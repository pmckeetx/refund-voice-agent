"""A scripted caller for the refund desk: no microphone, no speakers, a real call to Dialog-RSN-1.

It plays the WAVs in samples/ as the caller, at the moments each script says, while the agent's audio plays into a
simulated room at real time. At the end it prints the ledger.

    uv run caller.py --script mm-hmm               # "Mm-hmm." over the holding line
    uv run caller.py --script mm-hmm-gap           # "Mm-hmm." after the line, while the refund is going through
    uv run caller.py --script wrong-card           # "Wait, wrong card" over the holding line
    uv run caller.py --script missed-confirmation  # the caller talks over the confirmation
    uv run caller.py --script did-it-go-through    # "Sorry, did that go through?" over the holding line
    uv run caller.py --script did-it-go-through --before  # keyed on call_id, with the usual reply guard
"""
import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import websockets

from agent import RefundDesk, connect, refused
from audio import RATE, Speaker
from ledger import Ledger

SAMPLES = Path(__file__).parent / "samples"
TICK = 0.02
CHUNK = int(RATE * TICK) * 2

# Each step: the line the caller says, and when. None waits for the agent to finish first. ("line", s) says it s
# seconds into the holding line, ("gap", s) s seconds after the line ends while the refund is still going through,
# and ("confirm", s) s seconds into the reply that follows the refund.
SCRIPTS = {
    "mm-hmm": [("refund", None), ("mm-hmm", ("line", 0.8))],
    "mm-hmm-gap": [("refund", None), ("mm-hmm", ("gap", 0.3))],
    "tv": [("refund", None), ("tv", ("line", 0.8))],
    "wrong-card": [("refund", None), ("wrong-card", ("line", 0.5))],
    "did-it-go-through": [("refund", None), ("did-it-go-through", ("line", 0.8))],
    "mm-hmm-confirmation": [("refund", None), ("mm-hmm", ("confirm", 1.2))],
    "missed-confirmation": [("refund", None), ("thanks", ("confirm", 1.2))],
}


class CallKeyedLedger(Ledger):
    """The ledger most people write first: one row per tool call."""

    schema = """CREATE TABLE IF NOT EXISTS refunds (
        refund_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, amount_cents INTEGER NOT NULL, card_last4 TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', session_id TEXT NOT NULL, call_ids TEXT NOT NULL,
        accepted INTEGER NOT NULL DEFAULT 0, told INTEGER NOT NULL DEFAULT 0)"""

    def refund(self, order_id, card_last4, session_id, call_id):
        from ledger import ORDERS
        refund_id = f"R-{order_id}-{len(self.rows()) + 1}"  # every call is a new refund
        self.db.execute("INSERT INTO refunds (refund_id, order_id, amount_cents, card_last4, session_id, call_ids) "
                        "VALUES (?, ?, ?, ?, ?, ?)", (refund_id, order_id, ORDERS[order_id]["amount_cents"],
                                                      card_last4, session_id, json.dumps([call_id])))
        return "refunded", dict(self.db.execute("SELECT * FROM refunds WHERE refund_id = ?", (refund_id,)).fetchone())

    def row(self, order_id):
        rows = [r for r in self.rows() if r["order_id"] == order_id]
        return rows[-1] if rows else None

    def mark_told(self, order_id):
        self.db.execute("UPDATE refunds SET told = 1 WHERE order_id = ?", (order_id,))

    def mark_accepted(self, order_id):
        self.db.execute("UPDATE refunds SET accepted = 1 WHERE order_id = ?", (order_id,))


class Before(RefundDesk):
    """A refund desk without the fixes: a row per tool call, no reminder, and the usual guard on response.create.

    The guard skips the reply when the caller has started talking since the call, so that a slow tool's result
    isn't read out over their next question.
    """

    caller_turns = 0

    async def handle(self, event: dict):
        if event["type"] == "input_audio_buffer.speech_started":
            self.caller_turns += 1
        await super().handle(event)

    async def run_tool(self, call: dict):
        turns_before = self.caller_turns
        result = await self.issue_refund(json.loads(call["arguments"]), call["call_id"])
        self.owed[result["order_id"]] = self.responses  # bookkeeping for mismatches(), not a fix
        await self.send({"type": "conversation.item.create",
                         "item": {"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result)}})
        while self.responding:
            await asyncio.sleep(0.05)
        if self.caller_turns == turns_before:
            await self.send({"type": "response.create"})

    async def remind(self, row: dict):
        pass


def read(name: str) -> bytes:
    with wave.open(str(SAMPLES / f"{name}.wav")) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1
        return w.readframes(w.getnframes())


class Room:
    """Plays the agent's audio at real time, feeds the caller's lines to the microphone, and notes what it heard when.

    speed runs the room faster than real time, for the offline tests.
    """

    def __init__(self, desk: RefundDesk, speaker: Speaker, speed: float = 1.0):
        self.desk, self.speaker, self.speed = desk, speaker, speed
        self.voice = bytearray()  # the caller's line, still to say
        self.last_heard = 0.0
        self.t0 = time.monotonic()
        self.events: list[dict] = []  # every event both ways except audio, with t in seconds from the start
        self.kind: dict[str, str] = {}  # item_id -> "line" (holding line) or "confirm" (first reply after a result)
        self.started: dict[str, float] = {}  # item_id -> when its audio first reached the room
        self.tool_responses: set[str] = set()
        self.output_sent = False
        self.asked = False  # a response.create is out and hasn't been answered yet
        self.call_items: list[str] = []
        self.recording = bytearray()
        self._watch()

    def _watch(self):
        """Wrap the desk's send and handle, so the room sees every event without the desk knowing."""
        handle, send = self.desk.handle, self.desk.send

        async def watched_handle(event):
            self.note("recv", event)
            match event["type"]:
                case "response.created" | "error":
                    self.asked = False
                case "response.function_call_arguments.done":
                    self.tool_responses.add(event["response_id"])
                    self.call_items.append(event["item_id"])
                case "response.output_audio.delta" if event["item_id"] not in self.kind:
                    if event["response_id"] in self.tool_responses:
                        self.kind[event["item_id"]] = "line"
                    elif self.output_sent and "confirm" not in self.kind.values():
                        self.kind[event["item_id"]] = "confirm"
            await handle(event)

        async def watched_send(event):
            self.note("sent", event)
            if event.get("item", {}).get("type") == "function_call_output":
                self.output_sent = True
            self.asked = self.asked or event["type"] == "response.create"
            await send(event)

        self.desk.handle, self.desk.send = watched_handle, watched_send

    def note(self, direction: str, event: dict):
        if event["type"] in ("input_audio_buffer.append", "response.output_audio.delta"):
            return
        if event["type"].startswith("response.output_audio_transcript.delta"):
            return
        self.events.append({"t": round(time.monotonic() - self.t0, 3), "dir": direction, **event})

    def now(self) -> float:
        return (time.monotonic() - self.t0) * self.speed  # seconds of call time

    async def sleep(self, seconds: float):
        await asyncio.sleep(seconds / self.speed)

    async def run(self):
        start, n = time.monotonic(), 0
        while True:
            playing = self.speaker.queue[0][0] if self.speaker.queue else None
            played = np.frombuffer(self.speaker.pull(CHUNK), dtype=np.int16)
            if np.abs(played).max() > 200:
                self.last_heard = self.now()
                if playing and playing not in self.started:
                    self.started[playing] = self.now()
                    self.note("room", {"type": "room.audible", "item_id": playing, "kind": self.kind.get(playing, "reply")})
            said, self.voice[:] = bytes(self.voice[:CHUNK]).ljust(CHUNK, b"\0"), self.voice[CHUNK:]
            await self.desk.send_audio(said)
            both = played.astype(np.float32) + np.frombuffer(said, dtype=np.int16)
            self.recording.extend(both.clip(-32768, 32767).astype(np.int16).tobytes())
            n += 1
            await asyncio.sleep(max(0, start + n * TICK / self.speed - time.monotonic()))

    def say(self, name: str):
        self.desk.log("", f"(caller says {name}.wav)")
        self.note("room", {"type": "room.caller_says", "line": name})
        self.voice.extend(read(name))

    async def until(self, condition, timeout=30.0):
        end = self.now() + timeout
        while not condition():
            if self.now() > end:
                return False
            await self.sleep(TICK)
        return True

    def playing(self, kind: str) -> str | None:
        return next((i for i, k in self.kind.items() if k == kind and i in self.started), None)

    async def until_moment(self, when) -> bool:
        kind, after = when
        if kind == "gap":  # the line has played out, and the refund still hasn't come back
            ok = await self.until(lambda: self.playing("line") and not self.speaker.busy and not self.desk.responding)
            ok = ok and not self.output_sent
        else:
            ok = await self.until(lambda: self.playing(kind))
        if not ok:
            self.desk.log("", f"(the {kind} moment never came: the caller stays quiet)")
            return False
        await self.sleep(after)
        return True

    async def until_agent_done(self, timeout=60.0):
        await self.until(lambda: self.speaker.busy, timeout=30)
        await self.until(lambda: not (self.desk.responding or self.speaker.busy or self.desk.tasks or self.asked
                                      or self.now() - self.last_heard < 2.0), timeout)

    def heard(self) -> list[str]:
        """Every reply the caller heard, holding lines left out: what mismatches() checks the ledger against."""
        return [text for item_id, text in self.desk.replies if self.kind.get(item_id) != "line"]


async def run(script: str, before: bool = False, ledger: Ledger | None = None, speed: float = 1.0,
              audit: bool = False, client=None) -> Room:
    """Play one script against the server at DIALOG_REALTIME_URL and return the room, with everything it heard."""
    speaker = Speaker()
    ledger = ledger or (CallKeyedLedger("refunds-before.db") if before else Ledger())
    async with connect() as ws:
        desk = (client or (Before if before else RefundDesk))(ws, speaker, ledger=ledger)
        await desk.start()
        room = Room(desk, speaker, speed)
        tasks = [asyncio.create_task(room.run()), asyncio.create_task(desk.listen())]
        await room.sleep(0.5)
        for i, (line, when) in enumerate(SCRIPTS[script]):
            if when is None:
                if i > 0:  # the agent doesn't greet: the caller speaks first
                    await room.until_agent_done()
            elif not await room.until_moment(when):
                continue
            room.say(line)
            await room.sleep(len(read(line)) / (RATE * 2))
        await room.until_agent_done()
        if audit:  # is every tool call still in the server's history?
            for item_id in room.call_items:
                await desk.send({"type": "conversation.item.retrieve", "item_id": item_id})
            await room.sleep(1.0)
        for task in tasks:
            task.cancel()
    return room


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--script", choices=SCRIPTS, default="mm-hmm")
    parser.add_argument("--before", action="store_true", help="a row per tool call, and the usual guard on asking for a reply")
    parser.add_argument("--keep", action="store_true", help="keep the ledger from earlier calls instead of starting empty")
    parser.add_argument("--log", help="write every event, both ways, to this JSON lines file")
    parser.add_argument("--save", help="write the call as the room heard it, both voices, to this WAV")
    args = parser.parse_args()

    ledger = CallKeyedLedger("refunds-before.db") if args.before else Ledger()
    if not args.keep:
        ledger.clear()
    room = await run(args.script, args.before, ledger)
    print("\n" + ledger.show())
    if args.log:
        Path(args.log).write_text("".join(json.dumps(e) + "\n" for e in room.events))
    if args.save:
        with wave.open(args.save, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(bytes(room.recording))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except websockets.InvalidStatus as error:
        raise SystemExit(refused(error))
