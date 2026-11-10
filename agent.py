"""The refund desk at Copperline Kitchen Supply: a voice agent on Dialog-RSN-1 whose refunds can't land twice.

Every refund goes through a SQLite ledger keyed on the order, so a repeated or reissued tool call is the same refund,
a change of card changes it instead of adding one, and the caller is told what the ledger says, again if they
didn't hear it the first time.

    uv run agent.py              # talk to it on your microphone, with headphones on
    uv run agent.py --speakers   # on laptop speakers: cancel the echo first
    uv run caller.py             # a scripted caller, no microphone needed
"""
import argparse
import asyncio
import json
import os

import websockets
from dotenv import load_dotenv

from audio import EchoCanceller, SoundCard, Speaker
from barge_in import BargeInClient
from ledger import CARDS, ORDERS, Ledger, describe, mentions

load_dotenv()

URL = os.environ.get("DIALOG_REALTIME_URL", "wss://api.studio.poly.ai/v1/realtime")
PROVIDER_SECONDS = 4.0  # how long the payment provider takes to accept a refund

INSTRUCTIONS = """You are the refund desk at Copperline Kitchen Supply. Everything you say is spoken aloud, so use no lists or symbols.
The caller has one order: order 4417, a cast iron skillet, 32 dollars, paid with their Visa ending 4242. They also have
a Mastercard ending 8812 on file.
When the caller asks for a refund, call issue_refund straight away with the card the order was paid with, unless they
ask for another card. Don't ask them to confirm first. If they change their mind about the card, call issue_refund
again with the new one.
When issue_refund returns, tell the caller what it says, with the amount and the card's last four digits written as
digits, like "32 dollars back to your Visa ending 4242". Keep replies to one or two sentences."""

TOOLS = [{
    "type": "function",
    "name": "issue_refund",
    "description": "Refund an order in full to one of the caller's cards.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order number, like 4417."},
            "card_last4": {"type": "string", "description": "The last four digits of the card to refund to."},
            "say_while_working": {
                "type": "string",
                "poly_spoken": True,  # the server speaks this argument while the tool runs
                "description": "A sentence to say while the refund goes through, like 'Let me process that refund for you now.'",
            },
        },
        "required": ["order_id", "card_last4", "say_while_working"],
    },
}]


def provider_key(row: dict) -> str:
    """The idempotency key the payment provider gets: one per refund and card, so a repeat is deduplicated by the
    provider too, and a change of card is a new request instead of a reused key with different parameters."""
    return f"{row['refund_id']}-{row['card_last4']}"


async def submit_refund(row: dict):
    """The payment provider: a stand-in that takes a few seconds. A real one gets provider_key(row) as its key."""
    await asyncio.sleep(PROVIDER_SECONDS)


class RefundDesk(BargeInClient):
    def __init__(self, ws, speaker: Speaker, echo: EchoCanceller | None = None, ledger: Ledger | None = None):
        super().__init__(ws, speaker, echo)
        self.ledger = ledger or Ledger()
        self.session_id = "unknown"
        self.responses = 0  # response.created events so far
        self.response_of: dict[str, int] = {}  # item_id -> which response it came in
        self.with_calls: set[int] = set()  # responses that made a tool call: their message is the holding line
        self.owed: dict[str, int] = {}  # order_id -> responses seen when its result went into history
        self.reminded: dict[str, int] = {}  # order_id -> reminders sent since its last result

    async def start(self):
        created = json.loads(await self.ws.recv())
        self.session_id = created.get("session", {}).get("poly_session_id", "unknown")
        await self.send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": INSTRUCTIONS,
                "tools": TOOLS,
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": {"type": "server_vad"}}},
            },
        })
        updated = json.loads(await self.ws.recv())
        if updated["type"] != "session.updated":
            raise RuntimeError(updated.get("error", {}).get("message", f"unexpected {updated['type']}"))

    async def handle(self, event: dict):
        match event["type"]:
            case "response.created":
                self.responses += 1
            case "response.output_audio.delta":
                self.response_of.setdefault(event["item_id"], self.responses)
            case "response.done":
                for item in event["response"].get("output", []):
                    self.response_of.setdefault(item["id"], self.responses)
            case "response.function_call_arguments.done":
                self.with_calls.add(self.responses)
                self.spawn(self.run_tool(event))  # the call is final: start the refund before the holding line plays
        await super().handle(event)

    async def run_tool(self, call: dict):
        try:
            result = await self.issue_refund(json.loads(call["arguments"]), call["call_id"])
        except Exception as error:  # every call gets an output, or the model waits for it forever
            self.log("", f"(issue_refund failed: {error})")
            result = {"status": "error", "note": "The refund system failed, so the refund isn't confirmed. Don't call "
                      "issue_refund again. Tell the caller the team will email them once it's sorted."}
        since = self.responses
        await self.send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call["call_id"], "output": json.dumps(result)},
        })
        if result.get("refund_id"):
            self.owed.setdefault(result["order_id"], since)  # from the first result the caller hasn't heard
            self.reminded.pop(result["order_id"], None)
        await self.ask_for_reply(since)

    async def ask_for_reply(self, since: int):
        """Ask the model to speak about the result, unless a reply has already started since it went into history."""
        while self.responding:  # one response at a time: let the holding line, or the answer to the caller, finish
            await asyncio.sleep(0.05)
        if self.responses == since:
            await self.send({"type": "response.create"})  # sent while the caller talks, it's held for their turn

    async def issue_refund(self, args: dict, call_id: str) -> dict:
        order_id, card = str(args.get("order_id", "")).strip(), str(args.get("card_last4", "")).strip()
        if order_id not in ORDERS:
            return {"status": "error", "note": f"There's no order {order_id} for this caller. Ask them to check the number."}
        if card not in CARDS:
            return {"status": "error", "note": f"The caller has no card ending {card}. Ask which card they want."}
        outcome, row = self.ledger.refund(order_id, card, self.session_id, call_id)
        self.log("", f"(issue_refund {order_id} to {card}: {outcome})")
        if not row["accepted"]:  # new, a new card, or a repeat of one the provider never accepted
            await submit_refund(row)
            self.ledger.mark_accepted(order_id)
        return {"status": outcome, "order_id": order_id, "refund_id": row["refund_id"], "note": self.note_for(outcome, row)}

    def note_for(self, outcome: str, row: dict) -> str:
        """What the model is told, worded from the ledger's row, so its next line says what actually happened."""
        if outcome == "refunded":
            return f"Refunded {describe(row)}. Tell the caller it's done."
        if outcome == "already_refunded":
            when = "earlier in this call" if row["session_id"] == self.session_id else "on an earlier call"
            return f"This refund was already issued {when}: {describe(row)}. Tell the caller it's done; don't refund again."
        if outcome == "card_changed":
            return f"Still one refund, now {describe(row)} instead. Tell the caller."
        return f"This refund already settled, {describe(row)}, and can't be moved. Tell the caller a person will call back."

    async def heard(self, item_id: str, text: str, whole: bool):
        await super().heard(item_id, text, whole)
        for order_id, since in list(self.owed.items()):
            n = self.response_of.get(item_id, 0)
            if n <= since or n in self.with_calls:
                continue  # a holding line, or a reply that started before the result was in history
            row = self.ledger.row(order_id)
            if mentions(text, row["card_last4"]):
                self.ledger.mark_told(order_id)
                del self.owed[order_id]
            elif self.reminded.get(order_id, 0) < 2:  # twice at most, so a model that won't say it can't loop
                self.spawn(self.remind(row))

    async def remind(self, row: dict):
        """The refund is real, but the caller didn't hear it. Put the ledger's word in history and ask for a reply."""
        self.reminded[row["order_id"]] = self.reminded.get(row["order_id"], 0) + 1
        note = (f"Refund {row['refund_id']} is done: {describe(row)}. The caller hasn't heard that yet. In your next "
                "reply, tell them it's done, with the amount and the card's last four digits.")
        self.log("", f"(reminder: {row['refund_id']} not heard yet)")
        since = self.responses
        await self.send({"type": "conversation.item.create",
                         "item": {"type": "message", "role": "system", "content": [{"type": "input_text", "text": note}]}})
        await self.ask_for_reply(since)


def connect():
    key = os.environ.get("POLY_API_KEY")
    if not key:
        raise SystemExit("POLY_API_KEY isn't set. Copy .env.example to .env and add your key from studio.poly.ai.")
    return websockets.connect(URL, additional_headers={"Authorization": f"Bearer {key}"})


async def call(speakers: bool):
    """A live call on your microphone. Ctrl+C to hang up."""
    speaker = Speaker()
    echo = EchoCanceller() if speakers else None  # on headphones there's no echo to cancel, and it would hide the caller
    if echo:
        speaker.on_played = echo.played
    async with connect() as ws:
        with SoundCard(speaker) as card:
            desk = RefundDesk(ws, speaker, echo)
            await desk.start()
            print("Connected to Copperline Kitchen Supply. Ask for a refund on order 4417. Ctrl+C to hang up.")

            async def stream_mic():
                while True:
                    await desk.send_audio(await card.queue.get())

            try:
                await asyncio.gather(stream_mic(), desk.listen())
            finally:
                print("\n" + desk.ledger.show())


def refused(error: websockets.InvalidStatus) -> str:
    """One line instead of a traceback when the server turns the connection down."""
    return (f"The server refused the connection (HTTP {error.response.status_code}). Check POLY_API_KEY, and set "
            f"DIALOG_REALTIME_URL if your key is for a different host than {URL}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--speakers", action="store_true", help="you're on speakers, not headphones: cancel the echo")
    try:
        asyncio.run(call(parser.parse_args().speakers))
    except KeyboardInterrupt:
        print("\nHung up.")
    except websockets.InvalidStatus as error:
        raise SystemExit(refused(error))
