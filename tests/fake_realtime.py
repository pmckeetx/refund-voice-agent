"""A small stand-in for the Dialog-RSN-1 realtime endpoint, so the tests run with no key and no network.

Its clock is the caller's audio: every input_audio_buffer.append moves it on by the length of the chunk, and reply
audio streams out at the same pace, the way the real server sends it at speaking speed. So a test that runs the room
ten times faster than real time sees events in the same order a real call would. What it copies from the real server
(the API reference and overview, read 6 November 2026, and probes on api.studio.poly.ai, October and 10 November 2026):

- A tool call's response sends response.function_call_arguments.done first, then streams the poly_spoken line. A
  barge-in cancels that response, line and all, but the call has already been made.
- While a reply is audible, a sound only becomes a turn once it's confirmed (here: 600 ms of caller audio). A shorter
  one, like "mm-hmm", is ignored and the reply plays on (10 of 10 runs over the holding line, 10 of 10 over the
  confirmation). A barge-in's cancelled response.done arrives before its speech_started.
- While the agent is quiet, speech_started is sent at the onset, and "mm-hmm" is a turn like any other (5 of 5). The
  docs say a sound later judged to be noise is dropped, with no speech_stopped; here that's anything under 200 ms.
- response.create while a response is being written is refused with conversation_already_has_active_response. One
  sent while a reply is only playing is answered when its audio ends, and one sent while the caller is talking is
  held and answered by the response to their turn.
- conversation.item.truncate cuts history to audio_end_ms; conversation.item.retrieve returns the heard part, ending
  in "-".
"""
import base64
import json

import numpy as np
import websockets

LOUD = 1000
BYTES_PER_MS = 48
END_OF_TURN_MS = 500  # quiet after speech before the turn ends (prod: about 560 ms)
CONFIRM_MS = 600  # caller audio it takes to talk over a reply that's playing
NOISE_MS = 200  # while the agent is quiet, anything shorter is dropped as noise
DELTA_MS = 100


class FakeRealtime:
    def __init__(self, steps, caller_lines=None):
        """steps: what the server does for each response, in order.

        ("say", text, ms) speaks a reply. ("tool", name, args, line, ms) calls a tool and speaks the poly_spoken line.
        """
        self.steps = list(steps)
        self.caller_lines = list(caller_lines or ["(caller)"] * 10)
        self.received: list[dict] = []  # everything the client sent, except audio
        self.history: dict[str, dict] = {}  # item_id -> {"text", "ms", "heard_ms"}
        self.starts: list[int] = []  # for each response, how many events the client had sent when it started
        self.url = None

    def sent(self, event_type: str) -> list[dict]:
        return [e for e in self.received if e["type"] == event_type]

    async def __aenter__(self):
        self.server = await websockets.serve(self.handler, "127.0.0.1", 0, max_size=None)
        self.url = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/v1/realtime"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def handler(self, ws):
        send = lambda event: ws.send(json.dumps(event))  # noqa: E731
        s = {"n": 0, "reply": None, "open": None, "speaking": False, "loud_ms": 0, "quiet_ms": 0, "held": False,
             "after_audio": False}
        await send({"type": "session.created", "session": {"poly_session_id": "conversation_test"}})

        async def respond():
            if not self.steps:
                return
            kind, *rest = self.steps.pop(0)
            self.starts.append(len(self.received))  # what the client had sent when this response started
            s["n"] += 1
            rid, iid = f"resp_{s['n']}", f"item_{s['n']}"
            s["open"] = {"rid": rid, "output": []}
            await send({"type": "response.created", "response": {"id": rid, "status": "in_progress", "output": []}})
            if kind == "tool":
                name, args, text, ms = rest
                args = {**args, "say_while_working": text}
                call = {"type": "function_call", "id": f"call_item_{s['n']}", "call_id": f"call_{s['n']}", "name": name,
                        "arguments": json.dumps(args)}
                s["open"]["output"].append(call)
                await send({"type": "response.function_call_arguments.done", "response_id": rid, "item_id": call["id"],
                            "call_id": call["call_id"], "name": name, "arguments": call["arguments"]})
            else:
                text, ms = rest
            self.history[iid] = {"text": text, "ms": ms, "heard_ms": ms}
            s["reply"] = {"rid": rid, "iid": iid, "text": text, "left_ms": ms, "due_ms": 0}

        async def finish(status: str):
            reply, s["reply"] = s["reply"], None
            if status == "cancelled":
                self.history[reply["iid"]]["heard_ms"] = self.history[reply["iid"]]["ms"] - reply["left_ms"]
            await send({"type": "response.output_audio_transcript.done", "response_id": reply["rid"], "item_id": reply["iid"],
                        "transcript": reply["text"]})
            output = s["open"]["output"] + [self.item(reply["iid"])]
            s["open"] = None
            await send({"type": "response.done", "response": {"id": reply["rid"], "status": status, "output": output}})
            held, s["after_audio"] = s["after_audio"], False
            if held and status == "completed":  # a response.create sent while this played is answered now
                await respond()

        async def stream(ms: int):
            """Send the reply's audio for the next ms of call time."""
            reply = s["reply"]
            if not reply:
                return
            reply["due_ms"] += ms
            while reply["due_ms"] >= DELTA_MS and reply["left_ms"] > 0:
                reply["due_ms"] -= DELTA_MS
                reply["left_ms"] -= DELTA_MS
                reply["playing"] = True
                await send({"type": "response.output_audio.delta", "response_id": reply["rid"], "item_id": reply["iid"],
                            "delta": base64.b64encode(np.full(DELTA_MS * 24, 500, np.int16).tobytes()).decode()})
            if reply["left_ms"] <= 0:
                await finish("completed")

        async def turn_ends():
            await send({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 0})
            await send({"type": "conversation.item.input_audio_transcription.completed", "item_id": "item_caller",
                        "transcript": self.caller_lines.pop(0)})
            s["held"] = False  # a held response.create is answered by this turn's response
            await respond()

        async def caller_audio(pcm: bytes):
            ms = len(pcm) // BYTES_PER_MS
            loud = np.abs(np.frombuffer(pcm, dtype=np.int16)).max(initial=0) > LOUD
            audible = s["reply"] is not None
            if loud:
                s["loud_ms"] += ms
                s["quiet_ms"] = 0
                if not s["speaking"] and not audible:
                    s["speaking"] = True
                    await send({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
                elif audible and s["loud_ms"] >= CONFIRM_MS:  # a barge-in, confirmed
                    await finish("cancelled")
                    s["speaking"] = True
                    await send({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
            elif s["loud_ms"]:
                s["quiet_ms"] += ms
                if s["quiet_ms"] >= END_OF_TURN_MS:
                    real = s["loud_ms"] >= NOISE_MS
                    was_speaking, s["loud_ms"], s["quiet_ms"], s["speaking"] = s["speaking"], 0, 0, False
                    if real and was_speaking:
                        await turn_ends()
                    elif s["held"]:  # dropped as noise: the held request gets its answer now
                        s["held"] = False
                        await respond()
            await stream(ms)

        async for raw in ws:
            event = json.loads(raw)
            if event["type"] != "input_audio_buffer.append":
                self.received.append(event)
            match event["type"]:
                case "session.update":
                    await send({"type": "session.updated", "session": event["session"]})
                case "input_audio_buffer.append":
                    await caller_audio(base64.b64decode(event["audio"]))
                case "conversation.item.truncate":
                    self.history[event["item_id"]]["heard_ms"] = event["audio_end_ms"]
                    await send({"type": "conversation.item.truncated", "item_id": event["item_id"], "content_index": 0,
                                "audio_end_ms": event["audio_end_ms"]})
                case "conversation.item.retrieve":
                    await send({"type": "conversation.item.retrieved", "item": self.item(event["item_id"])})
                case "response.create":
                    if s["reply"] and s["reply"].get("playing"):
                        s["after_audio"] = True  # the server is only speaking now: answered when the audio ends
                    elif s["open"]:
                        await send({"type": "error", "error": {"type": "invalid_request_error",
                                    "code": "conversation_already_has_active_response",
                                    "message": "Conversation already has an active response in progress."}})
                    elif s["speaking"]:
                        s["held"] = True
                    else:
                        await respond()

    def item(self, iid: str) -> dict:
        h = self.history.get(iid)
        if h is None:
            return {"id": iid, "type": "function_call"}
        heard = h["text"]
        if h["heard_ms"] < h["ms"]:
            heard = h["text"][: round(len(h["text"]) * h["heard_ms"] / h["ms"])].rstrip() + "-"
        return {"id": iid, "type": "message", "role": "assistant", "content": [{"type": "output_audio", "transcript": heard}]}
