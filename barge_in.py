"""Barge-in plumbing for a Dialog-RSN-1 client: stop the voice, truncate history to what played, log what was heard.

The refund desk in agent.py builds on this. None of it is about refunds: it makes sure that what the client logs as
said, and what the server's history holds, is what the caller actually heard.
"""
import asyncio
import base64
import json
import time

from audio import EchoCanceller, Speaker


class BargeInClient:
    def __init__(self, ws, speaker: Speaker, echo: EchoCanceller | None = None):
        self.ws = ws
        self.speaker = speaker
        self.echo = echo
        self.written: dict[str, str] = {}  # item_id -> the full text of a reply, heard or not
        self.cut: set[str] = set()  # replies the caller talked over
        self.responding = False
        self.tasks: set[asyncio.Task] = set()  # tool calls and replies still playing
        self.lines: list[tuple[float, str, str]] = []  # (time, who, text): everything log() printed
        self.replies: list[tuple[str, str]] = []  # (item_id, text): every reply as the caller heard it

    async def send(self, event: dict):
        await self.ws.send(json.dumps(event))

    async def send_audio(self, pcm: bytes):
        if self.echo:
            pcm = self.echo.clean(pcm)
        if pcm:
            await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def cut_off(self):
        """Stop the agent's voice now, then tell the server how much of each reply the caller heard."""
        for item_id, played_ms in self.speaker.stop().items():
            self.cut.add(item_id)
            await self.send({"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0,
                             "audio_end_ms": played_ms})
            self.log("", f"(barge-in: stopped the reply after {played_ms / 1000:.1f} s)")

    async def handle(self, event: dict):
        match event["type"]:
            case "response.output_audio.delta":
                self.speaker.play(event["item_id"], base64.b64decode(event["delta"]))
            case "input_audio_buffer.speech_started":
                await self.cut_off()  # if the agent was quiet, there's nothing to stop
            case "conversation.item.input_audio_transcription.completed":
                self.log("Caller", event["transcript"])
            case "response.output_audio_transcript.done":
                self.written[event["item_id"]] = event["transcript"]
            case "response.created":
                self.responding = True
            case "response.done":
                self.responding = False
                if event["response"]["status"] == "cancelled":
                    await self.cut_off()  # this can arrive before speech_started
                await self.record(event["response"])
            case "conversation.item.retrieved" if self.text_of(event["item"]):
                await self.heard(event["item"]["id"], self.text_of(event["item"]), whole=False)
            case "error":
                self.log("Error", event["error"]["message"])

    async def record(self, response: dict):
        """A cut-off reply's full text never reached the caller, so ask the server for the heard part."""
        for item in response.get("output", []):
            if item["type"] != "message":
                continue
            if response["status"] == "cancelled" or item["id"] in self.cut:
                await self.send({"type": "conversation.item.retrieve", "item_id": item["id"]})
            else:
                self.spawn(self.after_playback(item["id"], self.written.get(item["id"], self.text_of(item))))

    async def after_playback(self, item_id: str, text: str):
        """Wait for the reply to leave the speaker. If the caller talked over it on the way, it was cut, not heard."""
        while any(queued == item_id for queued, _ in list(self.speaker.queue)):
            await asyncio.sleep(0.02)
        if item_id in self.cut:
            await self.send({"type": "conversation.item.retrieve", "item_id": item_id})
        else:
            await self.heard(item_id, text, whole=True)

    async def heard(self, item_id: str, text: str, whole: bool):
        """Called once per reply with what the caller heard of it. Subclasses hook in here."""
        self.replies.append((item_id, text))
        self.log("Agent", text if whole else text + "  [cut off]")

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    @staticmethod
    def text_of(item: dict) -> str:
        return " ".join(c.get("transcript") or c.get("text") or "" for c in item.get("content", []))

    def log(self, who: str, text: str):
        self.lines.append((time.monotonic(), who, text))
        print(f"{who + ':' if who else '':8}{text}", flush=True)

    async def listen(self):
        async for raw in self.ws:
            await self.handle(json.loads(raw))
