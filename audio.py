"""Audio for the terminal: a speaker that knows how much of each reply it played, an echo canceller, and a microphone.

Everything is 16-bit mono PCM at 24 kHz, the format the realtime API takes in and sends back.
"""
import asyncio
import threading
from collections import deque

import numpy as np

RATE = 24000
BYTES_PER_MS = RATE * 2 // 1000  # 48
FRAME_MS = 10  # the echo canceller works in 10 ms frames
FRAME = RATE * FRAME_MS // 1000 * 2  # bytes


class Speaker:
    """Plays reply audio and counts, per reply item, how many milliseconds actually reached the speaker.

    pull() is what the sound card (or a simulated one) calls for the next chunk. stop() drops everything that
    hasn't played yet and says how far each cut reply got, which is what the server needs to hear on a barge-in.
    """

    def __init__(self):
        self.queue: deque[list] = deque()  # [item_id, bytearray still to play]
        self.played: dict[str, int] = {}  # item_id -> bytes played
        self.lock = threading.Lock()
        self.on_played = None  # called with every chunk that plays: the echo canceller's reference

    def play(self, item_id: str, pcm: bytes):
        with self.lock:
            if self.queue and self.queue[-1][0] == item_id:
                self.queue[-1][1].extend(pcm)
            else:
                self.queue.append([item_id, bytearray(pcm)])

    def pull(self, n: int) -> bytes:
        out = bytearray()
        with self.lock:
            while self.queue and len(out) < n:
                item_id, pcm = self.queue[0]
                take = pcm[: n - len(out)]
                del pcm[: len(take)]
                out.extend(take)
                self.played[item_id] = self.played.get(item_id, 0) + len(take)
                if not pcm:
                    self.queue.popleft()
        out.extend(b"\0" * (n - len(out)))
        if self.on_played:
            self.on_played(bytes(out))
        return bytes(out)

    def stop(self) -> dict[str, int]:
        """Drop the unplayed audio. Returns {item_id: ms played} for every reply that was cut short."""
        with self.lock:
            cut = {item_id: self.played.get(item_id, 0) // BYTES_PER_MS for item_id, _ in self.queue}
            self.queue.clear()
        return cut

    @property
    def busy(self) -> bool:
        return bool(self.queue)

    async def drain(self):
        while self.busy:
            await asyncio.sleep(0.05)


class EchoCanceller:
    """Removes the agent's own voice from the microphone, so the server never hears the reply coming back.

    It's WebRTC's audio processing module (via the livekit package) with only echo cancellation on. Noise
    suppression and gain control stay off: Dialog-RSN-1 hears the caller's audio directly, and those filters
    change what it hears.
    """

    def __init__(self):
        from livekit import rtc

        self.rtc = rtc
        self.apm = rtc.AudioProcessingModule(
            echo_cancellation=True, noise_suppression=False, high_pass_filter=True, auto_gain_control=False
        )
        self.far = bytearray()
        self.near = bytearray()

    def _frames(self, buf: bytearray):
        while len(buf) >= FRAME:
            frame = self.rtc.AudioFrame(bytes(buf[:FRAME]), RATE, 1, FRAME // 2)
            del buf[:FRAME]
            yield frame

    def played(self, pcm: bytes):
        """Feed it what the speaker just played: the echo it should expect."""
        self.far.extend(pcm)
        for frame in self._frames(self.far):
            self.apm.process_reverse_stream(frame)

    def clean(self, pcm: bytes) -> bytes:
        """Microphone audio in, the same audio with the speaker's echo removed out."""
        self.near.extend(pcm)
        out = bytearray()
        for frame in self._frames(self.near):
            self.apm.process_stream(frame)
            out.extend(bytes(frame.data))
        return bytes(out)


def level_db(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9))


class SoundCard:
    """The real microphone and speakers, through sounddevice. Mic chunks go on an asyncio queue."""

    def __init__(self, speaker: Speaker):
        import sounddevice as sd

        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        block = RATE // 50  # 20 ms

        def mic(indata, frames, time_info, status):
            loop.call_soon_threadsafe(self.queue.put_nowait, bytes(indata))

        def out(outdata, frames, time_info, status):
            outdata[:] = speaker.pull(len(outdata))

        self.streams = [
            sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=block, callback=mic),
            sd.RawOutputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=block, callback=out),
        ]

    def __enter__(self):
        for s in self.streams:
            s.start()
        return self

    def __exit__(self, *exc):
        for s in self.streams:
            s.stop()
            s.close()
