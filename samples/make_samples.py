"""Rebuild the caller's lines with macOS `say`, as 24 kHz 16-bit mono WAVs.

    uv run samples/make_samples.py

The caller is Samantha. tv.wav is a newsreader (Daniel) about 10 dB under the caller's level, with a little room
noise, the way a television sounds from across the room.
"""
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RATE = 24000
LINES = {
    "refund": "Hi, I need a refund on order 4417. The cast iron skillet arrived cracked.",
    "mm-hmm": "Mm-hmm.",
    "wrong-card": "Wait, wrong card. Can you put it on my Mastercard instead?",
    "did-it-go-through": "Sorry, did that go through?",
    "thanks": "Okay, thanks. How long does it take to show up?",
}
TV = "Officials say the bridge will stay closed through the weekend while crews finish the repairs."


def say(voice: str, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "line.wav"
        subprocess.run(["say", "-v", voice, "-o", str(out), f"--data-format=LEI16@{RATE}", text], check=True)
        with wave.open(str(out)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def write(name: str, pcm: np.ndarray):
    with wave.open(str(HERE / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.clip(-32768, 32767).astype(np.int16).tobytes())
    print(f"{name}.wav")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


if __name__ == "__main__":
    caller_level = 0.0
    for name, text in LINES.items():
        pcm = say("Samantha", text)
        caller_level = max(caller_level, rms(pcm[np.abs(pcm) > 300]))
        write(name, pcm)
    tv = say("Daniel", TV)
    tv *= caller_level * 10 ** (-10 / 20) / rms(tv[np.abs(tv) > 300])  # 10 dB under the caller
    noise = np.random.default_rng(4417).normal(0, caller_level * 10 ** (-35 / 20), len(tv))  # room tone
    write("tv", tv + noise)
