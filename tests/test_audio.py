"""The speaker's playback accounting and the echo canceller, on synthetic audio."""
import wave
from pathlib import Path

import numpy as np

from audio import BYTES_PER_MS, RATE, EchoCanceller, Speaker, level_db

SAMPLES = Path(__file__).parent.parent / "samples"


def test_speaker_counts_what_played_per_reply():
    speaker = Speaker()
    speaker.play("item_a", bytes(500 * BYTES_PER_MS))
    speaker.play("item_b", bytes(800 * BYTES_PER_MS))
    speaker.pull(700 * BYTES_PER_MS)  # all of item_a, 200 ms of item_b
    assert speaker.stop() == {"item_b": 200}
    assert not speaker.busy


def test_stop_with_nothing_playing_cuts_nothing():
    speaker = Speaker()
    speaker.play("item_a", bytes(100 * BYTES_PER_MS))
    speaker.pull(100 * BYTES_PER_MS)
    assert speaker.stop() == {}


def test_pull_pads_with_silence_and_reports_to_the_echo_canceller():
    heard = []
    speaker = Speaker()
    speaker.on_played = heard.append
    speaker.play("item_a", b"\x01\x00" * 10)
    out = speaker.pull(40)
    assert out == b"\x01\x00" * 10 + bytes(20)
    assert heard == [out]


def voice(name: str) -> np.ndarray:
    with wave.open(str(SAMPLES / f"{name}.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def test_echo_canceller_removes_the_agent_and_keeps_the_caller():
    agent_voice = np.concatenate([voice("refund"), voice("refund")])  # stands in for a reply
    delay = RATE * 40 // 1000  # speaker to mic
    echo = np.concatenate([np.zeros(delay), agent_voice[:-delay]]) * 0.3
    caller = np.zeros_like(agent_voice)
    line = voice("thanks")
    caller[RATE * 6 : RATE * 6 + len(line)] = line
    mic = echo + caller

    canceller, cleaned = EchoCanceller(), bytearray()
    step = RATE // 50 * 2  # 20 ms, as the sound card delivers it
    far, near = agent_voice.astype(np.int16).tobytes(), mic.astype(np.int16).tobytes()
    for i in range(0, len(far) - step, step):
        canceller.played(far[i : i + step])
        cleaned.extend(canceller.clean(near[i : i + step]))

    s = slice(RATE * 4 * 2, RATE * 6 * 2)  # bytes: seconds 4 to 6, echo only
    c = slice(RATE * 6 * 2, (RATE * 6 + len(line)) * 2)  # the caller talking over the echo
    assert level_db(near[s]) - level_db(bytes(cleaned[s])) > 30  # echo cut by more than 30 dB
    assert level_db(bytes(cleaned[c])) > level_db(caller[RATE * 6 : RATE * 6 + len(line)].astype(np.int16).tobytes()) - 6
