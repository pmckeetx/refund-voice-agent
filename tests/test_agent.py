"""The refund desk's plumbing against the fake server: barge-in, refused sessions, bad tool arguments."""
import asyncio
import json

import numpy as np
import pytest

import agent
from agent import RefundDesk, connect
from audio import BYTES_PER_MS, Speaker
from fake_realtime import FakeRealtime
from ledger import Ledger

TALK = np.full(480, 4000, np.int16).tobytes()  # 20 ms of a loud caller
QUIET = bytes(960)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("POLY_API_KEY", "test-key")


async def until(condition, timeout=3.0):
    end = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > end:
            raise TimeoutError
        await asyncio.sleep(0.005)


async def open_desk(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "URL", fake.url)
    ws = await connect()
    desk = RefundDesk(ws, Speaker(), ledger=Ledger(tmp_path / "refunds.db"))
    await desk.start()
    return ws, desk, asyncio.create_task(desk.listen())


async def say(desk, ms):
    for _ in range(ms // 20):
        await desk.send_audio(TALK)
    for _ in range(30):
        await desk.send_audio(QUIET)


async def test_session_id_comes_from_session_created(monkeypatch, tmp_path):
    async with FakeRealtime([]) as fake:
        ws, desk, listener = await open_desk(fake, monkeypatch, tmp_path)
        assert desk.session_id == "conversation_test"
        listener.cancel()
        await ws.close()


async def test_barge_in_stops_playback_and_truncates_to_what_played(monkeypatch, tmp_path):
    async with FakeRealtime([("say", "Your order 4417 was a cast iron skillet for 32 dollars.", 3000)]) as fake:
        ws, desk, listener = await open_desk(fake, monkeypatch, tmp_path)
        await say(desk, 800)
        for _ in range(100):  # 2 s of quiet: the reply streams in
            await desk.send_audio(QUIET)
        await until(lambda: desk.speaker.busy)
        desk.speaker.pull(1200 * BYTES_PER_MS)  # 1.2 s reaches the caller's ear
        await say(desk, 800)  # and they talk over it
        await until(lambda: fake.sent("conversation.item.truncate"))
        assert not desk.speaker.busy
        assert fake.sent("conversation.item.truncate")[0]["audio_end_ms"] == 1200
        listener.cancel()
        await ws.close()


async def test_an_unknown_order_or_card_refunds_nothing(monkeypatch, tmp_path):
    async with FakeRealtime([]) as fake:
        ws, desk, listener = await open_desk(fake, monkeypatch, tmp_path)
        no_order = await desk.issue_refund({"order_id": "9999", "card_last4": "4242"}, "call_1")
        no_card = await desk.issue_refund({"order_id": "4417", "card_last4": "0000"}, "call_2")
        assert no_order["status"] == no_card["status"] == "error"
        assert "Ask them to check the number" in no_order["note"]
        assert desk.ledger.rows() == []
        listener.cancel()
        await ws.close()


async def test_a_tool_that_throws_still_answers_the_call(monkeypatch, tmp_path):
    async def provider_down(row):
        raise ConnectionError("payment provider unreachable")

    monkeypatch.setattr(agent, "submit_refund", provider_down)
    async with FakeRealtime([]) as fake:
        ws, desk, listener = await open_desk(fake, monkeypatch, tmp_path)
        await desk.run_tool({"call_id": "call_1", "arguments": json.dumps({"order_id": "4417", "card_last4": "4242"})})
        await until(lambda: fake.sent("response.create"))
        [output] = fake.sent("conversation.item.create")
        result = json.loads(output["item"]["output"])
        assert result["status"] == "error" and "isn't confirmed" in result["note"]
        assert fake.sent("response.create")  # the model is asked to tell the caller
        listener.cancel()
        await ws.close()


async def test_refused_session_update_raises(monkeypatch):
    class Refuses(FakeRealtime):
        async def handler(self, ws):
            await ws.send('{"type": "session.created", "session": {}}')
            await ws.recv()
            await ws.send('{"type": "error", "error": {"message": "unknown parameter"}}')
            await asyncio.sleep(0.1)

    async with Refuses([]) as fake:
        monkeypatch.setattr(agent, "URL", fake.url)
        async with connect() as ws:
            with pytest.raises(RuntimeError, match="unknown parameter"):
                await RefundDesk(ws, Speaker(), ledger=Ledger(":memory:")).start()
