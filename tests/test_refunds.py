"""Whole calls against the fake server, ten times faster than real time. Each one ends with the same check: does the
ledger match what the caller heard?"""
import json

import pytest

import agent
from caller import CallKeyedLedger, run
from fake_realtime import FakeRealtime
from ledger import Ledger, mismatches

SPEED = 10
VISA = {"order_id": "4417", "card_last4": "4242"}
LINE = "Let me process that refund for you now."
DONE = "That's done: 32 dollars back to your Visa ending 4242."


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("POLY_API_KEY", "test-key")
    monkeypatch.setattr(agent, "PROVIDER_SECONDS", 4.0 / SPEED)


async def call(fake, monkeypatch, tmp_path, script, before=False):
    monkeypatch.setattr(agent, "URL", fake.url)
    ledger = (CallKeyedLedger if before else Ledger)(tmp_path / "refunds.db")
    room = await run(script, before=before, ledger=ledger, speed=SPEED)
    return room, ledger


def assert_ledger_matches_heard(room, ledger):
    assert mismatches(ledger.rows(), room.heard()) == []


async def test_mm_hmm_over_the_holding_line_changes_nothing(monkeypatch, tmp_path):
    async with FakeRealtime([("tool", "issue_refund", VISA, LINE, 2000), ("say", DONE, 2500)]) as fake:
        room, ledger = await call(fake, monkeypatch, tmp_path, "mm-hmm")
    assert fake.sent("conversation.item.truncate") == []  # a backchannel isn't a barge-in
    assert room.heard() == [DONE]
    assert_ledger_matches_heard(room, ledger)


async def test_a_reissued_call_is_the_same_refund(monkeypatch, tmp_path):
    steps = [("tool", "issue_refund", VISA, LINE, 2000), ("tool", "issue_refund", VISA, "Let me check on that.", 1500),
             ("say", DONE, 2500)]
    async with FakeRealtime(steps, ["I need a refund on order 4417.", "Sorry, did that go through?"]) as fake:
        room, ledger = await call(fake, monkeypatch, tmp_path, "did-it-go-through")
    outputs = [json.loads(e["item"]["output"]) for e in fake.sent("conversation.item.create")
               if e["item"]["type"] == "function_call_output"]
    assert sorted(o["status"] for o in outputs) == ["already_refunded", "refunded"]  # one refund, and one repeat
    assert len(ledger.rows()) == 1
    assert_ledger_matches_heard(room, ledger)


async def test_a_ledger_keyed_on_call_id_pays_out_twice(monkeypatch, tmp_path):
    steps = [("tool", "issue_refund", VISA, LINE, 2000), ("tool", "issue_refund", VISA, "Let me check on that.", 1500),
             ("say", DONE, 2500)]
    async with FakeRealtime(steps) as fake:
        room, ledger = await call(fake, monkeypatch, tmp_path, "did-it-go-through", before=True)
    assert mismatches(ledger.rows(), room.heard()) == ["order 4417 refunded 2 times"]


async def test_wrong_card_changes_the_refund_instead_of_adding_one(monkeypatch, tmp_path):
    steps = [("tool", "issue_refund", VISA, LINE, 2000),
             ("tool", "issue_refund", {"order_id": "4417", "card_last4": "8812"}, "Sure, let me switch that.", 1500),
             ("say", "Done: it's 32 dollars back to your Mastercard ending 8812 instead.", 2500)]
    async with FakeRealtime(steps) as fake:
        room, ledger = await call(fake, monkeypatch, tmp_path, "wrong-card")
    [row] = ledger.rows()
    assert row["card_last4"] == "8812"
    assert_ledger_matches_heard(room, ledger)


async def test_a_missed_confirmation_is_repeated_from_the_ledger(monkeypatch, tmp_path):
    steps = [("tool", "issue_refund", VISA, LINE, 2000), ("say", DONE, 3000),
             ("say", "Sorry, to be clear, your 32 dollars is back on your Visa ending 4242. It takes 3 to 5 days.", 3000)]
    async with FakeRealtime(steps) as fake:
        room, ledger = await call(fake, monkeypatch, tmp_path, "missed-confirmation")
    notes = [e for e in fake.received if e["type"] == "conversation.item.create" and e["item"].get("role") == "system"]
    assert len(notes) == 1 and "R-4417" in notes[0]["item"]["content"][0]["text"]
    assert fake.received.index(notes[0]) < fake.starts[2]  # in history before the reply to the caller's turn
    assert room.heard()[0].endswith("-")  # the first confirmation was cut before the card
    assert_ledger_matches_heard(room, ledger)


STILL_PROCESSING = "It's still processing. I'll let you know as soon as it goes through."


async def test_a_turn_answered_before_the_result_still_gets_the_confirmation(monkeypatch, tmp_path):
    steps = [("tool", "issue_refund", VISA, LINE, 2000), ("say", STILL_PROCESSING, 2500), ("say", DONE, 2500)]
    async with FakeRealtime(steps) as fake:
        room, ledger = await call(fake, monkeypatch, tmp_path, "did-it-go-through")
    assert room.heard() == [STILL_PROCESSING, DONE]
    assert_ledger_matches_heard(room, ledger)


async def test_the_usual_guard_never_says_it_went_through(monkeypatch, tmp_path):
    steps = [("tool", "issue_refund", VISA, LINE, 2000), ("say", STILL_PROCESSING, 2500), ("say", DONE, 2500)]
    async with FakeRealtime(steps) as fake:
        room, ledger = await call(fake, monkeypatch, tmp_path, "did-it-go-through", before=True)
    assert fake.sent("response.create") == []
    assert room.heard() == [STILL_PROCESSING]
    assert mismatches(ledger.rows(), room.heard()) == [
        "R-4417-1 is in the ledger, but the caller never heard it was done"]


async def test_a_reminder_the_model_ignores_is_sent_again(monkeypatch, tmp_path):
    steps = [("tool", "issue_refund", VISA, LINE, 2000), ("say", DONE, 3000),
             ("say", "I don't have information about that.", 2000),
             ("say", "And to confirm, your 32 dollars is back on your Visa ending 4242.", 2500)]
    async with FakeRealtime(steps) as fake:
        room, ledger = await call(fake, monkeypatch, tmp_path, "missed-confirmation")
    notes = [e for e in fake.received if e["type"] == "conversation.item.create" and e["item"].get("role") == "system"]
    assert len(notes) == 2
    assert room.heard()[-1].startswith("And to confirm")
    assert_ledger_matches_heard(room, ledger)


async def test_measure_counts_what_happened_on_a_call(monkeypatch, tmp_path):
    from measure import summarise

    async with FakeRealtime([("tool", "issue_refund", VISA, LINE, 2000), ("say", DONE, 2500)]) as fake:
        room, ledger = await call(fake, monkeypatch, tmp_path, "mm-hmm")
        s = summarise(room, ledger)
    assert (s["speech_started"], s["speech_stopped"], s["cancelled"]) == (1, 1, [])
    assert (s["calls"], s["outcomes"], s["response_create"]) == (["4417/4242"], ["refunded"], 1)
    assert (s["rows"], s["mismatches"]) == ([("4417", "4242", "told")], [])
