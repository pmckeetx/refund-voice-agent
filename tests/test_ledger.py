"""The ledger on its own: SQLite in a temp file, no network."""
import json
import threading

from caller import CallKeyedLedger
from ledger import Ledger, mentions, mismatches


def test_the_same_order_twice_is_one_refund(tmp_path):
    ledger = Ledger(tmp_path / "refunds.db")
    assert ledger.refund("4417", "4242", "conversation_a", "call_1")[0] == "refunded"
    outcome, row = ledger.refund("4417", "4242", "conversation_a", "call_2")
    assert outcome == "already_refunded"
    assert len(ledger.rows()) == 1
    assert json.loads(row["call_ids"]) == ["call_1", "call_2"]  # both calls are on the record


def test_a_caller_who_rings_back_gets_the_same_refund(tmp_path):
    ledger = Ledger(tmp_path / "refunds.db")
    ledger.refund("4417", "4242", "conversation_a", "call_1")
    outcome, row = ledger.refund("4417", "4242", "conversation_b", "call_1")  # a new call can reuse a call_id
    assert outcome == "already_refunded"
    assert row["session_id"] == "conversation_a"


def test_a_new_card_while_pending_changes_the_refund(tmp_path):
    ledger = Ledger(tmp_path / "refunds.db")
    ledger.refund("4417", "4242", "conversation_a", "call_1")
    ledger.mark_told("4417")
    outcome, row = ledger.refund("4417", "8812", "conversation_a", "call_2")
    assert outcome == "card_changed"
    assert (row["card_last4"], row["told"], len(ledger.rows())) == ("8812", 0, 1)  # the caller has to hear it again


def test_a_new_card_after_settlement_is_refused(tmp_path):
    ledger = Ledger(tmp_path / "refunds.db")
    ledger.refund("4417", "4242", "conversation_a", "call_1")
    ledger.settle("4417")
    outcome, row = ledger.refund("4417", "8812", "conversation_a", "call_2")
    assert (outcome, row["card_last4"]) == ("settled", "4242")


def test_two_connections_racing_for_one_order_make_one_refund(tmp_path):
    path, start, outcomes = tmp_path / "refunds.db", threading.Barrier(2), []
    Ledger(path)  # create the table before the race

    def ask(call_id):
        ledger = Ledger(path)
        start.wait()
        outcomes.append(ledger.refund("4417", "4242", "conversation_a", call_id)[0])

    threads = [threading.Thread(target=ask, args=(f"call_{i}",)) for i in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["already_refunded", "refunded"]
    assert len(Ledger(path).rows()) == 1


def test_a_ledger_keyed_on_call_id_refunds_twice(tmp_path):
    ledger = CallKeyedLedger(tmp_path / "refunds.db")
    ledger.refund("4417", "4242", "conversation_a", "call_1")
    ledger.refund("4417", "4242", "conversation_a", "call_2")
    assert len(ledger.rows()) == 2


def test_mentions_reads_the_card_however_it_is_written():
    assert mentions("32 dollars back to your Visa ending 4242.", "4242")
    assert mentions("your Visa ending in 4 2 4 2", "4242")
    assert not mentions("32 dollars back to your Visa ending-", "4242")


def test_mismatches_catches_each_way_the_ledger_and_the_caller_disagree():
    row = {"order_id": "4417", "refund_id": "R-4417", "card_last4": "8812", "told": 1}
    confirmed = "Done, 32 dollars back to your Mastercard ending 8812."
    assert mismatches([row], [confirmed]) == []
    assert mismatches([row, {**row, "refund_id": "R-call_2"}], [confirmed]) == ["order 4417 refunded 2 times"]
    assert mismatches([row], ["Let me process-"]) == ["R-4417 is in the ledger, but the caller never heard it was done"]
    assert mismatches([{**row, "told": 0}], [confirmed]) == ["R-4417 was heard, but the ledger says it wasn't"]
    stale = "That's 32 dollars to your Visa ending 4242."
    assert mismatches([row], [confirmed, stale]) == [
        f"the caller last heard {stale!r}, but no refund in the ledger goes to that card"]
    assert mismatches([], [confirmed]) != []  # told about a refund that doesn't exist
