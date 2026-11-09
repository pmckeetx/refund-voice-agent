"""Scripted calls against the real API. Runs only with POLYAI_LIVE=1 and a POLY_API_KEY.

    POLYAI_LIVE=1 uv run pytest -m live

It checks the shape of the result, not the model's wording: one refund for the order, and a ledger that matches what
the caller heard.
"""
import os

import pytest

from ledger import Ledger, mismatches

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("POLYAI_LIVE") != "1", reason="set POLYAI_LIVE=1 to call the real API"),
]


@pytest.mark.parametrize("script", ["mm-hmm", "missed-confirmation"])
async def test_one_refund_and_the_caller_heard_it(script, tmp_path):
    from caller import run

    ledger = Ledger(tmp_path / "refunds.db")
    room = await run(script, ledger=ledger)
    rows = ledger.rows()

    assert [e for e in room.events if e["type"] == "response.function_call_arguments.done"], "the model never called issue_refund"
    assert len(rows) == 1 and rows[0]["order_id"] == "4417"
    assert mismatches(rows, room.heard()) == []
