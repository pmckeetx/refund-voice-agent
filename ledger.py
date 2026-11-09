"""The refund ledger: one row per order, in SQLite, however many times the model asks.

    uv run ledger.py --show          # print the ledger
    uv run ledger.py --settle 4417   # the payment provider settles a refund (after this, the card can't change)
"""
import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

PATH = Path(__file__).parent / "refunds.db"

# The shop's side: what each order cost and the cards on file. The model never chooses an amount.
ORDERS = {"4417": {"item": "cast iron skillet", "amount_cents": 3200, "card_last4": "4242"}}
CARDS = {"4242": "Visa", "8812": "Mastercard"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS refunds (
    order_id     TEXT PRIMARY KEY,                -- the key: what the refund is for, not who asked
    refund_id    TEXT NOT NULL UNIQUE,
    amount_cents INTEGER NOT NULL,
    card_last4   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending', -- pending until the payment provider settles it
    session_id   TEXT NOT NULL,                   -- the call that asked first
    call_ids     TEXT NOT NULL,                   -- every tool call that asked for it, as a JSON list
    told         INTEGER NOT NULL DEFAULT 0       -- did a confirmation of this row play to the end?
)
"""


class Ledger:
    schema = SCHEMA

    def __init__(self, path: Path | str = PATH):
        self.db = sqlite3.connect(path, isolation_level=None, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute(self.schema)

    def refund(self, order_id: str, card_last4: str, session_id: str, call_id: str) -> tuple[str, dict]:
        """Record a refund for an order. Returns what happened, and the row as it now stands."""
        amount = ORDERS[order_id]["amount_cents"]
        self.db.execute("BEGIN IMMEDIATE")  # the check and the write happen under one lock
        try:
            added = self.db.execute(
                "INSERT INTO refunds (order_id, refund_id, amount_cents, card_last4, session_id, call_ids) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (order_id) DO NOTHING",
                (order_id, f"R-{order_id}", amount, card_last4, session_id, json.dumps([call_id])),
            ).rowcount
            if added:
                outcome = "refunded"
            else:
                row = self.row(order_id)
                self.db.execute("UPDATE refunds SET call_ids = ? WHERE order_id = ?",
                                (json.dumps(json.loads(row["call_ids"]) + [call_id]), order_id))
                if row["card_last4"] == card_last4:
                    outcome = "already_refunded"
                elif row["status"] == "pending":
                    self.db.execute("UPDATE refunds SET card_last4 = ?, told = 0 WHERE order_id = ?", (card_last4, order_id))
                    outcome = "card_changed"
                else:
                    outcome = "settled"  # the money has gone: moving it is a new refund and a reversal, not an update
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return outcome, self.row(order_id)

    def mark_told(self, order_id: str):
        self.db.execute("UPDATE refunds SET told = 1 WHERE order_id = ?", (order_id,))

    def settle(self, order_id: str):
        self.db.execute("UPDATE refunds SET status = 'settled' WHERE order_id = ?", (order_id,))

    def row(self, order_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM refunds WHERE order_id = ?", (order_id,)).fetchone()
        return dict(row) if row else None

    def rows(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM refunds ORDER BY refund_id")]

    def clear(self):
        self.db.execute("DELETE FROM refunds")

    def show(self) -> str:
        lines = ["refund_id  order  amount  card  status   told  calls"]
        for r in self.rows():
            lines.append(f"{r['refund_id']:<10} {r['order_id']:<6} {r['amount_cents'] / 100:>6.2f}  {r['card_last4']}  "
                         f"{r['status']:<8} {'yes' if r['told'] else 'no':<5} {len(json.loads(r['call_ids']))}")
        return "\n".join(lines)


def describe(row: dict) -> str:
    return f"{row['amount_cents'] // 100} dollars to the {CARDS[row['card_last4']]} ending {row['card_last4']}"


def mentions(text: str, last4: str) -> bool:
    """Did this reply say the card's last four digits? Allows "4242", "4 2 4 2" and "4-2-4-2"."""
    return re.search(r"\W?".join(last4), text) is not None


def mismatches(rows: list[dict], heard: list[str]) -> list[str]:
    """Where the ledger and the caller disagree. heard is every reply the caller heard, holding lines left out."""
    problems = [f"order {order} refunded {n} times" for order, n in Counter(r["order_id"] for r in rows).items() if n > 1]
    for row in rows:
        if not any(mentions(reply, row["card_last4"]) for reply in heard):
            problems.append(f"{row['refund_id']} is in the ledger, but the caller never heard it was done")
        elif not row["told"]:
            problems.append(f"{row['refund_id']} was heard, but the ledger says it wasn't")
    confirmations = [reply for reply in heard if any(mentions(reply, card) for card in CARDS)]
    if confirmations and not any(mentions(confirmations[-1], row["card_last4"]) for row in rows):
        problems.append(f"the caller last heard {confirmations[-1]!r}, but no refund in the ledger goes to that card")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--show", action="store_true", help="print the ledger")
    parser.add_argument("--settle", metavar="ORDER", help="mark a refund as settled by the payment provider")
    parser.add_argument("--path", default=PATH, help=f"the SQLite file (default {PATH.name})")
    args = parser.parse_args()
    ledger = Ledger(args.path)
    if args.settle:
        ledger.settle(args.settle)
    print(ledger.show())
