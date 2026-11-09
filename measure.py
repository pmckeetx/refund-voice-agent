"""Run a script several times against the real API, each on an empty ledger, and count what happened.

    uv run measure.py --script mm-hmm --runs 5                  # refunds per order, confirmations heard, mismatches
    uv run measure.py --script did-it-go-through --audit        # also ask the server for every tool call afterwards
    uv run measure.py --script mm-hmm-gap --before              # the client keyed on call_id, with the old guard
    uv run measure.py --script tv --runs 10 --log runs/tv       # keep every event of every run as JSON lines

For each run it prints what the turn detector did (speech_started, speech_stopped, which replies were cancelled),
every issue_refund call and what the ledger answered, what happened to each response.create, and whether the
ledger matches what the caller heard.
"""
import argparse
import asyncio
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from caller import SCRIPTS, CallKeyedLedger, run
from ledger import Ledger, mismatches


def summarise(room, ledger) -> dict:
    got = [e for e in room.events if e["dir"] == "recv"]
    count = lambda kind: sum(e["type"] == kind for e in got)  # noqa: E731
    cancelled = [room.kind.get(item["id"], "reply") for e in got if e["type"] == "response.done"
                 and e["response"]["status"] == "cancelled" for item in e["response"].get("output", [])
                 if item["type"] == "message"]
    calls = [json.loads(e["arguments"]) for e in got if e["type"] == "response.function_call_arguments.done"]
    retrieved = [e["item"].get("type") for e in got if e["type"] == "conversation.item.retrieved"
                 and e["item"]["id"] in room.call_items]
    rows = ledger.rows()
    return {
        "speech_started": count("input_audio_buffer.speech_started"),
        "speech_stopped": count("input_audio_buffer.speech_stopped"),
        "caller_turns": [text for _, who, text in room.desk.lines if who == "Caller"],
        "cancelled": cancelled,
        "calls": [f"{c.get('order_id')}/{c.get('card_last4')}" for c in calls],
        "outcomes": [text.split(": ")[-1].rstrip(")") for _, who, text in room.desk.lines if text.startswith("(issue_refund")],
        "response_create": sum(e["type"] == "response.create" for e in room.events if e["dir"] == "sent"),
        "errors": [e["error"].get("code") or e["error"].get("message") for e in got if e["type"] == "error"],
        "reminders": sum(text.startswith("(reminder") for _, who, text in room.desk.lines),
        "calls_in_history": retrieved,
        "rows": [(r["order_id"], r["card_last4"], "told" if r["told"] else "untold") for r in rows],
        "mismatches": mismatches(rows, room.heard()),
        "heard": room.heard(),
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--script", choices=SCRIPTS, default="mm-hmm")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--before", action="store_true", help="a row per tool call, and turns counted on speech_started")
    parser.add_argument("--audit", action="store_true", help="after each call, retrieve every tool call from history")
    parser.add_argument("--log", help="a folder to write each run's events to, as JSON lines")
    args = parser.parse_args()

    host = os.environ.get("DIALOG_REALTIME_URL", "wss://api.studio.poly.ai/v1/realtime").split("/")[2]
    print(f"{args.script}, {args.runs} runs on {host}{', before' if args.before else ''}", flush=True)
    totals = Counter()
    for i in range(args.runs):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = (CallKeyedLedger if args.before else Ledger)(Path(tmp) / "refunds.db")
            room = await run(args.script, args.before, ledger, audit=args.audit)
            s = summarise(room, ledger)
        print(f"RUN {i + 1}: {json.dumps(s)}", flush=True)
        if args.log:
            Path(args.log).mkdir(parents=True, exist_ok=True)
            Path(args.log, f"{args.script}-{i + 1}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in room.events))
        totals["refunded twice"] += any(n > 1 for n in Counter(r[0] for r in s["rows"]).values())
        totals["no refund"] += not s["rows"]
        totals["caller never told"] += any("never heard" in m for m in s["mismatches"])
        totals["any mismatch"] += bool(s["mismatches"])
        totals["more than one call"] += len(s["calls"]) > 1
    print(f"{args.runs} runs: " + ", ".join(f"{k} {totals[k]}" for k in
          ("more than one call", "refunded twice", "no refund", "caller never told", "any mismatch")))


if __name__ == "__main__":
    asyncio.run(main())
