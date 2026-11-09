# Refund voice agent

A Python voice agent on PolyAI's Dialog-RSN-1 that issues refunds without paying out twice. Every refund goes through
a SQLite ledger keyed on the order, so a reissued tool call is the same refund, a change of card changes the refund
instead of adding one, and a confirmation the caller talked over is repeated from what the ledger says. The tests pass
only when the ledger matches what the caller heard.

The walkthrough is the post: [Stop your voice agent refunding twice when a caller says mm-hmm](https://www.petermckee.dev/blog/stop-your-voice-agent-refunding-twice-when-a-caller-says-mm).

## What's here

`agent.py` is the refund desk at Copperline Kitchen Supply, a made-up shop, and the file the post walks through.
`ledger.py` is the SQLite ledger and the check that compares it with what the caller heard. `barge_in.py` stops the
agent's voice when the caller talks over it and keeps the server's history to what was heard. `audio.py` has the
speaker, the echo canceller and the sound card. `caller.py` is a scripted caller that plays the lines in `samples/`
into a simulated room on a real call, and `measure.py` repeats a script and counts what happened.

## Prerequisites

Python 3.13 and [uv](https://docs.astral.sh/uv/). A Dialog-RSN-1 API key: sign in at
[studio.poly.ai](https://studio.poly.ai) and create one in the **API Keys** tab. For `agent.py` you need a
microphone, and headphones unless you pass `--speakers`.

## Set up

```bash
git clone https://github.com/pmckeetx/refund-voice-agent.git
cd refund-voice-agent
cp .env.example .env    # then add your POLY_API_KEY
uv sync
```

## Run

```bash
uv run caller.py --script mm-hmm               # "Mm-hmm." over the holding line
uv run caller.py --script mm-hmm-gap           # "Mm-hmm." after the line, while the refund goes through
uv run caller.py --script wrong-card           # "Wait, wrong card" over the holding line
uv run caller.py --script did-it-go-through    # "Sorry, did that go through?" over the holding line
uv run caller.py --script missed-confirmation  # the caller talks over the confirmation
uv run caller.py --script mm-hmm-gap --before  # a client keyed on call_id, counting turns on speech_started
uv run measure.py --script mm-hmm --runs 5     # repeat a script and count refunds, confirmations and mismatches
uv run ledger.py --show                        # print the ledger
uv run agent.py                                # talk to it yourself, with headphones on
```

Each scripted call starts on an empty ledger (`--keep` keeps the last one), and `agent.py` keeps its ledger in
`refunds.db` between calls, so ringing back about the same order gets the same refund. The caller's lines were made
with macOS `say`; `uv run samples/make_samples.py` rebuilds them on a Mac.

A variable set in your shell beats `.env`. If the server refuses the connection (HTTP 401), check
`echo $POLY_API_KEY`, and set `DIALOG_REALTIME_URL` if your key is for another PolyAI host.

## Test

```bash
uv run pytest                           # offline: a fake realtime server and a temp ledger, no key needed
POLYAI_LIVE=1 uv run pytest -m live     # two scripted calls against the real API
```

## License

Apache 2.0. See [LICENSE](LICENSE).
