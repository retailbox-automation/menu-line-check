# decider-agent

> Entry **N2 Menu Line Check** for the AWS Builder Center "Agent Experience Hackathon: Strands Decider"
> (deadline 23.10.2026 23:59 PT). Sections still marked **TBD** are filled in before submission.
> `decider_kit` is the concept-agnostic toolkit, `menu_line_check` is the agent, and
> `patterns/menu_line_tool_selection/` is the reusable decision pattern.

## What it does

A car owner stands at a dealer's service counter with a printed service menu and takes one photo. The agent
splits the menu into lines and, for each line, Strands Decider 2B decides which tool fits: decline an optional
add-on, check the maintenance schedule against the odometer, look up recalls for this car (public NHTSA API),
ask about a fee, or leave a safety part (brakes, tires, alignment, steering) to the owner. How sure Decider must
be depends on what a wrong action costs: a decline needs 0.8, a lookup or a question 0.5, and a decline on a line
Decider flags as a safety item is never taken automatically. Lines below their threshold go to a general model on
Amazon Bedrock; Bedrock also writes one plain question for the service advisor per flagged line and the summary
card (Keep · Decline · Ask · Your call). On the demo menu the
line "Replace fuel pump module $1,089.00" goes to the recall lookup, which finds recall campaign 23V858000 for a
2019 Civic: "Is this covered by recall 23V858000?"

## The decision handed to Decider 2B

TBD. Fill in: the one repeated decision, the workflow it lives in, what the agent must get right to be trusted, how we know it works.

| | |
|---|---|
| Decision | which tool fits one line of a printed service menu |
| Question type(s) | `readable` (noul), `safety` (noul), `tool` (choice) |
| Option set | `decline_addon`, `check_schedule`, `check_recall_warranty`, `ask_about_fee`, `owner_decides` |
| Where in the loop | per line before any tool runs; plus a `before_tool_call` gate on the LLM picker's `decline_addon` |
| Plain-code policy | unreadable -> retake; tool owner_decides -> owner decides; decline with P(safety) >= 0.5 -> LLM picker, never declined; confidence >= the tool's threshold (decline 0.8, lookups and fee 0.5) -> run it; else LLM picker |

## How it works

Diagram TBD.

```
agent (Strands + Bedrock model)
   |  proposes a tool call
   v
DeciderGate (InterventionHandler.before_tool_call)
   |  state = recent conversation + proposed call      POST /v1/systemone
   |  typed questions ------------------------------>  Decider 2B (separate host)
   |  <------------- answers: p(true) / option / score + confidence
   |  decide(answers) -> Proceed | Deny | Guide | Confirm | Transform
   v
ledger.jsonl  (one line per decision: answers, confidences, action, latency, state hash)
```

The model classifies, plain code decides. If the Decider server is unreachable the gate fails closed by default
(`Confirm` at `before_tool_call`; the agent pauses for a human), and the ledger records why.

## The reusable decision pattern

`patterns/menu_line_tool_selection/` (`questions.yaml`, `handler.py`, `README.md`, measured numbers included);
see `patterns/README.md`.
`patterns/example_grounding/` is only a labelled example reproducing the official weather example's two yes/no questions.

Using a pattern in another agent is meant to be one line:

```python
from strands import Agent
agent = Agent(tools=[...], interventions=[build_gate(DeciderClient("http://decider-host:8099"))])
```

## Measurements

TBD. To fill with `decider-bench` on our own labelled cases:

| Metric | Decider 2B | LLM baseline (Bedrock) |
|---|---|---|
| Latency p50 / p90 | TBD | TBD |
| Cost per 1,000 decisions | TBD | TBD |
| Accuracy | TBD | TBD |
| Accuracy at confidence >= 0.9 / abstain rate | TBD | n/a |
| Handling of low confidence (CONFIRM / HUMAN bands) | TBD | n/a |

The Bedrock baseline is behind `--baseline bedrock --model-id <id> --i-approve-spend` and has not been run (spend not approved).
Thresholds in `ThresholdPolicy` (act >= 0.9, confirm 0.5 to 0.9, human < 0.5) are the convention from the Decider architecture docs, not
a calibration: they must be measured on our own cases.

## What this does not show

TBD. Candidates to state honestly once measured: no cost win if it does not appear; wall clock end to end; the 4096-token window; noul
calibration (middle probabilities are common); `score` is not reliable on unseen tasks; the model is labelled experimental.

## Platform findings

Our spike on a Mac mini (19.10.2026, Decider 2B v1, `strands-decider` at e69edb2); figures exactly as measured:

1. **choice without a "do we have enough data" question confidently gets it wrong**: "What's the weather?" selects `get_weather` at 0.994 against
   `ask_clarifying_question` at 0.004. A single choice does not catch missing arguments: a separate noul "are the arguments grounded" is needed,
   which is exactly how the official example is built.
2. **Cold process vs warm server is two orders of magnitude**: 4.6 to 9.3 s per cold CLI process against a warm MLX server at **57 ms median**
   (p90 58 ms, 20 requests; server-reported 56 ms). The pattern has to keep the server warm. The public figures are about 115 ms (RTX 3090) and 153 ms (M3).
3. MPS is clearly slower than MLX at the tail: median 88 ms, p90 154 ms against 58 ms.

Found while building the toolkit:

* Strands has **five** intervention actions (`Proceed`, `Deny`, `Guide`, `Confirm`, `Transform`), not four. `Confirm` works only on `before_tool_call`;
  `Deny` has no effect after the fact. See `docs/STRANDS-INTERVENTIONS-NOTES.md` (with file:line).
* The registry hooks only methods **overridden at class level**, so a handler whose lifecycle stage is a constructor argument must override all five.
* `Guide` on `after_model_call` retries with no cap; a gate that guides there must make sure it converges.
* For `noul` the server returns only P(true). With confidence = max(p, 1-p) a noul answer is never below 0.5, so the HUMAN band is empty unless
  the confirm threshold for that question is raised.
* PyYAML turns unquoted `yes/no/on/off/true/false` into booleans: a choice option called `no` silently becomes `False`. `load_questions` rejects it with a clear message.

## How AI was used

TBD (state plainly which parts were written with AI assistance, and how the output was checked).

## Run it

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
pytest -q            # fake HTTP server, no model, no AWS
ruff check .
```

The real Decider model runs on another machine; this code only talks HTTP.

```bash
# the menu check on the demo menu (hand-made Textract JSON), NHTSA served from the recorded response
menu-check --ocr-json tests/fixtures/textract_menu.json --vehicle "2019 Honda Civic 31200" \
  --decider-url http://127.0.0.1:8099 --nhtsa-recorded tests/fixtures/nhtsa_recalls_civic_2019.json \
  --ledger out/ledger.jsonl --json out/report.json
# per-tool act thresholds over the defaults (decline_addon 0.8, the others 0.5); Decider is asked about
# all lines concurrently, at most --concurrency at a time (default 12); the CLI prints "12 lines decided in N ms".
# Measured 2026-10-20 on our reference server (strands-decider serve, MPS, --vision): it answers one request at a
# time, so 12 concurrent lines still took 6.7 s (server_ms grew 1.2 s -> 6.6 s); concurrency only overlaps the
# network. What shortens a menu is fewer or shorter questions per line (1 yes/no ~100 ms, 3 long ~490 ms).
menu-check ... --thresholds decline_addon=0.85,check_schedule=0.6 --concurrency 12
# Decider vs keyword rules on the 103 labelled spike lines, plus what the pipeline policy does with
# Decider's answers (per-tool thresholds vs the previous policy): recorded answers (no server) or live
menu-race --replay-recorded
menu-race --decider-url http://127.0.0.1:8099 --out out/race.md
```

Nothing calls Bedrock without `--allow-bedrock` (paid; model ids from `N2_PICKER_MODEL` / `N2_WRITER_MODEL`,
default `us.amazon.nova-lite-v1:0`). Without it the advisor questions come from templates and low-confidence
lines become "ask the advisor".

On the recorded spike answers (synthetic menus, 2026-10-20) the per-tool policy acts on 90 of 103 lines at 0.89
accuracy with no safety line declined, no false "your call" and no wrong decline; the previous single-threshold
policy acted on 76 at 0.76 with 10 false "your call". Details: `patterns/menu_line_tool_selection/README.md`.

```bash
# on the Decider host (not here)
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
# replay labelled cases against it
export STRANDS_DECIDER_URL=http://decider-host:8099
decider-bench --cases cases.jsonl --questions questions.yaml --thresholds 0.5,0.7,0.9,0.95
```

Case file, one JSON object per line: `{"id": "c1", "state": "...", "questions": {...}, "expected": {"q": true}}`
(`questions` may be omitted when `--questions` is given).

Layout:

```
src/decider_kit/       client.py  questions.py  policy.py  intervention.py  ledger.py  bench.py
src/menu_line_check/  models.py  ocr.py  nhtsa.py  schedule.py  tools.py  baseline.py  pipeline.py  llm.py
                      cli.py  race.py  data/ (illustrative maintenance schedule, 103 labelled race lines)
patterns/             README.md  menu_line_tool_selection/  example_grounding/
scripts/              build_race_lines.py (race lines from the spike output)
tests/                fake-server tests, real Strands actions, real Agent loop with a scripted model
docs/                 STRANDS-INTERVENTIONS-NOTES.md
```

## Licence

MIT, copyright Mikhail Oskola 2026. The example pattern reproduces question wording from the official `strands-labs/strands-decider`
example (Apache-2.0).
