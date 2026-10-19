# decider-agent

> SKELETON. Concept not chosen yet: every section marked **TBD** is filled in after the concept is picked.
> What exists today is `decider_kit`, a concept-agnostic toolkit that any concept plugs into.
> Entry for the AWS Builder Center "Agent Experience Hackathon: Strands Decider" (deadline 23.10.2026 23:59 PT).

## What it does

TBD (one paragraph: the agent, who uses it, the wow moment in the first 60 seconds).

## The decision handed to Decider 2B

TBD. Fill in: the one repeated decision, the workflow it lives in, what the agent must get right to be trusted, how we know it works.

| | |
|---|---|
| Decision | TBD |
| Question type(s) | TBD (`noul` / `choice` / `score`) |
| Option set | TBD |
| Where in the loop | TBD (`before_tool_call` by default) |
| Plain-code policy | TBD |

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

TBD. The pattern lives in `patterns/<name>/{questions.yaml, handler.py, README.md}`; see `patterns/README.md`.
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
src/decider_kit/  client.py  questions.py  policy.py  intervention.py  ledger.py  bench.py
patterns/         README.md  example_grounding/
tests/            fake-server tests, real Strands actions, real Agent loop with a scripted model
docs/             STRANDS-INTERVENTIONS-NOTES.md
```

## Licence

MIT, copyright Mikhail Oskola 2026. The example pattern reproduces question wording from the official `strands-labs/strands-decider`
example (Apache-2.0).
