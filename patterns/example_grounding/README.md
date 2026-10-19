# example_grounding (EXAMPLE, not our entry)

Shows the shape every pattern in this folder has. It reproduces the two yes/no questions of the
official weather example (`strands-labs/strands-decider`, `examples/strands/tool_call_intervention.py`,
Apache-2.0) and the same plain-Python policy, wired through `DeciderGate`.

| Part | File |
|---|---|
| Typed questions (2 x noul) | `questions.yaml` |
| Wiring (`DeciderGate`, `before_tool_call`) and policy | `handler.py` |

Policy: if `args_grounded` < 0.45, `Guide` the model to ask the user for the missing value; else if
`premature` >= 0.45, `Guide` it to clarify first; otherwise `Proceed`. If the Decider server is down,
the gate fails closed (`Confirm`) by default.

The 0.45 knob is illustrative (the official example says the same). Measure it on labelled cases with
`decider-bench` before using it for anything real.
