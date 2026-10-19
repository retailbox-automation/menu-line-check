# patterns/

PLACEHOLDER. The hackathon requires at least one reusable, clearly labelled decision pattern: typed
question(s) with their option set, the wiring (here a Strands `InterventionHandler` via
`decider_kit.DeciderGate`), and a short note on what it does and how to use it.

Our pattern(s) will live here as:

```
patterns/<name>/
  questions.yaml   typed questions (noul / choice / score) with option sets
  handler.py       build_gate(...) -> DeciderGate, plus the decide(answers) policy
  README.md        what it decides, how to drop it into another agent, measured thresholds
```

`patterns/example_grounding/` is a worked EXAMPLE that reproduces the official weather example's two
yes/no questions (are the arguments grounded? is the call premature?). It is there to show the
format. It is not our entry, and it is not original work: the wording comes from the official example.

Rules for a pattern in this folder: no domain secrets, no client data, one-line install into someone
else's agent, a test that runs against a fake Decider server (no model load).
