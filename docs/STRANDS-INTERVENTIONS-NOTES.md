# Strands interventions: what the installed source says

Read from the INSTALLED package `strands-agents==1.61.1` (pinned in `pyproject.toml`), not from docs or memory.
Paths below are relative to `.venv/lib/python3.12/site-packages/`. Line numbers are for that exact version.
Nothing here is guessed; where a behaviour was observed rather than read, it says "observed" and names the test.

## 1. Base class: `strands/interventions/handler.py`

| Fact | Where |
|---|---|
| `class InterventionHandler(ABC)` | `handler.py:43` |
| `name` is an abstract property: a subclass must provide it (class attribute or property) | `handler.py:70-74` |
| `on_error` property, default `"throw"`; type `OnError = Literal["throw","proceed","deny"]` (`proceed` is fail-open, `deny` is fail-closed) | `handler.py:32-40`, `handler.py:76-79` |
| Lifecycle method names and signatures (all default to `Proceed()`; `**kwargs` is part of the signature) | |
| `before_invocation(self, event: BeforeInvocationEvent, **kwargs) -> Proceed \| Deny \| Guide \| Transform` | `handler.py:81-85` |
| `before_tool_call(self, event: BeforeToolCallEvent, **kwargs) -> Proceed \| Deny \| Guide \| Confirm \| Transform` | `handler.py:87-91` |
| `after_tool_call(self, event: AfterToolCallEvent, **kwargs) -> Proceed \| Transform` | `handler.py:93-95` |
| `before_model_call(self, event: BeforeModelCallEvent, **kwargs) -> Proceed \| Deny \| Guide \| Transform` | `handler.py:97-101` |
| `after_model_call(self, event: AfterModelCallEvent, **kwargs) -> Proceed \| Guide \| Transform` | `handler.py:103-107` |
| Methods may be `async def`; the registry awaits any awaitable result | `handler.py:51-56`, `registry.py:214-215` |
| Only methods **overridden at class level** are called; instance-level assignment is not detected | `handler.py:46-50`, `registry.py:59-63` |

Consequence for `DeciderGate`: the lifecycle stage is chosen by a constructor argument, but the registry decides what to hook by
comparing `type(handler).<method>` with the base class (`registry.py:61-63`). So `DeciderGate` overrides all five methods on the
class and each one returns `Proceed()` unless it is the configured stage (`src/decider_kit/intervention.py`).

## 2. Actions: `strands/interventions/actions.py`

All are frozen dataclasses with a non-init `type` string field. Import from `strands.interventions` (`interventions/__init__.py:24-31`);
`InterventionHandler` is also exported from top-level `strands` (`strands/__init__.py:8`).

| Action | Constructor fields (all keyword-able) | Where |
|---|---|---|
| `Proceed` | `reason: str \| None = None` (debug only, not shown to the model) | `actions.py:44-53` |
| `Deny` | `reason: str = ""` (shown to the model as the cancellation message) | `actions.py:56-61` |
| `Guide` | `feedback: str = ""`, `reason: str \| None = None` | `actions.py:64-85` |
| `Confirm` | `prompt: str = ""`, `reason: str \| None = None`, `response: Any = None`, `evaluate: Callable[[Any], bool] \| None = default_evaluate` | `actions.py:88-101` |
| `Transform` | `apply: Callable[[LifecycleEvent], None]`, `reason: str \| None = None` | `actions.py:104-114` |
| `InterventionAction = Proceed \| Deny \| Guide \| Confirm \| Transform` (FIVE actions) | `actions.py:117` |

`default_evaluate` accepts `True` and the strings `y`/`yes` (`actions.py:26-41`).

Action-to-stage matrix (the source of `ALLOWED_ACTIONS` in our code): `actions.py:120-136`.
`Deny` has no effect on `after_tool_call` / `after_model_call`; `Confirm` only works on `before_tool_call`; "no-op (warns at runtime)".

Gotchas written into the source:
* `Guide` on `after_model_call` discards the response and retries with the feedback; there is **no retry cap**, the handler must converge (`actions.py:72-75`).
  This is why `DeciderGate` does not default to `Guide` when the server is unavailable at `after_*` stages.
* `Guide` on `before_model_call` / `after_model_call` injects messages straight into `agent.messages`, bypassing session managers (`actions.py:77-80`).
* On `before_tool_call` and `before_invocation`, `Guide` sets the cancel message, so the model sees the feedback as the tool result (`registry.py:116-118`, `registry.py:142-144`).

## 3. Dispatch: `strands/interventions/registry.py`

* Duplicate handler names raise `ValueError` (`registry.py:45-49`): give every gate a unique `name`.
* Handlers run in registration order; `Deny` (and a failed `Confirm`) short-circuit; `Guide` feedback accumulates and is applied once at the end as
  `"[<handler name>] <feedback>"` joined by newlines (`registry.py:194-248`, feedback format at `registry.py:247`).
* `before_tool_call` application: `Deny` -> `event.cancel_tool = "DENIED: <reason>"`; `Confirm` -> `event.interrupt(handler_name, reason=prompt, [response=...])`,
  then `evaluate(result)`; failure sets `event.cancel_tool = "CONFIRMATION_FAILED: <prompt>"`; `Guide` -> `event.cancel_tool = "GUIDANCE: <feedback>"`;
  `Transform` -> `action.apply(event)` (`registry.py:127-151`).
* `InterruptException` always propagates, regardless of `on_error` (`registry.py:216-219`).
* Handler exceptions go through `_handle_error`: `throw` re-raises, `deny` returns `Deny("Handler threw: ...")`, `proceed` logs and skips (`registry.py:250-267`).
* Hook order constants: `INTERVENTION_OUTPUT = -90`, `INTERVENTION_INPUT = 90` (`strands/hooks/registry.py:30-41`); the registry uses INPUT for `before_*`, OUTPUT for `after_*` (`registry.py:65-95`).

## 4. How a handler is passed to an Agent: `strands/agent/agent.py`

* Parameter `interventions: list[InterventionHandler] | None = None` (`agent.py:224`); doc: cheap handlers first, LLM-based last (`agent.py:307-312`).
* The agent builds `InterventionRegistry(interventions or [], self.hooks)` in its constructor (`agent.py:579`).

```python
agent = Agent(model=..., tools=[...], interventions=[gate])
```

## 5. Event fields used by the gate: `strands/hooks/events.py`

* `BeforeToolCallEvent` (`events.py:221-257`): `agent` (from `HookEvent`, `hooks/registry.py:103-110`), `selected_tool`, `tool_use` (`ToolUse` TypedDict: `name`, `toolUseId`, `input`;
  `types/tools.py:71-85`), `invocation_state`, `cancel_tool: bool | str = False`.
* `AfterToolCallEvent` `events.py:261`, `BeforeInvocationEvent` `events.py:39`, `BeforeModelCallEvent` `events.py:318`, `AfterModelCallEvent` `events.py:348`.
* `event.interrupt(name, reason, response)` returns the human response or raises `InterruptException` (`strands/types/interrupt.py:82-111`).
* The conversation is `event.agent.messages` (list of `{"role", "content": [ {"text": ...} | toolUse | toolResult ]}`).

## 6. Observed behaviour (tests in `tests/test_agent_e2e.py`, real `Agent`, scripted fake model, no AWS)

* `Deny` from the gate: the tool does not run; the tool result is `status: error` with text `DENIED: <reason>`.
* `Guide`: the tool does not run; the tool result text is `GUIDANCE: [<gate name>] <feedback>`.
* `Proceed`: the tool runs.
* Decider server down with the default `on_unavailable` (`Confirm`, no preset response): `AgentResult.stop_reason == "interrupt"`, the tool does not run,
  `result.interrupts[0].reason` carries our prompt. A non-interactive run therefore PAUSES rather than proceeds: that is the fail-closed behaviour.

## 7. Decider server facts the client relies on (upstream `strands-labs/strands-decider` main, fetched 19.10.2026; NOT installed here)

* `src/strands_decider/schema.py`: wire types. noul answer `{"type":"noul","noul":p}` (no confidence field, `schema.py:35-45, 106-108`);
  choice `{"choice","probabilities","confidence"}` (`schema.py:111-115`); score `{"score","legend","probabilities","confidence"}` (`schema.py:118-124`);
  limits: choice 2..255 options (`schema.py:30,70-73`), score 2..10 levels (`schema.py:31-32,83-90`); request `images` list (`schema.py:101-103`).
* `src/strands_decider/server.py`: `GET /health` returns `{"status":"ok", "model", "checkpoint", ...}` (`server.py:79-92`);
  `POST /v1/systemone` adds `latency_ms` to the payload (`server.py:98-111`) and maps `ValueError` to HTTP 422 (`server.py:105-107`).
* The "at most 4 images" limit comes from `docs/vision.md` (via `docs/2026-10-19-brainstorm/platform-field-digest-2026-10-19.md`), not from `schema.py`; the client enforces it locally.
