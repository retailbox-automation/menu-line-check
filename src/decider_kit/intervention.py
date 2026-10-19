"""``DeciderGate``: a generic Strands ``InterventionHandler`` backed by Decider 2B.

The gate holds no domain logic. The caller supplies

* ``questions``      - typed questions (see :mod:`decider_kit.questions`),
* ``state_builder``  - ``event -> state`` text built from the tool call and recent conversation,
* ``decide``         - ``answers -> Proceed | Deny | Guide | Confirm | Transform``.

The model classifies, plain code decides. Every decision is written to the ledger.

Strands facts this module relies on (verified in strands-agents 1.61.1, see
``docs/STRANDS-INTERVENTIONS-NOTES.md``):

* ``Agent(interventions=[handler])`` registers handlers; lifecycle methods may be ``async``.
* The registry registers a hook only for methods **overridden at class level**
  (``strands/interventions/registry.py:59-63``). So this class overrides all five lifecycle
  methods and each one is a no-op (``Proceed``) unless it is the configured ``lifecycle``.
* ``Confirm`` is only valid on ``before_tool_call``; ``Deny`` is not valid on
  ``after_tool_call`` / ``after_model_call``. An action that is not valid at the configured
  stage is coerced (``Deny`` if the stage allows it, else ``Proceed``) and the ledger notes it.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from strands.interventions import (
    Confirm,
    Deny,
    Guide,
    InterventionAction,
    InterventionHandler,
    OnError,
    Proceed,
    Transform,
)

from .client import Answer, DeciderClient, DeciderUnavailable
from .ledger import Ledger
from .policy import ThresholdPolicy
from .questions import Content, Question

logger = logging.getLogger(__name__)

Stage = Literal[
    "before_invocation",
    "before_tool_call",
    "after_tool_call",
    "before_model_call",
    "after_model_call",
]

# Mirrors the action-to-event matrix in strands/interventions/actions.py:120-136.
ALLOWED_ACTIONS: dict[str, tuple[type, ...]] = {
    "before_invocation": (Proceed, Deny, Guide, Transform),
    "before_tool_call": (Proceed, Deny, Guide, Confirm, Transform),
    "after_tool_call": (Proceed, Transform),
    "before_model_call": (Proceed, Deny, Guide, Transform),
    "after_model_call": (Proceed, Guide, Transform),
}


_ACTION_TYPES = (Proceed, Deny, Guide, Confirm, Transform)


@dataclass(frozen=True)
class GateInput:
    """What a state builder returns when it also has images (at most 4)."""

    state: Content
    images: tuple[str, ...] = ()


StateBuilder = Callable[[Any], Content | GateInput | Awaitable[Content | GateInput]]
DecideFn = Callable[[Mapping[str, Answer]], InterventionAction | Awaitable[InterventionAction]]
UnavailableFn = Callable[[Any, DeciderUnavailable], InterventionAction]


def recent_conversation(messages: list[dict[str, Any]] | None, max_messages: int = 8) -> str:
    """Flatten the last ``max_messages`` Strands messages to ``role: text`` lines (text only)."""
    lines: list[str] = []
    for message in (messages or [])[-max_messages:]:
        role = message.get("role", "unknown")
        for block in message.get("content", []):
            if isinstance(block, dict) and "text" in block:
                lines.append(f"{role}: {block['text']}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def tool_call_state_builder(max_messages: int = 8) -> StateBuilder:
    """A default state builder: recent conversation plus the proposed tool call, as plain text."""

    def build(event: Any) -> str:
        parts = [f"--- CONVERSATION ---\n{recent_conversation(event.agent.messages, max_messages)}"]
        tool_use = getattr(event, "tool_use", None)
        if tool_use:
            args = json.dumps(tool_use.get("input") or {}, ensure_ascii=False, default=str)
            parts.append(
                f"--- PROPOSED TOOL CALL ---\ntool: {tool_use.get('name')}\narguments: {args}"
            )
        return "\n\n".join(parts)

    return build


def _default_unavailable(stage: str, exc: DeciderUnavailable) -> InterventionAction:
    """Fail closed where the stage allows it."""
    if stage == "before_tool_call":
        return Confirm(
            prompt=f"The decision service is unavailable ({exc.kind}). Approve this tool call?",
            reason=f"decider unavailable: {exc.kind}",
        )
    if stage in ("before_invocation", "before_model_call"):
        return Deny(reason=f"decision service unavailable ({exc.kind})")
    # after_tool_call / after_model_call: Strands offers no blocking action here, and Guide on
    # after_model_call retries without a cap, so a down server would loop. Proceed and record it.
    return Proceed(reason=f"decider unavailable: {exc.kind}")


class DeciderGate(InterventionHandler):
    """Ask Decider typed questions about an event and map the answers to a Strands action."""

    def __init__(
        self,
        *,
        client: DeciderClient,
        questions: Mapping[str, Question],
        state_builder: StateBuilder,
        decide: DecideFn,
        name: str = "decider-gate",
        lifecycle: Stage = "before_tool_call",
        on_unavailable: InterventionAction | UnavailableFn | None = None,
        ledger: Ledger | None = None,
        policy: ThresholdPolicy | None = None,
        on_error: OnError = "deny",
    ) -> None:
        if lifecycle not in ALLOWED_ACTIONS:
            raise ValueError(
                f"lifecycle must be one of {sorted(ALLOWED_ACTIONS)}, got {lifecycle!r}"
            )
        if not questions:
            raise ValueError("at least one question is required")
        if isinstance(on_unavailable, _ACTION_TYPES) and not isinstance(
            on_unavailable, ALLOWED_ACTIONS[lifecycle]
        ):
            raise ValueError(f"on_unavailable action is not valid at {lifecycle}")
        self._name = name
        self._client = client
        self._questions = dict(questions)
        self._state_builder = state_builder
        self._decide = decide
        self._lifecycle = lifecycle
        self._on_unavailable = on_unavailable
        self._ledger = ledger
        self._policy = policy
        self._on_error = on_error

    @property
    def name(self) -> str:
        return self._name

    @property
    def on_error(self) -> OnError:
        return self._on_error

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    # -- the five lifecycle methods must all be overridden at class level (see module docstring)

    async def before_invocation(self, event: Any, **kwargs: Any) -> InterventionAction:
        return await self._run("before_invocation", event)

    async def before_tool_call(self, event: Any, **kwargs: Any) -> InterventionAction:
        return await self._run("before_tool_call", event)

    async def after_tool_call(self, event: Any, **kwargs: Any) -> InterventionAction:
        return await self._run("after_tool_call", event)

    async def before_model_call(self, event: Any, **kwargs: Any) -> InterventionAction:
        return await self._run("before_model_call", event)

    async def after_model_call(self, event: Any, **kwargs: Any) -> InterventionAction:
        return await self._run("after_model_call", event)

    # ------------------------------------------------------------------------------ internals

    async def _run(self, stage: str, event: Any) -> InterventionAction:
        if stage != self._lifecycle:
            return Proceed()

        built = self._state_builder(event)
        if inspect.isawaitable(built):
            built = await built
        gate_input = built if isinstance(built, GateInput) else GateInput(state=built)
        tool_use = getattr(event, "tool_use", None)
        tool = tool_use.get("name") if isinstance(tool_use, dict) else None

        try:
            response = await self._client.aask(
                gate_input.state, self._questions, gate_input.images or None
            )
        except DeciderUnavailable as exc:
            action = self._unavailable_action(stage, event, exc)
            action, coerced = self._coerce(stage, action)
            self._write(stage, gate_input, None, action, tool, None, None, exc.kind, coerced)
            return action

        decided = self._decide(response.answers)
        if inspect.isawaitable(decided):
            decided = await decided
        action, coerced = self._coerce(stage, decided)
        self._write(
            stage,
            gate_input,
            response,
            action,
            tool,
            response.client_ms,
            response.server_ms,
            None,
            coerced,
        )
        return action

    def _unavailable_action(
        self, stage: str, event: Any, exc: DeciderUnavailable
    ) -> InterventionAction:
        if self._on_unavailable is None:
            return _default_unavailable(stage, exc)
        if isinstance(self._on_unavailable, _ACTION_TYPES):
            return self._on_unavailable
        return self._on_unavailable(event, exc)

    @staticmethod
    def _coerce(stage: str, action: Any) -> tuple[InterventionAction, str | None]:
        if isinstance(action, ALLOWED_ACTIONS[stage]):
            return action, None
        original = getattr(action, "type", type(action).__name__)
        logger.warning("action %s is not valid at %s; coercing", original, stage)
        if Deny in ALLOWED_ACTIONS[stage]:
            return Deny(reason=f"{original} is not valid at {stage}"), str(original)
        return Proceed(reason=f"{original} is not valid at {stage}"), str(original)

    def _write(
        self,
        stage: str,
        gate_input: GateInput,
        response: Any,
        action: InterventionAction,
        tool: str | None,
        client_ms: float | None,
        server_ms: float | None,
        unavailable: str | None,
        coerced_from: str | None,
    ) -> None:
        if self._ledger is None:
            return
        answers = response.answers if response is not None else None
        routes = None
        if self._policy is not None and answers:
            routes = {n: r.route.value for n, r in self._policy.route_all(answers).items()}
        self._ledger.record(
            gate=self._name,
            stage=stage,
            state=gate_input.state,
            answers=answers,
            action=action,
            routes=routes,
            tool=tool,
            client_ms=client_ms,
            server_ms=server_ms,
            unavailable=unavailable,
            coerced_from=coerced_from,
        )
