"""DeciderGate with a fake Decider server and the REAL Strands action/registry classes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from strands.hooks.events import BeforeToolCallEvent
from strands.hooks.registry import HookRegistry
from strands.interventions import Confirm, Deny, Guide, InterventionHandler, Proceed, Transform
from strands.interventions.registry import InterventionRegistry

from decider_kit import (
    DeciderGate,
    DeciderUnavailable,
    GateInput,
    Ledger,
    NoulQuestion,
    ThresholdPolicy,
    read_ledger,
    tool_call_state_builder,
)

from .conftest import FakeServer, noul, ok_response

QUESTIONS = {
    "grounded": NoulQuestion("Are the arguments grounded?", {"true": "stated", "false": "guessed"}),
    "premature": NoulQuestion("Is it premature?"),
}


def decide(answers):
    if answers["grounded"].probability < 0.45:
        return Guide(feedback="ask the user for the city")
    if answers["premature"].probability >= 0.45:
        return Deny(reason="too early")
    return Proceed()


def make_event(tool="get_weather", args=None, messages=None):
    agent = SimpleNamespace(
        messages=messages or [{"role": "user", "content": [{"text": "What's the weather?"}]}]
    )
    return BeforeToolCallEvent(
        agent=agent,
        selected_tool=None,
        tool_use={"name": tool, "toolUseId": "t1", "input": args or {"location": "Seattle"}},
        invocation_state={},
    )


def server_with(grounded, premature):
    return FakeServer(
        lambda body: ok_response({"grounded": noul(grounded), "premature": noul(premature)})
    )


def gate_for(server, **kwargs):
    return DeciderGate(
        client=server.client(),
        questions=QUESTIONS,
        state_builder=tool_call_state_builder(),
        decide=decide,
        **kwargs,
    )


def run(coro):
    return asyncio.run(coro)


def test_gate_is_a_real_intervention_handler_and_maps_answers_to_real_actions():
    assert issubclass(DeciderGate, InterventionHandler)
    g = gate_for(server_with(0.1, 0.1))
    assert isinstance(run(g.before_tool_call(make_event())), Guide)
    assert isinstance(run(gate_for(server_with(0.9, 0.9)).before_tool_call(make_event())), Deny)
    assert isinstance(run(gate_for(server_with(0.9, 0.1)).before_tool_call(make_event())), Proceed)


def test_state_sent_to_server_has_conversation_and_tool_call():
    server = server_with(0.9, 0.1)
    run(gate_for(server).before_tool_call(make_event(args={"location": "Paris"})))
    state = server.requests[0]["state"]
    assert "user: What's the weather?" in state and "tool: get_weather" in state
    assert '"location": "Paris"' in state
    assert set(server.requests[0]["questions"]) == {"grounded", "premature"}


def test_images_from_state_builder_are_forwarded():
    server = server_with(0.9, 0.1)
    g = DeciderGate(
        client=server.client(),
        questions=QUESTIONS,
        decide=decide,
        state_builder=lambda e: GateInput("photo state", ("aGk=",)),
    )
    run(g.before_tool_call(make_event()))
    assert server.requests[0]["images"] == ["aGk="]


def test_other_lifecycle_stages_do_nothing_and_do_not_call_the_server():
    server = server_with(0.1, 0.1)
    g = gate_for(server)  # configured for before_tool_call
    for method in ("before_invocation", "after_tool_call", "before_model_call", "after_model_call"):
        assert isinstance(run(getattr(g, method)(make_event())), Proceed)
    assert server.requests == []


def test_lifecycle_argument_selects_the_stage():
    server = server_with(0.1, 0.1)
    g = gate_for(server, lifecycle="before_model_call")
    assert isinstance(run(g.before_tool_call(make_event())), Proceed) and server.requests == []
    assert isinstance(run(g.before_model_call(make_event())), Guide)
    with pytest.raises(ValueError, match="lifecycle"):
        gate_for(server, lifecycle="during_lunch")


def test_unavailable_default_is_fail_closed_confirm_at_before_tool_call():
    def down(body):
        return 503, {"detail": "down"}

    action = run(gate_for(FakeServer(down)).before_tool_call(make_event()))
    assert isinstance(action, Confirm) and "unavailable" in action.prompt


def test_unavailable_defaults_per_stage():
    server = FakeServer(lambda body: (503, {}))
    assert isinstance(
        run(gate_for(server, lifecycle="before_model_call").before_model_call(make_event())), Deny
    )
    assert isinstance(
        run(gate_for(server, lifecycle="before_invocation").before_invocation(make_event())), Deny
    )
    # no blocking action exists after the fact: Proceed, but the ledger records why
    assert isinstance(
        run(gate_for(server, lifecycle="after_tool_call").after_tool_call(make_event())), Proceed
    )


def test_custom_on_unavailable_action_and_callable():
    server = FakeServer(lambda body: (503, {}))
    fail_open = run(
        gate_for(server, on_unavailable=Proceed(reason="fail-open")).before_tool_call(make_event())
    )
    assert isinstance(fail_open, Proceed)
    seen = []

    def custom(event, exc):
        seen.append((event.tool_use["name"], exc.kind))
        return Deny(reason="no decider, no tool")

    action = run(gate_for(server, on_unavailable=custom).before_tool_call(make_event()))
    assert isinstance(action, Deny) and seen == [("get_weather", "http_5xx")]
    with pytest.raises(ValueError, match="not valid"):
        gate_for(server, lifecycle="after_tool_call", on_unavailable=Confirm(prompt="?"))


def test_action_not_valid_at_stage_is_coerced():
    server = server_with(0.9, 0.1)
    g = DeciderGate(
        client=server.client(),
        questions=QUESTIONS,
        lifecycle="before_model_call",
        state_builder=lambda e: "s",
        decide=lambda a: Confirm(prompt="sure?"),
    )
    assert isinstance(run(g.before_model_call(make_event())), Deny)
    g2 = DeciderGate(
        client=server.client(),
        questions=QUESTIONS,
        lifecycle="after_tool_call",
        state_builder=lambda e: "s",
        decide=lambda a: Deny(reason="x"),
    )
    assert isinstance(run(g2.after_tool_call(make_event())), Proceed)


def test_transform_action_passes_through():
    server = server_with(0.9, 0.1)
    t = Transform(apply=lambda e: None)
    g = DeciderGate(
        client=server.client(), questions=QUESTIONS, state_builder=lambda e: "s", decide=lambda a: t
    )
    assert run(g.before_tool_call(make_event())) is t


def test_every_decision_is_written_to_the_ledger(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    g = gate_for(server_with(0.1, 0.1), ledger=ledger, policy=ThresholdPolicy(), name="wx-gate")
    run(g.before_tool_call(make_event()))
    down = gate_for(FakeServer(lambda body: (503, {})), ledger=ledger)
    run(down.before_tool_call(make_event()))
    rows = read_ledger(tmp_path / "l.jsonl")
    assert len(rows) == 2
    a, b = rows
    assert (
        a["gate"] == "wx-gate" and a["stage"] == "before_tool_call" and a["tool"] == "get_weather"
    )
    assert a["action"]["type"] == "guide" and a["questions"] == ["grounded", "premature"]
    assert a["routes"]["grounded"] in {"act", "confirm"}
    assert a["server_ms"] == 12.5 and a["client_ms"] > 0 and len(a["state_sha256"]) == 64
    assert "What's the weather" not in (tmp_path / "l.jsonl").read_text()  # redacted by default
    assert (
        b["unavailable"] == "http_5xx" and b["action"]["type"] == "confirm" and b["answers"] == {}
    )


def test_coercion_is_noted_in_the_ledger(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    server = server_with(0.9, 0.1)
    g = DeciderGate(
        client=server.client(),
        questions=QUESTIONS,
        lifecycle="before_model_call",
        state_builder=lambda e: "s",
        decide=lambda a: Confirm(prompt="?"),
        ledger=ledger,
    )
    run(g.before_model_call(make_event()))
    row = read_ledger(tmp_path / "l.jsonl")[0]
    assert row["coerced_from"] == "confirm" and row["action"]["type"] == "deny"


def test_decide_exception_is_not_swallowed_and_on_error_defaults_to_deny():
    server = server_with(0.9, 0.1)

    def broken(answers):
        raise RuntimeError("policy bug")

    g = DeciderGate(
        client=server.client(), questions=QUESTIONS, state_builder=lambda e: "s", decide=broken
    )
    assert g.on_error == "deny"
    with pytest.raises(RuntimeError, match="policy bug"):
        run(g.before_tool_call(make_event()))


def test_async_decide_and_state_builder_are_awaited():
    server = server_with(0.9, 0.1)

    async def adecide(answers):
        return Proceed()

    async def abuild(event):
        return "async state"

    g = DeciderGate(
        client=server.client(), questions=QUESTIONS, state_builder=abuild, decide=adecide
    )
    assert isinstance(run(g.before_tool_call(make_event())), Proceed)
    assert server.requests[0]["state"] == "async state"


def test_strands_registry_wires_the_gate_and_cancels_the_tool_call():
    """Real InterventionRegistry + real HookRegistry: Deny sets cancel_tool, Guide sets guidance."""
    for grounded, premature, expected in (
        (0.9, 0.9, "DENIED: too early"),
        (0.1, 0.1, "GUIDANCE: [decider-gate] ask the user for the city"),
    ):
        gate = gate_for(server_with(grounded, premature))
        hooks = HookRegistry()
        registry = InterventionRegistry([gate], hooks)
        event = make_event()
        run(registry._on_before_tool_call(event))
        assert event.cancel_tool == expected
    allowed = make_event()
    run(
        InterventionRegistry(
            [gate_for(server_with(0.9, 0.1))], HookRegistry()
        )._on_before_tool_call(allowed)
    )
    assert allowed.cancel_tool is False


def test_unavailable_exception_type_is_exported_and_carries_kind():
    exc = DeciderUnavailable("timeout", "slow")
    assert exc.kind == "timeout" and "timeout" in str(exc)
