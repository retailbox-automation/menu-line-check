"""The example pattern runs end to end against a fake Decider server."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from strands.hooks.events import BeforeToolCallEvent
from strands.interventions import Guide, Proceed

from decider_kit import Ledger, NoulQuestion, read_ledger

from .conftest import FakeServer, noul, ok_response

HANDLER = Path(__file__).resolve().parents[1] / "patterns" / "example_grounding" / "handler.py"


def load_handler():
    spec = importlib.util.spec_from_file_location("example_grounding_handler", HANDLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event():
    return BeforeToolCallEvent(
        agent=SimpleNamespace(
            messages=[{"role": "user", "content": [{"text": "What's the weather?"}]}]
        ),
        selected_tool=None,
        tool_use={"name": "get_weather", "toolUseId": "t", "input": {"location": "Seattle"}},
        invocation_state={},
    )


def test_pattern_loads_questions_and_guides_on_ungrounded_args(tmp_path):
    handler = load_handler()
    server = FakeServer(
        lambda body: ok_response({"args_grounded": noul(0.05), "premature": noul(0.9)})
    )
    ledger = Ledger(tmp_path / "l.jsonl")
    gate = handler.build_gate(server.client(), ledger)
    action = asyncio.run(gate.before_tool_call(event()))
    assert isinstance(action, Guide) and "not grounded" in action.feedback
    sent = server.requests[0]["questions"]
    assert set(sent) == {"args_grounded", "premature"}
    assert sent["args_grounded"]["criteria"]["false"].startswith("an argument value was guessed")
    assert read_ledger(tmp_path / "l.jsonl")[0]["action"]["type"] == "guide"


def test_pattern_policy_branches():
    handler = load_handler()
    A = lambda g, p: {"args_grounded": handler.NoulAnswer(g), "premature": handler.NoulAnswer(p)}  # noqa: E731
    assert isinstance(handler.decide(A(0.9, 0.1)), Proceed)
    assert isinstance(handler.decide(A(0.9, 0.6)), Guide)
    assert isinstance(handler.decide(A(0.3, 0.1)), Guide)
    assert all(
        isinstance(q, NoulQuestion) for q in handler.load_questions(handler.QUESTIONS_PATH).values()
    )
