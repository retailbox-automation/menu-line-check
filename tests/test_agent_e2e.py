"""End to end with the REAL Strands Agent loop and a scripted fake model (no Bedrock, no AWS).

The fake model proposes ``get_weather(location="Seattle")`` and then answers in text. The
Decider server is a fake HTTP transport. We check the real Agent honours the gate's action.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any

from strands import Agent, tool
from strands.interventions import Deny, Guide, Proceed
from strands.models.model import Model

from decider_kit import DeciderGate, Ledger, NoulQuestion, read_ledger, tool_call_state_builder

from .conftest import FakeServer, noul, ok_response


class ScriptedModel(Model):
    """First call: ask for the tool. Later calls: plain text."""

    def __init__(self) -> None:
        self.calls = 0

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> Any:
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(
        self, messages, tool_specs=None, system_prompt=None, **kwargs
    ) -> AsyncIterable:
        self.calls += 1
        if self.calls == 1:
            yield {"messageStart": {"role": "assistant"}}
            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": "tu1", "name": "get_weather"}}
                }
            }
            yield {
                "contentBlockDelta": {"delta": {"toolUse": {"input": '{"location": "Seattle"}'}}}
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 1},
            }
        }


CALLS: list[str] = []


@tool
def get_weather(location: str) -> str:
    """Get the weather.

    Args:
        location: City.
    """
    CALLS.append(location)
    return f"sunny in {location}"


def build_agent(grounded: float, tmp_path, decide):
    server = FakeServer(lambda body: ok_response({"grounded": noul(grounded)}))
    gate = DeciderGate(
        client=server.client(),
        questions={"grounded": NoulQuestion("Are the arguments grounded?")},
        state_builder=tool_call_state_builder(),
        decide=decide,
        ledger=Ledger(tmp_path / "l.jsonl"),
    )
    agent = Agent(
        model=ScriptedModel(), tools=[get_weather], interventions=[gate], callback_handler=None
    )
    return agent, server


def tool_results(agent) -> list[dict[str, Any]]:
    out = []
    for message in agent.messages:
        for block in message["content"]:
            if "toolResult" in block:
                out.append(block["toolResult"])
    return out


def test_deny_cancels_the_tool_in_the_real_agent_loop(tmp_path):
    CALLS.clear()
    agent, server = build_agent(
        0.05,
        tmp_path,
        lambda a: Deny(reason="guessed") if a["grounded"].probability < 0.5 else Proceed(),
    )
    agent("What's the weather?")
    assert CALLS == []  # the tool never ran
    result = tool_results(agent)[0]
    assert result["status"] == "error" and "DENIED: guessed" in result["content"][0]["text"]
    assert "get_weather" in server.requests[0]["state"] and "Seattle" in server.requests[0]["state"]
    assert read_ledger(tmp_path / "l.jsonl")[0]["action"]["type"] == "deny"


def test_guide_sends_feedback_as_the_tool_result(tmp_path):
    CALLS.clear()
    agent, _ = build_agent(0.05, tmp_path, lambda a: Guide(feedback="ask which city"))
    agent("What's the weather?")
    assert CALLS == []
    assert "GUIDANCE: [decider-gate] ask which city" in tool_results(agent)[0]["content"][0]["text"]


def test_proceed_lets_the_tool_run(tmp_path):
    CALLS.clear()
    agent, _ = build_agent(0.95, tmp_path, lambda a: Proceed())
    agent("What's the weather?")
    assert CALLS == ["Seattle"]
    assert tool_results(agent)[0]["status"] == "success"


def test_server_down_fails_closed_the_agent_pauses_with_an_interrupt(tmp_path):
    """Default on_unavailable at before_tool_call is Confirm; with no human response the real
    Agent stops with stop_reason 'interrupt' and the tool does not run."""
    CALLS.clear()
    server = FakeServer(lambda body: (503, {"detail": "down"}))
    gate = DeciderGate(
        client=server.client(),
        questions={"grounded": NoulQuestion("Are the arguments grounded?")},
        state_builder=tool_call_state_builder(),
        decide=lambda a: Proceed(),
        ledger=Ledger(tmp_path / "l.jsonl"),
    )
    agent = Agent(
        model=ScriptedModel(), tools=[get_weather], interventions=[gate], callback_handler=None
    )
    result = agent("What's the weather?")
    assert result.stop_reason == "interrupt" and CALLS == []
    assert "unavailable" in result.interrupts[0].reason
    assert read_ledger(tmp_path / "l.jsonl")[0]["unavailable"] == "http_5xx"
