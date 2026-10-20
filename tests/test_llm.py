"""The safety gate inside a REAL Strands Agent loop (scripted model), the picker and the writer.

No Bedrock, no AWS: the model is scripted, Decider is a fake HTTP transport, and the Bedrock
writer gets a fake Converse client.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable
from typing import Any

import pytest
from strands import Agent
from strands.models.model import Model

from decider_kit import Ledger, read_ledger
from menu_line_check import llm
from menu_line_check.llm import BedrockNotAllowed, BedrockWriter, FakeWriter, StrandsPicker
from menu_line_check.models import LineOutcome, Vehicle
from menu_line_check.ocr import parse_line_texts
from menu_line_check.tools import make_picker_tools
from menu_line_tool_selection import SAFETY_DENY_REASON, build_safety_gate

from .conftest import FakeServer, noul, ok_response

CIVIC = Vehicle(2019, "Honda", "Civic", 31200)
BRAKES = "Front brake pads & rotors ... $489.00"
CONDITIONER = "Engine oil conditioner ... $39.00"


class ScriptedModel(Model):
    """Each call plays the next step: ("tool", name, input) or ("text", words)."""

    def __init__(self, steps: list[tuple[Any, ...]]) -> None:
        self.steps = list(steps)
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
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        if step[0] == "tool":
            _, name, args = step
            start = {"toolUse": {"toolUseId": f"tu{self.calls}", "name": name}}
            yield {"contentBlockStart": {"start": start}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": step[1]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
                "metrics": {"latencyMs": 1},
            }
        }


def safety_server(p_safety: float) -> FakeServer:
    return FakeServer(lambda body: ok_response({"safety": noul(p_safety)}))


def tool_results(agent: Agent) -> list[dict[str, Any]]:
    return [
        block["toolResult"]
        for message in agent.messages
        for block in message["content"]
        if "toolResult" in block
    ]


def gated_agent(text: str, p_safety: float, tmp_path):
    ran: list[tuple[str, int, str]] = []
    server = safety_server(p_safety)
    gate = build_safety_gate(server.client(), ledger=Ledger(tmp_path / "l.jsonl"))
    model = ScriptedModel(
        [("tool", "decline_addon", {"line_index": 9, "line_text": text}), ("text", "done")]
    )
    agent = Agent(
        model=model,
        tools=make_picker_tools(lambda *a: ran.append(a)),
        interventions=[gate],
        callback_handler=None,
    )
    agent(f"Menu line 9: {text}")
    return agent, ran, server


# ------------------------------------------------------------ the gate in the real agent loop


def test_decline_on_the_brake_line_is_denied_and_the_tool_does_not_run(tmp_path):
    agent, ran, server = gated_agent(BRAKES, 0.93, tmp_path)
    assert ran == []
    (result,) = tool_results(agent)
    assert result["status"] == "error"
    assert result["content"][0]["text"] == f"DENIED: {SAFETY_DENY_REASON}"
    assert server.requests[0]["state"].endswith(f'"{BRAKES}"')
    assert read_ledger(tmp_path / "l.jsonl")[0]["action"]["type"] == "deny"


def test_decline_on_the_conditioner_line_proceeds_and_the_tool_runs(tmp_path):
    agent, ran, _ = gated_agent(CONDITIONER, 0.04, tmp_path)
    assert ran == [("decline_addon", 9, CONDITIONER)]
    (result,) = tool_results(agent)
    assert result["status"] == "success"


# ------------------------------------------------------------------------------ the picker


def line(text):
    return parse_line_texts([text])[0]


def test_picker_returns_the_tool_the_model_called_and_its_reason():
    model = ScriptedModel(
        [
            ("tool", "check_recall_warranty", {"line_index": 1, "line_text": "Tech diagnostic"}),
            ("text", "A diagnostic fee may be covered."),
        ]
    )
    picker = StrandsPicker(model=model)
    assert picker(line("Tech diagnostic $189.00"), CIVIC) == (
        "check_recall_warranty",
        "A diagnostic fee may be covered.",
    )
    assert picker.last_usage == {"inputTokens": 200, "outputTokens": 20}


def test_picker_behind_the_gate_moves_from_decline_to_owner_decides():
    model = ScriptedModel(
        [
            ("tool", "decline_addon", {"line_index": 9, "line_text": BRAKES}),
            ("tool", "owner_decides", {"line_index": 9, "line_text": BRAKES}),
            ("text", "Brakes are the owner's call."),
        ]
    )
    picker = StrandsPicker(safety_server(0.93).client(), model=model)
    assert picker(line(BRAKES), CIVIC)[0] == "owner_decides"


def test_picker_after_a_denied_decline_and_no_other_call_picks_nothing():
    model = ScriptedModel(
        [("tool", "decline_addon", {"line_index": 9, "line_text": BRAKES}), ("text", "ok")]
    )
    picker = StrandsPicker(safety_server(0.93).client(), model=model)
    tool, why = picker(line(BRAKES))
    assert tool is None and why.startswith("decline_addon was denied by the safety gate")


def test_picker_that_calls_no_tool_returns_none():
    picker = StrandsPicker(model=ScriptedModel([("text", "I am not sure.")]))
    assert picker(line("Mystery package $99.00")) == (None, "I am not sure.")


def test_bedrock_picker_needs_allow_bedrock(monkeypatch):
    with pytest.raises(BedrockNotAllowed, match="allow_bedrock=True"):
        StrandsPicker()
    built = {}

    class FakeBedrockModel:
        def __init__(self, **kwargs):
            built.update(kwargs)

    import strands.models.bedrock as bedrock

    monkeypatch.setattr(bedrock, "BedrockModel", FakeBedrockModel)
    monkeypatch.delenv("N2_PICKER_MODEL", raising=False)
    StrandsPicker(allow_bedrock=True)
    assert built["model_id"] == "us.amazon.nova-lite-v1:0" and built["region_name"] == "us-east-1"
    monkeypatch.setenv("N2_PICKER_MODEL", "us.amazon.nova-pro-v1:0")
    assert StrandsPicker(allow_bedrock=True).model_id == "us.amazon.nova-pro-v1:0"


def test_llm_module_does_not_import_boto3_at_import_time():
    source = (llm.__file__ and open(llm.__file__, encoding="utf-8").read()) or ""
    top_level = [ln for ln in source.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("boto3" in ln or "bedrock" in ln for ln in top_level)


# ------------------------------------------------------------------------------ the writer


class FakeConverse:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": f" {self.reply} "}]}},
            "usage": {"inputTokens": 50, "outputTokens": 12},
        }


def fuel_outcome() -> LineOutcome:
    fuel = line("Replace fuel pump module $1,089.00")
    return LineOutcome(
        line=fuel,
        route="tool",
        tool="check_recall_warranty",
        confidence=0.84,
        source="decider",
        verdict="ask",
        tool_result={"recall": {"campaign": "23V858000", "component": "FUEL PUMP"}, "note": "n"},
    )


def test_bedrock_writer_sends_the_facts_and_returns_the_text(monkeypatch):
    with pytest.raises(BedrockNotAllowed):
        BedrockWriter(FakeConverse("x"))
    monkeypatch.delenv("N2_WRITER_MODEL", raising=False)
    client = FakeConverse("Is this covered by recall 23V858000?")
    writer = BedrockWriter(client, allow_bedrock=True)
    outcome = fuel_outcome()
    assert writer.advisor_question(outcome.line, outcome, CIVIC) == (
        "Is this covered by recall 23V858000?"
    )
    call = client.calls[0]
    assert call["modelId"] == "us.amazon.nova-lite-v1:0"
    facts = json.loads(call["messages"][0]["content"][0]["text"])
    assert facts["recall_campaign"] == "23V858000" and facts["price"] == "$1,089.00"
    assert "recall campaign number" in call["system"][0]["text"]
    assert writer.usage == {"inputTokens": 50, "outputTokens": 12}


def test_fake_writer_questions_are_templates():
    outcome = fuel_outcome()
    assert FakeWriter().advisor_question(outcome.line, outcome, CIVIC) == (
        "Is this covered by recall 23V858000?"
    )
