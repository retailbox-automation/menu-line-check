"""The general-model parts: the LLM tool picker and the advisor-question writer.

* :class:`StrandsPicker` - a Strands ``Agent`` on Amazon Bedrock (Nova by default) that picks
  exactly one tool for a menu line Decider was not sure about, behind the pattern's safety gate.
* :class:`BedrockWriter` - Bedrock Converse: rephrases the code-made question for the service
  advisor on each flagged line, and may add one sentence (no digits) to the code-made summary.
  Code rejects any new topic, make or model name, or number (:mod:`menu_line_check.writing`).
* :class:`FakePicker` / :class:`FakeWriter` - deterministic stand-ins for tests and the offline
  demo; no network.

Anything that calls Bedrock costs money, so both Bedrock classes refuse to be built unless the
caller passes ``allow_bedrock=True``, and every Bedrock-backed object carries a
:class:`~menu_line_check.spend.SpendCap` that refuses the next call once the cap is reached.
boto3 is never imported by this module at import time.
"""

from __future__ import annotations

import json
import os
from typing import Any

from strands import Agent

from decider_kit import DeciderClient, Ledger
from menu_line_tool_selection import build_safety_gate

from .baseline import keyword_tool
from .models import MenuLine, Vehicle
from .spend import SpendCap
from .tools import make_picker_tools
from .writing import TextSource

DEFAULT_PICKER_MODEL = "us.amazon.nova-lite-v1:0"
DEFAULT_WRITER_MODEL = "us.amazon.nova-lite-v1:0"
DEFAULT_REGION = "us-east-1"

PICKER_SYSTEM_PROMPT = (
    "You help a car owner at a dealer's service counter. You get ONE line of the printed service "
    "menu. Pick exactly one tool for this one menu line and call it once, passing the line "
    "number and the line text exactly as given. Brakes, tires, alignment and steering are "
    "always the owner's decision. After the tool call, say in one short sentence why it fits."
)
REPHRASE_SYSTEM_PROMPT = (
    "You get the facts about one line of a car service menu and a draft question the owner will "
    "ask the service advisor. Rewrite the draft as one short, polite question the owner says at "
    "the counter. Keep the same meaning. Add no new topics. Mention a recall only if a recall "
    "campaign number is in the facts. Use no brand, make or model names. Use no numbers that "
    "are not in the facts. Reply with the question only."
)
SUMMARY_SYSTEM_PROMPT = (
    "You get a car owner's service menu summary, already written. Add ONE plain sentence of "
    "advice for the owner at the counter, from the facts only. No digits and no numbers at all, "
    "no brand, make or model names, no greeting. Reply with the sentence only."
)


class BedrockNotAllowed(RuntimeError):  # noqa: N818 - reads as a sentence at the raise site
    """A Bedrock-backed object was requested without ``allow_bedrock=True``."""


# ---------------------------------------------------------------------------------- offline


class FakePicker:
    """Deterministic picker: the keyword rules, or a fixed tool per line index."""

    def __init__(self, choices: dict[int, str] | None = None) -> None:
        self.choices = dict(choices or {})

    def __call__(self, line: MenuLine, vehicle: Vehicle | None = None) -> tuple[str | None, str]:
        if line.index in self.choices:
            return self.choices[line.index], "fixed choice (offline stand-in for the LLM picker)"
        return keyword_tool(line.text), "keyword rule (offline stand-in for the LLM picker)"


class FakeWriter:
    """The offline writer: it rephrases nothing, so every question is the code-made draft and
    the summary is the code-made one (sources ``template``). Same structure as with Bedrock."""

    source: TextSource = "template"

    def rephrase_question(self, facts: dict[str, Any]) -> str | None:
        return None

    def summary_sentence(self, facts: dict[str, Any]) -> str | None:
        return None


# ---------------------------------------------------------------------------------- Bedrock


class StrandsPicker:
    """A Strands Agent that picks one tool for one menu line, behind the decline safety gate.

    ``model`` may be any Strands model (tests pass a scripted one; then no Bedrock is used).
    Without ``model`` a ``BedrockModel`` is built (model id from ``N2_PICKER_MODEL``, default
    Nova Lite, region us-east-1), and only when ``allow_bedrock=True``.

    ``spend`` is the run's :class:`SpendCap`: checked before each agent run, added to after it.
    A picker that really uses Bedrock always has one (a default ``SpendCap()`` when none is
    passed); with a scripted ``model`` it is whatever the caller passes (``None`` = no checks).

    ``decider`` enables the safety gate (``build_safety_gate``): a ``decline_addon`` call on a line
    Decider calls a safety item is denied. Pass ``decider=None`` to measure the LLM alone.
    """

    def __init__(
        self,
        decider: DeciderClient | None = None,
        *,
        allow_bedrock: bool = False,
        model: Any = None,
        model_id: str | None = None,
        region: str = DEFAULT_REGION,
        ledger: Ledger | None = None,
        spend: SpendCap | None = None,
    ) -> None:
        if model is None:
            if not allow_bedrock:
                raise BedrockNotAllowed(
                    "StrandsPicker calls Amazon Bedrock, which costs money: pass "
                    "allow_bedrock=True (CLI: --allow-bedrock) once the spend is approved."
                )
            from strands.models.bedrock import BedrockModel  # lazy: imports boto3

            self.model_id = model_id or os.environ.get("N2_PICKER_MODEL", DEFAULT_PICKER_MODEL)
            model = BedrockModel(
                model_id=self.model_id, region_name=region, temperature=0.0, max_tokens=300
            )
            spend = spend or SpendCap()
        else:
            self.model_id = model_id or type(model).__name__
        self._model = model
        self.spend = spend
        self._gate = build_safety_gate(decider, ledger=ledger) if decider is not None else None
        self.last_usage: dict[str, int] = {"inputTokens": 0, "outputTokens": 0}

    def __call__(self, line: MenuLine, vehicle: Vehicle | None = None) -> tuple[str | None, str]:
        if self.spend is not None:
            self.spend.check(self.model_id)
        picks: list[str] = []
        agent = Agent(
            model=self._model,
            tools=make_picker_tools(lambda tool, index, text: picks.append(tool)),
            system_prompt=PICKER_SYSTEM_PROMPT,
            interventions=[self._gate] if self._gate is not None else None,
            callback_handler=None,
        )
        car = f"Vehicle: {vehicle.label}.\n" if vehicle is not None else ""
        result = agent(f"{car}Menu line {line.index}: {line.text}")
        usage = result.metrics.accumulated_usage
        self.last_usage = {
            "inputTokens": int(usage.get("inputTokens", 0)),
            "outputTokens": int(usage.get("outputTokens", 0)),
        }
        if self.spend is not None:
            self.spend.add(
                self.model_id, self.last_usage["inputTokens"], self.last_usage["outputTokens"]
            )
        rationale = _text_of(result.message) or f"stop reason: {result.stop_reason}"
        if picks:
            return picks[-1], rationale
        if _was_denied(agent.messages):
            return None, "decline_addon was denied by the safety gate; not declined automatically"
        return None, rationale


def _text_of(message: dict[str, Any] | None) -> str:
    blocks = (message or {}).get("content", [])
    return " ".join(b["text"].strip() for b in blocks if isinstance(b, dict) and "text" in b)


def _was_denied(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        for block in message.get("content", []):
            result = block.get("toolResult") if isinstance(block, dict) else None
            if result and any("DENIED:" in c.get("text", "") for c in result.get("content", [])):
                return True
    return False


class BedrockWriter:
    """Rephrases code-made advisor questions, and adds one sentence to the code-made summary,
    through the Bedrock Converse API. It writes no fact of its own: the pipeline checks every
    text it returns (:mod:`menu_line_check.writing`) and keeps the draft when the check fails.

    ``client`` is a boto3 ``bedrock-runtime`` client created by the caller. Model id from
    ``N2_WRITER_MODEL`` (default Nova Lite). Refuses to be built without ``allow_bedrock=True``.
    Every call is checked against ``spend`` first and counted after; without one passed, a
    default ``SpendCap()`` is made, so a writer is never uncapped.
    """

    source: TextSource = "bedrock"

    def __init__(
        self,
        client: Any,
        *,
        allow_bedrock: bool = False,
        model_id: str | None = None,
        spend: SpendCap | None = None,
    ) -> None:
        if not allow_bedrock:
            raise BedrockNotAllowed(
                "BedrockWriter calls Amazon Bedrock, which costs money: pass allow_bedrock=True "
                "(CLI: --allow-bedrock) once the spend is approved."
            )
        self.client = client
        self.model_id = model_id or os.environ.get("N2_WRITER_MODEL", DEFAULT_WRITER_MODEL)
        self.spend = spend or SpendCap()
        self.usage = {"inputTokens": 0, "outputTokens": 0}

    def _converse(self, system: str, facts: dict[str, Any], max_tokens: int) -> str:
        self.spend.check(self.model_id)
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": json.dumps(facts, default=str)}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2},
        )
        usage = response.get("usage") or {}
        for key in self.usage:
            self.usage[key] += int(usage.get(key, 0))
        self.spend.add(
            self.model_id, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
        )
        blocks = response["output"]["message"]["content"]
        return "".join(b.get("text", "") for b in blocks).strip()

    def rephrase_question(self, facts: dict[str, Any]) -> str | None:
        """The model's rephrasing of ``facts["draft"]``; the caller validates it."""
        return self._converse(REPHRASE_SYSTEM_PROMPT, facts, 100)

    def summary_sentence(self, facts: dict[str, Any]) -> str | None:
        """One sentence to add to the code-made summary; the caller validates it."""
        return self._converse(SUMMARY_SYSTEM_PROMPT, facts, 80)


__all__ = [
    "BedrockNotAllowed",
    "BedrockWriter",
    "FakePicker",
    "FakeWriter",
    "StrandsPicker",
]
