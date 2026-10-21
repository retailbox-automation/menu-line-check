"""The general-model parts: the LLM tool picker and the advisor-question writer.

* :class:`StrandsPicker` - a Strands ``Agent`` on Amazon Bedrock (Nova by default) that picks
  exactly one tool for a menu line Decider was not sure about, behind the pattern's safety gate.
* :class:`BedrockWriter` - Bedrock Converse: one plain question for the service advisor per
  flagged line, and a two-sentence summary. The numbers on the summary card come from code.
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
from .models import LineOutcome, MenuLine, MenuReport, Vehicle
from .spend import SpendCap
from .tools import make_picker_tools

DEFAULT_PICKER_MODEL = "us.amazon.nova-lite-v1:0"
DEFAULT_WRITER_MODEL = "us.amazon.nova-lite-v1:0"
DEFAULT_REGION = "us-east-1"

PICKER_SYSTEM_PROMPT = (
    "You help a car owner at a dealer's service counter. You get ONE line of the printed service "
    "menu. Pick exactly one tool for this one menu line and call it once, passing the line "
    "number and the line text exactly as given. Brakes, tires, alignment and steering are "
    "always the owner's decision. After the tool call, say in one short sentence why it fits."
)
ADVISOR_SYSTEM_PROMPT = (
    "You write one short, polite question that a car owner can ask the service advisor about "
    "one line of a service menu. Use only the facts given. If a recall campaign number is "
    "given, ask whether the recall covers this line. One sentence, no greeting, no new numbers."
)
SUMMARY_SYSTEM_PROMPT = (
    "You write two plain sentences for a car owner about their service menu, from the facts "
    "given. Do not change, add or round any number. No greeting, no advice beyond the facts."
)


class BedrockNotAllowed(RuntimeError):  # noqa: N818 - reads as a sentence at the raise site
    """A Bedrock-backed object was requested without ``allow_bedrock=True``."""


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


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
    """Template questions and summary. Same inputs, same text."""

    def advisor_question(self, line: MenuLine, outcome: LineOutcome, vehicle: Vehicle) -> str:
        name = (line.name or f"line {line.index}").lower()
        result = outcome.tool_result
        if outcome.tool == "check_recall_warranty":
            if result.get("recall"):
                return f"Is this covered by recall {result['recall']['campaign']}?"
            return f"Could the warranty cover the {name}, and what exactly failed?"
        if outcome.tool == "ask_about_fee":
            price = line.price_text or "this"
            return f"What does the {price} {name} cover, and can it be removed?"
        if outcome.tool == "check_schedule":
            schedule = result.get("schedule") or {}
            if schedule.get("status") == "due" and schedule.get("every_visit"):
                return f"The {name} is done at every visit: what did it find?"
            if schedule.get("status") == "due":
                return (
                    f"My schedule has {schedule['service']} due at {schedule['due_at_mi']:,} mi: "
                    "is this that service, and what does it include?"
                )
            return f"Where does the maker's schedule list the {name}, and at what mileage?"
        if outcome.tool == "owner_decides":
            return f"What did the inspection find on the {name}, and how soon does it need doing?"
        return f"What is the {name} for, and is it required by the maker's schedule?"

    def summary(self, report: MenuReport) -> str:
        c = report.counts
        declines = _plural(c["decline"], "line", "lines")
        sheet_total = report.totals.total if report.totals else None
        if sheet_total is None:
            text = (
                f"{report.summary}. The menu totals ${report.menu_total:,.2f}; the "
                f"{declines} to decline come to ${report.total_for('decline'):,.2f}."
            )
        elif any(o.line.price is None for o in report.outcomes if o.verdict == "decline"):
            text = (
                f"{report.summary}. The sheet totals ${sheet_total:,.2f}; it does not price the "
                f"{declines} to decline one by one: ask what the total is without them."
            )
        else:
            text = (
                f"{report.summary}. The sheet totals ${sheet_total:,.2f}; the {declines} to "
                f"decline come to ${report.total_for('decline'):,.2f}."
            )
        if c["your_call"]:
            text += f" {_plural(c['your_call'], 'safety line is', 'safety lines are')} your call."
        if c["retake"]:
            text += f" Retake {_plural(c['retake'], 'line', 'lines')} before deciding."
        return text


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
    """Advisor questions and the summary through the Bedrock Converse API.

    ``client`` is a boto3 ``bedrock-runtime`` client created by the caller. Model id from
    ``N2_WRITER_MODEL`` (default Nova Lite). Refuses to be built without ``allow_bedrock=True``.
    Every call is checked against ``spend`` first and counted after; without one passed, a
    default ``SpendCap()`` is made, so a writer is never uncapped.
    """

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

    def advisor_question(self, line: MenuLine, outcome: LineOutcome, vehicle: Vehicle) -> str:
        result = outcome.tool_result
        facts = {
            "vehicle": vehicle.label,
            "menu_line": line.text,
            "price": line.price_text,
            "decision": outcome.verdict,
            "tool": outcome.tool,
            "tool_note": result.get("note"),
            "recall_campaign": (result.get("recall") or {}).get("campaign"),
            "recall_component": (result.get("recall") or {}).get("component"),
            "schedule": result.get("schedule"),
        }
        return self._converse(ADVISOR_SYSTEM_PROMPT, facts, 120)

    def summary(self, report: MenuReport) -> str:
        facts = {
            "card": report.summary,
            "counts": report.counts,
            "menu_total_usd": report.menu_total,
            "declined_total_usd": report.total_for("decline"),
            "your_call_lines": [o.line.name for o in report.outcomes if o.verdict == "your_call"],
        }
        return f"{report.summary}. {self._converse(SUMMARY_SYSTEM_PROMPT, facts, 160)}"


__all__ = [
    "BedrockNotAllowed",
    "BedrockWriter",
    "FakePicker",
    "FakeWriter",
    "StrandsPicker",
]
