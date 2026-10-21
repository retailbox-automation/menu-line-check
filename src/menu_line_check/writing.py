"""The owner-facing text: advisor questions and the summary. Code drafts and checks; a model may
only rephrase.

* Only lines the owner still has to act on get a question: verdict ``ask`` or ``your_call``. A
  ``keep`` or ``decline`` line has its reason sentence and nothing else. (The recalls card carries
  its own fixed question.)
* :func:`draft_question` writes the question from the tool result, in code. A writer
  (``BedrockWriter``) may rephrase the draft; :func:`question_problem` then rejects the model's
  text, and the draft is used, when it mentions a recall the facts do not name, names the car's
  make or model, carries a number that is not in the facts, is not a question, or is longer than
  :data:`MAX_QUESTION` characters.
* The summary's numbers come from code only (:func:`summary_core`): the card, the sheet total or
  the priced-line totals, and the after-declines totals when there is a second sheet. A writer may
  add one plain sentence with no digits at all (:func:`sentence_problem`), or nothing.

Every question and the summary record their source, ``bedrock`` or ``template``, and why a
model's text was not used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .models import LineOutcome, MenuLine, MenuReport, Vehicle

TextSource = Literal["bedrock", "template"]

QUESTION_VERDICTS: tuple[str, ...] = ("ask", "your_call")
MAX_QUESTION = 160
MAX_SENTENCE = 200


class Writer(Protocol):
    """Rephrases code-made text. ``None`` means "no rephrasing": the draft is used."""

    source: TextSource

    def rephrase_question(self, facts: dict[str, Any]) -> str | None: ...

    def summary_sentence(self, facts: dict[str, Any]) -> str | None: ...


@dataclass(frozen=True)
class Written:
    text: str
    source: TextSource
    fallback: str | None = None
    """Why the writer's text was not used (None when it was, or when there was none)."""


def needs_advisor_question(outcome: LineOutcome) -> bool:
    return outcome.verdict in QUESTION_VERDICTS


def car(vehicle: Vehicle) -> str:
    """The car as the writer sees it: year and model, never the make."""
    return f"{vehicle.year} {vehicle.model}"


_ACTION = re.compile(r"^[A-Z]{2,}\s+(?=.*[a-z])")
_ASK_ABOUT = re.compile(r"^ask (?:your )?(?:service )?advisor about (?:the )?", re.IGNORECASE)


def plain_name(line: MenuLine) -> str:
    """The line name in lower case, without a leading upper-case action word ("PERFORM") or
    an "ask your service advisor about" opening."""
    name = line.name or f"line {line.index}"
    name = _ASK_ABOUT.sub("", name.strip())
    return _ACTION.sub("", name).strip().lower()


def _price(line: MenuLine) -> str | None:
    return line.price_text.replace("$ ", "$") if line.price_text else None


# ------------------------------------------------------------------------------------- drafts


def _schedule_draft(name: str, schedule: dict[str, Any], vehicle: Vehicle) -> str:
    status = schedule.get("status")
    if schedule.get("source") == "maker":
        listed = schedule.get("name")
        at = f"{schedule.get('due_at_mi') or vehicle.odometer_mi:,}"
        if status == "bundled":
            return (
                f"The maker's schedule lists {listed} at {at} mi: can you do just that, "
                f"without the {schedule.get('extra')}?"
            )
        if status == "unclear":
            return f"The maker's schedule lists {listed} at {at} mi: is this line that service?"
        if status == "due":
            return f"The maker's schedule lists {listed} at {at} mi: is this that service?"
        return (
            f"The maker's schedule does not list the {name} at {vehicle.odometer_mi:,} mi: "
            "why do you recommend it?"
        )
    if status == "due" and schedule.get("every_visit"):
        return f"The {name} is done at every visit: what did it find?"
    if status == "due":
        return (
            f"My schedule has {schedule['service']} due at {schedule['due_at_mi']:,} mi: "
            "is this that service, and what does it include?"
        )
    return f"Is this in the maker's schedule at {vehicle.odometer_mi:,} mi?"


def draft_question(line: MenuLine, outcome: LineOutcome, vehicle: Vehicle) -> str:
    """The advisor question for one line, from its tool result. Plain code, no model."""
    name = plain_name(line)
    result = outcome.tool_result
    if outcome.tool == "check_recall_warranty":
        if result.get("recall"):
            return f"Is this covered by recall {result['recall']['campaign']}?"
        return f"Could the warranty cover the {name}, and what exactly failed?"
    if outcome.tool == "ask_about_fee":
        price = _price(line)
        if price:
            return f"What does the {name} line of {price} cover, and can it be removed?"
        return f"What does the {name} cover, and can it be removed?"
    if outcome.tool == "check_schedule":
        return _schedule_draft(name, result.get("schedule") or {}, vehicle)
    if outcome.tool == "owner_decides":
        return f"What did the inspection find on the {name}, and how soon does it need doing?"
    return f"What is the {name} for, and is it required by the maker's schedule?"


def question_facts(
    line: MenuLine, outcome: LineOutcome, vehicle: Vehicle, draft: str
) -> dict[str, Any]:
    """Everything the writer may use: no make, no tool internals."""
    recall = outcome.tool_result.get("recall") or {}
    return {
        "vehicle": car(vehicle),
        "line": plain_name(line),
        "price": line.price_text,
        "verdict": outcome.verdict,
        "reason": outcome.tool_result.get("note"),
        "recall_campaign": recall.get("campaign"),
        "draft": draft,
    }


# --------------------------------------------------------------------------------- validation


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def _names(text: str, vehicle: Vehicle) -> str | None:
    low = f" {text.lower()} "
    if re.search(rf"\b{re.escape(vehicle.make.lower())}\b", low):
        return "names the make"
    model = " ".join(vehicle.model.lower().split())
    if model and re.search(rf"\b{re.escape(model)}\b", " ".join(low.split())):
        return "names the model"
    return None


def question_problem(text: str, facts: dict[str, Any], vehicle: Vehicle) -> str | None:
    """Why a model's question cannot be used, or None when it can."""
    if not text:
        return "empty"
    if len(text) > MAX_QUESTION:
        return f"longer than {MAX_QUESTION} characters"
    if not text.endswith("?"):
        return "not a question"
    if "recall" in text.lower() and not facts.get("recall_campaign"):
        return "mentions a recall but the facts have no campaign"
    named = _names(text, vehicle)
    if named:
        return named
    extra = _numbers(text) - _numbers(json.dumps(facts, default=str))
    if extra:
        return f"number not in the facts: {', '.join(sorted(extra))}"
    return None


def sentence_problem(text: str, vehicle: Vehicle) -> str | None:
    """Why a model's summary sentence cannot be used, or None when it can."""
    if not text:
        return "empty"
    if re.search(r"\d", text):
        return "contains a digit"
    if len(text) > MAX_SENTENCE:
        return f"longer than {MAX_SENTENCE} characters"
    return _names(text, vehicle)


def _clean(text: str | None) -> str:
    return (text or "").strip().strip('"').strip("“”").strip()


def write_question(
    writer: Writer | None, line: MenuLine, outcome: LineOutcome, vehicle: Vehicle
) -> Written:
    """The draft, or the writer's rephrasing of it when that passes :func:`question_problem`."""
    draft = draft_question(line, outcome, vehicle)
    if writer is None or writer.source == "template":
        return Written(draft, "template")
    facts = question_facts(line, outcome, vehicle, draft)
    text = _clean(writer.rephrase_question(facts))
    problem = question_problem(text, facts, vehicle)
    if problem:
        return Written(draft, "template", problem)
    return Written(text, writer.source)


# ------------------------------------------------------------------------------------- summary


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def summary_core(report: MenuReport) -> str:
    """The card and every number of the summary, composed in code."""
    c = report.counts
    parts = [f"{report.summary}."]
    total = report.totals.total if report.totals else None
    if total is not None:
        parts.append(f"Sheet total ${total:,.2f}.")
        if report.after is not None and report.after.after_total is not None:
            parts.append(
                f"After the owner's declines: ${report.after.after_total:,.2f}, "
                f"${report.after.difference:,.2f} less."
            )
    else:
        parts.append(
            f"The priced lines total ${report.menu_total:,.2f}; the "
            f"{_plural(c['decline'], 'line', 'lines')} to decline come to "
            f"${report.total_for('decline'):,.2f}."
        )
    if c["your_call"]:
        parts.append(f"{_plural(c['your_call'], 'safety line is', 'safety lines are')} your call.")
    if c["retake"]:
        parts.append(f"Retake {_plural(c['retake'], 'line', 'lines')} before deciding.")
    return " ".join(parts)


def summary_facts(report: MenuReport) -> dict[str, Any]:
    """What the writer may know for its one sentence: no make, and it may not use numbers."""
    return {
        "vehicle": car(report.vehicle),
        "summary": summary_core(report),
        "lines": [
            {"line": plain_name(o.line), "verdict": o.verdict, "reason": o.tool_result.get("note")}
            for o in report.outcomes
        ],
    }


def write_summary(writer: Writer | None, report: MenuReport) -> Written:
    """The code-made summary, plus the writer's one digit-free sentence when it passes."""
    core = summary_core(report)
    if writer is None or writer.source == "template":
        return Written(core, "template")
    sentence = _clean(writer.summary_sentence(summary_facts(report)))
    problem = sentence_problem(sentence, report.vehicle)
    if problem:
        return Written(core, "template", problem)
    return Written(f"{core} {sentence}", writer.source)
