"""``check_menu``: one Decider decision per menu line, plain-code routing, tools, summary.

Per line:

1. One Decider request with the three pattern questions in the OCR-text wording. If the line
   has images, a second request asks ``readable`` in the image wording on those images.
   Without images, readability is the code check "OCR found a name and a price" (the spike
   measured that check above Decider's OCR-text ``readable`` answer); the OCR-text answer is
   still asked and written to the ledger.
2. ``decide_line`` routes the line: retake / owner_decides / tool / ask_llm.
3. ``ask_llm`` goes to ``llm_picker(line, vehicle) -> (tool, rationale)``. With no picker, or
   when it picks nothing, the line is "ask the advisor" (route ``ask_advisor``, source code).
4. The chosen tool runs (NHTSA for check_recall_warranty, the schedule for check_schedule).
5. One ledger row per line. If Decider is unreachable for a line, nothing is declined: the
   line becomes the owner's call (route ``unavailable``).

Then ``writer`` (if given) writes an advisor question for every line the owner keeps, asks
about or decides (not for declines, not-due lines or retakes), and a plain summary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Protocol

from decider_kit import (
    Answer,
    DeciderClient,
    DeciderRequestError,
    DeciderUnavailable,
    Ledger,
    NoulAnswer,
)
from menu_line_tool_selection import (
    DEFAULT_ACT_THRESHOLD,
    TOOLS,
    LineDecision,
    decide_line,
    image_state,
    load_pattern_questions,
    ocr_state,
)

from .models import (
    DeciderTiming,
    LineOutcome,
    MenuLine,
    MenuReport,
    Vehicle,
    summary_card,
    summary_counts,
)
from .nhtsa import NhtsaClient
from .tools import ask_advisor, retake_line, run_tool

logger = logging.getLogger(__name__)

LEDGER_GATE = "menu-line-check"

Picker = Callable[[MenuLine, Vehicle], tuple[str | None, str]]


class Writer(Protocol):
    def advisor_question(self, line: MenuLine, outcome: LineOutcome, vehicle: Vehicle) -> str: ...

    def summary(self, report: MenuReport) -> str: ...


@dataclass(frozen=True)
class LineAction:
    """What the ledger records as the action for one line (``type`` = the route)."""

    type: str
    reason: str


def needs_advisor_question(outcome: LineOutcome) -> bool:
    return outcome.verdict in ("keep", "ask", "your_call")


def _ask_decider(
    line: MenuLine, decider: DeciderClient
) -> tuple[dict[str, Answer], NoulAnswer, str, DeciderTiming]:
    text_questions = load_pattern_questions("ocr_text")
    first = decider.ask(ocr_state(line.text), text_questions)
    answers = dict(first.answers)
    client_ms, server_ms, requests = first.client_ms, first.server_ms, 1
    if line.images:
        second = decider.ask(
            image_state(len(line.images)),
            load_pattern_questions("image", names=["readable"]),
            images=line.images,
        )
        readable = second.answers["readable"]
        if not isinstance(readable, NoulAnswer):
            raise DeciderUnavailable("bad_response", "image 'readable' answer is not noul")
        answers["readable_image"] = readable
        client_ms += second.client_ms
        server_ms = (
            None if server_ms is None or second.server_ms is None else server_ms + second.server_ms
        )
        requests, source = 2, "decider"
    else:
        readable = NoulAnswer(1.0 if line.has_name_and_price else 0.0)
        answers["readable_code"] = readable
        source = "code"
    return answers, readable, source, DeciderTiming(client_ms, server_ms, requests)


def _check_line(
    line: MenuLine,
    vehicle: Vehicle,
    decider: DeciderClient,
    act_threshold: float,
    llm_picker: Picker | None,
    nhtsa: NhtsaClient | None,
    ledger: Ledger | None,
) -> LineOutcome:
    state = ocr_state(line.text)
    try:
        answers, readable, readable_source, timing = _ask_decider(line, decider)
    except (DeciderUnavailable, DeciderRequestError) as exc:
        kind = getattr(exc, "kind", None) or f"http_{getattr(exc, 'status', '4xx')}"
        outcome = LineOutcome(
            line=line,
            route="unavailable",
            tool=None,
            confidence=None,
            source="code",
            verdict="your_call",
            tool_result={
                "tool": None,
                "verdict": "your_call",
                "note": f"Decision service unavailable ({kind}): nothing is declined "
                "automatically; the owner decides.",
            },
        )
        if ledger is not None:
            ledger.record(
                gate=LEDGER_GATE,
                stage="line",
                state=state,
                answers=None,
                action=LineAction("unavailable", f"line {line.index}: {exc}"),
                unavailable=kind,
            )
        return outcome

    decision = decide_line(
        {"readable": readable, "safety": answers["safety"], "tool": answers["tool"]},
        act_threshold,
    )
    outcome = _resolve(line, vehicle, decision, readable_source, llm_picker, nhtsa)
    outcome = replace(outcome, decider_ms=timing)
    if ledger is not None:
        ledger.record(
            gate=LEDGER_GATE,
            stage="line",
            state=state,
            answers=answers,
            action=LineAction(
                outcome.route,
                f"line {line.index}: {decision.reason}; tool={outcome.tool}; "
                f"source={outcome.source}; verdict={outcome.verdict}",
            ),
            routes={"line": outcome.route},
            tool=outcome.tool,
            client_ms=timing.client_ms,
            server_ms=timing.server_ms,
        )
    return outcome


def _resolve(
    line: MenuLine,
    vehicle: Vehicle,
    decision: LineDecision,
    readable_source: str,
    llm_picker: Picker | None,
    nhtsa: NhtsaClient | None,
) -> LineOutcome:
    common = {
        "line": line,
        "confidence": decision.confidence,
        "p_readable": decision.p_readable,
        "p_safety": decision.p_safety,
        "decider_choice": decision.decider_choice,
    }
    if decision.route == "retake":
        result = retake_line(line)
        return LineOutcome(
            route="retake",
            tool="retake_line",
            source=readable_source,  # type: ignore[arg-type]
            verdict="retake",
            tool_result=result,
            **common,
        )
    if decision.route in ("owner_decides", "tool"):
        assert decision.tool is not None
        result = run_tool(decision.tool, line, vehicle, nhtsa)
        return LineOutcome(
            route=decision.route,
            tool=decision.tool,
            source="decider",
            verdict=result["verdict"],
            tool_result=result,
            **common,
        )

    # ask_llm
    rationale = "no LLM picker configured"
    picked: str | None = None
    if llm_picker is not None:
        try:
            picked, rationale = llm_picker(line, vehicle)
        except Exception as exc:  # the line falls back to "ask the advisor"; logged and recorded
            logger.warning("LLM picker failed on line %s: %r", line.index, exc)
            picked, rationale = None, f"LLM picker failed: {exc!r}"
    if picked in TOOLS:
        result = run_tool(picked, line, vehicle, nhtsa)
        return LineOutcome(
            route="ask_llm",
            tool=picked,
            source="llm",
            verdict=result["verdict"],
            tool_result=result,
            rationale=rationale,
            **common,
        )
    return LineOutcome(
        route="ask_advisor",
        tool=None,
        source="code",
        verdict="ask",
        tool_result=ask_advisor(line),
        rationale=rationale,
        **common,
    )


def check_menu(
    lines: Iterable[MenuLine],
    vehicle: Vehicle,
    decider: DeciderClient,
    *,
    act_threshold: float = DEFAULT_ACT_THRESHOLD,
    llm_picker: Picker | None = None,
    writer: Writer | None = None,
    nhtsa: NhtsaClient | None = None,
    ledger: Ledger | None = None,
) -> MenuReport:
    """Check every line of one menu for one vehicle. Lines are checked one after another."""
    outcomes = [
        _check_line(line, vehicle, decider, act_threshold, llm_picker, nhtsa, ledger)
        for line in lines
    ]
    if writer is not None:
        outcomes = [
            replace(o, advisor_question=writer.advisor_question(o.line, o, vehicle))
            if needs_advisor_question(o)
            else o
            for o in outcomes
        ]
    counts = summary_counts(outcomes)
    report = MenuReport(vehicle, tuple(outcomes), counts, summary_card(counts))
    if writer is not None:
        report = replace(report, summary_text=writer.summary(report))
    return report
