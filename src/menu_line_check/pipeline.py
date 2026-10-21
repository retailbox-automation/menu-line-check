"""``check_menu``: one Decider decision per menu line, plain-code routing, tools, summary.

Decider is asked about all lines first, concurrently (at most ``concurrency`` requests at a
time, default 12); everything after that runs per line, in line order. Per line:

1. One Decider request with two questions, ``safety`` and ``tool``, in the OCR-text wording.
   Without images, readability is the code check ``MenuLine.readable_by_code``: "OCR found a
   name and a price" (the spike measured that check above Decider's OCR-text ``readable``
   answer), so ``readable`` is not asked: fewer questions is what makes a line faster on a
   server that answers one request at a time. On a sheet that prints no price for any service
   line (bullet lines, prices only in the totals block; ``prices_listed`` False) the name alone
   makes a line readable: a missing price there is the layout, not an unreadable photo. If the
   line has images, a second request asks ``readable`` in the image wording.
2. ``decide_line`` routes the line with per-tool act thresholds: retake / owner_decides / tool /
   ask_llm. A decline on a line Decider flags as a safety item is never taken here.
3. ``ask_llm`` goes to ``llm_picker(line, vehicle) -> (tool, rationale)``. With no picker, or
   when it picks nothing, the line is "ask the advisor" (route ``ask_advisor``, source code).
   Whatever picker is plugged in, its ``decline_addon`` on a line with P(safety) >= 0.5 is not
   applied either: the line becomes "ask the advisor" (the Strands picker's gate already
   denies that call; this keeps the rule for any other picker).
4. The chosen tool runs (NHTSA for check_recall_warranty, the schedule for check_schedule).
   Whatever tool says "decline" (an add-on, or a service the maker's schedule does not list),
   the decline is not taken on a line with P(safety) >= 0.5: it becomes the owner's call, with
   the reason in the note (:func:`guard_decline`).
5. One ledger row per line. If Decider is unreachable for a line, nothing is declined: the
   line becomes the owner's call (route ``unavailable``).

Then ``writer`` (if given) writes an advisor question for every line the owner keeps, asks
about or decides (not for declines, not-due lines or retakes), and a plain summary. With an
NHTSA client the report also carries the recalls card for the whole car
(:func:`~menu_line_check.nhtsa.vehicle_recalls`), whatever the lines are.

A :class:`~menu_line_check.spend.SpendCapReached` from the picker or the writer is not caught
anywhere in here: it surfaces from ``check_menu`` and ends the run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from decider_kit import (
    Answer,
    DeciderClient,
    DeciderRequestError,
    DeciderUnavailable,
    Ledger,
    NoulAnswer,
    Question,
)
from menu_line_tool_selection import (
    SAFETY_YES,
    TOOLS,
    CriteriaSet,
    LineDecision,
    decide_line,
    image_state,
    load_pattern_questions,
    ocr_state,
    resolve_thresholds,
)

from .after import compare_sheets
from .models import (
    DeciderTiming,
    LineOutcome,
    MenuLine,
    MenuReport,
    SheetTotals,
    Vehicle,
    summary_card,
    summary_counts,
)
from .nhtsa import NhtsaClient, vehicle_recalls
from .spend import SpendCapReached
from .tools import ask_advisor, retake_line, run_tool
from .writing import Writer, needs_advisor_question, write_question, write_summary

logger = logging.getLogger(__name__)

LEDGER_GATE = "menu-line-check"

Picker = Callable[[MenuLine, Vehicle], tuple[str | None, str]]


@dataclass(frozen=True)
class LineAction:
    """What the ledger records as the action for one line (``type`` = the route)."""

    type: str
    reason: str


DEFAULT_CONCURRENCY = 12
"""Lines asked at the same time. Our reference Decider server answers one request at a time, so
this only overlaps the network."""


@dataclass(frozen=True)
class _Asked:
    """Decider's answers for one line, or the error that stopped them."""

    answers: dict[str, Answer] | None = None
    readable: NoulAnswer | None = None
    readable_source: str = "code"
    timing: DeciderTiming | None = None
    error: DeciderUnavailable | DeciderRequestError | None = None


TEXT_QUESTIONS: tuple[str, ...] = ("safety", "tool")
"""Asked on the OCR text of every line. ``readable`` is asked only on line images."""


def questions_per_line(line: MenuLine) -> tuple[str, ...]:
    """The Decider questions asked for this line (``readable_image`` = readable on its images)."""
    return (*TEXT_QUESTIONS, "readable_image") if line.images else TEXT_QUESTIONS


async def _aask_line(
    line: MenuLine,
    decider: DeciderClient,
    text_questions: dict[str, Question],
    image_questions: dict[str, Question],
) -> _Asked:
    try:
        first = await decider.aask(ocr_state(line.text), text_questions)
        answers = dict(first.answers)
        client_ms, server_ms, requests = first.client_ms, first.server_ms, 1
        if line.images:
            second = await decider.aask(
                image_state(len(line.images)), image_questions, images=line.images
            )
            readable = second.answers["readable"]
            if not isinstance(readable, NoulAnswer):
                raise DeciderUnavailable("bad_response", "image 'readable' answer is not noul")
            answers["readable_image"] = readable
            client_ms += second.client_ms
            server_ms = (
                None
                if server_ms is None or second.server_ms is None
                else server_ms + second.server_ms
            )
            requests, source = 2, "decider"
        else:
            readable = NoulAnswer(1.0 if line.readable_by_code else 0.0)
            source = "code"
    except (DeciderUnavailable, DeciderRequestError) as exc:
        return _Asked(error=exc)
    timing = DeciderTiming(client_ms, server_ms, requests, questions_per_line(line))
    return _Asked(answers, readable, source, timing)


async def _aask_all(
    lines: list[MenuLine], decider: DeciderClient, concurrency: int, criteria: CriteriaSet
) -> list[_Asked]:
    text_questions = load_pattern_questions("ocr_text", names=TEXT_QUESTIONS, criteria=criteria)
    image_questions = load_pattern_questions("image", names=["readable"])
    slots = asyncio.Semaphore(concurrency)

    async def one(line: MenuLine) -> _Asked:
        async with slots:
            return await _aask_line(line, decider, text_questions, image_questions)

    return list(await asyncio.gather(*(one(line) for line in lines)))


def _run(coro: Coroutine[Any, Any, list[_Asked]]) -> list[_Asked]:
    """``asyncio.run``, or in a worker thread when the caller already runs an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def ask_decider_for_all(
    lines: list[MenuLine],
    decider: DeciderClient,
    concurrency: int = DEFAULT_CONCURRENCY,
    criteria: CriteriaSet = "spike",
) -> tuple[list[_Asked], float]:
    """Ask Decider about every line, at most ``concurrency`` at a time.

    Returns one result per line, in line order whatever order the answers arrive in, and the
    wall-clock ms of the whole phase. A line Decider could not answer carries its error.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    started = time.perf_counter()
    asked = _run(_aask_all(lines, decider, concurrency, criteria))
    return asked, (time.perf_counter() - started) * 1000.0


def _finish_line(
    line: MenuLine,
    asked: _Asked,
    vehicle: Vehicle,
    thresholds: dict[str, float],
    llm_picker: Picker | None,
    nhtsa: NhtsaClient | None,
    ledger: Ledger | None,
) -> LineOutcome:
    """Route one line from its answers, run the tool, write its ledger row."""
    state = ocr_state(line.text)
    if asked.error is not None:
        exc = asked.error
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

    answers, timing = asked.answers or {}, asked.timing
    assert asked.readable is not None and timing is not None
    decision = decide_line(
        {"readable": asked.readable, "safety": answers["safety"], "tool": answers["tool"]},
        thresholds,
    )
    outcome = _resolve(line, vehicle, decision, asked.readable_source, llm_picker, nhtsa)
    outcome = replace(outcome, decider_ms=timing)
    if ledger is not None:
        ledger.record(
            gate=LEDGER_GATE,
            stage="line",
            state=state,
            answers=answers,
            action=LineAction(
                outcome.route,
                f"line {line.index}: asked {', '.join(timing.questions)}; readable by "
                f"{'code check' if asked.readable_source == 'code' else 'line image'}; "
                f"{decision.reason}; tool={outcome.tool}; source={outcome.source}; "
                f"verdict={outcome.verdict}",
            ),
            routes={"line": outcome.route},
            tool=outcome.tool,
            client_ms=timing.client_ms,
            server_ms=timing.server_ms,
        )
    return outcome


def guard_decline(result: dict[str, Any], p_safety: float | None) -> dict[str, Any]:
    """A decline from any tool on a safety-flagged line becomes the owner's call."""
    if result.get("verdict") != "decline" or p_safety is None or p_safety < SAFETY_YES:
        return result
    note = str(result.get("note", "")).strip()
    if note and not note.endswith((".", "!", "?")):
        note += "."
    return {
        **result,
        "verdict": "your_call",
        "guarded": "decline",
        "note": f"{note} Not declined: P(safety)={p_safety:.2f} >= {SAFETY_YES}, so the owner "
        "decides.".strip(),
    }


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
        result = guard_decline(run_tool(decision.tool, line, vehicle, nhtsa), decision.p_safety)
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
        except SpendCapReached:
            raise  # the spend cap stops the run; it is not a picker failure to fall back from
        except Exception as exc:  # the line falls back to "ask the advisor"; logged and recorded
            logger.warning("LLM picker failed on line %s: %r", line.index, exc)
            picked, rationale = None, f"LLM picker failed: {exc!r}"
    if picked == "decline_addon" and decision.p_safety >= SAFETY_YES:
        picked = None
        rationale = (
            f"the picker chose decline_addon, but P(safety)={decision.p_safety:.2f}: "
            "not declined automatically"
        )
    if picked in TOOLS:
        result = guard_decline(run_tool(picked, line, vehicle, nhtsa), decision.p_safety)
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
    thresholds: dict[str, float] | None = None,
    act_threshold: float | None = None,
    llm_picker: Picker | None = None,
    writer: Writer | None = None,
    nhtsa: NhtsaClient | None = None,
    ledger: Ledger | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    criteria: CriteriaSet = "spike",
    totals: SheetTotals | None = None,
    after_sheet: Any = None,
) -> MenuReport:
    """Check every line of one menu for one vehicle. Lines are checked one after another.

    ``thresholds`` overrides some per-tool act thresholds (defaults in the pattern);
    ``act_threshold`` sets one threshold for every tool instead. Decider is asked about all
    lines first, ``concurrency`` at a time; then each line is routed, its tool run and its
    ledger row written, in line order. ``criteria`` picks the ``tool`` criteria sentences:
    ``"spike"`` (default) or ``"short"`` (same meaning; measured 2026-10-21, acts on fewer lines).
    ``totals`` is the sheet's own totals block (``parse_sheet``), kept on the report and used
    by the summary. ``after_sheet`` (a ``parse_sheet`` result) is the sheet printed after the
    owner declined lines: the report's ``after`` block compares the two.

    With a ``writer``, every ``ask`` / ``your_call`` line gets an advisor question and the report
    gets its summary (:mod:`menu_line_check.writing`): code drafts both, the writer may only
    rephrase, and text that fails the checks is replaced by the draft. Sources and fallback
    reasons are on the report and, per line, in the ledger (stage ``question``).
    """
    limits = resolve_thresholds(thresholds, act_threshold)
    lines = list(lines)
    asked, wall_ms = ask_decider_for_all(lines, decider, concurrency, criteria)
    outcomes = [
        _finish_line(line, result, vehicle, limits, llm_picker, nhtsa, ledger)
        for line, result in zip(lines, asked, strict=True)
    ]
    if writer is not None:
        outcomes = [_with_question(o, writer, vehicle, ledger) for o in outcomes]
    counts = summary_counts(outcomes)
    report = MenuReport(
        vehicle,
        tuple(outcomes),
        counts,
        summary_card(counts),
        decider_wall_ms=round(wall_ms, 1),
        totals=totals,
        recalls=vehicle_recalls(vehicle, nhtsa) if nhtsa is not None else None,
    )
    if after_sheet is not None:
        report = replace(
            report, after=compare_sheets(report, after_sheet.lines, after_sheet.totals)
        )
    if writer is not None:
        summary = write_summary(writer, report)
        report = replace(
            report,
            summary_text=summary.text,
            summary_source=summary.source,
            summary_fallback=summary.fallback,
        )
    return report


def _with_question(
    outcome: LineOutcome, writer: Writer, vehicle: Vehicle, ledger: Ledger | None
) -> LineOutcome:
    if not needs_advisor_question(outcome):
        return outcome
    written = write_question(writer, outcome.line, outcome, vehicle)
    if ledger is not None:
        reason = f"line {outcome.line.index}: question_source={written.source}"
        if written.fallback:
            reason += f"; writer text rejected: {written.fallback}"
        ledger.record(
            gate=LEDGER_GATE,
            stage="question",
            state=written.text,
            answers=None,
            action=LineAction("question", reason),
        )
    return replace(
        outcome,
        advisor_question=written.text,
        question_source=written.source,
        question_fallback=written.fallback,
    )
