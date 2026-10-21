"""PATTERN menu_line_tool_selection: which tool fits one line of a printed car-service menu.

Decider 2B answers three typed questions about one menu line (see ``questions.yaml``):

* ``readable`` (noul)  - can the service name and the price be read?
* ``safety``   (noul)  - is the line about a driving-safety part?
* ``tool``     (choice) - which of five tools fits the line?

Plain code then routes the line (:func:`decide_line`), in this order:

1. P(readable) < 0.5                                     -> ``retake``
2. tool choice is owner_decides (any confidence)         -> ``owner_decides``
3. tool choice is decline_addon and P(safety) >= 0.5     -> ``ask_llm`` (never declined here)
4. tool confidence >= that tool's threshold              -> ``tool`` (run the chosen tool)
5. otherwise                                             -> ``ask_llm`` (a general model picks)

The threshold follows the cost of a wrong action: declining is the one action the owner cannot
take back, so ``decline_addon`` needs 0.8; the lookups and the fee question only fetch
information or ask, so they act from 0.5 (:data:`DEFAULT_THRESHOLDS`, configurable).

The ``safety`` answer is a guardrail on that one irreversible action, not an override of the
lookups: on OCR text it says yes on every tire line (live run 2026-10-20). Where a general model
picks the tool, :func:`build_safety_gate` repeats the guardrail: a Strands ``before_tool_call``
gate that asks Decider the safety question about the line text the model passed to
``decline_addon`` and denies the call when P(safety) >= 0.5.

This module depends only on ``decider_kit``; it holds no menu parsing and no tool code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import yaml
from strands.interventions import Deny, InterventionAction, Proceed

from decider_kit import (
    Answer,
    ChoiceAnswer,
    DeciderClient,
    DeciderGate,
    Ledger,
    NoulAnswer,
    Question,
    parse_questions,
)

QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")

TOOLS: tuple[str, ...] = (
    "decline_addon",
    "check_schedule",
    "check_recall_warranty",
    "ask_about_fee",
    "owner_decides",
)
"""The five tools, in the order of the ``tool`` question's options."""

InputMode = Literal["ocr_text", "image"]
INPUT_MODES: tuple[str, ...] = ("ocr_text", "image")
CriteriaSet = Literal["spike", "short"]
CRITERIA_SETS: tuple[str, ...] = ("spike", "short")
Route = Literal["retake", "owner_decides", "tool", "ask_llm"]

READABLE_YES = 0.5
SAFETY_YES = 0.5

DEFAULT_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "decline_addon": 0.8,  # irreversible for the owner: high bar
        "check_schedule": 0.5,  # a lookup: a wrong pick costs one wasted check
        "check_recall_warranty": 0.5,  # a lookup
        "ask_about_fee": 0.5,  # a question to the advisor
    }
)
"""Act threshold per tool. ``owner_decides`` has none: it is followed at any confidence."""

# State captions used in the spike of 2026-10-20 (spike/n2-menu/run_n2_spike.py).
STATE_OCR_TEXT = 'OCR text of one line of a car dealer\'s printed service menu: "{text}"'
STATE_STRIP = "One line of a car dealer's printed service menu, photographed with a phone."
STATE_CELLS = (
    "One line of a car dealer's printed service menu. Image 1 is the service name, "
    "image 2 is its price."
)

SAFETY_DENY_REASON = "this line is a safety item; the owner decides"


def load_pattern_questions(
    mode: InputMode = "ocr_text",
    names: Iterable[str] | None = None,
    *,
    criteria: CriteriaSet = "spike",
) -> dict[str, Question]:
    """Load the questions in one wording. ``names`` selects a subset, in the given order.

    ``criteria="spike"`` (default) keeps the ``tool`` criteria sentences the spike measured;
    ``"short"`` swaps in shorter ones with the same meaning (measured 2026-10-21: accuracy 0.87,
    policy acts on 79/103 at 0.86). Only the ``tool`` criteria change; every instruction and the
    yes/no criteria stay as they are.
    """
    data = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if mode not in INPUT_MODES:
        raise ValueError(f"mode must be one of {list(INPUT_MODES)}, got {mode!r}")
    if criteria not in CRITERIA_SETS:
        raise ValueError(f"criteria must be one of {list(CRITERIA_SETS)}, got {criteria!r}")
    raw = dict(data[mode])
    if criteria == "short":
        raw["tool"] = {**raw["tool"], "criteria": data["criteria_short"]}
    questions = parse_questions(raw)
    if names is None:
        return questions
    return {name: questions[name] for name in names}


def ocr_state(text: str) -> str:
    """The state for the OCR-text wording: the line text, quoted, after a short caption."""
    return STATE_OCR_TEXT.format(text=text)


def image_state(n_images: int) -> str:
    """The state caption for the image wording: one strip, or name crop + price crop."""
    return STATE_CELLS if n_images >= 2 else STATE_STRIP


@dataclass(frozen=True)
class LineDecision:
    """The route for one menu line, and the numbers it was based on."""

    route: Route
    tool: str | None
    """The tool to run: the chosen tool for ``tool``, ``owner_decides`` for ``owner_decides``."""
    confidence: float | None
    """Confidence of Decider's tool choice."""
    p_readable: float
    p_safety: float
    decider_choice: str | None
    """Decider's top tool, whatever the route."""
    reason: str
    threshold: float | None = None
    """The act threshold of Decider's top tool, when one applied."""


def _noul(answers: Mapping[str, Answer], name: str) -> NoulAnswer:
    answer = answers[name]
    if not isinstance(answer, NoulAnswer):
        raise TypeError(f"answer {name!r} must be a noul answer, got {type(answer).__name__}")
    return answer


def resolve_thresholds(
    thresholds: Mapping[str, float] | None = None, act_threshold: float | None = None
) -> dict[str, float]:
    """Per-tool thresholds: ``thresholds`` over the defaults, or one ``act_threshold`` for all."""
    if act_threshold is not None:
        out = dict.fromkeys(DEFAULT_THRESHOLDS, float(act_threshold))
    else:
        out = dict(DEFAULT_THRESHOLDS)
        unknown = set(thresholds or {}) - set(DEFAULT_THRESHOLDS)
        if unknown:
            raise ValueError(
                f"unknown tool(s) {sorted(unknown)}; thresholds exist for {sorted(out)}"
            )
        out.update({k: float(v) for k, v in (thresholds or {}).items()})
    bad = {k: v for k, v in out.items() if not 0.0 <= v <= 1.0}
    if bad:
        raise ValueError(f"thresholds must be between 0 and 1, got {bad}")
    return out


def parse_thresholds(text: str) -> dict[str, float]:
    """``"decline_addon=0.8,check_schedule=0.5"`` -> a dict (validated by resolve_thresholds)."""
    out: dict[str, float] = {}
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        name, sep, value = part.partition("=")
        try:
            if not sep:
                raise ValueError
            out[name.strip()] = float(value)
        except ValueError:
            raise ValueError(f"thresholds must be tool=value pairs, got {part!r}") from None
    resolve_thresholds(out)
    return out


def decide_line(
    answers: Mapping[str, Answer],
    thresholds: Mapping[str, float] | None = None,
    *,
    act_threshold: float | None = None,
    readable_yes: float = READABLE_YES,
    safety_yes: float = SAFETY_YES,
) -> LineDecision:
    """Route one line from its ``readable``, ``safety`` and ``tool`` answers. Plain code only.

    ``thresholds`` overrides some or all of :data:`DEFAULT_THRESHOLDS`; ``act_threshold`` (kept
    for older callers) sets one threshold for every tool instead.
    """
    limits = resolve_thresholds(thresholds, act_threshold)
    readable = _noul(answers, "readable")
    safety = _noul(answers, "safety")
    tool = answers["tool"]
    if not isinstance(tool, ChoiceAnswer):
        raise TypeError(f"answer 'tool' must be a choice answer, got {type(tool).__name__}")
    p_r, p_s, conf, choice = readable.probability, safety.probability, tool.confidence, tool.choice
    limit = limits.get(choice)

    def made(route: Route, chosen: str | None, reason: str) -> LineDecision:
        return LineDecision(route, chosen, conf, p_r, p_s, choice, reason, limit)

    if p_r < readable_yes:
        return made("retake", None, f"P(readable)={p_r:.2f} < {readable_yes}")
    if choice == "owner_decides":
        return made("owner_decides", "owner_decides", f"tool choice is owner_decides at {conf:.2f}")
    if choice == "decline_addon" and p_s >= safety_yes:
        return made(
            "ask_llm",
            None,
            f"decline_addon with P(safety)={p_s:.2f} >= {safety_yes}: not declined automatically",
        )
    if limit is None:
        return made("ask_llm", None, f"unknown option {choice!r}")
    if conf >= limit:
        return made("tool", choice, f"{choice} at {conf:.2f} >= {limit}")
    return made("ask_llm", None, f"{choice} at {conf:.2f} < {limit}")


# ----------------------------------------------------------------------------- safety gate


class DeclineSafetyGate(DeciderGate):
    """A ``DeciderGate`` that only looks at the guarded tools; every other call proceeds."""

    def __init__(self, *, guarded_tools: Iterable[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._guarded = frozenset(guarded_tools)

    async def before_tool_call(self, event: Any, **kwargs: Any) -> InterventionAction:
        tool_use = getattr(event, "tool_use", None) or {}
        if tool_use.get("name") not in self._guarded:
            return Proceed()
        text = (tool_use.get("input") or {}).get("line_text")
        if not isinstance(text, str) or not text.strip():
            return Deny(reason="no line text was given, so the safety check cannot run")
        return await super().before_tool_call(event, **kwargs)


def build_safety_gate(
    client: DeciderClient,
    *,
    threshold: float = SAFETY_YES,
    ledger: Ledger | None = None,
    name: str = "menu-line-safety",
    guarded_tools: Iterable[str] = ("decline_addon",),
    on_unavailable: InterventionAction | None = None,
) -> DeciderGate:
    """A ``before_tool_call`` gate: deny ``decline_addon`` on a line Decider calls a safety item.

    The line text reaches the gate through the tool arguments (``line_text``). When Decider is
    unreachable the default is ``Deny``: an unchecked line cannot be declined automatically.
    """

    def state(event: Any) -> str:
        return ocr_state(str(event.tool_use["input"]["line_text"]))

    def decide(answers: Mapping[str, Answer]) -> InterventionAction:
        if _noul(answers, "safety").probability >= threshold:
            return Deny(reason=SAFETY_DENY_REASON)
        return Proceed()

    return DeclineSafetyGate(
        guarded_tools=guarded_tools,
        name=name,
        client=client,
        questions=load_pattern_questions("ocr_text", names=["safety"]),
        state_builder=state,
        decide=decide,
        lifecycle="before_tool_call",
        on_unavailable=on_unavailable
        or Deny(reason="the safety check is unavailable; the owner decides"),
        ledger=ledger,
    )
