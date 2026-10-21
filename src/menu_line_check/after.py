"""Two sheets of one visit: the dealer's recommended sheet and the one printed after declines.

The second sheet lists the lines the owner declined under a ``Declined`` header and prints new
totals. :func:`compare_sheets` matches each declined line to the line the agent checked on the
first sheet (by normalised text: OCR reads the same printed line slightly differently on two
photos, "Incl." against "Incl,") and records what the agent said about it, so the agent's
verdicts can be set against what the owner actually did.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .models import LineOutcome, MenuLine, MenuReport, SheetTotals

FUZZY_MATCH = 0.9
"""Least similarity (difflib ratio of the normalised texts) for a line matched without an
exact normalised match."""

TOTAL_ROWS: tuple[str, ...] = ("menu_total", "shop_charges", "tax", "fees", "total")


def normalise(text: str) -> str:
    """Lower-case letters and digits, one space between words: punctuation and OCR commas go."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _cents(value: float) -> float:
    return round(value, 2) + 0.0  # + 0.0 turns -0.0 into 0.0


@dataclass(frozen=True)
class DeclinedLine:
    """One line the owner declined (second sheet), with what the agent said on the first."""

    text: str
    """The line as read on the second sheet."""
    before_index: int | None
    """The matching line on the first sheet, or None when nothing matched."""
    before_text: str | None
    agent_verdict: str | None
    agent_tool: str | None
    agent_route: str | None
    agent_confidence: float | None

    @property
    def agent_said_decline(self) -> bool:
        return self.agent_verdict == "decline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "before_index": self.before_index,
            "before_text": self.before_text,
            "agent_verdict": self.agent_verdict,
            "agent_tool": self.agent_tool,
            "agent_route": self.agent_route,
            "agent_confidence": self.agent_confidence,
            "agent_said_decline": self.agent_said_decline,
        }


@dataclass(frozen=True)
class AfterBlock:
    """Totals before and after, the owner's declines, and how the agent's verdicts compare."""

    before: SheetTotals
    after: SheetTotals
    declined: tuple[DeclinedLine, ...]
    agent_declined_owner_kept: tuple[str, ...]
    """First-sheet lines the agent said to decline that the owner did not decline."""

    @property
    def before_total(self) -> float | None:
        return self.before.total

    @property
    def after_total(self) -> float | None:
        return self.after.total

    @property
    def difference(self) -> float | None:
        """Before total minus after total, in dollars and cents; None if either is missing."""
        if self.before.total is None or self.after.total is None:
            return None
        return _cents(self.before.total - self.after.total)

    def row_differences(self) -> dict[str, float | None]:
        """Per totals row: before minus after (None where either sheet lacks the row)."""
        out: dict[str, float | None] = {}
        for row in TOTAL_ROWS:
            b, a = getattr(self.before, row), getattr(self.after, row)
            out[row] = None if b is None or a is None else _cents(b - a)
        return out

    @property
    def agreed(self) -> int:
        """Owner-declined lines the agent also said to decline."""
        return sum(d.agent_said_decline for d in self.declined)

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_total": self.before_total,
            "after_total": self.after_total,
            "difference": self.difference,
            "row_differences": self.row_differences(),
            "totals_before": self.before.to_dict(),
            "totals_after": self.after.to_dict(),
            "owner_declined": len(self.declined),
            "agent_said_decline_on_owner_declined": self.agreed,
            "declined": [d.to_dict() for d in self.declined],
            "agent_declined_owner_kept": list(self.agent_declined_owner_kept),
        }


def _match(text: str, outcomes: list[LineOutcome], taken: set[int]) -> LineOutcome | None:
    key = normalise(text)
    free = [o for o in outcomes if o.line.index not in taken]
    for o in free:
        if normalise(o.line.text) == key:
            return o
    scored = [(SequenceMatcher(None, key, normalise(o.line.text)).ratio(), o) for o in free]
    best = max(scored, key=lambda pair: pair[0], default=None)
    return best[1] if best is not None and best[0] >= FUZZY_MATCH else None


def compare_sheets(
    report: MenuReport, after_lines: Iterable[MenuLine], after_totals: SheetTotals
) -> AfterBlock:
    """The after block: ``report`` is the agent's check of the first sheet.

    ``after_lines`` are the second sheet's lines (``parse_sheet(...).lines``); the declined ones
    carry ``declined_on_sheet``. Each is matched to one first-sheet line, at most once.
    """
    outcomes = list(report.outcomes)
    taken: set[int] = set()
    declined = []
    for line in after_lines:
        if not line.declined_on_sheet:
            continue
        found = _match(line.text, outcomes, taken)
        if found is not None:
            taken.add(found.line.index)
        declined.append(
            DeclinedLine(
                text=line.text,
                before_index=found.line.index if found else None,
                before_text=found.line.text if found else None,
                agent_verdict=found.verdict if found else None,
                agent_tool=found.tool if found else None,
                agent_route=found.route if found else None,
                agent_confidence=found.confidence if found else None,
            )
        )
    kept_but_declined = tuple(
        o.line.text for o in outcomes if o.verdict == "decline" and o.line.index not in taken
    )
    before = report.totals or SheetTotals()
    return AfterBlock(before, after_totals, tuple(declined), kept_but_declined)
