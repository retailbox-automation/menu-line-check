"""Value types for the menu check: vehicle, menu line, per-line outcome, report."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from menu_line_tool_selection import LineDecision

if TYPE_CHECKING:
    from .after import AfterBlock
    from .nhtsa import VehicleRecalls

__all__ = [
    "BBox",
    "DeciderTiming",
    "LineDecision",
    "LineOutcome",
    "MenuLine",
    "MenuReport",
    "OutcomeRoute",
    "SheetTotals",
    "Source",
    "Vehicle",
    "Verdict",
    "summary_card",
]

Verdict = Literal["keep", "decline", "ask", "your_call", "retake"]
"""The owner-facing bucket of a line: what the summary card counts."""

OutcomeRoute = Literal["retake", "owner_decides", "tool", "ask_llm", "ask_advisor", "unavailable"]
"""``decide_line`` routes plus ``ask_advisor`` (no LLM picker, or it picked nothing) and
``unavailable`` (Decider could not be reached for this line)."""

Source = Literal["decider", "llm", "code"]

VERDICT_LABELS: dict[str, str] = {
    "keep": "Keep",
    "decline": "Decline",
    "ask": "Ask",
    "your_call": "Your call",
    "retake": "Retake",
}


@dataclass(frozen=True)
class Vehicle:
    year: int
    make: str
    model: str
    odometer_mi: int

    @classmethod
    def parse(cls, text: str) -> Vehicle:
        """Parse ``"<year> <make> <model> <odometer>"``.

        The make is one word, the model is everything between it and the odometer, and a
        trailing ``mi`` is allowed.
        """
        tokens = text.split()
        if tokens and tokens[-1].lower() in ("mi", "miles"):
            tokens.pop()
        if len(tokens) < 4 or not tokens[0].isdigit():
            raise ValueError(f"vehicle must be 'year make model odometer', got {text!r}")
        odometer = tokens[-1].replace(",", "")
        if not odometer.isdigit():
            raise ValueError(f"vehicle must be 'year make model odometer', got {text!r}")
        return cls(int(tokens[0]), tokens[1], " ".join(tokens[2:-1]), int(odometer))

    @property
    def label(self) -> str:
        return f"{self.year} {self.make} {self.model}, {self.odometer_mi:,} mi"


class BBox(NamedTuple):
    """Normalised bounding box (0..1 of the page), the Textract convention."""

    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def union(self, other: BBox | None) -> BBox:
        if other is None:
            return self
        left, top = min(self.left, other.left), min(self.top, other.top)
        right, bottom = max(self.right, other.right), max(self.bottom, other.bottom)
        return BBox(left, top, right - left, bottom - top)


_LETTERS = re.compile(r"[A-Za-z]{2}")


@dataclass(frozen=True)
class MenuLine:
    """One line of a service menu or service sheet, as OCR read it."""

    index: int
    """1-based position on the menu."""
    text: str
    """The line as one string; this is what Decider reads."""
    name: str
    price: float | None
    bbox: BBox | None
    """Whole line (name, description, price); None when the OCR gave no geometry."""
    images: tuple[str, ...] = ()
    """Base64 PNG crops of the line (strip, or name + price cells); empty without a photo."""
    price_text: str | None = None
    description: str | None = None
    name_bbox: BBox | None = None
    price_bbox: BBox | None = None
    declined_on_sheet: bool = False
    """The line is listed under a "Declined" section header of the sheet it was read from."""
    prices_listed: bool = True
    """False on a sheet that prints no price for any service line (bullet lines, totals only):
    there a missing price is how the sheet is printed, not an OCR failure."""

    @property
    def has_name_and_price(self) -> bool:
        """OCR found a name (two letters or more) and a price."""
        return self.price is not None and bool(_LETTERS.search(self.name))

    @property
    def readable_by_code(self) -> bool:
        """The code readability check used when no line image is sent to Decider.

        A name and a price, as on a priced menu. On a sheet that lists no per-line prices
        (``prices_listed`` False) the name alone is enough.
        """
        if self.prices_listed:
            return self.has_name_and_price
        return bool(_LETTERS.search(self.name))


@dataclass(frozen=True)
class SheetTotals:
    """The totals block of a sheet; None for a row the sheet does not print."""

    menu_total: float | None = None
    """Printed as Menu Total or Subtotal."""
    shop_charges: float | None = None
    """Printed as Shop Charges or Shop Supplies."""
    tax: float | None = None
    fees: float | None = None
    """Printed as Fees or Fee."""
    total: float | None = None
    """Printed as TOTAL: what the owner pays."""

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class DeciderTiming:
    client_ms: float
    """Wall clock around the HTTP calls for this line (all requests)."""
    server_ms: float | None
    """Sum of server-reported ``latency_ms``; None if the server did not report it."""
    requests: int
    questions: tuple[str, ...] = ()
    """The Decider questions asked for this line (``readable_image`` = readable on its images)."""


@dataclass(frozen=True)
class LineOutcome:
    line: MenuLine
    route: OutcomeRoute
    tool: str | None
    """The tool that ran (``owner_decides`` and ``retake_line`` included); None for ask_advisor."""
    confidence: float | None
    """Decider's tool-choice confidence, when Decider answered."""
    source: Source
    """Who chose the route: Decider, the LLM picker, or plain code."""
    verdict: Verdict
    tool_result: dict[str, Any] = field(default_factory=dict)
    advisor_question: str | None = None
    decider_ms: DeciderTiming | None = None
    p_readable: float | None = None
    p_safety: float | None = None
    decider_choice: str | None = None
    rationale: str | None = None


def summary_counts(outcomes: tuple[LineOutcome, ...] | list[LineOutcome]) -> dict[str, int]:
    counts = Counter(o.verdict for o in outcomes)
    return {verdict: counts.get(verdict, 0) for verdict in VERDICT_LABELS}


def summary_card(counts: dict[str, int]) -> str:
    """``Keep 3 · Decline 4 · Ask 4 · Your call 1`` (Retake only when there is one)."""
    parts = [f"{VERDICT_LABELS[v]} {counts.get(v, 0)}" for v in ("keep", "decline", "ask")]
    parts.append(f"Your call {counts.get('your_call', 0)}")
    if counts.get("retake"):
        parts.append(f"Retake {counts['retake']}")
    return " · ".join(parts)


@dataclass(frozen=True)
class MenuReport:
    vehicle: Vehicle
    outcomes: tuple[LineOutcome, ...]
    counts: dict[str, int]
    summary: str
    """The card line, computed by code."""
    summary_text: str | None = None
    """Plain-English summary from the writer, if one was given."""
    decider_wall_ms: float | None = None
    """Wall clock of the Decider phase (all lines, asked concurrently)."""
    totals: SheetTotals | None = None
    """The sheet's own totals block, when the sheet prints one."""
    recalls: VehicleRecalls | None = None
    """Recall campaigns for the car's make, model and year (the recalls card)."""
    after: AfterBlock | None = None
    """The comparison with a second sheet printed after the owner declined lines."""

    @property
    def menu_total(self) -> float:
        """Sum of the line prices (not the sheet's own totals, see ``totals``)."""
        return round(sum(o.line.price or 0.0 for o in self.outcomes), 2)

    def total_for(self, verdict: Verdict) -> float:
        return round(sum(o.line.price or 0.0 for o in self.outcomes if o.verdict == verdict), 2)

    def to_dict(self) -> dict[str, Any]:
        def outcome(o: LineOutcome) -> dict[str, Any]:
            data = asdict(o)
            data["line"].pop("images", None)
            data["line"]["has_images"] = bool(o.line.images)
            return data

        return {
            "vehicle": asdict(self.vehicle),
            "summary": self.summary,
            "summary_text": self.summary_text,
            "counts": dict(self.counts),
            "menu_total": self.menu_total,
            "declined_total": self.total_for("decline"),
            "decider_wall_ms": self.decider_wall_ms,
            "totals": self.totals.to_dict() if self.totals else None,
            "recalls": self.recalls.to_dict() if self.recalls else None,
            "after": self.after.to_dict() if self.after else None,
            "outcomes": [outcome(o) for o in self.outcomes],
        }
