"""The five per-line tools (plus ``retake_line``) and their Strands wrappers for the LLM picker.

Each plain tool returns a small dict with at least ``tool``, ``verdict`` (the summary bucket:
keep / decline / ask / your_call / retake) and ``note``. Only two tools look anything up:
``check_schedule`` (the illustrative schedule) and ``check_recall_warranty`` (NHTSA recalls).

The Strands wrappers (:func:`make_picker_tools`) only record which tool the LLM picked for which
line; the pipeline then runs the plain tool with the vehicle and the NHTSA client.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from strands import tool

from . import schedule as schedule_mod
from .models import MenuLine, Vehicle
from .nhtsa import NhtsaClient, NhtsaError, match_recall

ToolResult = dict[str, Any]


def decline_addon(line: MenuLine) -> ToolResult:
    return {
        "tool": "decline_addon",
        "verdict": "decline",
        "note": "Optional add-on: the maintenance schedule does not require it.",
    }


def check_schedule(line: MenuLine, vehicle: Vehicle) -> ToolResult:
    result = schedule_mod.check_schedule(line, vehicle)
    verdict = {"due": "keep", "not_due": "decline", "unknown": "ask"}[result.status]
    return {
        "tool": "check_schedule",
        "verdict": verdict,
        "schedule": result.to_dict(),
        "note": result.note,
    }


def check_recall_warranty(
    line: MenuLine, vehicle: Vehicle, nhtsa: NhtsaClient | None
) -> ToolResult:
    out: ToolResult = {"tool": "check_recall_warranty", "verdict": "ask", "recall": None}
    if nhtsa is None:
        out["note"] = (
            "Recall lookup not configured: ask whether a recall or the warranty covers it."
        )
        return out
    try:
        recalls = nhtsa.recalls(vehicle)
    except NhtsaError as exc:
        out["error"] = str(exc)
        out["note"] = "Recall lookup failed: ask whether a recall or the warranty covers it."
        return out
    out["recalls_checked"] = len(recalls)
    found = match_recall(line, recalls)
    if found is None:
        out["note"] = (
            f"No recall campaign for a {vehicle.year} {vehicle.make} {vehicle.model} matches this "
            "part: ask whether the warranty covers it."
        )
    else:
        out["recall"] = found.to_dict()
        out["note"] = (
            f"Recall {found.campaign} ({found.component}) covers {vehicle.year} {vehicle.make} "
            f"{vehicle.model} vehicles; whether it is open for this VIN needs the VIN check."
        )
    return out


def ask_about_fee(line: MenuLine) -> ToolResult:
    return {
        "tool": "ask_about_fee",
        "verdict": "ask",
        "note": "A fee rather than a service: ask what it covers and whether it is optional.",
    }


def owner_decides(line: MenuLine) -> ToolResult:
    return {
        "tool": "owner_decides",
        "verdict": "your_call",
        "note": "Driving-safety item: the owner decides; it is never declined automatically.",
    }


def retake_line(line: MenuLine) -> ToolResult:
    return {
        "tool": "retake_line",
        "verdict": "retake",
        "note": f"Retake line {line.index}: the service name or the price could not be read.",
    }


def ask_advisor(line: MenuLine) -> ToolResult:
    """No tool was chosen with enough confidence: the owner asks the advisor what it is."""
    return {
        "tool": None,
        "verdict": "ask",
        "note": "No tool was chosen with enough confidence: ask the advisor what this line is.",
    }


def run_tool(
    name: str, line: MenuLine, vehicle: Vehicle, nhtsa: NhtsaClient | None = None
) -> ToolResult:
    """Run one of the five tools (or ``retake_line``) by name."""
    if name == "check_schedule":
        return check_schedule(line, vehicle)
    if name == "check_recall_warranty":
        return check_recall_warranty(line, vehicle, nhtsa)
    plain: dict[str, Callable[[MenuLine], ToolResult]] = {
        "decline_addon": decline_addon,
        "ask_about_fee": ask_about_fee,
        "owner_decides": owner_decides,
        "retake_line": retake_line,
    }
    if name not in plain:
        raise ValueError(f"unknown tool {name!r}")
    return plain[name](line)


# --------------------------------------------------------------------- Strands wrappers

Recorder = Callable[[str, int, str], None]


def make_picker_tools(record: Recorder) -> list[Any]:
    """Strands ``@tool`` versions of the five tools. Each calls ``record(tool, index, text)``."""

    @tool
    def decline_addon(line_index: int, line_text: str) -> str:
        """Decline this line: an optional add-on product or chemical treatment the maker's
        schedule does not require (conditioners, cleanings, additives, nitrogen, protection
        packages).

        Args:
            line_index: The menu line number.
            line_text: The text of that menu line, exactly as given.
        """
        record("decline_addon", line_index, line_text)
        return f"decline_addon selected for line {line_index}."

    @tool
    def check_schedule(line_index: int, line_text: str) -> str:
        """Check this line against the maintenance schedule: routine maintenance scheduled by
        mileage (oil and filter, filters, rotation, fluid changes, spark plugs, inspection).

        Args:
            line_index: The menu line number.
            line_text: The text of that menu line, exactly as given.
        """
        record("check_schedule", line_index, line_text)
        return f"check_schedule selected for line {line_index}."

    @tool
    def check_recall_warranty(line_index: int, line_text: str) -> str:
        """Look up recalls and warranty: a diagnostic fee or the repair of one specific failed
        part, which an open recall or the warranty may cover.

        Args:
            line_index: The menu line number.
            line_text: The text of that menu line, exactly as given.
        """
        record("check_recall_warranty", line_index, line_text)
        return f"check_recall_warranty selected for line {line_index}."

    @tool
    def ask_about_fee(line_index: int, line_text: str) -> str:
        """Ask about this fee: a fee or surcharge rather than a service (shop supplies,
        disposal, handling, admin).

        Args:
            line_index: The menu line number.
            line_text: The text of that menu line, exactly as given.
        """
        record("ask_about_fee", line_index, line_text)
        return f"ask_about_fee selected for line {line_index}."

    @tool
    def owner_decides(line_index: int, line_text: str) -> str:
        """Leave this line to the owner: brakes, tires, alignment, steering or another part that
        affects driving safety.

        Args:
            line_index: The menu line number.
            line_text: The text of that menu line, exactly as given.
        """
        record("owner_decides", line_index, line_text)
        return f"owner_decides selected for line {line_index}."

    return [decline_addon, check_schedule, check_recall_warranty, ask_about_fee, owner_decides]
