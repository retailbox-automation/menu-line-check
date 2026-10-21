"""Test-only fakes for the menu check: scripted Decider answers for the 12-line fixture menu."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from menu_line_tool_selection import TOOLS

from .conftest import FakeServer, choice, noul, ok_response

FIXTURES = Path(__file__).parent / "fixtures"
TEXTRACT = FIXTURES / "textract_menu.json"
NHTSA = FIXTURES / "nhtsa_recalls_civic_2019.json"
SHEET_801 = FIXTURES / "sheet_801_vision.json"
SHEET_391 = FIXTURES / "sheet_391_vision.json"
NHTSA_FAMILY = FIXTURES / "nhtsa_recalls_family_2024.json"
NHTSA_MODEL_EMPTY = FIXTURES / "nhtsa_recalls_model_2024_empty.json"
NHTSA_SHEET_RECORDED = (NHTSA_FAMILY, (NHTSA_MODEL_EMPTY, "LEXUS", "RX 350", 2024))
"""Both recorded responses for the sheet's car (model: empty; model family: 4 campaigns), as
``NhtsaClient.from_recorded(*NHTSA_SHEET_RECORDED)`` takes them."""

# line name fragment -> (tool choice, confidence, P(safety))
MENU_ANSWERS: dict[str, tuple[str, float, float]] = {
    "Oil & filter change": ("check_schedule", 0.93, 0.05),
    "Engine oil conditioner": ("decline_addon", 0.91, 0.04),
    "Fuel injection cleaning": ("decline_addon", 0.86, 0.06),
    "Tire rotation": ("check_schedule", 0.56, 0.68),  # live run: safety yes on tire lines
    "Cabin air filter": ("check_schedule", 0.95, 0.03),
    "Replace fuel pump module": ("check_recall_warranty", 0.64, 0.2),
    "Tech diagnostic": ("check_recall_warranty", 0.72, 0.08),
    "Shop supplies": ("ask_about_fee", 0.97, 0.02),
    "Front brake pads": ("owner_decides", 0.88, 0.72),
    "Nitrogen tire fill": ("decline_addon", 0.78, 0.66),  # safety-flagged decline: ask_llm
    "Throttle body service": ("decline_addon", 0.83, 0.07),
    "Environmental disposal fee": ("ask_about_fee", 0.9, 0.02),
}

# Scripted (not measured) answers for the 11 lines of the real sheet, for offline tests.
SHEET_ANSWERS: dict[str, tuple[str, float, float]] = {
    "Engine Oil and Oil Filter": ("check_schedule", 0.9, 0.05),
    "Oil Conditioner": ("decline_addon", 0.92, 0.04),
    "Tire Rotation": ("check_schedule", 0.6, 0.7),
    "Cabin Filter": ("check_schedule", 0.7, 0.03),
    "Multi-Point Inspection": ("check_schedule", 0.85, 0.3),
    "Lifetime Protection": ("ask_about_fee", 0.55, 0.05),
    "Wheel Alignment": ("owner_decides", 0.8, 0.9),
    "Crankcase": ("decline_addon", 0.88, 0.06),
    "A/C Evaporator": ("decline_addon", 0.7, 0.05),
    "Shop Charges": ("ask_about_fee", 0.95, 0.02),
    "Fees": ("ask_about_fee", 0.93, 0.02),
}


def textract_response() -> dict[str, Any]:
    return json.loads(TEXTRACT.read_text(encoding="utf-8"))


def tool_answer(tool: str, conf: float) -> dict[str, Any]:
    rest = (1.0 - conf) / (len(TOOLS) - 1)
    return choice(tool, {t: (conf if t == tool else rest) for t in TOOLS}, conf)


def menu_responder(readable_image: float = 0.97, p_readable_text: float = 0.9):
    """Answer like Decider would for the fixture menu; images get only the readable answer."""

    def respond(body: dict[str, Any]) -> tuple[int, Any]:
        if "images" in body:
            return ok_response({"readable": noul(readable_image)}, latency_ms=20.0)
        state = body["state"]
        found = [
            v for fragment, v in {**MENU_ANSWERS, **SHEET_ANSWERS}.items() if fragment in state
        ]
        tool, conf, p_safety = found[0] if found else ("decline_addon", 0.5, 0.1)
        answers = {
            "readable": noul(p_readable_text),
            "safety": noul(p_safety),
            "tool": tool_answer(tool, conf),
        }
        return ok_response({k: v for k, v in answers.items() if k in body["questions"]})

    return respond


def menu_server(**kwargs: Any) -> FakeServer:
    return FakeServer(menu_responder(**kwargs))
