"""Dealer sheets with bullet lines and a totals block (the real sheets, redacted)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from menu_line_check.models import SheetTotals
from menu_line_check.ocr import parse_sheet, parse_textract, total_field

FIXTURES = Path(__file__).parent / "fixtures"

SERVICES_801 = [
    "REPLACE Engine Oil and Oil Filter",
    "INSTALL Premium Engine Oil Conditioner",
    "PERFORM Tire Rotation",
    "REPLACE Cabin Filter, Incl. Sanitizer (EPA Registered Virucide, Furgicide, Kills Odors)",
    "PERFORM Multi-Point Inspection",
    "ASK Your Service Advisor about the FREE Lifetime Protection Planf",
    "PERFORM Wheel Alignment",
    "PERFORM Crankcase Cleaning Service",
    "PERFORM A/C Evaporator System Cleaning Service with Climate control disinfectant",
]
"""As OCR read them: "Furgicide" and "Planf" are OCR misreads on this photo, kept as read."""
DECLINED_391 = [
    "INSTALL Premium Engine Oil Conditioner",
    "REPLACE Cabin Filter, Incl. Sanitizer (EPA Registered Virucide, Fungicide, Kills Odors)",
    "PERFORM Crankcase Cleaning Service",
    "PERFORM A/C Evaporator System Cleaning Service with Climate control disinfectant",
]


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sheet_801():
    return parse_sheet(load("sheet_801_vision.json"))


@pytest.fixture(scope="module")
def sheet_391():
    return parse_sheet(load("sheet_391_vision.json"))


def block(text, left, top, width=0.3, height=0.012):
    return {
        "BlockType": "LINE",
        "Text": text,
        "Geometry": {"BoundingBox": {"Left": left, "Top": top, "Width": width, "Height": height}},
    }


# ------------------------------------------------------------------------- the $801.05 sheet


def test_the_nine_bullet_lines_are_the_service_lines_unpriced_in_order(sheet_801):
    assert sheet_801.layout == "bullets"
    services = sheet_801.lines[:9]
    assert [ln.text for ln in services] == SERVICES_801
    assert [ln.name for ln in services] == SERVICES_801
    assert all(ln.price is None and ln.price_text is None for ln in services)
    assert all(not ln.prices_listed and ln.readable_by_code for ln in services)
    assert [ln.index for ln in sheet_801.lines] == list(range(1, 12))


def test_shop_charges_and_fees_become_priced_lines_after_the_services(sheet_801):
    charges = sheet_801.lines[9:]
    assert [(ln.name, ln.price, ln.price_text) for ln in charges] == [
        ("Shop Charges", 78.29, "$ 78.29"),
        ("Fees", 1.99, "$ 1.99"),
    ]
    assert [ln.text for ln in charges] == ["Shop Charges ... $ 78.29", "Fees ... $ 1.99"]
    assert all(ln.prices_listed and ln.readable_by_code for ln in charges)


def test_totals_block_of_the_801_sheet(sheet_801):
    assert sheet_801.totals == SheetTotals(
        menu_total=672.0, shop_charges=78.29, tax=48.77, fees=1.99, total=801.05
    )


def test_header_badge_financing_and_footer_are_not_lines(sheet_801):
    texts = " | ".join(ln.text for ln in sheet_801.lines)
    for left_out in (
        "Customer",
        "Odometer",
        "Recommended Maintenance",
        "PREFERRED SERVICE",
        "LIFETIME PROTECTION",
        "6,000",
        "63.21",
        "month",
        "Tax",
        "TOTAL",
        "Menu Total",
        "Signature",
        "Page 1",
    ):
        assert left_out not in texts
    assert 6000.0 not in dataclasses.astuple(sheet_801.totals)
    assert 63.21 not in dataclasses.astuple(sheet_801.totals)


def test_no_declined_section_on_the_first_sheet(sheet_801):
    assert sheet_801.declined_from is None and sheet_801.declined == ()


# ------------------------------------------------------------------------- the $391.38 sheet


def test_declined_section_marks_exactly_the_four_lines_under_it(sheet_391):
    assert [ln.text for ln in sheet_391.declined] == DECLINED_391
    assert sheet_391.declined_from == 6
    flags = [ln.declined_on_sheet for ln in sheet_391.lines]
    assert flags == [False] * 5 + [True] * 4 + [False] * 2
    assert all("Declined" not in ln.text for ln in sheet_391.lines)


def test_totals_block_of_the_391_sheet(sheet_391):
    assert sheet_391.totals == SheetTotals(
        menu_total=309.85, shop_charges=55.77, tax=23.77, fees=1.99, total=391.38
    )
    assert [ln.name for ln in sheet_391.lines[9:]] == ["Shop Charges", "Fees"]


def test_rows_split_by_blanked_words_come_back_as_one_line(sheet_391):
    texts = [ln.text for ln in sheet_391.lines]
    assert "ASK Your Service Advisor about the FREE Lifetime Protection Plan!" in texts
    assert "PERFORM Crankcase Cleaning Service" in texts
    assert not any(t.startswith(("Lifetime", "Sanitizer", "Crankcase")) for t in texts)


def test_parse_textract_returns_the_sheet_lines(sheet_801):
    lines = parse_textract(load("sheet_801_vision.json"))
    assert tuple(lines) == sheet_801.lines


# ---------------------------------------------------------------------------- general rules


def test_priced_menus_parse_exactly_as_before_bullet_sheets():
    """The Civic fixture against the output of the parser before this change (f903196)."""
    snapshot = load("textract_menu_lines_f903196.json")["lines"]
    sheet = parse_sheet(load("textract_menu.json"))
    now = []
    for ln in sheet.lines:
        data = dataclasses.asdict(ln)
        assert data.pop("declined_on_sheet") is False and data.pop("prices_listed") is True
        now.append(data)
    assert json.loads(json.dumps(now)) == snapshot
    assert sheet.layout == "priced" and sheet.declined_from is None
    assert sheet.totals == SheetTotals(shop_charges=34.5)


@pytest.mark.parametrize("bullet", ["• ", "·", "▪ ", "- ", "* "])
def test_bullet_signs(bullet):
    response = {
        "Blocks": [
            block("Recommended service", 0.1, 0.05),
            block(f"{bullet}Replace wiper blades", 0.1, 0.10),
            block(f"{bullet}Tire rotation", 0.1, 0.13),
            block("TOTAL:", 0.7, 0.20, width=0.05),
            block("$ 59.90", 0.85, 0.20, width=0.05),
        ]
    }
    sheet = parse_sheet(response)
    assert [ln.name for ln in sheet.lines] == ["Replace wiper blades", "Tire rotation"]
    assert sheet.totals.total == 59.9 and sheet.totals.menu_total is None


def test_a_row_split_into_two_blocks_is_joined_left_to_right():
    response = {
        "Blocks": [
            block("Crankcase Cleaning Service", 0.32, 0.3062, width=0.17, height=0.0109),
            block("• PERFORM", 0.22, 0.3058, width=0.08, height=0.0119),
            block("• REPLACE Cabin Filter, Incl.", 0.22, 0.2422, width=0.16, height=0.0109),
            block("Sanitizer (EPA)", 0.45, 0.2352, width=0.3, height=0.0199),
            block("Maintenance", 0.12, 0.15, width=0.06, height=0.24),  # sideways, left margin
            block("Shop Charges:", 0.70, 0.5811, width=0.08),
            block("$ 78.29", 0.83, 0.5812, width=0.05),
        ]
    }
    sheet = parse_sheet(response)
    assert [ln.text for ln in sheet.lines] == [
        "REPLACE Cabin Filter, Incl. Sanitizer (EPA)",
        "PERFORM Crankcase Cleaning Service",
        "Shop Charges ... $ 78.29",
    ]
    assert sheet.lines[1].bbox.left == pytest.approx(0.22)
    assert sheet.lines[1].bbox.right == pytest.approx(0.49)


def test_a_bullet_sign_read_as_its_own_block_starts_the_row():
    response = {
        "Blocks": [
            block("•", 0.20, 0.10, width=0.01),
            block("Wiper blades", 0.22, 0.10),
            block("• Tire rotation", 0.20, 0.13),
        ]
    }
    assert [ln.text for ln in parse_sheet(response).lines] == ["Wiper blades", "Tire rotation"]


def test_text_on_another_row_or_much_taller_is_not_joined():
    response = {
        "Blocks": [
            block("• Oil change", 0.20, 0.10),
            block("Recommended at every visit", 0.6, 0.106),  # half a row lower: still joined
            block("• Tire rotation", 0.20, 0.13),
            block("Free", 0.6, 0.137, height=0.012),  # overlap 5/12 < 0.5: not joined
            block("SIDEWAYS", 0.9, 0.05, width=0.02, height=0.2),  # tall: not joined
        ]
    }
    texts = [ln.text for ln in parse_sheet(response).lines]
    assert texts == ["Oil change Recommended at every visit", "Tire rotation"]


def test_a_dash_without_a_space_is_not_a_bullet():
    response = {"Blocks": [block("-20% coupon", 0.1, 0.1), block("$5.00", 0.85, 0.1, width=0.05)]}
    sheet = parse_sheet(response)
    # the leading dash is trimmed as edge punctuation, as it always was
    assert sheet.layout == "priced" and [ln.name for ln in sheet.lines] == ["20% coupon"]


def test_priced_bullets_keep_their_prices_and_need_them_to_be_readable():
    response = {
        "Blocks": [
            block("• Oil change", 0.1, 0.10),
            block("$89.95", 0.85, 0.10, width=0.06),
            block("• Cabin air filter", 0.1, 0.14),  # its price was not read
        ]
    }
    lines = parse_sheet(response).lines
    assert [(ln.name, ln.price) for ln in lines] == [
        ("Oil change", 89.95),
        ("Cabin air filter", None),
    ]
    assert all(ln.prices_listed for ln in lines)
    assert [ln.readable_by_code for ln in lines] == [True, False]


def test_a_declined_section_ends_at_the_next_section_header():
    response = {
        "Blocks": [
            block("• Oil change", 0.1, 0.10),
            block("Declined", 0.1, 0.13, width=0.06),
            block("• Engine flush", 0.1, 0.16),
            block("Approved later", 0.1, 0.19, width=0.1),
            block("• Wiper blades", 0.1, 0.22),
        ]
    }
    sheet = parse_sheet(response)
    assert [ln.declined_on_sheet for ln in sheet.lines] == [False, True, False]
    assert sheet.declined_from == 2


def test_declined_inside_a_line_is_not_a_header():
    response = {
        "Blocks": [
            block("• Oil change", 0.1, 0.10),
            block("Declined", 0.1, 0.13, width=0.06),
            block("by owner 10/17", 0.3, 0.13, width=0.1),  # same row: not alone
            block("• Engine flush", 0.1, 0.16),
        ]
    }
    assert parse_sheet(response).declined == ()


@pytest.mark.parametrize(
    "label,field",
    [
        ("Menu Total:", "menu_total"),
        ("Subtotal", "menu_total"),
        ("Shop Charges:", "shop_charges"),
        ("Shop supplies", "shop_charges"),
        ("Tax:", "tax"),
        ("Fees:", "fees"),
        ("Fee", "fees"),
        ("TOTAL:", "total"),
        ("up to", None),
        ("/ month", None),
        ("Environmental disposal fee", None),
    ],
)
def test_total_labels(label, field):
    assert total_field(label) == field
