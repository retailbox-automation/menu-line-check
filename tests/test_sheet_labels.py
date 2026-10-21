"""The labels of the real sheet line up with what the parser reads and with the owner's declines."""

from __future__ import annotations

import json

from menu_line_check.after import compare_sheets
from menu_line_check.models import Vehicle
from menu_line_check.ocr import parse_sheet
from menu_line_check.pipeline import check_menu
from menu_line_tool_selection import TOOLS

from .menu_fakes import FIXTURES, LEXUS_391, LEXUS_801, menu_server

LABELS = json.loads((FIXTURES / "lexus_sheet_labels.json").read_text(encoding="utf-8"))


def sheet(path):
    return parse_sheet(json.loads(path.read_text(encoding="utf-8")))


def test_one_label_per_parsed_line_matched_by_index_and_text():
    lines = sheet(FIXTURES / LABELS["sheet"]).lines
    assert len(LABELS["lines"]) == len(lines) == 11
    for label, line in zip(LABELS["lines"], lines, strict=True):
        assert label["index"] == line.index and label["match"] in line.text
        assert label["tools"] and set(label["tools"]) <= set(TOOLS) and label["reason"]
    assert LABELS["label_source"].startswith("lane judgement 2026-10-21")


def test_owner_declined_flags_are_the_second_sheet_declines():
    report = check_menu(
        sheet(LEXUS_801).lines, Vehicle(2024, "Lexus", "RX 350", 20000), menu_server().client()
    )
    after = sheet(LEXUS_391)
    declined = {d.before_index for d in compare_sheets(report, after.lines, after.totals).declined}
    assert declined == {label["index"] for label in LABELS["lines"] if label["owner_declined"]}
    assert declined == {2, 4, 8, 9}
