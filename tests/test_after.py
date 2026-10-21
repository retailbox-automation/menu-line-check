"""Two sheets of one visit: totals before and after, the owner's declines, the agent's verdicts."""

from __future__ import annotations

import dataclasses
import json

import pytest

from menu_line_check.after import compare_sheets, normalise
from menu_line_check.models import SheetTotals, Vehicle
from menu_line_check.nhtsa import NhtsaClient
from menu_line_check.ocr import parse_sheet
from menu_line_check.pipeline import check_menu

from .menu_fakes import NHTSA_RX_RECORDED, SHEET_391, SHEET_801, menu_server

RX350 = Vehicle.parse("2024 Lexus RX 350 20000")


def sheet(path):
    return parse_sheet(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def checked():
    before = sheet(SHEET_801)
    report = check_menu(
        before.lines,
        RX350,
        menu_server().client(),
        nhtsa=NhtsaClient.from_recorded(*NHTSA_RX_RECORDED),
        totals=before.totals,
    )
    after = sheet(SHEET_391)
    return report, compare_sheets(report, after.lines, after.totals)


def test_totals_before_after_and_the_difference(checked):
    _, block = checked
    assert (block.before_total, block.after_total, block.difference) == (801.05, 391.38, 409.67)
    assert block.row_differences() == {
        "menu_total": 362.15,
        "shop_charges": 22.52,
        "tax": 25.0,
        "fees": 0.0,
        "total": 409.67,
    }


def test_each_owner_decline_is_matched_to_its_first_sheet_line(checked):
    report, block = checked
    assert [d.before_index for d in block.declined] == [2, 4, 8, 9]
    # "Fungicide" on the second photo, "Furgicide" on the first: matched by similarity
    cabin = block.declined[1]
    assert cabin.text != cabin.before_text and "Cabin Filter" in cabin.before_text
    verdicts = {o.line.index: o.verdict for o in report.outcomes}
    assert [d.agent_verdict for d in block.declined] == [verdicts[i] for i in (2, 4, 8, 9)]


def test_agreement_with_the_scripted_answers(checked):
    report, block = checked
    # scripted fakes: conditioner and crankcase decline at >= 0.8; the cabin filter goes to the
    # illustrative schedule (every 15,000 mi: not due at 20,000); the A/C cleaning at 0.7 is
    # below the decline threshold (ask the advisor)
    assert [d.agent_verdict for d in block.declined] == ["decline", "decline", "decline", "ask"]
    assert block.agreed == 3
    # oil and rotation: every 7,500 mi in the illustrative table, so not due at 20,000
    assert block.agent_declined_owner_kept == (
        "REPLACE Engine Oil and Oil Filter",
        "PERFORM Tire Rotation",
    )
    data = block.to_dict()
    assert data["owner_declined"] == 4 and data["agent_said_decline_on_owner_declined"] == 3
    assert data["difference"] == 409.67


def test_an_agent_decline_the_owner_kept_is_listed(checked):
    report, _ = checked
    after = sheet(SHEET_391)
    kept_all = [dataclasses.replace(ln, declined_on_sheet=False) for ln in after.lines]
    block = compare_sheets(report, kept_all, after.totals)
    assert block.declined == () and block.agreed == 0
    declined_by_agent = [o.line.text for o in report.outcomes if o.verdict == "decline"]
    assert block.agent_declined_owner_kept == tuple(declined_by_agent)


def test_missing_totals_give_no_difference(checked):
    report, _ = checked
    block = compare_sheets(dataclasses.replace(report, totals=None), [], SheetTotals())
    assert block.difference is None and block.before_total is None


def test_a_declined_line_with_no_counterpart_is_kept_unmatched(checked):
    report, _ = checked
    stray = dataclasses.replace(
        sheet(SHEET_391).lines[0], text="Replace timing belt", declined_on_sheet=True
    )
    (only,) = compare_sheets(report, [stray], SheetTotals()).declined
    assert only.before_index is None and only.agent_verdict is None


def test_normalise():
    assert normalise("REPLACE Cabin Filter, Incl, X") == normalise("replace cabin filter incl. x")
