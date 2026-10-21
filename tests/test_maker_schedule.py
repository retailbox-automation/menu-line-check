"""The maker's schedule for one car, the never-decline illustrative table, the decline guard."""

from __future__ import annotations

import json

import pytest

from menu_line_check import schedule as schedule_mod
from menu_line_check.llm import FakeWriter
from menu_line_check.models import Vehicle
from menu_line_check.ocr import parse_line_texts, parse_sheet
from menu_line_check.pipeline import check_menu, guard_decline
from menu_line_check.schedule import (
    MAKER_SCHEDULES_PATH,
    check_schedule,
    load_maker_schedules,
    maker_service,
)
from menu_line_check.tools import check_schedule as schedule_tool

from .conftest import FakeServer, noul, ok_response
from .menu_fakes import SHEET_801, tool_answer

SHEET_CAR = Vehicle.parse("2024 Lexus RX 350 20000")
SOURCE_URL = (
    "https://assets.sia.toyota.com/publications/en/omms-s/L-MMS-24RX350/pdf/L-MMS-24RX350.pdf"
)


def sheet_lines():
    return parse_sheet(json.loads(SHEET_801.read_text(encoding="utf-8"))).lines


def line(text):
    return parse_line_texts([text])[0]


def test_the_maker_schedule_names_its_source_and_lists_the_20000_mile_items():
    (rx,) = load_maker_schedules()
    assert (rx.make, rx.model, rx.year) == ("LEXUS", "RX 350", 2024)
    assert rx.url == SOURCE_URL and "printed page 50 (PDF page 52)" in rx.page
    (service,) = rx.services
    assert (service.mileage, service.months) == (20000, 24)
    assert [i.item for i in service.items] == [
        "replace engine oil with 0W-20; replace oil filter",
        "replace cabin air filter",
        "reset oil maintenance reminder if oil was changed",
        "reset scheduled maintenance reminder",
        "rotate tires",
        "inspect brake linings, drums, pads and discs",
        "inspect wiper blades",
        "check installation of driver's floor mat",
        "road test vehicle",
    ]
    assert "Hawaii" not in MAKER_SCHEDULES_PATH.read_text(encoding="utf-8")


def test_each_line_of_the_real_sheet_against_the_maker_schedule():
    got = [(c.status, c.source) for c in (check_schedule(ln, SHEET_CAR) for ln in sheet_lines())]
    assert [s for s, _ in got] == [
        "due",  # engine oil and oil filter
        "unclear",  # premium engine oil conditioner: names "engine oil", is not the oil change
        "due",  # tire rotation
        "bundled",  # cabin filter, incl. sanitizer
        "due",  # multi-point inspection -> the inspect items
        "not_listed",  # protection plan
        "not_listed",  # wheel alignment
        "not_listed",  # crankcase cleaning
        "not_listed",  # A/C evaporator cleaning
        "not_listed",  # shop charges
        "not_listed",  # fees
    ]
    assert {source for _, source in got} == {"maker"}
    cabin = check_schedule(sheet_lines()[3], SHEET_CAR)
    assert cabin.extra == "sanitizer" and cabin.name == "a cabin air filter"
    assert SOURCE_URL in cabin.reference


def test_tool_verdicts_and_reasons_with_the_maker_schedule():
    lines = sheet_lines()
    oil = schedule_tool(lines[0], SHEET_CAR)
    assert (oil["verdict"], oil["note"]) == ("keep", "in the maker's schedule at 20,000 mi")
    assert schedule_tool(lines[2], SHEET_CAR)["verdict"] == "keep"
    assert schedule_tool(lines[4], SHEET_CAR)["verdict"] == "keep"
    assert schedule_tool(lines[3], SHEET_CAR)["verdict"] == "ask"
    plan = schedule_tool(lines[5], SHEET_CAR)
    assert (plan["verdict"], plan["note"]) == (
        "decline",
        "not in the maker's schedule at 20,000 mi",
    )


def test_the_bundled_question_asks_for_the_listed_item_alone():
    report = check_menu(
        sheet_lines()[3:4],
        SHEET_CAR,
        _server({"Cabin": ("check_schedule", 0.9, 0.1)}),
        writer=FakeWriter(),
    )
    (cabin,) = report.outcomes
    assert cabin.verdict == "ask"
    assert cabin.advisor_question == (
        "The maker's schedule lists a cabin air filter at 20,000 mi: can you do just that, "
        "without the sanitizer?"
    )


def test_no_maker_schedule_away_from_its_mileage_or_for_another_car():
    oil = line("Oil & filter change $89.95")
    assert maker_service(Vehicle(2024, "Lexus", "RX 350", 31200)) is None
    assert check_schedule(oil, Vehicle(2024, "Lexus", "RX 350", 31200)).source == "illustrative"
    assert check_schedule(oil, Vehicle(2024, "Lexus", "RX 350h", 20000)).source == "illustrative"
    assert check_schedule(oil, Vehicle(2023, "Lexus", "RX 350", 20000)).source == "illustrative"
    assert check_schedule(oil, Vehicle(2024, "LEXUS", "rx 350", 21000)).source == "maker"


@pytest.mark.parametrize(
    "text,odometer,verdict",
    [
        ("Oil & filter change $89.95", 31200, "keep"),  # due in the table
        ("Transmission fluid exchange $249.00", 31200, "ask"),  # not due: a question
        ("Brake fluid flush $129.00", 31200, "ask"),  # time based: unknown
        ("Engine flush $99.00", 31200, "ask"),  # not in the table
    ],
)
def test_the_illustrative_table_never_declines(text, odometer, verdict):
    result = schedule_tool(line(text), Vehicle(2019, "Honda", "Civic", odometer))
    assert result["verdict"] == verdict and result["schedule"]["source"] == "illustrative"


def test_the_illustrative_question_asks_about_the_maker_schedule():
    civic = Vehicle(2019, "Honda", "Civic", 31200)
    report = check_menu(
        [line("Transmission fluid exchange $249.00")],
        civic,
        _server({"Transmission": ("check_schedule", 0.9, 0.1)}),
        writer=FakeWriter(),
    )
    assert report.outcomes[0].advisor_question == "Is this in the maker's schedule at 31,200 mi?"


# --------------------------------------------------------------------------- the decline guard


def _server(answers):
    def respond(body):
        found = [v for k, v in answers.items() if k in body["state"]]
        tool, conf, p_safety = found[0]
        return ok_response({"safety": noul(p_safety), "tool": tool_answer(tool, conf)})

    return FakeServer(respond).client()


def test_a_schedule_decline_on_a_safety_flagged_line_becomes_the_owners_call(tmp_path, monkeypatch):
    """Tire rotation at P(safety) 0.78 (the live answer), on a maker schedule without it."""
    data = MAKER_SCHEDULES_PATH.read_text(encoding="utf-8")
    start = data.index('          - item: "rotate tires"')
    end = data.index('          - item: "inspect brake linings')
    path = tmp_path / "maker.yaml"
    path.write_text(data[:start] + data[end:], encoding="utf-8")
    monkeypatch.setattr(schedule_mod, "MAKER_SCHEDULES_PATH", path)

    rotation = [ln for ln in sheet_lines() if "Tire Rotation" in ln.text]
    report = check_menu(
        rotation, SHEET_CAR, _server({"Tire Rotation": ("check_schedule", 0.79, 0.78)})
    )
    (o,) = report.outcomes
    assert o.tool_result["schedule"]["status"] == "not_listed"
    assert (o.route, o.tool, o.verdict) == ("tool", "check_schedule", "your_call")
    assert o.tool_result["guarded"] == "decline"
    assert o.tool_result["note"] == (
        "not in the maker's schedule at 20,000 mi. Not declined: P(safety)=0.78 >= 0.5, "
        "so the owner decides."
    )
    # the same line without the safety flag is declined
    calm = check_menu(
        rotation, SHEET_CAR, _server({"Tire Rotation": ("check_schedule", 0.79, 0.2)})
    )
    assert calm.outcomes[0].verdict == "decline"


def test_with_the_real_schedule_the_rotation_is_kept_whatever_its_safety_answer():
    rotation = [ln for ln in sheet_lines() if "Tire Rotation" in ln.text]
    report = check_menu(
        rotation, SHEET_CAR, _server({"Tire Rotation": ("check_schedule", 0.79, 0.78)})
    )
    assert report.outcomes[0].verdict == "keep"


def test_guard_decline_leaves_other_verdicts_and_calm_lines_alone():
    decline = {"tool": "decline_addon", "verdict": "decline", "note": "Optional add-on."}
    assert guard_decline(decline, 0.49) is decline
    assert guard_decline(decline, None) is decline
    assert guard_decline({"verdict": "keep"}, 0.9) == {"verdict": "keep"}
    guarded = guard_decline(decline, 0.5)
    assert guarded["verdict"] == "your_call" and guarded["tool"] == "decline_addon"
