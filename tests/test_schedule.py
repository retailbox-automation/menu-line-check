"""The illustrative maintenance schedule and the due / not due / unknown check."""

from __future__ import annotations

import pytest

from menu_line_check.models import Vehicle
from menu_line_check.ocr import parse_line_texts
from menu_line_check.schedule import SCHEDULE_PATH, check_schedule, load_schedule


def civic(odometer):
    return Vehicle(2019, "Honda", "Civic", odometer)


def line(text):
    return parse_line_texts([text])[0]


def test_schedule_file_says_it_is_illustrative_and_has_the_intervals():
    assert "ILLUSTRATIVE" in SCHEDULE_PATH.read_text(encoding="utf-8")
    schedule = load_schedule()
    every = {s.service: (s.every_mi, s.every_months, s.every_visit) for s in schedule}
    assert every == {
        "oil and filter": (7500, None, False),
        "tire rotation": (7500, None, False),
        "engine air filter": (30000, None, False),
        "cabin air filter": (15000, None, False),
        "transmission fluid": (60000, None, False),
        "coolant": (100000, None, False),
        "spark plugs": (100000, None, False),
        "brake fluid": (None, 36, False),
        "multi-point inspection": (None, None, True),
    }


@pytest.mark.parametrize(
    "text,service,status,interval,next_due",
    [
        ("Oil & filter change, full synthetic $89.95", "oil and filter", "due", 7500, 37500),
        ("Tire rotation $39.95", "tire rotation", "due", 7500, 37500),
        ("Cabin air filter $79.95", "cabin air filter", "due", 15000, 45000),
        ("Engine air filter $64.95", "engine air filter", "due", 30000, 60000),
        ("Transmission fluid exchange $249.00", "transmission fluid", "not_due", 60000, 60000),
        ("Spark plug replacement $329.00 4 cylinder", "spark plugs", "not_due", 100000, 100000),
        ("Coolant flush & fill $179.00", "coolant", "not_due", 100000, 100000),
    ],
)
def test_mileage_intervals_at_31200_mi(text, service, status, interval, next_due):
    result = check_schedule(line(text), civic(31200))
    assert (result.service, result.status, result.every_mi, result.next_due_mi) == (
        service,
        status,
        interval,
        next_due,
    )


def test_due_window_before_and_after_a_multiple():
    oil = line("Oil & filter change $89.95")
    assert check_schedule(oil, civic(29000)).status == "due"  # 1,000 before 30,000
    assert check_schedule(oil, civic(31400)).status == "due"  # 1,400 after
    late = check_schedule(oil, civic(34000))
    assert late.status == "not_due" and late.next_due_mi == 37500 and late.last_due_mi == 30000
    assert check_schedule(oil, civic(34000), window_mi=4000).status == "due"
    assert check_schedule(oil, civic(800)).status == "not_due"


def test_every_visit_time_based_and_unknown():
    inspection = line("Multi-point inspection $0.00 (no charge)")
    assert check_schedule(inspection, civic(31200)).status == "due"
    brake_fluid = check_schedule(line("Brake fluid flush $139.00"), civic(31200))
    assert brake_fluid.status == "unknown" and brake_fluid.every_months == 36
    assert "date" in brake_fluid.note
    unknown = check_schedule(line("Throttle body service $119.00"), civic(31200))
    assert unknown.status == "unknown" and unknown.service is None


def test_the_longest_matching_name_wins():
    assert check_schedule(line("Cabin air filter $79.95"), civic(1)).service == "cabin air filter"
    assert check_schedule(line("Air filter $40.00"), civic(1)).service == "engine air filter"


def test_result_as_dict():
    d = check_schedule(line("Tire rotation $39.95"), civic(31200)).to_dict()
    assert d["status"] == "due" and d["every_mi"] == 7500 and "illustrative" in d["note"].lower()
