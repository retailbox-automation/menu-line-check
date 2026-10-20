"""check_menu end to end: fake Decider (MockTransport), FakeWriter, recorded NHTSA response."""

from __future__ import annotations

import dataclasses

import pytest

from decider_kit import Ledger, read_ledger
from menu_line_check.llm import FakePicker, FakeWriter
from menu_line_check.models import MenuLine, Vehicle
from menu_line_check.nhtsa import NhtsaClient
from menu_line_check.ocr import parse_line_texts, parse_textract
from menu_line_check.pipeline import check_menu

from .conftest import FakeServer
from .menu_fakes import NHTSA, menu_server, textract_response

CIVIC = Vehicle(2019, "Honda", "Civic", 31200)


def lines() -> list[MenuLine]:
    return parse_textract(textract_response())


def run(server=None, **kwargs):
    server = server or menu_server()
    kwargs.setdefault("nhtsa", NhtsaClient.from_recorded(NHTSA))
    report = check_menu(lines(), CIVIC, server.client(), **kwargs)
    return report, server


def by_name(report, fragment):
    (outcome,) = [o for o in report.outcomes if fragment in o.line.name]
    return outcome


def test_fixture_menu_summary_card(tmp_path):
    report, server = run(writer=FakeWriter(), ledger=Ledger(tmp_path / "l.jsonl"))
    assert report.counts == {"keep": 3, "decline": 4, "ask": 4, "your_call": 1, "retake": 0}
    assert report.summary == "Keep 3 · Decline 4 · Ask 4 · Your call 1"
    assert report.summary_text.startswith(report.summary)
    assert "$2,361.25" in report.summary_text and "$336.95" in report.summary_text
    assert len(server.requests) == 12  # one request per line, no images in the fixture


def test_fuel_pump_line_finds_the_recall_and_asks_about_it():
    report, _ = run(writer=FakeWriter())
    fuel = by_name(report, "fuel pump")
    assert (fuel.route, fuel.tool, fuel.source, fuel.verdict) == (
        "tool",
        "check_recall_warranty",
        "decider",
        "ask",
    )
    assert fuel.confidence == pytest.approx(0.84)
    assert fuel.tool_result["recall"]["campaign"] == "23V858000"
    assert fuel.advisor_question == "Is this covered by recall 23V858000?"


def test_each_route_and_who_decided():
    report, _ = run(writer=FakeWriter())
    brakes = by_name(report, "brake pads")
    assert (brakes.route, brakes.tool, brakes.verdict) == (
        "owner_decides",
        "owner_decides",
        "your_call",
    )
    assert brakes.advisor_question and "brake" in brakes.advisor_question.lower()
    oil = by_name(report, "Oil & filter")
    assert oil.verdict == "keep" and oil.tool_result["schedule"]["status"] == "due"
    diag = by_name(report, "Tech diagnostic")
    assert (diag.route, diag.source, diag.tool, diag.verdict) == (
        "ask_advisor",
        "code",
        None,
        "ask",
    )
    assert diag.decider_choice == "check_recall_warranty"
    conditioner = by_name(report, "conditioner")
    assert conditioner.verdict == "decline" and conditioner.advisor_question is None
    fee = by_name(report, "Shop supplies")
    assert fee.tool == "ask_about_fee" and "$34.50" in fee.advisor_question


def test_requests_use_the_ocr_text_wording_and_latency_is_kept():
    report, server = run()
    body = server.requests[0]
    assert list(body["questions"]) == ["readable", "safety", "tool"]
    assert "OCR text of one menu line" in body["questions"]["tool"]["instructions"]
    assert body["state"].endswith('"Oil & filter change, full synthetic ... $89.95"')
    timing = report.outcomes[0].decider_ms
    assert timing.requests == 1 and timing.server_ms == 12.5 and timing.client_ms > 0


def test_every_line_is_written_to_the_ledger(tmp_path):
    path = tmp_path / "l.jsonl"
    run(ledger=Ledger(path))
    rows = read_ledger(path)
    assert len(rows) == 12
    fuel = rows[5]
    assert fuel["gate"] == "menu-line-check" and fuel["tool"] == "check_recall_warranty"
    assert fuel["action"]["type"] == "tool" and "line 6" in fuel["action"]["reason"]
    assert fuel["confidences"]["tool"] == pytest.approx(0.84)
    assert set(fuel["answers"]) == {"readable", "safety", "tool", "readable_code"}


def test_low_confidence_line_goes_to_the_llm_picker():
    picks = []

    def picker(line, vehicle):
        picks.append((line.index, vehicle))
        return "check_recall_warranty", "a diagnostic fee"

    report, _ = run(llm_picker=picker, writer=FakeWriter())
    diag = by_name(report, "Tech diagnostic")
    assert picks == [(7, CIVIC)]
    assert (diag.route, diag.source, diag.tool, diag.rationale) == (
        "ask_llm",
        "llm",
        "check_recall_warranty",
        "a diagnostic fee",
    )
    assert diag.tool_result["recall"] is None and diag.verdict == "ask"
    assert report.counts["ask"] == 4


def test_fake_picker_and_a_picker_that_picks_nothing_or_fails():
    report, _ = run(llm_picker=FakePicker())
    assert by_name(report, "Tech diagnostic").tool == "check_recall_warranty"

    nothing, _ = run(llm_picker=lambda line, vehicle: (None, "no tool was called"))
    diag = by_name(nothing, "Tech diagnostic")
    assert (diag.route, diag.source, diag.rationale) == (
        "ask_advisor",
        "code",
        "no tool was called",
    )

    def broken(line, vehicle):
        raise RuntimeError("throttled")

    failed, _ = run(llm_picker=broken)
    diag = by_name(failed, "Tech diagnostic")
    assert diag.route == "ask_advisor" and "throttled" in diag.rationale


def test_line_images_ask_readable_on_the_image_in_a_second_request():
    server = menu_server(readable_image=0.2)
    line = dataclasses.replace(lines()[1], images=("bmFtZQ==", "cHJpY2U="))
    report = check_menu([line], CIVIC, server.client())
    first, second = server.requests
    assert "images" not in first and second["images"] == ["bmFtZQ==", "cHJpY2U="]
    assert list(second["questions"]) == ["readable"]
    assert second["questions"]["readable"]["instructions"].endswith("on this menu line?")
    assert second["state"].startswith("One line of a car dealer's printed service menu. Image 1")
    (outcome,) = report.outcomes
    assert (outcome.route, outcome.verdict, outcome.source) == ("retake", "retake", "decider")
    assert outcome.decider_ms.requests == 2 and outcome.decider_ms.server_ms == 32.5


def test_without_images_a_line_missing_its_price_is_retaken_by_code():
    (line,) = parse_line_texts(["Wiper blades"])
    report = check_menu([line], CIVIC, menu_server().client())
    (outcome,) = report.outcomes
    assert (outcome.route, outcome.source, outcome.tool) == ("retake", "code", "retake_line")
    assert "line 1" in outcome.tool_result["note"].lower()
    assert report.summary.endswith("Retake 1")


def test_decider_down_nothing_is_declined(tmp_path):
    server = FakeServer(lambda body: (503, {"detail": "down"}))
    report = check_menu(lines()[:2], CIVIC, server.client(), ledger=Ledger(tmp_path / "l.jsonl"))
    assert [o.route for o in report.outcomes] == ["unavailable", "unavailable"]
    assert report.counts["your_call"] == 2 and report.counts["decline"] == 0
    assert read_ledger(tmp_path / "l.jsonl")[0]["unavailable"] == "http_5xx"


def test_report_to_dict_is_json_ready():
    import json

    report, _ = run(writer=FakeWriter())
    data = json.loads(json.dumps(report.to_dict()))
    assert data["counts"]["keep"] == 3 and len(data["outcomes"]) == 12
    assert data["outcomes"][5]["tool_result"]["recall"]["campaign"] == "23V858000"
    assert data["declined_total"] == 336.95
