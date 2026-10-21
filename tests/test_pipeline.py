"""check_menu end to end: fake Decider (MockTransport), FakeWriter, recorded NHTSA response."""

from __future__ import annotations

import dataclasses

import pytest

from decider_kit import Ledger, read_ledger
from menu_line_check.llm import FakePicker, FakeWriter
from menu_line_check.models import MenuLine, Vehicle
from menu_line_check.nhtsa import NhtsaClient
from menu_line_check.ocr import parse_line_texts, parse_textract
from menu_line_check.pipeline import check_menu, questions_per_line
from menu_line_tool_selection import load_pattern_questions

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
    assert report.counts == {"keep": 3, "decline": 3, "ask": 5, "your_call": 1, "retake": 0}
    assert report.summary == "Keep 3 · Decline 3 · Ask 5 · Your call 1"
    assert report.summary_text.startswith(report.summary)
    assert "$2,361.25" in report.summary_text and "$307.00" in report.summary_text
    assert len(server.requests) == 12  # one request per line, no images in the fixture


def test_fuel_pump_recall_at_064_runs_the_nhtsa_lookup():
    report, _ = run(writer=FakeWriter())
    fuel = by_name(report, "fuel pump")
    assert (fuel.route, fuel.tool, fuel.source, fuel.verdict) == (
        "tool",
        "check_recall_warranty",
        "decider",
        "ask",
    )
    assert fuel.confidence == pytest.approx(0.64)
    assert fuel.tool_result["recall"]["campaign"] == "23V858000"
    assert fuel.advisor_question == "Is this covered by recall 23V858000?"


def test_tire_rotation_flagged_by_safety_still_checks_the_schedule():
    report, _ = run(writer=FakeWriter())
    rotation = by_name(report, "Tire rotation")
    assert (rotation.route, rotation.tool, rotation.verdict) == ("tool", "check_schedule", "keep")
    assert rotation.p_safety == pytest.approx(0.68)
    assert rotation.tool_result["schedule"]["status"] == "due"


def test_nitrogen_fill_flagged_by_safety_is_not_declined():
    report, _ = run(writer=FakeWriter())
    nitrogen = by_name(report, "Nitrogen")
    assert (nitrogen.route, nitrogen.source, nitrogen.tool, nitrogen.verdict) == (
        "ask_advisor",
        "code",
        None,
        "ask",
    )
    assert nitrogen.decider_choice == "decline_addon" and nitrogen.p_safety == pytest.approx(0.66)
    assert nitrogen.advisor_question is not None


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
    assert (diag.route, diag.tool, diag.verdict) == ("tool", "check_recall_warranty", "ask")
    assert diag.tool_result["recall"] is None
    conditioner = by_name(report, "conditioner")
    assert conditioner.verdict == "decline" and conditioner.advisor_question is None
    fee = by_name(report, "Shop supplies")
    assert fee.tool == "ask_about_fee" and "$34.50" in fee.advisor_question


def test_requests_use_the_ocr_text_wording_and_latency_is_kept():
    report, server = run()
    body = server.requests[0]
    assert all(list(r["questions"]) == ["safety", "tool"] for r in server.requests)
    assert "OCR text of one menu line" in body["questions"]["tool"]["instructions"]
    assert body["state"].endswith('"Oil & filter change, full synthetic ... $89.95"')
    timing = report.outcomes[0].decider_ms
    assert timing.requests == 1 and timing.server_ms == 12.5 and timing.client_ms > 0
    assert timing.questions == ("safety", "tool")
    assert report.outcomes[0].p_readable == 1.0  # the code check, not a Decider answer


def test_every_line_is_written_to_the_ledger(tmp_path):
    path = tmp_path / "l.jsonl"
    run(ledger=Ledger(path))
    rows = read_ledger(path)
    assert len(rows) == 12
    fuel = rows[5]
    assert fuel["gate"] == "menu-line-check" and fuel["tool"] == "check_recall_warranty"
    assert fuel["action"]["type"] == "tool" and "line 6" in fuel["action"]["reason"]
    assert fuel["confidences"]["tool"] == pytest.approx(0.64)
    assert fuel["questions"] == ["safety", "tool"] and set(fuel["answers"]) == {"safety", "tool"}
    assert "asked safety, tool; readable by code check" in fuel["action"]["reason"]
    assert "not declined automatically" in rows[9]["action"]["reason"]  # nitrogen fill


def test_per_tool_thresholds_and_the_single_override():
    strict, _ = run(thresholds={"check_recall_warranty": 0.75})
    assert by_name(strict, "fuel pump").route == "ask_advisor"
    assert by_name(strict, "Tech diagnostic").route == "ask_advisor"
    assert by_name(strict, "Tire rotation").route == "tool"
    one, _ = run(act_threshold=0.9)
    assert [o.line.name for o in one.outcomes if o.route == "tool"] == [
        "Oil & filter change, full synthetic",
        "Engine oil conditioner",
        "Cabin air filter",
        "Shop supplies",
        "Environmental disposal fee",
    ]


def test_safety_flagged_decline_goes_to_the_llm_picker():
    picks = []

    def picker(line, vehicle):
        picks.append((line.index, vehicle))
        return "check_schedule", "a tire service"

    report, _ = run(llm_picker=picker, writer=FakeWriter())
    nitrogen = by_name(report, "Nitrogen")
    assert picks == [(10, CIVIC)]
    assert (nitrogen.route, nitrogen.source, nitrogen.tool, nitrogen.rationale) == (
        "ask_llm",
        "llm",
        "check_schedule",
        "a tire service",
    )
    assert nitrogen.tool_result["schedule"]["status"] == "unknown" and nitrogen.verdict == "ask"


def test_a_picker_decline_on_a_safety_flagged_line_is_not_applied():
    report, _ = run(llm_picker=lambda line, vehicle: ("decline_addon", "an add-on"))
    nitrogen = by_name(report, "Nitrogen")
    assert (nitrogen.route, nitrogen.tool, nitrogen.verdict) == ("ask_advisor", None, "ask")
    assert "not declined automatically" in nitrogen.rationale
    assert report.counts["decline"] == 3


def test_fake_picker_and_a_picker_that_picks_nothing_or_fails():
    report, _ = run(llm_picker=FakePicker())
    assert by_name(report, "Nitrogen").tool == "owner_decides"  # keyword rule: "tire"

    nothing, _ = run(llm_picker=lambda line, vehicle: (None, "no tool was called"))
    nitrogen = by_name(nothing, "Nitrogen")
    assert (nitrogen.route, nitrogen.source, nitrogen.rationale) == (
        "ask_advisor",
        "code",
        "no tool was called",
    )

    def broken(line, vehicle):
        raise RuntimeError("throttled")

    failed, _ = run(llm_picker=broken)
    nitrogen = by_name(failed, "Nitrogen")
    assert nitrogen.route == "ask_advisor" and "throttled" in nitrogen.rationale


def test_line_images_ask_readable_on_the_image_in_a_second_request():
    server = menu_server(readable_image=0.2)
    line = dataclasses.replace(lines()[1], images=("bmFtZQ==", "cHJpY2U="))
    report = check_menu([line], CIVIC, server.client())
    first, second = server.requests
    assert "images" not in first and second["images"] == ["bmFtZQ==", "cHJpY2U="]
    assert list(first["questions"]) == ["safety", "tool"]
    assert list(second["questions"]) == ["readable"]
    assert second["questions"]["readable"]["instructions"].endswith("on this menu line?")
    assert second["state"].startswith("One line of a car dealer's printed service menu. Image 1")
    (outcome,) = report.outcomes
    assert (outcome.route, outcome.verdict, outcome.source) == ("retake", "retake", "decider")
    assert outcome.decider_ms.requests == 2 and outcome.decider_ms.server_ms == 32.5
    assert outcome.decider_ms.questions == ("safety", "tool", "readable_image")


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
    assert data["declined_total"] == 307.0


def test_questions_per_line():
    plain = lines()[0]
    assert questions_per_line(plain) == ("safety", "tool")
    with_photo = dataclasses.replace(plain, images=("bmFtZQ==", "cHJpY2U="))
    assert questions_per_line(with_photo) == ("safety", "tool", "readable_image")


def test_short_criteria_are_sent_only_when_asked_for():
    _, spike = run()
    default_criteria = spike.requests[0]["questions"]["tool"]["criteria"]
    assert default_criteria == load_pattern_questions("ocr_text")["tool"].criteria
    _, short = run(criteria="short")
    sent = short.requests[0]["questions"]["tool"]
    assert sent["criteria"] == load_pattern_questions("ocr_text", criteria="short")["tool"].criteria
    assert sent["instructions"] == spike.requests[0]["questions"]["tool"]["instructions"]
    assert short.requests[0]["questions"]["safety"] == spike.requests[0]["questions"]["safety"]


# ------------------------------------------------------------------ readability without prices


def _sheet_lines(name: str) -> list[MenuLine]:
    import json

    from menu_line_check.ocr import parse_sheet

    path = NHTSA.parent / name
    return list(parse_sheet(json.loads(path.read_text(encoding="utf-8"))).lines)


def test_on_a_sheet_without_line_prices_a_found_name_is_readable():
    rx_sheet = _sheet_lines("lexus_sheet_801_vision.json")
    report = check_menu(rx_sheet, Vehicle(2024, "Lexus", "RX 350", 20000), menu_server().client())
    assert report.counts["retake"] == 0
    assert all(o.p_readable == 1.0 for o in report.outcomes)
    assert all(o.line.price is None for o in report.outcomes[:9])


def test_on_a_priced_menu_a_missing_price_is_still_a_retake():
    priced = parse_line_texts(["Oil change $89.95", "Wiper blades"])
    report = check_menu(priced, CIVIC, menu_server().client())
    assert [o.verdict for o in report.outcomes][1] == "retake"
    assert report.outcomes[1].p_readable == 0.0
    # the same two lines as bullets of a sheet that prices one of them: the rule is unchanged
    bullets = [dataclasses.replace(ln, prices_listed=True) for ln in priced]
    assert check_menu(bullets, CIVIC, menu_server().client()).outcomes[1].verdict == "retake"
    unpriced = [dataclasses.replace(ln, price=None, prices_listed=False) for ln in priced]
    assert check_menu(unpriced, CIVIC, menu_server().client()).counts["retake"] == 0


def test_a_line_with_no_name_is_a_retake_even_without_line_prices():
    (blank,) = parse_line_texts(["$12.00"])
    line = dataclasses.replace(blank, price=None, price_text=None, prices_listed=False)
    assert not line.readable_by_code
    assert check_menu([line], CIVIC, menu_server().client()).outcomes[0].verdict == "retake"


def test_the_report_carries_the_recalls_card_only_with_an_nhtsa_client():
    report, _ = run()
    assert report.recalls is not None and report.recalls.vehicle == "2019 Honda Civic"
    assert report.to_dict()["recalls"]["advisor_question"] == "Are there open recalls on my VIN?"
    assert run(nhtsa=None)[0].recalls is None


def test_sheet_totals_go_into_the_summary_when_the_sheet_prints_them():
    from menu_line_check.models import SheetTotals

    rx_sheet = _sheet_lines("lexus_sheet_801_vision.json")
    totals = SheetTotals(672.0, 78.29, 48.77, 1.99, 801.05)
    report = check_menu(
        rx_sheet,
        Vehicle(2024, "Lexus", "RX 350", 20000),
        menu_server().client(),
        writer=FakeWriter(),
        totals=totals,
    )
    assert report.totals == totals
    assert report.summary_text == (
        "Keep 3 · Decline 2 · Ask 5 · Your call 1. Sheet total $801.05. 1 safety line is your call."
    )
    assert report.summary_source == "template" and report.summary_fallback is None
