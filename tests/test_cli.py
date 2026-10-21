"""menu-check on the Textract fixture with a fake Decider (monkeypatched client factory)."""

from __future__ import annotations

import io
import json
import re

import pytest
from PIL import Image

from decider_kit import read_ledger
from menu_line_check import cli
from menu_line_check.llm import FakePicker, FakeWriter

from .conftest import FakeServer, ok_response
from .menu_fakes import (
    NHTSA,
    NHTSA_RX,
    NHTSA_RX350_EMPTY,
    SHEET_391,
    SHEET_801,
    TEXTRACT,
    menu_server,
)

ARGS = ["--ocr-json", str(TEXTRACT), "--vehicle", "2019 Honda Civic 31200"]
OFFLINE = [*ARGS, "--nhtsa-recorded", str(NHTSA)]


@pytest.fixture
def fake_decider(monkeypatch):
    server = menu_server()
    monkeypatch.setattr(cli, "make_decider_client", lambda url, timeout=30.0: server.client())

    def no_bedrock(*args, **kwargs):
        raise AssertionError("Bedrock parts were built without --allow-bedrock")

    monkeypatch.setattr(cli, "make_bedrock_parts", no_bedrock)
    return server


def test_cli_prints_the_table_and_the_summary(fake_decider, capsys, tmp_path):
    report_path, ledger_path = tmp_path / "out" / "report.json", tmp_path / "out" / "l.jsonl"
    rc = cli.main([*OFFLINE, "--json", str(report_path), "--ledger", str(ledger_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2019 Honda Civic, 31,200 mi" in out
    assert out.count("Keep 3 · Decline 3 · Ask 5 · Your call 1") == 1
    assert "act >= decline_addon 0.8, check_schedule 0.5" in out
    assert "(up to 12 at a time)" in out
    assert "questions per line: 2 (safety, tool; readable by code check)" in out
    assert "tool criteria: spike" in out
    assert all(list(r["questions"]) == ["safety", "tool"] for r in fake_decider.requests)
    assert re.search(r"^12 lines decided in \d+ ms$", out, re.MULTILINE)
    assert report_path.exists() and "decider_wall_ms" in report_path.read_text()
    assert "Is this covered by recall 23V858000?" in out
    fuel_row = next(r for r in out.splitlines() if "Replace fuel pump module" in r)
    for cell in ("$1,089.00", "tool", "check_recall_warranty", "0.64", "decider"):
        assert cell in fuel_row
    nitrogen_row = next(r for r in out.splitlines() if "Nitrogen tire fill" in r)
    assert "ask_advisor" in nitrogen_row and "code" in nitrogen_row
    report = json.loads(report_path.read_text())
    assert report["counts"] == {"keep": 3, "decline": 3, "ask": 5, "your_call": 1, "retake": 0}
    rows = read_ledger(ledger_path)
    assert [r["stage"] for r in rows].count("line") == 12
    questions = [r for r in rows if r["stage"] == "question"]  # 5 ask + 1 your call
    assert len(questions) == 6
    assert all("question_source=template" in r["action"]["reason"] for r in questions)


def row(out, name):
    return next(r for r in out.splitlines() if name in r)


def row_after(out, name):
    """The detail line printed under the table row of ``name``."""
    lines = out.splitlines()
    return lines[lines.index(row(out, name)) + 1]


def test_cli_thresholds_change_the_route(fake_decider, capsys):
    assert cli.main([*OFFLINE, "--thresholds", "check_schedule=0.6"]) == 0
    out = capsys.readouterr().out
    assert "ask_advisor" in row(out, "Tire rotation") and "check_schedule 0.6" in out
    assert "check_recall_warranty" in row(out, "Replace fuel pump")
    assert cli.main([*OFFLINE, "--act-threshold", "0.9"]) == 0
    assert "ask_advisor" in row(capsys.readouterr().out, "Replace fuel pump")


@pytest.mark.parametrize(
    "extra",
    [
        ["--thresholds", "owner_decides=0.5"],
        ["--thresholds", "decline_addon:0.8"],
        ["--thresholds", "decline_addon=0.8", "--act-threshold", "0.7"],
    ],
)
def test_cli_rejects_bad_thresholds(fake_decider, extra):
    with pytest.raises(SystemExit):
        cli.main([*OFFLINE, *extra])


def test_cli_with_a_photo_asks_readable_on_the_line_crops(fake_decider, tmp_path, capsys):
    photo = tmp_path / "menu.jpg"
    buf = io.BytesIO()
    Image.new("RGB", (900, 1300), "white").save(buf, "JPEG")
    photo.write_bytes(buf.getvalue())
    assert cli.main([*OFFLINE, "--image", str(photo)]) == 0
    with_images = [r for r in fake_decider.requests if "images" in r]
    assert len(fake_decider.requests) == 24 and len(with_images) == 12
    assert "questions per line: 3 (safety, tool; readable on the line image)" in (
        capsys.readouterr().out
    )
    assert all(len(r["images"]) == 2 for r in with_images)


def test_cli_stops_when_decider_is_down(monkeypatch, capsys):
    server = FakeServer(lambda body: ok_response({}))
    server.health_status = 503
    monkeypatch.setattr(cli, "make_decider_client", lambda url, timeout=30.0: server.client())
    assert cli.main(OFFLINE) == 3
    assert "unavailable" in capsys.readouterr().err


def test_cli_bad_input(fake_decider, tmp_path, capsys):
    assert cli.main(["--ocr-json", str(tmp_path / "none.json"), "--vehicle", "2019 A B 1"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["--ocr-json", str(TEXTRACT), "--vehicle", "Civic"])


def test_allow_bedrock_builds_the_bedrock_parts_only_with_the_flag(monkeypatch, capsys):
    server = menu_server()
    monkeypatch.setattr(cli, "make_decider_client", lambda url, timeout=30.0: server.client())
    built = []

    def parts(decider, ledger, region, spend):
        built.append(region)
        return FakeWriter(), FakePicker()

    monkeypatch.setattr(cli, "make_bedrock_parts", parts)
    assert cli.main(OFFLINE) == 0 and built == []
    assert cli.main([*OFFLINE, "--allow-bedrock"]) == 0 and built == ["us-east-1"]
    nitrogen = next(r for r in capsys.readouterr().out.splitlines()[::-1] if "Nitrogen" in r)
    assert "ask_llm" in nitrogen and "llm" in nitrogen and "owner_decides" in nitrogen


def test_cli_concurrency_option(fake_decider, capsys):
    assert cli.main([*OFFLINE, "--concurrency", "4"]) == 0
    out = capsys.readouterr().out
    assert "(up to 4 at a time)" in out and "12 lines decided in" in out
    for bad in ("0", "-2", "many"):
        with pytest.raises(SystemExit):
            cli.main([*OFFLINE, "--concurrency", bad])


def test_cli_criteria_switch(fake_decider, capsys):
    assert cli.main([*OFFLINE, "--criteria", "short"]) == 0
    assert "tool criteria: short" in capsys.readouterr().out
    sent = fake_decider.requests[0]["questions"]["tool"]["criteria"]
    assert (
        sent["owner_decides"] == "brakes, tires, alignment, steering or other driving-safety parts"
    )
    with pytest.raises(SystemExit):
        cli.main([*OFFLINE, "--criteria", "long"])


SHEET = [
    "--ocr-json",
    str(SHEET_801),
    "--vehicle",
    "2024 Lexus RX 350 20000",
    "--nhtsa-recorded",
    str(NHTSA_RX),
    "--nhtsa-recorded",
    f"{NHTSA_RX350_EMPTY}=LEXUS/RX 350/2024",
]


def test_cli_two_sheets_prints_totals_recalls_and_the_after_block(fake_decider, capsys, tmp_path):
    report_path = tmp_path / "report.json"
    rc = cli.main([*SHEET, "--after", str(SHEET_391), "--json", str(report_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2024 Lexus RX 350, 20,000 mi · 11 lines" in out
    assert re.search(r"^11 lines decided in \d+ ms$", out, re.MULTILINE)
    assert (
        "Sheet totals: menu $672.00 · shop charges $78.29 · tax $48.77 · fees $1.99 · "
        "TOTAL $801.05" in out
    )
    assert 'NHTSA returns no campaign for model "RX 350"' in out
    assert 'Ask: "Are there open recalls on my VIN?"' in out
    assert "TOTAL $801.05 -> $391.38, $409.67 less" in out
    assert "Owner declined 4 lines; the agent said decline on 2 of them:" in out
    # a keep line has its reason and no question
    assert row_after(out, "Engine Oil and Oil Filter").strip() == (
        "-> keep. In the maker's schedule at 20,000 mi."
    )
    assert "retake" not in row(out, "Crankcase Cleaning Service")

    report = json.loads(report_path.read_text())
    assert {"totals", "recalls", "after", "outcomes", "counts"} <= set(report)
    assert report["totals"] == {
        "menu_total": 672.0,
        "shop_charges": 78.29,
        "tax": 48.77,
        "fees": 1.99,
        "total": 801.05,
    }
    assert report["recalls"]["matched"] == ["24V482000", "25V059000", "26V162000"]
    assert report["recalls"]["advisor_question"] == "Are there open recalls on my VIN?"
    after = report["after"]
    assert (after["before_total"], after["after_total"], after["difference"]) == (
        801.05,
        391.38,
        409.67,
    )
    assert [d["before_index"] for d in after["declined"]] == [2, 4, 8, 9]
    assert report["outcomes"][0]["line"]["prices_listed"] is False
    assert report["counts"]["retake"] == 0


def test_cli_without_after_has_no_after_block(fake_decider, capsys, tmp_path):
    report_path = tmp_path / "report.json"
    assert cli.main([*SHEET, "--json", str(report_path)]) == 0
    assert "After the owner's declines" not in capsys.readouterr().out
    assert json.loads(report_path.read_text())["after"] is None


def test_cli_civic_report_keeps_its_numbers_and_gets_the_new_keys(fake_decider, tmp_path):
    report_path = tmp_path / "report.json"
    assert cli.main([*OFFLINE, "--json", str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    assert report["after"] is None and report["totals"]["shop_charges"] == 34.5
    assert report["totals"]["total"] is None and report["menu_total"] == 2361.25
    assert len(report["recalls"]["campaigns"]) == 4 and not report["recalls"]["family_fallback"]


def test_cli_rejects_a_bad_recorded_query(fake_decider):
    with pytest.raises(SystemExit):
        cli.main([*ARGS, "--nhtsa-recorded", "x.json=LEXUS/RX 350"])


def test_cli_paid_flags_on_the_real_sheet_are_wired_with_fakes_only(monkeypatch, capsys):
    """`--allow-bedrock --max-usd 0.10` on the sheet fixtures: the Bedrock parts get the cap; the
    fakes stand in for them, so nothing is called and nothing is spent."""
    server = menu_server()
    monkeypatch.setattr(cli, "make_decider_client", lambda url, timeout=30.0: server.client())
    built = []

    def parts(decider, ledger, region, spend):
        built.append((region, spend.max_usd))
        return FakeWriter(), FakePicker()

    monkeypatch.setattr(cli, "make_bedrock_parts", parts)
    rc = cli.main([*SHEET, "--after", str(SHEET_391), "--allow-bedrock", "--max-usd", "0.10"])
    out = capsys.readouterr().out
    assert rc == 0 and built == [("us-east-1", 0.10)]
    assert "offline writer" not in out.splitlines()[0]
    assert "TOTAL $801.05 -> $391.38, $409.67 less" in out
    assert "Bedrock spend: $0.0000 of $0.10 cap, 0 calls" in out  # fakes spend nothing
    a_c = row(out, "A/C Evaporator")  # 0.70 decline: below 0.8, goes to the picker
    assert "ask_llm" in a_c and "llm" in a_c
