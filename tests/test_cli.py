"""menu-check on the Textract fixture with a fake Decider (monkeypatched client factory)."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from decider_kit import read_ledger
from menu_line_check import cli
from menu_line_check.llm import FakePicker, FakeWriter

from .conftest import FakeServer, ok_response
from .menu_fakes import NHTSA, TEXTRACT, menu_server

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
    assert out.count("Keep 3 · Decline 4 · Ask 4 · Your call 1") == 1
    assert "Is this covered by recall 23V858000?" in out
    fuel_row = next(r for r in out.splitlines() if "Replace fuel pump module" in r)
    for cell in ("$1,089.00", "tool", "check_recall_warranty", "0.84", "decider"):
        assert cell in fuel_row
    diag_row = next(r for r in out.splitlines() if "Tech diagnostic" in r)
    assert "ask_advisor" in diag_row and "code" in diag_row
    report = json.loads(report_path.read_text())
    assert report["counts"] == {"keep": 3, "decline": 4, "ask": 4, "your_call": 1, "retake": 0}
    assert len(read_ledger(ledger_path)) == 12


def test_cli_threshold_changes_the_route(fake_decider, capsys):
    assert cli.main([*OFFLINE, "--act-threshold", "0.7"]) == 0
    diag_row = next(r for r in capsys.readouterr().out.splitlines() if "Tech diagnostic" in r)
    assert "check_recall_warranty" in diag_row and "decider" in diag_row


def test_cli_with_a_photo_asks_readable_on_the_line_crops(fake_decider, tmp_path, capsys):
    photo = tmp_path / "menu.jpg"
    buf = io.BytesIO()
    Image.new("RGB", (900, 1300), "white").save(buf, "JPEG")
    photo.write_bytes(buf.getvalue())
    assert cli.main([*OFFLINE, "--image", str(photo)]) == 0
    with_images = [r for r in fake_decider.requests if "images" in r]
    assert len(fake_decider.requests) == 24 and len(with_images) == 12
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

    def parts(decider, ledger, region):
        built.append(region)
        return FakeWriter(), FakePicker()

    monkeypatch.setattr(cli, "make_bedrock_parts", parts)
    assert cli.main(OFFLINE) == 0 and built == []
    assert cli.main([*OFFLINE, "--allow-bedrock"]) == 0 and built == ["us-east-1"]
    diag_row = next(r for r in capsys.readouterr().out.splitlines()[::-1] if "Tech diag" in r)
    assert "ask_llm" in diag_row and "llm" in diag_row
