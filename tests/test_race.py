"""menu-race: Decider vs keyword baseline (vs an LLM picker) on labelled lines, no network."""

from __future__ import annotations

import pytest

from decider_kit.bench import percentile
from menu_line_check import race
from menu_line_check.baseline import keyword_tool
from menu_line_check.llm import FakePicker
from menu_line_check.race import (
    RACE_LINES_PATH,
    decider_picks,
    keyword_picks,
    llm_picks,
    load_race_lines,
    race_table,
    recorded_picks,
)

from .conftest import FakeServer, noul, ok_response
from .menu_fakes import tool_answer


def slice10():
    return load_race_lines(RACE_LINES_PATH)[:10]


def truthful_server(lines, wrong_id):
    """Answers the truth, except one line: wrong tool at confidence 0.55."""
    by_text = {ln.text: ln for ln in lines}

    def respond(body):
        text = body["state"].split(': "', 1)[1][:-1]
        line = by_text[text]
        if line.id == wrong_id:
            tool, conf = "decline_addon", 0.55
        else:
            tool, conf = line.tool, 0.95
        return ok_response(
            {
                "readable": noul(0.9),
                "safety": noul(0.9 if line.safety else 0.1),
                "tool": tool_answer(tool, conf),
            },
            latency_ms=600.0,
        )

    return FakeServer(respond)


def test_race_lines_file_has_the_103_readable_spike_lines():
    lines = load_race_lines(RACE_LINES_PATH)
    assert len(lines) == 103 and all(ln.readable for ln in lines)
    assert sum(ln.safety for ln in lines) == 20
    assert lines[0].id == "m00l00" and lines[0].text == "Rear brake pads $289.00"
    assert lines[0].recorded["tool"] == "owner_decides"


def test_decider_and_keyword_on_a_10_line_slice():
    lines = slice10()
    wrong = next(ln for ln in lines if ln.tool != "decline_addon").id
    server = truthful_server(lines, wrong)
    decider = decider_picks(lines, server.client())
    keyword = keyword_picks(lines)
    assert len(server.requests) == 10
    assert list(server.requests[0]["questions"]) == ["readable", "safety", "tool"]
    table = race_table(lines, [("Decider 2B", decider, "self-hosted"), ("keyword", keyword, "$0")])
    expected_kw = sum(keyword_tool(ln.text) == ln.tool for ln in lines) / 10
    rows = {r.split("|")[1].strip(): r for r in table.splitlines() if r.startswith("| ")}
    assert "| 10 | 0.90 |" in rows["Decider 2B"]
    assert "1.00 (9/10)" in rows["Decider 2B"]  # at >= 0.9 the wrong line drops out
    assert "0.90 (10/10)" in rows["Decider 2B"]  # at >= 0.5 everything is kept
    assert f"| {expected_kw:.2f} |" in rows["keyword"]
    assert "0/" in rows["Decider 2B"].split("|")[8]  # no safety line missed
    assert float(rows["Decider 2B"].split("|")[9]) >= 0  # wall-clock p50
    assert rows["Decider 2B"].rstrip().endswith("self-hosted |")


def test_recorded_spike_answers_reproduce_the_measured_numbers():
    lines = load_race_lines(RACE_LINES_PATH)
    table = race_table(lines, [("Decider 2B (recorded)", recorded_picks(lines), "self-hosted")])
    row = next(r for r in table.splitlines() if r.startswith("| Decider"))
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert cells[1:7] == [
        "103",
        "0.88",
        "0.90 (97/103)",
        "0.90 (81/103)",
        "0.86 (59/103)",
        "1.00 (21/103)",
    ]
    assert cells[7] == "0/20"
    recorded_ms = [ln.recorded["client_ms"] for ln in lines]
    assert cells[8] == f"{percentile(recorded_ms, 0.5):.0f}"


def test_llm_contestant_has_no_confidence_and_price_not_set_by_default(monkeypatch):
    lines = slice10()

    class CountingPicker(FakePicker):
        last_usage = {"inputTokens": 300, "outputTokens": 20}

    picks = llm_picks(lines, CountingPicker())
    table = race_table(lines, [("LLM picker", picks, None)])
    row = next(r for r in table.splitlines() if r.startswith("| LLM"))
    assert "n/a" in row and "price not set" in row
    monkeypatch.setattr(race, "PRICE_IN_PER_M", 0.06)
    monkeypatch.setattr(race, "PRICE_OUT_PER_M", 0.24)
    priced = race_table(lines, [("LLM picker", picks, None)])
    cost = 1000 * (300 * 0.06 + 20 * 0.24) / 1e6
    assert f"${cost:.4f}" in priced


def test_cli_runs_the_race_with_a_fake_decider(monkeypatch, tmp_path, capsys):
    lines = slice10()
    server = truthful_server(lines, "none")
    monkeypatch.setattr(race, "make_decider_client", lambda url: server.client())
    out = tmp_path / "race.md"
    assert race.main(["--limit", "10", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "| Decider 2B |" in printed and "| keyword rules |" in printed
    assert "LLM picker" not in printed  # not without --allow-bedrock
    assert out.read_text(encoding="utf-8").startswith("| contestant |")


def test_cli_replays_recorded_answers_without_any_server(monkeypatch, capsys):
    def must_not_build(url):
        raise AssertionError("no Decider client in --replay-recorded")

    monkeypatch.setattr(race, "make_decider_client", must_not_build)
    assert race.main(["--replay-recorded"]) == 0
    assert "1.00 (21/103)" in capsys.readouterr().out


def test_cli_refuses_bedrock_without_the_flag_and_reports_a_down_server(monkeypatch, capsys):
    server = FakeServer(lambda body: ok_response({}))
    server.health_status = 503
    monkeypatch.setattr(race, "make_decider_client", lambda url: server.client())
    assert race.main(["--limit", "3"]) == 3
    assert "unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["--thresholds", "--limit"])
def test_cli_bad_arguments(bad):
    with pytest.raises(SystemExit):
        race.main([bad, "x"])
