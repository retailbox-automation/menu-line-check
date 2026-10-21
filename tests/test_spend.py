"""The Bedrock/Textract spend cap: math, the stop before the next call, and who must not swallow it.

No AWS: Bedrock is a fake Converse client or a scripted Strands model, Textract is a fake client,
Decider is a fake HTTP transport.
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from menu_line_check import cli, race
from menu_line_check.llm import BedrockWriter, StrandsPicker
from menu_line_check.nhtsa import NhtsaClient
from menu_line_check.pipeline import check_menu
from menu_line_check.spend import (
    DEFAULT_MAX_USD,
    TEXTRACT_USD_PER_PAGE,
    SpendCap,
    SpendCapReached,
)

from .menu_fakes import NHTSA, TEXTRACT, menu_server, textract_response
from .test_llm import CIVIC, FakeConverse, ScriptedModel, fuel_outcome, line
from .test_pipeline import lines as menu_lines
from .test_race import slice10, truthful_server

LITE = "us.amazon.nova-lite-v1:0"
OFFLINE = [
    "--vehicle",
    "2019 Honda Civic 31200",
    "--nhtsa-recorded",
    str(NHTSA),
]


@pytest.fixture(autouse=True)
def _no_env_cap(monkeypatch):
    monkeypatch.delenv("N2_MAX_USD", raising=False)


# ---------------------------------------------------------------------------- the cap itself


def test_cost_comes_from_the_price_table_and_the_summary_reads_it_back():
    cap = SpendCap(1.0)
    cap.add(LITE, 1_000_000, 1_000_000)  # Nova Lite: 0.06 in + 0.24 out per 1M tokens
    assert cap.usd == pytest.approx(0.30) and cap.calls == 1
    assert cap.summary == "Bedrock spend: $0.3000 of $1.00 cap, 1 calls"
    cap.add(LITE, 500, 100)
    assert cap.usd == pytest.approx(0.30 + (500 * 0.06 + 100 * 0.24) / 1e6)
    assert cap.calls == 2


def test_cap_defaults_to_one_dollar_and_reads_the_env(monkeypatch):
    assert SpendCap().max_usd == DEFAULT_MAX_USD == 1.0
    monkeypatch.setenv("N2_MAX_USD", "2.5")
    assert SpendCap().max_usd == 2.5
    assert SpendCap(0.25).max_usd == 0.25  # an explicit value wins over the env
    monkeypatch.setenv("N2_MAX_USD", "lots")
    with pytest.raises(ValueError, match="N2_MAX_USD"):
        SpendCap()


def test_check_passes_below_the_cap_and_raises_at_it():
    cap = SpendCap(0.30)
    cap.check(LITE)
    cap.add(LITE, 1_000_000, 1_000_000)  # exactly at the cap
    with pytest.raises(SpendCapReached, match=r"cap reached: \$0\.3000 of \$0\.30 after 1 calls"):
        cap.check(LITE)


def test_an_unpriced_model_cannot_be_capped_so_it_is_refused():
    cap = SpendCap(1.0)
    with pytest.raises(SpendCapReached, match="cannot be capped"):
        cap.check("us.example.unpriced-model-v1:0")


def test_textract_pages_are_priced_into_the_same_cap():
    cap = SpendCap(0.002)
    cap.check_textract()
    cap.add_textract()
    assert TEXTRACT_USD_PER_PAGE == 0.0015 and cap.usd == pytest.approx(0.0015)
    cap.add(LITE, 1_000_000, 0)  # +0.06
    with pytest.raises(SpendCapReached):
        cap.check_textract()
    assert cap.summary.startswith("AWS spend (Bedrock + Textract): $0.0615 of $0.00 cap")


# --------------------------------------------------------------------- the Bedrock writer


def test_writer_adds_up_usage_and_stops_before_the_call_that_would_be_over_the_cap():
    client = FakeConverse("Is this covered?")  # usage 50 in / 12 out = $0.00000588 a call
    spend = SpendCap(0.00001)
    writer = BedrockWriter(client, allow_bedrock=True, spend=spend)
    outcome = fuel_outcome()
    for _ in range(2):  # 0.00000588 < cap: allowed; 0.00001176 >= cap after the second
        writer.advisor_question(outcome.line, outcome, CIVIC)
    assert spend.calls == 2 and spend.usd == pytest.approx(2 * 5.88e-6)
    with pytest.raises(SpendCapReached):
        writer.advisor_question(outcome.line, outcome, CIVIC)
    assert len(client.calls) == 2  # the third call never reached the client


def test_a_writer_made_without_a_cap_gets_a_default_one():
    writer = BedrockWriter(FakeConverse("x"), allow_bedrock=True)
    assert writer.spend.max_usd == DEFAULT_MAX_USD


# --------------------------------------------------------------------- the Strands picker


def picker_model():
    """Two picks, each two model steps of 100 in / 10 out: 200 in / 20 out = $0.0000168 a pick."""
    pick = [
        ("tool", "check_recall_warranty", {"line_index": 1, "line_text": "Tech diagnostic"}),
        ("text", "A diagnostic fee may be covered."),
    ]
    return ScriptedModel(pick * 2)


def test_picker_accumulates_and_raises_at_the_cap_before_the_agent_runs():
    model = picker_model()
    spend = SpendCap(0.00002)
    picker = StrandsPicker(model=model, model_id=LITE, spend=spend)
    for _ in range(2):  # $0.0000168 < cap: allowed; $0.0000336 >= cap after the second
        assert picker(line("Tech diagnostic $189.00"), CIVIC)[0] == "check_recall_warranty"
    assert spend.calls == 2 and spend.usd == pytest.approx(2 * 1.68e-5)
    steps_so_far = model.calls
    with pytest.raises(SpendCapReached):
        picker(line("Tech diagnostic $189.00"), CIVIC)
    assert model.calls == steps_so_far  # the agent did not run


def test_a_scripted_picker_without_spend_is_unchecked_but_a_bedrock_one_is_always_capped(
    monkeypatch,
):
    assert StrandsPicker(model=picker_model()).spend is None
    import strands.models.bedrock as bedrock

    monkeypatch.setattr(bedrock, "BedrockModel", lambda **kwargs: object())
    monkeypatch.delenv("N2_PICKER_MODEL", raising=False)
    assert StrandsPicker(allow_bedrock=True).spend.max_usd == DEFAULT_MAX_USD


# ------------------------------------------------------------------------------ check_menu


def test_check_menu_lets_a_spend_cap_from_the_picker_stop_the_run():
    def capped(line, vehicle):
        raise SpendCapReached("cap")

    with pytest.raises(SpendCapReached):
        check_menu(
            menu_lines(), CIVIC, menu_server().client(), llm_picker=capped,
            nhtsa=NhtsaClient.from_recorded(NHTSA),
        )  # fmt: skip


def test_check_menu_still_falls_back_on_an_ordinary_picker_error():
    def broken(line, vehicle):
        raise RuntimeError("throttled")

    report = check_menu(
        menu_lines(), CIVIC, menu_server().client(), llm_picker=broken,
        nhtsa=NhtsaClient.from_recorded(NHTSA),
    )  # fmt: skip
    assert any(o.route == "ask_advisor" and "throttled" in o.rationale for o in report.outcomes)


def test_check_menu_lets_a_spend_cap_from_the_writer_stop_the_run():
    client = FakeConverse("q")
    writer = BedrockWriter(client, allow_bedrock=True, spend=SpendCap(0.0))
    with pytest.raises(SpendCapReached):
        check_menu(
            menu_lines(), CIVIC, menu_server().client(), writer=writer,
            nhtsa=NhtsaClient.from_recorded(NHTSA),
        )  # fmt: skip
    assert client.calls == []


# ------------------------------------------------------------------------------- menu-race


class CappedPicker:
    model_id = LITE
    last_usage = {"inputTokens": 0, "outputTokens": 0}

    def __call__(self, line, vehicle=None):
        raise SpendCapReached("AWS spend cap reached: stop")


def test_race_stops_with_exit_code_4_when_the_cap_is_reached(monkeypatch, capsys):
    server = truthful_server(slice10(), "none")
    monkeypatch.setattr(race, "make_decider_client", lambda url: server.client())
    monkeypatch.setattr(race, "make_llm_picker", lambda spend: CappedPicker())
    assert race.main(["--limit", "10", "--allow-bedrock", "--max-usd", "0.5"]) == 4
    captured = capsys.readouterr()
    assert "error: AWS spend cap reached" in captured.err
    assert "LLM picker failed" not in captured.err  # not scored as a wrong answer per line
    assert "| Decider 2B |" not in captured.out


def test_race_prints_the_spend_after_an_allowed_bedrock_race(monkeypatch, capsys):
    made = []

    def make(spend):
        made.append(spend)
        model = ScriptedModel(
            [("tool", "ask_about_fee", {"line_index": 1, "line_text": "x"}), ("text", "ok")]
        )
        return StrandsPicker(model=model, model_id=LITE, spend=spend)

    monkeypatch.setattr(race, "make_llm_picker", make)
    assert (
        race.main(["--replay-recorded", "--limit", "3", "--allow-bedrock", "--max-usd", "2"]) == 0
    )
    out = capsys.readouterr().out
    assert made[0].max_usd == 2.0 and made[0].calls == 3
    assert f"Bedrock spend: ${made[0].usd:.4f} of $2.00 cap, 3 calls" in out
    assert race.main(["--replay-recorded", "--limit", "3"]) == 0
    assert "Bedrock spend" not in capsys.readouterr().out  # none without --allow-bedrock


def test_race_rejects_a_negative_cap():
    with pytest.raises(SystemExit):
        race.main(["--allow-bedrock", "--max-usd", "-1"])


# ------------------------------------------------------------------------------ menu-check


@pytest.fixture
def fake_decider(monkeypatch):
    server = menu_server()
    monkeypatch.setattr(cli, "make_decider_client", lambda url, timeout=30.0: server.client())
    return server


def test_menu_check_exits_4_and_prints_the_error_when_the_cap_is_reached(
    fake_decider, monkeypatch, capsys
):
    def parts(decider, ledger, region, spend):
        assert spend.max_usd == 0.75
        return BedrockWriter(FakeConverse("q"), allow_bedrock=True, spend=spend), CappedPicker()

    monkeypatch.setattr(cli, "make_bedrock_parts", parts)
    args = ["--ocr-json", str(TEXTRACT), *OFFLINE, "--allow-bedrock", "--max-usd", "0.75"]
    assert cli.main(args) == 4
    assert "error: AWS spend cap reached" in capsys.readouterr().err


def test_menu_check_prints_one_spend_line_after_the_report(fake_decider, monkeypatch, capsys):
    shared = []

    def parts(decider, ledger, region, spend):
        shared.append(spend)
        client = FakeConverse("Ask about it.")
        return BedrockWriter(client, allow_bedrock=True, spend=spend), None

    monkeypatch.setattr(cli, "make_bedrock_parts", parts)
    assert cli.main(["--ocr-json", str(TEXTRACT), *OFFLINE, "--allow-bedrock"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert shared[0].calls > 0 and shared[0].max_usd == 1.0
    assert out[-1] == shared[0].summary and out[-1].startswith("Bedrock spend: $0.0")


# ----------------------------------------------------------------- menu-check --textract


class FakeTextract:
    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def detect_document_text(self, *, Document):  # noqa: N803 - boto3's argument name
        self.calls.append(Document["Bytes"])
        return textract_response()


@pytest.fixture
def photo(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (900, 1300), "white").save(buf, "JPEG")
    path = tmp_path / "menu1.jpg"
    path.write_bytes(buf.getvalue())
    return path


@pytest.fixture
def fake_textract(monkeypatch):
    client, regions = FakeTextract(), []

    def make(region):
        regions.append(region)
        return client

    monkeypatch.setattr(cli, "make_textract_client", make)
    client.regions = regions
    return client


def test_textract_is_refused_without_allow_textract(fake_decider, fake_textract, photo, capsys):
    assert cli.main(["--textract", str(photo), *OFFLINE]) == 2
    err = capsys.readouterr().err
    assert "--allow-textract" in err and "costs money" in err
    assert fake_textract.calls == [] and fake_textract.regions == []
    # --allow-bedrock alone does not unlock it either
    assert cli.main(["--textract", str(photo), *OFFLINE, "--allow-bedrock"]) == 2
    assert fake_textract.calls == []


def test_ocr_json_and_textract_are_mutually_exclusive_and_one_is_required(photo):
    with pytest.raises(SystemExit):
        cli.main(["--ocr-json", str(TEXTRACT), "--textract", str(photo), *OFFLINE])
    with pytest.raises(SystemExit):
        cli.main(OFFLINE)


def test_textract_runs_once_saves_the_response_counts_the_spend(
    fake_decider, fake_textract, photo, tmp_path, capsys
):
    out_dir = tmp_path / "out"
    args = ["--textract", str(photo), "--allow-textract", *OFFLINE, "--region", "us-west-2"]
    assert cli.main([*args, "--json", str(out_dir / "report.json")]) == 0
    out = capsys.readouterr().out
    saved = out_dir / "textract-menu1.json"
    assert len(fake_textract.calls) == 1 and fake_textract.regions == ["us-west-2"]
    assert json.loads(saved.read_text(encoding="utf-8")) == textract_response()
    assert f"Textract response saved to {saved}" in out
    assert "12 lines decided" in out and out.count("\n 1  ") + out.count("\n12  ") == 2
    assert out.splitlines()[-1] == "AWS spend (Bedrock + Textract): $0.0015 of $1.00 cap, 1 calls"
    # the photo is used for the line crops: 12 lines x (text request + image request)
    assert len(fake_decider.requests) == 24
    # the saved response replays at no cost
    assert cli.main(["--ocr-json", str(saved), *OFFLINE]) == 0
    assert len(fake_textract.calls) == 1 and "spend" not in capsys.readouterr().out


def test_textract_response_is_saved_next_to_the_ledger_when_there_is_no_json_out(
    fake_decider, fake_textract, photo, tmp_path
):
    ledger = tmp_path / "led" / "ledger.jsonl"
    args = ["--textract", str(photo), "--allow-textract", *OFFLINE, "--ledger", str(ledger)]
    assert cli.main(args) == 0
    assert (tmp_path / "led" / "textract-menu1.json").exists()


def test_textract_stops_at_the_cap_before_calling(fake_decider, fake_textract, photo, capsys):
    args = ["--textract", str(photo), "--allow-textract", *OFFLINE, "--max-usd", "0"]
    assert cli.main(args) == 4
    assert "error: AWS spend cap reached" in capsys.readouterr().err
    assert fake_textract.calls == []


def test_textract_response_goes_to_the_current_directory_by_default(
    fake_decider, fake_textract, photo, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # no --json, no --ledger: the current directory
    assert cli.main(["--textract", str(photo), "--allow-textract", *OFFLINE]) == 0
    assert (tmp_path / "textract-menu1.json").exists()
