"""The writer only rephrases: code drafts every question and every number, and rejects any model
text that adds a topic, the make or model, or a number. Fake Converse clients only; no Bedrock."""

from __future__ import annotations

import json
from typing import Any

import pytest

from decider_kit import Ledger, read_ledger
from menu_line_check.llm import BedrockWriter, FakeWriter
from menu_line_check.models import SheetTotals, Vehicle
from menu_line_check.ocr import parse_sheet
from menu_line_check.pipeline import check_menu
from menu_line_check.writing import question_problem, sentence_problem

from .menu_fakes import SHEET_391, SHEET_801, menu_server

SHEET_CAR = Vehicle.parse("2024 Lexus RX 350 20000")

# The texts Nova Lite wrote in the paid run of 2026-10-21 (out/run-rx350-nova-1.txt).
BAD_KEEP = (
    "Does the engine oil and oil filter change service correspond to any recall campaign for my "
    "2024 Lexus RX 350?"
)
BAD_CABIN = (
    'Does the "REPLACE Cabin Filter, Incl. Sanitizer" service include any recall components?'
)
BAD_MAKE = "Does the Lexus RX 350's wheel alignment service need to be performed at this time?"
BAD_SUMMARY = (
    "The total cost of the menu is 80.28 USD, and the total cost of the declined services is "
    "0.0 USD."
)
CABIN_DRAFT = (
    "The maker's schedule lists a cabin air filter at 20,000 mi: can you do just that, without "
    "the sanitizer?"
)


class ScriptedConverse:
    """A fake Bedrock Converse client: replies by the line in the facts, or to the summary."""

    def __init__(self, questions: dict[str, str], summary: str) -> None:
        self.questions, self.summary = questions, summary
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        facts = json.loads(kwargs["messages"][0]["content"][0]["text"])
        if "line" in facts:
            reply = next((v for k, v in self.questions.items() if k in facts["line"]), "")
        else:
            reply = self.summary
        return {
            "output": {"message": {"content": [{"text": reply}]}},
            "usage": {"inputTokens": 40, "outputTokens": 10},
        }


def sheet(path):
    return parse_sheet(json.loads(path.read_text(encoding="utf-8")))


def run(client, tmp_path=None):
    before = sheet(SHEET_801)
    writer = BedrockWriter(client, allow_bedrock=True) if client else FakeWriter()
    ledger = Ledger(tmp_path / "l.jsonl") if tmp_path else None
    return check_menu(
        before.lines,
        SHEET_CAR,
        menu_server().client(),
        writer=writer,
        ledger=ledger,
        totals=before.totals,
        after_sheet=sheet(SHEET_391),
    )


def by_text(report, fragment):
    (o,) = [o for o in report.outcomes if fragment in o.line.text]
    return o


@pytest.fixture
def bad():
    return ScriptedConverse(
        {
            "engine oil and oil filter": BAD_KEEP,
            "cabin filter": BAD_CABIN,
            "wheel alignment": BAD_MAKE,
        },
        BAD_SUMMARY,
    )


def test_keep_and_decline_lines_get_no_question_and_the_writer_is_not_asked(bad):
    report = run(bad)
    for o in report.outcomes:
        if o.verdict in ("keep", "decline"):
            assert o.advisor_question is None and o.question_source is None
        else:
            assert o.advisor_question and o.question_source in ("bedrock", "template")
    asked = [json.loads(c["messages"][0]["content"][0]["text"]) for c in bad.calls]
    lines = [f["line"] for f in asked if "line" in f]
    assert not any("engine oil and oil filter" == line for line in lines)
    assert by_text(report, "Engine Oil and Oil Filter").verdict == "keep"


def test_the_bad_texts_are_rejected_and_the_drafts_used(bad, tmp_path):
    report = run(bad, tmp_path)
    cabin = by_text(report, "Cabin Filter")
    assert cabin.verdict == "ask" and cabin.advisor_question == CABIN_DRAFT
    assert cabin.question_source == "template"
    assert cabin.question_fallback == "mentions a recall but the facts have no campaign"
    alignment = by_text(report, "Wheel Alignment")
    assert alignment.verdict == "your_call" and alignment.question_source == "template"
    assert alignment.question_fallback == "names the make"
    assert alignment.advisor_question == (
        "What did the inspection find on the wheel alignment, and how soon does it need doing?"
    )
    rows = [r for r in read_ledger(tmp_path / "l.jsonl") if r["stage"] == "question"]
    reasons = [r["action"]["reason"] for r in rows]
    assert any(
        "question_source=template; writer text rejected: names the make" in r for r in reasons
    )


def test_a_summary_with_numbers_is_dropped_and_the_code_summary_kept(bad):
    report = run(bad)
    assert report.summary_source == "template" and report.summary_fallback == "contains a digit"
    assert "80.28" not in report.summary_text and "USD" not in report.summary_text
    assert report.summary_text.startswith(f"{report.summary}. Sheet total $801.05. ")
    assert "After the owner's declines: $391.38, $409.67 less." in report.summary_text
    data = report.to_dict()
    assert data["summary_source"] == "template" and data["summary_fallback"] == "contains a digit"
    assert {"question_source", "question_fallback"} <= set(data["outcomes"][0])


def test_the_writer_never_sees_the_make():
    client = ScriptedConverse({}, "")
    run(client)
    assert client.calls and all("Lexus" not in json.dumps(c) for c in client.calls)
    facts = json.loads(client.calls[0]["messages"][0]["content"][0]["text"])
    assert facts["vehicle"] == "2024 RX 350"


def test_a_good_rephrase_is_accepted():
    good = ScriptedConverse(
        {"cabin filter": "Could you replace just the cabin air filter, without the sanitizer?"},
        "Ask the advisor about every open line before you sign.",
    )
    report = run(good)
    cabin = by_text(report, "Cabin Filter")
    assert cabin.question_source == "bedrock" and cabin.question_fallback is None
    assert cabin.advisor_question == (
        "Could you replace just the cabin air filter, without the sanitizer?"
    )
    assert report.summary_source == "bedrock" and report.summary_fallback is None
    assert report.summary_text.endswith(
        "$409.67 less. 1 safety line is your call. Ask the advisor about every open line before "
        "you sign."
    )


def test_offline_and_bedrock_reports_have_the_same_structure(bad):
    offline, paid = run(None).to_dict(), run(bad).to_dict()
    assert set(offline) == set(paid)
    assert [set(o) for o in offline["outcomes"]] == [set(o) for o in paid["outcomes"]]
    assert [o["advisor_question"] is None for o in offline["outcomes"]] == [
        o["advisor_question"] is None for o in paid["outcomes"]
    ]
    assert offline["summary_source"] == "template" and offline["summary_fallback"] is None


@pytest.mark.parametrize(
    "text,facts,problem",
    [
        (BAD_KEEP, {"draft": "x?"}, "mentions a recall but the facts have no campaign"),
        ("Is this covered by recall 23V858000?", {"recall_campaign": "23V858000"}, None),
        ("Can I skip it on my Lexus?", {}, "names the make"),
        ("Is the RX 350 due for it?", {"vehicle": "2024 RX 350"}, "names the model"),
        ("Can you remove it.", {}, "not a question"),
        ("Is it due at 30,000 mi?", {"reason": "at 20,000 mi"}, "number not in the facts: 30"),
        ("Is the $78.29 charge optional?", {"price": "$ 78.29"}, None),
        ("Why" + " really" * 30 + "?", {}, "longer than 160 characters"),
        ("", {}, "empty"),
    ],
)
def test_question_checks(text, facts, problem):
    assert question_problem(text, facts, SHEET_CAR) == problem


@pytest.mark.parametrize(
    "text,problem",
    [
        (BAD_SUMMARY, "contains a digit"),
        ("Your Lexus needs nothing else.", "names the make"),
        ("Ask before you sign" + "." * 200, "longer than 200 characters"),
        ("Ask before you sign.", None),
    ],
)
def test_summary_sentence_checks(text, problem):
    assert sentence_problem(text, SHEET_CAR) == problem


def test_without_sheet_totals_the_summary_numbers_are_the_priced_lines(tmp_path):
    report = check_menu(
        sheet(SHEET_801).lines[9:],
        SHEET_CAR,
        menu_server().client(),
        writer=FakeWriter(),
        totals=SheetTotals(),
    )
    assert report.summary_text == (
        "Keep 0 · Decline 0 · Ask 2 · Your call 0. The priced lines total $80.28; the 0 lines to "
        "decline come to $0.00."
    )


def test_offline_drafts_on_the_real_sheet():
    questions = {o.line.index: o.advisor_question for o in run(None).outcomes}
    assert {i for i, q in questions.items() if q is None} == {1, 2, 3, 5, 8}
    assert questions[4] == CABIN_DRAFT
    assert questions[6] == "What does the lifetime protection planf cover, and can it be removed?"
    assert questions[9] == (
        "What is the a/c evaporator system cleaning service with climate control disinfectant "
        "for, and is it required by the maker's schedule?"
    )
    assert (
        questions[10] == "What does the shop charges line of $78.29 cover, and can it be removed?"
    )


def test_the_fee_reason_is_neutral_for_charges_and_plans_alike():
    report = run(None)
    for index in (6, 10, 11):  # the protection plan and the two charges (scripted ask_about_fee)
        outcome = next(o for o in report.outcomes if o.line.index == index)
        assert outcome.tool_result["note"] == "Ask what it covers and whether it is optional."
