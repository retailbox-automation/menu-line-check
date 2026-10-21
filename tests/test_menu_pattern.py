"""The menu_line_tool_selection pattern: questions, routing code, and the decline safety gate."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from strands.hooks.events import BeforeToolCallEvent
from strands.interventions import Confirm, Deny, Proceed

from decider_kit import (
    ChoiceAnswer,
    ChoiceQuestion,
    DeciderGate,
    Ledger,
    NoulAnswer,
    NoulQuestion,
    read_ledger,
)
from menu_line_tool_selection import (
    DEFAULT_THRESHOLDS,
    TOOLS,
    LineDecision,
    build_safety_gate,
    decide_line,
    load_pattern_questions,
    ocr_state,
    parse_thresholds,
    resolve_thresholds,
)

from .conftest import FakeServer, noul, ok_response


def answers(p_readable=0.95, p_safety=0.05, tool="decline_addon", conf=0.9):
    probs = {t: (conf if t == tool else (1 - conf) / 4) for t in TOOLS}
    return {
        "readable": NoulAnswer(p_readable),
        "safety": NoulAnswer(p_safety),
        "tool": ChoiceAnswer(tool, probs, conf),
    }


# ---------------------------------------------------------------------------- questions


def test_both_wordings_load_with_the_spike_tested_text():
    text = load_pattern_questions("ocr_text")
    image = load_pattern_questions("image")
    assert list(text) == list(image) == ["readable", "safety", "tool"]
    assert isinstance(text["readable"], NoulQuestion) and isinstance(text["tool"], ChoiceQuestion)
    assert text["readable"].instructions == (
        "Can you read both the name of the service and its dollar price on this OCR text of one "
        "menu line?"
    )
    assert image["readable"].instructions == (
        "Can you read both the name of the service and its dollar price on this menu line?"
    )
    assert image["safety"].instructions == (
        "Is this menu line about brakes, tires, wheel alignment, steering, or another part that "
        "affects driving safety?"
    )
    assert text["tool"].instructions == (
        "A car owner at the dealer's service counter holds the menu. Which action fits this OCR "
        "text of one menu line?"
    )
    assert text["readable"].criteria == {
        "true": "both the service name and the price are legible",
        "false": "blurred, cut off, or the name or the price cannot be read",
    }
    assert text["safety"].criteria == {
        "true": "a driving-safety part",
        "false": "not a driving-safety part",
    }
    assert tuple(text["tool"].criteria) == TOOLS
    assert text["tool"].criteria == image["tool"].criteria
    assert text["tool"].criteria["ask_about_fee"] == (
        "a fee or surcharge rather than a service (shop supplies, disposal, handling, admin)"
    )


SPIKE_CRITERIA = {
    "decline_addon": "an optional add-on product or chemical treatment the maker's schedule "
    "does not require (conditioners, cleanings, additives, nitrogen, protection packages)",
    "check_schedule": "routine maintenance the owner's manual schedules by mileage "
    "(oil and filter, filters, rotation, fluid changes, spark plugs, inspection)",
    "check_recall_warranty": "a diagnostic fee or the repair of one specific failed part, "
    "which an open recall or the warranty may cover",
    "ask_about_fee": "a fee or surcharge rather than a service "
    "(shop supplies, disposal, handling, admin)",
    "owner_decides": "brakes, tires, alignment, steering or another part that affects "
    "driving safety",
}


def test_default_tool_criteria_are_the_spike_sentences_word_for_word():
    for mode in ("ocr_text", "image"):
        assert load_pattern_questions(mode)["tool"].criteria == SPIKE_CRITERIA
        assert load_pattern_questions(mode, criteria="spike")["tool"].criteria == SPIKE_CRITERIA


def test_short_criteria_keep_options_and_change_only_the_criteria():
    for mode in ("ocr_text", "image"):
        spike, short = load_pattern_questions(mode), load_pattern_questions(mode, criteria="short")
        assert list(short["tool"].criteria) == list(TOOLS)
        assert short["tool"].instructions == spike["tool"].instructions
        assert short["safety"] == spike["safety"] and short["readable"] == spike["readable"]
        for option in TOOLS:
            assert len(short["tool"].criteria[option]) < len(spike["tool"].criteria[option])
    with pytest.raises(ValueError, match="criteria"):
        load_pattern_questions(criteria="tiny")


def test_question_subset_and_bad_mode():
    assert list(load_pattern_questions("image", names=["readable"])) == ["readable"]
    with pytest.raises(ValueError, match="mode"):
        load_pattern_questions("braille")
    with pytest.raises(KeyError):
        load_pattern_questions("ocr_text", names=["colour"])


def test_ocr_state_quotes_the_line_text():
    assert ocr_state("Shop supplies $34.50") == (
        'OCR text of one line of a car dealer\'s printed service menu: "Shop supplies $34.50"'
    )


# ---------------------------------------------------------------------------- routing


def test_unreadable_line_is_retaken_before_anything_else():
    d = decide_line(answers(p_readable=0.2, p_safety=0.99, tool="owner_decides", conf=0.99))
    assert isinstance(d, LineDecision) and d.route == "retake" and d.tool is None


def test_tire_rotation_flagged_by_safety_still_checks_the_schedule():
    # live run 2026-10-20: P(safety) 0.68 on "Tire rotation"; safety is no override of lookups
    d = decide_line(answers(p_safety=0.68, tool="check_schedule", conf=0.56))
    assert (d.route, d.tool, d.threshold) == ("tool", "check_schedule", 0.5)


def test_decline_on_a_line_flagged_by_safety_is_not_declined_but_goes_to_the_llm():
    d = decide_line(answers(p_safety=0.66, tool="decline_addon", conf=0.78))  # nitrogen tire fill
    assert (d.route, d.tool) == ("ask_llm", None) and "not declined" in d.reason
    sure = decide_line(answers(p_safety=0.55, tool="decline_addon", conf=0.99))
    assert sure.route == "ask_llm" and sure.decider_choice == "decline_addon"
    assert sure.p_safety == pytest.approx(0.55)


def test_fuel_pump_recall_at_064_runs_the_lookup():
    d = decide_line(answers(p_safety=0.2, tool="check_recall_warranty", conf=0.64))
    assert (d.route, d.tool, d.confidence) == ("tool", "check_recall_warranty", 0.64)


def test_owner_decides_choice_routes_to_owner_at_any_confidence():
    d = decide_line(answers(p_safety=0.1, tool="owner_decides", conf=0.3))
    assert (d.route, d.tool, d.confidence) == ("owner_decides", "owner_decides", 0.3)


def test_safety_alone_no_longer_makes_a_line_the_owners_call():
    for tool in ("check_schedule", "check_recall_warranty", "ask_about_fee"):
        d = decide_line(answers(p_safety=0.95, tool=tool, conf=0.9))
        assert (d.route, d.tool) == ("tool", tool)


def test_per_tool_default_thresholds():
    assert DEFAULT_THRESHOLDS == {
        "decline_addon": 0.8,
        "check_schedule": 0.5,
        "check_recall_warranty": 0.5,
        "ask_about_fee": 0.5,
    }
    assert decide_line(answers(tool="decline_addon", conf=0.8)).route == "tool"
    assert decide_line(answers(tool="decline_addon", conf=0.79)).route == "ask_llm"
    assert decide_line(answers(tool="ask_about_fee", conf=0.5)).route == "tool"
    low = decide_line(answers(tool="check_schedule", conf=0.49))
    assert (low.route, low.tool, low.decider_choice) == ("ask_llm", None, "check_schedule")


def test_thresholds_are_configurable_and_act_threshold_overrides_all():
    strict = {"check_schedule": 0.9}
    assert decide_line(answers(tool="check_schedule", conf=0.85), strict).route == "ask_llm"
    assert decide_line(answers(tool="ask_about_fee", conf=0.55), strict).route == "tool"
    one = decide_line(answers(tool="check_recall_warranty", conf=0.64), act_threshold=0.8)
    assert one.route == "ask_llm" and one.threshold == 0.8


def test_resolve_and_parse_thresholds():
    assert resolve_thresholds({"decline_addon": 0.9})["decline_addon"] == 0.9
    assert resolve_thresholds(None, act_threshold=0.7) == dict.fromkeys(DEFAULT_THRESHOLDS, 0.7)
    assert parse_thresholds("decline_addon=0.85, check_schedule=0.6") == {
        "decline_addon": 0.85,
        "check_schedule": 0.6,
    }
    with pytest.raises(ValueError, match="unknown tool"):
        resolve_thresholds({"owner_decides": 0.5})
    with pytest.raises(ValueError, match="between 0 and 1"):
        resolve_thresholds({"decline_addon": 1.5})
    with pytest.raises(ValueError, match="tool=value"):
        parse_thresholds("decline_addon:0.8")


def test_boundaries_of_the_yes_no_cut_offs():
    assert decide_line(answers(p_readable=0.5)).route == "tool"
    assert decide_line(answers(p_readable=0.4999)).route == "retake"
    assert decide_line(answers(p_safety=0.5, tool="decline_addon")).route == "ask_llm"
    assert decide_line(answers(p_safety=0.4999, tool="decline_addon")).route == "tool"


def test_an_unknown_option_goes_to_the_llm():
    d = decide_line(
        answers(tool="check_schedule", conf=0.99) | {"tool": ChoiceAnswer("x", {}, 1.0)}
    )
    assert d.route == "ask_llm" and "unknown" in d.reason


def test_wrong_answer_types_are_rejected():
    bad = answers()
    bad["safety"] = ChoiceAnswer("x", {"x": 1.0}, 1.0)
    with pytest.raises(TypeError, match="safety"):
        decide_line(bad)
    with pytest.raises(KeyError):
        decide_line({"readable": NoulAnswer(0.9)})


# ---------------------------------------------------------------------------- safety gate


def tool_event(name, args):
    return BeforeToolCallEvent(
        agent=SimpleNamespace(messages=[]),
        selected_tool=None,
        tool_use={"name": name, "toolUseId": "t1", "input": args},
        invocation_state={},
    )


def run(coro):
    return asyncio.run(coro)


def test_gate_denies_a_decline_on_a_safety_line_and_sends_the_line_text(tmp_path):
    server = FakeServer(lambda body: ok_response({"safety": noul(0.93)}))
    ledger = Ledger(tmp_path / "l.jsonl")
    gate = build_safety_gate(server.client(), ledger=ledger)
    assert isinstance(gate, DeciderGate)
    event = tool_event("decline_addon", {"line_index": 9, "line_text": "Front brake pads $489.00"})
    action = run(gate.before_tool_call(event))
    assert isinstance(action, Deny)
    assert action.reason == "this line is a safety item; the owner decides"
    sent = server.requests[0]
    assert sent["state"] == ocr_state("Front brake pads $489.00")
    assert list(sent["questions"]) == ["safety"]
    assert "OCR text of one menu line" in sent["questions"]["safety"]["instructions"]
    assert read_ledger(tmp_path / "l.jsonl")[0]["action"]["type"] == "deny"


def test_gate_lets_a_decline_on_a_non_safety_line_proceed():
    server = FakeServer(lambda body: ok_response({"safety": noul(0.04)}))
    gate = build_safety_gate(server.client())
    event = tool_event("decline_addon", {"line_index": 2, "line_text": "Engine oil conditioner"})
    assert isinstance(run(gate.before_tool_call(event)), Proceed)


def test_gate_ignores_other_tools_without_calling_decider():
    server = FakeServer(lambda body: ok_response({"safety": noul(0.99)}))
    gate = build_safety_gate(server.client())
    event = tool_event("check_schedule", {"line_index": 1, "line_text": "Front brake pads"})
    assert isinstance(run(gate.before_tool_call(event)), Proceed)
    assert server.requests == []


def test_gate_denies_when_the_line_text_is_missing_or_decider_is_down():
    server = FakeServer(lambda body: (503, {}))
    gate = build_safety_gate(server.client())
    missing = run(gate.before_tool_call(tool_event("decline_addon", {"line_index": 3})))
    assert isinstance(missing, Deny) and "line text" in missing.reason
    down = run(
        gate.before_tool_call(
            tool_event("decline_addon", {"line_index": 3, "line_text": "Nitrogen tire fill"})
        )
    )
    assert isinstance(down, Deny) and "unavailable" in down.reason
    confirm = build_safety_gate(server.client(), on_unavailable=Confirm(prompt="decline it?"))
    asked = run(
        confirm.before_tool_call(tool_event("decline_addon", {"line_text": "Nitrogen tire fill"}))
    )
    assert isinstance(asked, Confirm)


def test_gate_threshold_is_configurable():
    server = FakeServer(lambda body: ok_response({"safety": noul(0.35)}))
    event = tool_event("decline_addon", {"line_text": "Nitrogen tire fill $29.95"})
    assert isinstance(run(build_safety_gate(server.client()).before_tool_call(event)), Proceed)
    strict = build_safety_gate(server.client(), threshold=0.3)
    assert isinstance(run(strict.before_tool_call(event)), Deny)
