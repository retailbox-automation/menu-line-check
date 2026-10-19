from __future__ import annotations

import json

from strands.interventions import Confirm, Guide, Proceed

from decider_kit import ChoiceAnswer, Ledger, NoulAnswer, read_ledger
from decider_kit.ledger import state_hash

ANSWERS = {"q": NoulAnswer(0.83), "c": ChoiceAnswer("x", {"x": 0.9, "y": 0.1}, 0.8)}


def test_record_fields_and_append_only(tmp_path):
    path = tmp_path / "sub" / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.record(
        gate="g",
        stage="before_tool_call",
        state="the secret state",
        answers=ANSWERS,
        action=Guide(feedback="ask first"),
        routes={"q": "confirm"},
        tool="get_weather",
        client_ms=61.234,
        server_ms=57.0,
    )
    ledger.record(
        gate="g", stage="before_tool_call", state="other", answers=ANSWERS, action=Proceed()
    )
    rows = read_ledger(path)
    assert len(rows) == 2
    first = rows[0]
    assert first["ts"].endswith("Z") and first["questions"] == ["q", "c"]
    assert first["answers"]["q"]["probability"] == 0.83
    assert first["confidences"]["q"] == 0.83 and first["confidences"]["c"] == 0.8
    assert first["action"] == {"type": "guide", "feedback": "ask first"}
    assert first["routes"] == {"q": "confirm"} and first["tool"] == "get_weather"
    assert first["client_ms"] == 61.23 and first["server_ms"] == 57.0
    assert first["state_sha256"] == state_hash("the secret state")
    assert rows[1]["action"] == {"type": "proceed"}
    assert path.read_text().count("\n") == 2


def test_redact_true_never_stores_raw_state(tmp_path):
    path = tmp_path / "l.jsonl"
    Ledger(path, redact=True).record(
        gate="g", stage="s", state="TOP-SECRET-STATE", answers=ANSWERS, action=Proceed()
    )
    text = path.read_text()
    assert "TOP-SECRET-STATE" not in text and '"state"' not in text
    assert "state_sha256" in text


def test_redact_false_stores_state(tmp_path):
    path = tmp_path / "l.jsonl"
    Ledger(path, redact=False).record(
        gate="g", stage="s", state={"a": 1}, answers=ANSWERS, action=Confirm(prompt="ok?")
    )
    row = json.loads(path.read_text())
    assert row["state"] == {"a": 1} and row["action"] == {"type": "confirm", "prompt": "ok?"}


def test_unavailable_entry_has_no_answers(tmp_path):
    path = tmp_path / "l.jsonl"
    Ledger(path).record(
        gate="g", stage="s", state="x", answers=None, action=Proceed(), unavailable="timeout"
    )
    row = read_ledger(path)[0]
    assert row["unavailable"] == "timeout" and row["answers"] == {} and row["questions"] == []


def test_state_hash_is_stable_for_structures():
    assert state_hash({"a": 1, "b": 2}) == state_hash({"b": 2, "a": 1})
    assert state_hash("x") != state_hash("y")
