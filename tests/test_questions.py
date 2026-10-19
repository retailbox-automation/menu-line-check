from __future__ import annotations

import pytest

from decider_kit import (
    ChoiceQuestion,
    NoulQuestion,
    QuestionError,
    ScoreQuestion,
    load_questions,
    parse_questions,
)


def write(tmp_path, text):
    p = tmp_path / "q.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_file_all_three_types(tmp_path):
    p = write(
        tmp_path,
        """
a:
  type: noul
  instructions: Is it ready?
  criteria:
    "true": ready
    "false": not ready
b:
  type: choice
  instructions: Which one?
  criteria:
    first: the first
    second: null
c:
  type: score
  instructions: How risky?
  criteria: [low, medium, high]
""",
    )
    qs = load_questions(p)
    assert isinstance(qs["a"], NoulQuestion) and isinstance(qs["b"], ChoiceQuestion)
    assert isinstance(qs["c"], ScoreQuestion)
    assert qs["a"].to_wire()["criteria"] == {"true": "ready", "false": "not ready"}
    assert qs["b"].to_wire()["criteria"] == {"first": "the first", "second": None}


def test_yaml_unquoted_true_false_keys_are_normalised(tmp_path):
    p = write(
        tmp_path,
        "a:\n  type: noul\n  instructions: x?\n  criteria:\n    true: ok\n    false: not ok\n",
    )
    assert load_questions(p)["a"].criteria == {"true": "ok", "false": "not ok"}


def test_yaml_unquoted_yes_no_description_is_rejected(tmp_path):
    p = write(
        tmp_path,
        "a:\n  type: noul\n  instructions: x?\n  criteria:\n    true: yes\n    false: no\n",
    )
    with pytest.raises(QuestionError, match="quote"):
        load_questions(p)


def test_yaml_unquoted_boolean_looking_option_is_rejected(tmp_path):
    p = write(
        tmp_path, "b:\n  type: choice\n  instructions: ok?\n  criteria:\n    yes: a\n    no: b\n"
    )
    with pytest.raises(QuestionError, match="quote the option name"):
        load_questions(p)


def test_noul_rejects_unknown_criteria_key():
    with pytest.raises(QuestionError, match="subset"):
        NoulQuestion("x?", {"maybe": "..."})


@pytest.mark.parametrize("n,ok", [(1, False), (2, True), (255, True), (256, False)])
def test_choice_option_count_limits(n, ok):
    criteria = {f"o{i}": None for i in range(n)}
    if ok:
        assert len(ChoiceQuestion("pick", criteria).criteria) == n
    else:
        with pytest.raises(QuestionError, match="options"):
            ChoiceQuestion("pick", criteria)


@pytest.mark.parametrize("n,ok", [(1, False), (2, True), (10, True), (11, False)])
def test_score_level_limits(n, ok):
    levels = [f"l{i}" for i in range(n)]
    if ok:
        assert ScoreQuestion("rate", levels).to_wire()["criteria"] == levels
    else:
        with pytest.raises(QuestionError, match="levels"):
            ScoreQuestion("rate", levels)


def test_empty_instructions_and_bad_type_and_missing_fields():
    with pytest.raises(QuestionError):
        NoulQuestion("   ")
    with pytest.raises(QuestionError, match="type must be"):
        parse_questions({"a": {"type": "multi", "instructions": "x"}})
    with pytest.raises(QuestionError, match="missing 'instructions'"):
        parse_questions({"a": {"type": "noul"}})
    with pytest.raises(QuestionError, match="unknown keys"):
        parse_questions({"a": {"type": "noul", "instructions": "x", "extra": 1}})
    with pytest.raises(QuestionError, match="non-empty mapping"):
        parse_questions({})


def test_error_message_names_the_question_and_file(tmp_path):
    p = write(tmp_path, "bad:\n  type: choice\n  instructions: pick\n  criteria: {only: one}\n")
    with pytest.raises(QuestionError) as exc:
        load_questions(p)
    assert "bad" in str(exc.value) and "q.yaml" in str(exc.value)
