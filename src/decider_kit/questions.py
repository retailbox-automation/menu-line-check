"""Typed question definitions for the Decider ``/v1/systemone`` endpoint.

Three primitives, mirroring the server schema (``src/strands_decider/schema.py`` in
strands-labs/strands-decider):

* ``noul``   - yes/no, optional ``criteria`` ``{"true": ..., "false": ...}``.
* ``choice`` - one of 2..255 options; ``criteria`` maps option name -> description (or None).
* ``score``  - ordered rubric of 2..10 levels; ``criteria`` is an ascending list.

The size limits are validated here so a bad question file fails at load time, not as an
HTTP 422 in the middle of an agent run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

Content = str | dict[str, Any] | list[Any]

MAX_CHOICE_OPTIONS = 255
MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class QuestionError(ValueError):
    """A question definition is invalid."""


def _check_instructions(value: Any) -> None:
    if isinstance(value, str):
        ok = bool(value.strip())
    elif isinstance(value, dict | list):
        ok = bool(value)
    else:
        ok = False
    if not ok:
        raise QuestionError("instructions must be a non-empty string, mapping or list")


@dataclass(frozen=True)
class NoulQuestion:
    """Yes/no question. The answer is P(true)."""

    instructions: Content
    criteria: dict[str, Content | None] | None = None
    type: str = field(default="noul", init=False)

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if self.criteria is not None:
            bad = set(self.criteria) - {"true", "false"}
            if bad:
                raise QuestionError(
                    f"noul criteria keys must be a subset of 'true'/'false', got {sorted(bad)}"
                )

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = dict(self.criteria)
        return out


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick one of N named options. ``criteria`` maps option -> description (or None)."""

    instructions: Content
    criteria: dict[str, Content | None]
    type: str = field(default="choice", init=False)

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        n = len(self.criteria)
        if n < MIN_CHOICE_OPTIONS:
            raise QuestionError(f"choice requires at least {MIN_CHOICE_OPTIONS} options, got {n}")
        if n > MAX_CHOICE_OPTIONS:
            raise QuestionError(f"choice allows at most {MAX_CHOICE_OPTIONS} options, got {n}")
        for key in self.criteria:
            if not isinstance(key, str) or not key:
                raise QuestionError(f"choice option names must be non-empty strings, got {key!r}")

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class ScoreQuestion:
    """Rate against an ordered rubric. ``criteria`` is ascending: index 0 is the low end."""

    instructions: Content
    criteria: list[str]
    type: str = field(default="score", init=False)

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        n = len(self.criteria)
        if not (MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS):
            raise QuestionError(
                f"score requires between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels, got {n}"
            )
        if not all(isinstance(level, str) for level in self.criteria):
            raise QuestionError("score levels must be strings")

    def to_wire(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


def _check_description(key: Any, value: Any) -> None:
    if value is not None and not isinstance(value, str | dict | list):
        raise QuestionError(
            f"description for {key!r} is {value!r} ({type(value).__name__}), not text. YAML reads "
            "unquoted yes/no/on/off/true/false and bare numbers as non-strings: quote the text."
        )


def _normalise_noul_criteria(raw: Mapping[Any, Any]) -> dict[str, Content | None]:
    # PyYAML (YAML 1.1) turns an unquoted `true:` / `false:` key into a bool.
    out: dict[str, Content | None] = {}
    for key, value in raw.items():
        if key is True:
            key = "true"
        elif key is False:
            key = "false"
        _check_description(key, value)
        out[key] = value
    return out


def _normalise_choice_criteria(raw: Mapping[Any, Any], name: str) -> dict[str, Content | None]:
    out: dict[str, Content | None] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise QuestionError(
                f"question {name!r}: option name {key!r} is not a string. YAML reads unquoted "
                "yes/no/on/off/true/false and bare numbers as non-strings: quote the option name."
            )
        _check_description(key, value)
        out[key] = value
    return out


def parse_question(name: str, raw: Mapping[str, Any]) -> Question:
    """Build one question from its dict form (the wire shape)."""
    if not isinstance(raw, Mapping):
        raise QuestionError(f"question {name!r} must be a mapping")
    unknown = set(raw) - {"type", "instructions", "criteria"}
    if unknown:
        raise QuestionError(f"question {name!r}: unknown keys {sorted(unknown)}")
    qtype = raw.get("type")
    if "instructions" not in raw:
        raise QuestionError(f"question {name!r}: missing 'instructions'")
    instructions = raw["instructions"]
    criteria = raw.get("criteria")
    try:
        if qtype == "noul":
            if criteria is not None and not isinstance(criteria, Mapping):
                raise QuestionError("noul criteria must be a mapping")
            return NoulQuestion(
                instructions, None if criteria is None else _normalise_noul_criteria(criteria)
            )
        if qtype == "choice":
            if not isinstance(criteria, Mapping):
                raise QuestionError("choice requires 'criteria' as a mapping option -> description")
            return ChoiceQuestion(instructions, _normalise_choice_criteria(criteria, name))
        if qtype == "score":
            if not isinstance(criteria, list):
                raise QuestionError("score requires 'criteria' as an ascending list")
            return ScoreQuestion(instructions, [str(x) for x in criteria])
    except QuestionError as exc:
        raise QuestionError(f"question {name!r}: {exc}") from exc
    raise QuestionError(f"question {name!r}: type must be noul, choice or score, got {qtype!r}")


def parse_questions(raw: Mapping[str, Any]) -> dict[str, Question]:
    """Build a name -> Question mapping from a dict (YAML or JSON content)."""
    if not isinstance(raw, Mapping) or not raw:
        raise QuestionError("questions must be a non-empty mapping of name -> definition")
    return {str(name): parse_question(str(name), body) for name, body in raw.items()}


def load_questions(path: str | Path) -> dict[str, Question]:
    """Load and validate questions from a YAML file (top level: name -> definition)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise QuestionError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return parse_questions(data)
    except QuestionError as exc:
        raise QuestionError(f"{path}: {exc}") from exc


def questions_to_wire(questions: Mapping[str, Question]) -> dict[str, dict[str, Any]]:
    """Serialise questions for the request body."""
    return {name: q.to_wire() for name, q in questions.items()}
