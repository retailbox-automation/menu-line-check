"""Confidence-threshold routing: ACT / CONFIRM / HUMAN.

The convention comes from the Decider architecture docs (docs/architecture.md, "The
routing convention"): confidence >= 0.9 act without a check, 0.5 to 0.9 confirm first,
below 0.5 send the case to a person.

**These numbers are a starting point, not a calibration.** The Decider docs themselves say
the right thresholds depend on the cost of a wrong action and must be measured on your own
traffic. Measure them on our own labelled cases (``decider-bench``) before relying on them.

Confidence per answer type:

* ``noul``   - ``max(p, 1 - p)``; the verdict is the side with the higher probability.
* ``choice`` - the server's normalised max-probability ``confidence``; verdict = chosen option.
* ``score``  - the server's ordinal ``confidence``; verdict = the rounded level index.

Note for noul: ``max(p, 1 - p)`` is never below 0.5, so with the default ``confirm=0.5`` a
noul answer is never routed to HUMAN. Raise ``confirm`` for noul questions (``per_question``)
if you want an "unsure" band, e.g. 0.65.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .client import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer


class Route(StrEnum):
    ACT = "act"
    CONFIRM = "confirm"
    HUMAN = "human"


@dataclass(frozen=True)
class Thresholds:
    act: float = 0.9
    confirm: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.confirm <= self.act <= 1.0):
            raise ValueError(
                f"thresholds must satisfy 0 <= confirm <= act <= 1, got confirm={self.confirm} "
                f"act={self.act}"
            )


@dataclass(frozen=True)
class RouteResult:
    route: Route
    verdict: str
    confidence: float


def confidence_and_verdict(answer: Answer) -> tuple[float, str]:
    if isinstance(answer, NoulAnswer):
        return answer.confidence, "true" if answer.verdict else "false"
    if isinstance(answer, ChoiceAnswer):
        return answer.confidence, answer.choice
    if isinstance(answer, ScoreAnswer):
        return answer.confidence, str(round(answer.score))
    raise TypeError(f"unsupported answer type: {type(answer)!r}")


@dataclass(frozen=True)
class ThresholdPolicy:
    default: Thresholds = field(default_factory=Thresholds)
    per_question: Mapping[str, Thresholds] = field(default_factory=dict)

    def thresholds_for(self, name: str | None) -> Thresholds:
        if name is not None and name in self.per_question:
            return self.per_question[name]
        return self.default

    def route(self, answer: Answer, name: str | None = None) -> RouteResult:
        confidence, verdict = confidence_and_verdict(answer)
        t = self.thresholds_for(name)
        if confidence >= t.act:
            route = Route.ACT
        elif confidence >= t.confirm:
            route = Route.CONFIRM
        else:
            route = Route.HUMAN
        return RouteResult(route=route, verdict=verdict, confidence=confidence)

    def route_all(self, answers: Mapping[str, Answer]) -> dict[str, RouteResult]:
        return {name: self.route(answer, name) for name, answer in answers.items()}

    def weakest(self, answers: Mapping[str, Answer]) -> Route:
        """The most cautious route across all answers (HUMAN > CONFIRM > ACT)."""
        order = {Route.ACT: 0, Route.CONFIRM: 1, Route.HUMAN: 2}
        routes = [r.route for r in self.route_all(answers).values()]
        return max(routes, key=order.__getitem__) if routes else Route.HUMAN
