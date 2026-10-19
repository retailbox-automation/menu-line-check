from __future__ import annotations

import pytest

from decider_kit import ChoiceAnswer, NoulAnswer, Route, ScoreAnswer, ThresholdPolicy, Thresholds


def ch(conf):
    return ChoiceAnswer("a", {"a": 0.9, "b": 0.1}, conf)


@pytest.mark.parametrize(
    "conf,route",
    [
        (0.95, Route.ACT),
        (0.9, Route.ACT),
        (0.8999, Route.CONFIRM),
        (0.5, Route.CONFIRM),
        (0.4999, Route.HUMAN),
        (0.0, Route.HUMAN),
    ],
)
def test_default_thresholds_on_choice(conf, route):
    assert ThresholdPolicy().route(ch(conf)).route is route


def test_noul_uses_max_of_p_and_complement_and_side():
    policy = ThresholdPolicy()
    hi = policy.route(NoulAnswer(0.97))
    assert (hi.route, hi.verdict, round(hi.confidence, 2)) == (Route.ACT, "true", 0.97)
    lo = policy.route(NoulAnswer(0.03))
    assert (lo.route, lo.verdict, round(lo.confidence, 2)) == (Route.ACT, "false", 0.97)
    mid = policy.route(NoulAnswer(0.7))
    assert (mid.route, mid.verdict) == (Route.CONFIRM, "true")


def test_noul_is_never_human_with_default_confirm_threshold():
    # max(p, 1-p) >= 0.5 always: the HUMAN band only exists if `confirm` is raised.
    assert ThresholdPolicy().route(NoulAnswer(0.5)).route is Route.CONFIRM
    strict = ThresholdPolicy(per_question={"q": Thresholds(act=0.9, confirm=0.65)})
    assert strict.route(NoulAnswer(0.55), "q").route is Route.HUMAN
    assert strict.route(NoulAnswer(0.55), "other").route is Route.CONFIRM


def test_score_uses_server_confidence_and_rounded_level():
    r = ThresholdPolicy().route(ScoreAnswer(2.4, 0.95, {"0": 0.1, "2": 0.9}))
    assert (r.route, r.verdict) == (Route.ACT, "2")


def test_custom_thresholds_and_validation():
    p = ThresholdPolicy(Thresholds(act=0.99, confirm=0.8))
    assert p.route(ch(0.95)).route is Route.CONFIRM
    assert p.route(ch(0.7)).route is Route.HUMAN
    with pytest.raises(ValueError):
        Thresholds(act=0.5, confirm=0.9)
    with pytest.raises(ValueError):
        Thresholds(act=1.2, confirm=0.5)


def test_weakest_route_across_answers():
    p = ThresholdPolicy()
    assert p.weakest({"a": ch(0.95), "b": ch(0.7)}) is Route.CONFIRM
    assert p.weakest({"a": ch(0.95), "b": ch(0.2)}) is Route.HUMAN
    assert p.weakest({"a": ch(0.95)}) is Route.ACT
    assert p.weakest({}) is Route.HUMAN
