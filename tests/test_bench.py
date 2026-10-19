from __future__ import annotations

import json

import pytest

from decider_kit import bench
from decider_kit.bench import (
    BaselineReply,
    Case,
    SpendNotApproved,
    build_baseline_prompt,
    load_cases,
    percentile,
    require_spend_approval,
    run_baseline,
    run_bench,
)
from decider_kit.questions import ChoiceQuestion, NoulQuestion

from .conftest import FakeServer, choice, noul, ok_response

QS = {
    "ready": NoulQuestion("Ready?"),
    "tool": ChoiceQuestion("Which tool?", {"get_weather": None, "ask": None}),
}

# state -> (answers, latency). Three cases; the last answer on case 3 is wrong and low-confidence.
SCRIPT = {
    "s1": (
        {
            "ready": noul(0.97),
            "tool": choice("get_weather", {"get_weather": 0.98, "ask": 0.02}, 0.96),
        },
        50.0,
    ),
    "s2": (
        {"ready": noul(0.08), "tool": choice("ask", {"get_weather": 0.2, "ask": 0.8}, 0.6)},
        60.0,
    ),
    "s3": (
        {
            "ready": noul(0.6),
            "tool": choice("get_weather", {"get_weather": 0.55, "ask": 0.45}, 0.1),
        },
        70.0,
    ),
}
CASES = [
    Case("c1", "s1", QS, {"ready": True, "tool": "get_weather"}),
    Case("c2", "s2", QS, {"ready": False, "tool": "ask"}),
    Case("c3", "s3", QS, {"ready": False, "tool": "ask"}),  # both wrong
]


def fake_server():
    def responder(body):
        answers, latency = SCRIPT[body["state"]]
        return ok_response(answers, latency)

    return FakeServer(responder)


def test_accuracy_abstain_and_latency():
    report = run_bench(CASES, fake_server().client(), thresholds=(0.5, 0.9))
    d = report.to_dict()
    assert d["cases"] == 3 and d["answers"] == 6 and d["unavailable"] == 0
    assert d["accuracy_all"] == pytest.approx(4 / 6)
    # confidences: c1 .97/.96, c2 .92/.6, c3 .6/.1
    t09 = report.at_threshold(0.9)
    assert (t09["answered"], t09["accuracy"], t09["abstain_rate"]) == (3, 1.0, pytest.approx(0.5))
    t05 = report.at_threshold(0.5)
    assert t05["answered"] == 5 and t05["accuracy"] == pytest.approx(4 / 5)
    assert d["server_ms"] == {"p50": 60.0, "p90": 70.0}
    assert d["client_ms"]["p50"] is not None
    text = report.render()
    assert "threshold" in text and "abstain" in text and "p50" in text


def test_unavailable_cases_are_counted_not_scored():
    server = FakeServer(lambda body: (503, {}))
    report = run_bench(CASES, server.client())
    assert report.unavailable == 3 and report.items == [] and report.accuracy_all() is None
    assert "n/a" in report.render()


def test_warmup_requests_are_discarded():
    server = fake_server()
    report = run_bench(CASES, server.client(), warmup=2)
    assert len(server.requests) == 5 and len(report.client_ms) == 3


def test_percentile_nearest_rank():
    assert percentile([], 0.5) is None
    assert percentile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 0.9) == 90
    assert percentile([5], 0.9) == 5


def test_load_cases_from_jsonl_with_inline_and_shared_questions(tmp_path):
    p = tmp_path / "cases.jsonl"
    inline = {
        "id": "a",
        "state": "s1",
        "expected": {"ready": True},
        "questions": {"ready": {"type": "noul", "instructions": "Ready?"}},
    }
    shared = {"state": "s2", "expected": {"ready": False}}
    p.write_text(json.dumps(inline) + "\n\n" + json.dumps(shared) + "\n")
    with pytest.raises(ValueError, match="no 'questions'"):
        load_cases(p)
    cases = load_cases(p, {"ready": NoulQuestion("Ready?")})
    assert [c.id for c in cases] == ["a", "3"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**inline, "expected": {"nope": 1}}) + "\n")
    with pytest.raises(ValueError, match="unknown questions"):
        load_cases(bad)


def test_cli_end_to_end_with_fake_server(tmp_path, monkeypatch, capsys):
    server = fake_server()
    monkeypatch.setattr(bench, "DeciderClient", lambda url, timeout=0: server.client())
    p = tmp_path / "cases.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": c.id,
                    "state": c.state,
                    "questions": {n: q.to_wire() for n, q in c.questions.items()},
                    "expected": c.expected,
                }
            )
            for c in CASES
        )
        + "\n"
    )
    out = tmp_path / "r.json"
    assert (
        bench.main(
            ["--cases", str(p), "--thresholds", "0.5,0.9", "--warmup", "0", "--json", str(out)]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "accuracy over all answers:  66.7%" in printed
    assert json.loads(out.read_text())["decider"]["answers"] == 6


def test_baseline_refuses_without_spend_flag_and_never_builds_a_client(
    tmp_path, monkeypatch, capsys
):
    with pytest.raises(SpendNotApproved):
        require_spend_approval("bedrock", False)
    require_spend_approval(None, False)  # no baseline requested: fine
    require_spend_approval("bedrock", True)

    def must_not_be_called(*a, **k):
        raise AssertionError("a Bedrock client was built without approval")

    monkeypatch.setattr(bench, "make_bedrock_invoker", must_not_be_called)
    p = tmp_path / "cases.jsonl"
    p.write_text(
        json.dumps(
            {
                "state": "s",
                "expected": {"ready": True},
                "questions": {"ready": {"type": "noul", "instructions": "Ready?"}},
            }
        )
        + "\n"
    )
    rc = bench.main(["--cases", str(p), "--baseline", "bedrock", "--model-id", "some-model"])
    assert rc == 2 and "--i-approve-spend" in capsys.readouterr().err


def test_baseline_scoring_with_a_fake_invoker():
    replies = iter(
        [
            BaselineReply('{"answers": {"ready": true, "tool": "get_weather"}}', 100, 20),
            BaselineReply('Sure! {"answers": {"ready": true, "tool": "ask"}}', 100, 20),
            BaselineReply("I cannot answer that", 100, 5),
        ]
    )
    report = run_baseline(CASES, lambda prompt: next(replies), "fake-model", 3.0, 15.0)
    assert report.answers == 6 and report.correct == 2 + 1 + 0 and report.unparsable == 1
    assert (report.input_tokens, report.output_tokens) == (300, 45)
    assert report.cost_usd == pytest.approx((300 * 3.0 + 45 * 15.0) / 1e6)
    assert "fake-model" in report.render()


def test_baseline_prompt_carries_options_and_levels():
    from decider_kit.questions import ScoreQuestion

    prompt = build_baseline_prompt(
        "a state",
        {
            "tool": ChoiceQuestion("Which tool?", {"get_weather": "weather", "ask": None}),
            "risk": ScoreQuestion("Risk?", ["low", "high"]),
        },
    )
    assert "a state" in prompt and "get_weather: weather" in prompt and "1: high" in prompt
    assert prompt.rstrip().endswith('"<question name>": <answer>, ...}}')
