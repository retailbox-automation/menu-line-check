from __future__ import annotations

import asyncio

import httpx
import pytest

from decider_kit import (
    ChoiceAnswer,
    ChoiceQuestion,
    DeciderClient,
    DeciderRequestError,
    DeciderUnavailable,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

from .conftest import FakeServer, choice, noul, ok_response

QS = {
    "ready": NoulQuestion("Ready?", {"true": "yes", "false": "no"}),
    "tool": ChoiceQuestion("Which tool?", {"get_weather": None, "ask": "ask the user"}),
    "risk": ScoreQuestion("How risky?", ["low", "mid", "high"]),
}
ANSWERS = {
    "ready": noul(0.83),
    "tool": choice("get_weather", {"get_weather": 0.994, "ask": 0.006}, 0.99),
    "risk": {
        "type": "score",
        "score": 1.2,
        "legend": {"0": "low", "1": "mid", "2": "high"},
        "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2},
        "confidence": 0.55,
    },
}


def test_request_shape_and_typed_answers():
    server = FakeServer(lambda body: ok_response(ANSWERS, 12.5))
    resp = server.client().ask("the state", QS)
    body = server.requests[0]
    assert body["state"] == "the state"
    assert body["questions"]["ready"] == {
        "type": "noul",
        "instructions": "Ready?",
        "criteria": {"true": "yes", "false": "no"},
    }
    assert body["questions"]["tool"]["criteria"] == {"get_weather": None, "ask": "ask the user"}
    assert body["questions"]["risk"]["criteria"] == ["low", "mid", "high"]
    assert "images" not in body and "model" not in body

    ready, tool, risk = resp.answers["ready"], resp.answers["tool"], resp.answers["risk"]
    assert isinstance(ready, NoulAnswer) and ready.probability == 0.83 and ready.verdict is True
    assert ready.confidence == pytest.approx(0.83)
    assert (
        isinstance(tool, ChoiceAnswer) and tool.choice == "get_weather" and tool.confidence == 0.99
    )
    assert isinstance(risk, ScoreAnswer) and risk.score == 1.2 and risk.legend["1"] == "mid"
    assert resp.server_ms == 12.5 and resp.client_ms > 0
    assert resp.usage == {"input_tokens": 10, "output_tokens": 1}


def test_noul_confidence_is_max_of_p_and_complement():
    assert NoulAnswer(0.2).confidence == pytest.approx(0.8)
    assert NoulAnswer(0.2).verdict is False
    assert NoulAnswer(0.5).confidence == 0.5


def test_images_are_sent_and_capped_at_four():
    server = FakeServer(lambda body: ok_response({"ready": noul(0.5)}))
    client = server.client()
    client.ask("s", {"ready": QS["ready"]}, images=["aGk="])
    assert server.requests[0]["images"] == ["aGk="]
    with pytest.raises(ValueError, match="at most 4"):
        client.ask("s", {"ready": QS["ready"]}, images=["a"] * 5)


def test_missing_server_latency_is_none():
    server = FakeServer(
        lambda body: (200, {"model": "m", "answers": {"ready": noul(0.9)}, "usage": {}})
    )
    assert server.client().ask("s", {"ready": QS["ready"]}).server_ms is None


def test_timeout_and_connect_errors_raise_unavailable():
    def boom_timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def boom_connect(request):
        raise httpx.ConnectError("refused", request=request)

    for handler, kind in ((boom_timeout, "timeout"), (boom_connect, "transport")):
        client = DeciderClient("http://x.test", transport=httpx.MockTransport(handler))
        with pytest.raises(DeciderUnavailable) as exc:
            client.ask("s", {"ready": QS["ready"]})
        assert exc.value.kind == kind


def test_5xx_is_unavailable_4xx_is_request_error():
    s5 = FakeServer(lambda body: (503, {"detail": "busy"}))
    with pytest.raises(DeciderUnavailable) as exc:
        s5.client().ask("s", {"ready": QS["ready"]})
    assert exc.value.kind == "http_5xx"
    s4 = FakeServer(lambda body: (422, {"detail": "state too long"}))
    with pytest.raises(DeciderRequestError) as exc2:
        s4.client().ask("s", {"ready": QS["ready"]})
    assert exc2.value.status == 422


def test_malformed_bodies_are_bad_response():
    cases = [
        (200, {"model": "m", "usage": {}}),  # no answers
        (200, {"answers": {}, "model": "m"}),  # answer missing
        (200, {"answers": {"ready": {"type": "noul"}}}),  # no probability
        (200, {"answers": {"ready": {"type": "weird"}}}),  # unknown type
    ]
    for status, payload in cases:
        server = FakeServer(lambda body, s=status, p=payload: (s, p))
        with pytest.raises(DeciderUnavailable) as exc:
            server.client().ask("s", {"ready": QS["ready"]})
        assert exc.value.kind == "bad_response"


def test_health_ok_and_failures():
    server = FakeServer(lambda body: ok_response({}))
    assert server.client().health()["status"] == "ok"
    server.health_status = 500
    with pytest.raises(DeciderUnavailable):
        server.client().health()
    server.health_status, server.health_body = 200, {"status": "loading"}
    with pytest.raises(DeciderUnavailable, match="health reports"):
        server.client().health()


def test_async_path_matches_sync_path():
    server = FakeServer(lambda body: ok_response(ANSWERS))
    client = server.client()
    resp = asyncio.run(client.aask("s", QS))
    assert resp.answers["ready"].probability == 0.83
    assert asyncio.run(client.ahealth())["status"] == "ok"
    assert len(server.requests) == 1
