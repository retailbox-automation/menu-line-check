"""Shared test helpers: a fake Decider server built on httpx.MockTransport (no real model)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from decider_kit import DeciderClient


class FakeServer:
    """Records requests and answers from a responder ``(body) -> (status, json)``."""

    def __init__(self, responder: Callable[[dict[str, Any]], tuple[int, Any]]) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self.health_status = 200
        self.health_body: Any = {"status": "ok", "model": "fake-2b"}

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(self.health_status, json=self.health_body)
        assert request.url.path == "/v1/systemone"
        body = json.loads(request.content)
        self.requests.append(body)
        status, payload = self.responder(body)
        return httpx.Response(status, json=payload)

    def client(self, **kwargs: Any) -> DeciderClient:
        return DeciderClient(
            "http://decider.test", transport=httpx.MockTransport(self.handle), **kwargs
        )


def noul(p: float) -> dict[str, Any]:
    return {"type": "noul", "noul": p}


def choice(option: str, probs: dict[str, float], conf: float) -> dict[str, Any]:
    return {"type": "choice", "choice": option, "probabilities": probs, "confidence": conf}


def ok_response(answers: dict[str, Any], latency_ms: float = 12.5) -> tuple[int, Any]:
    return 200, {
        "model": "fake-2b",
        "answers": answers,
        "usage": {"input_tokens": 10, "output_tokens": 1},
        "latency_ms": latency_ms,
    }


@pytest.fixture
def server_factory():
    return FakeServer
