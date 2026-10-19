"""HTTP client for a Strands Decider server (``POST /v1/systemone``, ``GET /health``).

The real model runs on another machine; this module only talks HTTP. Transport trouble
(connection refused, timeout, 5xx, unreadable body) raises :class:`DeciderUnavailable`:
the caller decides whether that means fail-open or fail-closed. A 4xx (for example the
422 the server returns for an over-long state under ``--strict-window``, or too many
options) is a caller error and raises :class:`DeciderRequestError` instead.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from .questions import Content, Question, questions_to_wire

DEFAULT_TIMEOUT_S = 5.0
MAX_IMAGES = 4


class DeciderError(Exception):
    """Base class for client errors."""


class DeciderUnavailable(DeciderError):  # noqa: N818 - the name is part of the public contract
    """The server could not be reached or did not return a usable answer.

    ``kind`` is one of ``timeout``, ``transport``, ``http_5xx``, ``bad_response``.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"decider unavailable ({kind}): {message}")
        self.kind = kind
        self.message = message


class DeciderRequestError(DeciderError):
    """The server rejected the request (HTTP 4xx). Retrying the same request will not help."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"decider rejected the request (HTTP {status}): {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class NoulAnswer:
    """Yes/no. ``probability`` is P(true). The server returns no confidence for noul."""

    probability: float
    type: Literal["noul"] = "noul"

    @property
    def verdict(self) -> bool:
        return self.probability >= 0.5

    @property
    def confidence(self) -> float:
        """max(p, 1 - p): 0.5 = coin flip, 1.0 = certain. Derived client-side."""
        return max(self.probability, 1.0 - self.probability)


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: Literal["choice"] = "choice"


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    confidence: float
    probabilities: dict[str, float]
    legend: dict[str, str] = field(default_factory=dict)
    type: Literal["score"] = "score"


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class DeciderResponse:
    answers: dict[str, Answer]
    model: str
    client_ms: float
    """Wall-clock time measured around the HTTP call, including network and parsing."""
    server_ms: float | None
    """``latency_ms`` reported by the server (None if the server did not send it)."""
    usage: dict[str, int] = field(default_factory=dict)


def parse_answer(name: str, raw: Mapping[str, Any]) -> Answer:
    """Parse one answer object from the response body."""
    try:
        kind = raw["type"]
        if kind == "noul":
            return NoulAnswer(probability=float(raw["noul"]))
        if kind == "choice":
            return ChoiceAnswer(
                choice=str(raw["choice"]),
                probabilities={str(k): float(v) for k, v in raw["probabilities"].items()},
                confidence=float(raw["confidence"]),
            )
        if kind == "score":
            return ScoreAnswer(
                score=float(raw["score"]),
                confidence=float(raw["confidence"]),
                probabilities={str(k): float(v) for k, v in raw["probabilities"].items()},
                legend={str(k): str(v) for k, v in (raw.get("legend") or {}).items()},
            )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise DeciderUnavailable("bad_response", f"answer {name!r} is malformed: {exc!r}") from exc
    raise DeciderUnavailable("bad_response", f"answer {name!r} has unknown type {kind!r}")


class DeciderClient:
    """Synchronous and asynchronous client for one Decider server.

    ``transport`` is for tests (``httpx.MockTransport``); leave it None in real use.
    The async path opens a short-lived ``httpx.AsyncClient`` per call: Strands may run
    each agent call on a fresh event loop, and a pooled async client must not outlive its loop.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self._transport = transport
        self._sync: httpx.Client | None = None

    # ------------------------------------------------------------------ request building

    def _body(
        self,
        state: Content,
        questions: Mapping[str, Question],
        images: list[str] | tuple[str, ...] | None,
    ) -> dict[str, Any]:
        if not questions:
            raise ValueError("at least one question is required")
        if images and len(images) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images are allowed, got {len(images)}")
        body: dict[str, Any] = {"state": state, "questions": questions_to_wire(questions)}
        if self.model:
            body["model"] = self.model
        if images:
            body["images"] = list(images)
        return body

    def _parse(
        self,
        response: httpx.Response,
        questions: Mapping[str, Question],
        client_ms: float,
    ) -> DeciderResponse:
        status = response.status_code
        if status >= 500:
            raise DeciderUnavailable("http_5xx", f"HTTP {status}: {response.text[:200]}")
        if status >= 400:
            raise DeciderRequestError(status, response.text[:500])
        try:
            data = response.json()
            raw_answers = data["answers"]
        except (ValueError, KeyError, TypeError) as exc:
            raise DeciderUnavailable("bad_response", f"unreadable body: {exc!r}") from exc
        if not isinstance(raw_answers, dict):
            raise DeciderUnavailable("bad_response", "'answers' is not an object")
        answers: dict[str, Answer] = {}
        for name in questions:
            if name not in raw_answers:
                raise DeciderUnavailable("bad_response", f"answer {name!r} missing")
            answers[name] = parse_answer(name, raw_answers[name])
        server_ms = data.get("latency_ms")
        usage = data.get("usage") or {}
        return DeciderResponse(
            answers=answers,
            model=str(data.get("model", "")),
            client_ms=client_ms,
            server_ms=float(server_ms) if isinstance(server_ms, int | float) else None,
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int | float)},
        )

    @staticmethod
    def _transport_error(exc: httpx.HTTPError) -> DeciderUnavailable:
        if isinstance(exc, httpx.TimeoutException):
            return DeciderUnavailable("timeout", repr(exc))
        return DeciderUnavailable("transport", repr(exc))

    # ------------------------------------------------------------------ sync API

    def _client(self) -> httpx.Client:
        if self._sync is None:
            self._sync = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self._transport
                if isinstance(self._transport, httpx.BaseTransport)
                else None,
            )
        return self._sync

    def ask(
        self,
        state: Content,
        questions: Mapping[str, Question],
        images: list[str] | tuple[str, ...] | None = None,
    ) -> DeciderResponse:
        body = self._body(state, questions, images)
        started = time.perf_counter()
        try:
            response = self._client().post("/v1/systemone", json=body)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        return self._parse(response, questions, (time.perf_counter() - started) * 1000.0)

    def health(self) -> dict[str, Any]:
        """GET /health. Raises DeciderUnavailable if the server is not up and healthy."""
        try:
            response = self._client().get("/health")
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        return self._parse_health(response)

    @staticmethod
    def _parse_health(response: httpx.Response) -> dict[str, Any]:
        if response.status_code != 200:
            raise DeciderUnavailable("http_5xx", f"health returned HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise DeciderUnavailable("bad_response", f"health body unreadable: {exc!r}") from exc
        if not isinstance(data, dict) or data.get("status") != "ok":
            raise DeciderUnavailable("bad_response", f"health reports {data!r}")
        return data

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    def __enter__(self) -> DeciderClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ async API

    def _async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self._transport
            if isinstance(self._transport, httpx.AsyncBaseTransport)
            else None,
        )

    async def aask(
        self,
        state: Content,
        questions: Mapping[str, Question],
        images: list[str] | tuple[str, ...] | None = None,
    ) -> DeciderResponse:
        body = self._body(state, questions, images)
        started = time.perf_counter()
        try:
            async with self._async_client() as client:
                response = await client.post("/v1/systemone", json=body)
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        return self._parse(response, questions, (time.perf_counter() - started) * 1000.0)

    async def ahealth(self) -> dict[str, Any]:
        try:
            async with self._async_client() as client:
                response = await client.get("/health")
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        return self._parse_health(response)
