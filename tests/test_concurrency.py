"""check_menu asks Decider for all lines concurrently: order, bound, failures, timing."""

from __future__ import annotations

import asyncio
import gc
import json

import httpx
import pytest

from decider_kit import DeciderClient, Ledger, read_ledger
from menu_line_check.models import Vehicle
from menu_line_check.nhtsa import NhtsaClient
from menu_line_check.ocr import parse_textract
from menu_line_check.pipeline import DEFAULT_CONCURRENCY, check_menu

from .menu_fakes import MENU_ANSWERS, NHTSA, menu_responder, textract_response

CIVIC = Vehicle(2019, "Honda", "Civic", 31200)
FRAGMENTS = list(MENU_ANSWERS)


def fragment_of(state: str) -> str:
    return next(f for f in FRAGMENTS if f in state)


class AsyncDecider:
    """An async fake Decider: answers line k after ``delay(k)`` seconds; can fail some lines."""

    def __init__(self, delay=lambda pos: 0.0, status_for=None, raise_for=()):
        self.delay = delay
        self.status_for = status_for or {}
        self.raise_for = set(raise_for)
        self.respond = menu_responder()
        self.in_flight = 0
        self.max_in_flight = 0
        self.completed: list[str] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        fragment = fragment_of(body["state"])
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay(FRAGMENTS.index(fragment)))
        finally:
            self.in_flight -= 1
        self.completed.append(fragment)
        if fragment in self.raise_for:
            raise httpx.ReadTimeout("slow", request=request)
        if fragment in self.status_for:
            return httpx.Response(self.status_for[fragment], json={"detail": "down"})
        status, payload = self.respond(body)
        return httpx.Response(status, json=payload)

    def client(self) -> DeciderClient:
        return DeciderClient("http://decider.test", transport=httpx.MockTransport(self.handle))


def run(fake: AsyncDecider, **kwargs):
    lines = parse_textract(textract_response())
    kwargs.setdefault("nhtsa", NhtsaClient.from_recorded(NHTSA))
    return check_menu(lines, CIVIC, fake.client(), **kwargs)


def test_answers_arriving_out_of_order_keep_line_order_and_ledger_order(tmp_path):
    # The last line answers first. Gaps of 10 ms and a collection first: a garbage-collector
    # pause while the requests are being sent (13 ms measured) must not reorder the answers.
    gc.collect()
    fake = AsyncDecider(delay=lambda pos: 0.01 * (12 - pos))
    report = run(fake, ledger=Ledger(tmp_path / "l.jsonl"))
    assert fake.completed[0] == FRAGMENTS[-1] and fake.completed != FRAGMENTS  # really reordered
    assert [o.line.index for o in report.outcomes] == list(range(1, 13))
    for outcome, fragment in zip(report.outcomes, FRAGMENTS, strict=True):
        assert fragment in outcome.line.name
        assert outcome.decider_choice == MENU_ANSWERS[fragment][0]
        assert outcome.confidence == pytest.approx(MENU_ANSWERS[fragment][1])
    rows = read_ledger(tmp_path / "l.jsonl")
    assert [r["action"]["reason"].split(":")[0] for r in rows] == [
        f"line {i}" for i in range(1, 13)
    ]
    assert report.summary == "Keep 3 · Decline 3 · Ask 5 · Your call 1"


def test_concurrency_bound_is_respected():
    assert DEFAULT_CONCURRENCY == 12
    bounded = AsyncDecider(delay=lambda pos: 0.01)
    run(bounded, concurrency=3)
    assert bounded.max_in_flight == 3
    default = AsyncDecider(delay=lambda pos: 0.01)
    run(default)
    assert default.max_in_flight == 12
    one = AsyncDecider(delay=lambda pos: 0.001)
    run(one, concurrency=1)
    assert one.max_in_flight == 1 and one.completed == FRAGMENTS
    with pytest.raises(ValueError, match="concurrency"):
        run(AsyncDecider(), concurrency=0)


def test_failing_lines_land_as_before_and_do_not_break_the_others(tmp_path):
    fake = AsyncDecider(status_for={"Shop supplies": 503}, raise_for={"Tech diagnostic"})
    report = run(fake, ledger=Ledger(tmp_path / "l.jsonl"))
    by_route = {o.line.name: o for o in report.outcomes}
    for name in ("Shop supplies", "Tech diagnostic"):
        assert (by_route[name].route, by_route[name].verdict) == ("unavailable", "your_call")
    assert by_route["Replace fuel pump module"].tool_result["recall"]["campaign"] == "23V858000"
    assert sum(o.route == "unavailable" for o in report.outcomes) == 2
    rows = read_ledger(tmp_path / "l.jsonl")
    assert [r["unavailable"] for r in rows if r["unavailable"]] == ["timeout", "http_5xx"]


def test_per_line_latency_and_the_wall_clock_of_the_decider_phase():
    report = run(AsyncDecider(delay=lambda pos: 0.05))
    per_line = [o.decider_ms.client_ms for o in report.outcomes]
    assert all(ms >= 50 for ms in per_line)
    assert all(o.decider_ms.server_ms == 12.5 for o in report.outcomes)
    assert 50 <= report.decider_wall_ms < sum(per_line) / 2  # concurrent, not one after another
    assert report.to_dict()["decider_wall_ms"] == report.decider_wall_ms


def test_works_when_called_inside_a_running_event_loop():
    async def caller():
        return run(AsyncDecider())

    report = asyncio.run(caller())
    assert len(report.outcomes) == 12 and report.counts["keep"] == 3
