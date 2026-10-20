"""NHTSA recalls client on the recorded live response (httpx.MockTransport, no network)."""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

import httpx
import pytest

from menu_line_check.models import Vehicle
from menu_line_check.nhtsa import NhtsaClient, NhtsaError, Recall, match_recall
from menu_line_check.ocr import parse_line_texts

FIXTURE = Path(__file__).parent / "fixtures" / "nhtsa_recalls_civic_2019.json"
CIVIC = Vehicle(2019, "Honda", "Civic", 31200)


class Recorder:
    def __init__(self, status=200, body=None):
        self.requests: list[httpx.Request] = []
        self.status = status
        self.body = body if body is not None else json.loads(FIXTURE.read_text())

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)


def client(recorder, **kwargs):
    return NhtsaClient(transport=httpx.MockTransport(recorder), **kwargs)


def line(text):
    return parse_line_texts([text])[0]


def test_request_and_parsed_recalls():
    rec = Recorder()
    recalls = client(rec).recalls(CIVIC)
    url = rec.requests[0].url
    assert url.host == "api.nhtsa.gov" and url.path == "/recalls/recallsByVehicle"
    assert dict(url.params) == {"make": "Honda", "model": "Civic", "modelYear": "2019"}
    assert [r.campaign for r in recalls] == ["20V314000", "21V215000", "23V858000", "26V332000"]
    fuel = recalls[2]
    assert isinstance(fuel, Recall) and fuel.component == "FUEL SYSTEM, GASOLINE:DELIVERY:FUEL PUMP"
    assert fuel.report_date == date(2023, 12, 18)
    assert fuel.remedy.startswith("Dealers will replace the fuel pump module")
    assert "engine stall" in fuel.consequence


def test_disk_cache_by_make_model_year(tmp_path):
    rec = Recorder()
    first = client(rec, cache_dir=tmp_path).recalls(CIVIC)
    again = client(rec, cache_dir=tmp_path).recalls(Vehicle(2019, "HONDA", "civic", 50000))
    assert first == again and len(rec.requests) == 1
    client(rec, cache_dir=tmp_path).recalls(Vehicle(2020, "Honda", "Civic", 100))
    assert len(rec.requests) == 2


def test_stale_cache_is_refetched(tmp_path):
    rec = Recorder()
    client(rec, cache_dir=tmp_path).recalls(CIVIC)
    (cached,) = tmp_path.glob("*.json")
    old = time.time() - 3 * 86400
    os.utime(cached, (old, old))
    client(rec, cache_dir=tmp_path, max_age_days=1).recalls(CIVIC)
    assert len(rec.requests) == 2


@pytest.mark.parametrize(
    "handler,kind",
    [
        (Recorder(status=500), "http"),
        (Recorder(body={"Count": 0, "Message": "odd"}), "bad_response"),
    ],
)
def test_errors_raise_a_typed_exception(handler, kind):
    with pytest.raises(NhtsaError) as exc:
        client(handler).recalls(CIVIC)
    assert exc.value.kind == kind


def test_timeout_and_connection_errors_are_typed():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    for handler, kind in ((slow, "timeout"), (refused, "transport")):
        with pytest.raises(NhtsaError) as exc:
            NhtsaClient(transport=httpx.MockTransport(handler)).recalls(CIVIC)
        assert exc.value.kind == kind
    assert NhtsaClient().timeout == 8.0


def test_recorded_response_serves_only_its_own_vehicle():
    nhtsa = NhtsaClient.from_recorded(FIXTURE)
    assert len(nhtsa.recalls(CIVIC)) == 4
    with pytest.raises(NhtsaError, match="no recorded response"):
        nhtsa.recalls(Vehicle(2019, "Honda", "Accord", 10))


def test_fuel_pump_line_matches_the_newest_fuel_pump_campaign():
    recalls = client(Recorder()).recalls(CIVIC)
    found = match_recall(line("Replace fuel pump module $1,089.00"), recalls)
    assert found is not None and found.campaign == "23V858000"


@pytest.mark.parametrize(
    "text,campaign",
    [
        ("Airbag warning light diagnosis $189.00", "26V332000"),
        ("Front passenger air bag sensor $220.00", "26V332000"),
        ("Tech diagnostic ... $189.00 Required before any repair", None),
        ("Check engine light diagnosis $159.00", None),
        ("Fuel injection cleaning $149.00", None),
    ],
)
def test_keyword_overlap_matching(text, campaign):
    recalls = client(Recorder()).recalls(CIVIC)
    found = match_recall(line(text), recalls)
    assert (found.campaign if found else None) == campaign


def test_oxygen_sensor_spellings_match_each_other():
    o2 = Recall(
        campaign="99V000000",
        component="ENGINE AND ENGINE COOLING:EXHAUST SYSTEM:OXYGEN SENSOR",
        summary="The oxygen sensor may fail.",
        consequence="",
        remedy="Dealers will replace the sensor.",
        report_date=date(2024, 1, 1),
    )
    for text in ("Replace upstream O2 sensor $412.00", "Replace upstream 02 sensor $412.00"):
        assert match_recall(line(text), [o2]) is o2
    assert match_recall(line("Cabin air filter $79.95"), [o2]) is None
