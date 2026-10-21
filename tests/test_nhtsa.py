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
from menu_line_check.nhtsa import (
    VEHICLE_RECALLS_QUESTION,
    NhtsaClient,
    NhtsaError,
    Recall,
    match_recall,
    vehicle_recalls,
)
from menu_line_check.ocr import parse_line_texts

from .menu_fakes import NHTSA_FAMILY, NHTSA_MODEL_EMPTY, NHTSA_SHEET_RECORDED

FIXTURE = Path(__file__).parent / "fixtures" / "nhtsa_recalls_civic_2019.json"
CIVIC = Vehicle(2019, "Honda", "Civic", 31200)
SHEET_CAR = Vehicle.parse("2024 Lexus RX 350 20000")


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


def test_http_400_with_an_empty_result_means_no_campaigns_and_is_cached(tmp_path):
    empty = {"Count": 0, "Message": "Results returned successfully", "results": []}
    rec = Recorder(status=400, body=empty)
    assert client(rec, cache_dir=tmp_path).recalls(CIVIC) == []
    assert client(rec, cache_dir=tmp_path).recalls(CIVIC) == [] and len(rec.requests) == 1
    for status, body in ((400, {"Count": 0, "Message": "bad model"}), (500, empty)):
        with pytest.raises(NhtsaError) as exc:
            client(Recorder(status=status, body=body)).recalls(CIVIC)
        assert exc.value.kind == "http"


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


# --------------------------------------------------------------------------- the recalls card


def test_recorded_responses_serve_several_vehicles_and_an_empty_one_needs_its_query():
    nhtsa = NhtsaClient.from_recorded(FIXTURE, *NHTSA_SHEET_RECORDED)
    assert len(nhtsa.recalls(CIVIC)) == 4
    assert nhtsa.recalls(SHEET_CAR) == []
    assert len(nhtsa.recalls(Vehicle(2024, "LEXUS", "rx", 1))) == 4
    with pytest.raises(NhtsaError, match="no recorded response"):
        nhtsa.recalls(Vehicle(2025, "Lexus", "RX 350", 10))
    with pytest.raises(ValueError, match="needs its query"):
        NhtsaClient.from_recorded(NHTSA_MODEL_EMPTY)
    with pytest.raises(ValueError):
        NhtsaClient.from_recorded()


def test_model_with_no_campaigns_falls_back_to_its_family_and_keeps_those_naming_it():
    rec = Recorder(body=None)
    bodies = {
        "RX 350": json.loads(NHTSA_MODEL_EMPTY.read_text()),
        "RX": json.loads(NHTSA_FAMILY.read_text()),
    }

    def handle(request):
        rec.requests.append(request)
        return httpx.Response(200, json=bodies[request.url.params["model"]])

    card = vehicle_recalls(SHEET_CAR, NhtsaClient(transport=httpx.MockTransport(handle)))
    assert [r.url.params["model"] for r in rec.requests] == ["RX 350", "RX"]
    assert card.family_fallback and card.family == "RX" and card.queried_model == "RX"
    assert card.matched == ("24V482000", "25V059000", "26V162000")
    assert card.unmatched == ("25V744000",)
    assert [r.campaign for r in card.campaigns] == list(card.matched)
    assert card.advisor_question == VEHICLE_RECALLS_QUESTION == "Are there open recalls on my VIN?"
    assert card.vehicle == "2024 Lexus RX 350" and card.error is None
    data = card.to_dict()
    assert data["campaigns"][0] == {
        "campaign": "24V482000",
        "component": "SEATS:FRONT ASSEMBLY:HEAD RESTRAINT",
        "report_date": "2024-06-27",
        "summary": "The head restraints on the driver and front passenger seats may be "
        "inadvertently removed without pressing the lock release button.",
    }
    assert data["campaigns"][2]["summary"] == (
        "The rearview camera image may not display when the vehicle is placed in reverse."
    )
    assert 'no campaign for model "RX 350"' in data["note"] and "3 name the RX 350" in data["note"]


def test_one_line_summary_skips_the_opening_and_the_report_pointer():
    (camera,) = [
        r
        for r in NhtsaClient.from_recorded(NHTSA_FAMILY).recalls(Vehicle(2024, "Lexus", "RX", 1))
        if r.campaign == "25V744000"
    ]
    assert camera.one_line.startswith("A software error may cause the rearview camera")


def test_no_fallback_when_the_model_has_campaigns_or_is_one_word():
    civic = vehicle_recalls(CIVIC, NhtsaClient.from_recorded(FIXTURE))
    assert not civic.family_fallback and len(civic.campaigns) == 4
    assert civic.queried_model == "Civic" and civic.unmatched == ()
    rec = Recorder(body={"Count": 0, "Message": "ok", "results": []})
    card = vehicle_recalls(Vehicle(2024, "Acme", "Zed", 1), client(rec))
    assert len(rec.requests) == 1 and card.campaigns == () and not card.family_fallback
    assert "0 recall campaigns" in card.note


def test_a_failed_lookup_is_on_the_card_not_raised():
    card = vehicle_recalls(SHEET_CAR, client(Recorder(status=500)))
    assert card.error and "HTTP 500" in card.error and card.campaigns == ()
    assert card.advisor_question == VEHICLE_RECALLS_QUESTION
    assert "ask the advisor to check the VIN" in card.note
    # the model itself answered nothing, then the family lookup failed
    nhtsa = NhtsaClient.from_recorded(*NHTSA_SHEET_RECORDED[1:])
    card = vehicle_recalls(SHEET_CAR, nhtsa)
    assert card.family_fallback and card.error and "no recorded response" in card.error
