"""NHTSA recalls lookup (public API, no key) and matching a menu line to a recall campaign.

``GET https://api.nhtsa.gov/recalls/recallsByVehicle?make=&model=&modelYear=`` lists the recall
campaigns that cover a make, model and model year. It does not say whether a campaign is still
open for one particular car: that needs the VIN, which the service advisor can check. So the
advisor question is "is this covered by recall N?", not "this is covered".

:func:`vehicle_recalls` builds the recalls card for the whole car. The API matches the model
name exactly: for a 2024 Lexus "RX 350" it returns no campaign at all, while "RX" returns four,
three of which name the 2024 RX 350 in their summary (live, 2026-10-21). So when a model of two
or more words gets nothing, the card asks again for the model family (the first word) and keeps
the campaigns whose summary names the model.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

from .models import MenuLine, Vehicle

DEFAULT_BASE_URL = "https://api.nhtsa.gov"
RECALLS_PATH = "/recalls/recallsByVehicle"
DEFAULT_TIMEOUT_S = 8.0
VEHICLE_RECALLS_QUESTION = "Are there open recalls on my VIN?"
"""The advisor question on the recalls card: NHTSA lists campaigns per model, not per car."""

Recording = str | Path | tuple[str | Path, str, str, int | str]
"""A recorded response: a path, or (path, make, model, year) for a response with no results."""


class NhtsaError(Exception):
    """The recalls lookup failed.

    ``kind`` is one of ``timeout``, ``transport``, ``http``, ``bad_response``.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"NHTSA lookup failed ({kind}): {message}")
        self.kind = kind


@dataclass(frozen=True)
class Recall:
    campaign: str
    component: str
    summary: str
    consequence: str
    remedy: str
    report_date: date | None
    make: str = ""
    model: str = ""
    model_year: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["report_date"] = self.report_date.isoformat() if self.report_date else None
        return data

    @property
    def one_line(self) -> str:
        """The defect in one sentence: the first summary sentence that is not the
        "<maker> is recalling certain ..." opening or a "please see the report" pointer."""
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", self.summary) if x.strip()]
        for sentence in sentences:
            low = sentence.lower()
            if "is recalling" not in low and not low.startswith("please see"):
                return sentence
        return sentences[0] if sentences else ""

    def card_entry(self) -> dict[str, Any]:
        return {
            "campaign": self.campaign,
            "component": self.component,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "summary": self.one_line,
        }


def _date(value: Any) -> date | None:
    # NHTSA sends ReportReceivedDate as DD/MM/YYYY
    try:
        return datetime.strptime(str(value), "%d/%m/%Y").date()
    except ValueError:
        return None


def parse_recalls(body: Any) -> list[Recall]:
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise NhtsaError("bad_response", "no 'results' list in the response")
    out = []
    for raw in body["results"]:
        if not isinstance(raw, dict) or not raw.get("NHTSACampaignNumber"):
            raise NhtsaError("bad_response", f"recall without a campaign number: {raw!r:.200}")
        out.append(
            Recall(
                campaign=str(raw["NHTSACampaignNumber"]),
                component=str(raw.get("Component") or ""),
                summary=str(raw.get("Summary") or ""),
                consequence=str(raw.get("Consequence") or ""),
                remedy=str(raw.get("Remedy") or ""),
                report_date=_date(raw.get("ReportReceivedDate")),
                make=str(raw.get("Make") or ""),
                model=str(raw.get("Model") or ""),
                model_year=str(raw.get("ModelYear") or ""),
            )
        )
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class NhtsaClient:
    """Recalls by make, model and year, with an optional disk cache.

    ``transport`` is for tests and recorded replays (``httpx.MockTransport``). The disk cache is
    keyed by (make, model, year), case-insensitive; entries older than ``max_age_days`` are
    fetched again (None keeps them forever).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: str | Path | None = None,
        transport: httpx.BaseTransport | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_age_days: float | None = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.timeout = timeout
        self.max_age_days = max_age_days
        self._transport = transport
        self._memory: dict[tuple[str, str, int], list[Recall]] = {}

    @classmethod
    def from_recorded(cls, *recordings: Recording) -> NhtsaClient:
        """Serve recorded responses, each for its own make, model and year only; no network.

        A path is keyed by the make, model and year of its first result. A response with no
        results (``Count`` 0) says nothing about what was asked, so it is given as
        ``(path, make, model, year)``. Anything else asked gets HTTP 404.
        """
        if not recordings:
            raise ValueError("from_recorded needs at least one recorded response")
        served: dict[tuple[str, str, str], Any] = {}
        for recording in recordings:
            if isinstance(recording, tuple):
                path, make, model, year = recording
                key = (str(make).upper(), str(model).upper(), str(year))
                body = json.loads(Path(path).read_text(encoding="utf-8"))
            else:
                body = json.loads(Path(recording).read_text(encoding="utf-8"))
                results = body.get("results") or []
                if not results:
                    raise ValueError(
                        f"{recording}: a response with no results needs its query: "
                        "(path, make, model, year)"
                    )
                first = results[0]
                key = (
                    str(first.get("Make", "")).upper(),
                    str(first.get("Model", "")).upper(),
                    str(first.get("ModelYear", "")),
                )
            served[key] = body

        def handle(request: httpx.Request) -> httpx.Response:
            p = request.url.params
            asked = (p.get("make", "").upper(), p.get("model", "").upper(), p.get("modelYear"))
            if asked in served:
                return httpx.Response(200, json=served[asked])
            return httpx.Response(404, text=f"no recorded response for {p}")

        return cls(transport=httpx.MockTransport(handle))

    def _cache_path(self, vehicle: Vehicle) -> Path | None:
        if self.cache_dir is None:
            return None
        name = f"recalls_{_slug(vehicle.make)}_{_slug(vehicle.model)}_{vehicle.year}.json"
        return self.cache_dir / name

    def _fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self.max_age_days is None:
            return True
        return time.time() - path.stat().st_mtime < self.max_age_days * 86400

    def _fetch(self, vehicle: Vehicle) -> Any:
        params = {"make": vehicle.make, "model": vehicle.model, "modelYear": str(vehicle.year)}
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as http:
                response = http.get(self.base_url + RECALLS_PATH, params=params)
        except httpx.TimeoutException as exc:
            raise NhtsaError("timeout", repr(exc)) from exc
        except httpx.HTTPError as exc:
            raise NhtsaError("transport", repr(exc)) from exc
        if response.status_code >= 400:
            raise NhtsaError("http", f"HTTP {response.status_code}: {response.text[:200]}")
        try:
            return response.json()
        except ValueError as exc:
            raise NhtsaError("bad_response", f"body is not JSON: {exc!r}") from exc

    def recalls(self, vehicle: Vehicle) -> list[Recall]:
        key = (vehicle.make.upper(), vehicle.model.upper(), vehicle.year)
        if key in self._memory:
            return self._memory[key]
        path = self._cache_path(vehicle)
        if path is not None and self._fresh(path):
            recalls = parse_recalls(json.loads(path.read_text(encoding="utf-8")))
        else:
            body = self._fetch(vehicle)
            recalls = parse_recalls(body)  # validated before it is cached
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(body), encoding="utf-8")
        self._memory[key] = recalls
        return recalls


# ------------------------------------------------------------------------- the recalls card


@dataclass(frozen=True)
class VehicleRecalls:
    """Recall campaigns NHTSA lists for the car's make, model and year (the recalls card).

    These are campaigns for the model, not open recalls on this car: whether one is open needs
    the VIN, hence ``advisor_question``.
    """

    year: int
    make: str
    model: str
    campaigns: tuple[Recall, ...]
    queried_model: str
    """The model name the kept campaigns came from: the model, or its family."""
    family_fallback: bool = False
    family: str | None = None
    matched: tuple[str, ...] = ()
    """Family campaigns whose summary names the model (kept)."""
    unmatched: tuple[str, ...] = ()
    """Family campaigns whose summary does not name the model (not kept)."""
    error: str | None = None
    advisor_question: str = VEHICLE_RECALLS_QUESTION

    @property
    def vehicle(self) -> str:
        return f"{self.year} {self.make} {self.model}"

    @property
    def note(self) -> str:
        if self.error:
            return f"Recall lookup failed ({self.error}); ask the advisor to check the VIN."
        n = len(self.campaigns)
        listed = f"{n} recall campaign{'s' if n != 1 else ''}"
        if not self.family_fallback:
            return f"NHTSA lists {listed} for the {self.vehicle} (by model, not by VIN)."
        return (
            f'NHTSA returns no campaign for model "{self.model}"; model family "{self.family}" '
            f"has {n + len(self.unmatched)}, of which {n} name the {self.model} in their summary "
            "(by model, not by VIN)."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle": self.vehicle,
            "queried_model": self.queried_model,
            "campaigns": [r.card_entry() for r in self.campaigns],
            "family_fallback": self.family_fallback,
            "family": self.family,
            "matched": list(self.matched),
            "unmatched": list(self.unmatched),
            "error": self.error,
            "note": self.note,
            "advisor_question": self.advisor_question,
        }


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).upper()


def vehicle_recalls(vehicle: Vehicle, client: NhtsaClient) -> VehicleRecalls:
    """The recalls card: campaigns for the car, with the model-family fallback.

    Ask for make + model + year. If that returns nothing and the model has more than one word,
    ask for the family (the first word, "RX" for "RX 350") and keep the campaigns whose summary
    names the model (upper case, spaces removed: "RX350" in the summary). The agent never says
    a recall is open: the card's question for the advisor is about the VIN.
    """
    card = {"year": vehicle.year, "make": vehicle.make, "model": vehicle.model}
    try:
        recalls = client.recalls(vehicle)
    except NhtsaError as exc:
        return VehicleRecalls(**card, campaigns=(), queried_model=vehicle.model, error=str(exc))
    words = vehicle.model.split()
    if recalls or len(words) < 2:
        return VehicleRecalls(**card, campaigns=tuple(recalls), queried_model=vehicle.model)
    family = words[0]
    try:
        family_recalls = client.recalls(replace(vehicle, model=family))
    except NhtsaError as exc:
        return VehicleRecalls(
            **card,
            campaigns=(),
            queried_model=family,
            family_fallback=True,
            family=family,
            error=str(exc),
        )
    name = _compact(vehicle.model)
    kept = tuple(r for r in family_recalls if name in _compact(r.summary))
    return VehicleRecalls(
        **card,
        campaigns=kept,
        queried_model=family,
        family_fallback=True,
        family=family,
        matched=tuple(r.campaign for r in kept),
        unmatched=tuple(r.campaign for r in family_recalls if r not in kept),
    )


# ------------------------------------------------------------------------------------ matching

_PHRASES = [
    (re.compile(r"\bair\s*bags?\b"), "airbag"),
    (re.compile(r"\b(?:o2|02|oxygen)\s+sensors?\b"), "oxygen_sensor"),
    (re.compile(r"\bfuel\s+pumps?\b"), "fuel_pump"),
    (re.compile(r"\bseat\s*belts?\b"), "seatbelt"),
]
_STOP = frozenset(
    "a an and or of the to for in on with any new tech labor part parts assembly "
    "replace replacement repair service install installed check inspect inspection "
    "diagnosis diagnostic diagnose front rear left right upstream downstream may fail will".split()
)
# Category words of NHTSA component names: they do not identify a part on their own.
_GENERIC = frozenset(
    "system gasoline diesel delivery fuel engine cooling electrical equipment sensor unknown "
    "other power train".split()
)


def _tokens(text: str) -> set[str]:
    text = text.lower()
    for pattern, token in _PHRASES:
        text = pattern.sub(token, text)
    out = set()
    for word in re.findall(r"[a-z0-9_]+", text):
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if word not in _STOP:
            out.add(word)
    return out


def match_recall(line: MenuLine, recalls: Iterable[Recall]) -> Recall | None:
    """The recall whose component shares a part word with the line name, best overlap first.

    A match needs at least one shared non-generic word with the recall's component ("fuel pump",
    "air bags", "oxygen sensor"). Ties go to more shared words with the summary and remedy, then
    to the newest report (expansions of a campaign are filed later).
    """
    key = _tokens(line.name)
    best: tuple[tuple[int, int, date], Recall] | None = None
    for recall in recalls:
        component = len((key & _tokens(recall.component)) - _GENERIC)
        if component == 0:
            continue
        text = len(key & _tokens(f"{recall.summary} {recall.remedy}"))
        rank = (component, text, recall.report_date or date.min)
        if best is None or rank > best[0]:
            best = (rank, recall)
    return best[1] if best else None
