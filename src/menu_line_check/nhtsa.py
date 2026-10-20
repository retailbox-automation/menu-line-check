"""NHTSA recalls lookup (public API, no key) and matching a menu line to a recall campaign.

``GET https://api.nhtsa.gov/recalls/recallsByVehicle?make=&model=&modelYear=`` lists the recall
campaigns that cover a make, model and model year. It does not say whether a campaign is still
open for one particular car: that needs the VIN, which the service advisor can check. So the
advisor question is "is this covered by recall N?", not "this is covered".
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

from .models import MenuLine, Vehicle

DEFAULT_BASE_URL = "https://api.nhtsa.gov"
RECALLS_PATH = "/recalls/recallsByVehicle"
DEFAULT_TIMEOUT_S = 8.0


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
    def from_recorded(cls, path: str | Path) -> NhtsaClient:
        """Serve one recorded response (for its own make, model and year only); no network."""
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        first = (body.get("results") or [{}])[0]
        key = (str(first.get("Make", "")).upper(), str(first.get("Model", "")).upper())
        year = str(first.get("ModelYear", ""))

        def handle(request: httpx.Request) -> httpx.Response:
            p = request.url.params
            asked = (p.get("make", "").upper(), p.get("model", "").upper())
            if asked == key and p.get("modelYear") == year:
                return httpx.Response(200, json=body)
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
