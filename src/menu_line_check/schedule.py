"""Is a menu line a service the schedule lists at this odometer reading?

Two sources, tried in this order:

1. A MAKER schedule for this car (``data/maker_schedules.yaml``: make, model and year, with the
   maker's publication, URL and page). When the odometer is within ``window_mi`` of one of its
   services (the 20,000-mile service of the real sheet's car), a line is ``due`` when the
   maker lists it there, ``bundled`` when it is a listed item sold together with an extra the
   maker does not list ("Cabin Filter, Incl. Sanitizer"), ``unclear`` when it names a listed
   item but also words the schedule does not have ("Premium Engine Oil Conditioner"), and
   ``not_listed`` otherwise.
   Only this source may lead to a decline.
2. Otherwise the ILLUSTRATIVE generic table (``data/maintenance_schedule.yaml``), which is no
   maker's schedule. Without the car's service history the check assumes earlier services were
   done on schedule. A mileage service is ``due`` when the odometer is within ``window_mi``
   (default 1,500 mi) of a multiple of its interval, before or after it; otherwise ``not_due``
   with the next multiple. Time-based services (brake fluid) and services not in the table are
   ``unknown``. The tool turns everything but ``due`` into a question, never a decline.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml

from .models import MenuLine, Vehicle

SCHEDULE_PATH = Path(__file__).parent / "data" / "maintenance_schedule.yaml"
MAKER_SCHEDULES_PATH = Path(__file__).parent / "data" / "maker_schedules.yaml"
DEFAULT_WINDOW_MI = 1500
ILLUSTRATIVE = "illustrative schedule"

Status = Literal["due", "not_due", "unknown", "not_listed", "bundled", "unclear"]
ScheduleSource = Literal["maker", "illustrative"]


@dataclass(frozen=True)
class ScheduledService:
    service: str
    match: tuple[str, ...]
    every_mi: int | None = None
    every_months: int | None = None
    every_visit: bool = False


@dataclass(frozen=True)
class ScheduleCheck:
    status: Status
    service: str | None
    every_mi: int | None
    every_months: int | None
    every_visit: bool
    last_due_mi: int | None
    """The last multiple of the interval at or below the odometer (0 = none yet)."""
    next_due_mi: int | None
    """The next multiple of the interval above the odometer."""
    due_at_mi: int | None
    """The multiple the service is due at, when it is due (maker: the service's mileage)."""
    note: str
    source: ScheduleSource = "illustrative"
    """Which schedule answered: the maker's for this car, or the illustrative table."""
    name: str | None = None
    """Maker schedule: the listed item in plain words ("a cabin air filter")."""
    extra: str | None = None
    """Maker schedule, ``bundled`` / ``unclear``: what the line adds to the listed item."""
    reference: str | None = None
    """Maker schedule: publication, page and URL."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------- maker schedules


@dataclass(frozen=True)
class MakerItem:
    item: str
    """As the maker prints it."""
    name: str
    """In plain words for a question to the advisor."""
    match: tuple[str, ...]


@dataclass(frozen=True)
class MakerService:
    """One service of a maker schedule (e.g. 24 months or 20,000 miles)."""

    mileage: int
    months: int | None
    items: tuple[MakerItem, ...]
    inspection_match: tuple[str, ...] = ()
    inspection_name: str = "the inspections"


@dataclass(frozen=True)
class MakerSchedule:
    make: str
    model: str
    year: int
    source: str
    url: str
    page: str
    services: tuple[MakerService, ...]

    @property
    def reference(self) -> str:
        return f"{self.source}, {self.page}: {self.url}"

    def fits(self, vehicle: Vehicle) -> bool:
        return (
            vehicle.make.upper() == self.make.upper()
            and " ".join(vehicle.model.upper().split()) == " ".join(self.model.upper().split())
            and vehicle.year == self.year
        )


@cache
def _load_maker(path: Path) -> tuple[MakerSchedule, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for raw in data["schedules"]:
        services = tuple(
            MakerService(
                mileage=int(s["mileage"]),
                months=s.get("months"),
                items=tuple(
                    MakerItem(
                        str(i["item"]), str(i["name"]), tuple(str(m).lower() for m in i["match"])
                    )
                    for i in s["items"]
                ),
                inspection_match=tuple(str(m).lower() for m in s.get("inspection_match", ())),
                inspection_name=str(s.get("inspection_name", "the inspections")),
            )
            for s in raw["intervals"]
        )
        out.append(
            MakerSchedule(
                str(raw["make"]),
                str(raw["model"]),
                int(raw["year"]),
                str(raw["source"]),
                str(raw["url"]),
                str(raw["page"]),
                services,
            )
        )
    return tuple(out)


def load_maker_schedules(path: str | Path | None = None) -> tuple[MakerSchedule, ...]:
    return _load_maker(Path(path or MAKER_SCHEDULES_PATH))


def maker_service(
    vehicle: Vehicle, *, window_mi: int = DEFAULT_WINDOW_MI, path: str | Path | None = None
) -> tuple[MakerSchedule, MakerService] | None:
    """The maker schedule of this car and its service nearest the odometer, within the window."""
    for schedule in load_maker_schedules(path):
        if not schedule.fits(vehicle):
            continue
        near = [s for s in schedule.services if abs(s.mileage - vehicle.odometer_mi) <= window_mi]
        if near:
            return schedule, min(near, key=lambda s: abs(s.mileage - vehicle.odometer_mi))
    return None


# Words that say how a service is done, not which service it is.
_PLAIN = frozenset(
    "a an and or of the to for in on at by perform replace replaced install installed change "
    "changed rotate rotation service services check inspect inspection your vehicle new full "
    "synthetic conventional blend standard genuine factory front rear all four".split()
)
# "Cabin Filter, Incl. Sanitizer", "Oil change with additive", "Filter plus treatment"
_BUNDLE = re.compile(r"\b(?:incl\b\.?|including|includes|with|plus)\s|\s\+\s", re.IGNORECASE)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3 and w not in _PLAIN}


def _vocabulary(service: MakerService) -> set[str]:
    words: set[str] = set()
    for item in service.items:
        words |= _words(item.item) | _words(item.name) | _words(" ".join(item.match))
        words |= {w[:-1] for w in words if w.endswith("s")}  # tires -> tire
    return words | _words(" ".join(service.inspection_match))


def _longest(text: str, phrases: tuple[str, ...]) -> int:
    hits = [len(p) for p in phrases if re.search(rf"\b{re.escape(p)}", text)]
    return max(hits, default=0)


def check_maker_schedule(
    line: MenuLine,
    vehicle: Vehicle,
    *,
    window_mi: int = DEFAULT_WINDOW_MI,
    path: str | Path | None = None,
) -> ScheduleCheck | None:
    """The maker-schedule check, or None when there is no maker schedule for this car here."""
    found = maker_service(vehicle, window_mi=window_mi, path=path)
    if found is None:
        return None
    schedule, service = found
    at = service.mileage
    marker = _BUNDLE.search(line.name)
    head = line.name[: marker.start()] if marker else line.name
    extra = line.name[marker.end() :].strip() if marker else ""
    text = head.lower()

    best: tuple[int, str, str] | None = None
    for item in service.items:
        n = _longest(text, item.match)
        if n and (best is None or n > best[0]):
            best = (n, item.item, item.name)
    n = _longest(text, service.inspection_match)
    if n and (best is None or n > best[0]):
        best = (n, "inspect and check items of this service", service.inspection_name)

    def result(status: Status, note: str, item=None, name=None, added=None) -> ScheduleCheck:
        return ScheduleCheck(
            status,
            item,
            None,
            service.months,
            False,
            None,
            None,
            at if item else None,
            note,
            source="maker",
            name=name,
            extra=added,
            reference=schedule.reference,
        )

    where = f"the maker's schedule at {at:,} mi"
    if best is None:
        return result("not_listed", f"not in {where}")
    _, item, name = best
    vocabulary = _vocabulary(service)
    unknown = [w for w in re.findall(r"[a-z]+", text) if _words(w) and w not in vocabulary]
    if unknown:
        added = " ".join(dict.fromkeys(unknown))
        return result(
            "unclear", f'{where} lists {name}; this line also says "{added}"', item, name, added
        )
    if extra and _words(extra) - vocabulary:
        added = re.split(r"[(,;]", extra)[0].strip().lower() or extra.lower()
        return result(
            "bundled",
            f"in {where} ({name}), bundled with {added}, which the schedule does not list",
            item,
            name,
            added,
        )
    return result("due", f"in {where}", item, name)


@cache
def _load(path: Path) -> tuple[ScheduledService, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(
        ScheduledService(
            service=str(s["service"]),
            match=tuple(str(m).lower() for m in s["match"]),
            every_mi=s.get("every_mi"),
            every_months=s.get("every_months"),
            every_visit=bool(s.get("every_visit", False)),
        )
        for s in data["services"]
    )


def load_schedule(path: str | Path = SCHEDULE_PATH) -> tuple[ScheduledService, ...]:
    return _load(Path(path))


def find_service(
    name: str, schedule: tuple[ScheduledService, ...] | None = None
) -> ScheduledService | None:
    """The scheduled service whose longest match phrase starts a word in ``name``."""
    text = name.lower()
    best: tuple[int, ScheduledService] | None = None
    for service in schedule or load_schedule():
        for phrase in service.match:
            if re.search(rf"\b{re.escape(phrase)}", text) and (
                best is None or len(phrase) > best[0]
            ):
                best = (len(phrase), service)
    return best[1] if best else None


def check_schedule(
    line: MenuLine,
    vehicle: Vehicle,
    *,
    window_mi: int = DEFAULT_WINDOW_MI,
    schedule: tuple[ScheduledService, ...] | None = None,
    maker_path: str | Path | None = None,
) -> ScheduleCheck:
    """The maker's schedule for this car when there is one at this mileage, else the
    illustrative table (``schedule`` replaces the table, e.g. in tests)."""
    if schedule is None:
        maker = check_maker_schedule(line, vehicle, window_mi=window_mi, path=maker_path)
        if maker is not None:
            return maker
    return check_illustrative(line, vehicle, window_mi=window_mi, schedule=schedule)


def check_illustrative(
    line: MenuLine,
    vehicle: Vehicle,
    *,
    window_mi: int = DEFAULT_WINDOW_MI,
    schedule: tuple[ScheduledService, ...] | None = None,
) -> ScheduleCheck:
    """Due / not due / unknown for one menu line in the ILLUSTRATIVE table."""
    service = find_service(line.name, schedule)
    if service is None:
        return ScheduleCheck(
            "unknown", None, None, None, False, None, None, None, f"not in the {ILLUSTRATIVE}"
        )

    def result(status: Status, last=None, nxt=None, due_at=None, note="") -> ScheduleCheck:
        return ScheduleCheck(
            status,
            service.service,
            service.every_mi,
            service.every_months,
            service.every_visit,
            last,
            nxt,
            due_at,
            note,
        )

    if service.every_visit:
        return result("due", note=f"checked at every visit ({ILLUSTRATIVE})")
    if service.every_mi is None:
        months = service.every_months
        return result(
            "unknown",
            note=f"time-based (every {months} months); the date of the last service is needed",
        )

    every, odometer = service.every_mi, vehicle.odometer_mi
    last = (odometer // every) * every
    nxt = last + every
    if last > 0 and odometer - last <= window_mi:
        due_at = last
    elif nxt - odometer <= window_mi:
        due_at = nxt
    else:
        return result(
            "not_due",
            last,
            nxt,
            None,
            f"not due until {nxt:,} mi (every {every:,} mi, {ILLUSTRATIVE})",
        )
    return result(
        "due", last, nxt, due_at, f"due at {due_at:,} mi (every {every:,} mi, {ILLUSTRATIVE})"
    )
