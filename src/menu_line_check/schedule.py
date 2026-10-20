"""Is a scheduled service due at this odometer reading? (ILLUSTRATIVE schedule, see the YAML.)

Without the car's service history the check assumes earlier services were done on schedule. A
mileage service is ``due`` when the odometer is within ``window_mi`` (default 1,500 mi) of a
multiple of its interval, before or after it; otherwise ``not_due`` with the next multiple.
Time-based services (brake fluid) and services not in the schedule are ``unknown``.
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
DEFAULT_WINDOW_MI = 1500
ILLUSTRATIVE = "illustrative schedule"

Status = Literal["due", "not_due", "unknown"]


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
    """The multiple the service is due at, when it is due."""
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
) -> ScheduleCheck:
    """Due / not due / unknown for one menu line at the vehicle's odometer reading."""
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
