"""Append-only JSONL decision ledger.

One line per gate decision: UTC timestamp, question names, answers, confidences, the
action taken, client and server latency and a SHA-256 of the state. With ``redact=True``
(the default) the raw state is never written, only its hash, so the ledger can be shared
or attached to an article. With ``redact=False`` the raw state is stored too.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .client import Answer, NoulAnswer


def state_hash(state: Any) -> str:
    """Stable SHA-256 of a state (string, or JSON-serialisable structure)."""
    if isinstance(state, str):
        raw = state
    else:
        raw = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    data = asdict(answer)
    if isinstance(answer, NoulAnswer):
        data["confidence"] = answer.confidence
    return data


def action_to_dict(action: Any) -> dict[str, Any]:
    """Serialise a Strands action: its type plus the plain-text fields it carries."""
    out: dict[str, Any] = {"type": getattr(action, "type", type(action).__name__.lower())}
    for key in ("reason", "feedback", "prompt"):
        value = getattr(action, key, None)
        if value:
            out[key] = value
    return out


class Ledger:
    def __init__(self, path: str | Path, *, redact: bool = True) -> None:
        self.path = Path(path)
        self.redact = redact
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        gate: str,
        stage: str,
        state: Any,
        answers: Mapping[str, Answer] | None,
        action: Any,
        routes: Mapping[str, str] | None = None,
        tool: str | None = None,
        client_ms: float | None = None,
        server_ms: float | None = None,
        unavailable: str | None = None,
        coerced_from: str | None = None,
    ) -> dict[str, Any]:
        answers = answers or {}
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "gate": gate,
            "stage": stage,
            "tool": tool,
            "questions": list(answers),
            "answers": {name: answer_to_dict(a) for name, a in answers.items()},
            "confidences": {name: answer_to_dict(a)["confidence"] for name, a in answers.items()},
            "routes": dict(routes or {}),
            "action": action_to_dict(action),
            "client_ms": None if client_ms is None else round(client_ms, 2),
            "server_ms": None if server_ms is None else round(server_ms, 2),
            "state_sha256": state_hash(state),
            "unavailable": unavailable,
            "coerced_from": coerced_from,
        }
        if not self.redact:
            entry["state"] = state
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return entry


def read_ledger(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
