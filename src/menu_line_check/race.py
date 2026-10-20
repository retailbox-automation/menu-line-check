"""``menu-race``: replay labelled menu lines through Decider, a keyword baseline and an LLM picker.

Lines file (JSONL): ``{"id", "text", "price", "tool", "safety", "readable"}`` per line, optionally
``"recorded"`` (Decider answers recorded in the spike). The default file is the 103 readable lines
of the spike of 2026-10-20 (``data/race_lines.jsonl``, built by ``scripts/build_race_lines.py``).

Contestants:

* Decider 2B over HTTP (``--decider-url``), or its answers recorded in the spike
  (``--replay-recorded``, no server needed). Same three questions as the pipeline, OCR text.
* The keyword rules (``baseline.keyword_tool``), no model.
* The Strands LLM picker on Bedrock, alone (no Decider gate), only with ``--allow-bedrock``.

Output: a markdown table with tool-choice accuracy, accuracy and coverage at confidence
thresholds, safety lines missed, p50/p90 latency per line and cost per 1,000 lines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decider_kit import ChoiceAnswer, DeciderClient, DeciderError, DeciderUnavailable, NoulAnswer
from decider_kit.bench import percentile
from menu_line_tool_selection import load_pattern_questions, ocr_state

from .baseline import keyword_tool
from .models import MenuLine
from .ocr import parse_line_texts

RACE_LINES_PATH = Path(__file__).parent / "data" / "race_lines.jsonl"
DEFAULT_URL = "http://127.0.0.1:8099"
THRESHOLDS = (0.5, 0.7, 0.8, 0.9)

# USD per 1M tokens, us-east-1 on-demand — fill from https://aws.amazon.com/bedrock/pricing/
# before publishing. While they are None the table prints "price not set".
PRICE_IN_PER_M: float | None = None
PRICE_OUT_PER_M: float | None = None


@dataclass(frozen=True)
class RaceLine:
    id: str
    text: str
    price: float | None
    tool: str
    safety: bool
    readable: bool
    recorded: dict[str, Any] | None = None


@dataclass(frozen=True)
class Pick:
    """One contestant's answer for one line."""

    tool: str | None
    confidence: float | None = None
    safety_flag: bool | None = None
    """The contestant's own safety yes/no, when it has one."""
    ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0


Contestant = tuple[str, list[Pick], str | None]
"""(name, picks, cost note). A None cost note means: price the tokens."""


def load_race_lines(path: str | Path = RACE_LINES_PATH) -> list[RaceLine]:
    lines = []
    with Path(path).open(encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                d = json.loads(raw)
                lines.append(
                    RaceLine(
                        id=str(d["id"]),
                        text=str(d["text"]),
                        price=d.get("price"),
                        tool=str(d["tool"]),
                        safety=bool(d["safety"]),
                        readable=bool(d.get("readable", True)),
                        recorded=d.get("recorded"),
                    )
                )
    return lines


def menu_line(line: RaceLine, index: int) -> MenuLine:
    parsed = parse_line_texts([line.text])[0]
    return MenuLine(
        index=index,
        text=line.text,
        name=parsed.name,
        price=parsed.price,
        bbox=None,
        price_text=parsed.price_text,
        description=parsed.description,
    )


# --------------------------------------------------------------------------------- contestants


def decider_picks(lines: Sequence[RaceLine], client: DeciderClient) -> list[Pick]:
    """Ask Decider the pattern's three OCR-text questions per line (a down line scores wrong)."""
    questions = load_pattern_questions("ocr_text")
    picks = []
    for line in lines:
        try:
            response = client.ask(ocr_state(line.text), questions)
        except DeciderUnavailable:
            picks.append(Pick(None))
            continue
        tool, safety = response.answers["tool"], response.answers["safety"]
        if not (isinstance(tool, ChoiceAnswer) and isinstance(safety, NoulAnswer)):
            picks.append(Pick(None))  # wrong answer types: scored as wrong
            continue
        picks.append(Pick(tool.choice, tool.confidence, safety.verdict, response.client_ms))
    return picks


def recorded_picks(lines: Sequence[RaceLine]) -> list[Pick]:
    """Decider answers as recorded in the spike (wall-clock ms as recorded there)."""
    picks = []
    for line in lines:
        rec = line.recorded
        if not rec:
            raise ValueError(f"line {line.id} has no recorded answers")
        picks.append(
            Pick(rec["tool"], float(rec["tool_conf"]), rec["p_safety"] >= 0.5, rec["client_ms"])
        )
    return picks


def keyword_picks(lines: Sequence[RaceLine]) -> list[Pick]:
    picks = []
    for line in lines:
        started = time.perf_counter()
        tool = keyword_tool(line.text)
        ms = (time.perf_counter() - started) * 1000.0
        picks.append(Pick(tool, 1.0, tool == "owner_decides", ms))
    return picks


def llm_picks(
    lines: Sequence[RaceLine], picker: Callable[[MenuLine, Any], tuple[str | None, str]]
) -> list[Pick]:
    """Ask an LLM picker per line. Token counts come from ``picker.last_usage`` if it has one."""
    picks = []
    for i, line in enumerate(lines, 1):
        started = time.perf_counter()
        try:
            tool, _ = picker(menu_line(line, i), None)
        except Exception as exc:  # scored as wrong and reported on stderr
            print(f"LLM picker failed on {line.id}: {exc!r}", file=sys.stderr)
            tool = None
        ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(picker, "last_usage", None) or {}
        picks.append(
            Pick(
                tool,
                None,
                None,
                ms,
                int(usage.get("inputTokens", 0)),
                int(usage.get("outputTokens", 0)),
            )
        )
    return picks


# ------------------------------------------------------------------------------------- scoring


def _ms(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}" if x < 10 else f"{x:.0f}"


def _cost(picks: Sequence[Pick], note: str | None) -> str:
    if note is not None:
        return note
    if PRICE_IN_PER_M is None or PRICE_OUT_PER_M is None or not picks:
        return "price not set"
    tokens_in = sum(p.input_tokens for p in picks)
    tokens_out = sum(p.output_tokens for p in picks)
    usd = (tokens_in * PRICE_IN_PER_M + tokens_out * PRICE_OUT_PER_M) / 1_000_000
    return f"${usd / len(picks) * 1000:.4f}"


def race_row(
    name: str,
    lines: Sequence[RaceLine],
    picks: Sequence[Pick],
    cost_note: str | None,
    thresholds: Sequence[float] = THRESHOLDS,
) -> list[str]:
    n = len(lines)
    pairs = list(zip(lines, picks, strict=True))
    correct = sum(p.tool == ln.tool for ln, p in pairs)
    cells = [name, str(n), f"{correct / n:.2f}" if n else "n/a"]
    for t in thresholds:
        if all(p.confidence is None for p in picks):
            cells.append("n/a")
            continue
        kept = [(ln, p) for ln, p in pairs if p.confidence is not None and p.confidence >= t]
        if kept:
            acc = sum(p.tool == ln.tool for ln, p in kept) / len(kept)
            cells.append(f"{acc:.2f} ({len(kept)}/{n})")
        else:
            cells.append(f"- (0/{n})")
    safety = [(ln, p) for ln, p in pairs if ln.safety]
    missed = sum(1 for _, p in safety if not (p.safety_flag or p.tool == "owner_decides"))
    cells.append(f"{missed}/{len(safety)}")
    ms = [p.ms for p in picks if p.ms is not None]
    cells += [_ms(percentile(ms, 0.5)), _ms(percentile(ms, 0.9)), _cost(picks, cost_note)]
    return cells


def race_table(
    lines: Sequence[RaceLine],
    contestants: Sequence[Contestant],
    thresholds: Sequence[float] = THRESHOLDS,
) -> str:
    head = ["contestant", "lines", "accuracy"]
    head += [f">= {t} acc (kept)" for t in thresholds]
    head += ["safety lines missed", "p50 ms", "p90 ms", "cost / 1,000 lines"]
    rows = [head, ["---"] * len(head)]
    rows += [race_row(name, lines, picks, note, thresholds) for name, picks, note in contestants]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


NOTES = """\
accuracy: share of lines where the chosen tool equals the label.
>= t acc (kept): accuracy on the lines whose confidence is at least t (lines kept / all lines).
safety lines missed: labelled safety lines where neither the safety answer was yes nor the tool
  was owner_decides (the pipeline's safety route).
latency: wall clock per line (for recorded answers: as recorded in the spike).
Decider runs on our own machine: no per-call API price. The keyword rules cost nothing."""


# ----------------------------------------------------------------------------------------- CLI


def make_decider_client(url: str) -> DeciderClient:
    return DeciderClient(url, timeout=30.0)


def _thresholds(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"thresholds must be numbers, got {text!r}") from exc
    if not values or not all(0.0 <= v <= 1.0 for v in values):
        raise argparse.ArgumentTypeError(f"thresholds must be between 0 and 1, got {text!r}")
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="menu-race", description=__doc__.split("\n\n")[0])
    p.add_argument("--lines", default=str(RACE_LINES_PATH), help="labelled lines, JSONL")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--decider-url", default=os.environ.get("STRANDS_DECIDER_URL", DEFAULT_URL))
    src.add_argument(
        "--replay-recorded",
        action="store_true",
        help="use the Decider answers recorded in the lines file instead of a server",
    )
    p.add_argument("--allow-bedrock", action="store_true", help="also race the LLM picker (paid)")
    p.add_argument("--limit", type=int, help="only the first N lines")
    p.add_argument("--thresholds", type=_thresholds, default=THRESHOLDS)
    p.add_argument("--out", help="also write the table to this markdown file")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lines = load_race_lines(args.lines)
    if args.limit:
        lines = lines[: args.limit]
    contestants: list[Contestant] = []
    if args.replay_recorded:
        contestants.append(("Decider 2B (recorded)", recorded_picks(lines), "self-hosted"))
    else:
        client = make_decider_client(args.decider_url)
        try:
            client.health()
            contestants.append(("Decider 2B", decider_picks(lines, client), "self-hosted"))
        except DeciderError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        finally:
            client.close()
    contestants.append(("keyword rules", keyword_picks(lines), "$0"))
    if args.allow_bedrock:
        from .llm import StrandsPicker

        picker = StrandsPicker(allow_bedrock=True)
        contestants.append((f"LLM picker ({picker.model_id})", llm_picks(lines, picker), None))
    table = race_table(lines, contestants, args.thresholds)
    print(table)
    print()
    print(NOTES)
    if args.out:
        Path(args.out).write_text(table + "\n\n" + NOTES + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
