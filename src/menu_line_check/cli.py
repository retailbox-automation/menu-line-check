"""``menu-check``: check a service menu from its OCR JSON and print one row per line.

    menu-check --ocr-json menu.json --vehicle "<year> <make> <model> <odometer>" \\
        [--decider-url URL] [--thresholds decline_addon=0.8,check_schedule=0.5] \\
        [--image menu.jpg] \\
        [--nhtsa-recorded recalls.json | --nhtsa-cache DIR] [--allow-bedrock] \\
        [--ledger out/ledger.jsonl] [--json out/report.json]

``--ocr-json`` is an Amazon Textract ``DetectDocumentText`` response. ``--image`` is the photo
it came from: the name and price crops of each line are then sent to Decider for ``readable``.
Without ``--allow-bedrock`` no Bedrock call is made: the advisor questions come from templates
(``FakeWriter``) and low-confidence lines go to "ask the advisor" instead of an LLM picker.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from decider_kit import DeciderClient, DeciderError, Ledger
from menu_line_tool_selection import parse_thresholds, resolve_thresholds

from .llm import DEFAULT_REGION, FakeWriter
from .models import LineOutcome, MenuLine, MenuReport, Vehicle
from .nhtsa import NhtsaClient
from .ocr import crop_line, parse_textract
from .pipeline import check_menu

DEFAULT_URL = "http://127.0.0.1:8099"
NAME_WIDTH = 36


def make_decider_client(url: str, timeout: float = 30.0) -> DeciderClient:
    return DeciderClient(url, timeout=timeout)


def make_bedrock_parts(decider: DeciderClient, ledger: Ledger | None, region: str) -> Any:
    """(BedrockWriter, StrandsPicker). Imports boto3 here, only when --allow-bedrock is given."""
    import boto3

    from .llm import BedrockWriter, StrandsPicker

    runtime = boto3.client("bedrock-runtime", region_name=region)
    writer = BedrockWriter(runtime, allow_bedrock=True)
    picker = StrandsPicker(decider, allow_bedrock=True, region=region, ledger=ledger)
    return writer, picker


def attach_crops(lines: list[MenuLine], image_bytes: bytes) -> list[MenuLine]:
    """Add the name + price crops (or a strip when a box is missing) to every line."""
    out = []
    for line in lines:
        mode = "cells" if line.name_bbox and line.price_bbox else "strip"
        images = crop_line(image_bytes, line, mode) if line.bbox else ()
        out.append(dataclasses.replace(line, images=images))
    return out


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def format_row(o: LineOutcome) -> str:
    price = o.line.price_text or "?"
    conf = f"{o.confidence:.2f}" if o.confidence is not None else "-"
    ms = f"{o.decider_ms.client_ms:.0f}" if o.decider_ms else "-"
    name = _cut(o.line.name or "(unreadable)", NAME_WIDTH)
    return (
        f"{o.line.index:>2}  {name:<{NAME_WIDTH}}  {price:>10}  {o.route:<13}  "
        f"{o.tool or '-':<21}  {conf:>4}  {o.source:<7}  {ms:>5}"
    )


def format_detail(o: LineOutcome) -> str:
    if o.advisor_question:
        return f'    -> {o.verdict}. Ask: "{o.advisor_question}"'
    return f"    -> {o.verdict}. {o.tool_result.get('note', '')}".rstrip()


def render(report: MenuReport, header: str) -> str:
    head = (
        f"{'#':>2}  {'line':<{NAME_WIDTH}}  {'price':>10}  {'route':<13}  {'tool':<21}  "
        f"{'conf':>4}  {'src':<7}  {'ms':>5}"
    )
    rows = [header, "", head, "-" * len(head)]
    for o in report.outcomes:
        rows += [format_row(o), format_detail(o)]
    rows.append("")
    if report.summary_text and not report.summary_text.startswith(report.summary):
        rows.append(report.summary)
    rows.append(report.summary_text or report.summary)
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="menu-check", description=__doc__.split("\n\n")[0])
    p.add_argument("--ocr-json", required=True, help="Textract DetectDocumentText response")
    p.add_argument(
        "--vehicle", required=True, type=_vehicle, help='"<year> <make> <model> <odometer mi>"'
    )
    p.add_argument("--decider-url", default=os.environ.get("STRANDS_DECIDER_URL", DEFAULT_URL))
    limits = p.add_mutually_exclusive_group()
    limits.add_argument(
        "--thresholds",
        type=_thresholds,
        default={},
        help="per-tool act thresholds over the defaults, e.g. decline_addon=0.8,check_schedule=0.5",
    )
    limits.add_argument("--act-threshold", type=float, help="one act threshold for every tool")
    p.add_argument("--image", help="the menu photo the OCR JSON came from (adds line crops)")
    nhtsa = p.add_mutually_exclusive_group()
    nhtsa.add_argument("--nhtsa-recorded", help="serve this recorded NHTSA response (offline)")
    nhtsa.add_argument("--nhtsa-cache", help="disk cache directory for live NHTSA lookups")
    p.add_argument(
        "--allow-bedrock",
        action="store_true",
        help="use Bedrock (paid) for the LLM picker and the advisor questions",
    )
    p.add_argument("--region", default=DEFAULT_REGION, help="AWS region for Bedrock")
    p.add_argument("--ledger", help="append one JSONL row per line to this file")
    p.add_argument("--json", dest="json_out", help="also write the report as JSON")
    return p


def _thresholds(text: str) -> dict[str, float]:
    try:
        return parse_thresholds(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _vehicle(text: str) -> Vehicle:
    try:
        return Vehicle.parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lines = parse_textract(json.loads(Path(args.ocr_json).read_text(encoding="utf-8")))
        if args.image:
            lines = attach_crops(lines, Path(args.image).read_bytes())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    nhtsa = (
        NhtsaClient.from_recorded(args.nhtsa_recorded)
        if args.nhtsa_recorded
        else NhtsaClient(cache_dir=args.nhtsa_cache)
    )
    ledger = Ledger(args.ledger) if args.ledger else None
    decider = make_decider_client(args.decider_url)
    try:
        decider.health()
    except DeciderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        decider.close()
        return 3

    writer: Any = FakeWriter()
    picker = None
    if args.allow_bedrock:
        writer, picker = make_bedrock_parts(decider, ledger, args.region)
    try:
        report = check_menu(
            lines,
            args.vehicle,
            decider,
            thresholds=args.thresholds,
            act_threshold=args.act_threshold,
            llm_picker=picker,
            writer=writer,
            nhtsa=nhtsa,
            ledger=ledger,
        )
    finally:
        decider.close()

    limits = resolve_thresholds(args.thresholds, args.act_threshold)
    shown = ", ".join(f"{k} {v:g}" for k, v in limits.items())
    header = (
        f"Menu Line Check \u00b7 {args.vehicle.label} \u00b7 {len(lines)} lines \u00b7 "
        f"Decider {args.decider_url} \u00b7 act >= {shown}"
        + ("" if args.allow_bedrock else " \u00b7 offline writer, no LLM picker")
    )
    print(render(report, header))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
