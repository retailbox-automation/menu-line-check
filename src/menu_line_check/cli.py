"""``menu-check``: check a service menu from its OCR JSON and print one row per line.

    menu-check (--ocr-json menu.json | --textract menu.jpg --allow-textract) \\
        --vehicle "<year> <make> <model> <odometer>" \\
        [--decider-url URL] [--thresholds decline_addon=0.8,check_schedule=0.5] \\
        [--image menu.jpg] \\
        [--nhtsa-recorded recalls.json | --nhtsa-cache DIR] [--allow-bedrock] \\
        [--max-usd 1.0] [--ledger out/ledger.jsonl] [--json out/report.json]

``--ocr-json`` is an Amazon Textract ``DetectDocumentText`` response. ``--image`` is the photo
it came from: the name and price crops of each line are then sent to Decider for ``readable``.
``--textract PHOTO`` instead runs Textract on the photo (paid, $0.0015 per page, refused without
``--allow-textract``), saves the raw response as ``textract-<photo>.json`` next to ``--json`` /
``--ledger`` (or in the current directory) so later runs replay it free with ``--ocr-json``, and
uses the photo for the line crops. Textract and Bedrock share one spend cap.
Without ``--allow-bedrock`` no Bedrock call is made: the advisor questions come from templates
(``FakeWriter``) and low-confidence lines go to "ask the advisor" instead of an LLM picker.
With it, every Bedrock call is checked against a spend cap (``--max-usd``, default env
``N2_MAX_USD`` or 1.0): the run stops with exit code 4 once the cap is reached, and the spend is
printed after the report.
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
from menu_line_tool_selection import CRITERIA_SETS, parse_thresholds, resolve_thresholds

from .llm import DEFAULT_REGION, FakeWriter
from .models import LineOutcome, MenuLine, MenuReport, Vehicle
from .nhtsa import NhtsaClient
from .ocr import crop_line, parse_textract, textract_detect
from .pipeline import DEFAULT_CONCURRENCY, check_menu, questions_per_line
from .spend import SpendCap, SpendCapReached

DEFAULT_URL = "http://127.0.0.1:8099"
NAME_WIDTH = 36


def make_decider_client(url: str, timeout: float = 30.0) -> DeciderClient:
    return DeciderClient(url, timeout=timeout)


def make_bedrock_parts(
    decider: DeciderClient, ledger: Ledger | None, region: str, spend: SpendCap
) -> Any:
    """(BedrockWriter, StrandsPicker) sharing one ``spend`` cap.

    Imports boto3 here, only when --allow-bedrock is given.
    """
    import boto3

    from .llm import BedrockWriter, StrandsPicker

    runtime = boto3.client("bedrock-runtime", region_name=region)
    writer = BedrockWriter(runtime, allow_bedrock=True, spend=spend)
    picker = StrandsPicker(decider, allow_bedrock=True, region=region, ledger=ledger, spend=spend)
    return writer, picker


def attach_crops(lines: list[MenuLine], image_bytes: bytes) -> list[MenuLine]:
    """Add the name + price crops (or a strip when a box is missing) to every line."""
    out = []
    for line in lines:
        mode = "cells" if line.name_bbox and line.price_bbox else "strip"
        images = crop_line(image_bytes, line, mode) if line.bbox else ()
        out.append(dataclasses.replace(line, images=images))
    return out


def describe_questions(lines: list[MenuLine]) -> str:
    """``2 (safety, tool; readable by code check)`` or ``3 (...; readable on the line image)``."""
    counts = {len(questions_per_line(line)) for line in lines}
    if counts == {3}:
        return "3 (safety, tool; readable on the line image)"
    if counts == {2} or not counts:
        return "2 (safety, tool; readable by code check)"
    return "2 or 3 (safety, tool; readable on the line image where there is one)"


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
    if report.decider_wall_ms is not None:
        rows.append(f"{len(report.outcomes)} lines decided in {report.decider_wall_ms:.0f} ms")
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="menu-check", description=__doc__.split("\n\n")[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--ocr-json", help="Textract DetectDocumentText response (saved, free)")
    source.add_argument(
        "--textract",
        metavar="PHOTO",
        help="run Amazon Textract on this menu photo (paid, needs --allow-textract); the "
        "response is saved so later runs can replay it with --ocr-json; the photo is also "
        "used for line crops unless --image is given",
    )
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
    p.add_argument(
        "--criteria",
        choices=list(CRITERIA_SETS),
        default="spike",
        help="tool criteria sentences: spike (measured, default) or short (same meaning, "
        "not measured)",
    )
    p.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
        help=f"lines asked of Decider at the same time (default {DEFAULT_CONCURRENCY})",
    )
    p.add_argument("--image", help="the menu photo the OCR JSON came from (adds line crops)")
    nhtsa = p.add_mutually_exclusive_group()
    nhtsa.add_argument("--nhtsa-recorded", help="serve this recorded NHTSA response (offline)")
    nhtsa.add_argument("--nhtsa-cache", help="disk cache directory for live NHTSA lookups")
    p.add_argument(
        "--allow-bedrock",
        action="store_true",
        help="use Bedrock (paid) for the LLM picker and the advisor questions",
    )
    p.add_argument(
        "--max-usd",
        type=_non_negative_float,
        help="hard cap on AWS spend (Bedrock + Textract) for this run, checked before each call "
        "(default env N2_MAX_USD or 1.0)",
    )
    p.add_argument(
        "--allow-textract",
        action="store_true",
        help="allow the paid Textract call of --textract ($0.0015 per page)",
    )
    p.add_argument("--region", default=DEFAULT_REGION, help="AWS region for Bedrock")
    p.add_argument("--ledger", help="append one JSONL row per line to this file")
    p.add_argument("--json", dest="json_out", help="also write the report as JSON")
    return p


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a whole number of at least 1, got {text!r}")
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a number of dollars, 0 or more, got {text!r}")
    return value


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


def make_textract_client(region: str) -> Any:
    """A boto3 Textract client. Imports boto3 here, only when --textract is given."""
    import boto3

    return boto3.client("textract", region_name=region)


def textract_menu(
    photo_bytes: bytes, photo: Path, spend: SpendCap, region: str, out_dir: Path
) -> tuple[list[MenuLine], Path]:
    """One paid Textract call on the photo: guard, call, count, save the raw response, parse.

    The raw response is saved before it is parsed, so a paid answer is never lost; replay it
    later with ``--ocr-json`` at no cost.
    """
    spend.check_textract()
    response = textract_detect(photo_bytes, make_textract_client(region))
    spend.add_textract()
    saved = out_dir / f"textract-{photo.stem}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(response, indent=2, ensure_ascii=False), "utf-8")
    return parse_textract(response), saved


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.textract and not args.allow_textract:
        print(
            "error: --textract calls Amazon Textract, which costs money ($0.0015 per page): "
            "add --allow-textract once the spend is approved, or pass a saved response with "
            "--ocr-json.",
            file=sys.stderr,
        )
        return 2
    lines: list[MenuLine] = []
    photo_bytes: bytes | None = None
    try:
        if args.textract:
            photo_bytes = Path(args.textract).read_bytes()
        else:
            lines = parse_textract(json.loads(Path(args.ocr_json).read_text(encoding="utf-8")))
            if args.image:
                lines = attach_crops(lines, Path(args.image).read_bytes())
        spend = SpendCap(args.max_usd) if args.allow_bedrock or args.textract else None
    except (OSError, ValueError) as exc:  # includes a bad N2_MAX_USD
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
    try:
        if photo_bytes is not None:
            assert spend is not None
            anchor = Path(args.json_out).parent if args.json_out else None
            anchor = anchor or (Path(args.ledger).parent if args.ledger else Path.cwd())
            lines, saved = textract_menu(
                photo_bytes, Path(args.textract), spend, args.region, anchor
            )
            print(f"Textract response saved to {saved} (replay free: --ocr-json {saved})")
            crops = Path(args.image).read_bytes() if args.image else photo_bytes
            lines = attach_crops(lines, crops)
        if args.allow_bedrock:
            assert spend is not None
            writer, picker = make_bedrock_parts(decider, ledger, args.region, spend)
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
            concurrency=args.concurrency,
            criteria=args.criteria,
        )
    except SpendCapReached as exc:
        print(f"error: {exc}", file=sys.stderr)
        if spend is not None:
            print(spend.summary, file=sys.stderr)
        return 4
    except (OSError, ValueError) as exc:  # --textract: unreadable --image, unparsable response
        if photo_bytes is None:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        decider.close()

    limits = resolve_thresholds(args.thresholds, args.act_threshold)
    shown = ", ".join(f"{k} {v:g}" for k, v in limits.items())
    header = "\n".join(
        [
            f"Menu Line Check \u00b7 {args.vehicle.label} \u00b7 {len(lines)} lines"
            + ("" if args.allow_bedrock else " \u00b7 offline writer, no LLM picker"),
            f"Decider {args.decider_url} (up to {args.concurrency} at a time) \u00b7 "
            f"questions per line: {describe_questions(lines)} \u00b7 tool criteria: "
            f"{args.criteria}",
            f"act >= {shown}",
        ]
    )
    print(render(report, header))
    if spend is not None:
        print(spend.summary)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
