"""``decider-bench``: replay labelled cases against a Decider server and report the numbers.

Case file (JSONL, one object per line)::

    {"id": "c1", "state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}},
     "expected": {"q": true}}

``questions`` may be omitted if ``--questions questions.yaml`` is given. ``expected`` per
question: noul -> true/false; choice -> the option name; score -> the level index (compared
with the rounded score).

Output: accuracy at each confidence threshold (accuracy among answers whose confidence is at
least the threshold), the abstain rate (share of answers below it), and latency p50/p90 (client
wall clock and server-reported).

An OPTIONAL ``--baseline bedrock --model-id X`` path asks a Bedrock model the same questions.
It costs money, so it refuses to run without ``--i-approve-spend`` and is never used in tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import (
    Answer,
    ChoiceAnswer,
    DeciderClient,
    DeciderError,
    DeciderUnavailable,
    NoulAnswer,
)
from .policy import confidence_and_verdict
from .questions import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    load_questions,
    parse_questions,
)

DEFAULT_URL = "http://127.0.0.1:8099"
DEFAULT_THRESHOLDS = (0.5, 0.7, 0.9, 0.95)


class SpendNotApproved(RuntimeError):
    """The Bedrock baseline was requested without ``--i-approve-spend``."""


@dataclass(frozen=True)
class Case:
    id: str
    state: Any
    questions: dict[str, Question]
    expected: dict[str, Any]


def load_cases(
    path: str | Path, shared_questions: Mapping[str, Question] | None = None
) -> list[Case]:
    cases: list[Case] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if "questions" in raw:
                questions = parse_questions(raw["questions"])
            elif shared_questions:
                questions = dict(shared_questions)
            else:
                raise ValueError(f"{path}:{lineno}: no 'questions' and no --questions file")
            expected = raw.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise ValueError(f"{path}:{lineno}: 'expected' must be a non-empty object")
            missing = set(expected) - set(questions)
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: expected refers to unknown questions {sorted(missing)}"
                )
            cases.append(Case(str(raw.get("id", lineno)), raw["state"], questions, expected))
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def _norm_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def is_correct(answer: Answer, expected: Any) -> bool:
    if isinstance(answer, NoulAnswer):
        return answer.verdict == _norm_bool(expected)
    if isinstance(answer, ChoiceAnswer):
        return answer.choice == str(expected)
    return round(answer.score) == int(expected)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


@dataclass
class Item:
    case_id: str
    question: str
    correct: bool
    confidence: float


@dataclass
class BenchReport:
    items: list[Item] = field(default_factory=list)
    client_ms: list[float] = field(default_factory=list)
    server_ms: list[float] = field(default_factory=list)
    unavailable: int = 0
    cases: int = 0
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS

    def accuracy_all(self) -> float | None:
        return sum(i.correct for i in self.items) / len(self.items) if self.items else None

    def at_threshold(self, t: float) -> dict[str, float | int | None]:
        kept = [i for i in self.items if i.confidence >= t]
        total = len(self.items)
        return {
            "threshold": t,
            "answered": len(kept),
            "accuracy": (sum(i.correct for i in kept) / len(kept)) if kept else None,
            "abstain_rate": ((total - len(kept)) / total) if total else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "answers": len(self.items),
            "unavailable": self.unavailable,
            "accuracy_all": self.accuracy_all(),
            "thresholds": [self.at_threshold(t) for t in self.thresholds],
            "client_ms": {
                "p50": percentile(self.client_ms, 0.5),
                "p90": percentile(self.client_ms, 0.9),
            },
            "server_ms": {
                "p50": percentile(self.server_ms, 0.5),
                "p90": percentile(self.server_ms, 0.9),
            },
        }

    def render(self) -> str:
        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{100 * x:5.1f}%"

        def ms(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.1f} ms"

        d = self.to_dict()
        lines = [
            f"cases: {d['cases']}  answers: {d['answers']}  server unavailable: {d['unavailable']}",
            f"accuracy over all answers: {pct(d['accuracy_all'])}",
            "",
            "threshold  answered  accuracy(answered)  abstain",
        ]
        for row in d["thresholds"]:
            lines.append(
                f"{row['threshold']:>9.2f}  {row['answered']:>8}  {pct(row['accuracy']):>18}  "
                f"{pct(row['abstain_rate']):>7}"
            )
        lines += [
            "",
            f"latency client p50 {ms(d['client_ms']['p50'])}  p90 {ms(d['client_ms']['p90'])}",
            f"latency server p50 {ms(d['server_ms']['p50'])}  p90 {ms(d['server_ms']['p90'])}",
        ]
        return "\n".join(lines)


def run_bench(
    cases: Sequence[Case],
    client: DeciderClient,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    warmup: int = 0,
) -> BenchReport:
    """Replay ``cases``. ``warmup`` requests (the first case, repeated) are sent and discarded."""
    report = BenchReport(thresholds=tuple(thresholds), cases=len(cases))
    for _ in range(warmup):
        try:
            client.ask(cases[0].state, cases[0].questions)
        except DeciderUnavailable:
            break
    for case in cases:
        try:
            response = client.ask(case.state, case.questions)
        except DeciderUnavailable:
            report.unavailable += 1
            continue
        report.client_ms.append(response.client_ms)
        if response.server_ms is not None:
            report.server_ms.append(response.server_ms)
        for name, expected in case.expected.items():
            answer = response.answers[name]
            confidence, _ = confidence_and_verdict(answer)
            report.items.append(Item(case.id, name, is_correct(answer, expected), confidence))
    return report


# ----------------------------------------------------------------------------------- baseline


@dataclass(frozen=True)
class BaselineReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


def build_baseline_prompt(state: Any, questions: Mapping[str, Question]) -> str:
    """The same questions as the Decider request, phrased for a general chat model."""
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    lines = [
        "Read the state, then answer every question.",
        "",
        "STATE:",
        state_text,
        "",
        "QUESTIONS:",
    ]
    for name, q in questions.items():
        lines.append(f"- {name}: {q.instructions}")
        if isinstance(q, NoulQuestion):
            lines.append("  answer with true or false")
            for side, text in (q.criteria or {}).items():
                lines.append(f"  {side}: {text}")
        elif isinstance(q, ChoiceQuestion):
            lines.append("  answer with exactly one of these option names:")
            for option, text in q.criteria.items():
                lines.append(f"  {option}" + (f": {text}" if text else ""))
        elif isinstance(q, ScoreQuestion):
            lines.append("  answer with the integer index of one level (0 is the lowest):")
            for i, level in enumerate(q.criteria):
                lines.append(f"  {i}: {level}")
    lines += ["", 'Reply with JSON only: {"answers": {"<question name>": <answer>, ...}}']
    return "\n".join(lines)


def parse_baseline_reply(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    data = json.loads(text[start : end + 1])
    answers = data["answers"]
    if not isinstance(answers, dict):
        raise ValueError("'answers' is not an object")
    return answers


def baseline_is_correct(question: Question, given: Any, expected: Any) -> bool:
    if isinstance(question, NoulQuestion):
        return _norm_bool(given) == _norm_bool(expected)
    if isinstance(question, ChoiceQuestion):
        return str(given) == str(expected)
    try:
        return int(round(float(given))) == int(expected)
    except (TypeError, ValueError):
        return False


@dataclass
class BaselineReport:
    model_id: str
    answers: int = 0
    correct: int = 0
    unparsable: int = 0
    latency_ms: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def render(self) -> str:
        acc = f"{100 * self.correct / self.answers:.1f}%" if self.answers else "n/a"
        p50, p90 = percentile(self.latency_ms, 0.5), percentile(self.latency_ms, 0.9)
        cost = (
            "n/a (no --price-in/--price-out)" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        )
        return "\n".join(
            [
                f"baseline model: {self.model_id}",
                f"answers: {self.answers}  accuracy: {acc}  unparsable replies: {self.unparsable}",
                f"latency p50 {p50 if p50 is None else round(p50, 1)} ms  "
                f"p90 {p90 if p90 is None else round(p90, 1)} ms",
                f"tokens in {self.input_tokens} out {self.output_tokens}  cost {cost}",
            ]
        )


def run_baseline(
    cases: Sequence[Case],
    invoke: Callable[[str], BaselineReply],
    model_id: str,
    price_in_per_m: float | None = None,
    price_out_per_m: float | None = None,
) -> BaselineReport:
    """Ask a general model the same questions. ``invoke`` makes the paid call; tests pass a fake."""
    report = BaselineReport(model_id=model_id)
    for case in cases:
        prompt = build_baseline_prompt(case.state, case.questions)
        started = time.perf_counter()
        reply = invoke(prompt)
        report.latency_ms.append((time.perf_counter() - started) * 1000.0)
        report.input_tokens += reply.input_tokens
        report.output_tokens += reply.output_tokens
        try:
            given = parse_baseline_reply(reply.text)
        except (ValueError, KeyError, json.JSONDecodeError):
            report.unparsable += 1
            report.answers += len(case.expected)
            continue
        for name, expected in case.expected.items():
            report.answers += 1
            if name in given and baseline_is_correct(case.questions[name], given[name], expected):
                report.correct += 1
    if price_in_per_m is not None and price_out_per_m is not None:
        report.cost_usd = (
            report.input_tokens * price_in_per_m + report.output_tokens * price_out_per_m
        ) / 1_000_000
    return report


def require_spend_approval(baseline: str | None, approved: bool) -> None:
    """Raise unless the paid baseline is explicitly approved. Called before any import of boto3."""
    if baseline and not approved:
        raise SpendNotApproved(
            f"--baseline {baseline} makes paid API calls. Re-run with --i-approve-spend only after "
            "the spend has been approved by the account owner."
        )


def make_bedrock_invoker(model_id: str, region: str | None) -> Callable[[str], BaselineReply]:
    """Build a Bedrock ``converse`` invoker. Imports boto3 lazily; makes no call until used."""
    import boto3  # imported lazily: the baseline is optional

    client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(prompt: str) -> BaselineReply:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.0},
        )
        text = "".join(block.get("text", "") for block in resp["output"]["message"]["content"])
        usage = resp.get("usage", {})
        return BaselineReply(text, usage.get("inputTokens", 0), usage.get("outputTokens", 0))

    return invoke


# ---------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decider-bench", description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", required=True, help="labelled cases, JSONL")
    p.add_argument("--questions", help="shared questions YAML used when a case has none")
    p.add_argument("--base-url", default=os.environ.get("STRANDS_DECIDER_URL", DEFAULT_URL))
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    p.add_argument("--warmup", type=int, default=1, help="discarded warm-up requests (default 1)")
    p.add_argument("--json", dest="json_out", help="also write the report as JSON to this path")
    p.add_argument("--baseline", choices=["bedrock"], help="ALSO ask a Bedrock model (costs money)")
    p.add_argument("--model-id", help="Bedrock model id for --baseline")
    p.add_argument("--region", help="AWS region for --baseline")
    p.add_argument("--price-in", type=float, help="USD per 1M input tokens, to price the baseline")
    p.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    p.add_argument("--i-approve-spend", action="store_true", help="required for --baseline")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_spend_approval(args.baseline, args.i_approve_spend)
        if args.baseline and not args.model_id:
            raise SystemExit("--baseline needs --model-id")
        shared = load_questions(args.questions) if args.questions else None
        cases = load_cases(args.cases, shared)
        thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())
    except SpendNotApproved as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    client = DeciderClient(args.base_url, timeout=args.timeout)
    try:
        client.health()
        report = run_bench(cases, client, thresholds, warmup=args.warmup)
    except DeciderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    finally:
        client.close()
    print(report.render())
    out: dict[str, Any] = {"decider": report.to_dict()}

    if args.baseline == "bedrock":
        invoke = make_bedrock_invoker(args.model_id, args.region)
        base = run_baseline(cases, invoke, args.model_id, args.price_in, args.price_out)
        print()
        print(base.render())
        out["baseline"] = {
            "model_id": base.model_id,
            "answers": base.answers,
            "correct": base.correct,
            "cost_usd": base.cost_usd,
        }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
