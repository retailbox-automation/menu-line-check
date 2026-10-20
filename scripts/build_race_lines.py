"""Build ``src/menu_line_check/data/race_lines.jsonl`` from the N2 spike output (read-only).

    python scripts/build_race_lines.py <spike_out_dir> [out.jsonl]

``<spike_out_dir>`` holds ``labels.json`` (truth per line) and ``results.json`` (Apple Vision OCR
text per line under ``rows[i]["ocr"]["text"]`` and the recorded Decider answers). Only the lines
labelled readable are kept (103 of 120 in the run of 2026-10-20). Each output line:

    {"id", "text", "price", "tool", "safety", "readable", "condition",
     "recorded": {"source", "tool", "tool_conf", "p_safety", "p_readable",
                  "server_ms", "client_ms"}}

``recorded`` is approach C_ocr of the spike (Decider 2B on the OCR text, four questions per
request including ``over_100``). noul answers were stored as verdict + max(p, 1 - p), so the
probability is rebuilt as ``conf`` when the verdict is true and ``1 - conf`` when it is false.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "src" / "menu_line_check" / "data" / "race_lines.jsonl"
)


def p_true(answer: dict) -> float:
    conf = float(answer["conf"])
    return round(conf if answer["verdict"] else 1.0 - conf, 4)


def build(spike_dir: Path) -> list[dict]:
    labels = {x["id"]: x for x in json.loads((spike_dir / "labels.json").read_text())["labels"]}
    results = json.loads((spike_dir / "results.json").read_text())
    out = []
    for row in results["rows"]:
        truth = labels[row["id"]]
        if not truth["readable"]:
            continue
        rec = row["C_ocr"]
        answers = rec["answers"]
        out.append(
            {
                "id": row["id"],
                "text": row["ocr"]["text"],
                "price": truth["price"],
                "tool": truth["tool"],
                "safety": truth["safety"],
                "readable": truth["readable"],
                "condition": truth["condition"],
                "recorded": {
                    "source": f"spike {spike_dir.name}, approach C_ocr",
                    "tool": answers["tool"]["verdict"],
                    "tool_conf": answers["tool"]["conf"],
                    "p_safety": p_true(answers["safety"]),
                    "p_readable": p_true(answers["readable"]),
                    "server_ms": rec.get("server_ms"),
                    "client_ms": rec.get("ms"),
                },
            }
        )
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    rows = build(Path(argv[0]))
    out = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} lines to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
