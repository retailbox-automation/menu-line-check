"""Write ``textract_menu.json``: a HAND-MADE Amazon Textract ``DetectDocumentText`` response.

It is not the output of a real Textract call. It has the real response shape (PAGE, LINE and WORD
blocks, normalised ``BoundingBox`` and ``Polygon``, ``Relationships``) for a printed 12-line
service menu photographed straight on, so the parser can be tested without AWS.

Layout choices that mirror what Textract does on such a menu:

* the service name, the dotted leader and the price come back as separate LINE blocks;
* on "Shop supplies" the leader is glued to the name; on "Environmental disposal fee" name,
  leader and price come back as ONE line;
* "Tech diagnostic" has a smaller description line directly under it;
* the header line has no price.

Run: ``python tests/fixtures/make_textract_menu.py`` (deterministic).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).with_name("textract_menu.json")

HEADER = "Recommended service — 30,000 mi"
LINES = [
    ("Oil & filter change, full synthetic", "$89.95", None),
    ("Engine oil conditioner", "$39.00", None),
    ("Fuel injection cleaning", "$149.00", None),
    ("Tire rotation", "$39.95", None),
    ("Cabin air filter", "$79.95", None),
    ("Replace fuel pump module", "$1,089.00", None),
    ("Tech diagnostic", "$189.00", "Required before any repair"),
    ("Shop supplies", "$34.50", None),
    ("Front brake pads & rotors", "$489.00", None),
    ("Nitrogen tire fill", "$29.95", None),
    ("Throttle body service", "$119.00", None),
    ("Environmental disposal fee", "$12.95", None),
]
LEADER_GLUED = {"Shop supplies"}
ONE_LINE = {"Environmental disposal fee"}

CHAR_W = 0.0112  # normalised width of one character at the item font size
ROW_H = 0.024
DESC_H = 0.017
LEFT = 0.08
RIGHT = 0.92
FIRST_TOP = 0.125
STEP = 0.064

_counter = 0


def _id() -> str:
    global _counter
    _counter += 1
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"n2-textract-fixture/{_counter}"))


def _geometry(left: float, top: float, width: float, height: float) -> dict:
    r = lambda x: round(x, 6)  # noqa: E731
    return {
        "BoundingBox": {"Width": r(width), "Height": r(height), "Left": r(left), "Top": r(top)},
        "Polygon": [
            {"X": r(left), "Y": r(top)},
            {"X": r(left + width), "Y": r(top)},
            {"X": r(left + width), "Y": r(top + height)},
            {"X": r(left), "Y": r(top + height)},
        ],
    }


def _words(text: str, left: float, top: float, width: float, height: float, conf: float):
    tokens = text.split(" ")
    total = sum(len(t) for t in tokens) + len(tokens) - 1
    x, out = left, []
    for t in tokens:
        w = width * len(t) / total
        out.append(
            {
                "BlockType": "WORD",
                "Confidence": conf,
                "Text": t,
                "TextType": "PRINTED",
                "Geometry": _geometry(x, top, w, height),
                "Id": _id(),
            }
        )
        x += w + width / total
    return out


def build() -> dict:
    lines: list[dict] = []
    words: list[dict] = []

    def add(text: str, left: float, top: float, height: float, conf: float, width=None):
        width = width if width is not None else len(text) * CHAR_W * (height / ROW_H)
        ws = _words(text, left, top, width, height, conf)
        words.extend(ws)
        lines.append(
            {
                "BlockType": "LINE",
                "Confidence": conf,
                "Text": text,
                "Geometry": _geometry(left, top, width, height),
                "Id": _id(),
                "Relationships": [{"Type": "CHILD", "Ids": [w["Id"] for w in ws]}],
            }
        )

    add(HEADER, LEFT, 0.050, 0.032, 99.71)
    for i, (name, price, desc) in enumerate(LINES):
        top = FIRST_TOP + i * STEP + 0.0015 * (i % 3)  # a little row-to-row drift
        name_w = len(name) * CHAR_W
        price_w = len(price) * CHAR_W
        price_left = RIGHT - price_w
        lead_left, lead_right = LEFT + name_w + 0.012, price_left - 0.012
        dots = "." * max(3, int((lead_right - lead_left) / 0.0075))
        conf = 99.9 - 0.11 * (i % 4)
        if name in ONE_LINE:
            add(f"{name} {dots} {price}", LEFT, top, ROW_H, 97.84, width=RIGHT - LEFT)
        elif name in LEADER_GLUED:
            add(f"{name} {dots}", LEFT, top, ROW_H, 96.12, width=lead_right - LEFT)
            add(price, price_left, top + 0.0008, ROW_H, conf)
        else:
            add(name, LEFT, top, ROW_H, conf)
            add(dots, lead_left, top + 0.004, ROW_H * 0.7, 88.4, width=lead_right - lead_left)
            add(price, price_left, top + 0.0008, ROW_H, conf)
        if desc:
            add(desc, LEFT + 0.02, top + ROW_H + 0.004, DESC_H, 99.02)

    page = {
        "BlockType": "PAGE",
        "Geometry": _geometry(0.0, 0.0, 1.0, 1.0),
        "Id": _id(),
        "Relationships": [{"Type": "CHILD", "Ids": [line["Id"] for line in lines]}],
    }
    return {
        "DocumentMetadata": {"Pages": 1},
        "Blocks": [page, *lines, *words],
        "DetectDocumentTextModelVersion": "1.0",
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
