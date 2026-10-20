"""From OCR output to menu lines: Amazon Textract JSON or one OCR string per line; line crops.

Textract ``DetectDocumentText`` returns LINE blocks with text and a normalised bounding box. On a
printed service menu the service name, the dotted leader and the price usually come back as
separate LINE blocks on the same row; sometimes the leader or the price is glued to the name.
:func:`parse_textract` pairs each name with the price on its row (vertical overlap), drops the
leaders, attaches a smaller description line directly under a name, and skips header and footer
lines that carry no price. A line without a price between priced lines is kept with
``price=None`` so it is flagged for a retake rather than silently dropped.
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import BBox, MenuLine

CropMode = Literal["strip", "cells"]

# "$1,089.00", "$29.95", "$12", "$0.00 (no charge)"; "S0.00" is a common OCR misread of "$0.00"
# and is accepted only with cents, so "S60" in a name is not a price.
_PRICE = re.compile(
    r"(?:\$\s?(?P<d>\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d+(?:\.\d{2})?)"
    r"|(?<![A-Za-z0-9])S(?P<s>\d{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2}))"
    r"(?P<nc>\s*\((?i:no charge|free|n/c)\))?"
)
# Dotted leaders: two or more dots, bullets, middle dots, ellipses or underscores.
_LEADER = re.compile(r"(?:[.•·…_]\s*){2,}|…")
_EDGE = " .:,;-–—"

ROW_OVERLAP = 0.5
"""Minimum vertical overlap (share of the shorter box) for a name and a price to share a row."""
DESCRIPTION_HEIGHT = 0.9
"""A description line is shorter than this share of the name line's height."""


def parse_price(text: str) -> tuple[float, str] | None:
    """The first price in ``text``: (value, text as printed with ``$``), or None."""
    m = _PRICE.search(text)
    if m is None:
        return None
    digits = m.group("d") or m.group("s")
    printed = m.group(0).strip()
    if printed.startswith("S"):
        printed = "$" + printed[1:]
    return float(digits.replace(",", "")), printed


def clean_text(text: str) -> str:
    """Drop dotted leaders and collapse spaces; trim punctuation at the ends."""
    return " ".join(_LEADER.sub(" ", text).split()).strip(_EDGE)


def compose_text(name: str, price_text: str | None, description: str | None) -> str:
    """The line as Decider reads it: ``name ... $price description``."""
    parts = " ... ".join(p for p in (name, price_text) if p)
    return f"{parts} {description}" if description else parts


def _split(cleaned: str) -> tuple[str, tuple[float, str] | None, str | None]:
    """Split one cleaned row into (name, price, description after the price)."""
    m = _PRICE.search(cleaned)
    if m is None:
        return cleaned, None, None
    price = parse_price(m.group(0))
    before, after = cleaned[: m.start()].strip(_EDGE), cleaned[m.end() :].strip(_EDGE)
    if before and after:
        return before, price, after
    return before or after, price, None


def parse_line_texts(rows: Iterable[str]) -> list[MenuLine]:
    """One OCR string per menu line (e.g. Apple Vision on a line strip) -> menu lines.

    The price may come first or last. Text after a price that follows the name is taken as the
    description. ``text`` keeps the OCR string unchanged: it is what Decider reads.
    """
    lines = []
    for i, row in enumerate(rows, 1):
        name, price, description = _split(clean_text(row))
        lines.append(
            MenuLine(
                index=i,
                text=row,
                name=name,
                price=price[0] if price else None,
                bbox=None,
                price_text=price[1] if price else None,
                description=description,
            )
        )
    return lines


# ----------------------------------------------------------------------------------- Textract


@dataclass
class _Frag:
    text: str
    box: BBox
    price: tuple[float, str] | None = None
    price_box: BBox | None = None
    description: str | None = None
    desc_box: BBox | None = None
    extra: list[str] = field(default_factory=list)


def _bbox(block: Mapping[str, Any]) -> BBox:
    b = block["Geometry"]["BoundingBox"]
    return BBox(float(b["Left"]), float(b["Top"]), float(b["Width"]), float(b["Height"]))


def _overlap(a: BBox, b: BBox) -> float:
    inter = min(a.bottom, b.bottom) - max(a.top, b.top)
    return max(0.0, inter) / max(min(a.height, b.height), 1e-9)


def _word_boxes(
    line: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[BBox | None, BBox | None]:
    """(name box, price box) from the WORD children of a line that holds both."""
    name_box: BBox | None = None
    price_box: BBox | None = None
    for rel in line.get("Relationships") or []:
        if rel.get("Type") != "CHILD":
            continue
        for wid in rel.get("Ids", []):
            word = by_id.get(wid)
            if not word or word.get("BlockType") != "WORD":
                continue
            text, box = str(word.get("Text", "")), _bbox(word)
            if _PRICE.search(text) or text.lower().strip("()") in ("no", "charge"):
                price_box = box.union(price_box)
            elif clean_text(text):
                name_box = box.union(name_box)
    return name_box, price_box


def parse_textract(response: Mapping[str, Any]) -> list[MenuLine]:
    """Amazon Textract ``DetectDocumentText`` JSON -> priced menu lines, top to bottom."""
    blocks = response.get("Blocks") or []
    by_id = {b["Id"]: b for b in blocks if "Id" in b}
    texts: list[_Frag] = []
    prices: list[_Frag] = []
    for block in blocks:
        if block.get("BlockType") != "LINE":
            continue
        cleaned = clean_text(str(block.get("Text", "")))
        if not cleaned:
            continue  # a dotted leader on its own
        box = _bbox(block)
        name, price, description = _split(cleaned)
        if price and not name:
            prices.append(_Frag("", box, price, box))
        elif price:
            name_box, price_box = _word_boxes(block, by_id)
            texts.append(
                _Frag(name, name_box or box, price, price_box, description, None, [cleaned])
            )
        else:
            texts.append(_Frag(cleaned, box))

    # pair each price with the unpriced text on its row, to its left, with the most overlap
    for price in sorted(prices, key=lambda f: f.box.top):
        candidates = [
            t
            for t in texts
            if t.price is None
            and t.box.left < price.box.left
            and _overlap(t.box, price.box) >= ROW_OVERLAP
        ]
        if candidates:
            best = max(candidates, key=lambda t: _overlap(t.box, price.box))
            best.price, best.price_box = price.price, price.box
        else:
            texts.append(price)  # a price whose name could not be read

    items = sorted((t for t in texts if t.price is not None), key=lambda f: f.box.top)
    loose = sorted((t for t in texts if t.price is None), key=lambda f: f.box.top)
    first_top = items[0].box.top if items else 0.0
    last_bottom = max((i.box.bottom for i in items), default=0.0)
    for frag in loose:
        above = [i for i in items if i.box.top < frag.box.top]
        parent = max(above, key=lambda i: i.box.top) if above else None
        on_a_row = any(_overlap(frag.box, i.box) >= ROW_OVERLAP for i in items)
        if (
            parent is not None
            and parent.description is None
            and not on_a_row
            and frag.box.top - parent.box.bottom <= parent.box.height
            and frag.box.height < DESCRIPTION_HEIGHT * parent.box.height
        ):
            parent.description, parent.desc_box = frag.text, frag.box
        elif frag.box.top < first_top or frag.box.top > last_bottom:
            continue  # header or footer text without a price
        else:
            items.append(frag)  # a line whose price could not be read

    lines = []
    for i, f in enumerate(sorted(items, key=lambda f: f.box.top), 1):
        has_name = bool(f.text)
        name_box = f.box.union(f.desc_box) if has_name else f.desc_box
        whole = (name_box or f.price_box or f.box).union(f.price_box)
        price_text = f.price[1] if f.price else None
        lines.append(
            MenuLine(
                index=i,
                text=compose_text(f.text, price_text, f.description),
                name=f.text,
                price=f.price[0] if f.price else None,
                bbox=whole,
                price_text=price_text,
                description=f.description,
                name_bbox=name_box,
                price_bbox=f.price_box,
            )
        )
    return lines


def textract_detect(image_bytes: bytes, client: Any) -> dict[str, Any]:
    """Call ``DetectDocumentText`` on a boto3 Textract client passed in; return the raw response.

    No client is created here. Parse the result with :func:`parse_textract`.
    """
    return client.detect_document_text(Document={"Bytes": image_bytes})


# --------------------------------------------------------------------------------------- crops


def crop_line(
    image_bytes: bytes, line: MenuLine, mode: CropMode = "cells", *, pad: float = 0.3
) -> tuple[str, ...]:
    """Crop one line out of the menu photo; return base64 PNG strings for Decider.

    ``strip``: one full-width band around the line. ``cells``: the name (with its description)
    and the price as two separate crops. ``pad`` is a share of the box height added on each side.
    """
    if mode not in ("strip", "cells"):
        raise ValueError(f"mode must be 'strip' or 'cells', got {mode!r}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is a dev/demo dependency
        raise RuntimeError("crop_line needs pillow: pip install pillow") from exc

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    def png(box: tuple[int, int, int, int]) -> str:
        buf = io.BytesIO()
        image.crop(box).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def around(b: BBox, full_width: bool) -> tuple[int, int, int, int]:
        dy = pad * b.height * h
        top, bottom = max(0, int(b.top * h - dy)), min(h, int(b.bottom * h + dy) + 1)
        if full_width:
            return 0, top, w, bottom
        return max(0, int(b.left * w - dy)), top, min(w, int(b.right * w + dy) + 1), bottom

    if mode == "strip":
        if line.bbox is None:
            raise ValueError(f"line {line.index} has no bounding box to crop")
        return (png(around(line.bbox, True)),)
    if line.name_bbox is None or line.price_bbox is None:
        raise ValueError(f"line {line.index} needs a name and a price bounding box for cells")
    return png(around(line.name_bbox, False)), png(around(line.price_bbox, False))
