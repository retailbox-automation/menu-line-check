"""From OCR output to menu lines: Amazon Textract JSON or one OCR string per line; line crops.

Textract ``DetectDocumentText`` returns LINE blocks with text and a normalised bounding box. On a
printed service menu the service name, the dotted leader and the price usually come back as
separate LINE blocks on the same row; sometimes the leader or the price is glued to the name.
:func:`parse_textract` pairs each name with the price on its row (vertical overlap), drops the
leaders, attaches a smaller description line directly under a name, and skips header and footer
lines that carry no price. A line without a price between priced lines is kept with
``price=None`` so it is flagged for a retake rather than silently dropped.

Some dealer sheets print no price per line: a bulleted list of services, then a totals block
(Menu Total, Shop Charges, Tax, Fees, TOTAL). :func:`parse_sheet` reads both layouts. When the
page has bullet lines (``•``, ``·``, ``▪``, a leading ``- `` or ``* ``), the service lines are
exactly the bullet rows (a bullet block joined with the text that continues it on its row, see
:func:`_merge_bullet_rows`), in reading order, plus the Shop Charges and Fees rows of the totals
block (a charge the owner can ask about); every other text (header fields, titles, promotion
badges, financing offers, footers) is ignored. Bullet lines under a ``Declined`` section header
are marked ``declined_on_sheet``. A page without bullets parses exactly as before.
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import BBox, MenuLine, SheetTotals

CropMode = Literal["strip", "cells"]
Layout = Literal["priced", "bullets"]

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

# A bullet at the start of a line: one bullet sign, or a dash / star followed by a space.
_BULLET = re.compile(r"^\s*(?:[•·▪]|[-*](?=\s))\s*")
# Totals-block row labels (lower case, words only) -> the SheetTotals field they fill.
_TOTAL_LABELS: dict[str, str] = {
    "menu total": "menu_total",
    "subtotal": "menu_total",
    "sub total": "menu_total",
    "shop charges": "shop_charges",
    "shop supplies": "shop_charges",
    "tax": "tax",
    "fees": "fees",
    "fee": "fees",
    "total": "total",
}
CHARGE_FIELDS = frozenset({"shop_charges", "fees"})
"""Totals rows that are also menu lines on a bullet sheet: charges the owner can ask about."""
DECLINED_HEADER = "declined"


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
    bullet: bool = False
    raw: str = ""
    """The OCR text (bullet removed) before trimming, so a joined row keeps "Incl." whole."""


def _words(text: str) -> str:
    """Lower-case letters only, one space between words: ``"Shop Charges:"`` -> ``shop charges``."""
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def total_field(text: str) -> str | None:
    """The :class:`SheetTotals` field a totals-row label fills, or None."""
    return _TOTAL_LABELS.get(_words(text))


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


@dataclass(frozen=True)
class Sheet:
    """One parsed page: its lines, its totals block and where a Declined section starts."""

    lines: tuple[MenuLine, ...]
    totals: SheetTotals
    layout: Layout
    """``priced``: a price on every line row; ``bullets``: bullet lines, prices in the totals."""
    declined_from: int | None = None
    """Index of the first line under a Declined header, or None."""

    @property
    def declined(self) -> tuple[MenuLine, ...]:
        return tuple(line for line in self.lines if line.declined_on_sheet)


def _fragments(response: Mapping[str, Any]) -> list[_Frag]:
    """LINE blocks -> fragments; standalone prices paired with the text on their row."""
    blocks = response.get("Blocks") or []
    by_id = {b["Id"]: b for b in blocks if "Id" in b}
    texts: list[_Frag] = []
    prices: list[_Frag] = []
    for block in blocks:
        if block.get("BlockType") != "LINE":
            continue
        raw = str(block.get("Text", ""))
        if not clean_text(raw):
            continue  # a dotted leader on its own
        box = _bbox(block)
        bullet = _BULLET.match(raw)
        if bullet:
            raw = raw[bullet.end() :]
        cleaned = clean_text(raw)
        if not cleaned:
            texts.append(_Frag("", box, bullet=True))  # a bullet sign read as its own block
            continue
        name, price, description = _split(cleaned)
        if price and not name:
            prices.append(_Frag("", box, price, box))
        elif price:
            name_box, price_box = _word_boxes(block, by_id)
            texts.append(
                _Frag(
                    name,
                    name_box or box,
                    price,
                    price_box,
                    description,
                    None,
                    [cleaned],
                    bullet=bool(bullet),
                )
            )
        else:
            texts.append(_Frag(cleaned, box, bullet=bool(bullet), raw=raw))

    # On a bullet sheet a price belongs to a bullet line or a totals row, nothing else.
    bullets = any(t.bullet for t in texts)
    if bullets:
        texts = _merge_bullet_rows(texts)
    # pair each price with the unpriced text on its row, to its left, with the most overlap
    for price in sorted(prices, key=lambda f: f.box.top):
        candidates = [
            t
            for t in texts
            if t.price is None
            and t.box.left < price.box.left
            and _overlap(t.box, price.box) >= ROW_OVERLAP
            and (not bullets or t.bullet or total_field(t.text) is not None)
        ]
        if candidates:
            best = max(candidates, key=lambda t: _overlap(t.box, price.box))
            best.price, best.price_box = price.price, price.box
        else:
            texts.append(price)  # a price whose name could not be read
    return texts


ROW_HEIGHT_RATIO = 3.0
"""Two blocks of one printed row differ in height by less than this factor."""


def _merge_bullet_rows(texts: list[_Frag]) -> list[_Frag]:
    """Join each bullet line with the unpriced text that continues it on the same row.

    OCR splits one printed row into two LINE blocks where a word is blanked out or the gap
    between words is wide ("• PERFORM" + "Crankcase Cleaning Service"). A block continues a
    bullet row when it lies to the right of the bullet block, overlaps it vertically by
    :data:`ROW_OVERLAP` (as a price pairs with its name), is of a similar height, has no price,
    is not itself a bullet and is not a totals label. The pieces are joined left to right with
    one space. Only bullet sheets are merged: priced menus keep name, leader and price apart.
    """
    merged: set[int] = set()
    for frag in sorted((t for t in texts if t.bullet), key=lambda f: f.box.top):
        if frag.price is not None:
            continue
        parts = sorted(
            (
                t
                for t in texts
                if id(t) not in merged
                and not t.bullet
                and t.price is None
                and t.box.left > frag.box.left
                and total_field(t.text) is None
                and _overlap(t.box, frag.box) >= ROW_OVERLAP
                and max(t.box.height, frag.box.height)
                < ROW_HEIGHT_RATIO * max(min(t.box.height, frag.box.height), 1e-9)
            ),
            key=lambda f: f.box.left,
        )
        for part in parts:
            frag.raw = f"{frag.raw} {part.raw}"
            frag.text = clean_text(frag.raw)
            frag.box = frag.box.union(part.box)
            merged.add(id(part))
    return [t for t in texts if id(t) not in merged and (t.text or not t.bullet)]


def _line(index: int, f: _Frag, **extra: Any) -> MenuLine:
    has_name = bool(f.text)
    name_box = f.box.union(f.desc_box) if has_name else f.desc_box
    whole = (name_box or f.price_box or f.box).union(f.price_box)
    price_text = f.price[1] if f.price else None
    return MenuLine(
        index=index,
        text=compose_text(f.text, price_text, f.description),
        name=f.text,
        price=f.price[0] if f.price else None,
        bbox=whole,
        price_text=price_text,
        description=f.description,
        name_bbox=name_box,
        price_bbox=f.price_box,
        **extra,
    )


def _priced_lines(texts: list[_Frag]) -> list[MenuLine]:
    """A menu with a price on each row (the layout every synthetic menu uses)."""
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

    return [_line(i, f) for i, f in enumerate(sorted(items, key=lambda f: f.box.top), 1)]


def _same_row(a: BBox, b: BBox) -> bool:
    """Both boxes sit on one text row: the overlap is most of the TALLER box.

    Stricter than :func:`_overlap`: a tall box (sideways text in the margin, a logo) overlaps
    many rows but shares none of them.
    """
    inter = min(a.bottom, b.bottom) - max(a.top, b.top)
    return max(0.0, inter) / max(a.height, b.height, 1e-9) >= ROW_OVERLAP


def _section_headers(texts: list[_Frag], bullets: list[_Frag]) -> list[_Frag]:
    """Unpriced, non-bullet text alone on its row, between the first and the last bullet."""
    first, last = bullets[0].box.top, bullets[-1].box.top
    return sorted(
        (
            t
            for t in texts
            if not t.bullet
            and t.price is None
            and first < t.box.top < last
            and not any(o is not t and _same_row(o.box, t.box) for o in texts)
        ),
        key=lambda f: f.box.top,
    )


def _bullet_lines(texts: list[_Frag]) -> tuple[list[MenuLine], int | None]:
    """A sheet of bullet lines: the bullets, then the Shop Charges and Fees rows."""
    bullets = sorted((t for t in texts if t.bullet), key=lambda f: f.box.top)
    charges = sorted(
        (
            t
            for t in texts
            if not t.bullet and t.price is not None and total_field(t.text) in CHARGE_FIELDS
        ),
        key=lambda f: f.box.top,
    )
    prices_listed = any(b.price is not None for b in bullets)
    headers = _section_headers(texts, bullets)
    lines: list[MenuLine] = []
    declined_from: int | None = None
    for i, frag in enumerate(bullets, 1):
        above = [h for h in headers if h.box.top < frag.box.top]
        declined = bool(above) and _words(above[-1].text) == DECLINED_HEADER
        if declined and declined_from is None:
            declined_from = i
        lines.append(_line(i, frag, declined_on_sheet=declined, prices_listed=prices_listed))
    start = len(lines)
    lines += [_line(start + i, frag) for i, frag in enumerate(charges, 1)]
    return lines, declined_from


def _totals(texts: list[_Frag]) -> SheetTotals:
    """The first priced row for each totals label, top to bottom; bullet lines never count."""
    found: dict[str, float] = {}
    for frag in sorted(texts, key=lambda f: f.box.top):
        key = None if frag.bullet or frag.price is None else total_field(frag.text)
        if key is not None and key not in found:
            found[key] = frag.price[0]  # type: ignore[index]
    return SheetTotals(**found)


def parse_sheet(response: Mapping[str, Any]) -> Sheet:
    """Amazon Textract ``DetectDocumentText`` JSON -> lines, totals and the Declined section.

    The totals block is read on both layouts (on a priced menu its rows may also be lines, as
    they always were). A badge such as "up to $6,000" or a "$63.21 / month" financing offer
    has no totals label and is neither a line on a bullet sheet nor a total.
    """
    texts = _fragments(response)
    totals = _totals(texts)
    if not any(t.bullet for t in texts):
        return Sheet(tuple(_priced_lines(texts)), totals, "priced")
    lines, declined_from = _bullet_lines(texts)
    return Sheet(tuple(lines), totals, "bullets", declined_from)


def parse_textract(response: Mapping[str, Any]) -> list[MenuLine]:
    """Amazon Textract ``DetectDocumentText`` JSON -> menu lines, top to bottom.

    The lines of :func:`parse_sheet`; kept for callers that need only the lines.
    """
    return list(parse_sheet(response).lines)


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
