"""Menu parsing from Textract JSON and from per-line OCR text; line crops."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from menu_line_check.models import BBox, MenuLine, Vehicle
from menu_line_check.ocr import (
    crop_line,
    parse_line_texts,
    parse_price,
    parse_textract,
    textract_detect,
)

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = [
    ("Oil & filter change, full synthetic", 89.95),
    ("Engine oil conditioner", 39.0),
    ("Fuel injection cleaning", 149.0),
    ("Tire rotation", 39.95),
    ("Cabin air filter", 79.95),
    ("Replace fuel pump module", 1089.0),
    ("Tech diagnostic", 189.0),
    ("Shop supplies", 34.5),
    ("Front brake pads & rotors", 489.0),
    ("Nitrogen tire fill", 29.95),
    ("Throttle body service", 119.0),
    ("Environmental disposal fee", 12.95),
]


def textract_fixture() -> dict:
    return json.loads((FIXTURES / "textract_menu.json").read_text(encoding="utf-8"))


def line_block(text, left, top, width=0.2, height=0.024):
    return {
        "BlockType": "LINE",
        "Text": text,
        "Confidence": 99.0,
        "Geometry": {"BoundingBox": {"Left": left, "Top": top, "Width": width, "Height": height}},
    }


# ---------------------------------------------------------------------------- Textract


def test_fixture_pairs_every_name_with_its_price_in_order():
    lines = parse_textract(textract_fixture())
    assert [(ln.name, ln.price) for ln in lines] == EXPECTED
    assert [ln.index for ln in lines] == list(range(1, 13))


def test_header_without_a_price_is_skipped():
    names = [ln.name for ln in parse_textract(textract_fixture())]
    assert not any("Recommended service" in n for n in names)


def test_description_line_is_attached_to_the_name_above():
    lines = parse_textract(textract_fixture())
    diag = lines[6]
    assert diag.description == "Required before any repair"
    assert diag.text == "Tech diagnostic ... $189.00 Required before any repair"
    assert all(ln.description is None for ln in lines if ln is not diag)


def test_leaders_are_dropped_and_glued_or_one_line_rows_are_split():
    lines = parse_textract(textract_fixture())
    assert lines[7].text == "Shop supplies ... $34.50"
    assert lines[11].name == "Environmental disposal fee" and lines[11].price_text == "$12.95"
    assert lines[5].price_text == "$1,089.00" and lines[5].price == 1089.0


def test_bounding_boxes_are_kept_for_cropping():
    oil = parse_textract(textract_fixture())[0]
    assert isinstance(oil.bbox, BBox) and oil.name_bbox and oil.price_bbox
    assert oil.bbox.left == pytest.approx(0.08) and oil.bbox.right == pytest.approx(0.92)
    assert oil.price_bbox.left > 0.8 and oil.name_bbox.right < 0.5


def test_no_charge_price_and_a_middle_line_without_price_and_a_price_without_name():
    response = {
        "Blocks": [
            line_block("Service menu", 0.08, 0.02),
            line_block("Multi-point inspection", 0.08, 0.10),
            line_block("$0.00 (no charge)", 0.75, 0.101, width=0.17),
            line_block("Wiper blades", 0.08, 0.16),  # price cut off
            line_block("Cabin air filter", 0.08, 0.22),
            line_block("$79.95", 0.85, 0.22, width=0.07),
            line_block("$64.95", 0.85, 0.28, width=0.07),  # name unreadable
        ]
    }
    lines = parse_textract(response)
    assert [(ln.name, ln.price) for ln in lines] == [
        ("Multi-point inspection", 0.0),
        ("Wiper blades", None),
        ("Cabin air filter", 79.95),
        ("", 64.95),
    ]
    assert lines[0].price_text == "$0.00 (no charge)"
    assert [ln.has_name_and_price for ln in lines] == [True, False, True, False]


def test_textract_detect_is_a_thin_wrapper_over_the_client_passed_in():
    calls = []

    class FakeTextract:
        def detect_document_text(self, **kwargs):
            calls.append(kwargs)
            return textract_fixture()

    response = textract_detect(b"jpeg-bytes", FakeTextract())
    assert calls == [{"Document": {"Bytes": b"jpeg-bytes"}}]
    assert len(parse_textract(response)) == 12


# ---------------------------------------------------------------------------- plain OCR text


@pytest.mark.parametrize(
    "row,name,price,desc",
    [
        ("Engine oil conditioner ... $39.00", "Engine oil conditioner", 39.0, None),
        ("$189.00 Airbag warning light diagnosis", "Airbag warning light diagnosis", 189.0, None),
        ("S0.00 (no charge) Multi-point inspection", "Multi-point inspection", 0.0, None),
        ("Multi-point inspection $0.00 (no charge)", "Multi-point inspection", 0.0, None),
        (
            "Throttle body service $119.00 Recommended every 15,000 mi",
            "Throttle body service",
            119.0,
            "Recommended every 15,000 mi",
        ),
        (
            "Inner tie rod ends .••• $356.00 Steering play noted",
            "Inner tie rod ends",
            356.0,
            "Steering play noted",
        ),
        ("$34.50 Shop supplies.", "Shop supplies", 34.5, None),
        ("Replace fuel pump module $1,089.00", "Replace fuel pump module", 1089.0, None),
        ("Fuel injection cleaning ...  $149.00", "Fuel injection cleaning", 149.0, None),
    ],
)
def test_parse_line_texts_on_spike_style_ocr(row, name, price, desc):
    (line,) = parse_line_texts([row])
    assert (line.name, line.price, line.description) == (name, price, desc)
    assert line.text == row and line.bbox is None and line.has_name_and_price


@pytest.mark.parametrize("row", ["", "S0", "00", "Wiper blades"])
def test_unreadable_rows_have_no_name_and_price(row):
    (line,) = parse_line_texts([row])
    assert not line.has_name_and_price


def test_parse_line_texts_numbers_lines_from_one():
    lines = parse_line_texts(["Tire rotation $39.95", "Shop supplies $34.50"])
    assert [ln.index for ln in lines] == [1, 2]


def test_parse_price_forms():
    assert parse_price("$1,089.00") == (1089.0, "$1,089.00")
    assert parse_price("$29.95") == (29.95, "$29.95")
    assert parse_price("$0.00 (no charge)") == (0.0, "$0.00 (no charge)")
    assert parse_price("every 15,000 mi") is None
    assert parse_price("S60 wiper") is None


# ---------------------------------------------------------------------------- crops


def photo_bytes(w=1000, h=1400) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "JPEG")
    return buf.getvalue()


def decode(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    return Image.open(io.BytesIO(raw))


def test_strip_crop_is_one_full_width_png_band():
    line = parse_textract(textract_fixture())[0]
    (strip,) = crop_line(photo_bytes(), line, "strip")
    img = decode(strip)
    assert img.format == "PNG" and img.width == 1000 and 30 < img.height < 80


def test_cells_crop_is_name_then_price():
    line = parse_textract(textract_fixture())[6]  # has a description line
    name_png, price_png = crop_line(photo_bytes(), line, "cells")
    name_img, price_img = decode(name_png), decode(price_png)
    assert name_img.width > price_img.width and name_img.height > price_img.height


def test_crop_needs_geometry_and_a_known_mode():
    (no_box,) = parse_line_texts(["Tire rotation $39.95"])
    with pytest.raises(ValueError, match="bounding box"):
        crop_line(photo_bytes(), no_box, "strip")
    with pytest.raises(ValueError, match="mode"):
        crop_line(photo_bytes(), parse_textract(textract_fixture())[0], "mosaic")


# ---------------------------------------------------------------------------- models


def test_vehicle_parse_and_label():
    v = Vehicle.parse("2019 Honda Civic 31200")
    assert v == Vehicle(2019, "Honda", "Civic", 31200)
    assert v.label == "2019 Honda Civic, 31,200 mi"
    assert Vehicle.parse("2021 Toyota Grand Highlander 12,400 mi") == Vehicle(
        2021, "Toyota", "Grand Highlander", 12400
    )
    with pytest.raises(ValueError, match="year make model odometer"):
        Vehicle.parse("Honda Civic")


def test_menu_line_is_a_frozen_value():
    line = MenuLine(index=1, text="x $1.00", name="x", price=1.0, bbox=None)
    assert line.images == ()
    with pytest.raises(AttributeError):
        line.name = "y"  # type: ignore[misc]
