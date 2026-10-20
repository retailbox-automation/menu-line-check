"""The keyword baseline: regex rules on the line text, no model. The skeptic's comparison.

The rules are the ones of the spike of 2026-10-20 (``spike/n2-menu/run_n2_spike.py``,
``kw_tool``), checked in this order; the first match wins.
"""

from __future__ import annotations

import re

KEYWORD_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"brake|tire|alignment|tie rod|steering"), "owner_decides"),
    (re.compile(r"fee|supplies|charge|hazmat"), "ask_about_fee"),
    (re.compile(r"diagnos|replace|sensor|pump|software|update"), "check_recall_warranty"),
    (re.compile(r"oil &|oil and|filter|rotation|fluid|coolant|spark|inspection"), "check_schedule"),
)
DEFAULT_TOOL = "decline_addon"


def keyword_tool(text: str) -> str:
    """The tool the keyword rules pick for one line of text (lower-cased first)."""
    lowered = text.lower()
    for pattern, tool in KEYWORD_RULES:
        if pattern.search(lowered):
            return tool
    return DEFAULT_TOOL
