"""A hard USD cap on every paid AWS call the agent makes (Amazon Bedrock, Amazon Textract).

* :data:`BEDROCK_PRICES_PER_M` / :func:`bedrock_price` - the price table (used by ``menu-race``).
* :class:`SpendCap` - a running total, checked before each call and added to after it. One cap
  is shared by the Bedrock picker, the Bedrock writer and the Textract call of a run.
* :class:`SpendCapReached` - raised when the next call is not allowed; nothing swallows it.

Used by :class:`~menu_line_check.llm.StrandsPicker` and :class:`~menu_line_check.llm.BedrockWriter`,
which share one ``SpendCap`` per run (``menu-check --max-usd`` / ``menu-race --max-usd``).
"""

from __future__ import annotations

import os

# USD per 1M tokens (input, output), us-east-1 on-demand standard tier, read from the AWS Price List
# API (service AmazonBedrock, usage types USE1-Nova*-input/output-tokens) on 2026-10-21. A model id
# that is not listed prints "price not set".
BEDROCK_PRICES_PER_M: dict[str, tuple[float, float]] = {
    "us.amazon.nova-micro-v1:0": (0.035, 0.14),
    "us.amazon.nova-lite-v1:0": (0.06, 0.24),
    "us.amazon.nova-2-lite-v1:0": (0.33, 2.75),
}

# USD per page, Amazon Textract DetectDocumentText ($1.50 per 1,000 pages), us-east-1, read from
# the AWS Price List API on 2026-10-21.
TEXTRACT_USD_PER_PAGE = 0.0015

DEFAULT_MAX_USD = 1.0
MAX_USD_ENV = "N2_MAX_USD"


def bedrock_price(model_id: str) -> tuple[float, float] | None:
    return BEDROCK_PRICES_PER_M.get(model_id)


class SpendCapReached(RuntimeError):  # noqa: N818 - reads as a sentence at the raise site
    """The next Bedrock call is not allowed: the spend cap is reached, or the model is unpriced."""


class SpendCap:
    """Running USD total for Bedrock (and Textract) calls, with a hard stop.

    Call :meth:`check` before every Bedrock call and :meth:`add` after it with the token usage
    the response reports. The guarantee is honest, not exact: the cap is checked BEFORE a call,
    so a run can exceed it by at most the cost of one call (the one that crosses it). For the
    picker a "call" is one agent run, which may make a few model round-trips; the writer's call is
    one Converse request. A call that fails before reporting usage is not counted. A model
    id with no price in the table cannot be costed, so its calls are refused outright.

    ``max_usd`` defaults to env ``N2_MAX_USD``, else :data:`DEFAULT_MAX_USD`.
    """

    def __init__(self, max_usd: float | None = None) -> None:
        if max_usd is None:
            raw = os.environ.get(MAX_USD_ENV)
            try:
                max_usd = float(raw) if raw else DEFAULT_MAX_USD
            except ValueError as exc:
                raise ValueError(f"{MAX_USD_ENV} must be a number, got {raw!r}") from exc
        self.max_usd = max_usd
        self.usd = 0.0
        self.calls = 0
        self.textract_pages = 0

    def check(self, model_id: str) -> None:
        """Raise :class:`SpendCapReached` unless one more call to ``model_id`` is allowed."""
        if bedrock_price(model_id) is None:
            raise SpendCapReached(
                f"no price for Bedrock model {model_id!r}, so its spend cannot be capped: "
                "add it to BEDROCK_PRICES_PER_M (menu_line_check/spend.py) or use a priced model."
            )
        self._check_room()

    def check_textract(self) -> None:
        """Raise :class:`SpendCapReached` unless one more Textract page is allowed."""
        self._check_room()

    def add_textract(self, pages: int = 1) -> None:
        """Count finished Textract pages at :data:`TEXTRACT_USD_PER_PAGE` each."""
        self.usd += pages * TEXTRACT_USD_PER_PAGE
        self.textract_pages += pages
        self.calls += 1

    def _check_room(self) -> None:
        if self.usd >= self.max_usd:
            raise SpendCapReached(
                f"AWS spend cap reached: ${self.usd:.4f} of ${self.max_usd:.2f} "
                f"after {self.calls} calls; stopping (raise it with --max-usd or {MAX_USD_ENV})."
            )

    def add(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        """Count one finished call. A model without a price adds nothing (``check`` refuses it)."""
        price = bedrock_price(model_id)
        if price is not None:
            self.usd += (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
        self.calls += 1

    @property
    def summary(self) -> str:
        label = "AWS spend (Bedrock + Textract)" if self.textract_pages else "Bedrock spend"
        return f"{label}: ${self.usd:.4f} of ${self.max_usd:.2f} cap, {self.calls} calls"


__all__ = [
    "BEDROCK_PRICES_PER_M",
    "DEFAULT_MAX_USD",
    "SpendCap",
    "SpendCapReached",
    "TEXTRACT_USD_PER_PAGE",
    "bedrock_price",
]
