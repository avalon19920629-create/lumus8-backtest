"""Ticker and date normalization helpers."""

from __future__ import annotations

from datetime import date, datetime
import re

_VALID_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.-]*$")


def parse_date(value: str | date) -> date:
    """Parse an ISO ``YYYY-MM-DD`` date or return an existing date."""
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError("date must be a YYYY-MM-DD string")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def normalize_ticker(ticker: str | None) -> str:
    """Normalize source tickers to a Yahoo-compatible canonical form.

    Examples: ``BRK.B`` -> ``BRK-B`` and ``bf.b`` -> ``BF-B``.
    Blank/unknown values remain blank so ingestion never invents missing tickers.
    """
    if ticker is None:
        return ""
    value = str(ticker).strip().upper()
    if not value or value in {"N/A", "NA", "NONE", "—", "-"}:
        return ""
    value = value.replace("\u00a0", " ").split()[0]
    value = value.replace("/", "-").replace(".", "-")
    value = re.sub(r"[^A-Z0-9-]", "", value)
    return value


def ticker_is_normalized(ticker: str) -> bool:
    """Return whether a ticker already matches the canonical representation."""
    return ticker == normalize_ticker(ticker) and bool(_VALID_TICKER.match(ticker))
