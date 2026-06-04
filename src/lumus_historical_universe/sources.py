"""Source ingestion helpers for S&P 500 universe reconstruction."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from .normalize import normalize_ticker, parse_date

BASE_COLUMNS = [
    "anchor_date",
    "ticker",
    "security",
    "gics_sector",
    "gics_sub_industry",
    "source",
    "source_url",
    "verified_at",
    "original_ticker",
    "normalized_ticker",
]
CHANGE_COLUMNS = [
    "effective_date",
    "added_ticker",
    "added_security",
    "removed_ticker",
    "removed_security",
    "reason",
    "source",
    "source_url",
    "verified_at",
    "confidence",
    "notes",
    "added_original_ticker",
    "removed_original_ticker",
]
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def project_root() -> Path:
    """Return the repository root when using the default src layout."""
    return Path(__file__).resolve().parents[2]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cell(row: object, *keys: object) -> object:
    for key in keys:
        try:
            value = row[key]
        except (KeyError, TypeError):
            continue
        if value == value:
            return value
    return ""


def ingest_wikipedia(root: Path | None = None, anchor_date: date | None = None) -> None:
    """Fetch current constituents and change history from Wikipedia into processed CSVs."""
    import pandas as pd

    root = root or project_root()
    anchor = (anchor_date or date.today()).isoformat()
    tables = pd.read_html(WIKIPEDIA_SP500_URL)
    constituents = tables[0]
    changes = tables[1] if len(tables) > 1 else pd.DataFrame()

    base_rows: list[dict[str, object]] = []
    for _, row in constituents.iterrows():
        ticker = normalize_ticker(_cell(row, "Symbol"))
        if not ticker:
            continue
        base_rows.append(
            {
                "anchor_date": anchor,
                "ticker": ticker,
                "security": _cell(row, "Security"),
                "gics_sector": _cell(row, "GICS Sector"),
                "gics_sub_industry": _cell(row, "GICS Sub-Industry"),
                "source": "Wikipedia current S&P 500 constituents",
                "source_url": WIKIPEDIA_SP500_URL,
                "verified_at": anchor,
                "original_ticker": _cell(row, "Symbol"),
                "normalized_ticker": ticker,
            }
        )

    change_rows: list[dict[str, object]] = []
    for _, row in changes.iterrows():
        effective = _cell(row, ("Date", "Date"), "Date")
        try:
            effective_date = parse_date(effective).isoformat()
        except (TypeError, ValueError):
            continue
        added_ticker = normalize_ticker(_cell(row, ("Added", "Ticker"), "Added Ticker", "Added ticker"))
        removed_ticker = normalize_ticker(_cell(row, ("Removed", "Ticker"), "Removed Ticker", "Removed ticker"))
        change_rows.append(
            {
                "effective_date": effective_date,
                "added_ticker": added_ticker,
                "added_security": _cell(row, ("Added", "Security"), "Added Security", "Added security"),
                "removed_ticker": removed_ticker,
                "removed_security": _cell(row, ("Removed", "Security"), "Removed Security", "Removed security"),
                "reason": _cell(row, ("Reason", "Reason"), "Reason"),
                "source": "Wikipedia selected S&P 500 component changes",
                "source_url": WIKIPEDIA_SP500_URL,
                "verified_at": anchor,
                "confidence": "medium",
                "notes": "Fetched from Wikipedia selected changes table.",
                "added_original_ticker": _cell(row, ("Added", "Ticker"), "Added Ticker", "Added ticker"),
                "removed_original_ticker": _cell(row, ("Removed", "Ticker"), "Removed Ticker", "Removed ticker"),
            }
        )

    _write_csv(root / "data" / "processed" / "base_constituents.csv", BASE_COLUMNS, base_rows)
    _write_csv(root / "data" / "processed" / "membership_changes.csv", CHANGE_COLUMNS, change_rows)
