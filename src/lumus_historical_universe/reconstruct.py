"""Point-in-time S&P 500 membership reconstruction."""

from __future__ import annotations

import csv
import warnings
from datetime import date, timedelta
from pathlib import Path

from .normalize import normalize_ticker, parse_date
from .sources import BASE_COLUMNS, CHANGE_COLUMNS, ingest_wikipedia, project_root

HISTORICAL_COLUMNS = ["date", "ticker", "is_member"]
QUALITY_COLUMNS = [
    "date",
    "constituent_count",
    "expected_count",
    "missing_count",
    "duplicate_count",
    "source_coverage",
    "quality_flag",
    "notes",
]
SUPPORTED_MINIMUM_DATE = date(2010, 1, 1)
EXPECTED_COUNT = 503


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_processed_data(root: Path | None = None, allow_fetch: bool = True) -> None:
    """Ensure base/change CSVs exist, optionally fetching public sources."""
    root = root or project_root()
    base = root / "data" / "processed" / "base_constituents.csv"
    changes = root / "data" / "processed" / "membership_changes.csv"
    if allow_fetch and (not base.exists() or not changes.exists()):
        ingest_wikipedia(root=root)
    else:
        if not base.exists():
            _write_csv(base, BASE_COLUMNS, [])
        if not changes.exists():
            _write_csv(changes, CHANGE_COLUMNS, [])


def _load_base_and_changes(root: Path | None = None) -> tuple[date, set[str], list[dict[str, str]]]:
    root = root or project_root()
    ensure_processed_data(root=root)
    base_rows = _read_csv(root / "data" / "processed" / "base_constituents.csv")
    change_rows = _read_csv(root / "data" / "processed" / "membership_changes.csv")
    if not base_rows:
        raise RuntimeError(
            "No base constituents are available. Run the build command in an environment "
            "that can access public source data, or provide data/processed/base_constituents.csv."
        )
    anchor = max(parse_date(r["anchor_date"]) for r in base_rows if r.get("anchor_date"))
    members = {normalize_ticker(r.get("ticker") or r.get("normalized_ticker")) for r in base_rows}
    members.discard("")
    return anchor, members, change_rows


def _quality_for(date_value: date, members: list[str], changes: list[dict[str, str]]) -> dict[str, object]:
    unique = set(members)
    duplicate_count = len(members) - len(unique)
    count = len(unique)
    missing_count = max(EXPECTED_COUNT - count, 0)
    first_change = min((parse_date(r["effective_date"]) for r in changes if r.get("effective_date")), default=None)
    if first_change is None:
        coverage = "anchor_only"
        flag = "low"
        notes = "No change events available; result is current anchor membership only."
    elif date_value < first_change:
        coverage = "below_change_history"
        flag = "low"
        notes = f"Requested date predates first available change event ({first_change})."
    elif duplicate_count:
        coverage = "changes_available"
        flag = "review"
        notes = "Duplicate tickers detected after reconstruction."
    elif 490 <= count <= 510:
        coverage = "changes_available"
        flag = "ok"
        notes = "Constituent count is within approximate S&P 500 share-class range."
    else:
        coverage = "changes_available"
        flag = "review"
        notes = "Constituent count is outside the expected approximate range."
    return {
        "date": date_value.isoformat(),
        "constituent_count": count,
        "expected_count": EXPECTED_COUNT,
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "source_coverage": coverage,
        "quality_flag": flag,
        "notes": notes,
    }


def get_sp500_members(date: str, root: Path | None = None, warn_on_low_quality: bool = True) -> list[str]:
    """Return reconstructed normalized S&P 500 tickers as of ``date``.

    The method starts from the frozen anchor constituent list and reverses every
    known addition/removal effective after the requested date. Dates before the
    Phase-1 supported minimum or after the anchor date raise clear errors.
    """
    target = parse_date(date)
    if target < SUPPORTED_MINIMUM_DATE:
        raise ValueError(f"{target} is outside supported range; earliest supported date is {SUPPORTED_MINIMUM_DATE}")
    anchor, members, changes = _load_base_and_changes(root=root)
    if target > anchor:
        raise ValueError(f"{target} is outside supported range; latest anchor date is {anchor}")

    dated_changes = [r for r in changes if r.get("effective_date") and parse_date(r["effective_date"]) > target]
    dated_changes.sort(key=lambda r: parse_date(r["effective_date"]), reverse=True)
    for row in dated_changes:
        added = normalize_ticker(row.get("added_ticker"))
        removed = normalize_ticker(row.get("removed_ticker"))
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
    result = sorted(members)
    quality = _quality_for(target, result, changes)
    if warn_on_low_quality and quality["quality_flag"] in {"low", "review"}:
        warnings.warn(
            f"S&P 500 reconstruction quality for {target}: {quality['quality_flag']} - {quality['notes']}",
            RuntimeWarning,
            stacklevel=2,
        )
    return result



def get_sp500_quality(date: str, root: Path | None = None) -> dict[str, object]:
    """Return reconstruction quality metadata for ``date`` without emitting warnings."""
    target = parse_date(date)
    _, members, changes = _load_base_and_changes(root=root)
    dated_changes = [r for r in changes if r.get("effective_date") and parse_date(r["effective_date"]) > target]
    dated_changes.sort(key=lambda r: parse_date(r["effective_date"]), reverse=True)
    for row in dated_changes:
        added = normalize_ticker(row.get("added_ticker"))
        removed = normalize_ticker(row.get("removed_ticker"))
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
    return _quality_for(target, sorted(members), changes)

def reconstruct_sp500_history(start_date: str, end_date: str, root: Path | None = None) -> list[dict[str, object]]:
    """Build daily ``date,ticker,is_member`` rows and quality artifacts."""
    root = root or project_root()
    start = parse_date(start_date)
    end = parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    historical_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    _, _, changes = _load_base_and_changes(root=root)
    day = start
    while day <= end:
        members = get_sp500_members(day.isoformat(), root=root, warn_on_low_quality=False)
        historical_rows.extend({"date": day.isoformat(), "ticker": ticker, "is_member": 1} for ticker in members)
        quality_rows.append(_quality_for(day, members, changes))
        day += timedelta(days=1)
    _write_csv(root / "data" / "processed" / "historical_constituents_daily.csv", HISTORICAL_COLUMNS, historical_rows)
    _write_csv(root / "artifacts" / "reconstruction_quality.csv", QUALITY_COLUMNS, quality_rows)
    return quality_rows
