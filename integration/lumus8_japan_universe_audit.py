"""Audit the static Japanese equity universe used by L.U.M.U.S.-8.

This script does not alter strategy logic.  It extracts the current JP ticker list
from ``integration/lumus8_current_screening.py`` and writes audit-only artifacts
that document metadata gaps, current-index source limitations, and likely bias.
"""

from __future__ import annotations

import ast
import csv
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LUMUS_SCRIPT = REPO_ROOT / "integration" / "lumus8_current_screening.py"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
ACCESSED_AT = date.today().isoformat()

JP_TICKER_SOURCE = "integration/lumus8_current_screening.py:get_tickers_lumus():jp_tickers"

SOURCE_MANIFEST_COLUMNS = [
    "source_name", "source_url", "accessed_at", "fields_used", "license_terms_uncertain", "known_limitations",
]


def extract_jp_tickers() -> list[str]:
    """Extract the first ``jp_tickers`` list literal from the L.U.M.U.S. script."""
    tree = ast.parse(LUMUS_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "jp_tickers":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, list):
                        return [str(item) for item in value]
    raise RuntimeError(f"jp_tickers list not found in {LUMUS_SCRIPT}")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_lumus_universe_rows(tickers: list[str]) -> list[dict[str, str]]:
    return [
        {
            "ticker": ticker,
            "normalized_ticker": ticker,
            "company_name": "",
            "source_in_code": JP_TICKER_SOURCE,
            "notes": "Company name not fetched in this audit; ticker is taken directly from the static manual code list.",
        }
        for ticker in tickers
    ]


def build_profile_rows(tickers: list[str]) -> list[dict[str, str]]:
    return [
        {
            "ticker": ticker,
            "company_name": "",
            "sector": "unknown",
            "industry": "unknown",
            "market_segment": "unknown",
            "market_cap": "",
            "market_cap_bucket": "unknown",
            "liquidity_proxy": "",
            "source": JP_TICKER_SOURCE,
            "notes": "Metadata not fabricated. Public metadata/index source access was insufficient for reliable per-ticker fields in this phase.",
        }
        for ticker in tickers
    ]


def build_source_manifest_rows() -> list[dict[str, str]]:
    return [
        {
            "source_name": "L.U.M.U.S.-8 local JP ticker list",
            "source_url": "integration/lumus8_current_screening.py",
            "accessed_at": ACCESSED_AT,
            "fields_used": "ticker",
            "license_terms_uncertain": "false",
            "known_limitations": "Manual static current-only list; no point-in-time membership, company metadata, market segment, liquidity, or delisting history.",
        },
        {
            "source_name": "JPX TOPIX official index page",
            "source_url": "https://www.jpx.co.jp/english/markets/indices/topix/index.html",
            "accessed_at": ACCESSED_AT,
            "fields_used": "source investigated only",
            "license_terms_uncertain": "true",
            "known_limitations": "Current-only official index reference; full constituent/download data was not incorporated into this local audit artifact.",
        },
        {
            "source_name": "JPX Prime 150 Index official page",
            "source_url": "https://www.jpx.co.jp/english/markets/indices/jpx-prime150/",
            "accessed_at": ACCESSED_AT,
            "fields_used": "source investigated only",
            "license_terms_uncertain": "true",
            "known_limitations": "Current-only index reference; constituent list with weights requires separate official download/validation before overlap can be trusted.",
        },
        {
            "source_name": "JPX TOPIX Core30 factsheet/reference pages",
            "source_url": "https://www.jpx.co.jp/english/markets/indices/line-up/index.html",
            "accessed_at": ACCESSED_AT,
            "fields_used": "source investigated only",
            "license_terms_uncertain": "true",
            "known_limitations": "Official index family reference; this audit did not import a validated member file, so overlap is marked unavailable.",
        },
        {
            "source_name": "Nikkei 225 official component/profile pages",
            "source_url": "https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225",
            "accessed_at": ACCESSED_AT,
            "fields_used": "source investigated only",
            "license_terms_uncertain": "true",
            "known_limitations": "Current-only component/profile reference; not historical truth and not imported as a validated constituent file in this phase.",
        },
        {
            "source_name": "JPX listed/delisted company information pages",
            "source_url": "https://www.jpx.co.jp/english/listing/stocks/delisted/index.html",
            "accessed_at": ACCESSED_AT,
            "fields_used": "source investigated only",
            "license_terms_uncertain": "true",
            "known_limitations": "Useful for future delisting research; this phase records qualitative risk rather than reconstructing historical JP membership.",
        },
    ]


def build_index_overlap_rows(lumus_count: int) -> list[dict[str, Any]]:
    indexes = [
        ("Nikkei 225", "https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225"),
        ("JPX Prime 150", "https://www.jpx.co.jp/english/markets/indices/jpx-prime150/"),
        ("TOPIX 100", "https://www.jpx.co.jp/english/markets/indices/line-up/index.html"),
        ("TOPIX Core30", "https://www.jpx.co.jp/english/markets/indices/line-up/index.html"),
    ]
    return [
        {
            "index_name": name,
            "index_constituent_count": "",
            "lumus_overlap_count": "",
            "lumus_overlap_ratio": "",
            "index_overlap_ratio": "",
            "missing_from_lumus_count": "",
            "extra_in_lumus_count": lumus_count,
            "source": source,
            "notes": "Schema-only overlap row: no validated current constituent member file was imported; do not infer overlap from this row.",
        }
        for name, source in indexes
    ]


def build_index_overlap_by_ticker_rows(tickers: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "ticker": ticker,
            "company_name": "",
            "in_lumus": 1,
            "in_nikkei225": "unknown",
            "in_jpx_prime150": "unknown",
            "in_topix100": "unknown",
            "in_topix_core30": "unknown",
            "notes": "Index membership not asserted because validated current index constituent files were not imported.",
        }
        for ticker in tickers
    ]


def build_sector_concentration_rows(tickers: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "sector": "unknown",
            "ticker_count": len(tickers),
            "universe_weight_by_count": 1.0,
            "notes": "Unknown-sector ratio is 100%; sector concentration cannot be measured without validated issuer metadata.",
        }
    ]


def build_survivorship_risk_rows() -> list[dict[str, str]]:
    return [
        {
            "risk_item": "Static surviving large-cap list",
            "description": "The JP universe is a current manual list of live .T tickers and is not point-in-time. Delisted, acquired, bankrupt, and reorganized firms are structurally absent.",
            "severity": "high",
            "evidence_source": JP_TICKER_SOURCE,
            "notes": "This is the central audit finding; the list is investable today but not a valid historical universe by itself.",
        },
        {
            "risk_item": "Toshiba-style disappearance risk",
            "description": "Major former Japanese listed companies can disappear from public equity universes after take-private or restructuring events; current-only lists omit those historical paths.",
            "severity": "high",
            "evidence_source": "JPX delisted company information / public Toshiba delisting coverage",
            "notes": "Use as qualitative evidence only; do not treat this audit as a full delisting reconstruction.",
        },
        {
            "risk_item": "Bankruptcy/reorganization omission",
            "description": "Examples such as Takata, Renown, Skymark, Daiei, and Sanyo Electric illustrate that historical losers and reorganizations are missing from a live manual universe.",
            "severity": "high",
            "evidence_source": "Public company event histories and JPX delisting references to investigate in the next phase",
            "notes": "Specific event dates and ticker histories require a dedicated Japanese historical membership/delisting dataset.",
        },
        {
            "risk_item": "M&A disappearance risk",
            "description": "Acquired or merged companies that were once material index or market constituents are omitted unless they still trade under a current ticker.",
            "severity": "medium",
            "evidence_source": "JPX delisted company information pages",
            "notes": "M&A omissions can bias historical tests toward firms that survived as independent listings.",
        },
        {
            "risk_item": "No historical JP index membership",
            "description": "No TOPIX/Nikkei point-in-time membership engine exists in this repository, so rebalance-date JP membership cannot be validated.",
            "severity": "critical",
            "evidence_source": "Repository audit",
            "notes": "This prevents claims that JP-side backtests are free of survivorship bias.",
        },
    ]


def build_bias_rows() -> list[dict[str, str]]:
    return [
        {"bias_dimension": "survivor-only risk", "rating": "high", "reason": "Manual list contains only currently listed tickers.", "evidence": JP_TICKER_SOURCE},
        {"bias_dimension": "large-cap bias", "rating": "high", "reason": "Ticker set is dominated by famous Japanese blue chips and liquid megacaps by construction/appearance, but market-cap metadata was not validated.", "evidence": "Static ticker composition plus missing small-cap/delisted coverage."},
        {"bias_dimension": "quality bias", "rating": "high", "reason": "Manual curation likely favors surviving high-quality firms and excludes historical failures.", "evidence": "Survivorship risk rows and absence of delisted constituents."},
        {"bias_dimension": "liquidity bias", "rating": "medium", "reason": "The list appears to prefer liquid well-known names, but liquidity proxies were not fetched.", "evidence": "Manual ticker list; profile liquidity_proxy fields unknown."},
        {"bias_dimension": "famous-company/manual-selection bias", "rating": "high", "reason": "Universe definition is a hard-coded curated list rather than a published rule-based index file.", "evidence": JP_TICKER_SOURCE},
        {"bias_dimension": "sector concentration risk", "rating": "unknown", "reason": "Sector data was not reliably obtained; unknown-sector ratio is 100% in this phase.", "evidence": "artifacts/japan_sector_concentration.csv"},
        {"bias_dimension": "historical delisting omission risk", "rating": "critical", "reason": "There is no JP historical membership/delisting engine; major disappeared firms are absent by design.", "evidence": "artifacts/japan_survivorship_risk.csv"},
        {"bias_dimension": "reproducibility risk", "rating": "medium", "reason": "The list is reproducible as code, but the selection rationale is not documented as an external rule set.", "evidence": JP_TICKER_SOURCE},
        {"bias_dimension": "point-in-time validity risk", "rating": "critical", "reason": "A current manual list cannot be treated as rebalance-date historical truth.", "evidence": "No Japanese historical universe engine present."},
    ]


def write_report(tickers: list[str]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# L.U.M.U.S.-8 Japanese Universe Audit

## 1. Audit purpose

This phase audits only the Japanese equity universe used by L.U.M.U.S.-8. It is **not** a strategy improvement, alpha optimization, or Japanese historical reconstruction.

## 2. Current Japanese universe size

The extracted L.U.M.U.S. Japanese universe contains **{len(tickers)} tickers**.

## 3. How the ticker list is defined in code

The universe is a static Python list named `jp_tickers` inside `get_tickers_lumus()` in `integration/lumus8_current_screening.py`. This makes it reproducible as code, but it is a hand-curated current list rather than a documented published index methodology.

## 4. Basic profile summary

`artifacts/japan_universe_profile.csv` preserves all tickers but leaves company name, sector, industry, market segment, market cap, and liquidity fields unknown where reliable source data was not imported. No missing metadata was invented.

## 5. Sector concentration summary

Sector concentration cannot be measured honestly in this phase because issuer sector metadata was not reliably imported. `artifacts/japan_sector_concentration.csv` therefore reports a 100% unknown-sector ratio.

## 6. Index overlap summary

Nikkei 225, JPX Prime 150, TOPIX 100, and TOPIX Core30 official/current sources were investigated, but validated constituent member files were not imported into this repository audit. The overlap artifacts are preserved with schema-complete rows and `unknown` ticker-level membership flags. These current-only index sources must not be treated as historical truth.

## 7. Survivor-only risk assessment

The Japanese universe has **high survivor-only risk**. It consists only of currently listed `.T` tickers selected manually. Delisted companies, bankruptcies, reorganizations, acquired firms, and former large constituents are absent unless they still survive as current listings.

## 8. Historical loser / delisting omission discussion

Potentially relevant historical omissions include Toshiba-style take-private/disappearance risk, old Japan Airlines restructuring history, Sharp before restructuring, Takata, Renown, Skymark, Sanyo Electric, Daiei, Seiyu, and other formerly important listed companies. This audit does not reconstruct those histories; it flags that such names are structurally missing from a current manual universe.

## 9. Reproducibility assessment

The list is reproducible because it is in code, but the selection rule is not reproducible as a public index methodology. A future auditor can reproduce the tickers but cannot verify why these names, and not other Japanese stocks, were selected.

## 10. Final risk rating

**Final Japanese universe bias risk rating: high**

The rating is high because survivor-only, large-cap/manual-selection, quality, and point-in-time validity risks are material. It is not marked critical only because the list is explicit and reproducible as code, and this audit does not prove the strategy result is invalid; it proves the JP universe is not an unbiased historical universe.

## 11. Recommended next step

Recommended next step: **build Japanese historical universe engine**. If that is not immediately possible, keep the current manual list only with a clear warning, or replace the audit universe with a current rule-based published index such as Nikkei 225 / TOPIX 100 / Core30 for current-only experiments. For institutional-grade historical claims, use paid point-in-time Japanese equity data.

## 12. What must not be concluded yet

- Do not conclude that the JP-side backtest is point-in-time valid.
- Do not conclude that current Nikkei 225 / JPX Prime 150 / TOPIX lists are historical truth.
- Do not conclude that sector exposure is balanced; sector metadata is missing.
- Do not conclude that missing delisted companies had no impact.
- Do not replace or optimize the strategy based on this audit alone.

## Artifacts

- `artifacts/japan_lumus_universe.csv`
- `artifacts/japan_universe_profile.csv`
- `artifacts/japan_index_overlap.csv`
- `artifacts/japan_index_overlap_by_ticker.csv`
- `artifacts/japan_sector_concentration.csv`
- `artifacts/japan_survivorship_risk.csv`
- `artifacts/japan_bias_classification.csv`
- `artifacts/japan_source_manifest.csv`
"""
    (REPORTS_DIR / "japan_universe_audit.md").write_text(report, encoding="utf-8")


def run_audit() -> None:
    tickers = extract_jp_tickers()
    write_csv(ARTIFACTS_DIR / "japan_lumus_universe.csv", ["ticker", "normalized_ticker", "company_name", "source_in_code", "notes"], build_lumus_universe_rows(tickers))
    write_csv(ARTIFACTS_DIR / "japan_universe_profile.csv", ["ticker", "company_name", "sector", "industry", "market_segment", "market_cap", "market_cap_bucket", "liquidity_proxy", "source", "notes"], build_profile_rows(tickers))
    write_csv(ARTIFACTS_DIR / "japan_index_overlap.csv", ["index_name", "index_constituent_count", "lumus_overlap_count", "lumus_overlap_ratio", "index_overlap_ratio", "missing_from_lumus_count", "extra_in_lumus_count", "source", "notes"], build_index_overlap_rows(len(tickers)))
    write_csv(ARTIFACTS_DIR / "japan_index_overlap_by_ticker.csv", ["ticker", "company_name", "in_lumus", "in_nikkei225", "in_jpx_prime150", "in_topix100", "in_topix_core30", "notes"], build_index_overlap_by_ticker_rows(tickers))
    write_csv(ARTIFACTS_DIR / "japan_sector_concentration.csv", ["sector", "ticker_count", "universe_weight_by_count", "notes"], build_sector_concentration_rows(tickers))
    write_csv(ARTIFACTS_DIR / "japan_survivorship_risk.csv", ["risk_item", "description", "severity", "evidence_source", "notes"], build_survivorship_risk_rows())
    write_csv(ARTIFACTS_DIR / "japan_bias_classification.csv", ["bias_dimension", "rating", "reason", "evidence"], build_bias_rows())
    write_csv(ARTIFACTS_DIR / "japan_source_manifest.csv", SOURCE_MANIFEST_COLUMNS, build_source_manifest_rows())
    write_report(tickers)


if __name__ == "__main__":
    run_audit()
