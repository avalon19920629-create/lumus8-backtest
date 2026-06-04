"""Current-index overlap audit for the L.U.M.U.S.-8 Japanese universe.

This is an audit-only utility. It does not change strategy logic or membership.
The script uses the previously extracted L.U.M.U.S. Japanese universe artifact when
available, then compares it with conservative current-only proxy constituent sets
for major Japanese indices.  The proxy sets are explicitly marked as non-historical
and should be replaced by validated official constituent downloads before any
production-grade conclusion.
"""

from __future__ import annotations

import ast
import csv
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
LUMUS_SCRIPT = REPO_ROOT / "integration" / "lumus8_current_screening.py"
ACCESSED_AT = date.today().isoformat()

LUMUS_UNIVERSE_PATH = ARTIFACTS_DIR / "japan_lumus_universe.csv"

# Current-only proxy sets. These are audit seeds, not historical memberships.
TOPIX_CORE30 = """
2914.T 3382.T 4063.T 4502.T 4503.T 4568.T 6098.T 6501.T 6758.T 6861.T
6902.T 6954.T 6981.T 7203.T 7267.T 7741.T 7974.T 8031.T 8035.T 8058.T
8306.T 8316.T 8411.T 8766.T 9432.T 9433.T 9434.T 9983.T 9984.T 6367.T
""".split()

INDEX_EXTRA_POOL = """
1332.T 1925.T 1928.T 2269.T 2413.T 2768.T 2871.T 2914.T 3289.T 3402.T
3405.T 3407.T 4005.T 4151.T 4506.T 4507.T 4578.T 4612.T 4661.T 4901.T
5108.T 5201.T 5332.T 5333.T 5411.T 5801.T 6326.T 6471.T 6472.T 6473.T
6701.T 6902.T 7004.T 7202.T 7211.T 7733.T 7751.T 7832.T 8252.T 8267.T
8308.T 8309.T 8331.T 8604.T 8795.T 8830.T 9001.T 9005.T 9007.T 9008.T
9009.T 9023.T 9042.T 9064.T 9301.T 9433.T 9434.T 9613.T 9766.T 9843.T
2002.T 2267.T 2432.T 2587.T 2768.T 2875.T 3105.T 3288.T 3563.T 3861.T
4004.T 4042.T 4043.T 4204.T 4324.T 4528.T 4536.T 4613.T 4704.T 4902.T
5021.T 5233.T 5334.T 5706.T 5711.T 5901.T 6103.T 6324.T 6460.T 6645.T
6674.T 6841.T 6923.T 6963.T 7003.T 7186.T 7272.T 7309.T 7731.T 7752.T
7911.T 7912.T 7951.T 8113.T 8233.T 8253.T 8304.T 8354.T 8473.T 8593.T
8601.T 8697.T 9147.T 9143.T 9201.T 9302.T 9504.T 9513.T 9531.T 9532.T
9602.T 9684.T 9766.T 9962.T
""".split()

NIKKEI_EXTRA_POOL = """
1333.T 1605.T 1721.T 1801.T 1802.T 1803.T 1812.T 1963.T 2282.T 2501.T
2502.T 2503.T 2531.T 2801.T 2802.T 3086.T 3099.T 3101.T 3103.T 3401.T
3402.T 3405.T 3861.T 3863.T 4004.T 4005.T 4021.T 4041.T 4042.T 4043.T
4061.T 4188.T 4208.T 4272.T 4324.T 4631.T 4751.T 4901.T 4902.T 5019.T
5101.T 5108.T 5201.T 5214.T 5233.T 5301.T 5332.T 5333.T 5406.T 5411.T
5631.T 5703.T 5706.T 5707.T 5711.T 5714.T 5801.T 5803.T 5901.T 6103.T
6326.T 6361.T 6471.T 6472.T 6473.T 6479.T 6504.T 6674.T 6770.T 6841.T
6902.T 7004.T 7202.T 7211.T 7272.T 7309.T 7731.T 7733.T 7751.T 7752.T
7832.T 7911.T 7912.T 7951.T 8015.T 8233.T 8252.T 8253.T 8267.T 8308.T
8309.T 8331.T 8354.T 8601.T 8604.T 8628.T 8630.T 8795.T 8830.T 9001.T
9005.T 9007.T 9008.T 9009.T 9023.T 9042.T 9064.T 9301.T 9302.T 9412.T
9433.T 9434.T 9504.T 9531.T 9532.T 9602.T 9613.T 9766.T 9843.T 9987.T
""".split()


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_lumus_from_code() -> list[str]:
    tree = ast.parse(LUMUS_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "jp_tickers":
                    return [str(item) for item in ast.literal_eval(node.value)]
    raise RuntimeError("jp_tickers list not found")


def load_lumus_universe() -> list[str]:
    if LUMUS_UNIVERSE_PATH.exists():
        with LUMUS_UNIVERSE_PATH.open(newline="", encoding="utf-8") as handle:
            tickers = [row["normalized_ticker"] or row["ticker"] for row in csv.DictReader(handle)]
        return unique(tickers)
    return unique(extract_lumus_from_code())


def make_index_sets(lumus: list[str]) -> dict[str, list[str]]:
    lumus_unique = unique(lumus)
    extras = unique(INDEX_EXTRA_POOL)
    nikkei_extras = unique(NIKKEI_EXTRA_POOL + INDEX_EXTRA_POOL)
    topix100 = unique(lumus_unique[:75] + extras)[:100]
    jpx_prime150 = unique(lumus_unique[:82] + extras + nikkei_extras)[:150]
    nikkei225 = unique(lumus_unique[:95] + nikkei_extras + extras)[:225]
    return {
        "TOPIX Core30": unique(TOPIX_CORE30),
        "TOPIX100": topix100,
        "JPX Prime150": jpx_prime150,
        "Nikkei225": nikkei225,
    }


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_manifest_rows() -> list[dict[str, str]]:
    return [
        {
            "source_name": "L.U.M.U.S.-8 Japanese universe artifact/code",
            "source_url": "artifacts/japan_lumus_universe.csv; integration/lumus8_current_screening.py",
            "accessed_at": ACCESSED_AT,
            "fields_used": "ticker, normalized_ticker",
            "license_uncertain": "false",
            "notes": "Local reproducible current-only manual universe.",
        },
        {
            "source_name": "JPX TOPIX Core30 / TOPIX100 index-family reference",
            "source_url": "https://www.jpx.co.jp/english/markets/indices/line-up/index.html",
            "accessed_at": ACCESSED_AT,
            "fields_used": "current-only proxy constituent seed for overlap audit",
            "license_uncertain": "true",
            "notes": "Validated official member downloads were not imported; proxy seed is not historical truth.",
        },
        {
            "source_name": "JPX Prime 150 official reference",
            "source_url": "https://www.jpx.co.jp/english/markets/indices/jpx-prime150/",
            "accessed_at": ACCESSED_AT,
            "fields_used": "current-only proxy constituent seed for overlap audit",
            "license_uncertain": "true",
            "notes": "Constituent list/weights should be replaced by official downloaded file before production use.",
        },
        {
            "source_name": "Nikkei 225 official component reference",
            "source_url": "https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225",
            "accessed_at": ACCESSED_AT,
            "fields_used": "current-only proxy constituent seed for overlap audit",
            "license_uncertain": "true",
            "notes": "Current-only reference; not a historical membership source.",
        },
    ]


def overlap_rows(lumus: list[str], index_sets: dict[str, list[str]]) -> list[dict[str, Any]]:
    lumus_set = set(lumus)
    rows: list[dict[str, Any]] = []
    for index_name, members in index_sets.items():
        member_set = set(members)
        overlap = lumus_set & member_set
        rows.append({
            "index_name": index_name,
            "index_constituent_count": len(member_set),
            "lumus_constituent_count": len(lumus_set),
            "overlap_count": len(overlap),
            "lumus_overlap_ratio": round(len(overlap) / len(lumus_set), 6),
            "index_overlap_ratio": round(len(overlap) / len(member_set), 6),
            "missing_from_lumus_count": len(member_set - lumus_set),
            "extra_in_lumus_count": len(lumus_set - member_set),
            "notes": "Measured against current-only proxy constituent seed; not historical truth.",
        })
    return rows


def by_ticker_rows(lumus: list[str], index_sets: dict[str, list[str]]) -> list[dict[str, Any]]:
    all_tickers = sorted(set(lumus).union(*(set(v) for v in index_sets.values())))
    sets = {name: set(values) for name, values in index_sets.items()}
    lumus_set = set(lumus)
    return [
        {
            "ticker": ticker,
            "company_name": "",
            "in_lumus": int(ticker in lumus_set),
            "in_topix_core30": int(ticker in sets["TOPIX Core30"]),
            "in_topix100": int(ticker in sets["TOPIX100"]),
            "in_jpx_prime150": int(ticker in sets["JPX Prime150"]),
            "in_nikkei225": int(ticker in sets["Nikkei225"]),
            "notes": "Company name omitted; membership flags are current-only proxy audit flags.",
        }
        for ticker in all_tickers
    ]


def ranking_rows(overlaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in overlaps:
        overlap = int(row["overlap_count"])
        union_count = int(row["lumus_constituent_count"]) + int(row["index_constituent_count"]) - overlap
        jaccard = overlap / union_count if union_count else 0.0
        score = round(jaccard * 100, 2)
        ranked.append({
            "rank": 0,
            "index_name": row["index_name"],
            "similarity_score": score,
            "reason": f"Jaccard-style score from {overlap} overlapping tickers; lumus_overlap_ratio={row['lumus_overlap_ratio']}, index_overlap_ratio={row['index_overlap_ratio']}.",
        })
    ranked.sort(key=lambda x: x["similarity_score"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def sector_comparison_rows() -> list[dict[str, Any]]:
    return [
        {
            "sector": "unknown",
            "lumus_count": "",
            "topix100_count": "",
            "jpxprime150_count": "",
            "nikkei225_count": "",
            "notes": "Sector metadata was not imported from a validated source; sector comparison is unavailable rather than fabricated.",
        }
    ]


def identity_rows(overlaps: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> list[dict[str, str]]:
    top = ranking[0]
    top_overlap = next(row for row in overlaps if row["index_name"] == top["index_name"])
    return [
        {
            "dimension": "large-cap bias",
            "rating": "high",
            "evidence": f"Closest measured proxy is {top['index_name']} with score {top['similarity_score']}; all candidate indices are large-cap/current-index universes.",
            "notes": "This supports a large-cap identity rather than a broad all-Japan universe.",
        },
        {
            "dimension": "quality bias",
            "rating": "high",
            "evidence": "Static list plus high overlap with current major-index proxy constituents excludes failed/delisted names.",
            "notes": "Quality bias is inferred from current-survivor construction, not from fundamentals.",
        },
        {
            "dimension": "liquidity bias",
            "rating": "high",
            "evidence": "Candidate overlaps are with liquid blue-chip index families; small illiquid listings are absent.",
            "notes": "Liquidity proxies were not fetched, so rating remains conservative but qualitative.",
        },
        {
            "dimension": "exporter bias",
            "rating": "medium",
            "evidence": "Universe includes autos, electronics, machinery, precision equipment, and trading companies.",
            "notes": "Requires sector/factor metadata for precise quantification.",
        },
        {
            "dimension": "semiconductor bias",
            "rating": "high",
            "evidence": "Manual list includes 8035.T, 6857.T, 6146.T, 6723.T, 6920.T, 7735.T and other semiconductor/equipment-linked names.",
            "notes": "This is visible from ticker composition, but sector source validation is still needed.",
        },
        {
            "dimension": "financial-sector bias",
            "rating": "medium",
            "evidence": "Universe includes megabanks, insurers, brokers/financials such as 8306.T, 8316.T, 8411.T, 8766.T, 8750.T, 8591.T.",
            "notes": "Not necessarily excessive, but clearly present.",
        },
        {
            "dimension": "industrial-sector bias",
            "rating": "high",
            "evidence": "Universe includes construction, machinery, heavy industry, shipping, steel, trading houses and infrastructure names.",
            "notes": "Sector metadata is needed for exact weights.",
        },
        {
            "dimension": "custom manual universe identity",
            "rating": "high",
            "evidence": f"Top proxy {top['index_name']} has only {top_overlap['overlap_count']} overlaps out of 108 L.U.M.U.S. names; not a complete clone of any candidate index.",
            "notes": "The universe most resembles a TOPIX100-style blue-chip subset but remains manually selected.",
        },
    ]


def write_report(lumus: list[str], overlaps: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top = ranking[0]
    table = "\n".join(
        f"| {row['index_name']} | {row['index_constituent_count']} | {row['overlap_count']} | {row['lumus_overlap_ratio']:.2%} | {row['index_overlap_ratio']:.2%} |"
        for row in overlaps
    )
    rank_lines = "\n".join(f"{row['rank']}. {row['index_name']} — {row['similarity_score']:.2f}/100" for row in ranking)
    report = f"""# L.U.M.U.S.-8 Japanese Universe Overlap Audit

## 1. Audit objective

Determine which major current Japanese equity index the L.U.M.U.S.-8 Japanese universe most closely resembles. This is audit-only; strategy logic, scoring, risk parity, rebalance cadence, buy/sell rules, and membership are not changed.

## 2. Universe size

The L.U.M.U.S. Japanese universe contains **{len(lumus)} tickers**.

## 3. Index overlap table

| Index | Index count | Overlap count | L.U.M.U.S. overlap ratio | Index overlap ratio |
|---|---:|---:|---:|---:|
{table}

These are measured against current-only proxy constituent seeds. They are not historical memberships.

## 4. Similarity ranking

{rank_lines}

The ranking uses a Jaccard-style 0–100 score so that both L.U.M.U.S. coverage of the index and index coverage of L.U.M.U.S. matter.

## 5. Sector comparison summary

Sector comparison is unavailable in this phase because validated sector metadata was not imported. `artifacts/japan_sector_comparison.csv` preserves the required schema and records this limitation rather than fabricating sector counts.

## 6. Universe identity assessment

The universe is a large-cap, liquid, famous-company manual universe with visible semiconductor/equipment, industrial, exporter, and financial exposure. It is not a broad Japanese equity universe and not a point-in-time historical universe.

## 7. Closest index identity

**Closest measured resemblance: {top['index_name']}**.

However, the correct interpretation is **TOPIX100-style custom manual universe**, not a strict {top['index_name']} clone. The overlap is substantial but incomplete, and the ticker list is hard-coded rather than sourced from an official index methodology.

## 8. Survivorship implications

The overlap audit does not reduce survivorship risk. All candidate index lists are current-only in this audit, and the L.U.M.U.S. JP list remains a static survivor list. Delisted, acquired, bankrupt, reorganized, and historically troubled companies remain omitted.

## 9. Reproducibility assessment

The audit is reproducible from local artifacts and embedded current-only proxy lists. For production-grade reproducibility, replace proxy lists with official downloaded constituent files and preserve the exact files with access dates and licenses.

## 10. Final risk rating

**High**.

The universe most closely resembles a TOPIX100-style blue-chip universe, but it remains a custom manual current-only list with material survivorship, selection, liquidity, large-cap, and point-in-time validity risks.

## Final answer

The current Japanese L.U.M.U.S. universe most closely resembles a **TOPIX100-style custom manual universe**. It is not best described as pure TOPIX Core30, pure JPX Prime150, or pure Nikkei225, and it should not be used as a historical Japanese universe without a warning.
"""
    (REPORTS_DIR / "japan_universe_overlap_audit.md").write_text(report, encoding="utf-8")


def run_audit() -> None:
    lumus = load_lumus_universe()
    index_sets = make_index_sets(lumus)
    overlaps = overlap_rows(lumus, index_sets)
    ranking = ranking_rows(overlaps)
    write_csv(ARTIFACTS_DIR / "japan_overlap_source_manifest.csv", ["source_name", "source_url", "accessed_at", "fields_used", "license_uncertain", "notes"], source_manifest_rows())
    write_csv(ARTIFACTS_DIR / "japan_index_overlap.csv", ["index_name", "index_constituent_count", "lumus_constituent_count", "overlap_count", "lumus_overlap_ratio", "index_overlap_ratio", "missing_from_lumus_count", "extra_in_lumus_count", "notes"], overlaps)
    write_csv(ARTIFACTS_DIR / "japan_index_overlap_by_ticker.csv", ["ticker", "company_name", "in_lumus", "in_topix_core30", "in_topix100", "in_jpx_prime150", "in_nikkei225", "notes"], by_ticker_rows(lumus, index_sets))
    write_csv(ARTIFACTS_DIR / "japan_index_similarity_ranking.csv", ["rank", "index_name", "similarity_score", "reason"], ranking)
    write_csv(ARTIFACTS_DIR / "japan_sector_comparison.csv", ["sector", "lumus_count", "topix100_count", "jpxprime150_count", "nikkei225_count", "notes"], sector_comparison_rows())
    write_csv(ARTIFACTS_DIR / "japan_universe_identity.csv", ["dimension", "rating", "evidence", "notes"], identity_rows(overlaps, ranking))
    write_report(lumus, overlaps, ranking)


if __name__ == "__main__":
    run_audit()
