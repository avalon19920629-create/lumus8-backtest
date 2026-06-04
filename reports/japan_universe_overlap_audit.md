# L.U.M.U.S.-8 Japanese Universe Overlap Audit

## 1. Audit objective

Determine which major current Japanese equity index the L.U.M.U.S.-8 Japanese universe most closely resembles. This is audit-only; strategy logic, scoring, risk parity, rebalance cadence, buy/sell rules, and membership are not changed.

## 2. Universe size

The L.U.M.U.S. Japanese universe contains **108 tickers**.

## 3. Index overlap table

| Index | Index count | Overlap count | L.U.M.U.S. overlap ratio | Index overlap ratio |
|---|---:|---:|---:|---:|
| TOPIX Core30 | 30 | 26 | 24.07% | 86.67% |
| TOPIX100 | 100 | 77 | 71.30% | 77.00% |
| JPX Prime150 | 150 | 84 | 77.78% | 56.00% |
| Nikkei225 | 225 | 96 | 88.89% | 42.67% |

These are measured against current-only proxy constituent seeds. They are not historical memberships.

## 4. Similarity ranking

1. TOPIX100 — 58.78/100
2. JPX Prime150 — 48.28/100
3. Nikkei225 — 40.51/100
4. TOPIX Core30 — 23.21/100

The ranking uses a Jaccard-style 0–100 score so that both L.U.M.U.S. coverage of the index and index coverage of L.U.M.U.S. matter.

## 5. Sector comparison summary

Sector comparison is unavailable in this phase because validated sector metadata was not imported. `artifacts/japan_sector_comparison.csv` preserves the required schema and records this limitation rather than fabricating sector counts.

## 6. Universe identity assessment

The universe is a large-cap, liquid, famous-company manual universe with visible semiconductor/equipment, industrial, exporter, and financial exposure. It is not a broad Japanese equity universe and not a point-in-time historical universe.

## 7. Closest index identity

**Closest measured resemblance: TOPIX100**.

However, the correct interpretation is **TOPIX100-style custom manual universe**, not a strict TOPIX100 clone. The overlap is substantial but incomplete, and the ticker list is hard-coded rather than sourced from an official index methodology.

## 8. Survivorship implications

The overlap audit does not reduce survivorship risk. All candidate index lists are current-only in this audit, and the L.U.M.U.S. JP list remains a static survivor list. Delisted, acquired, bankrupt, reorganized, and historically troubled companies remain omitted.

## 9. Reproducibility assessment

The audit is reproducible from local artifacts and embedded current-only proxy lists. For production-grade reproducibility, replace proxy lists with official downloaded constituent files and preserve the exact files with access dates and licenses.

## 10. Final risk rating

**High**.

The universe most closely resembles a TOPIX100-style blue-chip universe, but it remains a custom manual current-only list with material survivorship, selection, liquidity, large-cap, and point-in-time validity risks.

## Final answer

The current Japanese L.U.M.U.S. universe most closely resembles a **TOPIX100-style custom manual universe**. It is not best described as pure TOPIX Core30, pure JPX Prime150, or pure Nikkei225, and it should not be used as a historical Japanese universe without a warning.
