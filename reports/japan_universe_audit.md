# L.U.M.U.S.-8 Japanese Universe Audit

## 1. Audit purpose

This phase audits only the Japanese equity universe used by L.U.M.U.S.-8. It is **not** a strategy improvement, alpha optimization, or Japanese historical reconstruction.

## 2. Current Japanese universe size

The extracted L.U.M.U.S. Japanese universe contains **108 tickers**.

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
