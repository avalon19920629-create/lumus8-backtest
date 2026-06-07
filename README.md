# L.U.M.U.S.-8 Core Backtest

`lumus8_backtest.py` compares SPY, a 60/40 portfolio, a Dalio-style All Weather portfolio, and L.U.M.U.S.-8 Core both with and without BTC. It downloads adjusted prices with `yfinance`, runs daily valuation with month-end target-weight rebalancing, and saves reproducible CSV reports and PNG charts.

## Run

```bash
python -m pip install -r requirements.txt
python lumus8_backtest.py --start 2005-01-01 --output-dir output
```

## Data policy

- `auto_adjust=True` is explicitly passed to `yfinance`, so adjusted Close prices include distributions and splits.
- Each ticker is downloaded separately and retried up to three times. Successful downloads are cached under the output directory; the cache is used as a fallback after a later provider failure.
- Prices are never backfilled before inception or after the final observation. Only internal gaps of up to five observations are forward-filled and disclosed in `data_coverage.csv`.
- SPY trading sessions are the valuation calendar. BTC weekend moves are captured cumulatively on the next SPY trading day, avoiding artificial zero-return ETF weekend rows in annualized statistics.
- Every portfolio starts at its own first common valid date. Consequently, the BTC version starts later than the ETF-only variants. Compare start dates in `metrics.csv` before interpreting relative final values; use `common_period_metrics.csv` for an apples-to-apples comparison from the BTC portfolio inception.
- The original L.U.M.U.S.-8 allocations total 90%. The script normalizes them to 100%; the BTC-free variant removes BTC and normalizes the remaining allocations.

## Outputs

The output directory contains:

- `metrics.csv`: CAGR, volatility, Sharpe, Sortino, Calmar, worst year, maximum drawdown peak/trough and recovery dates, and recovery durations.
- `common_period_metrics.csv`: the same metrics with every strategy fixed to the BTC portfolio inception for a fair comparison.
- `annual_returns.csv`: calendar-year returns.
- `stress_periods.csv`: portfolio return and within-period maximum drawdown for 2008, 2020, and 2022.
- `risk_contributions.csv`: covariance-based ex-post volatility contributions based on each portfolio's available history.
- `data_coverage.csv`: ticker availability and gap-fill disclosure.
- `equity_curves.csv`, `equity_curves.png`, and `drawdowns.png`: portfolio histories and saved charts.
- `correlation_matrix.csv` and `correlation_heatmap.png`: daily-return correlations for the nine L.U.M.U.S.-8 assets over their common available period.
- `cluster_results.csv`, `cluster_dendrogram.png`, and `cluster_dendrogram_average.png`: four-cluster assignments with `Ticker`, `Cluster`, and `LinkageMethod`, plus Ward and average-linkage dendrograms based on `1 - correlation` dissimilarity.
- `annual_asset_returns.csv`: adjusted-price calendar-year returns for each of the nine Core assets.
- `contribution_2022.csv`: 2022 return attribution for the Core variants with and without BTC.
- `contribution_by_stress_period.csv`: the same asset-level attribution for GFC 2008, COVID 2020, and inflation-driven 2022 stress periods.
- `real_estate_variant_metrics.csv`: fair-period CAGR, maximum drawdown, Sharpe, Sortino, and Calmar comparison for BTC-enabled `LUMUS_VNQ` and `LUMUS_XLRE`. Both variants begin on the later XLRE-compatible inception date.
- `real_estate_variant_risk_contributions.csv`: fair-period ex-post volatility contribution comparison for the VNQ and XLRE variants.
- `real_estate_variant_contribution_2022.csv`: asset-level 2022 return attribution for the VNQ and XLRE variants using the same drifting-weight methodology.
- `lumus_vnq_*` and `lumus_xlre_*` correlation and clustering artifacts: daily-return correlation matrices, heatmaps, Ward dendrograms, average-linkage dendrograms, and cluster assignments for each real-estate sleeve variant over the same XLRE-compatible period.
- `real_estate_variant_cluster_results.csv`: combined VNQ-versus-XLRE cluster assignments for direct comparison.

## Phase-2 audit methodology

Correlation clustering uses daily returns after the existing conservative price preparation and SPY-session alignment steps. `cluster_dendrogram.png` is the required Ward-method audit view. The average-linkage companion chart is also emitted because `1 - correlation` is a direct dissimilarity measure and average linkage provides a useful robustness check.

Contribution reports use each Core variant's actual modeled beginning-of-day weights: weights drift with daily asset returns and reset to the normalized target allocation on the first SPY trading day of every month. Daily contributions are scaled by intra-period portfolio growth before aggregation, so asset contributions add up to the compounded portfolio return shown in each row. This is more precise than a fixed month-start-weight approximation. BTC weekend moves remain cumulatively reflected on the next SPY trading day.

## VNQ-versus-XLRE sleeve audit

`LUMUS_VNQ` preserves the existing BTC-enabled Core allocation. `LUMUS_XLRE` changes only the 6% real-estate sleeve from VNQ to XLRE before the original 90% allocation is normalized. The dedicated `real_estate_variant_*` reports intentionally use a shared XLRE-compatible start date, preventing the longer VNQ history from biasing the direct comparison. Existing benchmark and phase-2 reports remain available unchanged.

## GENKI dynamic-conditioning preliminary audit

`genki_backtest.py` compares the threshold-rebalanced Base Strategy with Conservative
and Standard GENKI variants. The historical proxy uses trailing-only momentum,
200-day moving-average distance, drawdown, and volatility to rank role groups before
applying a conserved, bounded temporary target tilt. Year-end restores the base target.
The simulator reports both gross and after-tax results using an explicitly simplified
average-cost tax model and configurable slippage/fees.

```bash
python genki_backtest.py --start 2010-01-01 --end 2026-06-30 \
  --tax-rate 0.20315 --slippage-bps 5 --fee-bps 0 \
  --output-dir artifacts/genki_audit
```

The audit writes the requested Japanese summary, metrics, annual returns, tilt-event
log, equity curves, and drawdown/equity charts. Because the existing model has no
periodic contribution or cash-flow ledger, New-money-only is disclosed but not modeled.
The audit uses the repository's historical sleeve proxies IEF/GLD/BTC-USD for
BNDX/GLDM/BTC and therefore begins only when every required asset (including XLRE) has
a valid observation.
