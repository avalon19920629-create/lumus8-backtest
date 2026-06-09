# L.U.M.U.S.-8 Core Backtest

`lumus8_backtest.py` compares SPY, a 60/40 portfolio, a Dalio-style All Weather portfolio, and the formal `LUMUS_EX_ALPHA_VT_REPLACED` baseline, while retaining legacy L.U.M.U.S.-8 sample allocations. It downloads adjusted prices with `yfinance`, runs daily valuation with month-end target-weight rebalancing, and saves reproducible CSV reports and PNG charts.

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
- Every portfolio starts at its own first common valid date in `metrics.csv`. Fair comparisons in `common_period_metrics.csv` and `annual_returns.csv` use the formal BTC-enabled `LUMUS_EX_ALPHA_VT_REPLACED` inception, when BTC-USD, BNDX, XLRE, and all other required assets are available.
- The original L.U.M.U.S.-8 allocations total 90%. The script normalizes them to 100%; the BTC-free variant removes BTC and normalizes the remaining allocations.

## Outputs

The output directory contains:

- `metrics.csv`: CAGR, volatility, Sharpe, Sortino, Calmar, worst year, maximum drawdown peak/trough and recovery dates, and recovery durations.
- `common_period_metrics.csv`: the same metrics with every strategy fixed to the formal ex-Alpha baseline inception for a fair comparison.
- `annual_returns.csv`: common-period calendar-year returns.
- `portfolio_period_annual_returns.csv`: calendar-year returns over each portfolio’s individually available period.
- `drawdowns.csv`: drawdown series for every portfolio.
- `comparison_2022.csv`: common-period 2022 return and within-year drawdown comparison.
- `lumus_ex_alpha_vt_replaced_report.md`: Japanese research report centered on the formal ex-Alpha / VT-replaced baseline; `LUMUS_CORE_WITH_BTC` and `LUMUS_XLRE` are explicitly labeled legacy sample allocations.
- `stress_periods.csv`: common-period portfolio return and within-period maximum drawdown for 2008, 2020, and 2022.
- `portfolio_period_stress_periods.csv`: the same stress table over each portfolio’s individually available history.
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

## Allocation robustness preliminary audit

`allocation_robustness_audit.py` compares the current fixed allocation with a small,
predefined set of growth, defense, real-asset, TLT, BNDX-proxy, BTC, simplified, and
near-current variants. It applies the L.U.M.U.S.-8 ±5 percentage-point core / ±10
percentage-point support thresholds plus year-end rebalancing, and records a GENKI-style
simple average-cost tax estimate and trading friction. This is a robustness audit, not
an allocation optimizer.

```bash
python allocation_robustness_audit.py --start 2010-01-01 --end 2026-06-30 \
  --tax-rate 0.20315 --slippage-bps 5 --fee-bps 0 \
  --output-dir artifacts/allocation_audit
```

The Japanese summary and requested CSV/PNG artifacts are written under the selected
output directory. As in the GENKI audit, IEF/GLD/BTC-USD provide the long-history
proxies for BNDX/GLDM/BTC; the report explicitly discloses this limitation.

## Growth_Heavy decomposition preliminary audit

`growth_heavy_decomposition_audit.py` decomposes the previous audit's Growth_Heavy
result with twelve predefined counterfactual allocations. It separately probes VT,
BTC, defense, TLT, and SHY changes while retaining the same threshold/year-end
rebalancing and simplified after-tax cost ledger. The fixed candidate grid is not
optimized from historical prices, and the result is explicitly an input to further
validation rather than a live-allocation change.

```bash
python growth_heavy_decomposition_audit.py --start 2015-10-08 --end 2026-06-06 \
  --tax-rate 0.20315 --slippage-bps 5 --fee-bps 0 \
  --output-dir artifacts/growth_decomposition
```

A 2010 start is accepted, but the report discloses the later all-asset common start
forced by assets such as BTC-USD and XLRE. The output directory receives the Japanese
summary, metrics, annual returns, rebalance events, equity/drawdown CSVs, and charts.

## Rebalance-band robustness preliminary audit

`rebalance_band_robustness_audit.py` keeps the current normalized L.U.M.U.S.-8 allocation
fixed and compares eight predefined band/year-end policies, including conditional,
biennial, loss-position-only, and minimum-sale rebalancing. It reports after-tax return,
drawdown, risk ratios, turnover, trade/tax counts, tax cost, 2022 resilience, annual
wins/losses, and rolling three-/five-year win rates.

```bash
python rebalance_band_robustness_audit.py --start 2010-01-01 --end 2026-06-06 \
  --tax-rate 0.20315 --slippage-bps 5 --fee-bps 0 \
  --output-dir artifacts/rebalance_band_audit
```

The output directory receives CSV tables, PNG charts, price coverage, and a Japanese
summary report with an explicit adoption classification. The historical proxy and
simplified average-cost tax-model limitations are disclosed in the report.

### Tax-loss carryforward rebalance audit (Japan-inspired approximation)

The rebalance-band audit can optionally compare the same eight policies with an
annual-netting, three-year `tax_loss_pool` approximation. A net realized loss in year
Y is available to offset net realized gains in Y+1 through Y+3; the oldest pool is used
first, and any unused balance expires after Y+3. This is a deliberately simplified
backtest model inspired by Japan's listed-securities loss carryforward framework, not
tax advice and not a complete reproduction of filing requirements, eligible income,
account types, or dividend-tax elections.

```bash
python rebalance_band_robustness_audit.py \
  --start 2015-10-08 --end 2026-06-06 \
  --tax-rate 0.20315 --slippage-bps 5 --fee-bps 0 \
  --enable-tax-loss-carryforward \
  --output-dir artifacts/rebalance_band_audit_loss_carryforward
```

The carryforward run writes the Japanese loss-carryforward summary, policy metrics,
tax-loss events, rebalance events, annual returns, equity curves, and equity/drawdown
charts. Real-data execution requires an environment that can connect to Yahoo Finance;
the test suite uses synthetic prices to validate the model and artifact path offline.

### Conditional Year-End Threshold Sweep

The independent Conditional Year-End Threshold Sweep keeps the current core ±5-point
and support ±10-point emergency rebalance bands enabled, while comparing year-end
rebalance triggers at 25%, 50%, 75%, 100%, and 125% of each asset's applicable band.
`Threshold_Only` and `Annual_Only` are included as reference cases. The primary adoption
judgment uses the tax-loss-carryforward-adjusted results; if the differences are small
or effectively tied, the existing annual rebalance is preferred to preserve the
understandable fixed-weight homeostasis.

```bash
python rebalance_band_robustness_audit.py \
  --start 2015-10-08 --end 2026-06-06 \
  --tax-rate 0.20315 --slippage-bps 5 --fee-bps 0 \
  --enable-tax-loss-carryforward \
  --run-conditional-threshold-sweep \
  --output-dir artifacts/rebalance_condition_threshold_audit
```

The sweep writes `conditional_threshold_summary_report.md`, metrics, rebalance and
tax-loss events, annual returns, equity/drawdown CSVs, and PNG charts. Real-data
execution requires an environment that can connect to Yahoo Finance; synthetic-price
tests validate the sweep and artifact generation when provider access is unavailable.

## Constrained L.U.M.U.S.-8 allocation optimization research

`lumus8_optimization.py` searches a deterministic sample of feasible allocations for the
nine-asset L.U.M.U.S.-8 research universe. It applies the documented asset and group
bounds, selects max-Sharpe, max-Sortino, max-Calmar, max-CAGR-with-MDD-limit, and
min-volatility-with-CAGR-target candidates, then builds a robust convex-average candidate.
Search ranking uses daily constant-weight returns for tractability; all published candidate
metrics use the repository's monthly-rebalanced simulator. This is in-sample portfolio
design research, not investment advice or a claim of future-optimal weights.

```bash
python lumus8_optimization.py --start 2015-10-08 --end 2026-06-10 \
  --samples 5000 --output-dir output
```

The command writes `optimized_weights.csv`, optimized metric/annual/equity/drawdown/2022
and SP500-delta tables, `optimized_report.md`, and the requested efficient-frontier,
equity-curve, and drawdown charts. Increase `--samples` for a broader search; compare
multiple seeds and add walk-forward validation before drawing allocation conclusions.
