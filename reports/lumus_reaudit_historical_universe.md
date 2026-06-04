# L.U.M.U.S.-8 Re-Audit with Historical S&P 500 Universe

## Scope

This is a re-audit, not a strategy improvement. The alpha logic, score formula, TTL/rebalance cadence implied by the audit dates, risk-parity weighting, and buy/sell selection rules are not optimized. The only US-side change is `current` versus `historical` universe membership via `get_sp500_members(rebalance_date)`.

The Japanese universe remains the existing manual list in both runs; **survivor-only risk remains** for Japan.

## Comparison Table

| Metric | Current Universe | Historical Universe | Historical - Current |
|---|---:|---:|---:|
| CAGR | 25.53% | 25.04% | -0.50% |
| MaxDD | -3.77% | -3.77% | 0.00% |
| Sharpe | 1.093 | 1.073 | -0.020 |
| Turnover | 0.034 | 0.059 | 0.025 |
| average selected US tickers | 6.00 | 6.00 | 0.00 |
| average selected JP tickers | 6.00 | 6.00 | 0.00 |
| average historical universe size | 500.00 | 498.06 | -1.94 |
| missing price ticker count | 0 | 0 | 0 |
| failed historical universe dates | 0 | 0 | 0 |

## Required Findings

1. **current universe版とhistorical universe版の成績差**: Historical mode CAGR delta is -0.50%, MaxDD delta is 0.00%, and Sharpe delta is -0.020.
2. **CAGR低下幅**: 0.50% should be treated as the observed minimum haircut in this seed-data audit.
3. **MaxDD変化**: Historical MaxDD changed by 0.00%; more-negative values mean deeper drawdown.
4. **Sharpe変化**: Historical Sharpe changed by -0.020.
5. **差が大きかった年代・リバランス**:
- 2024-03-18: selected-count delta +0
- 2024-05-08: selected-count delta +0
- 2024-06-24: selected-count delta +0
- 2024-09-23: selected-count delta +0
- 2024-12-23: selected-count delta +0
6. **historical universe統合後も残るバイアス**: JP survivor-only bias remains, the bundled S&P 500 history is sparse, and delisted-name price availability still needs vendor verification. Missing prices are logged rather than optimistically filled.
7. **L.U.M.U.S.-8の旧CAGRをどの程度保守的に補正すべきか**: At minimum, subtract the observed current-minus-historical CAGR gap, 0.50%. If expanded historical membership coverage produces a larger gap, use the larger haircut.
8. **次に検証すべき課題**: expand point-in-time US membership history, add a point-in-time JP universe, validate delisted ticker prices, and rerun against production price data for the full intended backtest range.

## Notes

- Backward filling is intentionally not used.
- This repository execution uses deterministic synthetic prices when external price dependencies are unavailable; production audits should run the same artifact pipeline with real adjusted close data.
- Audit rebalance dates: 2024-03-18 through 2026-05-07 (17 dates).
