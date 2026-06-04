# L.U.M.U.S.-8 Tax Drag Audit

## 1. Audit objective

Evaluate whether the practical TTL remains attractive after tax drag, turnover, slippage, and transaction costs. This is an audit only; alpha logic, scoring model, factor definitions, risk parity, universe construction, and position sizing are not modified.

## 2. TTL sensitivity results

| TTL | CAGR | MaxDD | Sharpe | Volatility | Turnover | Trades/year |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 23.57% | -20.15% | 1.144 | 20.60% | 0.6000 | 146.10 |
| 60 | 24.05% | -19.71% | 1.230 | 19.55% | 0.3000 | 73.05 |
| 90 | 24.26% | -19.56% | 1.264 | 19.20% | 0.2000 | 48.70 |
| 120 | 24.35% | -19.49% | 1.280 | 19.02% | 0.1500 | 36.53 |
| 180 | 24.25% | -19.42% | 1.287 | 18.85% | 0.1000 | 24.35 |
| 365 | 21.97% | -19.34% | 1.177 | 18.67% | 0.0493 | 12.01 |

## 3. Tax drag results

Realized gains on reductions/sales are taxed; realized losses do not create refunds. The 20% tax-only best TTL is **180**, with CAGR 23.91% versus pre-tax CAGR 24.25%.

## 4. Slippage results

Slippage is charged on traded notional per side. Higher-turnover TTLs lose more CAGR as slippage rises; detailed rows are in `artifacts/slippage_scenarios.csv`.

## 5. Combined realistic scenario

Assumptions: tax_rate=20%, slippage=0.1% per side, transaction_cost=0.1% per side.

| TTL | CAGR | MaxDD | Sharpe | Turnover |
|---:|---:|---:|---:|---:|
| 30 | 8.59% | -23.90% | 0.417 | 0.6000 |
| 60 | 20.25% | -20.66% | 1.036 | 0.3000 |
| 90 | 22.56% | -19.99% | 1.175 | 0.2000 |
| 120 | 23.39% | -19.73% | 1.229 | 0.1500 |
| 180 | 23.83% | -19.52% | 1.264 | 0.1000 |
| 365 | 21.88% | -19.37% | 1.172 | 0.0493 |

## 6. Best TTL before tax

Best pre-tax TTL by CAGR: **120** with CAGR 24.35%.

## 7. Best TTL after tax

Best TTL under 20% tax-only scenario by CAGR: **180** with CAGR 23.91%.

## 8. Best TTL after all frictions

Ranking under the fixed audit score:

1. TTL 180 — score 96.67
2. TTL 120 — score 93.22
3. TTL 365 — score 92.12
4. TTL 90 — score 88.48
5. TTL 60 — score 76.92
6. TTL 30 — score 27.78

Best practical TTL after all frictions: **180**.

## 9. How much performance is lost to taxation

At the 20% tax-only best TTL (180), observed CAGR tax drag is **0.34%**. This is model-based and assumes no loss refunds or jurisdiction-specific tax optimization.

## 10. How much performance is lost to turnover

At the realistic best TTL (180), combined tax/slippage/cost drag versus no-friction CAGR is **0.43%**. Shorter TTLs trade more frequently and are more vulnerable to this drag.

## 11. Practical recommendation

If L.U.M.U.S.-8 is operated in a normal taxable account, the most practical setting from this audit is **TTL 180**. It balances post-friction CAGR, Sharpe, and lower turnover better than faster TTL settings.

## Assumptions and limitations

- Synthetic deterministic prices are used because no production price provider is bundled in this repository.
- Taxes are applied only to realized gains on sale/reduction; losses are not refunded.
- Slippage and transaction costs are charged on traded notional.
- This audit does not optimize TTL beyond the requested values.
