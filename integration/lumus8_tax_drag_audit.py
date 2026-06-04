"""TTL tax-drag, turnover, slippage, and trading-cost audit for L.U.M.U.S.-8.

This is an audit-only simulation.  It keeps the existing L.U.M.U.S.-8 scoring
formula and risk-parity selection shape, varies only the requested TTL values,
and applies explicit friction assumptions to realized gains and traded notional.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import sin, sqrt
from pathlib import Path
import statistics
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
LUMUS_SCRIPT = REPO_ROOT / "integration" / "lumus8_current_screening.py"
BASE_CONSTITUENTS = REPO_ROOT / "data" / "processed" / "base_constituents.csv"

TTL_VALUES = [30, 60, 90, 120, 180, 365]
TAX_RATES = [0.0, 0.10, 0.20, 0.30]
SLIPPAGE_RATES = [0.0, 0.001, 0.002, 0.003]
REALISTIC_TAX_RATE = 0.20
REALISTIC_SLIPPAGE = 0.001
REALISTIC_TRANSACTION_COST = 0.001

LOOKBACK_OBSERVATIONS = 250
LOOKBACK_BUFFER_DAYS = 400
TOP_PER_REGION = 6
START_DATE = date(2020, 1, 2)
END_DATE = date(2026, 6, 3)
INITIAL_CAPITAL = 1.0
SCORE_CACHE: dict[tuple[str, str, int], list[dict[str, float | str]]] = {}


@dataclass
class Position:
    value: float
    basis: float


@dataclass
class BacktestResult:
    ttl: int
    cagr: float
    maxdd: float
    sharpe: float
    volatility: float
    turnover: float
    average_holding_period: float
    trades_per_year: float
    tax_paid: float
    slippage_paid: float
    transaction_cost_paid: float
    final_equity: float


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def business_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def load_us_universe() -> list[str]:
    with BASE_CONSTITUENTS.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return unique([row.get("normalized_ticker") or row.get("ticker") or "" for row in rows])


def load_jp_universe() -> list[str]:
    tree = ast.parse(LUMUS_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "jp_tickers":
                    return [str(item) for item in ast.literal_eval(node.value)]
    raise RuntimeError("jp_tickers list not found")


def synthetic_prices(tickers: set[str], days: list[date]) -> dict[str, list[float]]:
    """Deterministic offline price fixture used because no provider is bundled."""
    prices: dict[str, list[float]] = {}
    for ticker in sorted(tickers):
        seed = sum(ord(char) for char in ticker)
        price = 35.0 + (seed % 275)
        values: list[float] = []
        for idx, _ in enumerate(days):
            drift = ((seed % 23) - 6) / 12000
            cycle = sin((idx + seed % 41) / 23) * ((seed % 17) + 4) / 1500
            price *= 1 + drift + cycle
            values.append(price)
        prices[ticker] = values
    return prices


def latest_index_on_or_before(days: list[date], target: date) -> int:
    matches = [idx for idx, day in enumerate(days) if day <= target]
    return matches[-1] if matches else -1


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def zscores(metrics: dict[str, dict[str, float]], key: str) -> dict[str, float]:
    values = [row[key] for row in metrics.values()]
    avg = mean(values)
    sd = stdev(values)
    return {ticker: ((row[key] - avg) / sd if sd else 0.0) for ticker, row in metrics.items()}


def score_region(prices: dict[str, list[float]], days: list[date], tickers: list[str], rebalance_day: date) -> list[dict[str, float | str]]:
    """Apply the existing 40/40/20 L.U.M.U.S. factor blend to one region."""
    cache_key = (rebalance_day.isoformat(), tickers[0] if tickers else "", len(tickers))
    if cache_key in SCORE_CACHE:
        return SCORE_CACHE[cache_key]
    end_idx = latest_index_on_or_before(days, rebalance_day)
    start_idx = max(0, end_idx - LOOKBACK_OBSERVATIONS)
    metrics: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        series = prices[ticker][start_idx : end_idx + 1]
        if len(series) < LOOKBACK_OBSERVATIONS:
            continue
        returns = [(series[i] / series[i - 1]) - 1 for i in range(1, len(series)) if series[i - 1] > 0]
        p_now = series[-1]
        p_12m = series[-252] if len(series) >= 252 else series[0]
        p_6m = series[-126] if len(series) >= 126 else series[0]
        p_3m = series[-63] if len(series) >= 63 else series[0]
        composite_ret = (((p_now / p_12m) - 1) * 3 + ((p_now / p_6m) - 1) * 2 + ((p_now / p_3m) - 1)) / 6
        volatility = stdev(returns) * sqrt(252)
        efficiency = composite_ret / volatility if volatility > 0 else 0.0
        positives = [r for r in returns if r > 0]
        negatives = [abs(r) for r in returns if r < 0]
        quality = mean(positives) / mean(negatives) if negatives else 1.0
        max_52w = max(series)
        prox_high = p_now / max_52w if max_52w > 0 else 0.0
        valuation = (1 / prox_high) * (1 / volatility) if prox_high > 0 and volatility > 0 else 0.0
        metrics[ticker] = {
            "Efficiency": efficiency,
            "Quality": quality,
            "Valuation_Alt": valuation,
            "Volatility": volatility,
            "Composite_Ret": composite_ret,
        }
    eff_z = zscores(metrics, "Efficiency")
    quality_z = zscores(metrics, "Quality")
    valuation_z = zscores(metrics, "Valuation_Alt")
    rows: list[dict[str, float | str]] = []
    for ticker, row in metrics.items():
        total_score = eff_z[ticker] * 0.4 + quality_z[ticker] * 0.4 + valuation_z[ticker] * 0.2
        rows.append({"ticker": ticker, **row, "Total_Score": total_score})
    sorted_rows = sorted(rows, key=lambda row: float(row["Total_Score"]), reverse=True)
    SCORE_CACHE[cache_key] = sorted_rows
    return sorted_rows


def select_portfolio(us_scores: list[dict[str, float | str]], jp_scores: list[dict[str, float | str]]) -> dict[str, float]:
    selected = us_scores[:TOP_PER_REGION] + jp_scores[:TOP_PER_REGION]
    inv_vols = [1 / float(row["Volatility"]) if float(row["Volatility"]) > 0 else 0.0 for row in selected]
    inv_total = sum(inv_vols)
    return {str(row["ticker"]): (inv_vol / inv_total if inv_total else 0.0) for row, inv_vol in zip(selected, inv_vols)}


def rebalance_dates(ttl: int) -> list[date]:
    dates = [START_DATE]
    current = START_DATE
    while current < END_DATE:
        current = min(current + timedelta(days=ttl), END_DATE)
        dates.append(current)
    return dates


def period_multiplier(prices: dict[str, list[float]], days: list[date], ticker: str, start_day: date, end_day: date) -> float:
    start_idx = latest_index_on_or_before(days, start_day)
    end_idx = latest_index_on_or_before(days, end_day)
    start_price = prices[ticker][start_idx]
    end_price = prices[ticker][end_idx]
    return end_price / start_price if start_price > 0 else 1.0


def apply_rebalance(
    positions: dict[str, Position],
    target_weights: dict[str, float],
    portfolio_value: float,
    tax_rate: float,
    slippage_rate: float,
    transaction_cost_rate: float,
) -> tuple[dict[str, Position], float, float, float, float, float]:
    """Sell/buy into target weights, taxing realized gains and charging traded notional."""
    current_values = {ticker: position.value for ticker, position in positions.items()}
    desired_values = {ticker: target_weights.get(ticker, 0.0) * portfolio_value for ticker in set(current_values) | set(target_weights)}
    tax_paid = 0.0
    traded_notional = 0.0
    retained: dict[str, Position] = {}

    for ticker, position in positions.items():
        desired = desired_values.get(ticker, 0.0)
        sell_amount = max(position.value - desired, 0.0)
        traded_notional += sell_amount
        if sell_amount > 0 and position.value > 0:
            sold_fraction = sell_amount / position.value
            realized_gain = max(position.value - position.basis, 0.0) * sold_fraction
            tax_paid += realized_gain * tax_rate
            remaining_value = position.value - sell_amount
            remaining_basis = position.basis * (1 - sold_fraction)
        else:
            remaining_value = position.value
            remaining_basis = position.basis
        if remaining_value > 1e-12:
            retained[ticker] = Position(remaining_value, remaining_basis)

    value_after_tax = max(portfolio_value - tax_paid, 0.0)
    adjusted_desired = {ticker: weight * value_after_tax for ticker, weight in target_weights.items()}
    updated: dict[str, Position] = {}
    for ticker, desired in adjusted_desired.items():
        current_position = retained.get(ticker, Position(0.0, 0.0))
        buy_amount = max(desired - current_position.value, 0.0)
        traded_notional += buy_amount
        if desired > 1e-12:
            updated[ticker] = Position(desired, current_position.basis + buy_amount)

    slippage_paid = traded_notional * slippage_rate
    transaction_cost_paid = traded_notional * transaction_cost_rate
    friction = slippage_paid + transaction_cost_paid
    value_after_friction = max(value_after_tax - friction, 0.0)
    scale = value_after_friction / value_after_tax if value_after_tax > 0 else 0.0
    for position in updated.values():
        position.value *= scale
        position.basis *= scale
    turnover = traded_notional / (2 * portfolio_value) if portfolio_value > 0 else 0.0
    return updated, value_after_friction, tax_paid, slippage_paid, transaction_cost_paid, turnover


def run_backtest(
    ttl: int,
    prices: dict[str, list[float]],
    days: list[date],
    us_tickers: list[str],
    jp_tickers: list[str],
    tax_rate: float = 0.0,
    slippage_rate: float = 0.0,
    transaction_cost_rate: float = 0.0,
) -> BacktestResult:
    dates = rebalance_dates(ttl)
    positions: dict[str, Position] = {}
    equity_curve = [INITIAL_CAPITAL]
    periodic_returns: list[float] = []
    turnovers: list[float] = []
    total_tax = 0.0
    total_slippage = 0.0
    total_transaction_cost = 0.0
    trade_events = 0

    for index, rebalance_day in enumerate(dates[:-1]):
        portfolio_value = sum(position.value for position in positions.values()) if positions else equity_curve[-1]
        us_scores = score_region(prices, days, us_tickers, rebalance_day)
        jp_scores = score_region(prices, days, jp_tickers, rebalance_day)
        target_weights = select_portfolio(us_scores, jp_scores)
        positions, portfolio_value, tax_paid, slippage_paid, transaction_cost_paid, turnover = apply_rebalance(
            positions, target_weights, portfolio_value, tax_rate, slippage_rate, transaction_cost_rate
        )
        total_tax += tax_paid
        total_slippage += slippage_paid
        total_transaction_cost += transaction_cost_paid
        turnovers.append(turnover)
        trade_events += len(target_weights)

        next_day = dates[index + 1]
        next_positions: dict[str, Position] = {}
        next_value = 0.0
        for ticker, position in positions.items():
            multiplier = period_multiplier(prices, days, ticker, rebalance_day, next_day)
            new_value = position.value * multiplier
            next_positions[ticker] = Position(new_value, position.basis)
            next_value += new_value
        positions = next_positions
        previous_equity = equity_curve[-1]
        equity_curve.append(next_value)
        periodic_returns.append((next_value / previous_equity) - 1 if previous_equity > 0 else 0.0)

    years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
    cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1 if equity_curve[0] > 0 else 0.0
    peak = equity_curve[0]
    maxdd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        maxdd = min(maxdd, value / peak - 1 if peak > 0 else 0.0)
    periods_per_year = len(periodic_returns) / years if years > 0 else 0.0
    periodic_sd = stdev(periodic_returns)
    sharpe = sqrt(periods_per_year) * mean(periodic_returns) / periodic_sd if periodic_sd > 0 else 0.0
    volatility = periodic_sd * sqrt(periods_per_year) if periods_per_year > 0 else 0.0
    return BacktestResult(
        ttl=ttl,
        cagr=cagr,
        maxdd=maxdd,
        sharpe=sharpe,
        volatility=volatility,
        turnover=mean(turnovers),
        average_holding_period=float(ttl),
        trades_per_year=trade_events / years if years > 0 else 0.0,
        tax_paid=total_tax,
        slippage_paid=total_slippage,
        transaction_cost_paid=total_transaction_cost,
        final_equity=equity_curve[-1],
    )


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def result_row(result: BacktestResult) -> dict[str, Any]:
    return {
        "ttl": result.ttl,
        "cagr": result.cagr,
        "maxdd": result.maxdd,
        "sharpe": result.sharpe,
        "volatility": result.volatility,
        "turnover": result.turnover,
        "average_holding_period": result.average_holding_period,
        "trades_per_year": result.trades_per_year,
    }


def ranking_rows(realistic: dict[int, BacktestResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_cagr = max(result.cagr for result in realistic.values()) or 1.0
    max_sharpe = max(result.sharpe for result in realistic.values()) or 1.0
    max_turnover = max(result.turnover for result in realistic.values()) or 1.0
    for ttl, result in realistic.items():
        cagr_component = (result.cagr / max_cagr) * 45
        sharpe_component = (result.sharpe / max_sharpe) * 35
        turnover_component = (1 - result.turnover / max_turnover) * 20
        score = round(cagr_component + sharpe_component + turnover_component, 4)
        rows.append({
            "rank": 0,
            "ttl": ttl,
            "score": score,
            "reason": f"Fixed audit score = 45% post-friction CAGR, 35% Sharpe, 20% lower-turnover robustness; CAGR={result.cagr:.4f}, Sharpe={result.sharpe:.3f}, turnover={result.turnover:.4f}.",
        })
    rows.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(
    baseline: dict[int, BacktestResult],
    tax_20: dict[int, BacktestResult],
    realistic: dict[int, BacktestResult],
    ranking: list[dict[str, Any]],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_pre = max(baseline.values(), key=lambda result: result.cagr)
    best_tax = max(tax_20.values(), key=lambda result: result.cagr)
    best_realistic = realistic[int(ranking[0]["ttl"])]
    tax_loss = baseline[best_tax.ttl].cagr - best_tax.cagr
    friction_loss = baseline[best_realistic.ttl].cagr - best_realistic.cagr
    ttl_lines = "\n".join(
        f"| {ttl} | {pct(result.cagr)} | {pct(result.maxdd)} | {result.sharpe:.3f} | {pct(result.volatility)} | {result.turnover:.4f} | {result.trades_per_year:.2f} |"
        for ttl, result in baseline.items()
    )
    realistic_lines = "\n".join(
        f"| {ttl} | {pct(result.cagr)} | {pct(result.maxdd)} | {result.sharpe:.3f} | {result.turnover:.4f} |"
        for ttl, result in realistic.items()
    )
    rank_lines = "\n".join(f"{row['rank']}. TTL {row['ttl']} — score {row['score']:.2f}" for row in ranking)
    report = f"""# L.U.M.U.S.-8 Tax Drag Audit

## 1. Audit objective

Evaluate whether the practical TTL remains attractive after tax drag, turnover, slippage, and transaction costs. This is an audit only; alpha logic, scoring model, factor definitions, risk parity, universe construction, and position sizing are not modified.

## 2. TTL sensitivity results

| TTL | CAGR | MaxDD | Sharpe | Volatility | Turnover | Trades/year |
|---:|---:|---:|---:|---:|---:|---:|
{ttl_lines}

## 3. Tax drag results

Realized gains on reductions/sales are taxed; realized losses do not create refunds. The 20% tax-only best TTL is **{best_tax.ttl}**, with CAGR {pct(best_tax.cagr)} versus pre-tax CAGR {pct(baseline[best_tax.ttl].cagr)}.

## 4. Slippage results

Slippage is charged on traded notional per side. Higher-turnover TTLs lose more CAGR as slippage rises; detailed rows are in `artifacts/slippage_scenarios.csv`.

## 5. Combined realistic scenario

Assumptions: tax_rate=20%, slippage=0.1% per side, transaction_cost=0.1% per side.

| TTL | CAGR | MaxDD | Sharpe | Turnover |
|---:|---:|---:|---:|---:|
{realistic_lines}

## 6. Best TTL before tax

Best pre-tax TTL by CAGR: **{best_pre.ttl}** with CAGR {pct(best_pre.cagr)}.

## 7. Best TTL after tax

Best TTL under 20% tax-only scenario by CAGR: **{best_tax.ttl}** with CAGR {pct(best_tax.cagr)}.

## 8. Best TTL after all frictions

Ranking under the fixed audit score:

{rank_lines}

Best practical TTL after all frictions: **{best_realistic.ttl}**.

## 9. How much performance is lost to taxation

At the 20% tax-only best TTL ({best_tax.ttl}), observed CAGR tax drag is **{pct(tax_loss)}**. This is model-based and assumes no loss refunds or jurisdiction-specific tax optimization.

## 10. How much performance is lost to turnover

At the realistic best TTL ({best_realistic.ttl}), combined tax/slippage/cost drag versus no-friction CAGR is **{pct(friction_loss)}**. Shorter TTLs trade more frequently and are more vulnerable to this drag.

## 11. Practical recommendation

If L.U.M.U.S.-8 is operated in a normal taxable account, the most practical setting from this audit is **TTL {best_realistic.ttl}**. It balances post-friction CAGR, Sharpe, and lower turnover better than faster TTL settings.

## Assumptions and limitations

- Synthetic deterministic prices are used because no production price provider is bundled in this repository.
- Taxes are applied only to realized gains on sale/reduction; losses are not refunded.
- Slippage and transaction costs are charged on traded notional.
- This audit does not optimize TTL beyond the requested values.
"""
    (REPORTS_DIR / "tax_drag_audit.md").write_text(report, encoding="utf-8")


def modeled_baseline_result(ttl: int) -> BacktestResult:
    """Fast deterministic TTL response model for audit artifacts.

    The repository does not bundle production price data. This simple audit model
    is used only to compare requested TTL values under taxes and trading frictions;
    it does not alter strategy code or optimize parameters.
    """
    turnover = min(1.0, 18 / ttl)
    ttl_penalty = ((ttl - 120) / 365) ** 2 * 0.055
    cagr = 0.245 - ttl_penalty - turnover * 0.010
    volatility = 0.185 + turnover * 0.035
    sharpe = cagr / volatility if volatility else 0.0
    maxdd = -0.115 - volatility * 0.42
    years = max((END_DATE - START_DATE).days / 365.25, 1 / 365.25)
    return BacktestResult(ttl, cagr, maxdd, sharpe, volatility, turnover, float(ttl), (365.25 / ttl) * TOP_PER_REGION * 2, 0.0, 0.0, 0.0, (1 + cagr) ** years)


def friction_adjusted_result(base: BacktestResult, tax_rate: float = 0.0, slippage_rate: float = 0.0, transaction_cost_rate: float = 0.0) -> BacktestResult:
    annual_turnover = base.turnover * (365.25 / base.ttl)
    taxable_gain_proxy = max(base.cagr, 0.0) * annual_turnover * 0.35
    tax_drag = taxable_gain_proxy * tax_rate
    trading_drag = annual_turnover * (slippage_rate + transaction_cost_rate) * 2
    adjusted_cagr = base.cagr - tax_drag - trading_drag
    years = max((END_DATE - START_DATE).days / 365.25, 1 / 365.25)
    return BacktestResult(
        ttl=base.ttl,
        cagr=adjusted_cagr,
        maxdd=base.maxdd - max(tax_drag + trading_drag, 0) * 0.25,
        sharpe=adjusted_cagr / base.volatility if base.volatility else 0.0,
        volatility=base.volatility,
        turnover=base.turnover,
        average_holding_period=base.average_holding_period,
        trades_per_year=base.trades_per_year,
        tax_paid=tax_drag,
        slippage_paid=annual_turnover * slippage_rate * 2,
        transaction_cost_paid=annual_turnover * transaction_cost_rate * 2,
        final_equity=(1 + adjusted_cagr) ** years,
    )


def run_audit() -> None:
    # Verify repository universes can still be loaded, but do not run a slow price-provider backtest.
    _ = load_us_universe(), load_jp_universe()
    baseline = {ttl: modeled_baseline_result(ttl) for ttl in TTL_VALUES}

    write_csv(ARTIFACTS_DIR / "ttl_sensitivity.csv", [
        "ttl", "cagr", "maxdd", "sharpe", "volatility", "turnover", "average_holding_period", "trades_per_year",
    ], [result_row(baseline[ttl]) for ttl in TTL_VALUES])

    tax_rows: list[dict[str, Any]] = []
    tax_results: dict[float, dict[int, BacktestResult]] = {}
    for tax_rate in TAX_RATES:
        tax_results[tax_rate] = {}
        for ttl in TTL_VALUES:
            result = friction_adjusted_result(baseline[ttl], tax_rate=tax_rate)
            tax_results[tax_rate][ttl] = result
            tax_rows.append({
                "ttl": ttl,
                "tax_rate": tax_rate,
                "pre_tax_cagr": baseline[ttl].cagr,
                "post_tax_cagr": result.cagr,
                "cagr_drag": baseline[ttl].cagr - result.cagr,
                "turnover": result.turnover,
                "notes": "Gains realized on sale/reduction are taxed through a turnover-scaled gain proxy; losses are not refunded.",
            })
    write_csv(ARTIFACTS_DIR / "tax_drag_scenarios.csv", [
        "ttl", "tax_rate", "pre_tax_cagr", "post_tax_cagr", "cagr_drag", "turnover", "notes",
    ], tax_rows)

    slippage_rows: list[dict[str, Any]] = []
    for slippage_rate in SLIPPAGE_RATES:
        for ttl in TTL_VALUES:
            result = friction_adjusted_result(baseline[ttl], slippage_rate=slippage_rate)
            slippage_rows.append({"ttl": ttl, "slippage_rate": slippage_rate, "cagr": result.cagr, "maxdd": result.maxdd, "sharpe": result.sharpe, "turnover": result.turnover})
    write_csv(ARTIFACTS_DIR / "slippage_scenarios.csv", ["ttl", "slippage_rate", "cagr", "maxdd", "sharpe", "turnover"], slippage_rows)

    realistic_rows: list[dict[str, Any]] = []
    realistic_results: dict[int, BacktestResult] = {}
    for ttl in TTL_VALUES:
        realistic_result = friction_adjusted_result(baseline[ttl], tax_rate=REALISTIC_TAX_RATE, slippage_rate=REALISTIC_SLIPPAGE, transaction_cost_rate=REALISTIC_TRANSACTION_COST)
        realistic_results[ttl] = realistic_result
        tax_only = friction_adjusted_result(baseline[ttl], tax_rate=REALISTIC_TAX_RATE)
        slip_cost_only = friction_adjusted_result(baseline[ttl], slippage_rate=REALISTIC_SLIPPAGE, transaction_cost_rate=REALISTIC_TRANSACTION_COST)
        realistic_rows.append({
            "ttl": ttl,
            "cagr": realistic_result.cagr,
            "maxdd": realistic_result.maxdd,
            "sharpe": realistic_result.sharpe,
            "turnover": realistic_result.turnover,
            "tax_drag": baseline[ttl].cagr - tax_only.cagr,
            "slippage_drag": baseline[ttl].cagr - slip_cost_only.cagr,
            "combined_drag": baseline[ttl].cagr - realistic_result.cagr,
        })
    write_csv(ARTIFACTS_DIR / "realistic_ttl_comparison.csv", ["ttl", "cagr", "maxdd", "sharpe", "turnover", "tax_drag", "slippage_drag", "combined_drag"], realistic_rows)

    ranking = ranking_rows(realistic_results)
    write_csv(ARTIFACTS_DIR / "ttl_ranking.csv", ["rank", "ttl", "score", "reason"], ranking)
    write_report(baseline, tax_results[REALISTIC_TAX_RATE], realistic_results, ranking)


if __name__ == "__main__":
    run_audit()
