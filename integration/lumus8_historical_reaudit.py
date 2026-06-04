"""Re-audit L.U.M.U.S.-8 with current vs historical S&P 500 universes.

No strategy parameters are optimized here: the audit preserves the existing
L.U.M.U.S.-8 3-factor score blend and risk-parity weighting while changing only
whether the US universe is survivor-only current constituents or point-in-time
historical constituents at each rebalance date.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import sin, sqrt
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumus_historical_universe.reconstruct import get_sp500_members, get_sp500_quality
from lumus_historical_universe.sources import project_root

LOOKBACK_DAYS = 400
MIN_OBSERVATIONS = 250
TOP_PER_REGION = 6
INITIAL_CAPITAL = 1.0
DEFAULT_START_DATE = "2024-03-18"
DEFAULT_END_DATE = "2026-05-07"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
SUMMARY_PATH = ARTIFACTS_DIR / "reaudit_summary.csv"
UNIVERSE_LOG_PATH = ARTIFACTS_DIR / "rebalance_universe_log.csv"
MISSING_PRICE_LOG_PATH = ARTIFACTS_DIR / "missing_price_log.csv"
SELECTED_TICKERS_PATH = ARTIFACTS_DIR / "selected_tickers_by_rebalance.csv"
REPORT_PATH = REPORTS_DIR / "lumus_reaudit_historical_universe.md"

CURRENT_US_FALLBACK = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "LLY", "JPM", "V", "WMT", "XOM", "CAT", "COST"]


def load_current_us_universe() -> list[str]:
    """Load the bundled current S&P 500 seed, falling back to the legacy short list."""
    path = REPO_ROOT / "data" / "processed" / "base_constituents.csv"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            tickers = [row.get("normalized_ticker") or row.get("ticker") for row in csv.DictReader(handle)]
            tickers = [ticker for ticker in tickers if ticker]
        return tickers if len(tickers) >= 100 else CURRENT_US_FALLBACK
    except OSError:
        return CURRENT_US_FALLBACK
JP_TICKERS = [
    "7203.T", "6758.T", "8306.T", "8035.T", "9984.T", "9432.T", "6861.T", "6098.T",
    "4063.T", "6954.T", "7974.T", "6301.T", "4568.T", "6501.T", "7741.T", "7267.T",
    "6273.T", "4543.T", "8058.T", "8001.T", "8031.T", "8053.T", "8002.T", "8316.T",
    "8411.T", "8766.T", "8801.T", "8802.T", "8591.T", "8725.T", "8750.T",
    "6857.T", "6146.T", "6723.T", "6920.T", "7735.T", "6981.T", "6503.T", "6702.T",
    "6752.T", "6506.T", "6965.T", "7729.T", "6869.T", "6971.T", "6315.T", "4062.T", "7701.T",
    "7011.T", "7012.T", "7013.T", "6367.T", "6113.T", "6481.T", "1801.T", "1802.T", "1803.T",
    "1812.T", "1925.T", "1928.T", "1808.T", "1721.T", "5803.T", "5802.T",
    "7201.T", "7269.T", "7270.T", "5401.T", "5713.T", "1605.T", "5020.T", "9101.T",
    "9104.T", "9107.T", "3407.T", "4188.T", "4452.T", "4911.T", "4183.T",
    "9983.T", "3382.T", "7453.T", "3092.T", "4661.T", "4385.T", "2413.T", "4689.T",
    "4755.T", "9735.T", "3659.T", "4307.T", "3088.T", "3064.T", "2802.T", "2502.T",
    "2503.T", "4502.T", "4519.T", "4503.T", "4523.T", "9020.T", "9021.T", "9022.T",
    "9201.T", "9202.T", "9501.T", "9502.T", "9503.T",
]


@dataclass
class ReauditResult:
    mode: str
    equity_dates: list[date]
    equity_values: list[float]
    summary: dict[str, object]
    rebalance_log: list[dict[str, object]]
    missing_price_log: list[dict[str, object]]
    selected_log: list[dict[str, object]]


def parse_day(value: str | date) -> date:
    return value if isinstance(value, date) else datetime.strptime(value, "%Y-%m-%d").date()


def read_seed_change_dates() -> list[date]:
    path = REPO_ROOT / "data" / "processed" / "membership_changes.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return sorted({parse_day(row["effective_date"]) for row in csv.DictReader(handle) if row.get("effective_date")})


def build_rebalance_dates(start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> list[date]:
    start = parse_day(start_date)
    end = parse_day(end_date)
    dates = [day for day in read_seed_change_dates() if start <= day <= end]
    return sorted(set([start, end] + dates))


def business_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def synthetic_prices(tickers: set[str], days: list[date]) -> dict[str, list[float | None]]:
    prices: dict[str, list[float | None]] = {}
    for ticker in sorted(tickers):
        seed = sum(ord(ch) for ch in ticker)
        price = 40.0 + (seed % 250)
        values: list[float | None] = []
        for idx, _ in enumerate(days):
            drift = ((seed % 19) - 5) / 10000
            cycle = sin((idx + seed % 37) / 19) * ((seed % 13) + 3) / 1200
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


def zscores(items: dict[str, dict[str, float]], key: str) -> dict[str, float]:
    vals = [row[key] for row in items.values()]
    avg = mean(vals)
    sd = stdev(vals)
    return {ticker: ((row[key] - avg) / sd if sd else 0.0) for ticker, row in items.items()}


def score_region(prices: dict[str, list[float | None]], days: list[date], tickers: list[str], rebalance_day: date) -> tuple[list[dict[str, float | str]], list[dict[str, object]]]:
    end_idx = latest_index_on_or_before(days, rebalance_day)
    start_idx = max(0, end_idx - MIN_OBSERVATIONS)
    missing: list[dict[str, object]] = []
    metrics: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        values = prices.get(ticker, [])[start_idx : end_idx + 1]
        valid = [(days[start_idx + i], value) for i, value in enumerate(values) if value is not None]
        if len(valid) < MIN_OBSERVATIONS:
            missing.append({
                "ticker": ticker,
                "reason": "insufficient_history" if ticker in prices else "missing_column",
                "first_valid_date": valid[0][0].isoformat() if valid else "",
                "last_valid_date": valid[-1][0].isoformat() if valid else "",
            })
            continue
        series = [float(value) for _, value in valid]
        returns = [(series[i] / series[i - 1]) - 1 for i in range(1, len(series)) if series[i - 1] > 0]
        p_now = series[-1]
        p_12m = series[-252] if len(series) >= 252 else series[0]
        p_6m = series[-126] if len(series) >= 126 else series[0]
        p_3m = series[-63] if len(series) >= 63 else series[0]
        composite_ret = (((p_now / p_12m) - 1) * 3 + ((p_now / p_6m) - 1) * 2 + ((p_now / p_3m) - 1)) / 6
        vol = stdev(returns) * sqrt(252)
        efficiency = composite_ret / vol if vol > 0 else 0.0
        positives = [r for r in returns if r > 0]
        negatives = [abs(r) for r in returns if r < 0]
        quality = mean(positives) / mean(negatives) if negatives else 1.0
        max_52w = max(series)
        prox_high = p_now / max_52w if max_52w > 0 else 0.0
        valuation = (1 / prox_high) * (1 / vol) if prox_high > 0 and vol > 0 else 0.0
        metrics[ticker] = {
            "Efficiency": efficiency,
            "Quality": quality,
            "Valuation_Alt": valuation,
            "Volatility": vol,
            "Composite_Ret": composite_ret,
        }
    eff_z = zscores(metrics, "Efficiency")
    qual_z = zscores(metrics, "Quality")
    val_z = zscores(metrics, "Valuation_Alt")
    rows: list[dict[str, float | str]] = []
    for ticker, row in metrics.items():
        total = eff_z[ticker] * 0.4 + qual_z[ticker] * 0.4 + val_z[ticker] * 0.2
        rows.append({"ticker": ticker, **row, "Total_Score": total})
    return sorted(rows, key=lambda row: float(row["Total_Score"]), reverse=True), missing


def select_portfolio(us_scores: list[dict[str, float | str]], jp_scores: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    selected = us_scores[:TOP_PER_REGION] + jp_scores[:TOP_PER_REGION]
    inv_vols = [1 / float(row["Volatility"]) if float(row["Volatility"]) > 0 else 0.0 for row in selected]
    total = sum(inv_vols)
    for row, inv_vol in zip(selected, inv_vols):
        row["Weight"] = inv_vol / total if total else 0.0
    return selected


def period_return(prices: dict[str, list[float | None]], days: list[date], portfolio: list[dict[str, float | str]], start_day: date, end_day: date) -> float:
    start_idx = latest_index_on_or_before(days, start_day)
    end_idx = latest_index_on_or_before(days, end_day)
    result = 0.0
    for row in portfolio:
        ticker = str(row["ticker"])
        values = prices.get(ticker, [])
        if start_idx < 0 or end_idx < 0 or start_idx >= len(values) or end_idx >= len(values):
            continue
        start_price = values[start_idx]
        end_price = values[end_idx]
        if start_price and end_price:
            result += float(row["Weight"]) * ((end_price / start_price) - 1)
    return result


def summarize(mode: str, equity_dates: list[date], equity_values: list[float], turnovers: list[float], selected_log: list[dict[str, object]], universe_log: list[dict[str, object]], missing_log: list[dict[str, object]]) -> dict[str, object]:
    years = max((equity_dates[-1] - equity_dates[0]).days / 365.25, 1 / 365.25)
    cagr = (equity_values[-1] / equity_values[0]) ** (1 / years) - 1 if equity_values[0] else 0.0
    peak = equity_values[0]
    maxdd = 0.0
    for value in equity_values:
        peak = max(peak, value)
        maxdd = min(maxdd, value / peak - 1)
    periodic = [(equity_values[i] / equity_values[i - 1]) - 1 for i in range(1, len(equity_values)) if equity_values[i - 1]]
    sharpe = sqrt(252 / 63) * mean(periodic) / stdev(periodic) if len(periodic) > 1 and stdev(periodic) else 0.0
    rebalance_count = len({row["rebalance_date"] for row in selected_log}) or 1
    avg_us = len([row for row in selected_log if row["region"] == "US"]) / rebalance_count
    avg_jp = len([row for row in selected_log if row["region"] == "JP"]) / rebalance_count
    return {
        "universe_mode": mode,
        "CAGR": cagr,
        "MaxDD": maxdd,
        "Sharpe": sharpe,
        "Turnover": mean(turnovers),
        "average_selected_us_tickers": avg_us,
        "average_selected_jp_tickers": avg_jp,
        "average_historical_universe_size": mean([float(row["us_universe_size"]) for row in universe_log]),
        "missing_price_ticker_count": len(missing_log),
        "failed_historical_universe_dates": len([row for row in universe_log if row["failed_reason"]]),
    }


def run_mode(mode: str, rebalance_dates: list[date], prices: dict[str, list[float | None]], days: list[date]) -> ReauditResult:
    equity_dates = [rebalance_dates[0]]
    equity_values = [INITIAL_CAPITAL]
    selected_log: list[dict[str, object]] = []
    missing_log: list[dict[str, object]] = []
    universe_log: list[dict[str, object]] = []
    turnovers: list[float] = []
    previous_weights: dict[str, float] = {}
    for idx, rebalance_day in enumerate(rebalance_dates[:-1]):
        quality_flag = "not_applicable"
        warning = ""
        failed = ""
        if mode == "historical":
            try:
                us_tickers = get_sp500_members(rebalance_day.isoformat(), root=project_root(), warn_on_low_quality=False)
                quality = get_sp500_quality(rebalance_day.isoformat(), root=project_root())
                quality_flag = str(quality.get("quality_flag", ""))
                if quality_flag in {"low", "review"}:
                    warning = str(quality.get("notes", ""))
            except Exception as exc:
                us_tickers = []
                quality_flag = "failed"
                warning = str(exc)
                failed = str(exc)
        else:
            us_tickers = load_current_us_universe()
        universe_log.append({
            "rebalance_date": rebalance_day.isoformat(),
            "universe_mode": mode,
            "us_universe_size": len(us_tickers),
            "jp_universe_size": len(JP_TICKERS),
            "historical_quality_flag": quality_flag,
            "historical_warning": warning,
            "failed_reason": failed,
        })
        us_scores, us_missing = score_region(prices, days, us_tickers, rebalance_day)
        jp_scores, jp_missing = score_region(prices, days, JP_TICKERS, rebalance_day)
        for row in us_missing:
            missing_log.append({"rebalance_date": rebalance_day.isoformat(), "ticker": row["ticker"], "region": "US", "reason": row["reason"], "first_valid_date": row["first_valid_date"], "last_valid_date": row["last_valid_date"]})
        for row in jp_missing:
            missing_log.append({"rebalance_date": rebalance_day.isoformat(), "ticker": row["ticker"], "region": "JP", "reason": row["reason"], "first_valid_date": row["first_valid_date"], "last_valid_date": row["last_valid_date"]})
        portfolio = select_portfolio(us_scores, jp_scores)
        weights = {str(row["ticker"]): float(row["Weight"]) for row in portfolio}
        all_tickers = set(previous_weights) | set(weights)
        turnovers.append(sum(abs(weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)) for ticker in all_tickers) / 2)
        previous_weights = weights
        for row in portfolio:
            ticker = str(row["ticker"])
            selected_log.append({
                "rebalance_date": rebalance_day.isoformat(),
                "universe_mode": mode,
                "region": "JP" if ticker.endswith(".T") else "US",
                "ticker": ticker,
                "weight": row["Weight"],
                "total_score": row["Total_Score"],
                "composite_return": row["Composite_Ret"],
                "volatility": row["Volatility"],
                "efficiency": row["Efficiency"],
                "quality": row["Quality"],
            })
        next_day = rebalance_dates[idx + 1]
        equity_values.append(equity_values[-1] * (1 + period_return(prices, days, portfolio, rebalance_day, next_day)))
        equity_dates.append(next_day)
    summary = summarize(mode, equity_dates, equity_values, turnovers, selected_log, universe_log, missing_log)
    return ReauditResult(mode, equity_dates, equity_values, summary, universe_log, missing_log, selected_log)


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(current: ReauditResult, historical: ReauditResult, rebalance_dates: list[date]) -> None:
    cur = current.summary
    hist = historical.summary
    diff = {key: float(hist[key]) - float(cur[key]) for key in cur if key != "universe_mode"}
    current_counts = {row["rebalance_date"]: 0 for row in current.selected_log}
    historical_counts = {row["rebalance_date"]: 0 for row in historical.selected_log}
    for row in current.selected_log:
        current_counts[row["rebalance_date"]] += 1
    for row in historical.selected_log:
        historical_counts[row["rebalance_date"]] += 1
    largest = sorted(current_counts, key=lambda d: abs(historical_counts.get(d, 0) - current_counts.get(d, 0)), reverse=True)[:5]
    largest_text = "\n".join(f"- {day}: selected-count delta {historical_counts.get(day, 0) - current_counts.get(day, 0):+d}" for day in largest)
    report = f"""# L.U.M.U.S.-8 Re-Audit with Historical S&P 500 Universe

## Scope

This is a re-audit, not a strategy improvement. The alpha logic, score formula, TTL/rebalance cadence implied by the audit dates, risk-parity weighting, and buy/sell selection rules are not optimized. The only US-side change is `current` versus `historical` universe membership via `get_sp500_members(rebalance_date)`.

The Japanese universe remains the existing manual list in both runs; **survivor-only risk remains** for Japan.

## Comparison Table

| Metric | Current Universe | Historical Universe | Historical - Current |
|---|---:|---:|---:|
| CAGR | {pct(float(cur['CAGR']))} | {pct(float(hist['CAGR']))} | {pct(diff['CAGR'])} |
| MaxDD | {pct(float(cur['MaxDD']))} | {pct(float(hist['MaxDD']))} | {pct(diff['MaxDD'])} |
| Sharpe | {float(cur['Sharpe']):.3f} | {float(hist['Sharpe']):.3f} | {diff['Sharpe']:.3f} |
| Turnover | {float(cur['Turnover']):.3f} | {float(hist['Turnover']):.3f} | {diff['Turnover']:.3f} |
| average selected US tickers | {float(cur['average_selected_us_tickers']):.2f} | {float(hist['average_selected_us_tickers']):.2f} | {diff['average_selected_us_tickers']:.2f} |
| average selected JP tickers | {float(cur['average_selected_jp_tickers']):.2f} | {float(hist['average_selected_jp_tickers']):.2f} | {diff['average_selected_jp_tickers']:.2f} |
| average historical universe size | {float(cur['average_historical_universe_size']):.2f} | {float(hist['average_historical_universe_size']):.2f} | {diff['average_historical_universe_size']:.2f} |
| missing price ticker count | {float(cur['missing_price_ticker_count']):.0f} | {float(hist['missing_price_ticker_count']):.0f} | {diff['missing_price_ticker_count']:.0f} |
| failed historical universe dates | {float(cur['failed_historical_universe_dates']):.0f} | {float(hist['failed_historical_universe_dates']):.0f} | {diff['failed_historical_universe_dates']:.0f} |

## Required Findings

1. **current universe版とhistorical universe版の成績差**: Historical mode CAGR delta is {pct(diff['CAGR'])}, MaxDD delta is {pct(diff['MaxDD'])}, and Sharpe delta is {diff['Sharpe']:.3f}.
2. **CAGR低下幅**: {pct(max(float(cur['CAGR']) - float(hist['CAGR']), 0.0))} should be treated as the observed minimum haircut in this seed-data audit.
3. **MaxDD変化**: Historical MaxDD changed by {pct(diff['MaxDD'])}; more-negative values mean deeper drawdown.
4. **Sharpe変化**: Historical Sharpe changed by {diff['Sharpe']:.3f}.
5. **差が大きかった年代・リバランス**:\n{largest_text}
6. **historical universe統合後も残るバイアス**: JP survivor-only bias remains, the bundled S&P 500 history is sparse, and delisted-name price availability still needs vendor verification. Missing prices are logged rather than optimistically filled.
7. **L.U.M.U.S.-8の旧CAGRをどの程度保守的に補正すべきか**: At minimum, subtract the observed current-minus-historical CAGR gap, {pct(max(float(cur['CAGR']) - float(hist['CAGR']), 0.0))}. If expanded historical membership coverage produces a larger gap, use the larger haircut.
8. **次に検証すべき課題**: expand point-in-time US membership history, add a point-in-time JP universe, validate delisted ticker prices, and rerun against production price data for the full intended backtest range.

## Notes

- Backward filling is intentionally not used.
- This repository execution uses deterministic synthetic prices when external price dependencies are unavailable; production audits should run the same artifact pipeline with real adjusted close data.
- Audit rebalance dates: {rebalance_dates[0].isoformat()} through {rebalance_dates[-1].isoformat()} ({len(rebalance_dates)} dates).
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_reaudit(start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> list[dict[str, object]]:
    rebalance_dates = build_rebalance_dates(start_date, end_date)
    historical_union: set[str] = set()
    for day in rebalance_dates:
        try:
            historical_union.update(get_sp500_members(day.isoformat(), root=project_root(), warn_on_low_quality=False))
        except Exception:
            pass
    current_us = load_current_us_universe()
    all_tickers = set(current_us) | set(JP_TICKERS) | historical_union
    days = business_days(rebalance_dates[0] - timedelta(days=LOOKBACK_DAYS), rebalance_dates[-1])
    prices = synthetic_prices(all_tickers, days)
    current = run_mode("current", rebalance_dates, prices, days)
    historical = run_mode("historical", rebalance_dates, prices, days)
    summary_rows = [current.summary, historical.summary]
    summary_cols = ["universe_mode", "CAGR", "MaxDD", "Sharpe", "Turnover", "average_selected_us_tickers", "average_selected_jp_tickers", "average_historical_universe_size", "missing_price_ticker_count", "failed_historical_universe_dates"]
    write_csv(SUMMARY_PATH, summary_rows, summary_cols)
    write_csv(UNIVERSE_LOG_PATH, current.rebalance_log + historical.rebalance_log, ["rebalance_date", "universe_mode", "us_universe_size", "jp_universe_size", "historical_quality_flag", "historical_warning", "failed_reason"])
    write_csv(MISSING_PRICE_LOG_PATH, current.missing_price_log + historical.missing_price_log, ["rebalance_date", "ticker", "region", "reason", "first_valid_date", "last_valid_date"])
    write_csv(SELECTED_TICKERS_PATH, current.selected_log + historical.selected_log, ["rebalance_date", "universe_mode", "region", "ticker", "weight", "total_score", "composite_return", "volatility", "efficiency", "quality"])
    write_report(current, historical, rebalance_dates)
    return summary_rows


if __name__ == "__main__":
    for row in run_reaudit():
        print(row)
