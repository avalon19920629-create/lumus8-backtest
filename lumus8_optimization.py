"""Constrained L.U.M.U.S.-8 allocation research built on the core backtester.

The optimizer deliberately compares several objectives and a robust average rather
than presenting one in-sample optimum as an investable answer. Search scores use
constant-weight daily returns for tractability; reported results use the repository's
monthly-rebalanced backtest model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lumus8_backtest import (
    PORTFOLIOS, TRADING_DAYS, annual_returns, backtest_portfolio, download_prices,
    metrics, prepare_prices, stress_analysis,
)

ASSETS = ("VT", "BNDX", "TLT", "TIP", "GLD", "XLRE", "DBC", "SHY", "BTC-USD")
BOUNDS = {
    "VT": (0.15, 0.45), "BNDX": (0.0, 0.15), "TLT": (0.05, 0.25),
    "TIP": (0.05, 0.20), "GLD": (0.05, 0.20), "XLRE": (0.0, 0.15),
    "DBC": (0.0, 0.10), "SHY": (0.05, 0.20), "BTC-USD": (0.0, 0.05),
}
GROUPS = {
    "Bonds": (("BNDX", "TLT", "TIP", "SHY"), (0.25, 0.55)),
    "RealAssets": (("GLD", "XLRE", "DBC"), (0.15, 0.40)),
    "GrowthRisk": (("VT", "BTC-USD"), (0.15, 0.50)),
}
OBJECTIVES = (
    "LUMUS_OPT_MAX_SHARPE", "LUMUS_OPT_MAX_SORTINO", "LUMUS_OPT_MAX_CALMAR",
    "LUMUS_OPT_MAX_CAGR_MDD25", "LUMUS_OPT_MIN_VOL_CAGR10",
)
OPTIMIZED_NAMES = (*OBJECTIVES, "LUMUS_OPT_ROBUST_MEDIAN")
COMPARISON_NAMES = (
    "SP500", "60_40", "DALIO_AW", "LUMUS_EX_ALPHA_VT_REPLACED",
    "LUMUS_GROWTH_SPY_QQQ_50_50", "LUMUS_XLRE",
)


def constraints_satisfied(weights: Mapping[str, float], tolerance: float = 1e-8) -> bool:
    w = pd.Series(weights, dtype=float).reindex(ASSETS, fill_value=0.0)
    if abs(w.sum() - 1) > tolerance:
        return False
    if any(w[a] < BOUNDS[a][0] - tolerance or w[a] > BOUNDS[a][1] + tolerance for a in ASSETS):
        return False
    return all(lo - tolerance <= w[list(names)].sum() <= hi + tolerance for names, (lo, hi) in GROUPS.values())


def feasible_weights(samples: int, seed: int = 8) -> pd.DataFrame:
    """Generate a deterministic, broad feasible set plus the formal baseline."""
    rng = np.random.default_rng(seed)
    lower = np.array([BOUNDS[a][0] for a in ASSETS])
    upper = np.array([BOUNDS[a][1] for a in ASSETS])
    accepted: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < samples and attempts < samples * 400:
        batch = rng.uniform(lower, upper, size=(max(1000, samples), len(ASSETS)))
        batch /= batch.sum(axis=1, keepdims=True)
        for row in batch:
            if constraints_satisfied(dict(zip(ASSETS, row))):
                accepted.append(row)
                if len(accepted) == samples:
                    break
        attempts += len(batch)
    if len(accepted) < samples:
        raise RuntimeError(f"Could only generate {len(accepted)} feasible portfolios")
    formal = pd.Series(PORTFOLIOS["LUMUS_EX_ALPHA_VT_REPLACED"]).reindex(ASSETS).to_numpy()
    accepted[0] = formal
    return pd.DataFrame(accepted, columns=ASSETS)


def search_statistics(asset_returns: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Score feasible candidates using daily constant-weight return streams."""
    rows = []
    values = asset_returns[list(ASSETS)].to_numpy()
    years = len(asset_returns) / TRADING_DAYS
    for weights in candidates.to_numpy():
        daily = values @ weights
        equity = np.cumprod(1 + daily)
        cagr = equity[-1] ** (1 / years) - 1
        vol = daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
        downside = daily[daily < 0].std(ddof=1)
        maxdd = np.min(equity / np.maximum.accumulate(equity) - 1)
        rows.append({
            "CAGR": cagr, "Volatility": vol,
            "Sharpe": daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS),
            "Sortino": daily.mean() / downside * np.sqrt(TRADING_DAYS) if downside else np.nan,
            "MaxDD": maxdd, "Calmar": cagr / abs(maxdd) if maxdd else np.nan,
        })
    return pd.DataFrame(rows, index=candidates.index)


def select_candidates(weights: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    selected: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    selected[OBJECTIVES[0]] = weights.loc[scores["Sharpe"].idxmax()]
    selected[OBJECTIVES[1]] = weights.loc[scores["Sortino"].idxmax()]
    selected[OBJECTIVES[2]] = weights.loc[scores["Calmar"].idxmax()]
    mdd = scores[scores["MaxDD"] >= -0.25]
    if mdd.empty:
        idx = scores["MaxDD"].idxmax()
        notes[OBJECTIVES[3]] = "MaxDD -25% feasible candidate not found; shallowest-MaxDD fallback used."
    else:
        idx = mdd["CAGR"].idxmax()
        notes[OBJECTIVES[3]] = "MaxDD -25% constraint satisfied in search sample."
    selected[OBJECTIVES[3]] = weights.loc[idx]
    eligible = scores[scores["CAGR"] >= 0.10]
    if eligible.empty:
        idx = scores["CAGR"].idxmax()
        notes[OBJECTIVES[4]] = "CAGR 10% feasible candidate not found; highest-CAGR feasible fallback used."
    else:
        idx = eligible["Volatility"].idxmin()
        notes[OBJECTIVES[4]] = "CAGR 10% constraint satisfied in search sample."
    selected[OBJECTIVES[4]] = weights.loc[idx]
    # Arithmetic mean is a convex combination, so every linear bound/group constraint remains valid.
    selected["LUMUS_OPT_ROBUST_MEDIAN"] = pd.DataFrame(selected).T.mean()
    result = pd.DataFrame(selected).T.reindex(columns=ASSETS)
    assert all(constraints_satisfied(row) for _, row in result.iterrows())
    return result, notes


def sp500_deltas(metric_table: pd.DataFrame) -> pd.DataFrame:
    spy = metric_table.loc["SP500"]
    result = metric_table[["CAGR", "Volatility", "MaxDD", "Calmar"]].copy()
    for column in result:
        result[f"{column}DiffVsSP500"] = result[column] - spy[column]
    return result[[c for c in result if c.endswith("DiffVsSP500")]]


def grade(row: pd.Series, spy: pd.Series) -> str:
    if row.CAGR >= .9 * spy.CAGR and row.Volatility <= .7 * spy.Volatility and row.MaxDD >= spy.MaxDD + .10 and row.Calmar >= spy.Calmar:
        return "S"
    if row.CAGR >= .8 * spy.CAGR and row.Volatility < spy.Volatility and row.MaxDD > spy.MaxDD and row.Sharpe > spy.Sharpe and row.Sortino > spy.Sortino and row.Calmar > spy.Calmar:
        return "A"
    if row.MaxDD >= spy.MaxDD + .10 and row.WorstYear > spy.WorstYear and (pd.isna(spy.RecoveryDays) or row.RecoveryDays < spy.RecoveryDays):
        return "B"
    if row.Volatility < spy.Volatility and row.MaxDD > spy.MaxDD:
        return "C"
    return "D"


def save_optimization_plots(equities: pd.DataFrame, scores: pd.DataFrame, output_dir: Path) -> None:
    ax = scores.plot.scatter(x="Volatility", y="CAGR", c="Sharpe", colormap="viridis", figsize=(10, 7), alpha=.45)
    ax.set_title("Constrained feasible set: return / volatility frontier")
    ax.grid(alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(output_dir / "efficient_frontier.png", dpi=160); plt.close(ax.figure)
    ax = equities.plot(logy=True, figsize=(13, 7), title="Optimized portfolios: growth of $1")
    ax.grid(alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(output_dir / "optimized_equity_curves.png", dpi=160); plt.close(ax.figure)
    drawdowns = equities / equities.cummax() - 1
    ax = drawdowns.plot(figsize=(13, 7), title="Optimized portfolio drawdowns")
    ax.grid(alpha=.25); ax.figure.tight_layout(); ax.figure.savefig(output_dir / "optimized_drawdowns.png", dpi=160); plt.close(ax.figure)


def write_report(tables: Mapping[str, pd.DataFrame], notes: Mapping[str, str], output_dir: Path) -> None:
    metrics_table = tables["optimized_common_period_metrics"]
    candidates = metrics_table.loc[list(OPTIMIZED_NAMES)].copy()
    candidates["_GradeRank"] = candidates["Grade"].map({"S": 0, "A": 1, "B": 2, "C": 3, "D": 4})
    candidates = candidates.sort_values(["_GradeRank", "Calmar"], ascending=[True, False])
    best = candidates.index[0]
    report = f"""# L.U.M.U.S.-8 制約付き最適化検証

## 位置づけ
本検証は投資助言ではなく、既存9資産の制約付き配分候補を複数目的で比較するポートフォリオ設計研究である。探索スコアは計算効率のため日次定率リバランス近似、最終報告値は既存コードと同じ月次リバランスで算出した。

## 最有力候補
合格判定とCalmarのバランス上の先頭候補は **{best}**（Grade {candidates.loc[best, 'Grade']}、CAGR {candidates.loc[best, 'CAGR']:.2%}、MaxDD {candidates.loc[best, 'MaxDD']:.2%}、Calmar {candidates.loc[best, 'Calmar']:.2f}）。最大リターンだけでなくCAGR・MaxDD・Calmar・WorstYearのバランスで最終採用候補を判断する。

## 制約と方法
全ウェイト100%、ロングオンリー、指定された資産別上下限、債券25〜55%、実物資産15〜40%、成長・リスク資産15〜50%を適用した。`LUMUS_OPT_ROBUST_MEDIAN` は5目的候補の算術平均であり、線形制約を保つロバスト候補である。

## 過剰適合に関する必須警告
- 最適化結果は将来の最適比率を保証しない。
- 特にBTCやNASDAQ的成長資産は、検証期間の影響を強く受ける。
- 単一の最適解ではなく、複数の最適化目的で安定して現れる比率を重視する。
- 最終採用候補は、最大リターンではなく、CAGR・MaxDD・Calmar・WorstYearのバランスで選定する。
- 本実装はインサンプル探索であり、walk-forward検証は今後の必須追加検証である。

## 実行注記
{chr(10).join(f'- {k}: {v}' for k, v in notes.items()) or '- すべての目的制約に適格候補あり。'}
"""
    (output_dir / "optimized_report.md").write_text(report, encoding="utf-8")


def run_optimization(raw_prices: pd.DataFrame, output_dir: Path, samples: int = 5000, seed: int = 8) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices, _ = prepare_prices(raw_prices)
    prices = prices.reindex(raw_prices["SPY"].dropna().index)
    common = prices[list(ASSETS)].dropna()
    asset_returns = common.pct_change(fill_method=None).dropna()
    universe = feasible_weights(samples, seed)
    scores = search_statistics(asset_returns, universe)
    optimized, notes = select_candidates(universe, scores)
    portfolios = {name: PORTFOLIOS[name] for name in COMPARISON_NAMES}
    portfolios.update({name: row.dropna().to_dict() for name, row in optimized.iterrows()})
    results = {name: backtest_portfolio(prices.loc[common.index[0]:], weights) for name, weights in portfolios.items()}
    equities = pd.DataFrame({name: result.equity for name, result in results.items()})
    metric_table = pd.DataFrame({name: metrics(result) for name, result in results.items()}).T
    annual = annual_returns({name: result.equity for name, result in results.items()})
    comparison_2022 = stress_analysis({name: result.equity for name, result in results.items()}).loc["INFLATION_2022"]
    metric_table["Return2022"] = annual.loc[2022] if 2022 in annual.index else np.nan
    spy = metric_table.loc["SP500"]
    metric_table["Grade"] = [grade(row, spy) for _, row in metric_table.iterrows()]
    tables = {
        "optimized_weights": optimized, "optimized_metrics": metric_table.loc[list(OPTIMIZED_NAMES)],
        "optimized_common_period_metrics": metric_table, "optimized_annual_returns": annual,
        "optimized_equity_curves": equities, "optimized_drawdowns": equities / equities.cummax() - 1,
        "optimized_comparison_2022": comparison_2022, "optimized_sp500_deltas": sp500_deltas(metric_table),
    }
    for name, table in tables.items(): table.to_csv(output_dir / f"{name}.csv")
    save_optimization_plots(equities, scores, output_dir)
    write_report(tables, notes, output_dir)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-10-08")
    parser.add_argument("--end", default=None)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    tickers = sorted(set(ASSETS) | {t for name in COMPARISON_NAMES for t in PORTFOLIOS[name]})
    raw = download_prices(tickers, args.start, args.end, args.output_dir / "cache")
    tables = run_optimization(raw, args.output_dir, args.samples, args.seed)
    print("\n=== 1. Optimized weights ===\n", tables["optimized_weights"].to_string())
    print("\n=== 2. Optimized metrics ===\n", tables["optimized_metrics"].to_string())
    print("\n=== 3. SP500 comparison deltas ===\n", tables["optimized_sp500_deltas"].to_string())
    print("\n=== 4. 2022 comparison ===\n", tables["optimized_comparison_2022"].to_string())
    ranked = tables["optimized_metrics"].copy()
    ranked["_GradeRank"] = ranked["Grade"].map({"S": 0, "A": 1, "B": 2, "C": 3, "D": 4})
    ranked = ranked.sort_values(["_GradeRank", "Calmar"], ascending=[True, False]).drop(columns="_GradeRank")
    print("\n=== 5. Best candidate summary ===\n", ranked.head(1).to_string())
    print("\n=== 6. Overfitting warnings ===\nOptimization does not guarantee future-optimal weights; prefer stable multi-objective weights and balanced CAGR/MaxDD/Calmar/WorstYear.")


if __name__ == "__main__": main()
