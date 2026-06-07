"""Audit L.U.M.U.S.-8 rebalance bands and year-end policies without changing weights.

This is a bounded policy comparison, not an optimizer. It retains the current normalized
allocation and the repository's IEF/GLD/BTC-USD historical proxies, while comparing eight
predefined rebalance rules with a simple average-cost tax ledger.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from allocation_robustness_audit import ALLOCATIONS, ASSETS, CORE_ASSETS, CURRENT, normalized_weights
from lumus8_backtest import download_prices, prepare_prices

TRADING_DAYS = 252
CURRENT_POLICY = "Current_5_10_Annual"
DECISIONS = {"現行リバランス維持", "リバランス幅を緩める候補", "年末リバランス条件付き化候補", "売却抑制リバランス候補", "追加検証候補", "不採用"}


@dataclass(frozen=True)
class RebalancePolicy:
    description: str
    core_band: float
    support_band: float
    year_end_rule: str
    sell_suppression: bool = False


POLICIES: dict[str, RebalancePolicy] = {
    CURRENT_POLICY: RebalancePolicy("現行：コア±5 / サポート±10 / 年末あり", .05, .10, "annual"),
    "Loose_7_5_12_5_Annual": RebalancePolicy("緩め：コア±7.5 / サポート±12.5 / 年末あり", .075, .125, "annual"),
    "Looser_10_15_Annual": RebalancePolicy("さらに緩め：コア±10 / サポート±15 / 年末あり", .10, .15, "annual"),
    "Threshold_Only": RebalancePolicy("年末なし：乖離時のみ", .05, .10, "none"),
    "Biennial_Year_End": RebalancePolicy("年末隔年：偶数年末のみ", .05, .10, "biennial"),
    "Conditional_Year_End": RebalancePolicy("年末条件付き：最大乖離がバンドの50%以上のときのみ", .05, .10, "conditional"),
    "Loss_Harvest_Year_End": RebalancePolicy("年末損出し優先：含み損ポジションがある場合のみ", .05, .10, "loss_only"),
    "Sell_Suppressed": RebalancePolicy("売却抑制版：乖離時にバンド境界までの最小売買", .05, .10, "annual", True),
}


@dataclass
class AuditResult:
    equity: pd.Series
    gross_equity: pd.Series
    events: list[dict[str, object]]
    turnover: float
    tax_cost: float
    slippage: float
    fees: float
    trade_count: int


def _bands(policy: RebalancePolicy) -> pd.Series:
    return pd.Series({asset: policy.core_band if asset in CORE_ASSETS else policy.support_band for asset in ASSETS})


def _minimum_trade_weights(current: pd.Series, target: pd.Series, bands: pd.Series) -> pd.Series:
    """Project weights into permitted bands while minimizing sales/trading.

    The bounded-simplex projection clips breached sleeves to their nearest boundary and
    allocates only the residual needed to make weights sum to one. This avoids restoring
    every sleeve to target and therefore suppresses unnecessary sales.
    """
    lower, upper = (target - bands).clip(lower=0), target + bands
    weights = current.clip(lower=lower, upper=upper)
    for _ in range(10):
        residual = 1.0 - float(weights.sum())
        if abs(residual) < 1e-12:
            break
        capacity = (upper - weights) if residual > 0 else (weights - lower)
        available = capacity[capacity > 1e-12]
        if available.empty:
            raise ValueError("Could not project weights into rebalance bands")
        weights.loc[available.index] += residual * available / available.sum()
        weights = weights.clip(lower=lower, upper=upper)
    return weights / weights.sum()


def _trade(values: pd.Series, basis: pd.Series, desired_weights: pd.Series, tax_rate: float,
           slippage_bps: float, fee_bps: float) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    total = float(values.sum()); desired = desired_weights * total
    sells = (values - desired).clip(lower=0); buys = (desired - values).clip(lower=0)
    sold_basis = (basis * (sells / values.replace(0, np.nan))).fillna(0)
    taxable_gain = float((sells - sold_basis).clip(lower=0).sum())
    tax = taxable_gain * tax_rate; traded = float(sells.sum() + buys.sum())
    slippage = traded * slippage_bps / 10000; fees = traded * fee_bps / 10000
    drag = tax + slippage + fees; scale = max(0.0, 1 - drag / total) if total else 0.0
    new_values = desired * scale
    new_basis = ((basis - sold_basis) + buys) * scale
    stats = {"traded_notional": traded, "turnover": traded / total if total else 0.0,
             "sold_notional": float(sells.sum()), "bought_notional": float(buys.sum()),
             "taxable_gain": taxable_gain, "tax_cost": tax, "slippage": slippage, "fees": fees,
             "trade_count": int((sells > 1e-12).sum() + (buys > 1e-12).sum())}
    return new_values, new_basis, stats


def _year_end_trigger(policy: RebalancePolicy, date: pd.Timestamp, drift: pd.Series,
                      bands: pd.Series, values: pd.Series, basis: pd.Series) -> bool:
    if policy.year_end_rule == "annual": return True
    if policy.year_end_rule == "none": return False
    if policy.year_end_rule == "biennial": return date.year % 2 == 0
    if policy.year_end_rule == "conditional": return bool((drift.abs() >= bands * .5).any())
    if policy.year_end_rule == "loss_only": return bool((values < basis - 1e-12).any())
    raise ValueError(f"Unknown year-end rule: {policy.year_end_rule}")


def simulate(prices: pd.DataFrame, name: str, policy: RebalancePolicy, tax_rate: float = .20315,
             slippage_bps: float = 5, fee_bps: float = 0) -> AuditResult:
    target = normalized_weights(ALLOCATIONS[CURRENT]); panel = prices.loc[:, ASSETS].dropna()
    if len(panel) < 2: raise ValueError("At least two common price observations are required")
    values = target.copy(); basis = target.copy(); gross_values = target.copy(); gross_basis = target.copy()
    bands = _bands(policy); net_points = []; gross_points = []; events = []
    turnover = tax = slippage = fees = 0.0; trade_count = 0
    for i, date in enumerate(panel.index):
        if i:
            relative = panel.iloc[i] / panel.iloc[i - 1]; values *= relative; gross_values *= relative
        current = values / values.sum(); drift = current - target; breached = drift.abs() > bands
        is_year_end = i < len(panel) - 1 and panel.index[i + 1].year != date.year
        year_end_due = is_year_end and _year_end_trigger(policy, date, drift, bands, values, basis)
        if bool(breached.any()) or year_end_due:
            desired = _minimum_trade_weights(current, target, bands) if policy.sell_suppression else target
            reason = "threshold" if bool(breached.any()) else f"year_end_{policy.year_end_rule}"
            before = float(values.sum())
            values, basis, stats = _trade(values, basis, desired, tax_rate, slippage_bps, fee_bps)
            gross_values, gross_basis, _ = _trade(gross_values, gross_basis, desired, 0, 0, 0)
            if stats["trade_count"]:
                turnover += stats["turnover"]; tax += stats["tax_cost"]; slippage += stats["slippage"]; fees += stats["fees"]
                trade_count += int(stats["trade_count"])
                events.append({"date": date, "policy": name, "reason": reason,
                               "breached_assets": "|".join(breached[breached].index),
                               "portfolio_value_before": before, **stats})
        net_points.append((date, float(values.sum()))); gross_points.append((date, float(gross_values.sum())))
    net = pd.Series(dict(net_points), name=name); gross = pd.Series(dict(gross_points), name=name)
    return AuditResult(net / net.iloc[0], gross / gross.iloc[0], events, turnover, tax, slippage, fees, trade_count)


def _cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan


def _ratio_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().dropna(); maxdd = float((equity / equity.cummax() - 1).min()); downside = returns[returns < 0].std(); cagr = _cagr(equity)
    return {"CAGR": cagr, "MaxDD": maxdd,
            "Sharpe": float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)) if returns.std() else np.nan,
            "Sortino": float(returns.mean() / downside * np.sqrt(TRADING_DAYS)) if downside else np.nan,
            "Calmar": cagr / abs(maxdd) if maxdd else np.nan}


def _annual_returns(equity: pd.Series) -> pd.Series:
    return equity.groupby(equity.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)


def _rolling_win_rate(candidate: pd.Series, baseline: pd.Series, years: int) -> float:
    periods = years * TRADING_DAYS; excess = (candidate / candidate.shift(periods) - baseline / baseline.shift(periods)).dropna()
    return float((excess > 0).mean()) if not excess.empty else np.nan


def _classify(name: str, row: Mapping[str, float], base: Mapping[str, float]) -> str:
    if name == CURRENT_POLICY: return "現行リバランス維持"
    robust = (row["AfterTaxCAGR"] >= base["AfterTaxCAGR"] - .001 and row["AfterTaxCalmar"] >= base["AfterTaxCalmar"] * .95
              and row["MaxDD"] >= base["MaxDD"] - .01 and (np.isnan(row["Excess2022VsCurrent"]) or row["Excess2022VsCurrent"] >= -.01))
    savings = row["Turnover"] <= base["Turnover"] * .9 or row["TaxCost"] <= base["TaxCost"] * .9
    if robust and savings:
        if name == "Sell_Suppressed": return "売却抑制リバランス候補"
        if name.startswith("Loose"): return "リバランス幅を緩める候補"
        return "年末リバランス条件付き化候補"
    if robust: return "追加検証候補"
    return "不採用"


def build_tables(results: Mapping[str, AuditResult]) -> dict[str, pd.DataFrame]:
    equities = pd.DataFrame({n: r.equity for n, r in results.items()}); gross = pd.DataFrame({n: r.gross_equity for n, r in results.items()})
    annual = pd.DataFrame({n: _annual_returns(r.equity) for n, r in results.items()}); years = (equities.index[-1] - equities.index[0]).days / 365.25
    rows = {}; baseline_annual = annual[CURRENT_POLICY]
    for name, result in results.items():
        net, pre = _ratio_metrics(result.equity), _ratio_metrics(result.gross_equity); excess = annual[name] - baseline_annual
        rows[name] = {"Description": POLICIES[name].description, "CoreBand": POLICIES[name].core_band, "SupportBand": POLICIES[name].support_band,
                      "PreTaxCAGR": pre["CAGR"], "AfterTaxCAGR": net["CAGR"], "MaxDD": net["MaxDD"], "AfterTaxCalmar": net["Calmar"],
                      "Sharpe": net["Sharpe"], "Sortino": net["Sortino"], "Turnover": result.turnover, "TradeCount": result.trade_count,
                      "RebalanceCount": len(result.events), "AnnualRebalanceCount": len(result.events) / years,
                      "TaxableEventCount": sum(e["tax_cost"] > 0 for e in result.events), "TaxCost": result.tax_cost,
                      "Slippage": result.slippage, "Fees": result.fees, "CostDragCAGR": pre["CAGR"] - net["CAGR"],
                      "Return2022": float(annual.loc[2022, name]) if 2022 in annual.index else np.nan,
                      "Excess2022VsCurrent": float(excess.loc[2022]) if 2022 in excess.index else np.nan,
                      "AnnualWins": int((excess > 1e-12).sum()), "AnnualLosses": int((excess < -1e-12).sum()), "AnnualTies": int(np.isclose(excess, 0).sum()),
                      "Rolling3YWinRate": _rolling_win_rate(result.equity, results[CURRENT_POLICY].equity, 3),
                      "Rolling5YWinRate": _rolling_win_rate(result.equity, results[CURRENT_POLICY].equity, 5)}
    metrics = pd.DataFrame.from_dict(rows, orient="index"); base = metrics.loc[CURRENT_POLICY]
    metrics["Decision"] = [_classify(name, row, base) for name, row in metrics.iterrows()]
    events = pd.DataFrame([event for result in results.values() for event in result.events])
    return {"metrics": metrics, "annual_returns": annual, "rebalance_events": events, "equity_curves": equities,
            "drawdown_curves": equities / equities.cummax() - 1}


def _overall_decision(metrics: pd.DataFrame) -> tuple[str, str]:
    candidates = metrics.drop(index=CURRENT_POLICY); accepted = candidates[candidates.Decision.isin(DECISIONS - {"現行リバランス維持", "追加検証候補", "不採用"})]
    if accepted.empty: return "現行リバランス維持", "堅牢性条件と売買・税コスト削減条件を同時に満たす代替案がない。"
    score = (accepted.AfterTaxCAGR - accepted.loc[:, "AfterTaxCAGR"].min()) - .001 * accepted.Turnover - accepted.CostDragCAGR
    best = accepted.loc[score.idxmax()]
    return str(best.Decision), f"{score.idxmax()} が堅牢性許容範囲内で売買・税コスト削減を示した。"


def _report_table(frame: pd.DataFrame) -> str:
    """Render a compact fixed-width table without an optional tabulate dependency."""
    return "```\n" + frame.to_string(float_format=lambda value: f"{value:.4f}") + "\n```"


def write_report(tables: Mapping[str, pd.DataFrame], output_dir: Path, start: pd.Timestamp, end: pd.Timestamp,
                 tax_rate: float, slippage_bps: float, fee_bps: float) -> None:
    metrics = tables["metrics"]; decision, reason = _overall_decision(metrics)
    cols = ["Description", "AfterTaxCAGR", "MaxDD", "AfterTaxCalmar", "Sharpe", "Sortino", "Turnover", "TradeCount", "RebalanceCount", "TaxableEventCount", "TaxCost", "CostDragCAGR", "Return2022", "Decision"]
    lines = ["# L.U.M.U.S.-8 リバランス幅・年末条件 頑健性監査", "", "## 【結論】", f"**{decision}** — {reason}", "",
             "## 【比較サマリー】", _report_table(metrics[cols]), "", "## 【年次勝敗・ローリング勝率】",
             _report_table(metrics[["AnnualWins", "AnnualLosses", "AnnualTies", "Rolling3YWinRate", "Rolling5YWinRate", "Excess2022VsCurrent"]]), "",
             "## 【判定基準】", "- 現行比率は固定し、事前定義した8ポリシーのみを比較する。", "- 候補は税後CAGR差-0.10%以内、税後Calmarが現行の95%以上、最大DD悪化1ポイント以内、2022年差-1ポイント以内を要求する。", "- 上記を満たし、TurnoverまたはTaxCostを10%以上削減した場合に具体的な候補分類とする。", "",
             "## 【重要な制約】", f"- 検証期間: {start.date()}〜{end.date()}。税率={tax_rate:.3%}、スリッページ={slippage_bps:g}bps、手数料={fee_bps:g}bps。",
             "- BNDX/GLDM/BTCの長期代理としてIEF/GLD/BTC-USDを使用する。配当課税、損益通算、ロット別取得価額、入出金は再現しない。", "- 年末条件付きはバンドの50%以上の乖離、年末隔年は偶数年末、損出し優先は含み損ポジション存在時と定義した。", "- 売却抑制版は乖離時に目標比率へ戻さず、許容バンド内の最寄り境界へ戻す。年末に乖離がなければ売買しない。", "- 過去データによる予備監査であり、将来成果を保証しない。"]
    (output_dir / "rebalance_band_summary_report_ja.md").write_text("\n".join(lines), encoding="utf-8")


def _plots(tables: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    ax = tables["equity_curves"].plot(figsize=(12, 7), alpha=.82, title="L.U.M.U.S.-8 rebalance policy audit: after-tax equity")
    ax.set_ylabel("Growth of 1"); ax.figure.tight_layout(); ax.figure.savefig(output_dir / "equity_curve.png", dpi=150); plt.close(ax.figure)
    metrics = tables["metrics"]; ax = metrics[["Turnover", "TaxCost"]].plot.bar(figsize=(12, 7), subplots=True, title="Trading and tax burden")
    fig = ax[0].figure; fig.tight_layout(); fig.savefig(output_dir / "trading_tax_burden.png", dpi=150); plt.close(fig)


def run_audit(raw_prices: pd.DataFrame, output_dir: Path, tax_rate: float = .20315,
              slippage_bps: float = 5, fee_bps: float = 0) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True); prices, coverage = prepare_prices(raw_prices); prices = prices.loc[:, ASSETS].dropna()
    results = {name: simulate(prices, name, policy, tax_rate, slippage_bps, fee_bps) for name, policy in POLICIES.items()}; tables = build_tables(results)
    files = {"metrics": "rebalance_band_metrics.csv", "annual_returns": "rebalance_band_annual_returns.csv", "rebalance_events": "rebalance_band_events.csv", "equity_curves": "rebalance_band_equity_curves.csv", "drawdown_curves": "rebalance_band_drawdown_curves.csv"}
    for key, filename in files.items(): tables[key].to_csv(output_dir / filename, index=key != "rebalance_events", index_label="policy" if key == "metrics" else "date")
    coverage.to_csv(output_dir / "price_coverage.csv", index_label="ticker"); _plots(tables, output_dir); write_report(tables, output_dir, prices.index[0], prices.index[-1], tax_rate, slippage_bps, fee_bps)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--start", default="2010-01-01"); parser.add_argument("--end", default=None)
    parser.add_argument("--tax-rate", type=float, default=.20315); parser.add_argument("--slippage-bps", type=float, default=5); parser.add_argument("--fee-bps", type=float, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rebalance_band_audit")); args = parser.parse_args()
    raw = download_prices(ASSETS, args.start, args.end, args.output_dir / "price_cache"); tables = run_audit(raw, args.output_dir, args.tax_rate, args.slippage_bps, args.fee_bps)
    print(tables["metrics"].to_string()); print(f"Artifacts saved under: {args.output_dir.resolve()}")


if __name__ == "__main__": main()
