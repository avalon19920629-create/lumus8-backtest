"""Preliminary robustness audit for conservative L.U.M.U.S.-8 allocations.

This module compares a small, deliberately bounded set of allocation candidates.  It
is not an optimizer.  To preserve the repository's longest common history, IEF and
GLD remain the historical proxies for the current BNDX and GLDM sleeves, as in the
GENKI audit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lumus8_backtest import download_prices, prepare_prices

TRADING_DAYS = 252
CURRENT = "Current_LUMUS8"
CORE_ASSETS = {"VT", "TLT", "TIP", "GLD", "BTC-USD"}
ASSETS = ["VT", "IEF", "TLT", "TIP", "GLD", "XLRE", "DBC", "SHY", "BTC-USD"]

# A bounded, hypothesis-led grid; no data-dependent weights are generated.
ALLOCATIONS: dict[str, dict[str, float]] = {
    CURRENT: {"VT": .25/.90, "IEF": .10/.90, "TLT": .12/.90, "TIP": .10/.90, "GLD": .10/.90, "XLRE": .06/.90, "DBC": .07/.90, "SHY": .05/.90, "BTC-USD": .05/.90},
    "Growth_Heavy": {"VT": .34, "IEF": .09, "TLT": .10, "TIP": .10, "GLD": .10, "XLRE": .07, "DBC": .07, "SHY": .06, "BTC-USD": .07},
    "Defense_Heavy": {"VT": .23, "IEF": .14, "TLT": .16, "TIP": .12, "GLD": .12, "XLRE": .06, "DBC": .06, "SHY": .08, "BTC-USD": .03},
    "Gold_RealAssets_Heavy": {"VT": .25, "IEF": .10, "TLT": .11, "TIP": .12, "GLD": .16, "XLRE": .08, "DBC": .10, "SHY": .04, "BTC-USD": .04},
    "TLT_Light": {"VT": .30, "IEF": .13, "TLT": .08, "TIP": .12, "GLD": .12, "XLRE": .07, "DBC": .08, "SHY": .06, "BTC-USD": .04},
    # IEF is the repository's long-history BNDX sleeve proxy; this is disclosed in the report.
    "BNDX_Proxy_Heavy": {"VT": .27, "IEF": .16, "TLT": .10, "TIP": .11, "GLD": .11, "XLRE": .07, "DBC": .07, "SHY": .07, "BTC-USD": .04},
    "BTC_Low": {"VT": .30, "IEF": .11, "TLT": .14, "TIP": .11, "GLD": .12, "XLRE": .07, "DBC": .08, "SHY": .06, "BTC-USD": .01},
    "BTC_High": {"VT": .25, "IEF": .10, "TLT": .12, "TIP": .10, "GLD": .11, "XLRE": .07, "DBC": .08, "SHY": .06, "BTC-USD": .11},
    "DBC_XLRE_Reduced": {"VT": .31, "IEF": .12, "TLT": .14, "TIP": .12, "GLD": .13, "XLRE": .03, "DBC": .03, "SHY": .07, "BTC-USD": .05},
    "Simple": {"VT": .30, "IEF": .12, "TLT": .14, "TIP": .12, "GLD": .14, "XLRE": 0, "DBC": 0, "SHY": .12, "BTC-USD": .06},
    "Near_Current_Growth": {"VT": .30, "IEF": .10, "TLT": .12, "TIP": .11, "GLD": .11, "XLRE": .07, "DBC": .08, "SHY": .06, "BTC-USD": .05},
    "Near_Current_Defense": {"VT": .26, "IEF": .12, "TLT": .14, "TIP": .12, "GLD": .12, "XLRE": .06, "DBC": .07, "SHY": .07, "BTC-USD": .04},
}

DESCRIPTIONS = {
    CURRENT: "現行L.U.M.U.S.-8比率（既存の90%表記を正規化した比較基準）",
    "Growth_Heavy": "成長厚め版", "Defense_Heavy": "防衛厚め版",
    "Gold_RealAssets_Heavy": "金・実物資産厚め版", "TLT_Light": "TLT薄め版",
    "BNDX_Proxy_Heavy": "BNDX厚め版（長期履歴確保のためIEFで代理）",
    "BTC_Low": "BTC低比率版", "BTC_High": "BTC高比率版",
    "DBC_XLRE_Reduced": "DBC / XLRE削減版", "Simple": "シンプル版",
    "Near_Current_Growth": "現行近傍・成長+2ポイント", "Near_Current_Defense": "現行近傍・防衛寄り",
}


@dataclass
class AllocationResult:
    equity: pd.Series
    gross_equity: pd.Series
    events: list[dict[str, object]]
    turnover: float
    tax_cost: float
    slippage: float
    fees: float
    trade_count: int


def normalized_weights(weights: Mapping[str, float]) -> pd.Series:
    series = pd.Series(weights, index=ASSETS, dtype=float).fillna(0)
    if (series < 0).any() or not np.isclose(series.sum(), 1.0):
        raise ValueError("Every allocation must be non-negative and sum to 1")
    return series


def _trade(values: pd.Series, basis: pd.Series, target: pd.Series, tax_rate: float,
           slippage_bps: float, fee_bps: float) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    total = float(values.sum())
    desired = target * total
    sells = (values - desired).clip(lower=0)
    buys = (desired - values).clip(lower=0)
    sold_basis = (basis * (sells / values.replace(0, np.nan))).fillna(0)
    taxable_gain = float((sells - sold_basis).clip(lower=0).sum())
    traded = float(sells.sum() + buys.sum())
    tax = taxable_gain * tax_rate
    slip = traded * slippage_bps / 10000
    fees = traded * fee_bps / 10000
    drag = tax + slip + fees
    scale = max(0.0, 1 - drag / total) if total else 0.0
    # This deliberately mirrors GENKI's simple average-cost model: losses generate no refund,
    # and friction is funded pro-rata from post-trade holdings.
    new_values = desired * scale
    new_basis = ((basis - sold_basis) + buys) * scale
    stats = {"traded_notional": traded, "turnover": traded / total if total else 0.0,
             "sold_notional": float(sells.sum()), "bought_notional": float(buys.sum()),
             "taxable_gain": taxable_gain, "tax_cost": tax, "slippage": slip, "fees": fees,
             "trade_count": int((sells > 1e-12).sum() + (buys > 1e-12).sum())}
    return new_values, new_basis, stats


def simulate(prices: pd.DataFrame, name: str, weights: Mapping[str, float], tax_rate: float = .20315,
             slippage_bps: float = 5, fee_bps: float = 0) -> AllocationResult:
    """Run threshold plus year-end rebalancing with a GENKI-style tax ledger."""
    target = normalized_weights(weights)
    prices = prices.loc[:, ASSETS].dropna()
    if len(prices) < 2:
        raise ValueError("At least two common price observations are required")
    values = target.copy(); basis = target.copy(); gross_values = target.copy(); gross_basis = target.copy()
    equity: list[tuple[pd.Timestamp, float]] = []; gross: list[tuple[pd.Timestamp, float]] = []
    events: list[dict[str, object]] = []; turnover = tax = slip = fees = 0.0; trades = 0
    thresholds = pd.Series({asset: .05 if asset in CORE_ASSETS else .10 for asset in ASSETS})
    for i, date in enumerate(prices.index):
        if i:
            relative = prices.iloc[i] / prices.iloc[i - 1]
            values *= relative; gross_values *= relative
        year_end = i < len(prices) - 1 and prices.index[i + 1].year != date.year
        current = values / values.sum()
        breached = (current - target).abs() > thresholds
        if year_end or bool(breached.any()):
            reason = "year_end" if year_end else "threshold"
            before = float(values.sum())
            values, basis, stats = _trade(values, basis, target, tax_rate, slippage_bps, fee_bps)
            gross_values, gross_basis, _ = _trade(gross_values, gross_basis, target, 0, 0, 0)
            if stats["trade_count"]:
                turnover += stats["turnover"]; tax += stats["tax_cost"]; slip += stats["slippage"]; fees += stats["fees"]; trades += int(stats["trade_count"])
                events.append({"date": date, "allocation": name, "reason": reason,
                               "breached_assets": "|".join(breached[breached].index), "portfolio_value_before": before,
                               **stats})
        equity.append((date, float(values.sum()))); gross.append((date, float(gross_values.sum())))
    net = pd.Series(dict(equity), name=name); pre = pd.Series(dict(gross), name=name)
    return AllocationResult(net / net.iloc[0], pre / pre.iloc[0], events, turnover, tax, slip, fees, trades)


def _cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan


def _ratio_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().dropna(); dd = equity / equity.cummax() - 1
    cagr = _cagr(equity); downside = returns[returns < 0].std(); maxdd = float(dd.min())
    return {"CAGR": cagr, "MaxDD": maxdd,
            "Sharpe": float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)) if returns.std() else np.nan,
            "Sortino": float(returns.mean() / downside * np.sqrt(TRADING_DAYS)) if downside else np.nan,
            "Calmar": cagr / abs(maxdd) if maxdd else np.nan}


def _annual_returns(equity: pd.Series) -> pd.Series:
    return equity.groupby(equity.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)


def _rolling_win_rate(candidate: pd.Series, baseline: pd.Series, years: int) -> float:
    periods = years * TRADING_DAYS
    candidate_return = candidate / candidate.shift(periods) - 1
    baseline_return = baseline / baseline.shift(periods) - 1
    excess = (candidate_return - baseline_return).dropna()
    return float((excess > 0).mean()) if not excess.empty else np.nan


def build_tables(results: Mapping[str, AllocationResult]) -> dict[str, pd.DataFrame]:
    equities = pd.DataFrame({name: result.equity for name, result in results.items()})
    gross = pd.DataFrame({name: result.gross_equity for name, result in results.items()})
    annual = pd.DataFrame({name: _annual_returns(result.equity) for name, result in results.items()})
    rows = {}
    baseline_annual = annual[CURRENT]
    years = (equities.index[-1] - equities.index[0]).days / 365.25
    for name, result in results.items():
        net = _ratio_metrics(result.equity); pre = _ratio_metrics(result.gross_equity)
        excess = annual[name] - baseline_annual
        row = {"Description": DESCRIPTIONS[name], "PreTaxCAGR": pre["CAGR"], "AfterTaxCAGR": net["CAGR"],
               "MaxDD": net["MaxDD"], "Sharpe": net["Sharpe"], "Sortino": net["Sortino"],
               "Calmar": pre["Calmar"], "AfterTaxCalmar": net["Calmar"],
               "AnnualWins": int((excess > 0).sum()), "AnnualLosses": int((excess < 0).sum()),
               "AnnualTies": int(np.isclose(excess, 0).sum()), "Rolling3YWinRate": _rolling_win_rate(result.equity, results[CURRENT].equity, 3),
               "Rolling5YWinRate": _rolling_win_rate(result.equity, results[CURRENT].equity, 5),
               "Turnover": result.turnover, "TradeCount": result.trade_count, "RebalanceCount": len(result.events),
               "AnnualRebalanceCount": len(result.events) / years, "TaxableEventCount": sum(e["tax_cost"] > 0 for e in result.events),
               "TaxCost": result.tax_cost, "Slippage": result.slippage, "Fees": result.fees,
               "CostDragCAGR": pre["CAGR"] - net["CAGR"], "WorstYearDeterioration": float(excess.min()),
               "Return2022": float(annual.loc[2022, name]) if 2022 in annual.index else np.nan,
               "Excess2022VsCurrent": float(excess.loc[2022]) if 2022 in excess.index else np.nan}
        rows[name] = row
    metrics = pd.DataFrame(rows).T
    metrics["AdoptionDecision"] = adoption_decisions(metrics)
    events = pd.DataFrame([event for result in results.values() for event in result.events]).reindex(columns=[
        "date", "allocation", "reason", "breached_assets", "portfolio_value_before", "traded_notional", "turnover",
        "sold_notional", "bought_notional", "taxable_gain", "tax_cost", "slippage", "fees", "trade_count"])
    weights = pd.DataFrame({name: normalized_weights(allocation) for name, allocation in ALLOCATIONS.items()}).T
    return {"metrics": metrics, "allocation_weights": weights, "annual_returns": annual, "rebalance_events": events,
            "equity_curves": equities, "drawdown_curves": equities / equities.cummax() - 1,
            "gross_equity_curves": gross}


def adoption_decisions(metrics: pd.DataFrame) -> pd.Series:
    base = metrics.loc[CURRENT]; decisions = {}
    for name, row in metrics.iterrows():
        if name == CURRENT:
            decisions[name] = "現行比率維持"; continue
        distance = sum(abs(normalized_weights(ALLOCATIONS[name]) - normalized_weights(ALLOCATIONS[CURRENT]))) / 2
        clear = (row.AfterTaxCAGR >= base.AfterTaxCAGR + .0025 and row.AfterTaxCalmar >= base.AfterTaxCalmar * 1.05
                 and row.MaxDD >= base.MaxDD - .01 and row.CostDragCAGR <= base.CostDragCAGR + .001)
        modest = row.AfterTaxCAGR > base.AfterTaxCAGR and row.AfterTaxCalmar > base.AfterTaxCalmar and row.WorstYearDeterioration > -.02
        if clear:
            decisions[name] = "軽微修正候補" if distance <= .05 else "新規資金で漸進修正候補"
        elif modest:
            decisions[name] = "軽微修正候補" if distance <= .05 else "追加検証候補"
        else:
            decisions[name] = "不採用"
    return pd.Series(decisions)


def _md(table: pd.DataFrame) -> str:
    frame = table.copy().reset_index(); cols = [str(c) for c in frame.columns]
    rows = [[f"{x:.6f}" if isinstance(x, (float, np.floating)) else str(x) for x in row] for row in frame.itertuples(index=False, name=None)]
    return "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_report(tables: Mapping[str, pd.DataFrame], output_dir: Path, start: pd.Timestamp, end: pd.Timestamp,
                 tax_rate: float, slippage_bps: float, fee_bps: float) -> None:
    metrics = tables["metrics"]; annual = tables["annual_returns"]
    eligible = metrics.drop(index=CURRENT); improving = eligible[eligible.AdoptionDecision != "不採用"]
    if improving.empty:
        conclusion = "現行比率維持"
    elif (improving.AdoptionDecision == "軽微修正候補").any():
        conclusion = "軽微修正候補"
    elif (improving.AdoptionDecision == "新規資金で漸進修正候補").any():
        conclusion = "新規資金で漸進修正候補"
    else:
        conclusion = "構造変更候補ではなく追加検証候補"
    main = ["Description", "PreTaxCAGR", "AfterTaxCAGR", "MaxDD", "Sharpe", "Sortino", "Calmar", "AfterTaxCalmar", "CostDragCAGR"]
    costs = ["Turnover", "TradeCount", "RebalanceCount", "AnnualRebalanceCount", "TaxableEventCount", "TaxCost", "Slippage", "Fees"]
    robustness = ["AfterTaxCAGR", "AfterTaxCalmar", "Rolling3YWinRate", "Rolling5YWinRate", "WorstYearDeterioration", "AdoptionDecision"]
    year_wins = metrics[["AnnualWins", "AnnualLosses", "AnnualTies", "WorstYearDeterioration"]]
    stress = metrics[["MaxDD", "Calmar", "AfterTaxCalmar", "Return2022", "Excess2022VsCurrent"]]
    lines = ["# L.U.M.U.S.-8 Allocation Robustness Audit", "", f"検証期間: {start.date()} ～ {end.date()}", "",
             "## 【結論】", f"判定は **{conclusion}**。本監査は少数の仮説主導候補だけを比較する予備監査であり、バックテスト上の最良比率をそのまま採用しない。", "",
             "## 【比較サマリー】", "候補比率（全候補は事前固定で、合計100%）:", _md(tables["allocation_weights"]), "", _md(metrics[main]), "",
             "## 【税後・コスト後評価】", f"GENKI監査と同系統の簡易平均取得単価モデルを使用（税率 {tax_rate:.5f}、slippage {slippage_bps:g}bps、fee {fee_bps:g}bps）。損失売却の税還付はなく、税後CAGR・税後Calmar・Cost Dragを優先する。", _md(metrics[["PreTaxCAGR", "AfterTaxCAGR", "AfterTaxCalmar", "CostDragCAGR", "TaxCost", "Slippage", "Fees"]]), "",
             "## 【リバランス負荷】", "コア資産は目標比率から±5ポイント、サポート資産は±10ポイントの乖離、加えて年末定期リバランスを適用した。TradeCountは売買した資産レッグ数、Turnoverは各イベントの両建て売買額÷直前資産額の累計。", _md(metrics[costs]), "",
             "## 【ドローダウン耐性】", "2022年は株債同時安の代表局面として税後年次リターンを比較した。", _md(stress), "",
             "## 【年次勝敗】", "税後・コスト後の年次リターンで現行比率に勝った年・負けた年を集計。", _md(year_wins), "", _md(annual), "",
             "## 【頑健性評価】", "現行近傍候補が類似した税後CAGR・税後Calmarを示す場合は『頑健な高原』の示唆とみなす。一方、明確な優位には税後CAGR +0.25ポイント以上、税後Calmar +5%以上、最大DD悪化1ポイント以内、Cost Drag悪化0.10ポイント以内を要求した。", _md(metrics[robustness]), "",
             "## 【採用判断】", _md(metrics[["Description", "AdoptionDecision"]]), "",
             "## 【重要な制約】", "- 過去データに対する予備監査であり、将来の成果を保証しない。", "- 過剰探索を避けるため、候補比率は事前固定し、データから最適化していない。", "- 長期共通履歴を維持するため、既存GENKI監査と同様にBNDXをIEF、GLDMをGLD、BTCをBTC-USDで代理する。BNDX厚め版もIEF代理袖を厚くしたものである。", "- 配当・分割込み調整価格を使うが、税モデルは分配金課税、損益通算、ロット別取得価額、口座区分を再現しない。"]
    (output_dir / "allocation_summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def _plots(equities: pd.DataFrame, drawdowns: pd.DataFrame, output_dir: Path) -> None:
    ax = equities.plot(figsize=(12, 7), title="L.U.M.U.S.-8 allocation audit: after-tax equity curves", alpha=.82)
    ax.set_ylabel("Growth of 1"); ax.figure.tight_layout(); ax.figure.savefig(output_dir / "equity_curve.png", dpi=150); plt.close(ax.figure)
    ax = drawdowns.plot(figsize=(12, 7), title="L.U.M.U.S.-8 allocation audit: after-tax drawdowns", alpha=.82)
    ax.set_ylabel("Drawdown"); ax.figure.tight_layout(); ax.figure.savefig(output_dir / "drawdown_curve.png", dpi=150); plt.close(ax.figure)


def run_audit(raw_prices: pd.DataFrame, output_dir: Path, tax_rate: float = .20315,
              slippage_bps: float = 5, fee_bps: float = 0) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices, _ = prepare_prices(raw_prices)
    prices = prices.loc[:, ASSETS].dropna()
    results = {name: simulate(prices, name, weights, tax_rate, slippage_bps, fee_bps) for name, weights in ALLOCATIONS.items()}
    tables = build_tables(results)
    filenames = {"metrics": "allocation_metrics.csv", "annual_returns": "allocation_annual_returns.csv",
                 "rebalance_events": "allocation_rebalance_events.csv", "equity_curves": "allocation_equity_curves.csv",
                 "drawdown_curves": "allocation_drawdown_curves.csv"}
    for key, filename in filenames.items():
        if key == "rebalance_events":
            tables[key].to_csv(output_dir / filename, index=False)
        else:
            tables[key].to_csv(output_dir / filename, index_label="allocation" if key == "metrics" else "date")
    _plots(tables["equity_curves"], tables["drawdown_curves"], output_dir)
    write_report(tables, output_dir, prices.index[0], prices.index[-1], tax_rate, slippage_bps, fee_bps)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None, help="exclusive end date accepted by yfinance")
    parser.add_argument("--tax-rate", type=float, default=.20315)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--fee-bps", type=float, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/allocation_audit"))
    args = parser.parse_args()
    raw = download_prices(ASSETS, args.start, args.end, args.output_dir / "price_cache")
    tables = run_audit(raw, args.output_dir, args.tax_rate, args.slippage_bps, args.fee_bps)
    print(tables["metrics"].to_string())
    print(f"Artifacts saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
