"""Hypothesis-led decomposition audit for the Growth_Heavy allocation.

The candidate allocations in this module are deliberately fixed before prices are
loaded.  This is a diagnostic counterfactual audit, not an optimizer and not a
recommendation to change a live taxable portfolio.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from allocation_robustness_audit import (
    ASSETS,
    CURRENT,
    ALLOCATIONS as ROBUSTNESS_ALLOCATIONS,
    _annual_returns,
    _md,
    _ratio_metrics,
    _rolling_win_rate,
    normalized_weights,
    simulate,
)
from lumus8_backtest import download_prices, prepare_prices

GROWTH = "Growth_Heavy"
# Audit invariant: these weights are hypothesis-led constants; no market data is used
# to create or alter them.  BTC is capped at Growth_Heavy's moderate 7%, well below the
# previous audit's 11% BTC_High stress candidate.
ALLOCATIONS: dict[str, dict[str, float]] = {
    CURRENT: dict(ROBUSTNESS_ALLOCATIONS[CURRENT]),
    GROWTH: dict(ROBUSTNESS_ALLOCATIONS[GROWTH]),
    "VT_Only_Heavy": {"VT": .340, "IEF": .095, "TLT": .105, "TIP": .110, "GLD": .110, "XLRE": .065, "DBC": .075, "SHY": .045, "BTC-USD": .055},
    "BTC_Only_Heavy": {"VT": .278, "IEF": .109, "TLT": .130, "TIP": .109, "GLD": .109, "XLRE": .066, "DBC": .075, "SHY": .059, "BTC-USD": .065},
    "VT_Heavy_BTC_Fixed": {"VT": .340, "IEF": .092, "TLT": .102, "TIP": .102, "GLD": .102, "XLRE": .068, "DBC": .068, "SHY": .070, "BTC-USD": .056},
    "VT_Current_BTC_Heavy": {"VT": .278, "IEF": .108, "TLT": .128, "TIP": .108, "GLD": .108, "XLRE": .065, "DBC": .073, "SHY": .062, "BTC-USD": .070},
    "Defense_Light_Only": {"VT": .278, "IEF": .090, "TLT": .100, "TIP": .100, "GLD": .130, "XLRE": .080, "DBC": .100, "SHY": .066, "BTC-USD": .056},
    "TLT_Light_Growth_Current": {"VT": .278, "IEF": .130, "TLT": .080, "TIP": .125, "GLD": .130, "XLRE": .070, "DBC": .080, "SHY": .051, "BTC-USD": .056},
    "SHY_Cash_Light_To_VT": {"VT": .313, "IEF": .111, "TLT": .133, "TIP": .111, "GLD": .111, "XLRE": .067, "DBC": .078, "SHY": .020, "BTC-USD": .056},
    "Mild_Growth_Heavy": {"VT": .310, "IEF": .100, "TLT": .120, "TIP": .105, "GLD": .105, "XLRE": .068, "DBC": .074, "SHY": .058, "BTC-USD": .060},
    "Growth_Heavy_No_BTC_Increase": {"VT": .340, "IEF": .090, "TLT": .100, "TIP": .100, "GLD": .100, "XLRE": .070, "DBC": .070, "SHY": .075, "BTC-USD": .055},
    "Growth_Heavy_2022_Safe": {"VT": .320, "IEF": .090, "TLT": .100, "TIP": .110, "GLD": .110, "XLRE": .070, "DBC": .070, "SHY": .070, "BTC-USD": .060},
}

DESCRIPTIONS = {
    CURRENT: "現行L.U.M.U.S.-8比率（90%表記を正規化）",
    GROWTH: "前回監査と同じ成長厚め版",
    "VT_Only_Heavy": "BTCを現行近傍に保ち、VTだけをGrowth_Heavy水準へ",
    "BTC_Only_Heavy": "VTを現行近傍に保ち、BTCを小幅増",
    "VT_Heavy_BTC_Fixed": "VTをGrowth_Heavy水準、BTCを現行水準に固定",
    "VT_Current_BTC_Heavy": "VTを現行近傍、BTCをGrowth_Heavy水準へ",
    "Defense_Light_Only": "VT/BTCを現行近傍のまま防衛資産を削減し分散先へ",
    "TLT_Light_Growth_Current": "VT/BTCを現行近傍のままTLTだけ削減",
    "SHY_Cash_Light_To_VT": "SHYを削減してVTへ移す",
    "Mild_Growth_Heavy": "現行とGrowth_Heavyの中間的な成長寄せ",
    "Growth_Heavy_No_BTC_Increase": "Growth_Heavy型だがBTCを現行以下に抑制",
    "Growth_Heavy_2022_Safe": "Growth_Heavy型にGLD/TIP/SHYを追加",
}

REQUIRED_METRICS = {
    "PreTaxCAGR", "AfterTaxCAGR", "MaxDD", "Sharpe", "Sortino", "Calmar", "AfterTaxCalmar",
    "CostDragCAGR", "Turnover", "TradeCount", "RebalanceCount", "AnnualRebalanceCount",
    "TaxableEventCount", "TaxCost", "Slippage", "Fees", "AnnualWins", "AnnualLosses",
    "Rolling3YWinRate", "Rolling5YWinRate", "WorstYearDeterioration", "Return2022", "Excess2022VsCurrent",
}


def allocation_distance(name: str) -> float:
    """Return one-way turnover needed to move from Current to a fixed candidate."""
    return float((normalized_weights(ALLOCATIONS[name]) - normalized_weights(ALLOCATIONS[CURRENT])).abs().sum() / 2)


def _period_cagr(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    section = equity.loc[start:end]
    return _ratio_metrics(section)["CAGR"] if len(section) >= 2 else np.nan


def adoption_decisions(metrics: pd.DataFrame) -> pd.Series:
    base = metrics.loc[CURRENT]
    decisions: dict[str, str] = {CURRENT: "現行比率維持"}
    for name, row in metrics.drop(index=CURRENT).iterrows():
        net_better = row.AfterTaxCAGR > base.AfterTaxCAGR and row.AfterTaxCalmar > base.AfterTaxCalmar
        tolerable = row.MaxDD >= base.MaxDD - .03 and row.Excess2022VsCurrent >= -.04
        if name == GROWTH and net_better and tolerable:
            decision = "Growth_Heavy追加検証候補"
        elif name in {"Mild_Growth_Heavy", "VT_Only_Heavy", "SHY_Cash_Light_To_VT"} and net_better and tolerable:
            decision = "軽微成長寄せ候補" if row.AllocationDistance <= .06 else "新規資金で漸進修正候補"
        elif net_better and tolerable:
            decision = "追加検証候補"
        else:
            decision = "不採用"
        decisions[name] = decision
    return pd.Series(decisions)


def build_tables(results: Mapping[str, object]) -> dict[str, pd.DataFrame]:
    equities = pd.DataFrame({name: result.equity for name, result in results.items()})
    gross = pd.DataFrame({name: result.gross_equity for name, result in results.items()})
    annual = pd.DataFrame({name: _annual_returns(result.equity) for name, result in results.items()})
    baseline_annual = annual[CURRENT]
    years = (equities.index[-1] - equities.index[0]).days / 365.25
    midpoint = equities.index[0] + (equities.index[-1] - equities.index[0]) / 2
    rows: dict[str, dict[str, object]] = {}
    for name, result in results.items():
        net = _ratio_metrics(result.equity); pre = _ratio_metrics(result.gross_equity)
        excess = annual[name] - baseline_annual
        early = _period_cagr(result.equity, equities.index[0], midpoint)
        late = _period_cagr(result.equity, midpoint, equities.index[-1])
        rows[name] = {
            "Description": DESCRIPTIONS[name], "PreTaxCAGR": pre["CAGR"], "AfterTaxCAGR": net["CAGR"],
            "MaxDD": net["MaxDD"], "Sharpe": net["Sharpe"], "Sortino": net["Sortino"],
            "Calmar": pre["Calmar"], "AfterTaxCalmar": net["Calmar"], "CostDragCAGR": pre["CAGR"] - net["CAGR"],
            "Turnover": result.turnover, "TradeCount": result.trade_count, "RebalanceCount": len(result.events),
            "AnnualRebalanceCount": len(result.events) / years, "TaxableEventCount": sum(e["tax_cost"] > 0 for e in result.events),
            "TaxCost": result.tax_cost, "Slippage": result.slippage, "Fees": result.fees,
            "AnnualWins": int((excess > 0).sum()), "AnnualLosses": int((excess < 0).sum()),
            "Rolling3YWinRate": _rolling_win_rate(result.equity, results[CURRENT].equity, 3),
            "Rolling5YWinRate": _rolling_win_rate(result.equity, results[CURRENT].equity, 5),
            "WorstYearDeterioration": float(excess.min()),
            "Return2022": float(annual.loc[2022, name]) if 2022 in annual.index else np.nan,
            "Excess2022VsCurrent": float(excess.loc[2022]) if 2022 in excess.index else np.nan,
            "AllocationDistance": allocation_distance(name), "EarlyHalfCAGR": early, "LateHalfCAGR": late,
            "EarlyHalfExcessVsCurrent": early - _period_cagr(results[CURRENT].equity, equities.index[0], midpoint),
            "LateHalfExcessVsCurrent": late - _period_cagr(results[CURRENT].equity, midpoint, equities.index[-1]),
        }
    metrics = pd.DataFrame(rows).T
    metrics["AdoptionDecision"] = adoption_decisions(metrics)
    events = pd.DataFrame([event for result in results.values() for event in result.events]).reindex(columns=[
        "date", "allocation", "reason", "breached_assets", "portfolio_value_before", "traded_notional", "turnover",
        "sold_notional", "bought_notional", "taxable_gain", "tax_cost", "slippage", "fees", "trade_count"])
    return {"metrics": metrics, "annual_returns": annual, "rebalance_events": events, "equity_curves": equities,
            "drawdown_curves": equities / equities.cummax() - 1, "gross_equity_curves": gross,
            "allocation_weights": pd.DataFrame({n: normalized_weights(w) for n, w in ALLOCATIONS.items()}).T}


def _main_conclusion(metrics: pd.DataFrame) -> str:
    base, growth = metrics.loc[CURRENT], metrics.loc[GROWTH]
    vt = metrics.loc["VT_Heavy_BTC_Fixed"]
    btc = metrics.loc["VT_Current_BTC_Heavy"]
    if growth.AfterTaxCAGR <= base.AfterTaxCAGR or growth.AfterTaxCalmar <= base.AfterTaxCalmar:
        return "1. 現行比率維持が最も妥当"
    if growth.EarlyHalfExcessVsCurrent <= 0 or growth.LateHalfExcessVsCurrent <= 0:
        return "5. 成長寄せ全般が過去相場に過剰適合しており、採用しない"
    if vt.AfterTaxCAGR > btc.AfterTaxCAGR and vt.AfterTaxCalmar >= base.AfterTaxCalmar:
        return "3. VTだけを少し厚くする軽微修正に妙味あり"
    if btc.AfterTaxCAGR > vt.AfterTaxCAGR and (btc.MaxDD < base.MaxDD - .02 or btc.Excess2022VsCurrent < -.02):
        return "4. BTC増加が主因だが、リスク悪化が大きく採用困難"
    return "2. Growth_Heavyは有力だが、移行には追加検証が必要"


def write_report(tables: Mapping[str, pd.DataFrame], output_dir: Path, start: pd.Timestamp, end: pd.Timestamp,
                 tax_rate: float, slippage_bps: float, fee_bps: float) -> None:
    m = tables["metrics"]; annual = tables["annual_returns"]; base = m.loc[CURRENT]; growth = m.loc[GROWTH]
    vt = m.loc["VT_Heavy_BTC_Fixed"]; btc = m.loc["VT_Current_BTC_Heavy"]
    defense = m.loc["Defense_Light_Only"]; cash = m.loc["SHY_Cash_Light_To_VT"]
    def delta(row: pd.Series, field: str) -> str: return f"{row[field] - base[field]:+.2%}"
    main_cols = ["Description", "AfterTaxCAGR", "AfterTaxCalmar", "MaxDD", "Return2022", "CostDragCAGR", "Turnover", "RebalanceCount", "AllocationDistance", "AdoptionDecision"]
    cost_cols = ["PreTaxCAGR", "AfterTaxCAGR", "CostDragCAGR", "TaxCost", "Slippage", "Fees"]
    load_cols = ["Turnover", "TradeCount", "RebalanceCount", "AnnualRebalanceCount", "TaxableEventCount", "AllocationDistance"]
    rolling_cols = ["AnnualWins", "AnnualLosses", "Rolling3YWinRate", "Rolling5YWinRate", "WorstYearDeterioration", "EarlyHalfExcessVsCurrent", "LateHalfExcessVsCurrent"]
    support_change = normalized_weights(ALLOCATIONS[GROWTH])[['GLD', 'XLRE', 'DBC']] - normalized_weights(ALLOCATIONS[CURRENT])[['GLD', 'XLRE', 'DBC']]
    lines = [
        "# Growth_Heavy 勝因分解監査", "", "【結論】", f"- **最重要結論: {_main_conclusion(m)}**",
        "- 本監査は事前固定した反実仮想の予備監査であり、本番比率変更を推奨・自動接続しない。結果が良くても追加検証材料に限定する。",
        f"- 共通評価期間は **{start.date()}〜{end.date()}**。指定開始日がこれより早い場合、BTC-USD/XLRE等を含む全資産共通履歴の制約で短縮された。",
        "", "【比較サマリー】", _md(m[main_cols]), "", "【Growth_Heavyの勝因分解】",
        f"- VT増加: BTCを現行水準に固定した `VT_Heavy_BTC_Fixed` の税後CAGR差は現行比 {delta(vt, 'AfterTaxCAGR')}。Growth_Heavyとの差は {vt.AfterTaxCAGR-growth.AfterTaxCAGR:+.2%}。",
        f"- BTC増加: VTを現行水準に固定した `VT_Current_BTC_Heavy` の税後CAGR差は現行比 {delta(btc, 'AfterTaxCAGR')}。Growth_Heavyとの差は {btc.AfterTaxCAGR-growth.AfterTaxCAGR:+.2%}。",
        f"- 防衛資産削減: `Defense_Light_Only` の税後CAGR差は {delta(defense, 'AfterTaxCAGR')}、税後Calmar差は {defense.AfterTaxCalmar-base.AfterTaxCalmar:+.3f}。成長資産を増やさない反実仮想で判定する。",
        f"- 現金/SHY削減: `SHY_Cash_Light_To_VT` の税後CAGR差は {delta(cash, 'AfterTaxCAGR')}、2022年差は {cash.Excess2022VsCurrent:+.2%}。",
        f"- サポート資産: Growth_Heavyの現行比変化は GLD {support_change.GLD:+.2%}, XLRE {support_change.XLRE:+.2%}, DBC {support_change.DBC:+.2%} と限定的。サポート削減単独候補は設けていないため、単独寄与の断定はしない。",
        f"- 2015年以降への適合: Growth_Heavyの前半/後半の現行超過CAGRは {growth.EarlyHalfExcessVsCurrent:+.2%} / {growth.LateHalfExcessVsCurrent:+.2%}。共通履歴以前を使えず、2015年以前への外挿検証はできない。片側だけの優位なら過剰適合懸念を強める。",
        f"- 税後・コスト後: Growth_Heavyの税後CAGR差は {delta(growth, 'AfterTaxCAGR')}、税後Calmar差は {growth.AfterTaxCalmar-base.AfterTaxCalmar:+.3f}、CostDrag差は {growth.CostDragCAGR-base.CostDragCAGR:+.2%}。",
        f"- 2022年耐性: Growth_Heavyは {growth.Return2022:.2%}（現行比 {growth.Excess2022VsCurrent:+.2%}）。悪化幅とMaxDD差 {growth.MaxDD-base.MaxDD:+.2%} をリターン差と交換可能とは本監査だけでは判断しない。",
        "", "【税後・コスト後評価】", _md(m[cost_cols]), "", "【リバランス負荷】", _md(m[load_cols]),
        "", "【2022年耐性】", _md(m[["Return2022", "Excess2022VsCurrent", "MaxDD", "WorstYearDeterioration"]]),
        "", "【年次勝敗】", _md(annual.T), "", "【ローリング勝率】", _md(m[rolling_cols]),
        "", "【採用候補の分類】", _md(m[["Description", "AllocationDistance", "AdoptionDecision"]]),
        "", "【最重要結論】", f"- **{_main_conclusion(m)}**", "- 良好な候補も即時移行ではなく、新規資金による漸進修正や別期間・別税務前提での追加検証に限定する。",
        "", "【重要な制約】",
        "- 候補比率はコード内で事前固定し、価格データから最適化していない。BTC上限は7%で、前回BTC_Highの11%を採用候補に含めない。",
        f"- 税率 {tax_rate:.3%}、スリッページ {slippage_bps:g}bps、手数料 {fee_bps:g}bps。平均取得価額による簡易税モデルで、含み益ロット、損益通算、分配金課税、口座区分、移行時の既存含み益課税は再現しない。",
        "- BNDX/GLDM/BTC袖は既存監査と同様にIEF/GLD/BTC-USDで代理する。共通履歴しか評価しないため、2015年以前の成長相場以外での頑健性は判定不能。",
        "- `Defense_Light_Only` の削減分は現金ではなくGLD/XLRE/DBC/SHYへ分散しており、純粋な現金保有の効果はモデル化しない。",
    ]
    (output_dir / "growth_decomposition_summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def _plots(equities: pd.DataFrame, drawdowns: pd.DataFrame, output_dir: Path) -> None:
    for data, filename, title, ylabel in [(equities, "equity_curve.png", "Growth decomposition: after-tax equity", "Growth of 1"), (drawdowns, "drawdown_curve.png", "Growth decomposition: after-tax drawdown", "Drawdown")]:
        ax = data.plot(figsize=(14, 8), title=title, alpha=.78); ax.set_ylabel(ylabel)
        ax.figure.tight_layout(); ax.figure.savefig(output_dir / filename, dpi=150); plt.close(ax.figure)


def run_audit(raw_prices: pd.DataFrame, output_dir: Path, tax_rate: float = .20315,
              slippage_bps: float = 5, fee_bps: float = 0) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices, _ = prepare_prices(raw_prices); prices = prices.loc[:, ASSETS].dropna()
    results = {name: simulate(prices, name, weights, tax_rate, slippage_bps, fee_bps) for name, weights in ALLOCATIONS.items()}
    tables = build_tables(results)
    filenames = {"metrics": "growth_decomposition_metrics.csv", "annual_returns": "growth_decomposition_annual_returns.csv", "rebalance_events": "growth_decomposition_rebalance_events.csv", "equity_curves": "growth_decomposition_equity_curves.csv", "drawdown_curves": "growth_decomposition_drawdown_curves.csv"}
    for key, filename in filenames.items():
        tables[key].to_csv(output_dir / filename, index=False if key == "rebalance_events" else True, index_label=None if key == "rebalance_events" else ("allocation" if key == "metrics" else "date"))
    _plots(tables["equity_curves"], tables["drawdown_curves"], output_dir)
    write_report(tables, output_dir, prices.index[0], prices.index[-1], tax_rate, slippage_bps, fee_bps)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-10-08"); parser.add_argument("--end", default=None, help="exclusive end date accepted by yfinance")
    parser.add_argument("--tax-rate", type=float, default=.20315); parser.add_argument("--slippage-bps", type=float, default=5); parser.add_argument("--fee-bps", type=float, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/growth_decomposition")); args = parser.parse_args()
    raw = download_prices(ASSETS, args.start, args.end, args.output_dir / "price_cache")
    tables = run_audit(raw, args.output_dir, args.tax_rate, args.slippage_bps, args.fee_bps)
    print(tables["metrics"].to_string()); print(f"Artifacts saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
