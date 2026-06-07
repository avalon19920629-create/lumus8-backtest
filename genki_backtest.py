"""Historical proxy audit for L.U.M.U.S.-8 Portfolio Conditioning (GENKI)."""
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
BASE_ALLOCATION = {
    "VT": .25, "IEF": .10, "TLT": .12, "TIP": .10, "GLD": .10,
    "XLRE": .06, "DBC": .07, "SHY": .05, "BTC-USD": .05,
}
# Historical-data aliases preserve the current repository's investable sleeves.
ROLE_GROUPS = {
    "成長・攻撃": ["VT", "BTC-USD"],
    "景気後退防衛": ["TLT", "IEF"],
    "インフレ防衛": ["TIP", "DBC", "GLD"],
    "実物資産・利回り": ["XLRE", "GLD", "DBC"],
    "危機時退避": ["GLD", "IEF", "TLT", "BTC-USD"],
}
VARIANT_BUDGETS = {
    "Base Strategy": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
    "GENKI Conservative": {0: 0, 1: 0, 2: 0, 3: .025, 4: .05},
    "GENKI Standard": {0: 0, 1: 0, 2: .025, 3: .05, 4: .10},
}
CORE = {"VT", "TLT", "TIP", "GLD", "BTC-USD"}


@dataclass
class AuditResult:
    equity: pd.Series
    gross_equity: pd.Series
    turnover: float
    tax_cost: float
    slippage: float
    fees: float
    events: list[dict[str, object]]


def normalized_base() -> pd.Series:
    base = pd.Series(BASE_ALLOCATION, dtype=float)
    return base / base.sum()


def historical_proxy(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build monthly role-regime scores and action levels from trailing-only data."""
    daily = prices.pct_change(fill_method=None)
    monthly_dates = prices.groupby(prices.index.to_period("M")).tail(1).index
    score_rows, level_rows = [], []
    for date in monthly_dates:
        p = prices.loc[:date]
        r = daily.loc[:date]
        if len(p) < 252:
            scores = pd.Series(0.0, index=ROLE_GROUPS)
            level = 0
        else:
            mom = p.iloc[-1] / p.iloc[-252] - 1
            ma_gap = p.iloc[-1] / p.iloc[-200:].mean() - 1
            dd = p.iloc[-1] / p.iloc[-252:].max() - 1
            vol = r.iloc[-63:].std() * np.sqrt(TRADING_DAYS)
            # Health is used only within a role after the regime-important roles are selected.
            health = (.45 * mom.rank(pct=True) + .25 * ma_gap.rank(pct=True)
                      + .20 * dd.rank(pct=True) + .10 * (1 - vol.rank(pct=True)))
            growth = float(health[["VT", "BTC-USD"]].mean())
            inflation = float(health[["TIP", "DBC", "GLD"]].mean())
            recession = float(health[["TLT", "IEF"]].mean())
            real = float(health[["XLRE", "GLD", "DBC"]].mean())
            stress = float((-dd[["VT", "BTC-USD"]]).mean() + vol[["VT", "BTC-USD"]].mean())
            scores = pd.Series({
                "成長・攻撃": growth, "景気後退防衛": recession + .35 * stress,
                "インフレ防衛": inflation, "実物資産・利回り": real,
                "危機時退避": float(health[["GLD", "IEF", "TLT"]].mean()) + .45 * stress,
            })
            spread = float(scores.max() - scores.min())
            level = 4 if spread >= .55 else 3 if spread >= .40 else 2 if spread >= .25 else 1
        score_rows.append(scores.rename(date))
        level_rows.append((date, level))
    return pd.DataFrame(score_rows), pd.DataFrame(level_rows, columns=["date", "action_level"]).set_index("date")


def make_tilt_target(base: pd.Series, prices_to_date: pd.DataFrame, group_scores: pd.Series,
                     action_level: int, budget: float) -> tuple[pd.Series, dict[str, object]]:
    """Shift a conserved budget between regime-selected roles, within ±10pp floors/ceilings."""
    if action_level <= 1 or budget <= 0 or len(prices_to_date) < 252:
        return base.copy(), {"budget": 0.0, "increase_groups": "", "decrease_groups": "", "increase_assets": "", "decrease_assets": ""}
    inc_group, dec_group = str(group_scores.idxmax()), str(group_scores.idxmin())
    trailing = prices_to_date.pct_change(fill_method=None).iloc[-252:]
    health = (prices_to_date.iloc[-1] / prices_to_date.iloc[-252] - 1).rank(pct=True)
    health += (1 - trailing.std().rank(pct=True)) * .25
    inc_assets = [a for a in ROLE_GROUPS[inc_group] if a not in ROLE_GROUPS[dec_group]] or ROLE_GROUPS[inc_group]
    dec_assets = [a for a in ROLE_GROUPS[dec_group] if a not in ROLE_GROUPS[inc_group]] or ROLE_GROUPS[dec_group]
    inc_assets = sorted(inc_assets, key=lambda x: health.get(x, 0), reverse=True)
    dec_assets = sorted(dec_assets, key=lambda x: health.get(x, 0))
    target = base.copy()
    remaining = min(budget, .10)
    for asset in dec_assets:
        amount = min(remaining, max(0.0, target[asset] - max(.01, base[asset] - .10)))
        target[asset] -= amount
        remaining -= amount
        if remaining <= 1e-12:
            break
    moved = min(budget, .10) - remaining
    remaining = moved
    for asset in inc_assets:
        amount = min(remaining, max(0.0, min(base[asset] + .10, .20 if asset == "BTC-USD" else 1.0) - target[asset]))
        target[asset] += amount
        remaining -= amount
        if remaining <= 1e-12:
            break
    actual = moved - remaining
    if remaining > 1e-12:  # return any unallocated increase to the reduced sleeve
        target[dec_assets[0]] += remaining
    return target, {"budget": actual, "increase_groups": inc_group, "decrease_groups": dec_group,
                    "increase_assets": "|".join(inc_assets), "decrease_assets": "|".join(dec_assets)}


def _trade_to_target(values: pd.Series, costs: pd.Series, target: pd.Series, total: float,
                     tax_rate: float, slippage_bps: float, fee_bps: float) -> tuple[pd.Series, pd.Series, float, float, float, float]:
    desired = target * total
    sells = (values - desired).clip(lower=0)
    buys = (desired - values).clip(lower=0)
    sold = float(sells.sum())
    gains = ((values - costs).clip(lower=0) * (sells / values.replace(0, np.nan))).fillna(0).sum()
    tax = float(gains * tax_rate)
    slip = float((sold + buys.sum()) * slippage_bps / 10000)
    fees = float((sold + buys.sum()) * fee_bps / 10000)
    drag = tax + slip + fees
    # Fund drag proportionally from desired holdings; average-cost basis rises only with buys.
    values = (desired * max(0, 1 - drag / total)).clip(lower=0)
    costs = (costs - (costs * (sells / (values + sells).replace(0, np.nan))).fillna(0) + buys).clip(lower=0)
    return values, costs, sold + float(buys.sum()), tax, slip, fees


def simulate(prices: pd.DataFrame, variant: str, group_scores: pd.DataFrame, levels: pd.DataFrame,
             tax_rate: float, slippage_bps: float, fee_bps: float) -> AuditResult:
    base = normalized_base(); dates = prices.dropna().index; prices = prices.loc[dates, base.index]
    values = base.copy(); costs = base.copy(); gross_values = base.copy(); gross_costs = base.copy()
    eq, gross_eq, events = [], [], []; turnover = tax = slip = fees = 0.0; last_target = base.copy()
    for i, date in enumerate(dates):
        if i:
            rel = prices.iloc[i] / prices.iloc[i - 1]
            values *= rel; gross_values *= rel
        year_end = i == len(dates)-1 or dates[i + 1].year != date.year
        month_end = i == len(dates)-1 or dates[i + 1].month != date.month
        target = last_target
        level = 0; info = {"budget": 0.0, "increase_groups": "", "decrease_groups": "", "increase_assets": "", "decrease_assets": ""}
        if year_end:
            target = base.copy()  # annual homeostatic restoration
        elif month_end and date in levels.index:
            level = int(levels.loc[date, "action_level"])
            budget = VARIANT_BUDGETS[variant][level]
            target, info = make_tilt_target(base, prices.loc[:date], group_scores.loc[date], level, budget)
        current = values / values.sum()
        thresholds = pd.Series({a: .05 if a in CORE else .10 for a in base.index})
        should_trade = year_end or bool(((current - target).abs() > thresholds).any())
        # A changed active tilt target is itself an approved conditioning rebalance.
        should_trade |= month_end and info["budget"] > 0 and not np.allclose(target, last_target)
        if should_trade:
            before = float(values.sum())
            values, costs, turn, tx, sl, fe = _trade_to_target(values, costs, target, before, tax_rate, slippage_bps, fee_bps)
            gross_values, gross_costs, *_ = _trade_to_target(gross_values, gross_costs, target, float(gross_values.sum()), 0, 0, 0)
            turnover += turn; tax += tx; slip += sl; fees += fe
            if info["budget"] > 0:
                events.append({"date": date, "variant": variant, "action_level": level, "tilt_budget": info["budget"],
                               **{k: info[k] for k in ["increase_groups", "decrease_groups", "increase_assets", "decrease_assets"]},
                               "estimated_tax_cost": tx, "estimated_slippage": sl})
            last_target = target
        eq.append((date, values.sum())); gross_eq.append((date, gross_values.sum()))
    equity = pd.Series(dict(eq), name=variant); gross = pd.Series(dict(gross_eq), name=variant)
    return AuditResult(equity / equity.iloc[0], gross / gross.iloc[0], turnover, tax, slip, fees, events)


def metric_row(result: AuditResult) -> dict[str, float]:
    r = result.equity.pct_change().dropna(); years = len(r) / TRADING_DAYS
    cagr = result.equity.iloc[-1] ** (1 / years) - 1 if years else np.nan
    dd = result.equity / result.equity.cummax() - 1; downside = r[r < 0].std()
    gross_cagr = result.gross_equity.iloc[-1] ** (1 / years) - 1 if years else np.nan
    return {"CAGR": cagr, "GrossCAGR": gross_cagr, "CostDragCAGR": gross_cagr-cagr, "MaxDD": dd.min(),
            "Sharpe": r.mean()/r.std()*np.sqrt(TRADING_DAYS), "Sortino": r.mean()/downside*np.sqrt(TRADING_DAYS),
            "Calmar": cagr/abs(dd.min()), "Turnover": result.turnover, "TaxCost": result.tax_cost,
            "Slippage": result.slippage, "Fees": result.fees, "TiltEvents": len(result.events),
            "AnnualTiltEvents": len(result.events)/years if years else 0}


def _forward_event_returns(events: pd.DataFrame, equities: pd.DataFrame) -> pd.DataFrame:
    for months in [1, 3, 6, 12]:
        vals=[]
        for _, e in events.iterrows():
            idx = equities.index.get_indexer([pd.Timestamp(e.date)], method="nearest")[0]
            end = min(len(equities)-1, idx + round(months * TRADING_DAYS / 12))
            vals.append(equities[e.variant].iloc[end]/equities[e.variant].iloc[idx] - equities["Base Strategy"].iloc[end]/equities["Base Strategy"].iloc[idx])
        events[f"forward_{months}m_excess"] = vals
    return events


def run_audit(raw_prices: pd.DataFrame, output_dir: Path, tax_rate: float=.20315, slippage_bps: float=5, fee_bps: float=0) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices, _ = prepare_prices(raw_prices)
    prices = prices.loc[:, normalized_base().index].dropna()
    scores, levels = historical_proxy(prices)
    results = {v: simulate(prices, v, scores, levels, tax_rate, slippage_bps, fee_bps) for v in VARIANT_BUDGETS}
    equities = pd.DataFrame({v: x.equity for v,x in results.items()}); equities.to_csv(output_dir/"equity_curves.csv", index_label="date")
    metrics = pd.DataFrame({v: metric_row(x) for v,x in results.items()}).T
    annual = equities.groupby(equities.index.year).apply(lambda x: x.iloc[-1]/x.iloc[0]-1); annual.index.name="year"
    base_ann = annual["Base Strategy"]
    for v in metrics.index:
        excess = annual[v]-base_ann
        metrics.loc[v,"AnnualWins"] = int((excess>0).sum()); metrics.loc[v,"AfterTaxExcessWinRate"] = float((excess>0).mean())
        metrics.loc[v,"AfterTaxExcessMedian"] = float(excess.median()); metrics.loc[v,"WorstYearDeterioration"] = float(excess.min())
        for yrs in [3,5]:
            roll = equities[v].pct_change().add(1).rolling(yrs*TRADING_DAYS).apply(np.prod, raw=True) - equities["Base Strategy"].pct_change().add(1).rolling(yrs*TRADING_DAYS).apply(np.prod, raw=True)
            metrics.loc[v,f"Rolling{yrs}YWinRate"] = float((roll.dropna()>0).mean()) if not roll.dropna().empty else np.nan
    metrics.to_csv(output_dir/"genki_metrics.csv", index_label="variant"); annual.to_csv(output_dir/"genki_annual_returns.csv")
    events = pd.DataFrame([e for x in results.values() for e in x.events])
    required = ["date","variant","action_level","tilt_budget","increase_groups","decrease_groups","increase_assets","decrease_assets","estimated_tax_cost","estimated_slippage"]
    events = events.reindex(columns=required)
    if not events.empty: events = _forward_event_returns(events, equities)
    else:
        for m in [1,3,6,12]: events[f"forward_{m}m_excess"]=[]
    events.to_csv(output_dir/"tilt_events.csv", index=False)
    _plots(equities, output_dir); _report(metrics, annual, events, prices.index.min(), prices.index.max(), output_dir, tax_rate, slippage_bps, fee_bps)
    return {"metrics":metrics,"annual_returns":annual,"events":events,"equity_curves":equities}


def _plots(equities: pd.DataFrame, out: Path) -> None:
    ax=equities.plot(figsize=(11,6), title="L.U.M.U.S.-8 GENKI audit equity curves"); ax.set_ylabel("Growth of 1"); ax.figure.tight_layout(); ax.figure.savefig(out/"equity_curve.png", dpi=150); plt.close(ax.figure)
    dd=equities/equities.cummax()-1; ax=dd.plot(figsize=(11,6), title="Drawdown curves"); ax.set_ylabel("Drawdown"); ax.figure.tight_layout(); ax.figure.savefig(out/"drawdown_curve.png", dpi=150); plt.close(ax.figure)


def _md(table: pd.DataFrame) -> str:
    """Render a compact markdown table without an optional tabulate dependency."""
    frame = table.copy().reset_index()
    cols = [str(c) for c in frame.columns]
    rows = [[f"{x:.6f}" if isinstance(x, (float, np.floating)) else str(x) for x in row] for row in frame.itertuples(index=False, name=None)]
    return "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _report(m: pd.DataFrame, annual: pd.DataFrame, events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, out: Path, tax: float, slip: float, fee: float) -> None:
    candidates=m.drop(index="Base Strategy"); best=candidates["CAGR"].idxmax(); improved=(candidates.CAGR>m.loc["Base Strategy","CAGR"]) & (candidates.Calmar>m.loc["Base Strategy","Calmar"])
    verdict="正式採用候補" if improved.all() else "Shadow Mode継続" if improved.any() else "不採用候補"
    lines=["# L.U.M.U.S.-8 元気玉システム Historical Backtest 予備監査","",f"検証期間: {start.date()} ～ {end.date()}","",f"## 【結論】\n採用判定は **{verdict}**。最良税後CAGR variantは **{best}**。本結果のみで本番売買には接続しない。","","## 【比較サマリー】",_md(m[["CAGR","MaxDD","Sharpe","Sortino","Calmar","Turnover","TiltEvents"]]),"",f"## 【税後パフォーマンス】\n簡易平均取得単価税モデル（税率 {tax:.5f}、slippage {slip:g}bps、fee {fee:g}bps）。GrossCAGRとの差をコストドラッグとして表示。",_md(m[["GrossCAGR","CAGR","CostDragCAGR","TaxCost","Slippage","Fees"]]),"","## 【ドローダウン】\n最大DDとCalmarは上表を参照。","","## 【売買コスト】\nTilt発動回数、turnover、課税ドラッグは上表を参照。通常版との差分が追加売買コストの目安。","","## 【年次勝敗】",_md(annual),"","## 【ローリング勝率】",_md(m[["Rolling3YWinRate","Rolling5YWinRate","AfterTaxExcessWinRate","AfterTaxExcessMedian","WorstYearDeterioration"]]),"","## 【Tiltイベント分析】",(_md(events.groupby("variant")[["forward_1m_excess","forward_3m_excess","forward_6m_excess","forward_12m_excess"]].median()) if not events.empty else "Tiltイベントなし。"),"",f"## 【採用判定】\n- **{verdict}**","","## 【注意】\nこの検証はHistorical El Shaddai Proxyによる予備監査であり、現在のEl Shaddai v2.0完全再現ではない。IEF/GLD/BTC-USDは、依頼記載のBNDX/GLDM/BTCに対する既存レポジトリ上の履歴代替である。New-money-onlyは定期入金・現金フローが既存モデルにないため未実装。"]
    (out/"genki_summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--start",default="2010-01-01"); p.add_argument("--end",default=None); p.add_argument("--tax-rate",type=float,default=.20315); p.add_argument("--slippage-bps",type=float,default=5); p.add_argument("--fee-bps",type=float,default=0); p.add_argument("--output-dir",type=Path,default=Path("artifacts/genki_audit")); a=p.parse_args()
    raw=download_prices(list(BASE_ALLOCATION),a.start,a.end,a.output_dir/"price_cache"); run_audit(raw,a.output_dir,a.tax_rate,a.slippage_bps,a.fee_bps)

if __name__ == "__main__": main()
