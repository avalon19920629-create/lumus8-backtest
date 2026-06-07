"""Audit L.U.M.U.S.-8 rebalance bands and year-end policies without changing weights.

This bounded comparison retains the current allocation and historical proxies.  The
optional carryforward mode adds an intentionally simplified annual-netting model for
three-year tax-loss carryforwards; it is a backtest approximation, not tax advice.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from allocation_robustness_audit import ALLOCATIONS, ASSETS, CORE_ASSETS, CURRENT, normalized_weights
from lumus8_backtest import download_prices, prepare_prices

TRADING_DAYS = 252
CURRENT_POLICY = "Current_5_10_Annual"
DECISIONS = {"現行リバランス維持", "年末1回のみ候補", "年末1回条件付き候補", "半年/四半期定期リバランス候補", "追加検証候補", "不採用"}
SIMPLIFICATION_POLICIES = [
    CURRENT_POLICY, "Annual_Only", "Annual_Only_Loss_Harvest", "Annual_Only_Conditional",
    "SemiAnnual_Only", "Quarterly_Only", "Threshold_Only",
]


@dataclass(frozen=True)
class RebalancePolicy:
    description: str
    core_band: float
    support_band: float
    year_end_rule: str
    sell_suppression: bool = False
    threshold_enabled: bool = True
    periodic_frequency: str = "annual"
    loss_harvest: bool = False


POLICIES: dict[str, RebalancePolicy] = {
    CURRENT_POLICY: RebalancePolicy("現行：コア±5 / サポート±10 / 年末あり", .05, .10, "annual"),
    "Loose_7_5_12_5_Annual": RebalancePolicy("緩め：コア±7.5 / サポート±12.5 / 年末あり", .075, .125, "annual"),
    "Looser_10_15_Annual": RebalancePolicy("さらに緩め：コア±10 / サポート±15 / 年末あり", .10, .15, "annual"),
    "Threshold_Only": RebalancePolicy("年末なし：乖離時のみ", .05, .10, "none"),
    "Biennial_Year_End": RebalancePolicy("年末隔年：偶数年末のみ", .05, .10, "biennial"),
    "Conditional_Year_End": RebalancePolicy("年末条件付き：最大乖離がバンドの50%以上のときのみ", .05, .10, "conditional"),
    "Loss_Harvest_Year_End": RebalancePolicy("年末損出し優先：含み損ポジションがある場合のみ", .05, .10, "loss_only"),
    "Sell_Suppressed": RebalancePolicy("売却抑制版：乖離時にバンド境界までの最小売買", .05, .10, "annual", True),
    "Annual_Only": RebalancePolicy("年末1回のみ：臨時リバランスなし", .05, .10, "annual", threshold_enabled=False),
    "Annual_Only_Loss_Harvest": RebalancePolicy("年末1回のみ：含み損ポジションを優先して損出し", .05, .10, "annual", threshold_enabled=False, loss_harvest=True),
    "Annual_Only_Conditional": RebalancePolicy("年末条件付きのみ：最大乖離がバンドの50%以上のとき", .05, .10, "conditional", threshold_enabled=False),
    "SemiAnnual_Only": RebalancePolicy("半年に1回のみ：臨時リバランスなし", .05, .10, "periodic", threshold_enabled=False, periodic_frequency="semiannual"),
    "Quarterly_Only": RebalancePolicy("四半期に1回のみ：臨時リバランスなし", .05, .10, "periodic", threshold_enabled=False, periodic_frequency="quarterly"),
}


@dataclass
class TaxLossCarryforwardLedger:
    """Annual-netting approximation of a loss pool usable in the next three years."""
    tax_rate: float
    pools: list[dict[str, float | int]] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    total_generated: float = 0.0
    total_used: float = 0.0
    total_expired: float = 0.0
    total_tax_before: float = 0.0
    total_tax_after: float = 0.0

    def settle_year(self, year: int, realized_gains: float, realized_losses: float,
                    policy: str = "", date: pd.Timestamp | None = None) -> float:
        """Net a calendar year's gains/losses, use oldest pools, and return tax due."""
        event_date = date if date is not None else pd.Timestamp(f"{year}-12-31")
        expired = sum(float(p["amount"]) for p in self.pools if int(p["expiry_year"]) < year)
        self.pools = [p for p in self.pools if int(p["expiry_year"]) >= year]
        net = realized_gains - realized_losses
        before = max(net, 0.0) * self.tax_rate
        used = 0.0
        if net > 0:
            remaining = net
            for pool in sorted(self.pools, key=lambda p: int(p["origin_year"])):
                offset = min(float(pool["amount"]), remaining)
                pool["amount"] = float(pool["amount"]) - offset
                remaining -= offset; used += offset
                if remaining <= 1e-15:
                    break
            self.pools = [p for p in self.pools if float(p["amount"]) > 1e-15]
            after = remaining * self.tax_rate
            generated = 0.0
        else:
            generated = max(-net, 0.0); after = 0.0
            if generated:
                self.pools.append({"origin_year": year, "expiry_year": year + 3, "amount": generated})
        # A pool from Y is usable through Y+3, then expires after that year's settlement.
        end_expired = sum(float(p["amount"]) for p in self.pools if int(p["expiry_year"]) == year)
        self.pools = [p for p in self.pools if int(p["expiry_year"]) != year]
        expired += end_expired
        self.total_generated += generated; self.total_used += used; self.total_expired += expired
        self.total_tax_before += before; self.total_tax_after += after
        self.events.append({"date": event_date, "year": year, "policy": policy,
                            "realized_gain_before_offset": realized_gains, "realized_loss": realized_losses,
                            "annual_net_realized_gain": net, "tax_loss_generated": generated,
                            "tax_loss_used": used, "tax_loss_expired": expired,
                            "tax_saved_by_carryforward": used * self.tax_rate,
                            "tax_cost_before_carryforward": before, "tax_cost_after_carryforward": after,
                            "tax_loss_pool_ending": sum(float(p["amount"]) for p in self.pools)})
        return after


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
    carryforward_equity: pd.Series | None = None
    tax_loss_events: list[dict[str, object]] = field(default_factory=list)
    realized_gain_before_offset: float = 0.0
    realized_loss: float = 0.0
    tax_loss_generated: float = 0.0
    tax_loss_used: float = 0.0
    tax_loss_expired: float = 0.0
    tax_cost_before_carryforward: float = 0.0
    tax_cost_after_carryforward: float = 0.0


def _bands(policy: RebalancePolicy) -> pd.Series:
    return pd.Series({asset: policy.core_band if asset in CORE_ASSETS else policy.support_band for asset in ASSETS})


def _minimum_trade_weights(current: pd.Series, target: pd.Series, bands: pd.Series) -> pd.Series:
    lower, upper = (target - bands).clip(lower=0), target + bands
    weights = current.clip(lower=lower, upper=upper)
    for _ in range(10):
        residual = 1.0 - float(weights.sum())
        if abs(residual) < 1e-12: break
        capacity = (upper - weights) if residual > 0 else (weights - lower)
        available = capacity[capacity > 1e-12]
        if available.empty: raise ValueError("Could not project weights into rebalance bands")
        weights.loc[available.index] += residual * available / available.sum()
        weights = weights.clip(lower=lower, upper=upper)
    return weights / weights.sum()


def _trade(values: pd.Series, basis: pd.Series, desired_weights: pd.Series, tax_rate: float,
           slippage_bps: float, fee_bps: float, loss_harvest: bool = False) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    total = float(values.sum()); desired = desired_weights * total
    sells = (values - desired).clip(lower=0); buys = (desired - values).clip(lower=0)
    sold_basis = (basis * (sells / values.replace(0, np.nan))).fillna(0)
    realized = sells - sold_basis
    realized_gain = float(realized.clip(lower=0).sum()); realized_loss = float((-realized.clip(upper=0)).sum())
    retained = pd.concat([values, desired], axis=1).min(axis=1)
    retained_basis = (basis * (retained / values.replace(0, np.nan))).fillna(0)
    harvest_mask = retained_basis > retained + 1e-12 if loss_harvest else pd.Series(False, index=values.index)
    harvested = retained.where(harvest_mask, 0.0); harvested_basis = retained_basis.where(harvest_mask, 0.0)
    realized_loss += float((harvested_basis - harvested).sum())
    tax = realized_gain * tax_rate
    traded = float(sells.sum() + buys.sum() + 2 * harvested.sum())
    slippage = traded * slippage_bps / 10000; fees = traded * fee_bps / 10000
    drag = tax + slippage + fees; scale = max(0.0, 1 - drag / total) if total else 0.0
    # A harvested retained lot is sold and repurchased, resetting its basis to market value.
    new_values = desired * scale
    new_basis = ((basis - sold_basis - harvested_basis) + buys + harvested) * scale
    stats = {"traded_notional": traded, "turnover": traded / total if total else 0.0,
             "sold_notional": float(sells.sum() + harvested.sum()), "bought_notional": float(buys.sum() + harvested.sum()),
             "taxable_gain": realized_gain, "realized_gain_before_offset": realized_gain,
             "realized_loss": realized_loss, "tax_cost": tax, "slippage": slippage, "fees": fees,
             "harvested_loss": float((harvested_basis - harvested).sum()),
             "trade_count": int((sells > 1e-12).sum() + (buys > 1e-12).sum() + 2 * harvest_mask.sum())}
    return new_values, new_basis, stats


def _period_end_due(policy: RebalancePolicy, panel: pd.DataFrame, i: int) -> bool:
    if i >= len(panel) - 1:
        return False
    date, next_date = panel.index[i], panel.index[i + 1]
    if policy.periodic_frequency == "annual":
        return next_date.year != date.year
    if policy.periodic_frequency == "semiannual":
        return (next_date.year, (next_date.month - 1) // 6) != (date.year, (date.month - 1) // 6)
    if policy.periodic_frequency == "quarterly":
        return (next_date.year, next_date.quarter) != (date.year, date.quarter)
    raise ValueError(f"Unknown periodic frequency: {policy.periodic_frequency}")


def _periodic_trigger(policy: RebalancePolicy, date: pd.Timestamp, drift: pd.Series,
                      bands: pd.Series, values: pd.Series, basis: pd.Series) -> bool:
    if policy.year_end_rule in {"annual", "periodic"}: return True
    if policy.year_end_rule == "none": return False
    if policy.year_end_rule == "biennial": return date.year % 2 == 0
    if policy.year_end_rule == "conditional": return bool((drift.abs() >= bands * .5).any())
    if policy.year_end_rule == "loss_only": return bool((values < basis - 1e-12).any())
    raise ValueError(f"Unknown periodic rule: {policy.year_end_rule}")


def simulate(prices: pd.DataFrame, name: str, policy: RebalancePolicy, tax_rate: float = .20315,
             slippage_bps: float = 5, fee_bps: float = 0, enable_tax_loss_carryforward: bool = False) -> AuditResult:
    target = normalized_weights(ALLOCATIONS[CURRENT]); panel = prices.loc[:, ASSETS].dropna()
    if len(panel) < 2: raise ValueError("At least two common price observations are required")
    values = target.copy(); basis = target.copy(); gross_values = target.copy(); gross_basis = target.copy()
    cf_values = target.copy(); cf_basis = target.copy(); ledger = TaxLossCarryforwardLedger(tax_rate)
    bands = _bands(policy); net_points = []; gross_points = []; cf_points = []; events = []
    turnover = tax = slippage = fees = realized_gains = realized_losses = 0.0; trade_count = 0
    annual_gain = annual_loss = 0.0
    for i, date in enumerate(panel.index):
        if i:
            relative = panel.iloc[i] / panel.iloc[i - 1]; values *= relative; gross_values *= relative; cf_values *= relative
        current = values / values.sum(); drift = current - target; breached = (drift.abs() > bands) if policy.threshold_enabled else pd.Series(False, index=ASSETS)
        is_year_end = i < len(panel) - 1 and panel.index[i + 1].year != date.year
        periodic_due = _period_end_due(policy, panel, i) and _periodic_trigger(policy, date, drift, bands, values, basis)
        if bool(breached.any()) or periodic_due:
            desired = _minimum_trade_weights(current, target, bands) if policy.sell_suppression else target
            reason = "threshold" if bool(breached.any()) else f"{policy.periodic_frequency}_{policy.year_end_rule}"; before = float(values.sum())
            values, basis, stats = _trade(values, basis, desired, tax_rate, slippage_bps, fee_bps, policy.loss_harvest)
            gross_values, gross_basis, _ = _trade(gross_values, gross_basis, desired, 0, 0, 0)
            cf_values, cf_basis, cf_stats = _trade(cf_values, cf_basis, desired, 0, slippage_bps, fee_bps, policy.loss_harvest)
            annual_gain += cf_stats["realized_gain_before_offset"]; annual_loss += cf_stats["realized_loss"]
            realized_gains += cf_stats["realized_gain_before_offset"]; realized_losses += cf_stats["realized_loss"]
            if stats["trade_count"]:
                turnover += stats["turnover"]; tax += stats["tax_cost"]; slippage += stats["slippage"]; fees += stats["fees"]; trade_count += int(stats["trade_count"])
                events.append({"date": date, "policy": name, "reason": reason, "breached_assets": "|".join(breached[breached].index),
                               "portfolio_value_before": before, **stats})
        settle = enable_tax_loss_carryforward and (is_year_end or i == len(panel) - 1)
        if settle:
            tax_due = ledger.settle_year(date.year, annual_gain, annual_loss, name, date)
            total = float(cf_values.sum()); scale = max(0.0, 1 - tax_due / total) if total else 0.0
            cf_values *= scale; cf_basis *= scale; annual_gain = annual_loss = 0.0
        net_points.append((date, float(values.sum()))); gross_points.append((date, float(gross_values.sum()))); cf_points.append((date, float(cf_values.sum())))
    net = pd.Series(dict(net_points), name=name); gross = pd.Series(dict(gross_points), name=name); cf = pd.Series(dict(cf_points), name=name)
    return AuditResult(net / net.iloc[0], gross / gross.iloc[0], events, turnover, tax, slippage, fees, trade_count,
                       cf / cf.iloc[0] if enable_tax_loss_carryforward else None, ledger.events, realized_gains, realized_losses,
                       ledger.total_generated, ledger.total_used, ledger.total_expired, ledger.total_tax_before, ledger.total_tax_after)


def _cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan


def _ratio_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().dropna(); maxdd = float((equity / equity.cummax() - 1).min()); downside = returns[returns < 0].std(); cagr = _cagr(equity)
    return {"CAGR": cagr, "MaxDD": maxdd, "Sharpe": float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)) if returns.std() else np.nan,
            "Sortino": float(returns.mean() / downside * np.sqrt(TRADING_DAYS)) if downside else np.nan, "Calmar": cagr / abs(maxdd) if maxdd else np.nan}


def _annual_returns(equity: pd.Series) -> pd.Series:
    return equity.groupby(equity.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)


def _rolling_win_rate(candidate: pd.Series, baseline: pd.Series, years: int) -> float:
    excess = (candidate / candidate.shift(years * TRADING_DAYS) - baseline / baseline.shift(years * TRADING_DAYS)).dropna()
    return float((excess > 0).mean()) if not excess.empty else np.nan


def _classify(name: str, row: Mapping[str, float], base: Mapping[str, float], carryforward: bool) -> str:
    if name == CURRENT_POLICY: return "現行リバランス維持"
    cagr_key = "CarryforwardAdjustedAfterTaxCAGR" if carryforward else "AfterTaxCAGR"
    calmar_key = "CarryforwardAdjustedAfterTaxCalmar" if carryforward else "AfterTaxCalmar"
    robust = (row[cagr_key] >= base[cagr_key] - .001 and row[calmar_key] >= base[calmar_key] * .95
              and row["MaxDD"] >= base["MaxDD"] - .01 and (np.isnan(row["Excess2022VsCurrent"]) or row["Excess2022VsCurrent"] >= -.01))
    burden_lower = (row["TradeCount"] < base["TradeCount"] and row["Turnover"] < base["Turnover"]
                    and row["TaxCost"] < base["TaxCost"] - 1e-12)
    if robust and burden_lower:
        if name == "Annual_Only": return "年末1回のみ候補"
        if name == "Annual_Only_Conditional": return "年末1回条件付き候補"
        if name in {"SemiAnnual_Only", "Quarterly_Only"}: return "半年/四半期定期リバランス候補"
    if robust: return "追加検証候補"
    return "不採用"


def build_tables(results: Mapping[str, AuditResult]) -> dict[str, pd.DataFrame]:
    equities = pd.DataFrame({n: r.equity for n, r in results.items()}); gross = pd.DataFrame({n: r.gross_equity for n, r in results.items()})
    cf_enabled = all(r.carryforward_equity is not None for r in results.values())
    cf_equities = pd.DataFrame({n: r.carryforward_equity for n, r in results.items()}) if cf_enabled else equities.copy()
    annual = pd.DataFrame({n: _annual_returns(cf_equities[n] if cf_enabled else r.equity) for n, r in results.items()}); years = (equities.index[-1] - equities.index[0]).days / 365.25
    rows = {}; baseline_annual = annual[CURRENT_POLICY]
    for name, result in results.items():
        net, pre, cf = _ratio_metrics(result.equity), _ratio_metrics(result.gross_equity), _ratio_metrics(cf_equities[name]); excess = annual[name] - baseline_annual
        rows[name] = {"Description": POLICIES[name].description, "CoreBand": POLICIES[name].core_band, "SupportBand": POLICIES[name].support_band,
          "PreTaxCAGR": pre["CAGR"], "AfterTaxCAGR": net["CAGR"], "MaxDD": cf["MaxDD"] if cf_enabled else net["MaxDD"], "AfterTaxCalmar": net["Calmar"],
          "CarryforwardAdjustedAfterTaxCAGR": cf["CAGR"], "CarryforwardAdjustedAfterTaxCalmar": cf["Calmar"], "Sharpe": cf["Sharpe"], "Sortino": cf["Sortino"],
          "Turnover": result.turnover, "TradeCount": result.trade_count, "RebalanceCount": len(result.events), "AnnualRebalanceCount": len(result.events) / years,
          "TaxableEventCount": sum(e["tax_cost"] > 0 for e in result.events), "TaxCost": result.tax_cost, "RealizedGainBeforeOffset": result.realized_gain_before_offset,
          "RealizedLoss": result.realized_loss, "TaxLossGenerated": result.tax_loss_generated, "TaxLossUsed": result.tax_loss_used, "TaxLossExpired": result.tax_loss_expired,
          "TaxSavedByCarryforward": 0.0,  # overwritten below from annual ledger events
          "TaxCostBeforeCarryforward": result.tax_cost_before_carryforward, "TaxCostAfterCarryforward": result.tax_cost_after_carryforward,
          "Slippage": result.slippage, "Fees": result.fees, "CostDragCAGR": pre["CAGR"] - net["CAGR"], "CarryforwardAdjustedCostDragCAGR": pre["CAGR"] - cf["CAGR"],
          "Return2022": float(annual.loc[2022, name]) if 2022 in annual.index else np.nan, "Excess2022VsCurrent": float(excess.loc[2022]) if 2022 in excess.index else np.nan,
          "AnnualWins": int((excess > 1e-12).sum()), "AnnualLosses": int((excess < -1e-12).sum()), "AnnualTies": int(np.isclose(excess, 0).sum()),
          "Rolling3YWinRate": _rolling_win_rate(cf_equities[name], cf_equities[CURRENT_POLICY], 3), "Rolling5YWinRate": _rolling_win_rate(cf_equities[name], cf_equities[CURRENT_POLICY], 5)}
        rows[name]["TaxSavedByCarryforward"] = sum(float(e["tax_saved_by_carryforward"]) for e in result.tax_loss_events)
    metrics = pd.DataFrame.from_dict(rows, orient="index"); base = metrics.loc[CURRENT_POLICY]
    metrics["Decision"] = [_classify(name, row, base, cf_enabled) for name, row in metrics.iterrows()]
    events = pd.DataFrame([event for result in results.values() for event in result.events]); tax_events = pd.DataFrame([event for result in results.values() for event in result.tax_loss_events])
    return {"metrics": metrics, "annual_returns": annual, "rebalance_events": events, "tax_loss_events": tax_events,
            "equity_curves": cf_equities if cf_enabled else equities, "standard_equity_curves": equities, "drawdown_curves": cf_equities / cf_equities.cummax() - 1}


def _overall_decision(metrics: pd.DataFrame, carryforward: bool) -> tuple[str, str]:
    candidates = metrics.loc[[name for name in SIMPLIFICATION_POLICIES if name != CURRENT_POLICY]]
    priority = ["年末1回のみ候補", "年末1回条件付き候補", "半年/四半期定期リバランス候補"]
    for decision in priority:
        matches = candidates[candidates.Decision == decision]
        if not matches.empty:
            return decision, f"{matches.index[0]} は現行比で耐性を概ね維持しつつ、売買・回転・税負担を明確に低下させた。即採用ではなく追加の実運用検証を要する。"
    return "現行リバランス維持", "単純化案が採用基準を満たさないか横一線であり、固定比率復元と年末損出しを兼ねる現行ホメオスタシスを優先する。"


def _report_table(frame: pd.DataFrame) -> str:
    return "```\n" + frame.to_string(float_format=lambda value: f"{value:.4f}") + "\n```"


def write_report(tables: Mapping[str, pd.DataFrame], output_dir: Path, start: pd.Timestamp, end: pd.Timestamp,
                 tax_rate: float, slippage_bps: float, fee_bps: float, carryforward: bool = False) -> None:
    metrics = tables["metrics"]; decision, reason = _overall_decision(metrics, carryforward)
    simple = metrics.loc[SIMPLIFICATION_POLICIES].copy(); base = metrics.loc[CURRENT_POLICY]
    effects = simple[["TradeCount", "Turnover", "TaxCost"]].subtract(base[["TradeCount", "Turnover", "TaxCost"]])
    effects.columns = ["TradeCountChangeVsCurrent", "TurnoverChangeVsCurrent", "TaxCostChangeVsCurrent"]
    side_effects = simple[["MaxDD", "AfterTaxCalmar", "CarryforwardAdjustedAfterTaxCalmar", "Return2022", "Excess2022VsCurrent"]]
    lines = ["# L.U.M.U.S.-8 リバランス幅・損失繰越 頑健性監査", "", "## 【結論】", f"**{decision}** — {reason}",
      "結果が良くても即本番採用せず、横一線なら現行ルールを優先する。現行ルールは固定比率復元と年末損出しを兼ねたホメオスタシス機構である。", "",
      "## 【前回監査との差分】", "既存8ポリシーを維持したまま、年末単独・半年・四半期の定期リバランス簡素化比較を追加した。通常簡易税モデルと、年内損益通算・3年繰越を近似した指標を並記する。", "",
      "## 【損失繰越モデルの説明】", "各年の実現益と実現損を年間でネットし、ネット損失を翌年以後3年間利用可能な tax_loss_pool とする。ネット利益には古いpoolから充当し、残額だけを課税する。Y年損失はY+1〜Y+3年に利用でき、Y+3年末の未使用残は失効する。", "",
      "## 【年末1回リバランス検証】", "現行の臨時+年末リバランスと、年末単独・定期リバランス簡素化ポリシー群を区別して比較する。", _report_table(simple[["Description", "PreTaxCAGR", "AfterTaxCAGR", "CarryforwardAdjustedAfterTaxCAGR", "MaxDD", "AfterTaxCalmar", "CarryforwardAdjustedAfterTaxCalmar", "Return2022", "Decision"]]), "",
      "## 【単純化による効果】", "負値は現行より負担が減ったことを示す。", _report_table(effects), "",
      "## 【単純化による副作用】", "MaxDD、2022年耐性、税後Calmarの悪化有無を確認する。", _report_table(side_effects), "",
      "## 【比較サマリー】", _report_table(metrics[["Description", "AfterTaxCAGR", "CarryforwardAdjustedAfterTaxCAGR", "CarryforwardAdjustedAfterTaxCalmar", "MaxDD", "Turnover", "Return2022", "Decision"]]), "",
      "## 【損失繰越効果】", _report_table(metrics[["TaxLossGenerated", "TaxLossUsed", "TaxLossExpired", "TaxSavedByCarryforward", "TaxCostBeforeCarryforward", "TaxCostAfterCarryforward"]]), "",
      "## 【税後・コスト後評価】", _report_table(metrics[["PreTaxCAGR", "AfterTaxCAGR", "CarryforwardAdjustedAfterTaxCAGR", "CostDragCAGR", "CarryforwardAdjustedCostDragCAGR"]]), "",
      "## 【リバランス負荷】", _report_table(metrics[["Turnover", "TradeCount", "RebalanceCount", "AnnualRebalanceCount", "TaxableEventCount", "TaxCost"]]), "",
      "## 【ドローダウン耐性】", _report_table(metrics[["MaxDD", "AfterTaxCalmar", "CarryforwardAdjustedAfterTaxCalmar", "Return2022", "Excess2022VsCurrent"]]), "",
      "## 【年次勝敗・ローリング勝率】", _report_table(metrics[["AnnualWins", "AnnualLosses", "AnnualTies", "Rolling3YWinRate", "Rolling5YWinRate", "Excess2022VsCurrent"]]), "",
      "## 【採用判断】", _report_table(simple[["Decision"]]), "", "## 【判定基準】", "- 年末1回のみは、税後CAGR・税後Calmarがほぼ同等、MaxDD悪化が1ポイント以内、2022年悪化が軽微で、TradeCount・Turnover・TaxCostがすべて低下した場合のみ候補とする。", "- 売買が減ってもMaxDDまたは2022年耐性が明確に悪化する場合は現行維持とし、横一線でも現行維持を優先する。", "",
      "## 【重要な制約】", f"- 検証期間: {start.date()}〜{end.date()}。税率={tax_rate:.3%}、スリッページ={slippage_bps:g}bps、手数料={fee_bps:g}bps。",
      "- 本モデルは税務助言ではなく、L.U.M.U.S.-8バックテスト上の近似である。実際の判断には確定申告、制度要件、損益通算対象、証券会社の口座区分、配当課税方式等が関係する。",
      "- 年間ネット損益近似であり、申告・配当・ロット別取得価額・口座間通算・ウォッシュセール等の制度要件を完全再現しない。", "- BNDX/GLDM/BTC等にはIEF/GLD/BTC-USDの代理資産制約がある。", "- 過去データによる予備監査であり、将来成果を保証しない。"]
    filename = "rebalance_band_loss_carryforward_summary_report.md" if carryforward else "rebalance_band_summary_report_ja.md"
    (output_dir / filename).write_text("\n".join(lines), encoding="utf-8")


def _plots(tables: Mapping[str, pd.DataFrame], output_dir: Path, carryforward: bool) -> None:
    ax = tables["equity_curves"].plot(figsize=(12, 7), alpha=.82, title="L.U.M.U.S.-8 rebalance policy audit: after-tax equity")
    ax.set_ylabel("Growth of 1"); ax.figure.tight_layout(); ax.figure.savefig(output_dir / "equity_curve.png", dpi=150); plt.close(ax.figure)
    ax = tables["drawdown_curves"].plot(figsize=(12, 7), alpha=.82, title="L.U.M.U.S.-8 rebalance policy audit: drawdown")
    ax.set_ylabel("Drawdown"); ax.figure.tight_layout(); ax.figure.savefig(output_dir / "drawdown_curve.png", dpi=150); plt.close(ax.figure)
    if not carryforward:
        ax = tables["metrics"][["Turnover", "TaxCost"]].plot.bar(figsize=(12, 7), subplots=True, title="Trading and tax burden")
        fig = ax[0].figure; fig.tight_layout(); fig.savefig(output_dir / "trading_tax_burden.png", dpi=150); plt.close(fig)


def run_audit(raw_prices: pd.DataFrame, output_dir: Path, tax_rate: float = .20315, slippage_bps: float = 5,
              fee_bps: float = 0, enable_tax_loss_carryforward: bool = False) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True); prices, coverage = prepare_prices(raw_prices); prices = prices.loc[:, ASSETS].dropna()
    results = {name: simulate(prices, name, policy, tax_rate, slippage_bps, fee_bps, enable_tax_loss_carryforward) for name, policy in POLICIES.items()}; tables = build_tables(results)
    if enable_tax_loss_carryforward:
        files = {"metrics": "rebalance_band_loss_carryforward_metrics.csv", "tax_loss_events": "rebalance_band_loss_carryforward_tax_loss_events.csv",
                 "rebalance_events": "rebalance_band_loss_carryforward_rebalance_events.csv", "annual_returns": "rebalance_band_loss_carryforward_annual_returns.csv",
                 "equity_curves": "rebalance_band_loss_carryforward_equity_curves.csv"}
    else:
        files = {"metrics": "rebalance_band_metrics.csv", "annual_returns": "rebalance_band_annual_returns.csv", "rebalance_events": "rebalance_band_events.csv",
                 "equity_curves": "rebalance_band_equity_curves.csv", "drawdown_curves": "rebalance_band_drawdown_curves.csv"}
    for key, filename in files.items(): tables[key].to_csv(output_dir / filename, index=key not in {"rebalance_events", "tax_loss_events"}, index_label="policy" if key == "metrics" else "date")
    coverage.to_csv(output_dir / "price_coverage.csv", index_label="ticker"); _plots(tables, output_dir, enable_tax_loss_carryforward)
    write_report(tables, output_dir, prices.index[0], prices.index[-1], tax_rate, slippage_bps, fee_bps, enable_tax_loss_carryforward)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--start", default="2010-01-01"); parser.add_argument("--end", default=None)
    parser.add_argument("--tax-rate", type=float, default=.20315); parser.add_argument("--slippage-bps", type=float, default=5); parser.add_argument("--fee-bps", type=float, default=0)
    parser.add_argument("--enable-tax-loss-carryforward", action="store_true", help="Enable annual-netting three-year tax-loss carryforward approximation")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rebalance_band_audit")); args = parser.parse_args()
    raw = download_prices(ASSETS, args.start, args.end, args.output_dir / "price_cache")
    tables = run_audit(raw, args.output_dir, args.tax_rate, args.slippage_bps, args.fee_bps, args.enable_tax_loss_carryforward)
    print(tables["metrics"].to_string()); print(f"Artifacts saved under: {args.output_dir.resolve()}")


if __name__ == "__main__": main()
