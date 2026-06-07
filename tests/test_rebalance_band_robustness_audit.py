import numpy as np
import pandas as pd

from allocation_robustness_audit import ASSETS
from rebalance_band_robustness_audit import (
    CONDITIONAL_THRESHOLD_POLICIES, CURRENT_POLICY, DECISIONS, POLICIES, _bands, _minimum_trade_weights,
    run_audit, run_conditional_threshold_audit, simulate,
)


def sample_prices(periods=2200):
    dates = pd.bdate_range("2015-10-08", periods=periods)
    rng = np.random.default_rng(812)
    returns = rng.normal(.00018, .008, (len(dates), len(ASSETS)))
    returns[:, ASSETS.index("BTC-USD")] += .00015
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=ASSETS)


def test_existing_eight_and_simplification_policies_cover_requested_variants():
    existing = {CURRENT_POLICY, "Loose_7_5_12_5_Annual", "Looser_10_15_Annual", "Threshold_Only",
                "Biennial_Year_End", "Conditional_Year_End", "Loss_Harvest_Year_End", "Sell_Suppressed"}
    requested = {CURRENT_POLICY, "Annual_Only", "Annual_Only_Loss_Harvest", "Annual_Only_Conditional",
                 "SemiAnnual_Only", "Quarterly_Only", "Threshold_Only"}
    assert existing <= set(POLICIES)
    assert requested <= set(POLICIES)
    assert len(POLICIES) == 13
    assert sum(p.sell_suppression for p in POLICIES.values()) == 1


def test_sell_suppression_projects_to_band_and_trades_less_than_full_rebalance():
    target = pd.Series({asset: 1 / len(ASSETS) for asset in ASSETS})
    current = target.copy(); current.iloc[0] += .08; current.iloc[1:] -= .08 / (len(ASSETS) - 1)
    bands = _bands(POLICIES[CURRENT_POLICY]); projected = _minimum_trade_weights(current, target, bands)
    assert np.isclose(projected.sum(), 1)
    assert ((projected - target).abs() <= bands + 1e-12).all()
    assert (projected - current).abs().sum() < (target - current).abs().sum()


def test_threshold_only_has_no_year_end_events():
    result = simulate(sample_prices(), "Threshold_Only", POLICIES["Threshold_Only"])
    assert all(event["reason"] == "threshold" for event in result.events)


def test_annual_only_never_triggers_threshold_and_rebalances_only_at_year_end():
    result = simulate(sample_prices(), "Annual_Only", POLICIES["Annual_Only"])
    assert result.events
    assert all(event["reason"] == "annual_annual" for event in result.events)
    assert all((event["date"] + pd.offsets.BDay()).year > event["date"].year for event in result.events)


def test_semiannual_and_quarterly_only_rebalance_at_requested_period_ends():
    prices = sample_prices()
    semi = simulate(prices, "SemiAnnual_Only", POLICIES["SemiAnnual_Only"])
    quarterly = simulate(prices, "Quarterly_Only", POLICIES["Quarterly_Only"])
    assert all(event["reason"] == "semiannual_periodic" for event in semi.events)
    assert all(event["reason"] == "quarterly_periodic" for event in quarterly.events)
    assert all(((event["date"] + pd.offsets.BDay()).year, ((event["date"] + pd.offsets.BDay()).month - 1) // 6)
               != (event["date"].year, (event["date"].month - 1) // 6) for event in semi.events)
    assert all(((event["date"] + pd.offsets.BDay()).year, (event["date"] + pd.offsets.BDay()).quarter)
               != (event["date"].year, event["date"].quarter) for event in quarterly.events)
    assert len(quarterly.events) >= len(semi.events)


def test_annual_only_loss_harvest_generates_at_least_as_much_realized_loss():
    plain = simulate(sample_prices(), "Annual_Only", POLICIES["Annual_Only"], enable_tax_loss_carryforward=True)
    harvest = simulate(sample_prices(), "Annual_Only_Loss_Harvest", POLICIES["Annual_Only_Loss_Harvest"], enable_tax_loss_carryforward=True)
    assert harvest.realized_loss >= plain.realized_loss


def test_required_metrics_artifacts_and_japanese_report(tmp_path):
    tables = run_audit(sample_prices(), tmp_path, fee_bps=1)
    required = ["rebalance_band_summary_report_ja.md", "rebalance_band_metrics.csv", "rebalance_band_annual_returns.csv",
                "rebalance_band_events.csv", "rebalance_band_equity_curves.csv", "rebalance_band_drawdown_curves.csv",
                "price_coverage.csv", "equity_curve.png", "trading_tax_burden.png"]
    assert all((tmp_path / name).is_file() for name in required)
    expected = {"AfterTaxCAGR", "MaxDD", "AfterTaxCalmar", "Sharpe", "Sortino", "Turnover", "TradeCount", "RebalanceCount",
                "TaxableEventCount", "TaxCost", "CostDragCAGR", "Return2022", "AnnualWins", "AnnualLosses", "Rolling3YWinRate", "Rolling5YWinRate", "Decision"}
    assert expected <= set(tables["metrics"].columns)
    assert set(tables["metrics"].Decision) <= DECISIONS
    report = (tmp_path / "rebalance_band_summary_report_ja.md").read_text(encoding="utf-8")
    for section in ["【結論】", "【年末1回リバランス検証】", "【単純化による効果】", "【単純化による副作用】",
                    "【比較サマリー】", "【年次勝敗・ローリング勝率】", "【判定基準】", "【重要な制約】"]:
        assert section in report


def test_tax_loss_pool_generation_use_expiry_and_savings():
    from rebalance_band_robustness_audit import TaxLossCarryforwardLedger

    ledger = TaxLossCarryforwardLedger(.20)
    assert ledger.settle_year(2022, 0, 100) == 0
    assert ledger.pools == [{"origin_year": 2022, "expiry_year": 2025, "amount": 100}]
    assert ledger.settle_year(2023, 40, 0) == 0
    assert ledger.total_used == 40
    assert ledger.events[-1]["tax_saved_by_carryforward"] == 8
    ledger.settle_year(2024, 0, 0)
    ledger.settle_year(2025, 0, 0)
    assert ledger.total_expired == 60
    assert not ledger.pools


def test_carryforward_mode_outputs_metrics_artifacts_and_required_report_sections(tmp_path):
    tables = run_audit(sample_prices(), tmp_path, enable_tax_loss_carryforward=True)
    required = ["rebalance_band_loss_carryforward_summary_report.md", "rebalance_band_loss_carryforward_metrics.csv",
                "rebalance_band_loss_carryforward_tax_loss_events.csv", "rebalance_band_loss_carryforward_rebalance_events.csv",
                "rebalance_band_loss_carryforward_annual_returns.csv", "rebalance_band_loss_carryforward_equity_curves.csv",
                "equity_curve.png", "drawdown_curve.png"]
    assert all((tmp_path / name).is_file() for name in required)
    expected = {"RealizedGainBeforeOffset", "RealizedLoss", "TaxLossGenerated", "TaxLossUsed", "TaxLossExpired",
                "TaxSavedByCarryforward", "TaxCostBeforeCarryforward", "TaxCostAfterCarryforward",
                "CarryforwardAdjustedAfterTaxCAGR", "CarryforwardAdjustedAfterTaxCalmar", "CarryforwardAdjustedCostDragCAGR"}
    assert expected <= set(tables["metrics"].columns)
    assert len(tables["metrics"]) == 13
    report = (tmp_path / "rebalance_band_loss_carryforward_summary_report.md").read_text(encoding="utf-8")
    for section in ["【結論】", "【前回監査との差分】", "【損失繰越モデルの説明】", "【年末1回リバランス検証】",
                    "【単純化による効果】", "【単純化による副作用】", "【比較サマリー】", "【損失繰越効果】",
                    "【税後・コスト後評価】", "【リバランス負荷】", "【ドローダウン耐性】", "【採用判断】", "【重要な制約】"]:
        assert section in report
    assert "税務助言ではなく" in report


def test_conditional_threshold_policies_keep_current_emergency_bands_enabled():
    for threshold in (25, 50, 75, 100):
        policy = CONDITIONAL_THRESHOLD_POLICIES[f"Conditional_Year_End_{threshold}"]
        assert policy.threshold_enabled
        assert policy.year_end_rule == "conditional"
        assert policy.conditional_year_end_fraction == threshold / 100


def test_conditional_threshold_only_suppresses_year_end_and_is_monotonic():
    prices = sample_prices(); shocked = prices.copy()
    shocked.loc[shocked.index[100]:, "BTC-USD"] *= 20
    counts = []
    for threshold in (25, 50, 75, 100, 125):
        name = f"Conditional_Year_End_{threshold}"
        policy = CONDITIONAL_THRESHOLD_POLICIES[name]
        shocked_result = simulate(shocked, name, policy)
        result = simulate(prices, name, policy)
        assert any(event["reason"] == "threshold" for event in shocked_result.events)
        assert all(event["reason"] in {"threshold", "annual_conditional"} for event in result.events)
        counts.append(sum(event["reason"] == "annual_conditional" for event in result.events))
    assert counts == sorted(counts, reverse=True)


def test_conditional_threshold_sweep_carryforward_artifacts_and_report(tmp_path):
    tables = run_conditional_threshold_audit(sample_prices(600), tmp_path, enable_tax_loss_carryforward=True)
    required = ["conditional_threshold_summary_report.md", "conditional_threshold_metrics.csv",
                "conditional_threshold_rebalance_events.csv", "conditional_threshold_tax_loss_events.csv",
                "conditional_threshold_annual_returns.csv", "conditional_threshold_equity_curves.csv",
                "conditional_threshold_drawdown_curves.csv", "equity_curve.png", "drawdown_curve.png"]
    assert all((tmp_path / name).is_file() for name in required)
    assert set(CONDITIONAL_THRESHOLD_POLICIES) == set(tables["metrics"].index)
    assert {"YearEndRebalanceCount", "MaxObservedDrift", "TaxCostAfterCarryforward"} <= set(tables["metrics"].columns)
    report = (tmp_path / "conditional_threshold_summary_report.md").read_text(encoding="utf-8")
    for section in ["【結論】", "【検証目的】", "【比較対象】", "【閾値感度サマリー】", "【現行リバランスとの比較】",
                    "【税後・損失繰越込み評価】", "【売買回数・課税負荷】", "【2022年耐性】",
                    "【年末リバランス抑制の副作用】", "【採用判断】", "【重要な制約】"]:
        assert section in report


def test_conditional_threshold_sweep_supports_standard_tax_model(tmp_path):
    tables = run_conditional_threshold_audit(sample_prices(600), tmp_path)
    assert tables["metrics"]["AfterTaxCAGR"].notna().all()
    assert (tmp_path / "conditional_threshold_tax_loss_events.csv").is_file()
    assert (tmp_path / "conditional_threshold_summary_report.md").is_file()
