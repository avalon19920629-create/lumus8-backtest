import numpy as np
import pandas as pd

from allocation_robustness_audit import ASSETS
from rebalance_band_robustness_audit import CURRENT_POLICY, DECISIONS, POLICIES, _bands, _minimum_trade_weights, run_audit, simulate


def sample_prices():
    dates = pd.bdate_range("2015-10-08", periods=2200)
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
