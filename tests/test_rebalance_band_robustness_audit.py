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


def test_eight_predefined_policies_cover_requested_variants():
    assert len(POLICIES) == 8
    assert {p.year_end_rule for p in POLICIES.values()} == {"annual", "none", "biennial", "conditional", "loss_only"}
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
    for section in ["【結論】", "【比較サマリー】", "【年次勝敗・ローリング勝率】", "【判定基準】", "【重要な制約】"]:
        assert section in report
