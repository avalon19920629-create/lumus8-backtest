import numpy as np
import pandas as pd

from allocation_robustness_audit import ALLOCATIONS, ASSETS, CORE_ASSETS, CURRENT, normalized_weights, run_audit, simulate


def sample_prices():
    dates = pd.bdate_range("2016-01-01", periods=2100)
    rng = np.random.default_rng(109)
    returns = rng.normal(.00018, .008, (len(dates), len(ASSETS)))
    returns[:, ASSETS.index("BTC-USD")] += .00015
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=ASSETS)


def test_candidate_grid_is_bounded_and_normalized():
    current = normalized_weights(ALLOCATIONS[CURRENT])
    assert len(ALLOCATIONS) >= 10
    for weights in ALLOCATIONS.values():
        candidate = normalized_weights(weights)
        assert np.isclose(candidate.sum(), 1)
        assert candidate.max() <= .35
        assert ((candidate - current).abs() <= .12).all()


def test_threshold_and_year_end_rebalances_are_recorded_with_costs():
    prices = sample_prices()
    result = simulate(prices, CURRENT, ALLOCATIONS[CURRENT], tax_rate=.2, slippage_bps=5, fee_bps=1)
    assert result.events
    assert {event["reason"] for event in result.events} <= {"year_end", "threshold"}
    assert result.trade_count == sum(event["trade_count"] for event in result.events)
    assert result.equity.iloc[-1] <= result.gross_equity.iloc[-1]
    assert result.slippage > 0 and result.fees > 0
    assert all((asset in CORE_ASSETS) == (asset in {"VT", "TLT", "TIP", "GLD", "BTC-USD"}) for asset in ASSETS)


def test_required_artifacts_metrics_and_japanese_report(tmp_path):
    tables = run_audit(sample_prices(), tmp_path)
    required = ["allocation_summary_report.md", "allocation_metrics.csv", "allocation_annual_returns.csv",
                "allocation_rebalance_events.csv", "allocation_equity_curves.csv", "allocation_drawdown_curves.csv",
                "equity_curve.png", "drawdown_curve.png"]
    assert all((tmp_path / name).is_file() for name in required)
    expected = {"PreTaxCAGR", "AfterTaxCAGR", "MaxDD", "Sharpe", "Sortino", "Calmar", "AfterTaxCalmar",
                "Turnover", "RebalanceCount", "TaxCost", "Slippage", "Fees", "CostDragCAGR", "Return2022"}
    assert expected <= set(tables["metrics"].columns)
    report = (tmp_path / "allocation_summary_report.md").read_text(encoding="utf-8")
    for section in ["【結論】", "【比較サマリー】", "【税後・コスト後評価】", "【リバランス負荷】",
                    "【ドローダウン耐性】", "【年次勝敗】", "【頑健性評価】", "【採用判断】"]:
        assert section in report
