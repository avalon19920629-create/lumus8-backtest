import numpy as np
import pandas as pd

from allocation_robustness_audit import ASSETS
from growth_heavy_decomposition_audit import ALLOCATIONS, REQUIRED_METRICS, allocation_distance, normalized_weights, run_audit


def sample_prices():
    dates = pd.bdate_range("2015-10-08", periods=2200)
    rng = np.random.default_rng(508)
    returns = rng.normal(.00018, .007, (len(dates), len(ASSETS)))
    returns[:, ASSETS.index("VT")] += .00008
    returns[:, ASSETS.index("BTC-USD")] += .00012
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=ASSETS)


def test_candidates_are_predefined_normalized_and_btc_is_bounded():
    required = {"Current_LUMUS8", "Growth_Heavy", "VT_Only_Heavy", "BTC_Only_Heavy"}
    assert required <= set(ALLOCATIONS)
    assert len(ALLOCATIONS) == 12
    for name, weights in ALLOCATIONS.items():
        candidate = normalized_weights(weights)
        assert np.isclose(candidate.sum(), 1)
        assert candidate["BTC-USD"] <= .07
        assert allocation_distance(name) <= .15
    # Plain constants are loaded before any prices and every candidate has exactly the known universe.
    assert all(set(weights) == set(ASSETS) for weights in ALLOCATIONS.values())


def test_required_artifacts_metrics_and_japanese_report(tmp_path):
    tables = run_audit(sample_prices(), tmp_path, fee_bps=1)
    required_files = ["growth_decomposition_summary_report.md", "growth_decomposition_metrics.csv",
                      "growth_decomposition_annual_returns.csv", "growth_decomposition_rebalance_events.csv",
                      "growth_decomposition_equity_curves.csv", "growth_decomposition_drawdown_curves.csv",
                      "equity_curve.png", "drawdown_curve.png"]
    assert all((tmp_path / filename).is_file() for filename in required_files)
    assert REQUIRED_METRICS <= set(tables["metrics"].columns)
    assert {"AllocationDistance", "AdoptionDecision", "EarlyHalfExcessVsCurrent", "LateHalfExcessVsCurrent"} <= set(tables["metrics"].columns)
    report = (tmp_path / "growth_decomposition_summary_report.md").read_text(encoding="utf-8")
    for section in ["【結論】", "【比較サマリー】", "【Growth_Heavyの勝因分解】", "【税後・コスト後評価】",
                    "【リバランス負荷】", "【2022年耐性】", "【年次勝敗】", "【ローリング勝率】",
                    "【採用候補の分類】", "【重要な制約】", "【最重要結論】"]:
        assert section in report
