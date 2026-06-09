import numpy as np
import pandas as pd

from lumus8_backtest import PORTFOLIOS
from lumus8_optimization import (
    ASSETS, OPTIMIZED_NAMES, constraints_satisfied, feasible_weights, run_optimization,
)


def test_feasible_search_and_robust_candidate_obey_all_constraints(tmp_path):
    candidates = feasible_weights(40, seed=3)
    assert all(constraints_satisfied(row) for _, row in candidates.iterrows())

    dates = pd.bdate_range("2018-01-02", "2023-12-29")
    tickers = sorted({ticker for weights in PORTFOLIOS.values() for ticker in weights} | set(ASSETS))
    rng = np.random.default_rng(12)
    common = rng.normal(0.00015, 0.004, len(dates))
    returns = np.column_stack([
        common + rng.normal(0, 0.003 + i * 0.0001, len(dates)) for i in range(len(tickers))
    ])
    raw = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=tickers)
    tables = run_optimization(raw, tmp_path, samples=80, seed=4)

    assert tables["optimized_weights"].index.tolist() == list(OPTIMIZED_NAMES)
    assert all(constraints_satisfied(row) for _, row in tables["optimized_weights"].iterrows())
    assert set(tables["optimized_metrics"]["Grade"]) <= set("SABCD")
    for filename in [
        "optimized_weights.csv", "optimized_metrics.csv", "optimized_common_period_metrics.csv",
        "optimized_annual_returns.csv", "optimized_equity_curves.csv", "optimized_drawdowns.csv",
        "optimized_comparison_2022.csv", "optimized_sp500_deltas.csv", "optimized_report.md",
        "efficient_frontier.png", "optimized_equity_curves.png", "optimized_drawdowns.png",
    ]:
        assert (tmp_path / filename).is_file()
    report = (tmp_path / "optimized_report.md").read_text(encoding="utf-8")
    assert "最適化結果は将来の最適比率を保証しない" in report
