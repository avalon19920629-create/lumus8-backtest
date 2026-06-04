import numpy as np
import pandas as pd

from lumus8_backtest import (
    LUMUS_ASSETS,
    PORTFOLIOS,
    REAL_ESTATE_AUDIT_ASSETS,
    REAL_ESTATE_VARIANTS,
    annual_asset_returns,
    annual_returns,
    backtest_portfolio,
    contribution_by_period,
    correlation_analysis,
    drawdown_details,
    prepare_prices,
    real_estate_variant_audit,
    risk_contributions,
    stress_analysis,
)


def test_prepare_prices_does_not_backfill_before_inception_and_limits_forward_fill():
    dates = pd.date_range("2020-01-01", periods=7)
    raw = pd.DataFrame({"A": [np.nan, 10, np.nan, 12, np.nan, np.nan, np.nan]}, index=dates)
    cleaned, coverage = prepare_prices(raw, max_ffill_days=2)
    assert pd.isna(cleaned.loc[dates[0], "A"])
    assert cleaned.loc[dates[2], "A"] == 10
    assert cleaned.loc[dates[5], "A"] == 12
    assert pd.isna(cleaned.loc[dates[6], "A"])
    assert coverage.loc["A", "FilledInternalGaps"] == 1


def test_backtest_starts_at_common_inception_and_rebalances_monthly():
    dates = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
    prices = pd.DataFrame({"A": [100, 110, 110, 110], "B": [np.nan, 100, 100, 110]}, index=dates)
    result = backtest_portfolio(prices, {"A": 0.5, "B": 0.5})
    assert result.start == pd.Timestamp("2020-01-31")
    assert result.equity.index[0] == pd.Timestamp("2020-01-31")
    assert np.isclose(result.equity.iloc[-1], 1.05)


def test_drawdown_details_reports_recovery_dates():
    equity = pd.Series([1.0, 1.2, 0.9, 1.0, 1.2], index=pd.date_range("2020-01-01", periods=5))
    details = drawdown_details(equity)
    assert details["MaxDDPeakDate"] == pd.Timestamp("2020-01-02")
    assert details["MaxDDTroughDate"] == pd.Timestamp("2020-01-03")
    assert details["RecoveryDate"] == pd.Timestamp("2020-01-05")
    assert np.isclose(details["MaxDD"], -0.25)


def test_reporting_helpers_produce_expected_tables():
    dates = pd.bdate_range("2007-12-28", "2022-12-30")
    tickers = sorted({ticker for weights in PORTFOLIOS.values() for ticker in weights})
    trend = np.linspace(100, 200, len(dates))
    prices = pd.DataFrame({ticker: trend * (1 + i / 100) for i, ticker in enumerate(tickers)}, index=dates)
    equities = {"SP500": backtest_portfolio(prices, {"SPY": 1}).equity}
    annual = annual_returns(equities)
    stress = stress_analysis(equities)
    risk = risk_contributions(prices, {"PAIR": {"SPY": 0.5, "IEF": 0.5}})
    assert 2022 in annual.index
    assert stress.loc[("GFC_2008", "SP500"), "Available"]
    assert np.isclose(risk.loc["PAIR", "RiskContributionPct"].sum(), 1)


def test_correlation_analysis_saves_matrix_clusters_and_plots(tmp_path):
    generator = np.random.default_rng(8)
    dates = pd.bdate_range("2020-01-01", periods=80)
    base_returns = generator.normal(0.0002, 0.01, (len(dates), len(LUMUS_ASSETS)))
    prices = pd.DataFrame(100 * np.cumprod(1 + base_returns, axis=0), index=dates, columns=LUMUS_ASSETS)
    correlation, clusters = correlation_analysis(prices, tmp_path)
    assert correlation.index.tolist() == LUMUS_ASSETS
    assert correlation.columns.tolist() == LUMUS_ASSETS
    assert set(clusters["LinkageMethod"]) == {"ward", "average"}
    assert set(clusters["Ticker"]) == set(LUMUS_ASSETS)
    assert clusters["Cluster"].between(1, 4).all()
    for filename in ["correlation_matrix.csv", "correlation_heatmap.png", "cluster_results.csv", "cluster_dendrogram.png", "cluster_dendrogram_average.png"]:
        assert (tmp_path / filename).is_file()


def test_phase_two_asset_returns_and_contributions_reconcile():
    dates = pd.bdate_range("2021-12-30", "2022-02-04")
    tickers = sorted({ticker for weights in PORTFOLIOS.values() for ticker in weights})
    prices = pd.DataFrame(100.0, index=dates, columns=tickers)
    prices.loc["2022-01-03":, "VT"] = 110.0
    prices.loc["2022-02-01":, "GLD"] = 105.0
    annual = annual_asset_returns(prices)
    contributions = contribution_by_period(prices, {"TEST_2022": ("2022-01-01", "2022-12-31")})
    assert 2022 in annual.index
    assert np.isclose(annual.loc[2022, "VT"], 0.1)
    for portfolio in ["LUMUS_CORE_WITH_BTC", "LUMUS_CORE_NO_BTC"]:
        rows = contributions.loc[("TEST_2022", portfolio)]
        assert rows["Available"].all()
        assert np.isclose(rows["ReturnContribution"].sum(), rows["PortfolioReturn"].iloc[0])


def test_real_estate_variant_audit_compares_vnq_and_xlre(tmp_path):
    generator = np.random.default_rng(13)
    dates = pd.bdate_range("2021-12-30", "2022-12-30")
    tickers = sorted({ticker for weights in PORTFOLIOS.values() for ticker in weights})
    returns = generator.normal(0.0001, 0.005, (len(dates), len(tickers)))
    prices = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=tickers)
    results = {name: backtest_portfolio(prices, PORTFOLIOS[name]) for name in REAL_ESTATE_VARIANTS}
    tables = real_estate_variant_audit(prices, results, tmp_path)
    assert tables["real_estate_variant_metrics"].index.tolist() == list(REAL_ESTATE_VARIANTS)
    assert set(tables["real_estate_variant_cluster_results"]["Portfolio"]) == set(REAL_ESTATE_VARIANTS)
    contribution = tables["real_estate_variant_contribution_2022"]
    assert set(contribution.index.get_level_values("Ticker")) == set(REAL_ESTATE_AUDIT_ASSETS)
    for portfolio in REAL_ESTATE_VARIANTS:
        rows = contribution.loc[("INFLATION_2022", portfolio)]
        assert np.isclose(rows["ReturnContribution"].sum(), rows["PortfolioReturn"].iloc[0])
    for stem in ["lumus_vnq", "lumus_xlre"]:
        assert (tmp_path / f"{stem}_correlation_matrix.csv").is_file()
        assert (tmp_path / f"{stem}_cluster_dendrogram.png").is_file()
