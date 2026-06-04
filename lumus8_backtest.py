"""L.U.M.U.S.-8 Core portfolio backtest using adjusted ETF prices.

The script intentionally keeps data handling conservative: prices before an asset's
inception are never invented, and only short gaps inside an existing series are
forward-filled.  Each portfolio starts when all of its required assets are usable.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

START = "2005-01-01"
DEFAULT_OUTPUT_DIR = Path("output")
TRADING_DAYS = 252
MAX_FORWARD_FILL_DAYS = 5
CLUSTER_COUNT = 4

PORTFOLIOS: dict[str, dict[str, float]] = {
    "SP500": {"SPY": 1.0},
    "60_40": {"SPY": 0.60, "IEF": 0.40},
    "DALIO_AW": {"SPY": 0.30, "TLT": 0.40, "IEF": 0.15, "GLD": 0.075, "DBC": 0.075},
    # The original weights sum to 90%; normalization is deliberate and reported.
    "LUMUS_CORE_WITH_BTC": {
        "VT": 0.25, "IEF": 0.10, "TLT": 0.12, "TIP": 0.10, "GLD": 0.10,
        "VNQ": 0.06, "DBC": 0.07, "SHY": 0.05, "BTC-USD": 0.05,
    },
    # BTC is removed and the remaining original weights are normalized.
    "LUMUS_CORE_NO_BTC": {
        "VT": 0.25, "IEF": 0.10, "TLT": 0.12, "TIP": 0.10, "GLD": 0.10,
        "VNQ": 0.06, "DBC": 0.07, "SHY": 0.05,
    },
    # Real-estate sleeve audit: compare broad REIT exposure with S&P 500 real estate.
    "LUMUS_VNQ": {
        "VT": 0.25, "IEF": 0.10, "TLT": 0.12, "TIP": 0.10, "GLD": 0.10,
        "VNQ": 0.06, "DBC": 0.07, "SHY": 0.05, "BTC-USD": 0.05,
    },
    "LUMUS_XLRE": {
        "VT": 0.25, "IEF": 0.10, "TLT": 0.12, "TIP": 0.10, "GLD": 0.10,
        "XLRE": 0.06, "DBC": 0.07, "SHY": 0.05, "BTC-USD": 0.05,
    },
}

LUMUS_ASSETS = list(PORTFOLIOS["LUMUS_CORE_WITH_BTC"])
CORE_COMPARISON_PORTFOLIOS = ("SP500", "60_40", "DALIO_AW", "LUMUS_CORE_WITH_BTC", "LUMUS_CORE_NO_BTC")
REAL_ESTATE_VARIANTS = ("LUMUS_VNQ", "LUMUS_XLRE")
REAL_ESTATE_ASSETS = {name: list(PORTFOLIOS[name]) for name in REAL_ESTATE_VARIANTS}
REAL_ESTATE_AUDIT_ASSETS = list(dict.fromkeys(
    ticker for name in REAL_ESTATE_VARIANTS for ticker in REAL_ESTATE_ASSETS[name]
))

STRESS_PERIODS = {
    "GFC_2008": ("2008-01-01", "2008-12-31"),
    "COVID_2020": ("2020-01-01", "2020-12-31"),
    "INFLATION_2022": ("2022-01-01", "2022-12-31"),
}


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    start: pd.Timestamp
    weights: pd.Series


def normalized_weights(weights: Mapping[str, float]) -> pd.Series:
    """Validate and normalize portfolio weights."""
    series = pd.Series(weights, dtype=float)
    if series.empty or (series < 0).any() or series.sum() <= 0:
        raise ValueError("Weights must be non-negative and have a positive sum")
    return series / series.sum()


def _close_series(downloaded: pd.DataFrame | pd.Series, ticker: str) -> pd.Series:
    """Extract one adjusted Close series across yfinance output shapes."""
    if isinstance(downloaded, pd.Series):
        series = downloaded
    elif isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise ValueError("download did not include Close prices")
        close = downloaded["Close"]
        series = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    elif "Close" in downloaded.columns:
        series = downloaded["Close"]
    else:
        raise ValueError("download did not include Close prices")
    return pd.to_numeric(series, errors="coerce").rename(ticker).dropna()


def download_prices(
    tickers: list[str], start: str, end: str | None, cache_dir: Path,
    retries: int = 3, pause_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch tickers independently, retry failures, and fall back to local cache.

    yfinance ``auto_adjust=True`` is explicit: adjusted Close therefore includes
    splits and distributions. Cached CSV data is useful when a later request fails.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices: dict[str, pd.Series] = {}
    errors: list[str] = []
    for ticker in tickers:
        cache_file = cache_dir / f"{ticker.replace('-', '_')}.csv"
        series: pd.Series | None = None
        last_error = "empty response"
        for attempt in range(1, retries + 1):
            try:
                raw = yf.download(
                    ticker, start=start, end=end, auto_adjust=True,
                    progress=False, threads=False,
                )
                series = _close_series(raw, ticker)
                if series.empty:
                    raise ValueError("no adjusted Close rows returned")
                series.to_frame().to_csv(cache_file, index_label="Date")
                break
            except Exception as exc:  # network/provider failures vary by environment
                last_error = f"attempt {attempt}/{retries}: {exc}"
                if attempt < retries:
                    time.sleep(pause_seconds * attempt)
        if series is None or series.empty:
            if cache_file.exists():
                cached = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
                series = pd.to_numeric(cached.iloc[:, 0], errors="coerce").rename(ticker).dropna()
                print(f"WARNING: using cached prices for {ticker}; download failed ({last_error})")
            else:
                errors.append(f"{ticker}: {last_error}")
                continue
        series.index = pd.to_datetime(series.index).tz_localize(None)
        prices[ticker] = series[~series.index.duplicated(keep="last")].sort_index()
    if errors:
        raise RuntimeError("Could not obtain required prices:\n  " + "\n  ".join(errors))
    return pd.DataFrame(prices).sort_index()


def prepare_prices(raw_prices: pd.DataFrame, max_ffill_days: int = MAX_FORWARD_FILL_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill only short internal gaps and return a ticker-level coverage report.

    Leading/trailing gaps remain missing. A portfolio starts only after all of its
    assets have valid prices, avoiding backfilled pre-inception performance.
    """
    if raw_prices.empty:
        raise ValueError("No price data supplied")
    raw_prices = raw_prices.sort_index()
    coverage_rows = []
    cleaned: dict[str, pd.Series] = {}
    for ticker in raw_prices.columns:
        original = raw_prices[ticker].dropna()
        if original.empty:
            raise ValueError(f"{ticker} has no usable prices")
        internal = raw_prices[ticker].loc[original.index.min(): original.index.max()]
        filled = internal.ffill(limit=max_ffill_days)
        cleaned[ticker] = filled.reindex(raw_prices.index)
        coverage_rows.append({
            "Ticker": ticker,
            "FirstValidDate": original.index.min(),
            "LastValidDate": original.index.max(),
            "RawObservations": len(original),
            "FilledInternalGaps": int(filled.notna().sum() - internal.notna().sum()),
            "RemainingMissing": int(filled.isna().sum()),
        })
    return pd.DataFrame(cleaned), pd.DataFrame(coverage_rows).set_index("Ticker")


def backtest_portfolio(prices: pd.DataFrame, weights: Mapping[str, float]) -> BacktestResult:
    """Run a daily valuation backtest with target-weight rebalancing each month-end."""
    target = normalized_weights(weights)
    missing = target.index.difference(prices.columns)
    if not missing.empty:
        raise ValueError(f"Prices missing required tickers: {', '.join(missing)}")
    panel = prices[target.index].dropna(how="any")
    if len(panel) < 2:
        raise ValueError("Not enough common price observations for portfolio")
    asset_returns = panel.pct_change().dropna(how="any")
    value = 1.0
    current = target.copy()
    values = [(panel.index[0], value)]
    previous_month: pd.Period | None = None
    for date, row in asset_returns.iterrows():
        month = date.to_period("M")
        # Reset on the first observed trading day of a new month. This is
        # equivalent to rebalancing after the final observation of the prior month.
        if previous_month is not None and month != previous_month:
            current = target.copy()
        daily_return = float(current.dot(row))
        value *= 1.0 + daily_return
        values.append((date, value))
        current = current * (1.0 + row)
        current /= current.sum()
        previous_month = month
    equity = pd.Series(dict(values), name="Equity").sort_index()
    return BacktestResult(equity, equity.pct_change().dropna(), panel.index[0], target)


def drawdown_details(equity: pd.Series) -> dict[str, object]:
    """Return maximum drawdown dates and recovery duration (if recovered)."""
    drawdown = equity / equity.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    recovered = equity.loc[trough:][equity.loc[trough:] >= equity.loc[peak]]
    recovery = recovered.index[0] if not recovered.empty else pd.NaT
    return {
        "MaxDD": drawdown.loc[trough], "MaxDDPeakDate": peak, "MaxDDTroughDate": trough,
        "RecoveryDate": recovery,
        "PeakToTroughDays": (trough - peak).days,
        "RecoveryDays": (recovery - trough).days if pd.notna(recovery) else np.nan,
        "PeakToRecoveryDays": (recovery - peak).days if pd.notna(recovery) else np.nan,
    }


def metrics(result: BacktestResult) -> dict[str, object]:
    equity, daily = result.equity, result.returns
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    volatility = daily.std() * np.sqrt(TRADING_DAYS)
    downside = daily[daily < 0].std()
    dd = drawdown_details(equity)
    annual = equity.resample("YE").last().pct_change().dropna()
    return {
        "StartDate": equity.index[0], "EndDate": equity.index[-1], "Years": years,
        "CAGR": cagr, "Volatility": volatility,
        "Sharpe": daily.mean() / daily.std() * np.sqrt(TRADING_DAYS) if daily.std() else np.nan,
        "Sortino": daily.mean() / downside * np.sqrt(TRADING_DAYS) if downside else np.nan,
        **dd, "Calmar": cagr / abs(dd["MaxDD"]) if dd["MaxDD"] else np.nan,
        "WorstYear": annual.min() if not annual.empty else np.nan, "FinalValue": equity.iloc[-1],
    }


def annual_returns(equities: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Compute calendar-year returns, including each portfolio's partial first year."""
    columns: dict[str, pd.Series] = {}
    for name, equity in equities.items():
        year_end = equity.resample("YE").last()
        returns = year_end.pct_change(fill_method=None)
        # No prior year-end exists for the first (possibly partial) calendar year.
        returns.iloc[0] = year_end.iloc[0] / equity.iloc[0] - 1
        returns.index = returns.index.year
        columns[name] = returns
    return pd.DataFrame(columns).rename_axis("Year")


def stress_analysis(equities: Mapping[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for period, (start, end) in STRESS_PERIODS.items():
        for portfolio, equity in equities.items():
            sample = equity.loc[start:end]
            if len(sample) < 2:
                rows.append({"Period": period, "Portfolio": portfolio, "Available": False})
                continue
            dd = sample / sample.cummax() - 1
            rows.append({"Period": period, "Portfolio": portfolio, "Available": True,
                         "StartDate": sample.index[0], "EndDate": sample.index[-1],
                         "Return": sample.iloc[-1] / sample.iloc[0] - 1, "MaxDDWithinPeriod": dd.min()})
    return pd.DataFrame(rows).set_index(["Period", "Portfolio"])


def risk_contributions(prices: pd.DataFrame, portfolios: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Estimate annualized volatility contribution from full-history daily covariance."""
    rows = []
    for name, raw_weights in portfolios.items():
        weights = normalized_weights(raw_weights)
        sample = prices[weights.index].dropna().pct_change().dropna()
        covariance = sample.cov() * TRADING_DAYS
        marginal = covariance.dot(weights)
        volatility = float(np.sqrt(weights.dot(marginal)))
        components = weights * marginal / volatility if volatility else weights * np.nan
        for ticker in weights.index:
            rows.append({"Portfolio": name, "Ticker": ticker, "Weight": weights[ticker],
                         "VolatilityContribution": components[ticker],
                         "RiskContributionPct": components[ticker] / volatility if volatility else np.nan,
                         "SampleStart": sample.index.min(), "SampleEnd": sample.index.max()})
    return pd.DataFrame(rows).set_index(["Portfolio", "Ticker"])


def annual_asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return calendar-year adjusted-price returns for each L.U.M.U.S.-8 asset."""
    columns: dict[str, pd.Series] = {}
    for ticker in LUMUS_ASSETS:
        asset_prices = prices[ticker].dropna()
        year_end = asset_prices.resample("YE").last()
        returns = year_end.pct_change(fill_method=None)
        if not returns.empty:
            # Report the available partial first year rather than silently dropping it.
            returns.iloc[0] = year_end.iloc[0] / asset_prices.iloc[0] - 1
        returns.index = returns.index.year
        columns[ticker] = returns
    return pd.DataFrame(columns).rename_axis("Year")


def portfolio_daily_attribution(prices: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    """Attribute daily returns using the portfolio's actual drifting weights.

    Beginning-of-day weights drift with asset returns and reset to target weights
    on the first SPY trading day of each month, matching ``backtest_portfolio``.
    """
    target = normalized_weights(weights)
    panel = prices[target.index].dropna(how="any")
    asset_returns = panel.pct_change(fill_method=None).dropna(how="any")
    current = target.copy()
    previous_month: pd.Period | None = None
    rows = []
    for date, row in asset_returns.iterrows():
        month = date.to_period("M")
        if previous_month is not None and month != previous_month:
            current = target.copy()
        contribution = current * row
        rows.append({**contribution.to_dict(), "PortfolioReturn": contribution.sum()})
        current = current * (1.0 + row)
        current /= current.sum()
        previous_month = month
    return pd.DataFrame(rows, index=asset_returns.index).reindex(columns=[*target.index, "PortfolioReturn"])


def contribution_by_period(
    prices: pd.DataFrame, periods: Mapping[str, tuple[str, str]],
    portfolio_names: tuple[str, ...] = ("LUMUS_CORE_WITH_BTC", "LUMUS_CORE_NO_BTC"),
    output_assets: list[str] = LUMUS_ASSETS,
) -> pd.DataFrame:
    """Return exact additive period attribution for selected L.U.M.U.S.-8 variants.

    Daily return contributions are scaled by the portfolio's intra-period growth.
    Their sum therefore reconciles to each portfolio's compounded period return.
    """
    rows = []
    for portfolio in portfolio_names:
        target = normalized_weights(PORTFOLIOS[portfolio]).reindex(output_assets, fill_value=0.0)
        daily = portfolio_daily_attribution(prices, PORTFOLIOS[portfolio])
        for period, (start, end) in periods.items():
            sample = daily.loc[start:end]
            if sample.empty:
                for ticker in output_assets:
                    rows.append({"Period": period, "Portfolio": portfolio, "Ticker": ticker,
                                 "Available": False, "TargetWeight": target[ticker]})
                continue
            growth_before_day = (1.0 + sample["PortfolioReturn"]).cumprod().shift(fill_value=1.0)
            scaled = sample.drop(columns="PortfolioReturn").multiply(growth_before_day, axis="index")
            contributions = scaled.sum().reindex(output_assets, fill_value=0.0)
            portfolio_return = (1.0 + sample["PortfolioReturn"]).prod() - 1.0
            for ticker in output_assets:
                rows.append({"Period": period, "Portfolio": portfolio, "Ticker": ticker,
                             "Available": True, "StartDate": sample.index[0], "EndDate": sample.index[-1],
                             "TargetWeight": target[ticker], "ReturnContribution": contributions[ticker],
                             "ContributionShare": contributions[ticker] / portfolio_return if portfolio_return else np.nan,
                             "PortfolioReturn": portfolio_return})
    return pd.DataFrame(rows).set_index(["Period", "Portfolio", "Ticker"])


def correlation_analysis(
    prices: pd.DataFrame, output_dir: Path, cluster_count: int = CLUSTER_COUNT,
    assets: list[str] = LUMUS_ASSETS, file_stem: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save L.U.M.U.S.-8 return correlations and hierarchy-clustering artifacts.

    The condensed distance matrix is based directly on ``1 - correlation`` as an
    audit-friendly dissimilarity. Ward output is required for the phase-2 audit;
    average linkage is emitted as a useful companion view.
    """
    if not 1 <= cluster_count <= len(assets):
        raise ValueError(f"cluster_count must be between 1 and {len(assets)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = pd.Index(assets).difference(prices.columns)
    if not missing.empty:
        raise ValueError(f"Prices missing L.U.M.U.S.-8 assets: {', '.join(missing)}")
    sample = prices[assets].dropna(how="any").pct_change(fill_method=None).dropna(how="any")
    if len(sample) < 2:
        raise ValueError("Not enough common L.U.M.U.S.-8 observations for correlation analysis")
    correlation = sample.corr()
    if correlation.isna().any().any():
        raise ValueError("Cannot cluster assets with undefined return correlations")

    distance = (1.0 - correlation).clip(lower=0.0)
    np.fill_diagonal(distance.values, 0.0)
    condensed_distance = squareform(distance, checks=True)
    assignment_tables = []
    prefix = f"{file_stem}_" if file_stem else ""
    for method, filename in (("ward", f"{prefix}cluster_dendrogram.png"), ("average", f"{prefix}cluster_dendrogram_average.png")):
        linkage = hierarchy.linkage(condensed_distance, method=method)
        clusters = hierarchy.fcluster(linkage, t=cluster_count, criterion="maxclust")
        leaves = hierarchy.leaves_list(linkage)
        leaf_order = {assets[index]: order + 1 for order, index in enumerate(leaves)}
        assignment_tables.append(pd.DataFrame({
            "Ticker": assets, "Cluster": clusters, "LinkageMethod": method,
            "DendrogramLeafOrder": [leaf_order[ticker] for ticker in assets],
            "SampleStart": sample.index.min(), "SampleEnd": sample.index.max(),
            "Observations": len(sample),
        }))
        figure, axis = plt.subplots(figsize=(10, 6))
        hierarchy.dendrogram(linkage, labels=assets, leaf_rotation=45, ax=axis)
        axis.set_title(f"L.U.M.U.S.-8 {method.title()} Linkage of Daily Return Correlations")
        axis.set_ylabel("1 - correlation linkage distance")
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)
    assignments = pd.concat(assignment_tables).sort_values(["LinkageMethod", "Cluster", "DendrogramLeafOrder"])
    correlation.to_csv(output_dir / f"{prefix}correlation_matrix.csv")
    assignments.to_csv(output_dir / f"{prefix}cluster_results.csv", index=False)

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation)), labels=correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation)), labels=correlation.index)
    for row in range(len(correlation)):
        for column in range(len(correlation)):
            axis.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("L.U.M.U.S.-8 Daily Return Correlations")
    figure.colorbar(image, ax=axis, label="Correlation")
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}correlation_heatmap.png", dpi=160)
    plt.close(figure)
    return correlation, assignments


def save_plots(equities: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(13, 7))
    equities.plot(ax=plt.gca(), logy=True)
    plt.title("Portfolio Backtest: Growth of $1 (log scale)")
    plt.ylabel("Growth of $1")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves.png", dpi=160)
    plt.close()

    drawdowns = equities / equities.cummax() - 1
    drawdowns.plot(figsize=(13, 7))
    plt.title("Portfolio Drawdowns")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "drawdowns.png", dpi=160)
    plt.close()


def real_estate_variant_audit(
    prices: pd.DataFrame, results: Mapping[str, BacktestResult], output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build fair-period VNQ-versus-XLRE comparison reports and clustering charts."""
    common_start = max(results[name].start for name in REAL_ESTATE_VARIANTS)
    common_results = {
        name: backtest_portfolio(prices.loc[common_start:], PORTFOLIOS[name])
        for name in REAL_ESTATE_VARIANTS
    }
    tables = {
        "real_estate_variant_metrics": pd.DataFrame({
            name: metrics(result) for name, result in common_results.items()
        }).T,
        "real_estate_variant_risk_contributions": risk_contributions(
            prices.loc[common_start:], {name: PORTFOLIOS[name] for name in REAL_ESTATE_VARIANTS}
        ),
        "real_estate_variant_contribution_2022": contribution_by_period(
            prices, {"INFLATION_2022": STRESS_PERIODS["INFLATION_2022"]},
            REAL_ESTATE_VARIANTS, REAL_ESTATE_AUDIT_ASSETS,
        ),
    }
    cluster_tables = []
    for name in REAL_ESTATE_VARIANTS:
        stem = name.lower()
        _, clusters = correlation_analysis(
            prices.loc[common_start:], output_dir, assets=REAL_ESTATE_ASSETS[name], file_stem=stem,
        )
        clusters.insert(0, "Portfolio", name)
        cluster_tables.append(clusters)
    tables["real_estate_variant_cluster_results"] = pd.concat(cluster_tables, ignore_index=True)
    return tables


def run_backtest(raw_prices: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices, coverage = prepare_prices(raw_prices)
    # SPY sessions are the valuation calendar. BTC weekend moves are therefore
    # captured cumulatively on the next US trading day without adding zero-return
    # ETF weekend rows that would distort annualized statistics.
    prices = prices.reindex(raw_prices["SPY"].dropna().index)
    results = {name: backtest_portfolio(prices, weights) for name, weights in PORTFOLIOS.items()}
    equities = pd.DataFrame({name: result.equity for name, result in results.items()})
    tables = {
        "metrics": pd.DataFrame({name: metrics(result) for name, result in results.items()}).T,
        "annual_returns": annual_returns({name: result.equity for name, result in results.items()}),
        "stress_periods": stress_analysis({name: result.equity for name, result in results.items()}),
        "risk_contributions": risk_contributions(prices, PORTFOLIOS),
        "data_coverage": coverage,
        "equity_curves": equities,
        "annual_asset_returns": annual_asset_returns(prices),
        "contribution_by_stress_period": contribution_by_period(prices, STRESS_PERIODS),
        "contribution_2022": contribution_by_period(prices, {"INFLATION_2022": STRESS_PERIODS["INFLATION_2022"]}),
    }
    # Preserve the original benchmark comparison at the BTC portfolio inception.
    # The VNQ-versus-XLRE audit has its own fair common-period metrics table.
    common_start = max(results[name].start for name in CORE_COMPARISON_PORTFOLIOS)
    common_results = {
        name: backtest_portfolio(prices.loc[common_start:], weights)
        for name, weights in PORTFOLIOS.items()
    }
    tables["common_period_metrics"] = pd.DataFrame(
        {name: metrics(result) for name, result in common_results.items()}
    ).T
    tables.update(real_estate_variant_audit(prices, results, output_dir))
    correlation, clusters = correlation_analysis(prices, output_dir)
    tables["correlation_matrix"] = correlation
    tables["cluster_results"] = clusters
    for filename, table in tables.items():
        if filename not in {"correlation_matrix", "cluster_results"}:
            table.to_csv(output_dir / f"{filename}.csv")
    save_plots(equities, output_dir)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=None, help="exclusive end date accepted by yfinance")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    tickers = sorted({ticker for weights in PORTFOLIOS.values() for ticker in weights})
    raw_prices = download_prices(tickers, args.start, args.end, args.output_dir / "cache")
    tables = run_backtest(raw_prices, args.output_dir)
    print("\n=== Backtest metrics ===")
    print(tables["metrics"].to_string())
    print("\n=== Annual returns ===")
    print((tables["annual_returns"] * 100).round(2).to_string())
    print(f"\nCSV files and plots saved under: {args.output_dir.resolve()}")
    for artifact in sorted(path.name for path in args.output_dir.iterdir() if path.is_file()):
        print(f"  - {artifact}")


if __name__ == "__main__":
    main()
