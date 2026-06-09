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
    "LUMUS_GROWTH_SPY_QQQ_50_50": {"SPY": 0.50, "QQQ": 0.50},
    # Formal ex-Alpha baseline: replace the 15% Alpha Engine growth sleeve with VT.
    "LUMUS_EX_ALPHA_VT_REPLACED": {
        "VT": 0.30, "BNDX": 0.075, "TLT": 0.125, "TIP": 0.10, "GLD": 0.10,
        "XLRE": 0.10, "DBC": 0.05, "SHY": 0.12, "BTC-USD": 0.03,
    },
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
    PORTFOLIOS: dict[str, dict[str, float]] = {
    "SP500": {"SPY": 1.0},
    "60_40": {"SPY": 0.60, "IEF": 0.40},
    "DALIO_AW": {...},

    "LUMUS_EX_ALPHA_VT_REPLACED": {...},
    "LUMUS_GROWTH_SPY_QQQ_50_50": {...},

}
}

FORMAL_BASELINE = "LUMUS_EX_ALPHA_VT_REPLACED"
LEGACY_SAMPLE_PORTFOLIOS = ("LUMUS_CORE_WITH_BTC", "LUMUS_XLRE")
LUMUS_ASSETS = list(PORTFOLIOS["LUMUS_CORE_WITH_BTC"])
CORE_COMPARISON_PORTFOLIOS = ("SP500", "60_40", "DALIO_AW", FORMAL_BASELINE, *LEGACY_SAMPLE_PORTFOLIOS)
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

    Leading and long trailing gaps remain missing. A portfolio starts only after all of its
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
        # Include the trailing calendar so a short provider gap after the latest
        # observation is handled consistently with other short internal gaps.
        internal = raw_prices[ticker].loc[original.index.min():]
        filled = internal.ffill(limit=max_ffill_days)
        cleaned[ticker] = filled.reindex(raw_prices.index)
        coverage_rows.append({
            "Ticker": ticker,
            "FirstValidDate": original.index.min(),
            "LastValidDate": original.index.max(),
            "RawObservations": len(original),
            "FilledInternalGaps": int(filled.loc[:original.index.max()].notna().sum() - internal.loc[:original.index.max()].notna().sum()),
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

    distance = (1.0 - correlation).clip(lower=0.0).copy()
    distance_values = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_values, 0.0)
    distance = pd.DataFrame(distance_values, index=distance.index, columns=distance.columns)
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


def _metric_summary(table: pd.DataFrame, name: str) -> str:
    row = table.loc[name]
    return (
        f"CAGR {row['CAGR']:.2%}、Volatility {row['Volatility']:.2%}、"
        f"MaxDD {row['MaxDD']:.2%}、Sharpe {row['Sharpe']:.2f}、Calmar {row['Calmar']:.2f}"
    )


def write_ex_alpha_report(tables: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    """Write the research report for the formal ex-Alpha / VT-replaced baseline."""
    common = tables["common_period_metrics"]
    stress = tables["stress_periods"]
    annual = tables["annual_returns"]
    start = pd.Timestamp(common.loc[FORMAL_BASELINE, "StartDate"]).date()
    end = pd.Timestamp(common.loc[FORMAL_BASELINE, "EndDate"]).date()
    comparison_lines = [
        f"- **{name}**: {_metric_summary(common, name)}"
        for name in (FORMAL_BASELINE, "SP500", "60_40", "DALIO_AW")
    ]
    year_2022 = annual.loc[2022] if 2022 in annual.index else pd.Series(dtype=float)
    year_lines = [f"- **{name}**: {year_2022[name]:.2%}" for name in CORE_COMPARISON_PORTFOLIOS if name in year_2022]
    inflation = stress.loc["INFLATION_2022"] if "INFLATION_2022" in stress.index else pd.DataFrame()
    stress_lines = []
    for name in (FORMAL_BASELINE, "60_40", "DALIO_AW", "SP500"):
        if name in inflation.index and bool(inflation.loc[name, "Available"]):
            row = inflation.loc[name]
            stress_lines.append(
                f"- **{name}**: 期間リターン {row['Return']:.2%}、期間内MaxDD {row['MaxDDWithinPeriod']:.2%}"
            )
    formal = common.loc[FORMAL_BASELINE]
    sp500 = common.loc["SP500"]
    assessment = (
        "同一期間では、正式基準はSP500より低いボラティリティとMaxDDを示した。"
        if formal["Volatility"] < sp500["Volatility"] and abs(formal["MaxDD"]) < abs(sp500["MaxDD"])
        else "同一期間では、正式基準がSP500より低いボラティリティとMaxDDを同時には示さなかった。"
    )
    report = f"""# L.U.M.U.S.-8 ex-Alpha / VT置換版 バックテスト・レポート

## 位置づけと検証目的

`{FORMAL_BASELINE}` は **L.U.M.U.S.-8完成版ではなく、Alpha Engine 15%をVTで置換した基準ポートフォリオ** である。Alpha Engineを完全削除してCore 85%を100%へ正規化しない理由は、Alpha Engineが「本来VTで持ってもよい成長枠15%の代替戦略」として設計されているためである。

本検証の目的は、Alpha Engineなしでもインデックス級のリターンに近づきつつ、明らかに低いボラティリティと最大ドローダウン（MDD）を維持できるかを研究することである。これは投資助言ではなく、ポートフォリオ設計検証用のバックテストである。

## 方法と共通比較期間

- 月次で目標比率へリバランスし、調整後終値を利用した。
- 現金代替はSHY、金はGLD、不動産はXLRE、暗号資産はBTC-USD 3%とした。
- 全戦略の公平比較期間は、正式基準に必要なBTC-USD、BNDX、XLREを含む全資産が利用可能な **{start}〜{end}** とした。
- `metrics.csv` は各戦略固有の利用可能期間、`common_period_metrics.csv` は上記共通期間を示す。
- 無リスク金利、税、売買コスト、スリッページは考慮していない。

## 同一期間の主要評価

{chr(10).join(comparison_lines)}

{assessment} CAGR、Volatility、MaxDD、Sharpe、Calmarの組み合わせから、リターン接近度と防御力のトレードオフを評価すべきである。VT単体はポートフォリオ構成資産として含まれるが、比較対象として指定されたSP500を株式インデックスの代表ベンチマークに用いた。

## 2022年（株債同時下落局面）

### 年次リターン
{chr(10).join(year_lines) if year_lines else '- 2022年データなし'}

### 期間内成績
{chr(10).join(stress_lines) if stress_lines else '- 2022年データなし'}

60/40およびDalio All Weatherとの比較では、年次リターンだけでなく期間内MaxDDも併せて防御力を判定する。

## 旧サンプル比率との違い

- `{FORMAL_BASELINE}` が今回の正式な評価対象で、比率合計は100%。VT 30%、BNDX 7.5%、XLRE 10%、BTC-USD 3%など、ex-Alpha / VT置換設計を直接表す。
- `LUMUS_CORE_WITH_BTC` と `LUMUS_XLRE` は **旧サンプル比率**。元の比率合計90%をバックテスト時に正規化し、BTC 5%や旧来の債券・実物資産配分を含むため、正式基準ではない。

## 解釈上の注意

ETFの設定時期により共通期間は限定され、過去の成績は将来を保証しない。BTCを含むため価格データ品質とリバランス仮定にも結果は左右される。詳細値はCSV成果物を参照すること。
"""
    (output_dir / "lumus_ex_alpha_vt_replaced_report.md").write_text(report, encoding="utf-8")


def concise_assessment(common: pd.DataFrame) -> str:
    formal, sp500 = common.loc[FORMAL_BASELINE], common.loc["SP500"]
    return (
        f"{FORMAL_BASELINE}: CAGR {formal['CAGR']:.2%} vs SP500 {sp500['CAGR']:.2%}; "
        f"Volatility {formal['Volatility']:.2%} vs {sp500['Volatility']:.2%}; "
        f"MaxDD {formal['MaxDD']:.2%} vs {sp500['MaxDD']:.2%}. "
        "Alpha EngineをVTで置換した研究用基準として、リターン接近度と防御力を評価してください。"
    )


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
        "portfolio_period_annual_returns": annual_returns({name: result.equity for name, result in results.items()}),
        "portfolio_period_stress_periods": stress_analysis({name: result.equity for name, result in results.items()}),
        "risk_contributions": risk_contributions(prices, PORTFOLIOS),
        "data_coverage": coverage,
        "equity_curves": equities,
        "drawdowns": equities / equities.cummax() - 1.0,
        "annual_asset_returns": annual_asset_returns(prices),
        "contribution_by_stress_period": contribution_by_period(prices, STRESS_PERIODS),
        "contribution_2022": contribution_by_period(prices, {"INFLATION_2022": STRESS_PERIODS["INFLATION_2022"]}),
    }
    # Fix every strategy to the formal BTC-enabled baseline inception. This may be
    # later than BTC inception because BNDX and XLRE must also be available.
    common_start = results[FORMAL_BASELINE].start
    common_results = {
        name: backtest_portfolio(prices.loc[common_start:], weights)
        for name, weights in PORTFOLIOS.items()
    }
    tables["common_period_metrics"] = pd.DataFrame(
        {name: metrics(result) for name, result in common_results.items()}
    ).T
    common_equities = {name: result.equity for name, result in common_results.items()}
    tables["annual_returns"] = annual_returns(common_equities)
    common_stress = stress_analysis(common_equities)
    tables["stress_periods"] = common_stress
    tables["comparison_2022"] = common_stress.loc["INFLATION_2022"].copy()
    tables.update(real_estate_variant_audit(prices, results, output_dir))
    correlation, clusters = correlation_analysis(prices, output_dir)
    tables["correlation_matrix"] = correlation
    tables["cluster_results"] = clusters
    for filename, table in tables.items():
        if filename not in {"correlation_matrix", "cluster_results"}:
            table.to_csv(output_dir / f"{filename}.csv")
    save_plots(equities, output_dir)
    write_ex_alpha_report(tables, output_dir)
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
    print("\n=== 1. Backtest metrics table ===")
    print(tables["metrics"].to_string())
    print("\n=== 2. Common-period metrics table ===")
    print(tables["common_period_metrics"].to_string())
    print("\n=== 3. Annual returns table (%) ===")
    print((tables["annual_returns"] * 100).round(2).to_string())
    print("\n=== 4. 2022 comparison table ===")
    print(tables["comparison_2022"].to_string())
    print(f"\n=== 5. Artifacts: {args.output_dir.resolve()} ===")
    for artifact in sorted(path.name for path in args.output_dir.iterdir() if path.is_file()):
        print(f"  - {artifact}")
    print("\n=== 6. Concise assessment ===")
    print(concise_assessment(tables["common_period_metrics"]))


if __name__ == "__main__":
    main()
