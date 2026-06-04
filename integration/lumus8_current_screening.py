# 日米株６００銘柄モメンタムスクリーニング Feat. Project A.U.R.A

import csv
import importlib.util
import io
import warnings
from datetime import datetime, timedelta
from pathlib import Path

if importlib.util.find_spec("numpy") is not None:
    import numpy as np
else:
    np = None

if importlib.util.find_spec("pandas") is not None:
    import pandas as pd
else:
    pd = None

if importlib.util.find_spec("requests") is not None:
    import requests
else:
    requests = None

if importlib.util.find_spec("yfinance") is not None:
    import yfinance as yf
else:
    yf = None

warnings.simplefilter(action="ignore", category=FutureWarning)


# ==========================================================
# 1. データ取得：広域ユニバース (Robust Edition)
# ==========================================================
def get_tickers_lumus(universe_mode="current", rebalance_date=None):
    """Return US/JP universes, optionally using point-in-time S&P 500 membership.

    ``universe_mode="current"`` preserves the existing survivor-only behavior.
    ``universe_mode="historical"`` replaces only the US universe by calling
    the local historical membership engine for ``rebalance_date``.  The Japan
    universe intentionally remains the existing manual list, so JP survivor-only
    risk is still present in any report using this helper.
    """
    print("🌌 L.U.M.U.S. ユニバース（S&P500 + 日本株精鋭）を構築中...")
    if universe_mode not in {"current", "historical"}:
        raise ValueError('universe_mode must be "current" or "historical"')

    us_tickers = []
    if universe_mode == "current":
        # Plan A: Wikipedia
        try:
            sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
            us_tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
        except Exception:
            pass

        # Plan B: GitHub CSV
        if len(us_tickers) < 100:
            try:
                url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
                s = requests.get(url).content
                df_csv = pd.read_csv(io.StringIO(s.decode("utf-8")))
                us_tickers = df_csv["Symbol"].tolist()
            except Exception:
                pass

        # Plan C: Local processed seed (CI/offline backup)
        if len(us_tickers) < 100:
            try:
                seed_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "base_constituents.csv"
                with seed_path.open(newline="", encoding="utf-8") as handle:
                    us_tickers = [row.get("normalized_ticker") or row.get("ticker") for row in csv.DictReader(handle)]
                    us_tickers = [ticker for ticker in us_tickers if ticker]
            except Exception:
                pass

        # Plan D: Static List (Backup)
        if len(us_tickers) < 100:
            us_tickers = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "LLY", "JPM", "V", "WMT", "XOM", "CAT", "COST"]

    # 日本株（新リスト）
    jp_tickers = [
        "7203.T", "6758.T", "8306.T", "8035.T", "9984.T", "9432.T", "6861.T", "6098.T",
        "4063.T", "6954.T", "7974.T", "6301.T", "4568.T", "6501.T", "7741.T", "7267.T",
        "6273.T", "4543.T", "8058.T", "8001.T", "8031.T", "8053.T", "8002.T", "8316.T",
        "8411.T", "8766.T", "8801.T", "8802.T", "8591.T", "8725.T", "8750.T",
        "6857.T", "6146.T", "6723.T", "6920.T", "7735.T", "6981.T", "6503.T", "6702.T",
        "6752.T", "6506.T", "6965.T", "7729.T", "6869.T", "6971.T", "6315.T", "4062.T", "7701.T",
        "7011.T", "7012.T", "7013.T", "6367.T", "6113.T", "6481.T", "1801.T", "1802.T", "1803.T",
        "1812.T", "1925.T", "1928.T", "1808.T", "1721.T", "5803.T", "5802.T",
        "7201.T", "7269.T", "7270.T", "5401.T", "5713.T", "1605.T", "5020.T", "9101.T",
        "9104.T", "9107.T", "3407.T", "4188.T", "4452.T", "4911.T", "4183.T",
        "9983.T", "3382.T", "7453.T", "3092.T", "4661.T", "4385.T", "2413.T", "4689.T",
        "4755.T", "9735.T", "3659.T", "4307.T", "3088.T", "3064.T", "2802.T", "2502.T",
        "2503.T", "4502.T", "4519.T", "4503.T", "4523.T", "9020.T", "9021.T", "9022.T",
        "9201.T", "9202.T", "9501.T", "9502.T", "9503.T"
    ]
    if universe_mode == "historical":
        if rebalance_date is None:
            raise ValueError('rebalance_date is required when universe_mode="historical"')
        from lumus_historical_universe.reconstruct import get_sp500_members

        us_tickers = get_sp500_members(str(rebalance_date))

    return us_tickers, jp_tickers


# ==========================================================
# 2. 市場レジーム判定 (防御システム)
# ==========================================================
def check_regime():
    if yf is None:
        return {"US": "BULL", "JP": "BULL"}
    indices = {"US": "^GSPC", "JP": "^N225"}
    regimes = {}
    print("📡 市場天気図（200日線）を観測中...")
    try:
        data = yf.download(list(indices.values()), period="2y", progress=False)["Close"].ffill()
        for region, ticker in indices.items():
            if ticker in data.columns:
                series = data[ticker].dropna()
                current = series.iloc[-1]
                ma200 = series.rolling(200).mean().iloc[-1]
                regimes[region] = "BULL" if current > ma200 else "BEAR"
            else:
                regimes[region] = "UNKNOWN"
    except Exception:
        regimes = {"US": "BULL", "JP": "BULL"}
    return regimes


# ==========================================================
# 3. コアロジック: 3因子モデル + マルチ期間 + 高速化
# ==========================================================
def analyze_lumus_engine(tickers, region, regime):
    if yf is None or pd is None or np is None:
        return pd.DataFrame() if pd is not None else None
    print(f"🔍 {region}市場: {len(tickers)} 銘柄を解析中...")
    end = datetime.today()
    start = end - timedelta(days=400)  # 1年以上確保

    try:
        # 価格データ取得
        data = yf.download(tickers, start=start, end=end, progress=False)["Close"].ffill()
        daily_ret = data.pct_change(fill_method=None)
    except Exception:
        return pd.DataFrame()

    metrics = {}
    for t in data.columns:
        if data[t].count() < 250:
            continue
        series = data[t].dropna()
        d_r = daily_ret[t].dropna()

        # --- A. Efficiency (マルチ期間モメンタム / ボラティリティ) ---
        p_now = series.iloc[-1]
        p_12m = series.iloc[-252] if len(series) >= 252 else series.iloc[0]
        p_6m = series.iloc[-126] if len(series) >= 126 else series.iloc[0]
        p_3m = series.iloc[-63] if len(series) >= 63 else series.iloc[0]

        r_12m, r_6m, r_3m = (p_now / p_12m) - 1, (p_now / p_6m) - 1, (p_now / p_3m) - 1
        # 合成リターン (12M重視)
        composite_ret = (r_12m * 3 + r_6m * 2 + r_3m * 1) / 6

        vol = d_r.std() * np.sqrt(252)
        efficiency = composite_ret / vol if vol > 0 else 0

        # --- B. Quality (非対称性: 上がりやすく下がりにくい) ---
        avg_pos = d_r[d_r > 0].mean()
        avg_neg = abs(d_r[d_r < 0].mean())
        quality = avg_pos / avg_neg if avg_neg > 0 else 1.0

        # --- C. Valuation (簡易PBR代替: 高値からの距離 & ボラティリティ逆数) ---
        # ※API高速化のため、厳密なPBRの代わりに「過熱感のなさ」をバリュー代替とする
        # (ボラが低く、極端な急騰をしていないものを割安とみなす実務的アプローチ)
        max_52w = series.max()
        prox_high = p_now / max_52w
        valuation_score = (1 / prox_high) * (1 / vol)  # 高値から少し調整していて、ボラが低いものを評価

        metrics[t] = {
            "Efficiency": efficiency,
            "Quality": quality,
            "Valuation_Alt": valuation_score,
            "Volatility": vol,
            "Composite_Ret": composite_ret,
        }

    df = pd.DataFrame(metrics).T
    if df.empty:
        return pd.DataFrame()

    # --- スコアリング (3因子モデル) ---
    def z(s):
        return (s - s.mean()) / s.std()

    # 初代の哲学: 40:40:20
    df["Total_Score"] = z(df["Efficiency"]) * 0.4 + z(df["Quality"]) * 0.4 + z(df["Valuation_Alt"]) * 0.2

    # レジームフィルター (弱気相場なら厳格化)
    if regime == "BEAR":
        df["Total_Score"] -= 3.0
        print(f"⚠️ {region}市場は弱気です。選定基準を引き上げました。")

    return df.sort_values("Total_Score", ascending=False)


# ==========================================================
# 4. ポートフォリオ構築 (Risk Parity)
# ==========================================================
def build_lumus_portfolio(df_us, df_jp):
    print("\n" + "=" * 80)
    print("🏰 L.U.M.U.S.-8 Alpha Portfolio (Risk Parity & 3-Factor)")
    print("=" * 80)

    top_us = df_us.head(6)
    top_jp = df_jp.head(6)
    portfolio = pd.concat([top_us, top_jp])

    # リスクパリティ・ウェイト
    inv_vol = 1 / portfolio["Volatility"]
    weights = inv_vol / inv_vol.sum()
    portfolio["Weight"] = (weights * 100).round(1)

    cols = ["Weight", "Total_Score", "Composite_Ret", "Volatility", "Efficiency", "Quality"]
    print(portfolio[cols].sort_values("Weight", ascending=False))
    return portfolio


# ==========================================================
# 5. レジーム連動型ポジションサイズ決定 (Final Safety Valve)
# ==========================================================
def determine_exposure(regimes):
    print("\n" + "=" * 60)
    print("🛡️ ポートフォリオ稼働率 (Market Exposure)")
    print("=" * 60)

    us_status = regimes.get("US", "UNKNOWN")
    jp_status = regimes.get("JP", "UNKNOWN")

    # ロジック: 日米の「天気」に応じて株式組入比率を変える
    if us_status == "BULL" and jp_status == "BULL":
        exposure = 1.0  # 100% 株式
        msg = "🌞 快晴 (Full Throttle): 株式 100% / 現金 0%"
    elif us_status == "BULL" or jp_status == "BULL":
        exposure = 0.6  # 60% 株式 (40%現金)
        msg = "⛅ 曇り (Caution): 株式 60% / 現金 40% (分散投資)"
    else:
        exposure = 0.2  # 20% 株式 (80%現金 - 防衛モード)
        msg = "⛈️ 嵐 (Defense Mode): 株式 20% / 現金 80% (シェルター退避)"

    print(f"市場環境: US={us_status} | JP={jp_status}")
    print(f"👉 推奨稼働率: {msg}")
    return exposure


# ==========================================================
# 6. L.U.M.U.S.-8 Fusion Command Center (統合司令部)
# ==========================================================
def bridge_aura_to_amedas(dominant_anchor, gri_score):
    """マクロ判定結果から、スキャンすべきティッカー群を決定する関数"""
    target_universe = {"US_Stocks": [], "JP_Stocks": []}

    if dominant_anchor == "DBC":
        # XBMエラー修正済（素材ETF: XMEに変更）
        target_universe["US_Stocks"] = ["XLE", "XME", "GLD", "GDX", "CVX"]
        target_universe["JP_Stocks"] = ["1605.T", "8031.T", "8058.T", "5713.T"]
    elif dominant_anchor == "BTC-USD" or (dominant_anchor == "SPY" and gri_score > 0):
        target_universe["US_Stocks"] = ["QQQ", "SMH", "WULF", "MSTR", "NVDA"]
        target_universe["JP_Stocks"] = ["8035.T", "9984.T", "6857.T"]
    elif dominant_anchor == "UUP":
        target_universe["US_Stocks"] = ["XLV", "XLP", "JNJ", "PG"]
        target_universe["JP_Stocks"] = ["4502.T", "9432.T", "2502.T"]
    elif dominant_anchor == "TLT":
        target_universe["US_Stocks"] = ["XLU", "TLT", "IEF", "O"]
        target_universe["JP_Stocks"] = ["9501.T", "9502.T", "8801.T"]

    return target_universe


def get_amedas_signals(tickers):
    """AMeDASのテクニカル判定を行い、シグナルを抽出する内部関数"""
    signals = {}
    if yf is None:
        return signals
    if not tickers:
        return signals
    try:
        data = yf.download(tickers, period="1y", progress=False)["Close"].ffill()
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])

        for ticker in tickers:
            if ticker not in data.columns or len(data[ticker].dropna()) < 200:
                continue
            close = data[ticker].dropna()

            current_price = close.iloc[-1]
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1]

            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rs = gain / loss if loss != 0 else 0
            rsi14 = 100 - (100 / (1 + rs))

            is_po = (sma20 > sma50) and (sma50 > sma200)
            if is_po:
                if rsi14 > 75:
                    signals[ticker] = "🔴 買われすぎ"
                elif rsi14 < 45:
                    signals[ticker] = "🟢 押し目買い"
                else:
                    signals[ticker] = "🔥 順張りGO"
            elif current_price > sma200 and current_price > sma50 and sma20 > sma200:
                signals[ticker] = "🟡 打診買い"
    except Exception:
        pass
    return signals


def execute_lumus_fusion(df_core, amedas_us, amedas_jp):
    """現行の3因子ポートフォリオと、A.U.R.A.マクロ戦術を融合する関数"""
    print("\n" + "★" * 70)
    print(" 🌌 L.U.M.U.S.-8 FUSION COMMAND CENTER (完全統合ポートフォリオ) 🌌")
    print("★" * 70)

    all_amedas_tickers = amedas_us + amedas_jp
    amedas_results = get_amedas_signals(all_amedas_tickers)

    buy_signals = ["🔥 順張りGO", "🟢 押し目買い", "🟡 打診買い"]
    tactical_buys = {k: v for k, v in amedas_results.items() if v in buy_signals}

    fusion_records = []
    core_volatility_mean = df_core["Volatility"].mean() if not df_core.empty else 0.2

    # 既存コア銘柄の評価（オーバーラップ判定）
    for ticker in df_core.index:
        base_weight = df_core.loc[ticker, "Weight"]
        volatility = df_core.loc[ticker, "Volatility"]

        if ticker in tactical_buys:
            category = "① 👑 OVERLAP (Conviction)"
            signal = tactical_buys[ticker]
            raw_weight = base_weight * 1.5  # 確信度ブースト
        else:
            category = "③ 🛡️ CORE (Stability)"
            signal = "💎 3因子維持"
            raw_weight = base_weight

        fusion_records.append({"Ticker": ticker, "Category": category, "Signal": signal, "Volatility": volatility, "Raw_Weight": raw_weight})

    # A.U.R.A.専用銘柄の追加（戦術的サテライト）
    for ticker, signal in tactical_buys.items():
        if ticker not in df_core.index:
            category = "② ⚔️ TACTICAL (Theme)"
            try:
                temp_data = yf.download(ticker, period="1y", progress=False)["Close"]
                vol = temp_data.pct_change().std() * np.sqrt(252)
                vol = vol.iloc[0] if isinstance(vol, pd.Series) else vol
            except Exception:
                vol = core_volatility_mean

            # リスクパリティ準拠のディスカウントウェイト
            pseudo_weight = (1 / vol) / (1 / core_volatility_mean) * (100 / len(df_core)) * 0.7
            fusion_records.append({"Ticker": ticker, "Category": category, "Signal": signal, "Volatility": vol, "Raw_Weight": pseudo_weight})

    if not fusion_records:
        print("フュージョン可能な銘柄がありません。現行ポートフォリオを維持します。")
        return df_core

    # 最終ウェイトの正規化
    df_fusion = pd.DataFrame(fusion_records).set_index("Ticker")
    total_raw_weight = df_fusion["Raw_Weight"].sum()
    df_fusion["Final_Weight(%)"] = (df_fusion["Raw_Weight"] / total_raw_weight * 100).round(1)

    df_fusion = df_fusion.sort_values(by=["Category", "Final_Weight(%)"], ascending=[True, False])
    display_cols = ["Category", "Signal", "Final_Weight(%)", "Volatility"]

    print(df_fusion[display_cols])

    overlap_count = len(df_fusion[df_fusion["Category"].str.contains("OVERLAP")])
    print("\n[ 戦略的サマリー ]")
    print(f"👑 コンビクション銘柄 (両システム合致): {overlap_count} 銘柄")
    print(f"⚔️ テーマ戦術銘柄 (A.U.R.A.追加): {len(tactical_buys) - overlap_count} 銘柄")
    print(f"🛡️ コア安定銘柄 (現行維持): {len(df_core) - overlap_count} 銘柄")

    return df_fusion


# ==========================================================
# 7. L.U.M.U.S.-8 Auto Order Generator (自動発注リスト生成)
# ==========================================================
def generate_trade_orders(df_fusion, total_budget_jpy):
    if yf is None:
        print("yfinance is not installed; order generation skipped.")
        return None
    print("\n" + "=" * 70)
    print(" 🛒 L.U.M.U.S.-8 自動オーダー生成 (発注ロット計算)")
    print("=" * 70)

    # 1. 最新のドル円為替レートを取得
    try:
        usdjpy = yf.download("JPY=X", period="1d", progress=False)["Close"].iloc[-1]
        if isinstance(usdjpy, pd.Series):
            usdjpy = usdjpy.iloc[0]
        print(f"🔄 適用為替レート: 1 USD = {usdjpy:.2f} JPY\n")
    except Exception:
        usdjpy = 150.0  # 取得失敗時のフェイルセーフ
        print(f"⚠️ 為替取得失敗。仮レート(1 USD = 150 JPY)を適用します。\n")

    orders = []
    tickers = df_fusion.index.tolist()

    # 2. 最新の株価を取得 (Robust Edition)
    try:
        # 期間を5日間に延ばし、ffill()で休場日のNaNを前営業日の終値で埋める
        prices = yf.download(tickers, period="5d", progress=False)["Close"].ffill().iloc[-1]
    except Exception as e:
        print(f"価格データの取得に失敗しました: {e}")
        return None

    for ticker in tickers:
        alloc_jpy = df_fusion.loc[ticker, "Allocation_Amt(JPY)"]
        if alloc_jpy <= 0:
            continue

        try:
            price_local = prices[ticker]
            if isinstance(price_local, pd.Series):
                price_local = price_local.iloc[0]
            if pd.isna(price_local):
                continue

            # 日本株と米国株で通貨を分岐
            if str(ticker).endswith(".T"):
                price_jpy = price_local
                currency = "JPY"
            else:
                price_jpy = price_local * usdjpy
                currency = "USD"

            # 3. 買付株数の計算 (1株単位での端株購入を想定し、小数点以下切り捨て)
            shares = int(alloc_jpy // price_jpy)
            actual_cost_jpy = shares * price_jpy

            orders.append({
                "Ticker": ticker,
                "Cat": df_fusion.loc[ticker, "Category"][:4],  # 視認性のため先頭アイコンのみ
                "Price_Local": f"{price_local:>7.2f} {currency}",
                "Shares(株)": shares,
                "Target_Amt(¥)": int(alloc_jpy),
                "Actual_Cost(¥)": int(actual_cost_jpy),
            })
        except Exception:
            pass

    # 結果の表示
    df_orders = pd.DataFrame(orders).set_index("Ticker")

    # 株数が0になってしまったもの（単価が高すぎて予算オーバー）は除外して表示
    df_executable = df_orders[df_orders["Shares(株)"] > 0]
    print(df_executable[["Cat", "Price_Local", "Shares(株)", "Target_Amt(¥)", "Actual_Cost(¥)"]])

    total_actual = df_executable["Actual_Cost(¥)"].sum()
    cash_remainder = total_budget_jpy - total_actual

    print("\n" + "-" * 70)
    print(f"✅ 発注予定総額: {total_actual:,.0f} 円")
    print(f"📦 残存キャッシュ (端数調整後): {cash_remainder:,.0f} 円")
    print("-" * 70)

    return df_executable


# ==========================================================
# 実行セクション
# ==========================================================
def main():
    regimes = check_regime()
    us_list, jp_list = get_tickers_lumus()
    df_us = analyze_lumus_engine(us_list, "US", regimes["US"])
    df_jp = analyze_lumus_engine(jp_list, "JP", regimes["JP"])
    final_port = build_lumus_portfolio(df_us, df_jp)
    print("\n✅ 完了。これが三体戦略における『泥に染まらない蓮』の候補です。")

    # final_port が計算された後に実行
    exposure_ratio = determine_exposure(regimes)

    # 最終的な推奨購入額（例：予算100万円の場合）
    print("\n💰 推奨アロケーション (例: 投資資金100万円)")
    final_port["Allocation_Amt"] = final_port["Weight"] * 10000 * exposure_ratio  # 万円単位
    display_cols = ["Weight", "Allocation_Amt", "Total_Score", "Efficiency"]
    print(final_port[display_cols].sort_values("Weight", ascending=False))

    print(f"\n📦 現金ポジション(CASH): {1000000 * (1 - exposure_ratio):,.0f} 円")

    # A.U.R.A.の最新出力結果を手動で定義
    current_anchor = "UUP"
    current_gri = 0.002

    # ブリッジ関数で監視対象を取得
    target_universe = bridge_aura_to_amedas(current_anchor, current_gri)

    # フュージョン実行（final_port は現行コードで生成済みの変数）
    df_ultimate = execute_lumus_fusion(final_port, target_universe["US_Stocks"], target_universe["JP_Stocks"])

    # 最終的な投資金額の計算 (予算と市場稼働率を適用)
    # 例: exposure_ratio は現行コードで計算済みの稼働率（0.2, 0.6, 1.0）
    total_budget = 4500000 * exposure_ratio  # 100万円をベースに稼働率を掛ける
    df_ultimate["Allocation_Amt(JPY)"] = (df_ultimate["Final_Weight(%)"] / 100 * total_budget).round(0)

    print("\n💰 【最終】フュージョン・アロケーション")
    print(df_ultimate[["Category", "Final_Weight(%)", "Allocation_Amt(JPY)"]])

    # total_budget は直前のコードで計算した 1000000 * exposure_ratio を使用
    generate_trade_orders(df_ultimate, total_budget)


if __name__ == "__main__":
    main()
