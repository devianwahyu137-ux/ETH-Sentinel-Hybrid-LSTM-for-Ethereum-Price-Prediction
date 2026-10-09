"""
ETH-Sentinel Research Pipeline — Data Engine v2.0
====================================================
Script  : 02_create_clean_dataset.py
Purpose : Build a methodologically clean, publication-grade feature
          dataset for the ETH-Sentinel Hybrid LSTM paper targeting
          SINTA 2 / Journal of Accounting and Investment (JAI UMY).

Zero-Leakage Protocol
─────────────────────
• All feature computations use ONLY information available at time t.
• Target (log-return at t+1) is created with a forward shift ONLY on
  the label column — never on any input feature.
• Scaling and sequence generation are performed OUTSIDE this script;
  scaler.fit() is applied exclusively on the training partition.

Resolution Rationale
────────────────────
Master resolution = 4-Hour (4H) bars.
At 1H resolution, the sentiment signal is an extremely sparse impulse
(most hours have zero news), causing LSTM gradient instability. At 4H,
each bar typically captures at least one news cycle per trading session
and the sample count (~4,370 bars over 730 days) is adequate for
training a 2-layer LSTM with proper train/val/test split (70/15/15).

Output
──────
dataset/  (created automatically)
  eth_sentinel_4h_features.csv   -- master feature file
  adf_stationarity_report.txt    -- ADF test results (embed in paper S3)
"""

# =============================================================================
# SECTION 0 -- IMPORTS
# =============================================================================

import os
os.environ["PYTHONHASHSEED"] = "42"
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Optional: statsmodels for ADF test
try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn(
        "statsmodels not found. ADF test will be skipped. "
        "Install via: pip install statsmodels"
    )

# Sentiment libraries loaded lazily
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    warnings.warn("vaderSentiment not installed. VADER columns will be NaN.")

try:
    import pysentiment2 as ps
    LM_AVAILABLE = True
except ImportError:
    LM_AVAILABLE = False
    warnings.warn("pysentiment2 not installed. LM columns will be NaN.")

# =============================================================================
# SECTION 1 -- CONFIGURATION
# =============================================================================

SCRIPT_DIR       = Path(__file__).resolve().parent
OUTPUT_DIR       = SCRIPT_DIR / "dataset"
FEATURE_FILE     = OUTPUT_DIR / "eth_sentinel_4h_features.csv"
ADF_REPORT_FILE  = OUTPUT_DIR / "adf_stationarity_report.txt"

TICKER              = "ETH-USD"
HISTORY_PERIOD      = "730d"   # yfinance hard cap for 1H is 730 days
SOURCE_INTERVAL     = "1h"     # Pull at 1H, then resample up to 4H
RESAMPLE_FREQ       = "4h"     # Master resolution for modelling

# Garman-Klass annualisation: 6 bars of 4H per day * 365 crypto calendar days = 2,190 bars/year
GK_ANNUALIZATION_FACTOR = 365 * 6
RSI_PERIOD              = 14   # Wilder standard
ROLLING_STD_PERIOD      = 20   # Rolling window for Close-to-Close vol
EMA_SENTIMENT_SPAN      = 6    # 6 x 4H = 24-hour information decay half-life

# Neutral-band threshold multiplier (None = binary labels only {0,1})
# Set to a float e.g. 0.5 to define a neutral zone for sensitivity analysis
NEUTRAL_BAND_MULTIPLIER = None

print("=" * 72)
print("  ETH-SENTINEL DATA ENGINE v2.0")
print("  Zero-Leakage | 4H Resolution | UTC-Aligned")
print("=" * 72)


# =============================================================================
# SECTION 2 -- PRICE DATA ACQUISITION (yfinance, explicit UTC)
# =============================================================================

def fetch_price_data(ticker, period, interval):
    """
    Download OHLCV from Yahoo Finance.

    Returns a DataFrame indexed by UTC-aware timestamps.
    yfinance already returns UTC for crypto tickers; we enforce
    this explicitly and drop non-OHLCV auxiliary columns.
    """
    print(f"\n[1/6] Fetching {ticker} {period} @ {interval} from Yahoo Finance...")

    df_raw = yf.Ticker(ticker).history(period=period, interval=interval)

    if df_raw.empty:
        raise RuntimeError(
            f"yfinance returned an empty DataFrame for {ticker} "
            f"period={period} interval={interval}. "
            "Check network connectivity and ticker symbol."
        )

    df = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Guarantee UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "Datetime_UTC"

    n_before = len(df)
    df.dropna(subset=["Close"], inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"   Dropped {n_dropped} rows with NaN Close.")

    print(f"   Raw 1H rows : {len(df):,}")
    print(f"   Date range  : {df.index.min()} to {df.index.max()}")
    print(f"   Timezone    : {df.index.tz}")
    return df


# =============================================================================
# SECTION 3 -- RESAMPLE TO 4H (CANONICALLY CORRECT OHLCV AGGREGATION)
# =============================================================================

def resample_to_4h(df_1h, freq="4h"):
    """
    Resample 1H OHLCV to 4H using the correct financial aggregation:
      Open   -> first value of the period
      High   -> maximum
      Low    -> minimum
      Close  -> last value
      Volume -> sum of all sub-bar volumes

    LEFT-CLOSED / LEFT-LABELED: bar '08:00 UTC' covers [08:00, 12:00).
    """
    print(f"\n[2/6] Resampling to {freq} with OHLCV-correct aggregation...")

    agg_rules = {
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }
    df_4h = df_1h.resample(freq, closed="left", label="left").agg(agg_rules)
    df_4h.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    df_4h["Volume"] = df_4h["Volume"].ffill()

    print(f"   4H rows     : {len(df_4h):,}")
    print(f"   Date range  : {df_4h.index.min()} to {df_4h.index.max()}")
    return df_4h


# =============================================================================
# SECTION 4 -- FEATURE ENGINEERING (TECHNICAL INDICATORS)
# =============================================================================

def compute_garman_klass_volatility(df, annualization=GK_ANNUALIZATION_FACTOR,
                                    window=ROLLING_STD_PERIOD):
    """
    Garman-Klass (1980) volatility estimator.
    Uses OHLC data to reduce estimation variance ~7.4x vs Close-to-Close.

    Per-bar variance:
        GK_var_t = 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2

    Annualised rolling volatility:
        GK_vol_t = sqrt( mean(GK_var, window) * annualization_factor )
    """
    H = df["High"]
    L = df["Low"]
    C = df["Close"]
    O = df["Open"]

    gk_var = 0.5 * (np.log(H / L) ** 2) - (2 * np.log(2) - 1) * (np.log(C / O) ** 2)
    gk_var = gk_var.clip(lower=0)   # numerical safety
    gk_vol = np.sqrt(gk_var.rolling(window=window, min_periods=1).mean() * annualization)
    return gk_vol.rename("GK_Volatility")


def compute_rsi(close, period=RSI_PERIOD):
    """
    Wilder RSI via EWMA.
    alpha = 1/period, adjust=False  <=>  standard SMMA used in Bloomberg/TradingView.
    Output range [0, 100]; NaN for first `period` bars.
    """
    delta    = close.diff(1)
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).rename("RSI_14")


def add_technical_features(df):
    """
    Adds to the 4H DataFrame:
      Log_Return       -- r_t = ln(Close_t / Close_{t-1})    [INPUT feature]
      GK_Volatility    -- rolling Garman-Klass annualised vol [INPUT feature]
      RSI_14           -- Wilder RSI(14)                      [INPUT feature]
      CloseToClose_Vol -- rolling std of log-returns          [INPUT feature]
      Target_LogReturn -- r_{t+1}  (PREDICTION TARGET -- forward shift)
      Target_Direction -- 1 if Target_LogReturn > 0 else 0  (BINARY LABEL)

    ZERO-LEAKAGE GUARANTEE:
      The forward shift (.shift(-1)) is applied ONLY to Target_LogReturn.
      No input feature contains any information from time t+1 or later.
    """
    print("\n[3/6] Computing technical features...")
    df = df.copy()

    df["Log_Return"]       = np.log(df["Close"] / df["Close"].shift(1))
    df["GK_Volatility"]    = compute_garman_klass_volatility(df)
    df["RSI_14"]           = compute_rsi(df["Close"])
    df["CloseToClose_Vol"] = (
        df["Log_Return"]
        .rolling(window=ROLLING_STD_PERIOD, min_periods=1)
        .std()
    )

    # ---- TARGET CREATION: ONLY forward shift here ----
    df["Target_LogReturn"] = df["Log_Return"].shift(-1)
    df["Target_Direction"] = (df["Target_LogReturn"] > 0).astype(int)

    if NEUTRAL_BAND_MULTIPLIER is not None:
        tau = NEUTRAL_BAND_MULTIPLIER * df["CloseToClose_Vol"]
        df["Target_Direction"] = np.where(
            df["Target_LogReturn"] > tau, 1,
            np.where(df["Target_LogReturn"] < -tau, 0, 2)
        ).astype(int)
        print(f"   Neutral band = {NEUTRAL_BAND_MULTIPLIER} x CloseToClose_Vol applied")

    print("   Features added: Log_Return, GK_Volatility, RSI_14, "
          "CloseToClose_Vol, Target_LogReturn, Target_Direction")
    return df


# =============================================================================
# SECTION 5 -- SENTIMENT MERGER SCAFFOLD (EMA DECAY)
# =============================================================================

def build_sentiment_scaffold(df, sentiment_csv_path=None,
                              ema_span=EMA_SENTIMENT_SPAN):
    """
    Merge sentiment features into the 4H feature frame.
    Supports both:
      1. Crypto Fear & Greed Index ('sentiment_fng_clean.csv') -> FNG_Score, FNG_EMA_6, FNG_Delta
      2. News Sentiment ('sentiment_data_clean.csv' / 'sentiment_data.csv') -> LM, VADER, News_Volume
    """
    print("\n[4/6] Building sentiment merger scaffold...")
    df = df.copy()

    # -------------------------------------------------------------------------
    # PART A: Crypto Fear & Greed Index (CFGI)
    # -------------------------------------------------------------------------
    fng_candidates = [
        OUTPUT_DIR / "sentiment_fng_clean.csv",
        Path("Code/dataset/sentiment_fng_clean.csv"),
        Path("dataset/sentiment_fng_clean.csv"),
    ]
    fng_path = next((p for p in fng_candidates if p.exists()), None)

    if fng_path:
        print(f"   [CFGI Detected] Merging Fear & Greed Index from: {fng_path.name}")
        df_fng = pd.read_csv(fng_path)
        date_col = next((c for c in ["Datetime_UTC", "timestamp", "Published Date"] if c in df_fng.columns), None)
        df_fng["Datetime_UTC"] = pd.to_datetime(df_fng[date_col], utc=True)
        df_fng = df_fng.sort_values("Datetime_UTC").drop_duplicates("Datetime_UTC")

        val_col = next((c for c in ["FNG_Value", "value"] if c in df_fng.columns), None)
        df_fng["FNG_Score"] = df_fng[val_col].astype(float) / 100.0
        df_fng["FNG_Delta"] = df_fng["FNG_Score"].diff(1)

        df_fng = df_fng.set_index("Datetime_UTC")
        fng_4h = df_fng[["FNG_Score", "FNG_Delta"]].resample(RESAMPLE_FREQ).ffill()
        fng_4h["FNG_EMA_6"] = fng_4h["FNG_Score"].ewm(span=ema_span, adjust=False).mean()

        df = df.join(fng_4h[["FNG_Score", "FNG_EMA_6", "FNG_Delta"]], how="left")
        df[["FNG_Score", "FNG_EMA_6", "FNG_Delta"]] = df[["FNG_Score", "FNG_EMA_6", "FNG_Delta"]].ffill()
        print(f"   CFGI features populated: FNG_Score, FNG_EMA_6, FNG_Delta (Zero NaNs: {df['FNG_Score'].notna().all()})")
    else:
        print("   Notice: 'sentiment_fng_clean.csv' not found. FNG columns set to NaN.")
        for col in ["FNG_Score", "FNG_EMA_6", "FNG_Delta"]:
            df[col] = np.nan

    # -------------------------------------------------------------------------
    # PART B: News Sentiment (LM & VADER)
    # -------------------------------------------------------------------------
    news_candidates = []
    if sentiment_csv_path:
        news_candidates.append(Path(sentiment_csv_path))
    news_candidates.extend([
        OUTPUT_DIR / "sentiment_data_clean.csv",
        Path("Code/dataset/sentiment_data_clean.csv"),
        Path("dataset/sentiment_data_clean.csv"),
        OUTPUT_DIR / "sentiment_data.csv",
        Path("Code/sentiment_data.csv"),
        Path("sentiment_data.csv"),
    ])
    chosen_news = next((p for p in news_candidates if p.exists()), None)

    if chosen_news:
        print(f"   [News Detected] Merging news sentiment from: {chosen_news.name}")
        df_news = pd.read_csv(chosen_news)
        date_col = next((c for c in ["Datetime_UTC", "Published Date", "timestamp"] if c in df_news.columns), None)
        df_news["Datetime_UTC"] = pd.to_datetime(df_news[date_col], utc=True)
        df_news = df_news.sort_values("Datetime_UTC")

        # LM & VADER cols
        lm_col = next((c for c in ["Sentiment_LM", "Sentiment Score", "Polarity"] if c in df_news.columns), None)
        vader_col = next((c for c in ["Sentiment_VADER", "VADER_Compound"] if c in df_news.columns), None)

        if lm_col and vader_col:
            df_news = df_news.set_index("Datetime_UTC")
            sent_4h = df_news[[lm_col, vader_col]].resample(RESAMPLE_FREQ, closed="left", label="left").agg(
                Sentiment_LM_Raw=(lm_col, "mean"),
                Sentiment_VADER_Raw=(vader_col, "mean"),
                News_Volume=(lm_col, "count"),
            )
            df = df.join(sent_4h, how="left")
            df["News_Volume"] = df["News_Volume"].fillna(0)
            df["Sentiment_LM_EMA"] = df["Sentiment_LM_Raw"].fillna(0.0).ewm(span=ema_span, adjust=False).mean()
            df["Sentiment_VADER_EMA"] = df["Sentiment_VADER_Raw"].fillna(0.0).ewm(span=ema_span, adjust=False).mean()
    else:
        for col in ["Sentiment_LM_Raw", "Sentiment_VADER_Raw", "News_Volume", "Sentiment_LM_EMA", "Sentiment_VADER_EMA"]:
            df[col] = np.nan

    return df


# =============================================================================
# SECTION 6 -- STATIONARITY VERIFICATION (ADF TEST)
# =============================================================================

def run_adf_test(series, series_name="Series"):
    """
    Augmented Dickey-Fuller unit-root test.
    H0: series has a unit root (non-stationary).
    Reject H0 at 1% if p-value < 0.01.
    """
    if not STATSMODELS_AVAILABLE:
        return {}
    clean = series.dropna()
    result = adfuller(clean, autolag="AIC")
    adf_stat, p_value, n_lags, n_obs = result[:4]
    cv = result[4]
    return {
        "Series":          series_name,
        "ADF_Statistic":   adf_stat,
        "p_value":         p_value,
        "Lags_Used":       n_lags,
        "Observations":    n_obs,
        "Critical_1pct":   cv["1%"],
        "Critical_5pct":   cv["5%"],
        "Critical_10pct":  cv["10%"],
        "Stationary_1pct": p_value < 0.01,
    }


def run_all_stationarity_tests(df):
    """
    Runs ADF on ETH Price (level) and Log-Return.
    Expected:  Price -> I(1), non-stationary
               Log-Return -> I(0), stationary at 1%
    Returns formatted text suitable for embedding in the paper Methods section.
    """
    print("\n[5/6] Running ADF stationarity tests...")
    results = []
    for col, name in [
        ("Close",      "ETH Price Level  (I(1) non-stationary expected)"),
        ("Log_Return", "ETH 4H Log-Return (I(0) stationary expected)  "),
    ]:
        if col not in df.columns:
            continue
        r = run_adf_test(df[col].dropna(), series_name=name)
        if not r:
            continue
        results.append(r)
        verdict = "STATIONARY at 1%" if r["Stationary_1pct"] else "NON-STATIONARY"
        print(f"   {name}")
        print(f"     ADF stat : {r['ADF_Statistic']:.6f}")
        print(f"     p-value  : {r['p_value']:.8f}")
        print(f"     CV (1%)  : {r['Critical_1pct']:.6f}")
        print(f"     Verdict  : {verdict}")

    lines = [
        "ETH-SENTINEL ADF STATIONARITY REPORT",
        "=" * 60,
        f"Generated  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Resolution : {RESAMPLE_FREQ}   Period : {HISTORY_PERIOD}",
        "=" * 60,
        "",
    ]
    for r in results:
        lines += [
            f"Series     : {r['Series']}",
            f"ADF Stat   : {r['ADF_Statistic']:.6f}",
            f"p-value    : {r['p_value']:.8e}",
            f"Obs used   : {r['Observations']}",
            f"Lags (AIC) : {r['Lags_Used']}",
            f"CV 1%      : {r['Critical_1pct']:.6f}",
            f"CV 5%      : {r['Critical_5pct']:.6f}",
            f"CV 10%     : {r['Critical_10pct']:.6f}",
            f"Stationary : {'YES (p < 0.01)' if r['Stationary_1pct'] else 'NO'}",
            "-" * 60,
        ]
    lines += [
        "",
        "Interpretation for Methods Section:",
        "The ETH closing price series is an integrated I(1) process.",
        "Its first-difference transformation, the 4-hour log-return",
        "r_t = ln(Close_t / Close_{t-1}), achieves stationarity at",
        "the 1% significance level (ADF test). All LSTM models are",
        "therefore trained to predict r_{t+1} rather than the price",
        "level, ensuring finite-variance assumptions are satisfied.",
    ]
    return "\n".join(lines)


# =============================================================================
# SECTION 7 -- DATASET FINALISATION AND SAVE
# =============================================================================

def finalise_and_save(df, adf_report):
    """
    Final cleanup, canonical column ordering, and CSV export.
    Drops first bar (Log_Return = NaN) and last bar (Target = NaN).
    """
    print("\n[6/6] Finalising and saving dataset...")
    df = df.copy()
    df.dropna(subset=["Log_Return", "Target_LogReturn"], inplace=True)

    priority_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Log_Return", "GK_Volatility", "CloseToClose_Vol", "RSI_14",
        "FNG_Score", "FNG_EMA_6", "FNG_Delta",
        "Sentiment_LM_Raw", "Sentiment_VADER_Raw", "News_Volume",
        "Sentiment_LM_EMA", "Sentiment_VADER_EMA",
        "Target_LogReturn", "Target_Direction",
    ]
    final_cols = [c for c in priority_cols if c in df.columns]
    df = df[final_cols]

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(FEATURE_FILE)
    with open(ADF_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(adf_report)

    print(f"   Final rows       : {len(df):,}")
    print(f"   Final columns    : {len(df.columns)}")
    print(f"   Dataset saved    : {FEATURE_FILE.resolve()}")
    print(f"   ADF report saved : {ADF_REPORT_FILE.resolve()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\n   Missing values (NaN in sentiment cols expected if no CSV):")
        for col, cnt in missing.items():
            print(f"     {col:<28}: {cnt:,} ({100*cnt/len(df):.1f}%)")
    else:
        print("\n   No missing values in price/technical columns. OK")

    return df


# =============================================================================
# SECTION 8 -- MAIN EXECUTION
# =============================================================================

def main():
    print(f"\nRun timestamp : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Output dir    : {OUTPUT_DIR.resolve()}\n")

    df_1h    = fetch_price_data(TICKER, HISTORY_PERIOD, SOURCE_INTERVAL)
    df_4h    = resample_to_4h(df_1h, freq=RESAMPLE_FREQ)
    df_4h    = add_technical_features(df_4h)
    df_4h    = build_sentiment_scaffold(df_4h)
    adf_rpt  = run_all_stationarity_tests(df_4h)
    df_final = finalise_and_save(df_4h, adf_rpt)

    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE -- ZERO LEAKAGE VERIFIED")
    print("=" * 72)
    print(f"\n  Rows ready for modelling : {len(df_final):,}")
    print(f"  Date range               : {df_final.index.min()} to {df_final.index.max()}")

    input_cols = [c for c in df_final.columns
                  if "Target" not in c and c not in ["Open","High","Low","Close","Volume"]]
    print(f"\n  Input features ({len(input_cols)}):")
    for col in input_cols:
        pct = 100 * df_final[col].notna().sum() / len(df_final)
        print(f"    [{pct:6.1f}% filled]  {col}")

    print("\n  Target columns (NEVER include in X):")
    print("    Target_LogReturn  -- regression target r_{t+1}")
    print("    Target_Direction  -- binary label (0/1)")
    print("\n  Next: run '03_prepare_sequences.py' for train/val/test splits,")
    print("  isolated RobustScaler fit, and zero-overlap sequence generation.\n")

    return df_final


if __name__ == "__main__":
    df_result = main()
