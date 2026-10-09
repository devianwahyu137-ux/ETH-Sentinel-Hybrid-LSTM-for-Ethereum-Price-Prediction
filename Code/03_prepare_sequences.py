"""
ETH-Sentinel Research Pipeline — Sequence Preparation Engine v2.5
===================================================================
Script  : 03_prepare_sequences.py
Purpose : Partition clean 4H ETH features chronologically (70/15/15),
          apply isolated RobustScaler (train-only fit), and construct
          strictly isolated lookback sequences (W = 18 bars / 72 hours).
          Supports both:
          1. Technical-Only Sequences (eth_technical_sequences.npz)
          2. Dual-Stream Hybrid Sequences (eth_hybrid_sequences.npz):
             - X_price: ['Log_Return', 'GK_Volatility', 'CloseToClose_Vol', 'RSI_14']
             - X_sent : ['Sentiment_LM_EMA', 'Sentiment_VADER_EMA', 'News_Volume']
Author  : ETH-Sentinel Research Team
Target  : Journal of Accounting and Investment (JAI - SINTA 2)
===================================================================
"""

from __future__ import annotations

import json
import os
os.environ["PYTHONHASHSEED"] = "42"
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# =============================================================================
# GLOBAL CONFIGURATION & CONSTANTS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
INPUT_CSV = DATASET_DIR / "eth_sentinel_4h_features.csv"

OUTPUT_TECHNICAL_NPZ = DATASET_DIR / "eth_technical_sequences.npz"
OUTPUT_HYBRID_NPZ    = DATASET_DIR / "eth_hybrid_sequences.npz"
OUTPUT_FNG_HYBRID_NPZ= DATASET_DIR / "eth_fng_hybrid_sequences.npz"

OUTPUT_PRICE_SCALER = DATASET_DIR / "eth_technical_scaler.pkl"
OUTPUT_SENT_SCALER  = DATASET_DIR / "eth_fng_sentiment_scaler.pkl"
OUTPUT_META         = DATASET_DIR / "eth_technical_metadata.json"

# Lookback window: 18 bars of 4H = 72 hours (3 days of market context)
LOOKBACK_WINDOW = 18

# Chronological partition ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Modality 1: Price Dynamics Features
PRICE_FEATURE_COLS = [
    "Log_Return",
    "GK_Volatility",
    "CloseToClose_Vol",
    "RSI_14",
]

# Modality 2: Crypto Fear & Greed Index (CFGI) Sentiment Features
SENT_FEATURE_COLS = [
    "FNG_Score",
    "FNG_EMA_6",
    "FNG_Delta",
]

TARGET_DIR_COL = "Target_Direction"
TARGET_RET_COL = "Target_LogReturn"


# =============================================================================
# SECTION 1 -- DATA LOADING & PRE-VALIDATION
# =============================================================================
def load_and_validate_data(
    csv_path: Path,
    price_cols: list[str],
    sent_cols: list[str],
    target_cols: list[str],
) -> Tuple[pd.DataFrame, bool]:
    """
    Loads dataset, enforces UTC datetime index, sorts chronologically,
    and drops initial warmup rows containing NaNs in technical indicators.
    Detects whether sentiment columns are populated or placeholders.
    """
    print(f"\n[1/5] Loading and validating dataset from '{csv_path.name}'...")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Please run 'Code/02_create_clean_dataset.py' first."
        )

    df = pd.read_csv(csv_path, index_col="Datetime_UTC", parse_dates=True)
    initial_rows = len(df)
    print(f"   Raw loaded rows       : {initial_rows:,}")

    # Verify chronological order
    if not df.index.is_monotonic_increasing:
        print("   Notice: Sorting index to guarantee chronological order.")
        df = df.sort_index()

    # Drop duplicates if any
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        print(f"   Warning: Removing {duplicates} duplicate timestamps.")
        df = df[~df.index.duplicated(keep="first")]

    # Check price & target columns
    required_core = price_cols + target_cols
    missing_core = [c for c in required_core if c not in df.columns]
    if missing_core:
        raise KeyError(f"Missing core columns in CSV: {missing_core}")

    # Check sentiment status
    has_sent_cols = all(c in df.columns for c in sent_cols)
    sentiment_active = False
    if has_sent_cols:
        sent_non_nan = df[sent_cols].notna().all(axis=1).sum()
        if sent_non_nan > 0.5 * len(df):
            sentiment_active = True
            print(f"   Sentiment Modality    : [ACTIVE] ({sent_non_nan:,} bars populated)")
        else:
            print("   Sentiment Modality    : [INACTIVE] (contains NaN placeholders)")
    else:
        print("   Sentiment Modality    : [NOT FOUND in CSV]")

    # Drop warmup rows (e.g. RSI_14 requires 14 bars warmup)
    eval_cols = required_core + (sent_cols if sentiment_active else [])
    clean_df = df.dropna(subset=eval_cols).copy()
    dropped_rows = initial_rows - len(clean_df)
    print(f"   Warmup rows dropped   : {dropped_rows} (due to indicator warmup)")
    print(f"   Clean usable rows     : {len(clean_df):,}")
    print(f"   Date span             : {clean_df.index.min()} -> {clean_df.index.max()}")

    return clean_df, sentiment_active


# =============================================================================
# SECTION 2 -- CHRONOLOGICAL PARTITIONING (70 / 15 / 15)
# =============================================================================
def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the DataFrame chronologically with zero forward shuffle.
    """
    print(f"\n[2/5] Performing strict chronological partitioning ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})...")
    total_samples = len(df)

    n_train = int(total_samples * train_ratio)
    n_val   = int(total_samples * val_ratio)
    n_test  = total_samples - n_train - n_val

    df_train = df.iloc[:n_train].copy()
    df_val   = df.iloc[n_train : n_train + n_val].copy()
    df_test  = df.iloc[n_train + n_val :].copy()

    # Zero-leakage chronological assertions
    assert df_train.index.max() < df_val.index.min(), "Leakage Error: Train overlaps with Val!"
    assert df_val.index.max() < df_test.index.min(), "Leakage Error: Val overlaps with Test!"
    assert len(df_train) + len(df_val) + len(df_test) == total_samples, "Partition count mismatch!"

    def summarize_split(name: str, split_df: pd.DataFrame):
        dir_counts = split_df[TARGET_DIR_COL].value_counts(normalize=True).to_dict()
        up_pct = dir_counts.get(1, 0.0) * 100.0
        down_pct = dir_counts.get(0, 0.0) * 100.0
        print(f"   {name:<12}: {len(split_df):>5} bars ({len(split_df)/total_samples:>6.2%}) | "
              f"{split_df.index.min()} to {split_df.index.max()} | "
              f"UP: {up_pct:.2f}% | DOWN: {down_pct:.2f}%")

    summarize_split("Train (70%)", df_train)
    summarize_split("Val   (15%)", df_val)
    summarize_split("Test  (15%)", df_test)

    return df_train, df_val, df_test


# =============================================================================
# SECTION 3 -- ISOLATED ROBUST SCALING (TRAIN-ONLY FIT)
# =============================================================================
def fit_and_scale_modality(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list[str],
    scaler_save_path: Path,
    modality_name: str = "Price",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """
    Fits RobustScaler EXCLUSIVELY on training features for a given modality.
    Transforms Validation and Test partitions using the frozen parameters.
    """
    print(f"\n[3/5] Fitting isolated RobustScaler for {modality_name} Modality (Train-only fit)...")

    scaler = RobustScaler()
    scaler.fit(df_train[cols])

    X_train_scaled = scaler.transform(df_train[cols]).astype(np.float32)
    X_val_scaled   = scaler.transform(df_val[cols]).astype(np.float32)
    X_test_scaled  = scaler.transform(df_test[cols]).astype(np.float32)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"   Saved fitted scaler -> {scaler_save_path.name}")

    print(f"   {modality_name} Scaling Parameters:")
    for col, center, scale in zip(cols, scaler.center_, scaler.scale_):
        print(f"     * {col:<20} : Median={center:>10.6f}, IQR={scale:>10.6f}")

    for name, arr in [("Train", X_train_scaled), ("Val", X_val_scaled), ("Test", X_test_scaled)]:
        assert not np.isnan(arr).any(), f"NaN detected in {name} scaled array!"
        assert not np.isinf(arr).any(), f"Inf detected in {name} scaled array!"

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# =============================================================================
# SECTION 4 -- SEQUENCE WINDOW GENERATOR (STRICT PARTITION ISOLATION)
# =============================================================================
def generate_isolated_sequences(
    X: np.ndarray,
    y_dir: np.ndarray,
    y_ret: np.ndarray,
    timestamps: pd.Index,
    window_size: int = 18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates 3D tensor sequences strictly isolated within a single partition:
      Input X_seq  : shape (N - W + 1, W, num_features)
      Target y_dir : shape (N - W + 1,) -- binary direction at end of window
      Target y_ret : shape (N - W + 1,) -- continuous return at end of window
      Dates seq    : shape (N - W + 1,) -- ISO-8601 timestamps of target prediction
    """
    n_samples = len(X) - window_size + 1
    if n_samples <= 0:
        raise ValueError(f"Partition length ({len(X)}) is smaller than lookback window ({window_size})!")

    num_features = X.shape[1]
    X_seq = np.empty((n_samples, window_size, num_features), dtype=np.float32)
    y_dir_seq = np.empty((n_samples,), dtype=np.int32)
    y_ret_seq = np.empty((n_samples,), dtype=np.float32)
    dates_seq = np.empty((n_samples,), dtype="<U32")

    for i in range(n_samples):
        X_seq[i] = X[i : i + window_size]
        target_idx = i + window_size - 1
        y_dir_seq[i] = y_dir[target_idx]
        y_ret_seq[i] = y_ret[target_idx]
        dates_seq[i] = str(timestamps[target_idx])

    return X_seq, y_dir_seq, y_ret_seq, dates_seq


# =============================================================================
# SECTION 5 -- MAIN ORCHESTRATION PIPELINE
# =============================================================================
def main():
    print("=" * 80)
    print("ETH-SENTINEL: SEQUENCE PREPARATION PIPELINE (HYBRID & TECHNICAL)")
    print("Zero-Leakage Protocol | W=18 (72H) | Isolated RobustScalers | 70/15/15")
    print("=" * 80)

    # 1. Load data
    df_clean, sentiment_active = load_and_validate_data(
        csv_path=INPUT_CSV,
        price_cols=PRICE_FEATURE_COLS,
        sent_cols=SENT_FEATURE_COLS,
        target_cols=[TARGET_DIR_COL, TARGET_RET_COL],
    )

    # 2. Chronological split
    df_train, df_val, df_test = chronological_split(
        df_clean,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    # 3. Isolated scaling: Price Modality
    X_train_price_s, X_val_price_s, X_test_price_s, price_scaler = fit_and_scale_modality(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        cols=PRICE_FEATURE_COLS,
        scaler_save_path=OUTPUT_PRICE_SCALER,
        modality_name="Price",
    )

    # 4. Generate Price Sequences
    print(f"\n[4/5] Constructing isolated sequence windows (W = {LOOKBACK_WINDOW} bars)...")
    X_train_price, y_train_dir, y_train_ret, dates_train = generate_isolated_sequences(
        X=X_train_price_s,
        y_dir=df_train[TARGET_DIR_COL].to_numpy(dtype=np.int32),
        y_ret=df_train[TARGET_RET_COL].to_numpy(dtype=np.float32),
        timestamps=df_train.index,
        window_size=LOOKBACK_WINDOW,
    )
    X_val_price, y_val_dir, y_val_ret, dates_val = generate_isolated_sequences(
        X=X_val_price_s,
        y_dir=df_val[TARGET_DIR_COL].to_numpy(dtype=np.int32),
        y_ret=df_val[TARGET_RET_COL].to_numpy(dtype=np.float32),
        timestamps=df_val.index,
        window_size=LOOKBACK_WINDOW,
    )
    X_test_price, y_test_dir, y_test_ret, dates_test = generate_isolated_sequences(
        X=X_test_price_s,
        y_dir=df_test[TARGET_DIR_COL].to_numpy(dtype=np.int32),
        y_ret=df_test[TARGET_RET_COL].to_numpy(dtype=np.float32),
        timestamps=df_test.index,
        window_size=LOOKBACK_WINDOW,
    )

    # Always save Technical-Only Sequences
    np.savez_compressed(
        OUTPUT_TECHNICAL_NPZ,
        X_train=X_train_price,
        y_train_dir=y_train_dir,
        y_train_ret=y_train_ret,
        X_val=X_val_price,
        y_val_dir=y_val_dir,
        y_val_ret=y_val_ret,
        X_test=X_test_price,
        y_test_dir=y_test_dir,
        y_test_ret=y_test_ret,
        dates_train=dates_train,
        dates_val=dates_val,
        dates_test=dates_test,
        feature_names=np.array(PRICE_FEATURE_COLS),
    )
    print(f"   [OK] Saved technical sequences -> {OUTPUT_TECHNICAL_NPZ.name}")

    # 5. Handle Sentiment Modality if Active
    if sentiment_active:
        X_train_sent_s, X_val_sent_s, X_test_sent_s, sent_scaler = fit_and_scale_modality(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            cols=SENT_FEATURE_COLS,
            scaler_save_path=OUTPUT_SENT_SCALER,
            modality_name="Sentiment",
        )

        X_train_sent, _, _, _ = generate_isolated_sequences(
            X=X_train_sent_s,
            y_dir=df_train[TARGET_DIR_COL].to_numpy(dtype=np.int32),
            y_ret=df_train[TARGET_RET_COL].to_numpy(dtype=np.float32),
            timestamps=df_train.index,
            window_size=LOOKBACK_WINDOW,
        )
        X_val_sent, _, _, _ = generate_isolated_sequences(
            X=X_val_sent_s,
            y_dir=df_val[TARGET_DIR_COL].to_numpy(dtype=np.int32),
            y_ret=df_val[TARGET_RET_COL].to_numpy(dtype=np.float32),
            timestamps=df_val.index,
            window_size=LOOKBACK_WINDOW,
        )
        X_test_sent, _, _, _ = generate_isolated_sequences(
            X=X_test_sent_s,
            y_dir=df_test[TARGET_DIR_COL].to_numpy(dtype=np.int32),
            y_ret=df_test[TARGET_RET_COL].to_numpy(dtype=np.float32),
            timestamps=df_test.index,
            window_size=LOOKBACK_WINDOW,
        )

        # Save Dual-Stream Hybrid Sequences
        for out_npz in [OUTPUT_FNG_HYBRID_NPZ, OUTPUT_HYBRID_NPZ]:
            np.savez_compressed(
                out_npz,
                X_train_price=X_train_price,
                X_train_sent=X_train_sent,
                y_train_dir=y_train_dir,
                y_train_ret=y_train_ret,
                X_val_price=X_val_price,
                X_val_sent=X_val_sent,
                y_val_dir=y_val_dir,
                y_val_ret=y_val_ret,
                X_test_price=X_test_price,
                X_test_sent=X_test_sent,
                y_test_dir=y_test_dir,
                y_test_ret=y_test_ret,
                dates_train=dates_train,
                dates_val=dates_val,
                dates_test=dates_test,
                price_features=np.array(PRICE_FEATURE_COLS),
                sent_features=np.array(SENT_FEATURE_COLS),
            )
            mb = os.path.getsize(out_npz) / (1024 * 1024)
            print(f"   [OK] Saved dual-stream hybrid sequences -> {out_npz.name} ({mb:.2f} MB)")

        print("\n[5/5] Dual-Stream Tensor Shape Verification:")
        print(f"   {'Partition':<12} | {'Price Tensor (X_price)':<24} | {'Sentiment Tensor (X_sent)':<26}")
        print("   " + "-" * 72)
        print(f"   {'Train':<12} | {str(X_train_price.shape):<24} | {str(X_train_sent.shape):<26}")
        print(f"   {'Validation':<12} | {str(X_val_price.shape):<24} | {str(X_val_sent.shape):<26}")
        print(f"   {'Test':<12} | {str(X_test_price.shape):<24} | {str(X_test_sent.shape):<26}")
    else:
        print("\n[5/5] Technical Tensor Shape Verification:")
        print(f"   Train X: {X_train_price.shape} | Val X: {X_val_price.shape} | Test X: {X_test_price.shape}")
        print("   (Note: Run '01_fetch_news_clean.py' and '02_create_clean_dataset.py' to activate Hybrid sequences).")

    print("\n" + "=" * 80)
    print("SEQUENCE PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
