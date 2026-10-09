"""
ETH-Sentinel Research Pipeline — Technical Baseline Benchmark
==============================================================
Script  : 04_train_technical_baselines.py
Purpose : Head-to-head empirical evaluation of 4 baseline models on
          Out-of-Sample Test Set (N=638 4H bars) under strict zero-leakage:
          1. Naive Benchmark (Buy-and-Hold / Class Prior)
          2. Logistic Regression (Ridge / L2 Regularized)
          3. Random Forest Classifier (n_estimators=100, max_depth=5)
          4. Proposed 2-Layer LSTM (32 -> 16 units, Dropout, EarlyStopping)
Evaluation:
  A. Statistical Classification:
     - Directional Accuracy (DA %)
     - Macro F1-Score
     - ROC-AUC Score
     - Matthews Correlation Coefficient (MCC)
     - Brier Score
  B. Financial / Trading Backtest (with 10 bps turnover friction):
     - Cumulative Wealth / Return (%)
     - Annualized Sharpe Ratio (2,190 4H bars/yr)
     - Maximum Drawdown (MDD %)
Artifacts:
  - results/technical_benchmark_metrics.csv
  - results/technical_equity_curves.png
  - results/best_technical_lstm.keras
Author  : ETH-Sentinel Research Team
Target  : Journal of Accounting and Investment (JAI - SINTA 2)
==============================================================
"""

from __future__ import annotations

import os
os.environ["PYTHONHASHSEED"] = "42"
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# Deep Learning Imports
import keras
from keras import callbacks, layers


# =============================================================================
# REPRODUCIBILITY SEED ENFORCEMENT
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DATASET_PATH = SCRIPT_DIR / "dataset" / "eth_technical_sequences.npz"

ROOT_RESULTS_DIR = WORKSPACE_ROOT / "results"
CODE_RESULTS_DIR = SCRIPT_DIR / "results"

for d in [ROOT_RESULTS_DIR, CODE_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 2,190 4-hour periods per calendar year (365 days * 24h / 4h)
ANNUAL_FACTOR_4H = np.sqrt(2190)
TRANSACTION_FEE = 0.0010  # 10 bps (0.10%) per turnover transition


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================
def load_sequences(npz_path: Path):
    """
    Loads pre-scaled, zero-leakage 3D sequences from .npz file.
    Creates flattened 2D representations for tabular/econometric baselines.
    """
    print(f"\n[1/5] Loading sequence tensors from '{npz_path.name}'...")
    if not npz_path.exists():
        raise FileNotFoundError(f"Sequences not found at {npz_path}. Run 03_prepare_sequences.py first.")

    data = np.load(npz_path, allow_pickle=False)

    X_train = data["X_train"].astype(np.float32)
    y_train_dir = data["y_train_dir"].astype(np.int32)
    y_train_ret = data["y_train_ret"].astype(np.float32)

    X_val = data["X_val"].astype(np.float32)
    y_val_dir = data["y_val_dir"].astype(np.int32)
    y_val_ret = data["y_val_ret"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test_dir = data["y_test_dir"].astype(np.int32)
    y_test_ret = data["y_test_ret"].astype(np.float32)
    dates_test = data["dates_test"]

    # Flatten 3D (N, W, D) -> 2D (N, W*D) for non-sequential baselines
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    print(f"   Train samples     : {len(X_train):>5} (3D: {X_train.shape} | 2D: {X_train_flat.shape})")
    print(f"   Validation samples: {len(X_val):>5} (3D: {X_val.shape} | 2D: {X_val_flat.shape})")
    print(f"   Test samples      : {len(X_test):>5} (3D: {X_test.shape} | 2D: {X_test_flat.shape})")
    print(f"   Test Date Range   : {dates_test[0]} -> {dates_test[-1]}")

    return {
        "X_train": X_train, "y_train_dir": y_train_dir, "y_train_ret": y_train_ret,
        "X_val": X_val, "y_val_dir": y_val_dir, "y_val_ret": y_val_ret,
        "X_test": X_test, "y_test_dir": y_test_dir, "y_test_ret": y_test_ret,
        "X_train_flat": X_train_flat, "X_val_flat": X_val_flat, "X_test_flat": X_test_flat,
        "dates_test": dates_test,
    }


# =============================================================================
# EVALUATION METRICS ENGINE
# =============================================================================
def evaluate_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes rigorous statistical classification metrics:
    DA (%), Macro F1, ROC-AUC, MCC, and Brier Score.
    """
    y_pred = (y_prob > 0.50).astype(int)

    da = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Handle single-class edge case for ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5000

    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "DA (%)": da,
        "Macro F1": f1,
        "ROC-AUC": auc,
        "MCC": mcc,
        "Brier Score": brier,
    }


def run_financial_backtest(
    y_prob: np.ndarray,
    y_ret: np.ndarray,
    fee: float = TRANSACTION_FEE,
    is_buy_and_hold: bool = False,
) -> Dict[str, any]:
    """
    Executes realistic out-of-sample trading simulation:
      - Signal: Long (1.0) if P(Up) > 0.50 else Cash (0.0)
      - Frictions: 10 bps (0.10%) per turnover transition
      - Wealth trajectory compounding
      - Metrics: Cumulative Return (%), Annualized Sharpe, Max Drawdown (MDD %)
    """
    n_bars = len(y_ret)
    asset_ret = np.exp(y_ret) - 1.0  # Simple asset return from log return

    if is_buy_and_hold:
        signals = np.ones(n_bars, dtype=np.float32)
        # Buy & hold pays 1 initial entry fee at bar 0
        turnover = np.zeros(n_bars, dtype=np.float32)
        turnover[0] = 1.0
    else:
        signals = (y_prob > 0.50).astype(np.float32)
        prev_signals = np.concatenate(([0.0], signals[:-1]))
        turnover = np.abs(signals - prev_signals)

    cost = turnover * fee
    strat_ret = signals * asset_ret - cost

    # Compounded wealth trajectory starting at 1.0
    wealth = np.cumprod(1.0 + strat_ret)
    wealth = np.concatenate(([1.0], wealth))

    cum_return = (wealth[-1] - 1.0) * 100.0

    # Annualized Sharpe Ratio (Rf = 0)
    mean_ret = np.mean(strat_ret)
    std_ret = np.std(strat_ret)
    if std_ret > 1e-9:
        sharpe = ANNUAL_FACTOR_4H * (mean_ret / std_ret)
    else:
        sharpe = 0.0

    # Maximum Drawdown
    peaks = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peaks) / peaks
    mdd = np.min(drawdowns) * 100.0

    return {
        "Cum Return (%)": cum_return,
        "Annualized Sharpe": sharpe,
        "Max Drawdown (%)": mdd,
        "Turnover Trades": int(np.sum(turnover > 0)),
        "Total Cost (%)": float(np.sum(cost) * 100.0),
        "Wealth": wealth,
        "Drawdowns": drawdowns,
    }


# =============================================================================
# MODEL ARCHITECTURES & TRAINING
# =============================================================================
def train_and_eval_naive(data: dict) -> Tuple[Dict[str, float], Dict[str, any]]:
    """
    Baseline 0: Buy-and-Hold / Majority Prior Class.
    """
    print("\n--- [Model 1/4] Baseline 0: Naive Benchmark (Buy & Hold) ---")
    y_test_dir = data["y_test_dir"]
    y_test_ret = data["y_test_ret"]

    # Predict train class prior probability
    prior_up = np.mean(data["y_train_dir"] == 1)
    y_prob = np.full(len(y_test_dir), prior_up, dtype=np.float32)

    stat_metrics = {
        "DA (%)": np.mean(y_test_dir == 1) * 100.0,
        "Macro F1": f1_score(y_test_dir, np.ones_like(y_test_dir), average="macro", zero_division=0),
        "ROC-AUC": 0.5000,
        "MCC": 0.0000,
        "Brier Score": brier_score_loss(y_test_dir, y_prob),
    }

    fin_metrics = run_financial_backtest(y_prob, y_test_ret, is_buy_and_hold=True)
    return stat_metrics, fin_metrics


def train_and_eval_logistic_regression(data: dict) -> Tuple[Dict[str, float], Dict[str, any]]:
    """
    Baseline 1: Logistic Regression with L2 Regularization (Ridge).
    """
    print("\n--- [Model 2/4] Baseline 1: Logistic Regression (Ridge L2) ---")
    X_train = data["X_train_flat"]
    y_train = data["y_train_dir"]
    X_test = data["X_test_flat"]
    y_test_dir = data["y_test_dir"]
    y_test_ret = data["y_test_ret"]

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1].astype(np.float32)
    stat_metrics = evaluate_classification_metrics(y_test_dir, y_prob)
    fin_metrics = run_financial_backtest(y_prob, y_test_ret)

    return stat_metrics, fin_metrics


def train_and_eval_random_forest(data: dict) -> Tuple[Dict[str, float], Dict[str, any]]:
    """
    Baseline 2: Random Forest Classifier (n_estimators=100, max_depth=5).
    """
    print("\n--- [Model 3/4] Baseline 2: Random Forest Classifier (n=100, depth=5) ---")
    X_train = data["X_train_flat"]
    y_train = data["y_train_dir"]
    X_test = data["X_test_flat"]
    y_test_dir = data["y_test_dir"]
    y_test_ret = data["y_test_ret"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1].astype(np.float32)
    stat_metrics = evaluate_classification_metrics(y_test_dir, y_prob)
    fin_metrics = run_financial_backtest(y_prob, y_test_ret)

    return stat_metrics, fin_metrics


def train_and_eval_lstm(data: dict) -> Tuple[Dict[str, float], Dict[str, any], keras.Model]:
    """
    Proposed Deep Learning: 2-Layer LSTM with Dropout, Recurrent Dropout, and EarlyStopping.
    """
    print("\n--- [Model 4/4] Proposed: 2-Layer Compact LSTM (Anti-Overfitting) ---")
    X_train = data["X_train"]
    y_train = data["y_train_dir"]
    X_val = data["X_val"]
    y_val = data["y_val_dir"]
    X_test = data["X_test"]
    y_test_dir = data["y_test_dir"]
    y_test_ret = data["y_test_ret"]

    # Model definition
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2]), name="input_3d_sequences"),
        layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            name="lstm_layer_1",
        ),
        layers.LSTM(
            16,
            dropout=0.2,
            name="lstm_layer_2",
        ),
        layers.Dense(
            1,
            activation="sigmoid",
            name="output_direction",
        ),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    es_callback = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    print("   Starting LSTM model training (max 60 epochs, patience=10)...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        callbacks=[es_callback],
        verbose=1,
    )

    stopped_epoch = len(history.history["loss"])
    best_val_loss = min(history.history["val_loss"])
    print(f"   Training completed at epoch {stopped_epoch}. Best Val Loss: {best_val_loss:.4f}")

    # Generate test predictions
    y_prob = model.predict(X_test, verbose=0).flatten().astype(np.float32)
    stat_metrics = evaluate_classification_metrics(y_test_dir, y_prob)
    fin_metrics = run_financial_backtest(y_prob, y_test_ret)

    return stat_metrics, fin_metrics, model


# =============================================================================
# PLOTTING & REPORT EXPORT
# =============================================================================
def plot_equity_curves(
    results: Dict[str, Dict[str, any]],
    dates: np.ndarray,
    save_paths: list[Path],
):
    """
    Generates high-resolution publication-quality 2-panel chart:
      Panel 1: Cumulative Wealth Trajectory ($W_t$)
      Panel 2: Underwater Drawdown Profile (%)
    """
    print("\nGenerating publication-grade comparative equity curves...")
    
    # Parse dates to pandas datetime
    dt_index = pd.to_datetime(dates)
    # Prepend the start timestamp for wealth[0] = 1.0
    start_dt = dt_index[0] - pd.Timedelta(hours=4)
    chart_dates = pd.DatetimeIndex([start_dt] + list(dt_index))

    # Styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )

    palette = {
        "Naive (Buy & Hold)": ("#7f7f7f", "--", 1.8),
        "Logistic Regression": ("#ff7f0e", "-.", 1.8),
        "Random Forest": ("#2ca02c", ":", 2.0),
        "Proposed 2-Layer LSTM": ("#1f77b4", "-", 2.5),
    }

    # Top Panel: Cumulative Wealth
    for model_name, res in results.items():
        color, style, lw = palette[model_name]
        wealth = res["fin"]["Wealth"]
        sharpe = res["fin"]["Annualized Sharpe"]
        cum_ret = res["fin"]["Cum Return (%)"]
        label = f"{model_name} | Return: {cum_ret:+.2f}% | Sharpe: {sharpe:.2f}"
        ax1.plot(chart_dates, wealth, label=label, color=color, linestyle=style, linewidth=lw)

    ax1.axhline(1.0, color="#black" if False else "#444444", linestyle=":", alpha=0.5, linewidth=1.0)
    ax1.set_ylabel("Cumulative Wealth ($W_t$) [Initial = 1.00]", fontweight="bold")
    ax1.set_title("ETH-Sentinel Technical Benchmark: Out-of-Sample Performance Comparison\n"
                  f"Test Set Evaluation (N = {len(dates)} 4H bars, June–September 2026, 10 bps Friction)",
                  fontweight="bold", pad=12)
    ax1.legend(loc="upper left", framealpha=0.92, facecolor="#ffffff", edgecolor="#cccccc")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}x"))

    # Bottom Panel: Underwater Drawdown
    for model_name, res in results.items():
        color, style, lw = palette[model_name]
        dd = res["fin"]["Drawdowns"]
        mdd = res["fin"]["Max Drawdown (%)"]
        ax2.plot(chart_dates, dd * 100.0, label=f"{model_name} (MDD: {mdd:.2f}%)", color=color, linestyle=style, linewidth=lw * 0.8)

    ax2.set_ylabel("Drawdown (%)", fontweight="bold")
    ax2.set_xlabel("Out-of-Sample Date (UTC)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # Format Date Axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()

    plt.tight_layout()

    for p in save_paths:
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"   [OK] Saved equity curve plot -> {p}")
    plt.close(fig)


# =============================================================================
# MAIN ORCHESTRATION PIPELINE
# =============================================================================
def main():
    print("=" * 80)
    print("ETH-SENTINEL: TECHNICAL-ONLY BASELINE BENCHMARK ENGINE")
    print("Zero-Leakage Test Set Evaluation | Statistical & Econometric Rigor")
    print("=" * 80)

    # 1. Load data
    data = load_sequences(DATASET_PATH)
    y_test_dir = data["y_test_dir"]

    # 2. Train & Evaluate Models
    results = {}

    # Model 1: Naive Buy & Hold
    stat_naive, fin_naive = train_and_eval_naive(data)
    results["Naive (Buy & Hold)"] = {"stat": stat_naive, "fin": fin_naive}

    # Model 2: Logistic Regression
    stat_lr, fin_lr = train_and_eval_logistic_regression(data)
    results["Logistic Regression"] = {"stat": stat_lr, "fin": fin_lr}

    # Model 3: Random Forest
    stat_rf, fin_rf = train_and_eval_random_forest(data)
    results["Random Forest"] = {"stat": stat_rf, "fin": fin_rf}

    # Model 4: Proposed 2-Layer LSTM
    stat_lstm, fin_lstm, lstm_model = train_and_eval_lstm(data)
    results["Proposed 2-Layer LSTM"] = {"stat": stat_lstm, "fin": fin_lstm}

    # 3. Save LSTM Model
    lstm_save_paths = [
        ROOT_RESULTS_DIR / "best_technical_lstm.keras",
        CODE_RESULTS_DIR / "best_technical_lstm.keras",
    ]
    for p in lstm_save_paths:
        lstm_model.save(p)
        print(f"   [OK] Saved best LSTM model -> {p}")

    # 4. Construct Benchmark Summary Table
    rows = []
    for model_name, res in results.items():
        row = {
            "Model": model_name,
            "Directional Accuracy (%)": f"{res['stat']['DA (%)']:.2f}%",
            "Macro F1-Score": f"{res['stat']['Macro F1']:.4f}",
            "ROC-AUC": f"{res['stat']['ROC-AUC']:.4f}",
            "MCC": f"{res['stat']['MCC']:.4f}",
            "Brier Score": f"{res['stat']['Brier Score']:.4f}",
            "Cumulative Return (%)": f"{res['fin']['Cum Return (%)']:+.2f}%",
            "Annualized Sharpe": f"{res['fin']['Annualized Sharpe']:.2f}",
            "Max Drawdown (%)": f"{res['fin']['Max Drawdown (%)']:.2f}%",
            "Turnover Trades": res["fin"]["Turnover Trades"],
            "Total Friction Cost (%)": f"{res['fin']['Total Cost (%)']:.2f}%",
        }
        rows.append(row)

    df_metrics = pd.DataFrame(rows)

    # Export metrics CSV to both results directories
    csv_paths = [
        ROOT_RESULTS_DIR / "technical_benchmark_metrics.csv",
        CODE_RESULTS_DIR / "technical_benchmark_metrics.csv",
    ]
    for p in csv_paths:
        df_metrics.to_csv(p, index=False)
        print(f"   [OK] Saved benchmark metrics table -> {p}")

    # 5. Plot Comparative Equity Curves
    img_paths = [
        ROOT_RESULTS_DIR / "technical_equity_curves.png",
        CODE_RESULTS_DIR / "technical_equity_curves.png",
    ]
    plot_equity_curves(results, data["dates_test"], img_paths)

    # 6. Display Clean Summary Table in Console
    print("\n" + "=" * 105)
    print("HEAD-TO-HEAD BENCHMARK RESULTS (OUT-OF-SAMPLE TEST SET, N = 638 BARS)")
    print("=" * 105)
    print(df_metrics.to_string(index=False))
    print("=" * 105)
    print("\n[PASSED] All technical benchmark experiments completed successfully!")


if __name__ == "__main__":
    main()
