"""
ETH-Sentinel Research Pipeline — Crypto Fear & Greed Index (CFGI) Ingestion
=============================================================================
Script  : 01_fetch_fng_sentiment.py
Purpose : Fetch complete historical sentiment data from Alternative.me
          Crypto Fear & Greed Index (CFGI) via open public API.
Endpoint: https://api.alternative.me/fng/?limit=0 (Free, Open-Access, 2018-Present)
Output  : Code/dataset/sentiment_fng_clean.csv
Author  : ETH-Sentinel Research Team
Target  : Journal of Accounting and Investment (JAI - SINTA 2)
=============================================================================
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = DATASET_DIR / "sentiment_fng_clean.csv"

API_URL = "https://api.alternative.me/fng/?limit=0"


def fetch_and_process_fng(output_path: Path = OUTPUT_CSV) -> pd.DataFrame:
    print("=" * 80)
    print("ETH-SENTINEL: CRYPTO FEAR & GREED INDEX (CFGI) INGESTION")
    print(f"Target Endpoint: {API_URL}")
    print("=" * 80)

    print("\n[1/3] Fetching complete historical CFGI dataset via HTTP GET...")
    try:
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        print(f"   [Error] Failed to fetch data from Alternative.me: {e}")
        sys.exit(1)

    items = payload.get("data", [])
    if not items:
        print("   [Error] Empty data array received from API.")
        sys.exit(1)

    print(f"   Received {len(items):,} raw daily sentiment records.")

    print("\n[2/3] Parsing and converting to explicit UTC ISO-8601...")
    records = []
    for item in items:
        ts = int(item.get("timestamp", 0))
        if ts <= 0:
            continue

        # Explicit UTC timestamp conversion
        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        dt_utc_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S+00:00")

        val_raw = float(item.get("value", 0.0))
        classification = item.get("value_classification", "Neutral").strip()

        records.append({
            "Datetime_UTC": dt_utc_str,
            "Timestamp_Unix": ts,
            "FNG_Value": val_raw,
            "FNG_Classification": classification,
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["Timestamp_Unix"])
    df = df.sort_values("Timestamp_Unix", ascending=True).reset_index(drop=True)

    print(f"   Processed records    : {len(df):,}")
    print(f"   Chronological span   : {df['Datetime_UTC'].iloc[0]} -> {df['Datetime_UTC'].iloc[-1]}")
    print(f"   Mean Sentiment Value : {df['FNG_Value'].mean():.2f} (Std: {df['FNG_Value'].std():.2f})")
    print(f"   Min / Max Value      : {df['FNG_Value'].min():.1f} / {df['FNG_Value'].max():.1f}")

    print("\n   Sentiment Classification Breakdown:")
    class_counts = df["FNG_Classification"].value_counts()
    for cat, cnt in class_counts.items():
        print(f"     * {cat:<18}: {cnt:>5} days ({cnt/len(df):>6.2%})")

    print(f"\n[3/3] Saving clean sentiment dataset to '{output_path.name}'...")
    df.to_csv(output_path, index=False)
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"   [OK] Saved -> {output_path} ({file_size_kb:.1f} KB)")
    print("=" * 80)
    print("CFGI INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return df


if __name__ == "__main__":
    fetch_and_process_fng()
