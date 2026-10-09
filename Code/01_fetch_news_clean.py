"""
ETH-Sentinel Research Pipeline — News & Sentiment Ingestion Engine v2.0
========================================================================
Script  : 01_fetch_news_clean.py
Purpose : Automated ingestion and dual-scoring of financial crypto news:
          1. CryptoCompare News API with backward pagination via 'lTs'.
          2. Explicit UTC ISO-8601 timestamp conversion (eliminates 7H offset).
          3. Dual-scoring in real time:
             - Loughran-McDonald Polarity (pysentiment2, financial dictionary)
             - VADER Compound Score (vaderSentiment, rule-based social valence)
          4. Robust batch checkpointing to prevent progress loss.
          5. Fallback mode: Ingests, validates, re-scores, and converts
             legacy CSV news archives to canonical UTC format.
Output  : Code/dataset/sentiment_data_clean.csv
Author  : ETH-Sentinel Research Team
Target  : Journal of Accounting and Investment (JAI - SINTA 2)
========================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Sentiment Analyzers
try:
    import pysentiment2 as ps
    LM_AVAILABLE = True
except ImportError:
    LM_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Environment Variable Loader
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# =============================================================================
# PATH CONFIGURATION & ENV LOADING
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DATASET_DIR = SCRIPT_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CLEAN_CSV = DATASET_DIR / "sentiment_data_clean.csv"
LEGACY_CANDIDATE_PATHS = [
    SCRIPT_DIR / "sentiment_data.csv",
    WORKSPACE_ROOT / "sentiment_data.csv",
    DATASET_DIR / "sentiment_data.csv",
    SCRIPT_DIR / "dataset" / "sentiment_data.csv",
]

# Attempt to load .env from multiple locations
if DOTENV_AVAILABLE:
    load_dotenv(WORKSPACE_ROOT / ".env")
    load_dotenv(SCRIPT_DIR / ".env")
else:
    # Lightweight manual .env fallback
    for env_path in [WORKSPACE_ROOT / ".env", SCRIPT_DIR / ".env"]:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


API_BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/"


# =============================================================================
# SENTIMENT SCORING ENGINE
# =============================================================================
class DualSentimentScorer:
    """
    Computes both Loughran-McDonald (Finance) and VADER (Valence) sentiment scores.
    """
    def __init__(self):
        print("   Initializing Sentiment Analyzers:")
        if LM_AVAILABLE:
            self.lm = ps.LM()
            print("     * Loughran-McDonald (pysentiment2) [ACTIVE]")
        else:
            self.lm = None
            print("     * Loughran-McDonald [NOT INSTALLED - fallback to 0.0]")

        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            print("     * VADER Sentiment Analyzer [ACTIVE]")
        else:
            self.vader = None
            print("     * VADER Sentiment Analyzer [NOT INSTALLED - fallback to 0.0]")

    def score_text(self, title: str, body: str = "") -> Tuple[float, float]:
        full_text = f"{title}. {body}".strip() if body else str(title).strip()
        if not full_text:
            return 0.0, 0.0

        # 1. Loughran-McDonald Polarity
        lm_score = 0.0
        if self.lm is not None:
            try:
                tokens = self.lm.tokenize(full_text)
                res = self.lm.get_score(tokens)
                lm_score = float(res.get("Polarity", 0.0))
            except Exception:
                lm_score = 0.0

        # 2. VADER Compound Score
        vader_score = 0.0
        if self.vader is not None:
            try:
                v_res = self.vader.polarity_scores(full_text)
                vader_score = float(v_res.get("compound", 0.0))
            except Exception:
                vader_score = 0.0

        return lm_score, vader_score


# =============================================================================
# BATCH CHECKPOINTING & STORAGE
# =============================================================================
def load_existing_checkpoint(csv_path: Path) -> Tuple[pd.DataFrame, Set[str], Optional[int]]:
    """
    Loads already collected articles to allow seamless continuation.
    Returns (DataFrame, set_of_seen_ids_or_urls, oldest_unix_timestamp).
    """
    if not csv_path.exists():
        return pd.DataFrame(), set(), None

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame(), set(), None

        # Build lookup set for deduplication
        seen_keys = set()
        if "Article_ID" in df.columns:
            seen_keys.update(df["Article_ID"].dropna().astype(str).tolist())
        if "Article_URL" in df.columns:
            seen_keys.update(df["Article_URL"].dropna().astype(str).tolist())

        oldest_ts = None
        if "Timestamp_Unix" in df.columns and not df["Timestamp_Unix"].dropna().empty:
            oldest_ts = int(df["Timestamp_Unix"].min())

        return df, seen_keys, oldest_ts
    except Exception as e:
        print(f"   [Notice] Could not read existing checkpoint ({e}). Starting fresh.")
        return pd.DataFrame(), set(), None


def save_checkpoint(df: pd.DataFrame, csv_path: Path):
    """
    Saves dataframe atomically with proper sorting.
    """
    if df.empty:
        return
    # Ensure Datetime_UTC sorting
    if "Datetime_UTC" in df.columns:
        df = df.sort_values(by="Datetime_UTC", ascending=True)
    df.to_csv(csv_path, index=False)


# =============================================================================
# API FETCH ENGINE WITH BACKWARD PAGINATION (lTs)
# =============================================================================
def fetch_cryptocompare_news(
    api_key: Optional[str],
    target_count: int = 50000,
    categories: Optional[str] = None,
    rate_limit_delay: float = 0.8,
    scorer: Optional[DualSentimentScorer] = None,
    checkpoint_file: Path = OUTPUT_CLEAN_CSV,
) -> pd.DataFrame:
    """
    Fetches news backwards in time using CryptoCompare's `lTs` pagination.
    Performs real-time sentiment scoring and saves checkpoints every batch.
    """
    if scorer is None:
        scorer = DualSentimentScorer()

    df_existing, seen_keys, oldest_ts = load_existing_checkpoint(checkpoint_file)
    print(f"\n[API Mode] Target: {target_count:,} articles | Existing in Checkpoint: {len(df_existing):,}")

    headers = {}
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"
        print("   CryptoCompare API Key: [DETECTED & CONFIGURED]")
    else:
        print("   CryptoCompare API Key: [NOT DETECTED] — Running in Public Rate-Limited Mode.")
        print("   * Note: Public endpoints may restrict historical pagination depth.")

    all_records = df_existing.to_dict("records") if not df_existing.empty else []
    last_timestamp = oldest_ts

    consecutive_empty = 0
    pbar = tqdm(total=target_count, initial=len(all_records), desc="Collecting & Scoring News", unit="art")

    while len(all_records) < target_count:
        params = {
            "lang": "EN",
            "sortOrder": "latest",
        }
        if categories:
            params["categories"] = categories
        if last_timestamp is not None:
            params["lTs"] = int(last_timestamp)

        try:
            resp = requests.get(API_BASE_URL, headers=headers, params=params, timeout=15)
            if resp.status_code == 429:
                print("\n   [Rate Limit Warning] 429 encountered. Sleeping 5 seconds...")
                time.sleep(5.0)
                continue
            if resp.status_code != 200:
                print(f"\n   [HTTP Error] Status {resp.status_code}: {resp.text[:200]}")
                break

            payload = resp.json()
            if payload.get("Type") != 100:
                msg = payload.get("Message", "Unknown API error")
                print(f"\n   [API Response] Message: {msg}")
                if "rate limit" in msg.lower() or "api key" in msg.lower():
                    print("   Terminating pagination due to API authentication/limit.")
                break

            articles = payload.get("Data", [])
            if not articles:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    print("\n   [Pagination End] No further historical articles returned.")
                    break
                time.sleep(1.0)
                continue
            else:
                consecutive_empty = 0

            batch_new = []
            for art in articles:
                art_id = str(art.get("id", ""))
                url = art.get("url", "")
                unique_key = art_id or url

                if unique_key and unique_key in seen_keys:
                    continue

                pub_ts = art.get("published_on", 0)
                if pub_ts <= 0:
                    continue

                # Explicit UTC ISO-8601 formatting
                dt_utc = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
                dt_utc_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S+00:00")

                title = art.get("title", "").strip()
                body = art.get("body", "").strip()
                source = art.get("source", "")
                cat = art.get("categories", "")

                # Real-time sentiment extraction
                lm_pol, vader_comp = scorer.score_text(title, body)

                record = {
                    "Article_ID": art_id,
                    "Datetime_UTC": dt_utc_str,
                    "Timestamp_Unix": pub_ts,
                    "Article_Title": title,
                    "Article_Body": body,
                    "Article_URL": url,
                    "News_Source": source,
                    "Categories": cat,
                    "Sentiment_LM": lm_pol,
                    "Sentiment_VADER": vader_comp,
                }
                batch_new.append(record)
                seen_keys.add(unique_key)

            if batch_new:
                all_records.extend(batch_new)
                pbar.update(len(batch_new))

                # Periodic Checkpoint Saving
                df_temp = pd.DataFrame(all_records)
                save_checkpoint(df_temp, checkpoint_file)

            # Move pagination cursor to oldest article in this batch
            oldest_in_batch = min(art.get("published_on", sys.maxsize) for art in articles)
            if oldest_in_batch == sys.maxsize or (last_timestamp is not None and oldest_in_batch >= last_timestamp):
                # Decrement by 1 to prevent getting stuck
                last_timestamp = (last_timestamp or int(time.time())) - 1
            else:
                last_timestamp = oldest_in_batch

            time.sleep(rate_limit_delay)

        except Exception as e:
            print(f"\n   [Exception] Connection interrupted: {e}. Progress is preserved in checkpoint.")
            break

    pbar.close()
    final_df = pd.DataFrame(all_records)
    save_checkpoint(final_df, checkpoint_file)
    print(f"\n   [Complete] Saved {len(final_df):,} articles to '{checkpoint_file.name}'")
    return final_df


# =============================================================================
# FALLBACK: INGEST & NORMALIZE LEGACY CSV FILES
# =============================================================================
def ingest_legacy_news_csv(
    legacy_csv_path: Path,
    scorer: Optional[DualSentimentScorer] = None,
    output_path: Path = OUTPUT_CLEAN_CSV,
) -> pd.DataFrame:
    """
    Ingests an existing/legacy CSV, detects date and text columns,
    enforces UTC ISO-8601 timestamps, computes missing sentiment scores,
    and formats output canonically.
    """
    print(f"\n[Fallback Ingestion Mode] Processing legacy file: '{legacy_csv_path.name}'...")
    if not legacy_csv_path.exists():
        raise FileNotFoundError(f"File not found: {legacy_csv_path}")

    if scorer is None:
        scorer = DualSentimentScorer()

    df_raw = pd.read_csv(legacy_csv_path)
    print(f"   Raw records loaded: {len(df_raw):,}")
    print(f"   Detected columns  : {list(df_raw.columns)}")

    # 1. Detect and parse timestamp/datetime
    date_col = None
    for candidate in ["published_on", "timestamp", "Published Date", "Published_Date", "date", "datetime"]:
        if candidate in df_raw.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError(f"Could not automatically detect date/timestamp column in {legacy_csv_path}")

    print(f"   Using datetime column: '{date_col}'")
    sample_val = df_raw[date_col].dropna().iloc[0]

    # Convert to Unix & UTC
    if isinstance(sample_val, (int, float, np.integer, np.floating)) or (isinstance(sample_val, str) and sample_val.isdigit()):
        # Epoch seconds or milliseconds
        ts_series = pd.to_numeric(df_raw[date_col], errors="coerce")
        # Check if milliseconds (> 1e11)
        if ts_series.max() > 1e11:
            ts_series = ts_series / 1000.0
        df_raw["Timestamp_Unix"] = ts_series.astype("Int64")
        df_raw["Datetime_UTC"] = pd.to_datetime(ts_series, unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    else:
        # String datetime: enforce UTC conversion
        dt_series = pd.to_datetime(df_raw[date_col], utc=True, errors="coerce")
        df_raw["Datetime_UTC"] = dt_series.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        df_raw["Timestamp_Unix"] = dt_series.astype("int64") // 10**9

    # Drop rows without valid dates
    df_raw = df_raw.dropna(subset=["Datetime_UTC"]).copy()

    # 2. Text columns
    title_col = next((c for c in ["Article Title", "Article_Title", "title", "headline"] if c in df_raw.columns), None)
    body_col  = next((c for c in ["Article Content", "Article_Content", "body", "content", "description"] if c in df_raw.columns), None)

    if title_col:
        df_raw["Article_Title"] = df_raw[title_col].fillna("").astype(str)
    else:
        df_raw["Article_Title"] = ""

    if body_col:
        df_raw["Article_Body"] = df_raw[body_col].fillna("").astype(str)
    else:
        df_raw["Article_Body"] = ""

    # URL / ID
    url_col = next((c for c in ["Article URL", "Article_URL", "url", "link", "id"] if c in df_raw.columns), None)
    if url_col:
        df_raw["Article_URL"] = df_raw[url_col].fillna("").astype(str)
    else:
        df_raw["Article_URL"] = ""

    df_raw["Article_ID"] = df_raw.get("Article_ID", df_raw["Article_URL"])
    df_raw["News_Source"] = df_raw.get("source", df_raw.get("News_Source", "Legacy_Archive"))
    df_raw["Categories"] = df_raw.get("categories", df_raw.get("Categories", "General"))

    # 3. Sentiment Scoring (Compute if missing or re-score)
    needs_lm = ("Sentiment_LM" not in df_raw.columns) or df_raw["Sentiment_LM"].isna().all()
    needs_vader = ("Sentiment_VADER" not in df_raw.columns) or df_raw["Sentiment_VADER"].isna().all()

    if needs_lm or needs_vader:
        print(f"   Calculating missing sentiment scores (Needs LM: {needs_lm} | Needs VADER: {needs_vader})...")
        lm_scores = []
        vader_scores = []
        for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Scoring Legacy Articles"):
            t = row["Article_Title"]
            b = row["Article_Body"]
            lm_p, v_c = scorer.score_text(t, b)
            lm_scores.append(lm_p)
            vader_scores.append(v_c)

        if needs_lm:
            df_raw["Sentiment_LM"] = lm_scores
        if needs_vader:
            df_raw["Sentiment_VADER"] = vader_scores

    # 4. Canonical column arrangement and deduplication
    canonical_cols = [
        "Article_ID",
        "Datetime_UTC",
        "Timestamp_Unix",
        "Article_Title",
        "Article_Body",
        "Article_URL",
        "News_Source",
        "Categories",
        "Sentiment_LM",
        "Sentiment_VADER",
    ]
    df_clean = df_raw[[c for c in canonical_cols if c in df_raw.columns]].copy()
    df_clean = df_clean.drop_duplicates(subset=["Datetime_UTC", "Article_Title"])
    df_clean = df_clean.sort_values(by="Datetime_UTC", ascending=True)

    save_checkpoint(df_clean, output_path)
    print(f"   [Success] Cleaned legacy data saved -> {output_path}")
    print(f"   Usable articles : {len(df_clean):,}")
    print(f"   Date span (UTC) : {df_clean['Datetime_UTC'].min()} -> {df_clean['Datetime_UTC'].max()}")

    return df_clean


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="ETH-Sentinel: News Fetching, UTC Cleaning, and Dual Sentiment Ingestion Engine"
    )
    parser.add_argument("--api-key", type=str, default=None, help="CryptoCompare API Key (or set CRYPTOCOMPARE_API_KEY)")
    parser.add_argument("--target-count", type=int, default=50000, help="Target total articles to collect")
    parser.add_argument("--categories", type=str, default=None, help="Categories filter (e.g. 'ETH,ETHEREUM')")
    parser.add_argument("--rate-limit", type=float, default=0.8, help="Seconds to sleep between API calls")
    parser.add_argument("--import-file", type=str, default=None, help="Path to legacy news CSV to normalize into UTC")

    args = parser.parse_args()

    print("=" * 80)
    print("ETH-SENTINEL: FINANCIAL NEWS INGESTION & DUAL-SENTIMENT ENGINE v2.0")
    print("UTC Enforced | LM + VADER Real-time Scoring | Checkpoint Safe")
    print("=" * 80)

    scorer = DualSentimentScorer()

    # Mode 1: Import existing legacy file if provided
    if args.import_file:
        import_path = Path(args.import_file)
        ingest_legacy_news_csv(import_path, scorer=scorer, output_path=OUTPUT_CLEAN_CSV)
        return

    # Check for legacy candidates automatically
    existing_legacy = next((p for p in LEGACY_CANDIDATE_PATHS if p.exists() and p != OUTPUT_CLEAN_CSV), None)
    if existing_legacy:
        print(f"\n[Auto-Detection] Found legacy news file at: {existing_legacy}")
        user_choice = input(f"Would you like to import and normalize '{existing_legacy.name}'? (y/n): ").strip().lower()
        if user_choice in ("y", "yes"):
            ingest_legacy_news_csv(existing_legacy, scorer=scorer, output_path=OUTPUT_CLEAN_CSV)
            return

    # Mode 2: Live API Pagination
    api_key = args.api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
    fetch_cryptocompare_news(
        api_key=api_key,
        target_count=args.target_count,
        categories=args.categories,
        rate_limit_delay=args.rate_limit,
        scorer=scorer,
        checkpoint_file=OUTPUT_CLEAN_CSV,
    )


if __name__ == "__main__":
    main()
