
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

def load_sentiment_data(filepath="sentiment_data.csv"):
    """Loads sentiment data and parses dates."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Please run 03_fetch_news.py first.")
        return None
    
    print(f"Loading sentiment data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Ensure 'Published Date' exists
    if 'Published Date' not in df.columns:
        print(f"Error: 'Published Date' column missing in {filepath}. Columns found: {df.columns}")
        return None

    # Parse dates
    try:
        df['datetime'] = pd.to_datetime(df['Published Date'])
    except Exception as e:
        print(f"Error parsing dates: {e}")
        return None
        
    # Sort by date
    df = df.sort_values('datetime')
    return df

def fetch_price_data(start_date, end_date):
    """Fetches ETH-USD price data from Yahoo Finance."""
    print(f"Fetching ETH-USD price data from {start_date} to {end_date}...")
    try:
        # Ticker
        eth = yf.Ticker("ETH-USD")
        df_price = eth.history(start=start_date, end=end_date, interval="1h")
        
        if df_price.empty:
            print("Warning: No hourly data fetched. Trying daily interval...")
            df_price = eth.history(start=start_date, end=end_date, interval="1d")
            
        if df_price.empty:
            print("Error: Could not fetch price data.")
            return None
            
        # Reset index to make Date a column
        df_price = df_price.reset_index()
        
        # Standardize timezone to UTC (yfinance returns tz-aware)
        if 'Datetime' in df_price.columns:
            df_price['datetime'] = pd.to_datetime(df_price['Datetime']).dt.tz_localize(None)
        elif 'Date' in df_price.columns:
            df_price['datetime'] = pd.to_datetime(df_price['Date']).dt.tz_localize(None)
            
        return df_price[['datetime', 'Close', 'Volume']]
        
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return None

def merge_and_process(df_sentiment, df_price):
    """Merges sentiment and price data, resampling to common frequency."""
    print("Processing and merging data...")
    
    # 1. Resample Sentiment to Hourly (taking mean of sentiment scores for that hour)
    df_sentiment = df_sentiment.set_index('datetime')
    sentiment_hourly = df_sentiment['Sentiment Score'].resample('1h').mean().fillna(0)
    sentiment_hourly = sentiment_hourly.reset_index()
    
    # 2. Merge with Price Data
    df_price['datetime'] = df_price['datetime'].dt.floor('h')
    merged_df = pd.merge(df_price, sentiment_hourly, on='datetime', how='left')
    merged_df['Sentiment Score'] = merged_df['Sentiment Score'].fillna(0.0)
    merged_df = merged_df.rename(columns={'Close': 'price', 'Volume': 'volume'})
    
    return merged_df

def main():
    # 1. Load Sentiment
    df_sentiment = load_sentiment_data()
    if df_sentiment is None or df_sentiment.empty:
        print("No sentiment data found. Please run fetch_news_api.py first.")
        return
    min_date = df_sentiment['datetime'].min()
    max_date = df_sentiment['datetime'].max()
    start_date = min_date - timedelta(days=1)
    end_date = max_date + timedelta(days=1)
    
    print(f"Sentiment Data Range: {min_date} to {max_date}")

    # 2. Fetch Price
    df_price = fetch_price_data(start_date, end_date)
    if df_price is None or df_price.empty:
        print("Failed to get price data.")
        return

    # 3. Merge
    final_df = merge_and_process(df_sentiment, df_price)
    
    # 4. Save
    output_file = "training_data.csv"
    print(f"Saving merged data to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print(f"Successfully created {output_file} with {len(final_df)} rows.")
    print("Columns:", final_df.columns.tolist())

if __name__ == "__main__":
    main()
