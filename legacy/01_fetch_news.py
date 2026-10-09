import requests
import pandas as pd
import time
from datetime import datetime
import pysentiment2 as ps
import numpy as np
from tqdm import tqdm

# Configuration
API_URL = "https://min-api.cryptocompare.com/data/v2/news/"
LIMIT_PER_REQUEST = 50
TARGET_TOTAL_NEWS = 100000
OUTPUT_FILE = "sentiment_data.csv" 

# Sentiment Model Setup
print("Initializing Loughran-McDonald Sentiment Analyzer (Finance Oriented)...")
lm = ps.LM()

def get_sentiment(text):
    """
    Predicts sentiment score for a given text using Loughran-McDonald Dictionary.
    Returns a Polarity score.
    """
    if not text:
        return 0.0
    
    # Tokenize and score
    tokens = lm.tokenize(text)
    score = lm.get_score(tokens)
    return score['Polarity']

def fetch_historical_news():
    all_news = []
    last_timestamp = None
    
    print(f"Starting to fetch news. Target: {TARGET_TOTAL_NEWS} articles...")
    
    while len(all_news) < TARGET_TOTAL_NEWS:
        params = {
            "lang": "EN",
            "sortOrder": "latest"
        }
        
        if last_timestamp:
            params["lTs"] = last_timestamp
            
        try:
            response = requests.get(API_URL, params=params)
            data = response.json()
            
            if data.get("Type") != 100:
                print(f"Error checking API: {data.get('Message')}")
                break
                
            articles = data.get("Data", [])
            
            if not articles:
                print("No more articles found.")
                break
            
            for article in articles:
                title = article.get("title", "")
                body = article.get("body", "")
                published_on = article.get("published_on", 0)
                url = article.get("url", "")
                source = article.get("source", "")
                full_text = f"{title}. {body}"
                dt_object = datetime.fromtimestamp(published_on)
                published_date_str = dt_object.strftime("%Y-%m-%d %H:%M:%S") 
                
                all_news.append({
                    "Article Title": title,
                    "Article URL": url, 
                    "Article Content": body, 
                    "Published Date": published_date_str,
                    "full_text_for_sentiment": full_text,
                    "timestamp": published_on
                })
            last_timestamp = articles[-1].get("published_on")
            
            print(f"Fetched {len(all_news)} articles so far... (Last date: {datetime.fromtimestamp(last_timestamp)})")

            time.sleep(0.5)
            
        except Exception as e:
            print(f"Exception occurred: {e}")
            break
            
    return pd.DataFrame(all_news)

# Main Execution
if __name__ == "__main__":
    df_news = fetch_historical_news()
    
    if not df_news.empty:
        print(f"\nPerforming sentiment analysis on {len(df_news)} articles...")
        
        tqdm.pandas(desc="Analyzing Sentiment")
        df_news['Sentiment Score'] = df_news['full_text_for_sentiment'].progress_apply(get_sentiment)

        final_df = df_news[['Article Title', 'Article URL', 'Article Content', 'Published Date', 'Sentiment Score']]

        final_df = final_df.drop_duplicates(subset=['Article URL'])
        
        print(f"Analysis complete. Saving {len(final_df)} unique articles to {OUTPUT_FILE}...")
        final_df.to_csv(OUTPUT_FILE, index=False)
        print("Done!")
    else:
        print("No news fetched.")
