import yfinance as yf
import pandas as pd
import argparse

# Argument parser for specifying the output file path
parser = argparse.ArgumentParser(description="Extract news data for Dow Jones companies.")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the extracted news data.")
args = parser.parse_args()

# Define the tickers for Dow Jones companies
tickers = ["AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "PG", "DIS", "HD", "UNH"]  # Add more as needed

# List to collect news data
all_news = []

# Fetch news for each ticker
for ticker in tickers:
    stock = yf.Ticker(ticker)
    news = stock.news  # This fetches recent news articles

    for article in news:
        all_news.append({
            "Date": article["providerPublishTime"],
            "Ticker": ticker,
            "Headline": article["title"]
        })

# Convert to DataFrame
news_data = pd.DataFrame(all_news)
news_data['Date'] = pd.to_datetime(news_data['Date'], unit='s').dt.date  # Convert UNIX time to readable format

# Save the data to the specified output file
news_data.to_csv(args.output_file, index=False)
print(f"News data saved to {args.output_file}")

