import yfinance as yf
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

# Argument parser for user input
parser = argparse.ArgumentParser(description="Preprocess stock data for technical analysis.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data.")
parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data for testing.")
args = parser.parse_args()

# Create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Define the Dow Jones 30 tickers
dow30_tickers = [
    "AAPL", "MSFT", "JNJ", "V", "WMT", "JPM", "PG", "UNH", "HD", "MA",
    "DIS", "MRK", "VZ", "CVX", "KO", "PFE", "PEP", "INTC", "CSCO", "TRV",
    "MCD", "GS", "AXP", "IBM", "NKE", "BA", "CAT", "MMM", "WBA", "DOW"
]

# Download data for the last 10 years
print("Downloading stock data...")
stock_data = yf.download(dow30_tickers, start="2013-01-01", end="2023-12-31", group_by="ticker")
stock_data = stock_data.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index()

# Save raw data
raw_data_path = os.path.join(args.output_dir, "raw_stock_data.csv")
stock_data.to_csv(raw_data_path, index=False)
print(f"Raw stock data saved to {raw_data_path}")

# Feature engineering
print("Performing feature engineering...")
stock_data['Daily Return'] = stock_data.groupby('Ticker')['Adj Close'].transform(lambda x: x.pct_change())
stock_data['SMA_50'] = stock_data.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=50).mean())
stock_data['SMA_200'] = stock_data.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=200).mean())

# Drop rows with missing values due to rolling window calculations
stock_data.dropna(inplace=True)

# Save processed data
processed_data_path = os.path.join(args.output_dir, "processed_stock_data.csv")
stock_data.to_csv(processed_data_path, index=False)
print(f"Processed stock data saved to {processed_data_path}")

# Prepare train and test splits
print("Splitting data into train and test sets...")
train_data, test_data = train_test_split(stock_data, test_size=args.test_size, shuffle=False)

# Save train and test data
train_data_path = os.path.join(args.output_dir, "train_data.csv")
test_data_path = os.path.join(args.output_dir, "test_data.csv")

train_data.to_csv(train_data_path, index=False)
test_data.to_csv(test_data_path, index=False)

print(f"Train data saved to {train_data_path}")
print(f"Test data saved to {test_data_path}")
