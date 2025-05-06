import os
import pandas as pd
import yfinance as yf
import argparse

# Argument parser for specifying options
parser = argparse.ArgumentParser(description="Preprocess Dow Jones 30 stock data using Yahoo Finance.")
parser.add_argument("--output_file", type=str, default="dow_jones_5yr_data.csv", help="Path to save the preprocessed data.")
args = parser.parse_args()

# List of Dow Jones 30 company symbols
DOW_JONES_30 = [
    "AAPL", "MSFT", "JPM", "V", "UNH", "HD", "PG", "BAC", "DIS", "XOM",
    "CVX", "KO", "MRK", "WMT", "INTC", "CSCO", "PFE", "VZ", "AXP", "BA",
    "MMM", "CAT", "GS", "IBM", "JNJ", "NKE", "TRV", "WBA", "DOW", "CRM"
]

def fetch_stock_data(symbol):
    """
    Fetch daily stock data for a given symbol using Yahoo Finance.
    """
    try:
        data = yf.download(symbol, period="5y", interval="1d")
        data['Symbol'] = symbol
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def preprocess_data(output_file):
    """
    Retrieve and preprocess Dow Jones 30 stock data for the past 5 years.
    """
    all_data = []
    for symbol in DOW_JONES_30:
        print(f"Fetching data for {symbol}...")
        stock_data = fetch_stock_data(symbol)
        if stock_data is not None:
            # Reset index and rename columns
            stock_data.reset_index(inplace=True)
            stock_data.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Adj Close': 'Adj Close',
                'Volume': 'Volume'
            }, inplace=True)
            all_data.append(stock_data)

    # Concatenate all company data into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        # Handle missing values (e.g., fill or drop)
        combined_data.fillna(method='ffill', inplace=True)
        combined_data.to_csv(output_file, index=False)
        print(f"Data successfully saved to {output_file}")
    else:
        print("No data fetched. Please check your internet connection or symbols.")

if __name__ == "__main__":
    preprocess_data(output_file=args.output_file)
