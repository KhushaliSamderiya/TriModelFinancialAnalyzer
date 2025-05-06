import yfinance as yf
import pandas as pd
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Fetch fundamental data for Dow 30 stocks.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the data.")
args = parser.parse_args()

# List of Dow Jones 30 ticker symbols
dow30_tickers = ["AAPL", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", "GS",
                 "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
                 "NKE", "PFE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT", "XOM"]

# Fetch and save data for each ticker
for ticker in dow30_tickers:
    stock = yf.Ticker(ticker)
    
    # Get financial data
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    income_statement = stock.financials
    
    # Save each type of report as a CSV
    balance_sheet.to_csv(f"{args.output_dir}/{ticker}_balance_sheet.csv")
    cash_flow.to_csv(f"{args.output_dir}/{ticker}_cash_flow.csv")
    income_statement.to_csv(f"{args.output_dir}/{ticker}_income_statement.csv")

    print(f"Saved data for {ticker}")

