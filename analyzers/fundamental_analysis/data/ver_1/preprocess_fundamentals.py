import argparse
import os
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser(description="Preprocess fundamental data by combining balance sheet, income statement, and cash flow for each ticker.")
parser.add_argument("--input_dir", type=str, required=True, help="Directory with individual fundamental CSV files (balance sheet, income statement, cash flow).")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the combined preprocessed data for each ticker.")
args = parser.parse_args()

# Ensure output directory exists
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Get list of tickers by extracting unique ticker names from file names
tickers = set()
for file_name in os.listdir(args.input_dir):
    if file_name.endswith('.csv'):
        ticker = file_name.split('_')[0]  # assumes format TICKER_type.csv
        tickers.add(ticker)

# Combine balance sheet, income statement, and cash flow for each ticker
for ticker in tickers:
    try:
        # Load individual files for each ticker
        balance_sheet_path = os.path.join(args.input_dir, f"{ticker}_balance_sheet.csv")
        income_statement_path = os.path.join(args.input_dir, f"{ticker}_income_statement.csv")
        cash_flow_path = os.path.join(args.input_dir, f"{ticker}_cash_flow.csv")

        # Check if all required files exist for this ticker
        if not (os.path.exists(balance_sheet_path) and os.path.exists(income_statement_path) and os.path.exists(cash_flow_path)):
            print(f"Missing one or more files for {ticker}. Skipping this company.")
            continue

        # Load data
        balance_sheet = pd.read_csv(balance_sheet_path)
        income_statement = pd.read_csv(income_statement_path)
        cash_flow = pd.read_csv(cash_flow_path)

        # Merge data on common columns, e.g., 'Date'
        company_data = pd.concat([balance_sheet, income_statement, cash_flow], axis=1)

        # Save the combined data for this ticker
        output_path = os.path.join(args.output_dir, f"{ticker}_combined_data.csv")
        company_data.to_csv(output_path, index=False)
        print(f"Saved combined data for {ticker} to {output_path}")

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

print("Preprocessing complete. Combined data saved by ticker.")

