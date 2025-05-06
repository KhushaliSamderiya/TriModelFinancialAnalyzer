import pandas as pd
import yfinance as yf

# Function to retrieve and display available keys for a given ticker
def list_keys(tickers):
    unique_keys = {
        "balance_sheet": set(),
        "income_statement": set(),
        "cash_flow": set()
    }
    
    for ticker in tickers:
        print(f"Processing {ticker}...")

        # Fetch data
        stock = yf.Ticker(ticker)
        
        # Get balance sheet, income statement, and cash flow data
        try:
            balance_sheet_data = stock.get_balance_sheet()
            income_statement_data = stock.get_financials()
            cash_flow_data = stock.get_cashflow()

            # Collect unique keys for each data type
            if isinstance(balance_sheet_data, pd.DataFrame):
                unique_keys["balance_sheet"].update(balance_sheet_data.index.tolist())
            if isinstance(income_statement_data, pd.DataFrame):
                unique_keys["income_statement"].update(income_statement_data.index.tolist())
            if isinstance(cash_flow_data, pd.DataFrame):
                unique_keys["cash_flow"].update(cash_flow_data.index.tolist())
        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Output all unique keys found
    print("\nUnique Balance Sheet Keys:")
    print(sorted(unique_keys["balance_sheet"]))

    print("\nUnique Income Statement Keys:")
    print(sorted(unique_keys["income_statement"]))

    print("\nUnique Cash Flow Statement Keys:")
    print(sorted(unique_keys["cash_flow"]))


# Dow Jones 30 tickers
dow30_tickers = [
    "AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "PG", "DIS", "MA", "UNH", "VZ", 
    "INTC", "KO", "PFE", "MRK", "CVX", "XOM", "BA", "MCD", "IBM", "GS", "CAT", 
    "MMM", "TRV", "AXP", "WBA", "DOW", "CSCO", "NKE", "HD"
]

# Run the key listing function
list_keys(dow30_tickers)

