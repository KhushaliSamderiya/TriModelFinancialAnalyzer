import yfinance as yf

def fetch_available_keys(ticker):
    stock = yf.Ticker(ticker)
    
    # Fetch financial data
    try:
        balance_sheet = stock.balance_sheet
        income_statement = stock.financials
        cash_flow = stock.cashflow

        print(f"\nAvailable keys for {ticker}:\n")

        if not balance_sheet.empty:
            print("Balance Sheet Keys:")
            for key in balance_sheet.index:
                print(f"  - {key}")
        else:
            print("No balance sheet data found.")
        
        if not income_statement.empty:
            print("\nIncome Statement Keys:")
            for key in income_statement.index:
                print(f"  - {key}")
        else:
            print("No income statement data found.")
        
        if not cash_flow.empty:
            print("\nCash Flow Statement Keys:")
            for key in cash_flow.index:
                print(f"  - {key}")
        else:
            print("No cash flow data found.")

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

if __name__ == "__main__":
    # Replace 'AAPL' with the ticker you want to check
    ticker = 'AAPL'
    fetch_available_keys(ticker)

