import os
import pandas as pd

def extract_key_financial_metrics(input_dir, summary_dir, metrics_file):
    key_metrics = ["Total Revenue", "Net Income", "Free Cash Flow", "Operating Income", "Gross Profit"]
    os.makedirs(summary_dir, exist_ok=True)

    evaluation_log = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith("_combined_data.csv"):
            ticker = filename.split("_")[0]
            filepath = os.path.join(input_dir, filename)
            df = pd.read_csv(filepath)
            
            summary_data = {}
            for metric in key_metrics:
                if metric in df.columns:
                    summary_data[metric] = df[metric].iloc[-1]
                else:
                    summary_data[metric] = "N/A"
            
            summary_text = f"Financial Summary for {ticker}:\n"
            for metric, value in summary_data.items():
                summary_text += f"{metric}: {value}\n"
            
            summary_filepath = os.path.join(summary_dir, f"{ticker}_summary.txt")
            with open(summary_filepath, "w") as f:
                f.write(summary_text)

            # Evaluate completeness for the ticker
            extracted_metrics = sum(1 for value in summary_data.values() if value != "N/A")
            evaluation_log.append({
                "Ticker": ticker,
                "Extracted_Metrics": extracted_metrics,
                "Missing_Metrics": len(key_metrics) - extracted_metrics
            })
            
            print(f"Generated summary for {ticker} and saved to {summary_filepath}")

    # Save evaluation metrics
    metrics_df = pd.DataFrame(evaluation_log)
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Evaluation metrics saved to {metrics_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract key financial metrics and evaluate")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with preprocessed data files")
    parser.add_argument("--summary_dir", type=str, required=True, help="Directory to save generated summaries")
    parser.add_argument("--metrics_file", type=str, required=True, help="File to save evaluation metrics")
    args = parser.parse_args()
    
    extract_key_financial_metrics(args.input_dir, args.summary_dir, args.metrics_file)

