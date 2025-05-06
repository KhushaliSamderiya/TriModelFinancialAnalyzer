import pandas as pd
import argparse

def clean_labels(input_file, output_file):
    # Load the CSV
    data = pd.read_csv(input_file)
    
    # Fill NaNs in the 'Risk Classification' column with a default value ('Low Risk')
    data['Risk Classification'].fillna('Low Risk', inplace=True)

    # Save the cleaned file
    data.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean labels in dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save cleaned CSV")
    args = parser.parse_args()

    clean_labels(args.input_file, args.output_file)

