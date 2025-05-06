import pandas as pd
import numpy as np
import argparse
import os

def fill_missing_values(df):
    """
    Fill missing values in each column with random values within the column's min-max range.
    """
    for column in df.columns:
        if df[column].isnull().any():
            min_val = df[column].min()
            max_val = df[column].max()
            df[column].fillna(np.random.uniform(min_val, max_val), inplace=True)
    return df

def process_and_save_file(file_path):
    """
    Reads a CSV file, fills missing values, and overwrites the file.
    """
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    filled_df = fill_missing_values(df)
    filled_df.to_csv(file_path, index=False)
    print(f"Completed filling missing values for {file_path}")

def main(train_file, val_file, test_file):
    process_and_save_file(train_file)
    process_and_save_file(val_file)
    process_and_save_file(test_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill missing values in train, validation, and test CSV files.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation CSV file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test CSV file.")
    
    args = parser.parse_args()
    
    main(args.train_file, args.val_file, args.test_file)

