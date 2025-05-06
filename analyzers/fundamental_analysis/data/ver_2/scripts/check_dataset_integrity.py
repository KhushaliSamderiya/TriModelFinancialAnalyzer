import pandas as pd
import argparse
import os

def check_missing_values(df, set_name):
    """
    Check for missing values in a DataFrame and report the columns that contain them.
    """
    missing_columns = df.columns[df.isnull().any()].tolist()
    if missing_columns:
        print(f"[WARNING] {set_name} has missing values in columns: {missing_columns}")
    else:
        print(f"{set_name} has no missing values.")

def check_overlap(train_df, val_df, test_df, key_column='Risk Classification'):
    """
    Check for overlaps between train, val, and test sets based on a key column.
    """
    train_val_overlap = pd.merge(train_df, val_df, on=key_column, how='inner')
    train_test_overlap = pd.merge(train_df, test_df, on=key_column, how='inner')
    val_test_overlap = pd.merge(val_df, test_df, on=key_column, how='inner')

    if not train_val_overlap.empty:
        print("[ERROR] Overlap found between train and validation sets.")
    if not train_test_overlap.empty:
        print("[ERROR] Overlap found between train and test sets.")
    if not val_test_overlap.empty:
        print("[ERROR] Overlap found between validation and test sets.")
    
    if train_val_overlap.empty and train_test_overlap.empty and val_test_overlap.empty:
        print("No overlap detected between train, validation, and test sets.")

def check_columns(df, expected_columns, set_name):
    """
    Check if a DataFrame has the expected columns.
    """
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"[WARNING] {set_name} is missing columns: {missing_columns}")
    else:
        print(f"{set_name} has all expected columns.")

def main(train_file, val_file, test_file):
    # Expected columns based on the dataset schema
    expected_columns = ['Current Ratio', 'Debt-to-Equity Ratio', 'Revenue', 
                        'Net Income', 'Free Cash Flow', 'Risk Classification']
    
    # Load datasets
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    # Check for missing values
    check_missing_values(train_df, "Train set")
    check_missing_values(val_df, "Validation set")
    check_missing_values(test_df, "Test set")
    
    # Check for overlaps between datasets
    check_overlap(train_df, val_df, test_df, key_column='Risk Classification')
    
    # Check columns
    check_columns(train_df, expected_columns, "Train set")
    check_columns(val_df, expected_columns, "Validation set")
    check_columns(test_df, expected_columns, "Test set")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the integrity of train, validation, and test datasets.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation CSV file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test CSV file.")
    
    args = parser.parse_args()
    
    main(args.train_file, args.val_file, args.test_file)

