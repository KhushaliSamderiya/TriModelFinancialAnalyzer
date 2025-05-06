import pandas as pd
import argparse

def remove_overlap(train_df, val_df, test_df, key_column='Risk Classification'):
    """
    Remove overlaps between train, val, and test sets based on a key column.
    """
    # Remove overlaps between train and validation
    val_df = val_df[~val_df[key_column].isin(train_df[key_column])]
    test_df = test_df[~test_df[key_column].isin(train_df[key_column])]
    
    # Remove remaining overlaps between validation and test
    test_df = test_df[~test_df[key_column].isin(val_df[key_column])]

    return train_df, val_df, test_df

def main(train_file, val_file, test_file, output_dir):
    # Load datasets
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    # Remove overlaps
    train_df, val_df, test_df = remove_overlap(train_df, val_df, test_df, key_column='Risk Classification')
    
    # Save cleaned datasets
    train_df.to_csv(f"{output_dir}/train_data_no_overlap.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_data_no_overlap.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_data_no_overlap.csv", index=False)
    
    print("Overlap removed and files saved to output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove overlap from train, val, and test datasets.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation CSV file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    
    args = parser.parse_args()
    
    main(args.train_file, args.val_file, args.test_file, args.output_dir)

