import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import argparse
import os

def preprocess_data(input_file, output_dir, train_ratio, val_ratio, test_ratio):
    # Load data
    data = pd.read_csv(input_file)
    print("Loaded data from:", input_file)
    
    # Display columns to verify column names
    print("Columns in the dataset:", data.columns.tolist())
    
    # Drop non-numeric columns (e.g., 'Ticker' and 'Date')
    data = data.drop(columns=['Ticker', 'Date'], errors='ignore')
    
    # Handle the 'Risk Classification' column, filling any missing values with the most common class
    label_column = 'Risk Classification' if 'Risk Classification' in data.columns else 'label'
    if data[label_column].isnull().any():
        data[label_column].fillna(data[label_column].mode()[0], inplace=True)
    
    # Encode the 'Risk Classification' column to numerical values
    label_encoder = LabelEncoder()
    data[label_column] = label_encoder.fit_transform(data[label_column].astype(str))
    
    # Fill any remaining missing values in features with column means
    data.fillna(data.mean(), inplace=True)
    
    # Verify no missing values remain
    assert not data.isnull().any().any(), "There are still missing values in the dataset after processing."
    
    # Separate features and labels
    X = data.drop(columns=[label_column])
    y = data[label_column]
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Apply SMOTE to balance classes
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("SMOTE applied. Class distribution after resampling:")
    print(pd.Series(y_resampled).value_counts(normalize=True) * 100)

    # Stratified split into train, val, and test sets
    train_size = train_ratio
    val_size = val_ratio / (val_ratio + test_ratio)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resampled, y_resampled, train_size=train_size, random_state=42, stratify=y_resampled)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_size, random_state=42, stratify=y_temp)
    
    # Check and display class distribution in each split
    def display_class_balance(label_data, split_name):
        print(f"Class distribution in {split_name} set:")
        class_counts = pd.Series(label_data).value_counts(normalize=True) * 100
        print(class_counts)
    
    display_class_balance(y_train, "training")
    display_class_balance(y_val, "validation")
    display_class_balance(y_test, "test")
    
    # Convert to DataFrames
    feature_columns = data.columns[:-1]  # Exclude the label column
    train_df = pd.DataFrame(X_train, columns=feature_columns)
    train_df[label_column] = y_train
    val_df = pd.DataFrame(X_val, columns=feature_columns)
    val_df[label_column] = y_val
    test_df = pd.DataFrame(X_test, columns=feature_columns)
    test_df[label_column] = y_test
    
    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

    print("Preprocessing complete. Files saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess financial dataset for model training.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed files.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio.")
    
    args = parser.parse_args()
    
    preprocess_data(args.input_file, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)

