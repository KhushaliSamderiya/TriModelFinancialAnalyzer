import pandas as pd
import argparse

def check_class_balance(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Display the count of each class
    class_counts = data['Risk Classification'].value_counts()
    print(f"Class balance for {file_path}:")
    print(class_counts)
    print("\nClass distribution as percentages:")
    print((class_counts / len(data)) * 100)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Check class balance for datasets.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the train dataset CSV file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation dataset CSV file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test dataset CSV file")
    
    args = parser.parse_args()
    
    # Check balance for each dataset
    print("Checking class balance for train set:")
    check_class_balance(args.train_file)
    
    print("\nChecking class balance for validation set:")
    check_class_balance(args.val_file)
    
    print("\nChecking class balance for test set:")
    check_class_balance(args.test_file)

