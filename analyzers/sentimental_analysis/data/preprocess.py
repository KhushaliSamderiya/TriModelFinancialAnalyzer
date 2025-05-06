import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from transformers import pipeline

# Argument parser for specifying file paths
parser = argparse.ArgumentParser(description="Preprocess, add sentiment labels, and split Dow Jones 30 news data.")
parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file with Dow Jones news data.")
parser.add_argument("--train_file", type=str, required=True, help="Path to save the training CSV file.")
parser.add_argument("--val_file", type=str, required=True, help="Path to save the validation CSV file.")
parser.add_argument("--test_file", type=str, required=True, help="Path to save the test CSV file.")
args = parser.parse_args()

# Load the data
data = pd.read_csv(args.input_file)

# Adjust column name to the actual column containing headlines
# Replace 'actual_column_name' with the column name that holds the news headlines
data['headline'] = data['Headline'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Add sentiment labels
data['label'] = data['headline'].apply(lambda x: sentiment_analyzer(x)[0]['label'].lower())

# Split the data into 70% train, 20% validation, and 10% test without overlap
train_data, temp_data = train_test_split(data, test_size=0.3, shuffle=True, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.33, shuffle=True, random_state=42)

# Save the splits
train_data.to_csv(args.train_file, index=False)
val_data.to_csv(args.val_file, index=False)
test_data.to_csv(args.test_file, index=False)

print("Data with sentiment labels split and saved:")
print(f"Training data: {args.train_file}")
print(f"Validation data: {args.val_file}")
print(f"Test data: {args.test_file}")

