import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser(description="Train a transformer model for fundamental analysis.")
parser.add_argument("--data_file", type=str, required=True, help="Path to the preprocessed CSV file.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
parser.add_argument("--seq_len", type=int, default=128, help="Maximum sequence length for the model.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
args = parser.parse_args()

# Dataset class
class StockDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len, task="classification"):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Combine features into a single text representation
        text = f"Open: {row['Open']}, High: {row['High']}, Low: {row['Low']}, Close: {row['Close']}, Volume: {row['Volume']}"
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.seq_len, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Remove extra dimension

        # Define label
        if self.task == "classification":
            label = row["Target"]
        else:  # For regression tasks
            label = row["Close"]

        inputs["labels"] = torch.tensor(label, dtype=torch.float if self.task == "regression" else torch.long)
        return inputs

# Evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Load and preprocess data
data = pd.read_csv(args.data_file, low_memory=False)

# Convert numeric columns to proper format
for col in ["Open", "High", "Low", "Close", "Volume"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")  # Convert and handle errors
data.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)  # Remove rows with invalid data

# Define a target column for classification (e.g., predicting "up" or "down" based on Close price)
data["Target"] = (data["Close"].pct_change().fillna(0) > 0).astype(int)

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Target"])

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create datasets
train_dataset = StockDataset(train_data, tokenizer, seq_len=args.seq_len, task="classification")
test_dataset = StockDataset(test_data, tokenizer, seq_len=args.seq_len, task="classification")

# Initialize model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    logging_dir=f"{args.output_dir}/logs",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save model and tokenizer
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"Model and tokenizer saved to {args.output_dir}")
