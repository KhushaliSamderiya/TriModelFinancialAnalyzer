import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import argparse
import logging
import os

# Argument parser for specifying file paths
parser = argparse.ArgumentParser(description="Train a sentiment analysis model on preprocessed data.")
parser.add_argument("--train_file", type=str, required=True, help="Path to the training CSV file.")
parser.add_argument("--val_file", type=str, required=True, help="Path to the validation CSV file.")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model and logs.")
args = parser.parse_args()

# Set up logging to file
log_dir = os.path.join(args.save_path, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "training.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load training and validation data
train_data = pd.read_csv(args.train_file)
val_data = pd.read_csv(args.val_file)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Encode the labels (positive, negative, neutral)
label_mapping = {"positive": 0, "neutral": 1, "negative": 2}
train_data["label"] = train_data["label"].map(label_mapping)
val_data["label"] = val_data["label"].map(label_mapping)

# Dataset class for the data
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.texts = data["headline"].tolist()
        self.labels = data["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        inputs = {key: val.squeeze() for key, val in inputs.items()}  # Remove extra dimensions
        inputs["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

# Prepare datasets
train_dataset = SentimentDataset(train_data, tokenizer)
val_dataset = SentimentDataset(val_data, tokenizer)

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

# Training arguments
training_args = TrainingArguments(
    output_dir=args.save_path,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=log_dir,
    logging_steps=10,
    log_level="info",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Start training
logger.info("Starting training process...")
trainer.train()

# Save the model and tokenizer
model.save_pretrained(args.save_path)
tokenizer.save_pretrained(args.save_path)
logger.info(f"Model and tokenizer saved to {args.save_path}")
print(f"Model and logs saved to {args.save_path}")

