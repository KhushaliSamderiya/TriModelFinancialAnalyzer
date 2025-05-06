import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

# Argument parser
parser = argparse.ArgumentParser(description="Test sentiment model on test data and evaluate.")
parser.add_argument("--test_file", type=str, required=True, help="Path to the test CSV file.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing.")
args = parser.parse_args()

# Load the test data
test_data = pd.read_csv(args.test_file)

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(args.model_path)
model = BertForSequenceClassification.from_pretrained(args.model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare the test dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Encode test texts and labels
test_texts = test_data['headline'].tolist()
test_labels = test_data['label'].map({"positive": 2, "neutral": 1, "negative": 0}).tolist()
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Function to evaluate the model
def evaluate(model, loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels)

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Log metrics to a file
    with open(f"{args.model_path}/test_results.txt", "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")

# Run evaluation
evaluate(model, test_loader)

