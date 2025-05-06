import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel  # Added BertModel import
import torch.nn as nn
import argparse
import os

# Custom Dataset class for test data
class FinancialDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.labels = {'Low Risk': 0, 'Moderate Risk': 1, 'High Risk': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = f"{row['Current Ratio']} {row['Debt-to-Equity Ratio']} {row['Revenue']} {row['Net Income']} {row['Free Cash Flow']}"
        inputs = self.tokenizer(features, padding='max_length', max_length=self.seq_len, return_tensors="pt", truncation=True)
        label = self.labels[row['Risk Classification']]
        return {**inputs, 'labels': torch.tensor(label)}

# Model class definition
class FinancialRiskClassifier(nn.Module):
    def __init__(self, seq_len, num_labels):
        super(FinancialRiskClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            labels = batch['labels']
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the financial risk classifier")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the saved model file (.pth)")
    parser.add_argument("--seq_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    # Load test data
    test_data = pd.read_csv(args.test_file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = FinancialDataset(test_data, tokenizer, args.seq_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Load the model
    model = FinancialRiskClassifier(seq_len=args.seq_len, num_labels=3)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    # Evaluate the model on test set
    evaluate_model(model, test_loader)

