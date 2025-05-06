import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm
import argparse
import os

# Custom Dataset
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
        label_str = str(row['Risk Classification']).strip()
        label = self.labels[label_str] if label_str in self.labels else 0  # default to 0 if label is invalid
        return {**inputs, 'labels': torch.tensor(label)}

# Model Definition
class FinancialRiskClassifier(nn.Module):
    def __init__(self, seq_len, num_labels):
        super(FinancialRiskClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

# Training Function with Progress Bar
def train_model(model, train_loader, val_loader, optimizer, epochs=3):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].squeeze(1)
            attention_mask = batch['attention_mask'].squeeze(1)
            labels = batch['labels']
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a financial risk classifier")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model")
    parser.add_argument("--seq_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()

    # Load Data
    train_data = pd.read_csv(args.train_file)
    val_data = pd.read_csv(args.val_file)
    test_data = pd.read_csv(args.test_file)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create Datasets
    train_dataset = FinancialDataset(train_data, tokenizer, args.seq_len)
    val_dataset = FinancialDataset(val_data, tokenizer, args.seq_len)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize Model
    model = FinancialRiskClassifier(seq_len=args.seq_len, num_labels=3)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Train Model
    train_model(model, train_loader, val_loader, optimizer, epochs=args.epochs)

    # Save Model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "financial_risk_classifier.pth"))
    print("Model saved successfully.")

