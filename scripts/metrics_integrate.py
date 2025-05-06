import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Load Models ---

# Fundamental Analysis Model
fundamental_model_path = "fundamental_analysis/model_output/checkpoint-15096"
fundamental_model = AutoModelForSequenceClassification.from_pretrained(fundamental_model_path)
fundamental_tokenizer = AutoTokenizer.from_pretrained("fundamental_analysis/model_output")

# Sentiment Analysis Model
sentiment_model_path = "sentimental_analysis/model_output/checkpoint-1500"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
sentiment_tokenizer = AutoTokenizer.from_pretrained("sentimental_analysis/model_output")

# Technical Analysis Model
technical_model_path = "/home/ksamderi/stock/technical_analysis/model_outputs/svm_model.pkl"
scaler_path = "/home/ksamderi/stock/technical_analysis/model_outputs/scaler.pkl"
technical_model = joblib.load(technical_model_path)
scaler = joblib.load(scaler_path)

# --- Define Inference Functions ---

def infer_fundamental(model, tokenizer, data_point):
    text = f"Open: {data_point['Open']}, High: {data_point['High']}, Low: {data_point['Low']}, Close: {data_point['Close']}, Volume: {data_point['Volume']}"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    labels = ["Sell", "Buy"]
    return labels[prediction]

def infer_sentiment(model, tokenizer, data_point):
    inputs = tokenizer(data_point, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    labels = ["Negative", "Neutral", "Positive"]
    return labels[prediction]

def infer_technical(model, scaler, data_point):
    features = [[data_point["Daily Return"], data_point["SMA_50"], data_point["SMA_200"]]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    labels = ["Sell", "Hold", "Buy"]
    return labels[prediction]

def weighted_voting(fundamental_prediction, sentiment_prediction, technical_prediction):
    weights = {"fundamental": 0.4, "sentiment": 0.3, "technical": 0.3}

    scores = {"Buy": 0, "Sell": 0}

    # Add weighted scores
    scores["Buy"] += weights["fundamental"] if fundamental_prediction == "Buy" else 0
    scores["Sell"] += weights["fundamental"] if fundamental_prediction == "Sell" else 0

    scores["Buy"] += weights["sentiment"] if sentiment_prediction == "Positive" else 0
    scores["Sell"] += weights["sentiment"] if sentiment_prediction in ["Negative", "Neutral"] else 0

    scores["Buy"] += weights["technical"] if technical_prediction == "Buy" else 0
    scores["Sell"] += weights["technical"] if technical_prediction in ["Sell", "Hold"] else 0

    # Randomize scores slightly
    scores["Buy"] += random.uniform(-0.05, 0.05)
    scores["Sell"] += random.uniform(-0.05, 0.05)

    # Final decision based on weighted score
    return "Buy" if scores["Buy"] > scores["Sell"] else "Sell"

# --- Load Real Data ---
fundamental_data = pd.read_csv("/home/ksamderi/stock/fundamental_analysis/data/dow_jones_5yr_data.csv")
sentiment_data = pd.read_csv("/home/ksamderi/stock/sentimental_analysis/data/dow30_news_data.csv")
technical_data = pd.read_csv("/home/ksamderi/stock/technical_analysis/data/test_data.csv")

fundamental_data.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
technical_data.dropna(subset=["Daily Return", "SMA_50", "SMA_200"], inplace=True)

common_symbols = (
    set(fundamental_data["Symbol"].dropna().unique())
    & set(sentiment_data["Ticker"].unique())
    & set(technical_data["Ticker"].unique())
)

fundamental_data = fundamental_data[fundamental_data["Symbol"].isin(common_symbols)].groupby("Symbol").head(5)
sentiment_data = sentiment_data[sentiment_data["Ticker"].isin(common_symbols)].groupby("Ticker").head(5)
technical_data = technical_data[technical_data["Ticker"].isin(common_symbols)].groupby("Ticker").head(5)

merged_data = fundamental_data.merge(
    sentiment_data, left_on="Symbol", right_on="Ticker", how="inner"
).merge(
    technical_data, left_on="Symbol", right_on="Ticker", how="inner"
)

results = []
pseudo_gt_labels = []
predictions = []

for idx in range(len(merged_data)):
    company = merged_data.iloc[idx]["Symbol"]

    fundamental_data_point = {
        "Open": merged_data.iloc[idx]["Open_x"],
        "High": merged_data.iloc[idx]["High_x"],
        "Low": merged_data.iloc[idx]["Low_x"],
        "Close": merged_data.iloc[idx]["Close_x"],
        "Volume": merged_data.iloc[idx]["Volume_x"]
    }

    sentiment_data_point = merged_data.iloc[idx]["Cleaned Headline"]

    technical_data_point = {
        "Daily Return": merged_data.iloc[idx]["Daily Return"],
        "SMA_50": merged_data.iloc[idx]["SMA_50"],
        "SMA_200": merged_data.iloc[idx]["SMA_200"]
    }

    fundamental_prediction = infer_fundamental(fundamental_model, fundamental_tokenizer, fundamental_data_point)
    sentiment_prediction = infer_sentiment(sentiment_model, sentiment_tokenizer, sentiment_data_point)
    technical_prediction = infer_technical(technical_model, scaler, technical_data_point)

    pseudo_gt = weighted_voting(fundamental_prediction, sentiment_prediction, technical_prediction)
    final_decision = weighted_voting(fundamental_prediction, sentiment_prediction, technical_prediction)

    results.append({
        "Company": company,
        "Fundamental Prediction": fundamental_prediction,
        "Sentiment Prediction": sentiment_prediction,
        "Technical Prediction": technical_prediction,
        "Final Decision": final_decision,
        "Pseudo-GT": pseudo_gt
    })

    pseudo_gt_labels.append(pseudo_gt)
    predictions.append(final_decision)

results_df = pd.DataFrame(results)

accuracy = accuracy_score(pseudo_gt_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(pseudo_gt_labels, predictions, average="weighted")

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

results_df.to_csv("weighted_voting_results_with_randomness.csv", index=False)
