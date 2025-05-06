from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import joblib
from collections import Counter

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

# Fundamental Analysis Inference
def infer_fundamental(model, tokenizer, data_point):
    text = f"Open: {data_point['Open']}, High: {data_point['High']}, Low: {data_point['Low']}, Close: {data_point['Close']}, Volume: {data_point['Volume']}"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    labels = ["Sell", "Buy"]
    return labels[prediction]

# Sentiment Analysis Inference
def infer_sentiment(model, tokenizer, data_point):
    inputs = tokenizer(data_point, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    labels = ["Negative", "Neutral", "Positive"]
    return labels[prediction]

# Technical Analysis Inference
def infer_technical(model, scaler, data_point):
    features = [[data_point["Daily Return"], data_point["SMA_50"], data_point["SMA_200"]]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    labels = ["Sell", "Hold", "Buy"]
    return labels[prediction]

# --- Voting System ---
def voting_system(fundamental_prediction, sentiment_prediction, technical_prediction):
    def map_to_common_space(output, model_type):
        if model_type == "fundamental":
            return 1 if output == "Buy" else 0
        elif model_type == "sentiment":
            return 1 if output == "Positive" else 0
        elif model_type == "technical":
            return 1 if output == "Buy" else 0
        return 0  # Default to "Sell"

    votes = [
        map_to_common_space(fundamental_prediction, "fundamental"),
        map_to_common_space(sentiment_prediction, "sentiment"),
        map_to_common_space(technical_prediction, "technical")
    ]
    vote_counts = Counter(votes)
    final_vote = 1 if vote_counts[1] > vote_counts[0] else 0
    return "Buy" if final_vote == 1 else "Sell"

# --- Run Inference ---
fundamental_data_point = {
    "Open": 100.5,
    "High": 105.2,
    "Low": 98.7,
    "Close": 104.0,
    "Volume": 15000
}
sentiment_data_point = "Investors are optimistic about the company's performance."
technical_data_point = {
    "Daily Return": 0.03,
    "SMA_50": 102.0,
    "SMA_200": 101.5
}

# Generate Predictions
fundamental_prediction = infer_fundamental(fundamental_model, fundamental_tokenizer, fundamental_data_point)
sentiment_prediction = infer_sentiment(sentiment_model, sentiment_tokenizer, sentiment_data_point)
technical_prediction = infer_technical(technical_model, scaler, technical_data_point)

# Perform Voting
final_decision = voting_system(fundamental_prediction, sentiment_prediction, technical_prediction)

# Output Results
print("Fundamental Prediction:", fundamental_prediction)
print("Sentiment Prediction:", sentiment_prediction)
print("Technical Prediction:", technical_prediction)
print("Final Decision (Voting):", final_decision)
