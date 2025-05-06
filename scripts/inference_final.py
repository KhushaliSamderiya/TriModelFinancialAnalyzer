from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
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

# --- Load Real Data ---

# Load data files
fundamental_data = pd.read_csv("/home/ksamderi/stock/fundamental_analysis/data/dow_jones_5yr_data.csv")
sentiment_data = pd.read_csv("/home/ksamderi/stock/sentimental_analysis/data/dow30_news_data.csv")
technical_data = pd.read_csv("/home/ksamderi/stock/technical_analysis/data/test_data.csv")

# Ensure all datasets have valid rows
fundamental_data.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
technical_data.dropna(subset=["Daily Return", "SMA_50", "SMA_200"], inplace=True)

# --- Align Datasets by Common Symbols ---
common_symbols = (
    set(fundamental_data["Symbol"].dropna().unique()) &
    set(sentiment_data["Ticker"].unique()) &
    set(technical_data["Ticker"].unique())
)

print(f"Common symbols across datasets: {common_symbols}")

# Filter datasets to include only rows with common symbols
fundamental_data = fundamental_data[fundamental_data["Symbol"].isin(common_symbols)]
sentiment_data = sentiment_data[sentiment_data["Ticker"].isin(common_symbols)]
technical_data = technical_data[technical_data["Ticker"].isin(common_symbols)]

# Reset indices for alignment
fundamental_data.reset_index(drop=True, inplace=True)
sentiment_data.reset_index(drop=True, inplace=True)
technical_data.reset_index(drop=True, inplace=True)

# --- Process Limited Data Points ---
LIMIT = min(len(fundamental_data), len(sentiment_data), len(technical_data))
results = []

for idx in range(LIMIT):
    # Extract the company symbol
    company = fundamental_data.iloc[idx]["Symbol"]

    # Fundamental data
    fundamental_data_point = {
        "Open": fundamental_data.iloc[idx]["Open"],
        "High": fundamental_data.iloc[idx]["High"],
        "Low": fundamental_data.iloc[idx]["Low"],
        "Close": fundamental_data.iloc[idx]["Close"],
        "Volume": fundamental_data.iloc[idx]["Volume"]
    }

    # Sentiment data
    sentiment_data_point = sentiment_data.iloc[idx]["Cleaned Headline"]

    # Technical data
    technical_data_point = {
        "Daily Return": technical_data.iloc[idx]["Daily Return"],
        "SMA_50": technical_data.iloc[idx]["SMA_50"],
        "SMA_200": technical_data.iloc[idx]["SMA_200"]
    }

    # Generate Predictions
    fundamental_prediction = infer_fundamental(fundamental_model, fundamental_tokenizer, fundamental_data_point)
    sentiment_prediction = infer_sentiment(sentiment_model, sentiment_tokenizer, sentiment_data_point)
    technical_prediction = infer_technical(technical_model, scaler, technical_data_point)

    # Perform Voting
    final_decision = voting_system(fundamental_prediction, sentiment_prediction, technical_prediction)

    # Append result
    results.append({
        "Company": company,
        "Fundamental Prediction": fundamental_prediction,
        "Sentiment Prediction": sentiment_prediction,
        "Technical Prediction": technical_prediction,
        "Final Decision": final_decision
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# --- Company-wise Analysis ---
company_analysis = results_df.groupby("Company").agg(
    Buy_Count=("Final Decision", lambda x: (x == "Buy").sum()),
    Sell_Count=("Final Decision", lambda x: (x == "Sell").sum()),
    Total=("Final Decision", "count"),
).reset_index()

# Add a final recommendation for each company
def recommend_action(row):
    if row["Buy_Count"] > row["Sell_Count"]:
        return "Buy"
    elif row["Sell_Count"] > row["Buy_Count"]:
        return "Sell"
    else:
        return "Hold"

company_analysis["Recommendation"] = company_analysis.apply(recommend_action, axis=1)

# Save results to CSV
results_df.to_csv("voting_results_limited.csv", index=False)
company_analysis.to_csv("company_analysis.csv", index=False)

# Output Results
print("Company-Wise Analysis")
print(company_analysis.head())
