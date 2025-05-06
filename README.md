📊 TriModelFinancialAnalyzer
An integrated machine learning system that predicts Buy or Sell stock recommendations using insights from fundamental analysis, technical indicators, and sentiment analysis — then combines them using a voting system.

Domains: Finance, NLP, Time Series
Models Used: BERT (Transformer), SVM (Scikit-learn)
Data Sources: Yahoo Finance, Financial News

🧠 Project Structure
bash
Copy
Edit
TriModelFinancialAnalyzer/
├── analyzers/
│   ├── fundamental_analysis/
│   ├── technical_analysis/
│   └── sentimental_analysis/
├── scripts/
└── README.md
Each analyzer has its own pipeline for data processing, training, and testing. The scripts/ folder integrates all models into a decision-making system.

🧭 Full Pipeline Overview
🔹 Step 1: Generate Datasets
1.1 Fundamentals (Financial Ratios + Risk)
bash
Copy
Edit
python analyzers/fundamental_analysis/data/ver_2/scripts/preprocess_fundamentals.py \
    --start_date 2018-01-01 \
    --end_date 2023-12-31 \
    --output_file dow_jones_5yr_data.csv
✔ Fetches financial data for Dow 30 companies
✔ Computes key ratios (Current Ratio, Debt-to-Equity, etc.)
✔ Assigns Risk Classification (Low, Moderate, High)

1.2 Sentiment (News Headlines)
bash
Copy
Edit
python analyzers/sentimental_analysis/data/extract_news.py \
    --output_file dow30_news_data.csv
✔ Collects recent headlines using yfinance
✔ Maps news to companies

1.3 Technical (Price-based Indicators)
bash
Copy
Edit
python analyzers/technical_analysis/data/preprocess_technical_data.py \
    --output_dir analyzers/technical_analysis/data/processed \
    --test_size 0.2
✔ Downloads 10 years of price data
✔ Computes daily return, SMA-50, SMA-200
✔ Outputs train_data.csv and test_data.csv

🔹 Step 2: Preprocess & Clean
Fundamental Risk Labels (Clean Missing)
bash
Copy
Edit
python analyzers/fundamental_analysis/model/ver_2/scripts/clean_labels.py \
    --input_file dow_jones_5yr_data.csv \
    --output_file cleaned_fundamental.csv
Sentiment Headlines + Labels
bash
Copy
Edit
python analyzers/sentimental_analysis/data/preprocess.py \
    --input_file dow30_news_data.csv \
    --train_file train.csv \
    --val_file val.csv \
    --test_file test.csv
✔ Applies Hugging Face pipeline("sentiment-analysis")
✔ Adds labels: positive, neutral, negative
✔ Splits dataset 70/20/10

🤖 Train Models
📘 Fundamental Model (Transformer-based Risk Classifier)
bash
Copy
Edit
python analyzers/fundamental_analysis/model/ver_2/scripts/train_transformer_model.py \
    --train_file train_data.csv \
    --val_file val_data.csv \
    --test_file test_data.csv \
    --output_dir fundamental_analysis/model_output
✔ Trains a BERT model on financial ratios
✔ Outputs: financial_risk_classifier.pth

📰 Sentiment Model (News Headline Classifier)
bash
Copy
Edit
python analyzers/sentimental_analysis/models/train_sentiment_model.py \
    --train_file train.csv \
    --val_file val.csv \
    --save_path sentimental_analysis/model_output
✔ Trains BERT for 3-class sentiment classification
✔ Saves model & tokenizer

📈 Technical Model (SVM)
bash
Copy
Edit
python analyzers/technical_analysis/models/train_svm.py \
    --train_file train_data.csv \
    --test_file test_data.csv \
    --output_dir technical_analysis/model_outputs
✔ SVM classifier on SMA + return
✔ Classifies price movements as Buy, Hold, or Sell

🔁 Integration & Inference
🧩 Run Inference on Sample
bash
Copy
Edit
python scripts/integrate.py
✔ Loads all 3 models
✔ Runs prediction on a single synthetic example
✔ Prints each model’s output and final voted decision

🧮 Batch Inference on Real Data
bash
Copy
Edit
python scripts/inference_final.py
✔ Loads cleaned datasets
✔ Aligns by Ticker/Symbol
✔ Runs inference for multiple companies
✔ Applies majority vote across models
✔ Outputs:

voting_results_limited.csv: Row-wise results

company_analysis.csv: Final Buy/Sell recommendation per company

📊 Metrics & Evaluation
🧪 Evaluate Integration Logic with Pseudo Ground Truth
bash
Copy
Edit
python scripts/metrics_integrate.py
✔ Uses weighted voting + noise to simulate real-world uncertainty
✔ Computes:

Accuracy

Precision

Recall

F1 Score
✔ Saves predictions to:

weighted_voting_results_with_randomness.csv

🧠 Why This Matters
Tri-modal prediction: Incorporates fundamental, technical, and sentiment signals

Transformer + Traditional ML: Combines modern deep learning with classical SVMs

Modular & Extensible: Each pipeline is cleanly separated but integratable

Voting Logic: Emulates ensemble decision making for robustness

💡 Final Thoughts
Each domain contributes unique insights:

Fundamental = Intrinsic financial health

Sentiment = Market perception

Technical = Price movement trends

Together, they create a more holistic and reliable stock prediction engine.

✅ Recommended Run Order
bash
Copy
Edit
# 1. Data generation
preprocess_fundamentals.py
extract_news.py
preprocess_technical_data.py

# 2. Cleaning & preprocessing
clean_labels.py
preprocess.py

# 3. Training models
train_transformer_model.py
train_sentiment_model.py
train_svm.py

# 4. Integration
inference_final.py

# 5. Evaluation
metrics_integrate.py
