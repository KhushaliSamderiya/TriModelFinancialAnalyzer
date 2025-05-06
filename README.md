# 📊 TriModelFinancialAnalyzer

An integrated machine learning system that predicts **Buy** or **Sell** stock recommendations using insights from:

- 🧾 **Fundamental Analysis**  
- 💹 **Technical Indicators**  
- 📰 **Sentiment Analysis**

These predictions are combined using a **majority voting ensemble**.

> **Domains:** Finance, NLP, Time Series  
> **Models Used:** BERT (Transformers), SVM (Scikit-learn)  
> **Data Sources:** Yahoo Finance, Financial News

---

## 🧠 Project Structure

```
TriModelFinancialAnalyzer/
├── analyzers/
│   ├── fundamental_analysis/
│   ├── technical_analysis/
│   └── sentimental_analysis/
├── scripts/
└── README.md
```

Each analyzer contains its own preprocessing, training, and testing pipeline. The `scripts/` folder integrates all models into a unified decision-making system.

---

## 🧭 Full Pipeline Overview

### 🔹 Step 1: Generate Datasets

#### 1.1 Fundamentals (Financial Ratios + Risk)

```bash
python analyzers/fundamental_analysis/data/ver_2/scripts/preprocess_fundamentals.py \
    --start_date 2018-01-01 \
    --end_date 2023-12-31 \
    --output_file dow_jones_5yr_data.csv
```

- ✔ Fetches financial data for Dow 30 companies  
- ✔ Computes ratios: Current Ratio, Debt-to-Equity, etc.  
- ✔ Assigns risk classification (Low, Moderate, High)

---

#### 1.2 Sentiment (News Headlines)

```bash
python analyzers/sentimental_analysis/data/extract_news.py \
    --output_file dow30_news_data.csv
```

- ✔ Collects headlines using `yfinance`  
- ✔ Maps news to respective companies

---

#### 1.3 Technical (Price-based Indicators)

```bash
python analyzers/technical_analysis/data/preprocess_technical_data.py \
    --output_dir analyzers/technical_analysis/data/processed \
    --test_size 0.2
```

- ✔ Downloads 10 years of price data  
- ✔ Computes indicators: daily return, SMA-50, SMA-200  
- ✔ Outputs: `train_data.csv`, `test_data.csv`

---

### 🔹 Step 2: Preprocess & Clean

#### Clean Fundamental Risk Labels

```bash
python analyzers/fundamental_analysis/model/ver_2/scripts/clean_labels.py \
    --input_file dow_jones_5yr_data.csv \
    --output_file cleaned_fundamental.csv
```

#### Clean and Label Sentiment Data

```bash
python analyzers/sentimental_analysis/data/preprocess.py \
    --input_file dow30_news_data.csv \
    --train_file train.csv \
    --val_file val.csv \
    --test_file test.csv
```

- ✔ Uses Hugging Face `pipeline("sentiment-analysis")`  
- ✔ Adds labels: `positive`, `neutral`, `negative`  
- ✔ Splits dataset 70/20/10

---

### 🤖 Train Models

#### 📘 Fundamental Model (BERT)

```bash
python analyzers/fundamental_analysis/model/ver_2/scripts/train_transformer_model.py \
    --train_file train_data.csv \
    --val_file val_data.csv \
    --test_file test_data.csv \
    --output_dir fundamental_analysis/model_output
```

- ✔ Trains BERT model on financial ratios  
- ✔ Outputs `financial_risk_classifier.pth`

---

#### 📰 Sentiment Model (News Headline Classifier)

```bash
python analyzers/sentimental_analysis/models/train_sentiment_model.py \
    --train_file train.csv \
    --val_file val.csv \
    --save_path sentimental_analysis/model_output
```

- ✔ Fine-tunes BERT for sentiment classification  
- ✔ Saves model and tokenizer

---

#### 📈 Technical Model (SVM)

```bash
python analyzers/technical_analysis/models/train_svm.py \
    --train_file train_data.csv \
    --test_file test_data.csv \
    --output_dir technical_analysis/model_outputs
```

- ✔ SVM classifier using SMA and return features  
- ✔ Predicts: `Buy`, `Hold`, or `Sell`

---

### 🔁 Integration & Inference

#### 🧩 Run Inference on a Sample

```bash
python scripts/integrate.py
```

- ✔ Loads all 3 models  
- ✔ Predicts on synthetic sample  
- ✔ Prints individual and final voted results

---

#### 🧮 Batch Inference on Real Data

```bash
python scripts/inference_final.py
```

- ✔ Loads cleaned datasets  
- ✔ Aligns companies by ticker  
- ✔ Applies model predictions and voting logic  
- ✔ Outputs:

  - `voting_results_limited.csv`: Row-wise predictions  
  - `company_analysis.csv`: Final Buy/Sell recommendation

---

### 📊 Metrics & Evaluation

#### Evaluate Ensemble Logic

```bash
python scripts/metrics_integrate.py
```

- ✔ Simulates uncertainty using random noise  
- ✔ Computes Accuracy, Precision, Recall, F1  
- ✔ Saves to: `weighted_voting_results_with_randomness.csv`

---

## ✅ Recommended Run Order

```bash
# 1. Data Generation
preprocess_fundamentals.py
extract_news.py
preprocess_technical_data.py

# 2. Cleaning & Preprocessing
clean_labels.py
preprocess.py

# 3. Training Models
train_transformer_model.py
train_sentiment_model.py
train_svm.py

# 4. Integration
inference_final.py

# 5. Evaluation
metrics_integrate.py
```

---

## 🧠 Why This Project Matters

- **Tri-Modal Learning**: Merges insights from three distinct analysis styles  
- **Transformer + Traditional ML**: Balances modern NLP with classic finance models  
- **Modular Design**: Pipelines are independent and scalable  
- **Robust Voting Logic**: Boosts generalization and reduces overfitting

---

Let’s build smarter stock strategies — across **data**, **domains**, and **models**.
