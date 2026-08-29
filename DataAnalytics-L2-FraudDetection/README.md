# 🕵️ Fraud Detection — Handling Severe Class Imbalance

> Building a fraud detection pipeline on a heavily imbalanced dataset (1.51% fraud rate) using SMOTE, with an honest look at the Precision/Recall trade-off and a scalability discussion for production volumes.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-orange?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project builds a machine learning pipeline to detect fraudulent transactions from a dataset where fraud is a rare event (1.51% of all transactions), treating class imbalance as the central modelling challenge. It covers:

- 📊 Class imbalance analysis
- 💵 EDA on transaction amount and time-of-day patterns for fraud vs. legitimate
- ⚠️ Why standard accuracy is misleading on imbalanced data — explained and quantified
- ⚖️ SMOTE oversampling, applied correctly (training set only, post-split)
- 🎯 Stratified train/test split
- 🤖 Two trained models — Logistic Regression & Random Forest
- 📈 Precision, Recall, F1-Score, and AUC-ROC evaluation
- ⚖️ A grounded Recall-vs-Precision trade-off discussion using this run's real numbers
- 🔍 Feature importance / coefficient analysis
- 🏗️ A scalability discussion: handling 1 million transactions/hour

---

## 🗂️ Dataset

`data/credit_card_fraud_10k.csv` — **10,000 transactions**, 10 columns (Kaggle-style credit card fraud dataset): `amount`, `transaction_hour`, `merchant_category`, `foreign_transaction`, `location_mismatch`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`, and the target `is_fraud`.

📦 Source: [Kaggle](https://www.kaggle.com/)

> **Note:** Only 151 of 10,000 transactions (1.51%) are fraudulent — a ~1:65 imbalance ratio. This is the core challenge the entire notebook is built around.

---

## 📁 Project Structure

```
DataAnalytics-L2-FraudDetection/
├── README.md
├── requirements.txt
├── data/
│   └── credit_card_fraud_10k.csv
├── notebooks/
│   └── fraud_detection.ipynb
└── scripts/
    └── fraud_detection.py
```

---

## 🔍 What's Inside the Notebook

The full pipeline lives in [`notebooks/fraud_detection.ipynb`](notebooks/fraud_detection.ipynb) — **outputs are pre-rendered**, so it displays directly on GitHub without needing to run anything. A standalone [`scripts/fraud_detection.py`](scripts/fraud_detection.py) version is also included, runnable directly from the command line.

- ✅ Class imbalance analysis with exact fraud percentage
- ✅ Transaction amount distribution (fraud vs. legitimate) + time-of-day fraud rate analysis
- ✅ Markdown explanation of why accuracy is misleading here, with the exact "always predict legitimate" accuracy calculated
- ✅ SMOTE applied correctly — post-split, training data only
- ✅ Stratified 80/20 split, ensuring fraud cases appear in both sets
- ✅ Logistic Regression + Random Forest, evaluated on the untouched imbalanced test set
- ✅ Precision, Recall, F1, AUC-ROC, confusion matrices, and ROC curve for both models
- ✅ A Recall-vs-Precision discussion grounded in this run's actual results
- ✅ Coefficient analysis (Logistic Regression) + feature importance (Random Forest)
- ✅ A detailed scalability discussion for 1M transactions/hour

---

## 🚀 Getting Started

```bash
git clone https://github.com/ayush-cyber1/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-FraudDetection

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/fraud_detection.ipynb

# ...or run the standalone script
cd scripts
python fraud_detection.py
```

---

## 💡 Key Findings

- ⚠️ A model that always predicts "not fraud" would score **98.49% accuracy** while catching zero fraud — the exact reason accuracy is the wrong metric here
- 🎯 **Logistic Regression**: 93% Recall, but only 29% Precision — catches almost all fraud, at the cost of many false alarms
- 🌲 **Random Forest**: a much more balanced 73% Recall / 79% Precision (F1 = 0.76), and the higher AUC-ROC (0.998 vs. 0.993)
- 🔑 `device_trust_score`, `velocity_last_24h`, `location_mismatch`, and `foreign_transaction` rank as top predictors in **both** models — a strong, consistent signal
- 🏗️ Scaling to 1M transactions/hour is primarily an infrastructure problem (streaming features, low-latency serving, drift monitoring), not a fundamentally different modelling approach

> Full metrics, plots, and reasoning are in the notebook.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `imbalanced-learn (SMOTE)` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## 📬 Contact

Built by **Ayush** — [GitHub](https://github.com/ayush-cyber1)

⭐ If you find this useful, consider starring the repo!
