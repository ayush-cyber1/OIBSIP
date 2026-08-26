# 💬 Sentiment Analysis — Tweet Classification

> Classifying tweets as positive, negative, or neutral using NLP preprocessing, TF-IDF, and two ML classifiers — with full evaluation and error analysis.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-black?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project builds a machine learning pipeline that classifies tweet sentiment as **positive**, **negative**, or **neutral**, comparing two classifiers and analysing where each one goes wrong. It covers:

- 📊 Class distribution inspection
- 🧹 A full NLP preprocessing pipeline (cleaning, tokenisation, stopword removal, lemmatisation)
- 🔢 TF-IDF feature extraction, explained
- 🤖 Two trained classifiers — Naive Bayes & Logistic Regression
- 📈 Full evaluation — accuracy, precision, recall, F1, confusion matrices
- ☁️ WordClouds per sentiment class
- 🔍 Error analysis on real misclassified tweets
- 💡 A conclusion on the best model and real-world applications

---

## 🗂️ Dataset

`data/tweet_sentiment_train.csv` — **27,481 tweets** labelled `positive` / `negative` / `neutral` (Kaggle "Tweet Sentiment Extraction" dataset).

📦 Source: [Kaggle](https://www.kaggle.com/)

---

## 📁 Project Structure

```
DataAnalytics-L1-SentimentAnalysis/
├── README.md
├── requirements.txt
├── data/
│   └── tweet_sentiment_train.csv
└── notebooks/
    └── sentiment_analysis.ipynb
```

---

## 🔍 What's Inside the Notebook

The full pipeline lives in [`notebooks/sentiment_analysis.ipynb`](notebooks/sentiment_analysis.ipynb) — **outputs are pre-rendered**, so it displays directly on GitHub without needing to run anything.

- ✅ Class distribution inspection + bar chart
- ✅ Preprocessing: lowercase → URL/mention/punctuation removal → tokenisation → stopword removal → lemmatisation
- ✅ TF-IDF vectorisation, with purpose explained in markdown
- ✅ 80/20 stratified train/test split
- ✅ Naive Bayes + Logistic Regression classifiers
- ✅ Accuracy, precision, recall, F1-score, and confusion matrices for both models
- ✅ WordCloud per sentiment class
- ✅ 5 misclassified examples with discussion of likely causes
- ✅ Model comparison + real-world application discussion

---

## 🚀 Getting Started

```bash
git clone https://github.com/ayush-cyber1/OIBSIP.git
cd OIBSIP/DataAnalytics-L1-SentimentAnalysis

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/sentiment_analysis.ipynb
```

---

## 💡 Key Findings

- 🏆 **Logistic Regression outperforms Naive Bayes** on weighted F1-score — it can learn weighted feature combinations rather than assuming word independence
- ⚖️ Most misclassifications happen at the **neutral ↔ polar sentiment boundary**, not between positive and negative directly
- 🎭 Sarcasm, short tweets, and mixed sentiment within a single tweet are the main sources of model error
- 📢 A model like this could power brand monitoring, social listening, or support-ticket triage

> Full metrics, confusion matrices, and error analysis are in the notebook.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `NLTK` · `WordCloud` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## 📬 Contact

Built by **Ayush** — [GitHub](https://github.com/ayush-cyber1)

⭐ If you find this useful, consider starring the repo!
