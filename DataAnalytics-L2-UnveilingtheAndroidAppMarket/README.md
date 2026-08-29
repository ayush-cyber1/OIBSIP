# 📱 Unveiling the Android App Market — Google Play Store Analysis

> A comprehensive analysis of the Google Play Store ecosystem — cleaning genuinely messy real-world data, exploring category saturation, ratings, pricing trends, and running sentiment analysis on real user reviews.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas)
![TextBlob](https://img.shields.io/badge/TextBlob-NLP-blue?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-black?style=flat-square&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project analyses the Google Play Store apps and user reviews datasets end-to-end. It covers:

- 🧹 Data cleaning of a genuinely messy real-world dataset (contaminated types, a corrupted row, duplicates)
- 📊 Category saturation analysis
- ⭐ Ratings distribution and average rating by category
- 📦 App size vs. installs correlation analysis
- 💰 Pricing analysis, free vs. paid split, and a revenue estimate by category
- 💬 Sentiment analysis on 60K+ real user reviews using TextBlob, validated against the dataset's original labels
- 🎭 Sentiment breakdown by app category
- 📈 Interactive Plotly visualisations
- 💡 3 data-driven insights for a developer planning a new app

---

## 🗂️ Datasets

- `data/googleplaystore.csv` — **10,841 apps**, 13 columns (category, rating, size, installs, price, etc.)
- `data/googleplaystore_user_reviews.csv` — **64,295 user reviews**, 5 columns (review text, pre-labelled sentiment, polarity, subjectivity)

📦 Source: [Kaggle — Google Play Store Apps](https://www.kaggle.com/)

> **Note:** This is a genuinely messy, real-world scraped dataset — `Installs` stored as `"10,000+"` strings, `Price` with `$` symbols, `Size` mixing `M`/`k` suffixes with `"Varies with device"`, a corrupted column-shift row, and duplicate app entries. All of this is identified and fixed explicitly in the notebook, with reasoning documented for every decision.

---

## 📁 Project Structure

```
DataAnalytics-L2-UnveilingtheAndroidAppMarket/
├── README.md
├── requirements.txt
├── data/
│   ├── googleplaystore.csv
│   └── googleplaystore_user_reviews.csv
├── notebooks/
│   └── android_app_market_analysis.ipynb
└── scripts/
    └── android_app_market_analysis.py
```

---

## 🔍 What's Inside the Notebook

The full analysis lives in [`notebooks/android_app_market_analysis.ipynb`](notebooks/android_app_market_analysis.ipynb) — **outputs are pre-rendered**, so it displays directly on GitHub without needing to run anything. A standalone [`scripts/android_app_market_analysis.py`](scripts/android_app_market_analysis.py) version is also included, runnable directly from the command line.

- ✅ Both datasets loaded and inspected separately
- ✅ Data cleaning: dropped a genuinely corrupted row, fixed `Installs`/`Price`/`Size`/`Reviews` dtypes, parsed dates, resolved duplicates, and handled nulls with documented reasoning (not blanket imputation)
- ✅ Category saturation bar chart
- ✅ Ratings distribution + average rating by category
- ✅ Size vs. installs scatter plot with correlation coefficient
- ✅ Free vs. paid split, paid app price distribution, and a category revenue estimate (with caveats)
- ✅ TextBlob sentiment classification, validated against the dataset's original sentiment labels (84.1% agreement)
- ✅ Sentiment breakdown by app category
- ✅ Two interactive Plotly charts (installs by category, rating vs. installs)
- ✅ 3 data-driven insights for a developer, grounded in the actual analysis

---

## 🚀 Getting Started

```bash
git clone https://github.com/ayush-cyber1/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-UnveilingtheAndroidAppMarket

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/android_app_market_analysis.ipynb

# ...or run the standalone script
cd scripts
python android_app_market_analysis.py
```

---

## 💡 Key Findings

- 🧹 Found and handled a genuine **column-shift data corruption bug** in the source dataset (a Rating value of 19, impossible on a 1–5 scale)
- 🏆 **Family, Game, and Tools** are the most saturated categories — a new app faces the steepest competition there
- 📦 App size shows only a **weak correlation (r ≈ 0.13)** with installs — size isn't a meaningful growth lever
- 💵 The vast majority of apps are **free**; successful paid apps cluster tightly in the **$0.99–$4.99** range
- 💬 TextBlob sentiment classification agrees with the dataset's original labels **84.1%** of the time — a solid independent validation
- 📊 Star ratings and review sentiment don't always tell the same story — both are worth monitoring separately

> Full analysis, charts, and reasoning are in the notebook.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `TextBlob` · `Plotly` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## 📬 Contact

Built by **Ayush** — [GitHub](https://github.com/ayush-cyber1)

⭐ If you find this useful, consider starring the repo!
