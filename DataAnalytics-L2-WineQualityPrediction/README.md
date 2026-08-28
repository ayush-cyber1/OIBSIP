# 🍷 Wine Quality Prediction — Multi-Model Classification

> Training and comparing Random Forest, SGD, and SVC classifiers to predict wine quality from physicochemical properties — with an honest treatment of severe class imbalance.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-black?style=flat-square&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project trains and compares three classification models to predict wine quality from chemical properties like acidity, density, and alcohol content. It covers:

- 📊 EDA — feature distributions and a correlation heatmap
- ⚖️ A full discussion of severe class imbalance in the raw quality scores
- 🧠 A justified decision to bin quality into Low/Medium/High classes
- 🎯 Stratified train/test splitting
- 🤖 Three trained classifiers — Random Forest, SGD, and SVC
- 📈 Full evaluation — accuracy, classification reports, confusion matrices
- 🌲 Random Forest feature importance
- 📋 A side-by-side model comparison table
- 💡 A deployment recommendation, grounded in the actual results

---

## 🗂️ Dataset

`data/WineQT.csv` — **1,143 wine samples**, 13 columns (Kaggle "Wine Quality" dataset, WineQT variant): 11 physicochemical features (acidity, sugar, chlorides, sulfur dioxide, density, pH, sulphates, alcohol) plus a `quality` score (3–8).

📦 Source: [Kaggle](https://www.kaggle.com/)

> **Note:** Raw quality scores are severely imbalanced — quality 3 has only 6 samples and quality 8 only 16, out of 1,143 total. This is discussed in depth in the notebook and directly motivates binning quality into Low/Medium/High classes before modelling.

---

## 📁 Project Structure

```
DataAnalytics-L2-WineQualityPrediction/
├── README.md
├── requirements.txt
├── data/
│   └── WineQT.csv
├── notebooks/
│   └── wine_quality_prediction.ipynb
└── scripts/
    └── wine_quality_prediction.py
```

---

## 🔍 What's Inside the Notebook

The full workflow lives in [`notebooks/wine_quality_prediction.ipynb`](notebooks/wine_quality_prediction.ipynb) — **outputs are pre-rendered**, so it displays directly on GitHub without needing to run anything. A standalone [`scripts/wine_quality_prediction.py`](scripts/wine_quality_prediction.py) version is also included, runnable directly from the command line.

- ✅ Dataset structure inspection + raw quality class distribution
- ✅ Distribution plots for all 11 chemical features + correlation heatmap
- ✅ Class imbalance discussion — which scores are underrepresented, and why it matters
- ✅ Feature engineering: quality binned into Low/Medium/High, with reasoning
- ✅ Stratified 80/20 train/test split
- ✅ Random Forest, SGD, and SVC classifiers, all trained with `class_weight='balanced'`
- ✅ Accuracy, classification report, and confusion matrix for each model
- ✅ Random Forest feature importance chart
- ✅ Side-by-side model comparison table (accuracy, macro precision/recall/F1)
- ✅ Deployment recommendation, backed by the macro-F1 results

---

## 🚀 Getting Started

```bash
git clone https://github.com/ayush-cyber1/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-WineQualityPrediction

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/wine_quality_prediction.ipynb

# ...or run the standalone script
cd scripts
python wine_quality_prediction.py
```

---

## 💡 Key Findings

- 🏆 **Random Forest** wins on both accuracy (87.8%) and macro F1 (0.52) — the fairest comparison metric given the class imbalance
- ⚠️ Even with `class_weight='balanced'`, Random Forest's recall on the `Low` quality class is **0.00** — with only 39 samples after binning, there's likely too little data for any model to learn that class reliably
- 🍇 `alcohol` and `volatile acidity` are consistently the strongest predictors of quality, both in correlation and Random Forest feature importance
- 📝 The real bottleneck isn't the model — it's the lack of labelled poor-quality wine examples; more data would likely help more than further tuning

> Full metrics, confusion matrices, and reasoning are in the notebook.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## 📬 Contact

Built by **Ayush** — [GitHub](https://github.com/ayush-cyber1)

⭐ If you find this useful, consider starring the repo!
