# 🏠 Predicting House Prices with Linear Regression

> Building and evaluating a linear regression model for house price prediction — full workflow from EDA through model interpretation, including an honest look at when a linear model doesn't have real predictive power.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-black?style=flat-square&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project builds an end-to-end linear regression pipeline to predict house prices from features like area, location, room counts, and age. It covers:

- 📊 EDA on the dataset and target variable distribution
- 🧠 A feature selection discussion, reasoned through before modelling
- 🔤 One-Hot Encoding of categorical features
- 🔗 A correlation heatmap of all features against price
- 🤖 A trained Linear Regression model, evaluated with MSE, RMSE, and R²
- 📈 Actual-vs-predicted and residual diagnostic plots
- 🧮 Coefficient analysis — which features push price up or down
- 🎁 Bonus: Ridge & Lasso regularised model comparison

---

## 🗂️ Dataset

`data/house_prices.csv` — **2,000 houses**, 10 columns (Kaggle "House Price Prediction Dataset"):

| Column | Description |
|---|---|
| `Id` | Row identifier (dropped before modelling) |
| `Area` | House area |
| `Bedrooms` / `Bathrooms` | Room counts |
| `Floors` | Number of floors |
| `YearBuilt` | Year of construction |
| `Location` | Downtown / Urban / Suburban / Rural |
| `Condition` | Poor / Fair / Good / Excellent |
| `Garage` | Yes / No |
| `Price` | Target variable ($) |

📦 Source: [Kaggle](https://www.kaggle.com/)

---

## 📁 Project Structure

```
DataAnalytics-L2-PredictingHousePrices/
├── README.md
├── requirements.txt
├── data/
│   └── house_prices.csv
└── notebooks/
    └── house_price_prediction.ipynb
```

---

## 🔍 What's Inside the Notebook

The full workflow lives in [`notebooks/house_price_prediction.ipynb`](notebooks/house_price_prediction.ipynb) — **outputs are pre-rendered**, so it displays directly on GitHub without needing to run anything.

- ✅ EDA — nulls, descriptive stats, target variable distribution
- ✅ Feature selection reasoning, explained in markdown before modelling
- ✅ Missing value check + One-Hot Encoding of categorical features
- ✅ Correlation heatmap
- ✅ 80/20 train/test split
- ✅ Linear Regression model training
- ✅ MSE, RMSE, and R² evaluation
- ✅ Actual vs. predicted scatter plot
- ✅ Residual plot + residual distribution
- ✅ Coefficient analysis (positive/negative price drivers)
- ✅ Bonus: Ridge & Lasso comparison

---

## 🚀 Getting Started

```bash
git clone https://github.com/ayush-cyber1/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-PredictingHousePrices

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/house_price_prediction.ipynb
```

---

## 💡 Key Findings

- ⚠️ **R² comes out slightly negative (≈ -0.01)** across Linear, Ridge, and Lasso models — this dataset's `Price` values appear to have little to no real linear relationship with the given features, consistent with a synthetically-generated practice dataset rather than a real housing market
- 🔍 Residual diagnostics confirm the model isn't violating linear regression assumptions (no heteroscedasticity or curved pattern) — the model is technically valid, it simply has very little signal to work with
- ⚖️ Ridge and Lasso perform nearly identically to plain Linear Regression here, since regularisation addresses overfitting/multicollinearity, not a fundamental lack of signal
- 📝 The notebook walks through the complete, correct workflow and calls out this result honestly rather than glossing over it — a model training without errors doesn't guarantee real predictive power

> Full metrics, diagnostic plots, and coefficient tables are in the notebook.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## 📬 Contact

Built by **Ayush** — [GitHub](https://github.com/ayush-cyber1)

⭐ If you find this useful, consider starring the repo!
