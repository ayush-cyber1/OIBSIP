# 🛍️ Retail Sales — Exploratory Data Analysis

> Uncovering sales trends, customer behavior, and product performance from 1,000 retail transactions — with actionable business recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-black?style=flat-square&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project performs a thorough **Exploratory Data Analysis (EDA)** on a retail sales transaction dataset to uncover:

- 📈 Time-based sales trends
- 👥 Customer demographic patterns
- 🛒 Product category performance
- 🔗 Relationships between key numerical variables
- 💡 Non-obvious cross-segment insights

...and translates the findings into **concrete, actionable business recommendations**.

---

## 🗂️ Dataset

`data/retail_sales_dataset.csv` — **1,000 transactions** across **9 columns**:

| Column | Description |
|---|---|
| `Transaction ID` | Unique identifier for each transaction |
| `Date` | Date of purchase |
| `Customer ID` | Unique customer identifier |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Product Category` | Beauty / Clothing / Electronics |
| `Quantity` | Units purchased |
| `Price per Unit` | Price per unit ($) |
| `Total Amount` | Total transaction value ($) |

📦 Source: [Kaggle](https://www.kaggle.com/)

---

## 📁 Project Structure

```
retail-sales-eda/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── retail_sales_dataset.csv
└── notebooks/
    └── retail_sales_eda.ipynb
```

---

## 🔍 What's Inside the Notebook

The full analysis lives in [`notebooks/retail_sales_eda.ipynb`](notebooks/retail_sales_eda.ipynb) — **outputs are pre-rendered**, so it displays directly on GitHub without needing to run anything.

- ✅ Data inspection — shape, dtypes, null checks
- ✅ Descriptive statistics — mean, median, mode, std
- ✅ Monthly & quarterly sales trend charts
- ✅ Customer demographics — age distribution & gender breakdown
- ✅ Product category analysis — revenue & units sold
- ✅ Correlation heatmap of numerical variables
- ✅ 💡 Bonus insight — gender × age-group spend breakdown
- ✅ Markdown commentary after every chart
- ✅ Business recommendations section

---

## 🚀 Getting Started

```bash
git clone https://github.com/ayush-cyber1/retail-sales-eda.git
cd retail-sales-eda

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/retail_sales_eda.ipynb
```

---

## 💡 Key Findings

- 📊 Sales are **right-skewed** — a small share of high-value transactions drives a disproportionate share of revenue
- 🎯 **Age has almost no correlation** with spending — category and gender matter more
- 🏆 One product category **clearly leads** on both revenue and units sold
- 👫 Average order value per category **differs meaningfully by gender**

> Full write-up and recommendations are in the notebook's conclusion section.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## 📬 Contact

Built by **Ayush** — [GitHub](https://github.com/ayush-cyber1)

