# 🧩 Customer Segmentation Analysis — K-Means Clustering

> Segmenting an e-commerce customer base into distinct behavioural groups using K-Means clustering — with targeted marketing recommendations for each segment.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-black?style=flat-square&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project applies **K-Means clustering** to segment an e-commerce company's customers into distinct behavioural groups, enabling targeted marketing strategies for each segment. It covers:

- 📊 Descriptive statistics on purchase value & engagement behaviour
- 🎯 Feature selection using an RFM-inspired framework
- ⚙️ Standardisation and the Elbow Method to find optimal K
- 🖼️ Cluster visualisation across multiple feature pairs
- 🧬 Per-cluster profiling and customer archetypes
- 💡 Marketing recommendations tailored to each segment

---

## 🗂️ Dataset

`data/ecommerce_customers.csv` — **500 customers**, 7 columns (Kaggle "Ecommerce Customers" dataset):

| Column | Description |
|---|---|
| `Email` | Customer email (identifier) |
| `Address` | Customer address |
| `Avatar` | Avatar color (categorical, unused in clustering) |
| `Time on App` | Avg. hours spent on the app |
| `Time on Website` | Avg. hours spent on the website |
| `Length of Membership` | Years as a member |
| `Yearly Amount Spent` | Total yearly spend ($) |

📦 Source: [Kaggle](https://www.kaggle.com/)

> **Note:** This dataset has one row per customer with no transaction-level dates, so classic RFM (Recency, Frequency, Monetary) is adapted using the closest available behavioural proxies — `Yearly Amount Spent` (Monetary), `Time on App` (Frequency proxy), and `Length of Membership` (loyalty/tenure proxy). This is explained in full in the notebook.

---

## 📁 Project Structure

```
DataAnalytics-L1-CustomerSegmentationAnalysis/
├── README.md
├── requirements.txt
├── data/
│   └── ecommerce_customers.csv
└── notebooks/
    └── customer_segmentation_analysis.ipynb
```

---

## 🔍 What's Inside the Notebook

The full analysis lives in [`notebooks/customer_segmentation_analysis.ipynb`](notebooks/customer_segmentation_analysis.ipynb) — **outputs are pre-rendered**, so it displays directly on GitHub without needing to run anything.

- ✅ Data inspection — shape, dtypes, nulls, duplicate check
- ✅ Descriptive statistics — avg. purchase value, engagement proxy, CLV proxy
- ✅ RFM-adapted feature selection with justification
- ✅ StandardScaler normalisation
- ✅ K-Means clustering with Elbow Method for optimal K
- ✅ Cluster scatter plots across 3 feature-pair combinations
- ✅ Per-cluster profiling (mean feature values + customer archetype)
- ✅ Bar chart — customers per cluster
- ✅ Segment-by-segment marketing recommendations

---

## 🚀 Getting Started

```bash
git clone https://github.com/ayush-cyber1/OIBSIP.git
cd OIBSIP/DataAnalytics-L1-CustomerSegmentationAnalysis

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/customer_segmentation_analysis.ipynb
```

---

## 💡 Key Findings

- 🎯 Optimal cluster count (K) determined via the Elbow Method
- 💰 Customer segments range from high-value "Loyal Champions" to low-engagement new browsers
- 📱 App engagement correlates more strongly with spend than website engagement — a signal for where to invest
- 🎁 Each segment maps to a distinct, actionable marketing strategy

> Full cluster profiles and recommendations are in the notebook's insights section.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## 📬 Contact

Built by **Ayush** — [GitHub](https://github.com/ayush-cyber1)

⭐ If you find this useful, consider starring the repo!
