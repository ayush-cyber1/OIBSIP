# 🧹 Data Cleaning — Highest-Grossing Concert Tours

> Taking a deliberately messy, real-world scraped dataset and systematically transforming it into a clean, analysis-ready dataset — with every cleaning decision documented and justified.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project demonstrates a professional, documented data-cleaning workflow on a genuinely messy dataset — highest-grossing concert tours, scraped in a Wikipedia-table style. It covers:

- 🔍 A full data quality report (nulls, duplicates, dtype issues, hidden formatting problems)
- 🧩 Missing-data strategy chosen and justified per column
- 🔁 Duplicate detection and documentation
- 🔤 Standardisation of currency, footnote-contaminated fields, and year ranges
- 📈 IQR-based outlier detection with a reasoned keep/cap/remove decision
- 🏷️ Full data type correction
- 📊 A before-vs-after data quality summary table
- 💾 A cleaned, analysis-ready CSV output

---

## 🗂️ Dataset

`data/concert_tours_raw.csv` — **20 rows**, 11 columns: highest-grossing concert tours (Wikipedia-style scraped data).

**Real messiness in this dataset:**
- Currency values stored as strings with `$`, commas, and stray footnote brackets (`"$229,100,000[b]"`)
- Ranking columns (`Peak`, `All Time Peak`) contaminated with citation references (`"1[4]"`)
- A **non-breaking space** hidden inside two column headers (invisible in a text editor, breaks direct column access)
- Year ranges using an en-dash (`"2023–2024"`) instead of separate year fields
- A genuine tied rank (two tours both listed as Rank 7)
- Tour titles with trailing symbols (`†`, `‡`, `*`) and citation brackets

📦 Source: [Kaggle](https://www.kaggle.com/)

---

## 📁 Project Structure

```
DataAnalytics-L1-CleaningData/
├── README.md
├── requirements.txt
├── data/
│   ├── concert_tours_raw.csv
│   └── concert_tours_cleaned.csv
└── notebooks/
    └── data_cleaning.ipynb
```

---

## 🔍 What's Inside the Notebook

The full workflow lives in [`notebooks/data_cleaning.ipynb`](notebooks/data_cleaning.ipynb) — **outputs are pre-rendered**, so it displays directly on GitHub without needing to run anything.

- ✅ Data quality report — nulls, duplicates, dtype issues, hidden formatting anomalies
- ✅ Per-column missing-data strategy, justified in markdown
- ✅ Duplicate row check and documentation of a tied-rank edge case
- ✅ Standardisation — currency parsing, footnote stripping, year-range splitting
- ✅ IQR-based outlier detection with a documented retain/cap/remove decision
- ✅ Full dtype correction (string IDs, nullable integers, floats)
- ✅ Before-vs-after summary table (row count, nulls, duplicates, dtype accuracy)
- ✅ Cleaned dataset exported to CSV

---

## 🚀 Getting Started

```bash
git clone https://github.com/ayush-cyber1/OIBSIP.git
cd OIBSIP/DataAnalytics-L1-CleaningData

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook notebooks/data_cleaning.ipynb
```

---

## 💡 Key Cleaning Decisions

- 🔢 Footnote-contaminated numeric fields parsed with regex, not manually re-typed
- ❓ Missing ranking values (`Peak`, `All Time Peak`) **retained as NaN**, not imputed — a fabricated chart rank would misrepresent the data
- 📊 Statistical outliers in gross revenue **retained**, not capped — they're the legitimate top performers a "highest-grossing" list is meant to capture
- 🆔 Synthetic string `Tour ID` added since `Rank` alone has a genuine tie and isn't a reliable unique key

> Full reasoning for every decision is documented directly in the notebook's markdown cells.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `Jupyter Notebook`

---

## 📬 Contact

Built by **Ayush** — [GitHub](https://github.com/ayush-cyber1)

⭐ If you find this useful, consider starring the repo!
