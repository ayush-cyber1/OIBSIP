# ⌨️ Autocomplete and Autocorrect Data Analytics

> Implementing and comparing frequency-based n-gram autocomplete models and edit-distance autocorrect approaches on a real 8.2-million-word Wikipedia corpus — with an honest discussion of evaluation limitations and how this compares to production systems like Google Keyboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-black?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-black?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-black?style=flat-square&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## 📌 Overview

This project analyses the efficiency and accuracy of autocomplete and autocorrect algorithms using classic NLP techniques, on a large real-world text corpus. It covers:

- 🧹 A full NLP preprocessing pipeline, with task-specific reasoning about when stopword removal helps vs. hurts
- 🔮 Autocomplete: frequency-based **bigram and trigram** n-gram models, tested on 12 input prefixes
- ✍️ Autocorrect: a **custom Levenshtein-distance corrector** built from the corpus vocabulary, compared against **pyspellchecker**
- 📊 Precision/Recall metrics, properly defined for both single-label (autocorrect) and top-k (autocomplete) tasks
- ⚖️ A grounded algorithm comparison using this run's actual results — including an honest explanation of a counter-intuitive finding
- 📈 Word frequency visualisation and an autocorrect outcome matrix
- 💬 A detailed discussion of limitations vs. production systems like Google Keyboard

---

## 🗂️ Dataset

`data/wikipedia_text_corpus.csv` — **10,859 Wikipedia article extracts**, ~52.7 million characters (~8.2 million word tokens after cleaning). A large, diverse, real-world English text sample — well above the volume needed for a statistically meaningful n-gram model.

📦 Source: Wikipedia article extracts (self-sourced text corpus)

---

## 📁 Project Structure

```
DataAnalytics-L2-AutocompleteAndAutocorrectDataAnalytics/
├── README.md
├── requirements.txt
├── data/
│   └── wikipedia_text_corpus.csv
├── notebooks/
│   └── autocomplete_autocorrect_analysis.ipynb
└── scripts/
    └── autocomplete_autocorrect_analysis.py
```

---

## 🔍 What's Inside the Notebook

The full analysis lives in [`notebooks/autocomplete_autocorrect_analysis.ipynb`](notebooks/autocomplete_autocorrect_analysis.ipynb) — **outputs are pre-rendered**, so it displays directly on GitHub without needing to run anything. A standalone [`scripts/autocomplete_autocorrect_analysis.py`](scripts/autocomplete_autocorrect_analysis.py) version is also included, runnable directly from the command line.

- ✅ Corpus loaded, with a self-sourcing note explaining its origin and scale
- ✅ Full preprocessing pipeline (tokenisation, lowercasing, punctuation removal, stopword removal), with task-specific application explained
- ✅ Bigram + trigram frequency models, tested on 12 prefixes with top-3 predictions each
- ✅ Custom Levenshtein autocorrect + pyspellchecker, tested on 20 deliberately misspelled words
- ✅ Precision/Recall properly defined and calculated for both tasks (including Recall@3/Precision@3 for autocomplete)
- ✅ A disclosed evaluation limitation (train/test overlap) that directly explains a counter-intuitive result
- ✅ Bigram vs. trigram and custom vs. pyspellchecker comparisons, grounded in real numbers
- ✅ Top-20 word frequency bar chart + an autocorrect outcome matrix
- ✅ A full limitations discussion vs. production systems like Google Keyboard

---

## 🚀 Getting Started

```bash
git clone https://github.com/ayush-cyber1/OIBSIP.git
cd OIBSIP/DataAnalytics-L2-AutocompleteAndAutocorrectDataAnalytics

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/autocomplete_autocorrect_analysis.ipynb

# ...or run the standalone script
cd scripts
python autocomplete_autocorrect_analysis.py
```

---

## 💡 Key Findings

- 🔀 **Trigram beat bigram on both Recall@3 (58.4% vs. 26.4%) and Precision@3** — the opposite of the usual "more context = sparser coverage" expectation, and the notebook explains exactly why: this evaluation reuses the training corpus, so the trigram model is partly "remembering" rather than generalising
- ✍️ **pyspellchecker (85% accuracy) outperformed the custom corrector (70% accuracy)** on 20 misspelled words — the custom corrector is capped by whether the correct word exists in this specific 140,517-word corpus vocabulary
- 📝 Autocorrect precision and recall are mathematically identical for a single-label task — reported as such, not padded out as if they were different insights
- 🤖 A detailed, honest comparison to production systems highlights exactly what this implementation *doesn't* have: neural language modelling, personalization, keyboard-geometry awareness, and held-out evaluation

> Full metrics, predictions, and reasoning are in the notebook.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NLTK` · `pyspellchecker` · `textdistance` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## 📬 Contact

Built by **Ayush** — [GitHub](https://github.com/ayush-cyber1)

⭐ If you find this useful, consider starring the repo!
