# 📰 Fake News Detection System

A machine learning pipeline that classifies news articles as **Real** or **Fake** using NLP and three classification models.

## 🔍 Overview

Built a text classification system on 44,000+ news articles using TF-IDF vectorization and compared Logistic Regression, Naive Bayes, and Random Forest classifiers.

## 📊 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~98% |
| Naive Bayes | ~94% |
| Random Forest | ~99% |

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** Scikit-learn, NumPy, Pandas
- **NLP:** NLTK, TF-IDF Vectorizer
- **Visualization:** Matplotlib, Seaborn

## 📁 Dataset

[ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) — 44,000+ labeled news articles (Real/Fake)

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/komalakaranam/fake-news-detector
cd fake-news-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Go to: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
# Download True.csv and Fake.csv → place in this folder

# 4. Run
python fake_news_detector.py
```

## 📈 Pipeline

```
Raw Text
   ↓
Preprocessing (lowercase, stopword removal, URL/punctuation stripping)
   ↓
TF-IDF Vectorization (50,000 features, unigrams + bigrams)
   ↓
Model Training (Logistic Regression / Naive Bayes / Random Forest)
   ↓
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
   ↓
Live Prediction on Custom Input
```

## 📷 Output

- `eda_plots.png` — Class distribution & article length histogram
- `confusion_matrices.png` — Side-by-side confusion matrices for all 3 models
- `model_comparison.png` — Accuracy bar chart

## 👤 Author

**Komala Karanam** — B.Tech CSE, ANITS (2027)
