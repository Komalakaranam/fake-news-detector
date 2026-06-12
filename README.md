# 📰 Fake News Detection System

A machine learning pipeline that classifies news articles as **Real** or **Fake** using NLP and three classification models.

🌐 **Live Demo:** https://fake-news-detector-jqkeb3oq3qjtbckabjyytj.streamlit.app

---

## 🔍 Overview

Built a text classification system on 44,000+ news articles using TF-IDF vectorization and compared Logistic Regression, Naive Bayes, and Random Forest classifiers. Achieved **99.62% accuracy** with Random Forest.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 99.03% | 0.99 | 0.99 | 0.99 |
| Naive Bayes | 95.91% | 0.96 | 0.96 | 0.96 |
| **Random Forest** | **99.62%** | **0.99** | **0.99** | **0.99** |

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** Scikit-learn, NumPy, Pandas
- **NLP:** NLTK, TF-IDF Vectorizer (50,000 features, unigrams + bigrams)
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Streamlit

---

## ✨ Features

- ✅ Real vs Fake news classification
- ✅ 3 ML models compared side by side
- ✅ Live prediction with confidence score
- ✅ Probability bar chart for each prediction
- ✅ EDA visualizations (class distribution, article length)
- ✅ Confusion matrices for all 3 models
- ✅ Interactive Streamlit web app

---

## 📁 Dataset

[ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) — 44,000+ labeled news articles (Real/Fake) from Reuters and other sources.

---

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

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/komalakaranam/fake-news-detector
cd fake-news-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# Go to: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
# Download True.csv and Fake.csv → place in this folder

# 4. Run the ML pipeline
python fake_news_detector.py

# 5. Or run the web app
streamlit run app.py
```

---

## 📂 Project Structure

```
fake-news-detector/
├── app.py                  ← Streamlit web app
├── fake_news_detector.py   ← Full ML pipeline
├── True_small.csv          ← Dataset (real news)
├── Fake_small.csv          ← Dataset (fake news)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📷 Output Files

- `eda_plots.png` — Class distribution & article length histogram
- `confusion_matrices.png` — Side-by-side confusion matrices for all 3 models
- `model_comparison.png` — Accuracy bar chart

---

## 🔮 Future Improvements

- Replace TF-IDF with BERT embeddings for better short-text accuracy
- Add multilingual fake news detection
- Integrate real-time news API for live checking
- Add explainability with SHAP values

---

## 👤 Author

**Komala Karanam** — B.Tech CSE, ANITS (2027)

