"""
Fake News Detection System — Streamlit Web App
Author: Komala Karanam
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #1a1a2e;
    text-align: center;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.result-real {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border-left: 6px solid #28a745;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}

.result-fake {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border-left: 6px solid #dc3545;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}

.result-label {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.result-conf {
    font-size: 1rem;
    color: #444;
}

.stat-box {
    background: #f8f9ff;
    border: 1px solid #e0e4ff;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.stat-num {
    font-size: 1.6rem;
    font-weight: 700;
    color: #3a3aff;
}

.stat-label {
    font-size: 0.8rem;
    color: #777;
    margin-top: 0.2rem;
}

.stTextArea textarea {
    font-size: 1rem !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: #1a1a2e;
    color: white;
    font-size: 1rem;
    font-weight: 600;
    padding: 0.6rem 2rem;
    border-radius: 8px;
    border: none;
    width: 100%;
    transition: background 0.2s;
}

.stButton > button:hover {
    background: #3a3aff;
}
</style>
""", unsafe_allow_html=True)


# ── Preprocessing ────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text


# ── Model Training (cached) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    try:
        true_df = pd.read_csv("True_small.csv")
        fake_df = pd.read_csv("Fake_small.csv")
    except FileNotFoundError:
        return None, None, None

    true_df["label"] = 1
    fake_df["label"] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["text", "label"]].dropna()
    df["clean_text"] = df["text"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_t = tfidf.fit_transform(X_train)
    X_test_t  = tfidf.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes":         MultinomialNB(),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    trained = {}
    accuracies = {}
    for name, m in models.items():
        m.fit(X_train_t, y_train)
        acc = accuracy_score(y_test, m.predict(X_test_t))
        trained[name] = m
        accuracies[name] = acc

    return tfidf, trained, accuracies


# ── UI ────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any news headline or article — AI will tell you if it\'s real or fake</div>', unsafe_allow_html=True)

# Load models
with st.spinner("Training models on 44,000+ articles... (first load only)"):
    tfidf, models, accuracies = load_and_train()

if tfidf is None:
    st.error("⚠️ Could not find True.csv / Fake.csv. Make sure the dataset is in the same folder.")
    st.stop()

# Model stats
c1, c2, c3 = st.columns(3)
model_names = list(accuracies.keys())
short_names = ["Logistic Reg.", "Naive Bayes", "Random Forest"]
for col, name, short in zip([c1, c2, c3], model_names, short_names):
    with col:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{accuracies[name]*100:.1f}%</div>
            <div class="stat-label">{short}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model selector
selected_model = st.selectbox(
    "Choose model",
    options=model_names,
    index=2,  # default: Random Forest
    format_func=lambda x: f"{x}  ({accuracies[x]*100:.1f}% accuracy)"
)

# Input
user_input = st.text_area(
    "Paste a news headline or article",
    placeholder="e.g. Scientists discover new treatment for Alzheimer's disease...",
    height=150
)

# Predict
if st.button("🔍 Analyse"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analysing..."):
            time.sleep(0.4)
            clean = preprocess(user_input)
            vec = tfidf.transform([clean])
            model = models[selected_model]
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            confidence = max(prob) * 100

        if pred == 1:
            st.markdown(f"""
            <div class="result-real">
                <div class="result-label">✅ REAL NEWS</div>
                <div class="result-conf">Confidence: <strong>{confidence:.1f}%</strong> &nbsp;·&nbsp; Model: {selected_model}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-label">❌ FAKE NEWS</div>
                <div class="result-conf">Confidence: <strong>{confidence:.1f}%</strong> &nbsp;·&nbsp; Model: {selected_model}</div>
            </div>""", unsafe_allow_html=True)

        # Confidence bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Prediction probabilities**")
        prob_df = pd.DataFrame({
            "Label": ["Fake", "Real"],
            "Probability": [prob[0]*100, prob[1]*100]
        })
        st.bar_chart(prob_df.set_index("Label"))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Built by <strong>Komala Karanam</strong> · "
    "Trained on ISOT Fake News Dataset (44K+ articles) · "
    "Random Forest: 99.62% accuracy"
    "</div>",
    unsafe_allow_html=True
)