"""
Fake News Detection System
Author: Komala Karanam
Dataset: ISOT Fake News Dataset (https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  FAKE NEWS DETECTION SYSTEM")
print("=" * 60)
print("\n[1/5] Loading dataset...")

# Download from Kaggle or place True.csv and Fake.csv in same folder
try:
    true_df  = pd.read_csv("True.csv")
    fake_df  = pd.read_csv("Fake.csv")
except FileNotFoundError:
    print("\n  ⚠  True.csv / Fake.csv not found.")
    print("  Download from: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets")
    print("  Place both CSV files in the same folder as this script and re-run.\n")
    exit()

true_df["label"] = 1   # Real
fake_df["label"] = 0   # Fake

df = pd.concat([true_df, fake_df], ignore_index=True)
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df[["text", "label"]].dropna()

print(f"  ✔  Dataset loaded: {len(df):,} articles")
print(f"     Real: {df['label'].sum():,}  |  Fake: {(df['label']==0).sum():,}")

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
print("\n[2/5] Exploratory Data Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Fake News Dataset — EDA", fontsize=14, fontweight='bold')

# Class distribution
counts = df['label'].value_counts()
axes[0].bar(['Fake', 'Real'], [counts[0], counts[1]], color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Number of Articles")
for i, v in enumerate([counts[0], counts[1]]):
    axes[0].text(i, v + 100, f"{v:,}", ha='center', fontweight='bold')

# Article length distribution
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df[df['label']==1]['word_count'].hist(ax=axes[1], bins=50, alpha=0.6,
                                       color='#2ecc71', label='Real')
df[df['label']==0]['word_count'].hist(ax=axes[1], bins=50, alpha=0.6,
                                       color='#e74c3c', label='Fake')
axes[1].set_title("Article Length Distribution")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].set_xlim(0, 2000)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✔  EDA plots saved → eda_plots.png")

# ─────────────────────────────────────────────
# 3. TEXT PREPROCESSING
# ─────────────────────────────────────────────
print("\n[3/5] Preprocessing text...")

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)           # remove brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>+', '', text)            # remove HTML tags
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)          # remove words with numbers
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

df['clean_text'] = df['text'].apply(preprocess)
print("  ✔  Text cleaned (lowercasing, stopword removal, URL/punctuation stripping)")

# ─────────────────────────────────────────────
# 4. FEATURE EXTRACTION & TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
print("\n[4/5] Extracting features with TF-IDF...")

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"  ✔  TF-IDF matrix: {X_train_tfidf.shape[0]:,} train | {X_test_tfidf.shape[0]:,} test")
print(f"     Vocabulary size: {len(tfidf.vocabulary_):,} tokens (unigrams + bigrams)")

# ─────────────────────────────────────────────
# 5. TRAIN & EVALUATE 3 MODELS
# ─────────────────────────────────────────────
print("\n[5/5] Training and evaluating models...")
print("-" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes":         MultinomialNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices — Model Comparison", fontsize=14, fontweight='bold')

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n  ▶  {name}")
    print(f"     Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred,
                                 target_names=["Fake", "Real"],
                                 digits=4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Fake", "Real"])
    disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
    axes[idx].set_title(f"{name}\nAccuracy: {acc*100:.2f}%")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n  ✔  Confusion matrices saved → confusion_matrices.png")

# ─────────────────────────────────────────────
# 6. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#3498db', '#e67e22', '#2ecc71']
bars = ax.bar(results.keys(), [v*100 for v in results.values()],
              color=colors, edgecolor='black', width=0.5)
ax.set_ylim(85, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison", fontweight='bold')
for bar, val in zip(bars, results.values()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{val*100:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("  ✔  Model comparison chart saved → model_comparison.png")

# ─────────────────────────────────────────────
# 7. PREDICT ON CUSTOM INPUT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LIVE PREDICTION DEMO")
print("=" * 60)

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

sample_texts = [
    "NASA confirms water found on the Moon's sunlit surface, opening new possibilities for future lunar missions.",
    "SHOCKING: Government secretly putting mind control chemicals in tap water, leaked documents reveal!!!"
]

print(f"\n  Using best model: {best_model_name} ({results[best_model_name]*100:.2f}%)\n")

for text in sample_texts:
    clean = preprocess(text)
    vec   = tfidf.transform([clean])
    pred  = best_model.predict(vec)[0]
    prob  = best_model.predict_proba(vec)[0] if hasattr(best_model, 'predict_proba') else None

    label = "✅ REAL" if pred == 1 else "❌ FAKE"
    conf  = f"  (confidence: {max(prob)*100:.1f}%)" if prob is not None else ""
    print(f"  [{label}]{conf}")
    print(f"  \"{text[:80]}...\"" if len(text) > 80 else f"  \"{text}\"")
    print()

print("=" * 60)
print("  Done! Check the saved PNG files for visualizations.")
print("=" * 60)
