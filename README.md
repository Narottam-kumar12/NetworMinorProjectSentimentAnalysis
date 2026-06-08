<div align="center">

<h1>🧠 Sentiment Analysis Ensemble Pipeline</h1>

<p>Production-grade multi-class sentiment classifier for tweets, reviews, and user feedback —<br>combining TF-IDF, SMOTE balancing, and a 4-model soft-voting ensemble.</p>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-latest-00BCD4?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-FF6600?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-NLP-85C1E9?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

| 🎯 Train Accuracy | ✅ Test Accuracy | 🤖 Ensemble Models | 📊 TF-IDF Features |
|:-:|:-:|:-:|:-:|
| **88.79%** | **77.70%** | **4** | **5,000** |

> 📄 Based on research paper: *"Study of Twitter Sentiment Analysis using Machine Learning Algorithms on Python"*  
> Narottam Kumar & Sandeep Kumar · MMMUT Gorakhpur · Supervisor: Dr. Vimal Kumar

</div>

---

## 📌 Overview

This project presents a **production-grade Sentiment Analysis System** that classifies tweets into three categories:

- 🔴 **Negative**
- ⚪ **Neutral**
- 🟢 **Positive**

The pipeline is engineered for real-world robustness — handling class imbalance via multi-class SMOTE, controlling overfitting through regularization, and combining four diverse classifiers through a soft-voting ensemble. The methodology is documented in a peer-reviewed research paper.

---

## 🧠 Research Contribution

Twitter sentiment analysis is uniquely challenging due to:

| Challenge | Our Solution |
|---|---|
| Short, noisy text (140–280 chars) | Multi-step preprocessing pipeline (regex, lemmatization, stopword removal) |
| Class imbalance | Multi-class SMOTE applied only on training set (no data leakage) |
| Single-model fragility | Soft-voting ensemble of 4 diverse classifiers |
| Overfitting | Depth limits, L1/L2 regularization, class-weighted loss |
| Sparse features | TF-IDF with uni-gram + bi-gram, 5,000 features |

---

## 🏗️ Pipeline Architecture

```
Raw Tweet Data (Kaggle CSV)
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  Text Preprocessing                                  │
│  • Remove URLs, mentions, hashtags, emojis           │
│  • Lowercase, stopword removal, lemmatization        │
│  • Tokenization, whitespace trimming                 │
└──────────────────────┬───────────────────────────────┘
                       │
     ▼
┌──────────────────────────────────────────────────────┐
│  TF-IDF Vectorization                                │
│  max_features = 5,000  │  ngram_range = (1, 2)       │
└──────────────────────┬───────────────────────────────┘
                       │
     ▼
┌──────────────────────────────────────────────────────┐
│  Multi-class SMOTE (training set only)               │
│  Oversample Neutral + Positive minority classes      │
└──────────────────────┬───────────────────────────────┘
                       │
     ▼
┌──────────────────────────────────────────────────────┐
│  Model Training                                      │
│  ┌───────────┐  ┌─────────┐  ┌────────────┐  ┌────┐ │
│  │ LightGBM  │  │ XGBoost │  │Rand Forest │  │ LR │ │
│  └───────────┘  └─────────┘  └────────────┘  └────┘ │
└──────────────────────┬───────────────────────────────┘
                       │
     ▼
┌──────────────────────────────────────────────────────┐
│  Soft Voting Ensemble                                │
│  Average predicted probabilities across all 4 models │
└──────────────────────┬───────────────────────────────┘
                       │
     ▼
  Prediction + Evaluation (unseen test set)
```

---

## 🤖 Models

| Model | Type | Regularization Applied |
|---|---|---|
| **LightGBM** | Gradient Boosting (leaf-wise) | Limited tree depth, L1/L2 (`reg_alpha`, `reg_lambda`) |
| **XGBoost** | Gradient Boosting (level-wise) | `reg_alpha`, `reg_lambda`, learning rate tuning |
| **Random Forest** | Bagging ensemble | Max depth constraint, class-weighted loss |
| **Logistic Regression** | Linear classifier | L2 penalty (C-regularization) |

> Final prediction uses **Soft Voting** — each model outputs a probability distribution; the ensemble averages them before classifying.  
> Formally: **ŷ = argmax Σ wₘ · Pₘ(y|X)**

---

## ✨ Features

- ✔️ Multi-step NLP preprocessing (regex, lemmatization, stopwords via NLTK)
- ✔️ TF-IDF with **uni-gram + bi-gram** support (`ngram_range=(1,2)`, `max_features=5000`)
- ✔️ **Multi-class SMOTE** applied only on training set — zero data leakage into test set
- ✔️ **Soft-voting ensemble** across 4 diverse classifiers
- ✔️ Overfitting control via L1/L2 regularization + depth constraints on all tree models
- ✔️ Stratified 80/20 train/test split — preserves original class distribution
- ✔️ Comprehensive evaluation: Accuracy · Precision · Recall · F1 (per class) · Confusion Matrix
- ✔️ Custom text prediction via simple API call

---

## 📊 Results

### Training Performance (after SMOTE)

| Class | Precision | Recall | F1-Score | Support |
|---|:-:|:-:|:-:|:-:|
| Negative | 0.93 | 0.90 | 0.92 | 7,343 |
| Neutral | 0.79 | 0.87 | 0.83 | 3,671 |
| Positive | 0.91 | 0.88 | 0.89 | 3,671 |
| **Overall** | **0.89** | **0.89** | **0.89** | **14,685** |

**Training Accuracy: 88.79%**

---

### Test Performance (unseen data)

| Class | Precision | Recall | F1-Score | Support |
|---|:-:|:-:|:-:|:-:|
| Negative | 0.87 | 0.84 | 0.85 | 1,835 |
| Neutral | 0.58 | 0.68 | 0.63 | 620 |
| Positive | 0.74 | 0.65 | 0.69 | 473 |
| **Overall** | **0.78** | **0.78** | **0.78** | **2,928** |

**Test Accuracy: 77.70%**

---

### Individual Model Accuracy (Test Set)

| Model | Test Accuracy |
|---|:-:|
| LightGBM | 80% |
| XGBoost | 79% |
| Random Forest | 78% |
| Logistic Regression | 77% |
| **Soft Voting Ensemble** | **77.70%** |

> The 11% train/test gap reflects **intentional regularization** — the model is tuned for generalization over leaderboard performance. Negative sentiment is predicted most accurately (F1: 0.85); Neutral is harder to distinguish due to overlapping expressions with Positive.

---

## 🚀 Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm xgboost matplotlib seaborn nltk
```

### Run the Pipeline

```python
import pandas as pd
from sentiment_ensemble_pipeline import SentimentEnsemblePipeline

# Dataset requires: clean_text, sentiment_label columns
df = pd.read_csv("tweets.csv")

model = SentimentEnsemblePipeline()
model.run_pipeline(df)
```

### Predict on New Text

```python
model.predict_new_tweet("I love this new feature, it's awesome!")
# → Positive
```

---

## 📁 Dataset Schema

| Column | Type | Description |
|---|---|---|
| `clean_text` | `str` | Preprocessed tweet text |
| `sentiment_label` | `int` | `0` = Negative · `1` = Neutral · `2` = Positive |

Data source: Kaggle Twitter Sentiment Dataset (CSV format, pre-labeled)

---

## 🔍 Preprocessing Steps

Each raw tweet goes through the following pipeline before vectorization:

1. Remove re-tweets, duplicate entries, and null values
2. Strip URLs, `@mentions`, `#hashtags`, numbers, emojis, special characters (regex)
3. Convert all text to lowercase
4. Remove stopwords (NLTK stopword corpus)
5. Tokenize into individual words
6. Lemmatize to root forms (e.g., `running → run`)
7. Trim extra whitespace → output stored in `clean_text`

---

## 📈 Roadmap

- [ ] 🔹 Fine-tune BERT / RoBERTa for deep contextual understanding
- [ ] 🔹 Automated hyperparameter optimization using Optuna
- [ ] 🔹 Real-time tweet streaming via Twitter / X API
- [ ] 🔹 REST API deployment via FastAPI
- [ ] 🔹 Interactive sentiment dashboard with Streamlit

---

## 💡 Why This Approach Works

| Problem | Naive Approach | Our Approach |
|---|---|---|
| Class imbalance | Ignore it → biased toward majority | Multi-class SMOTE on train only |
| Overfitting | No regularization → high variance | Depth limits + L1/L2 across all models |
| Single-model fragility | One model fails → wrong prediction | 4-model soft-voting ensemble |
| Sparse tweet features | Unigrams only → miss context | Bi-gram TF-IDF captures phrases |

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 👨‍💻 Authors

**Narottam Kumar** 
B.Tech Computer Science Engineering (3rd Year)  
Madan Mohan Malaviya University of Technology, Gorakhpur  


[GitHub →](https://github.com/Narottam-kumar12)

---

<div align="center">
<sub>If this project helped you, consider giving it a ⭐</sub>
</div>
