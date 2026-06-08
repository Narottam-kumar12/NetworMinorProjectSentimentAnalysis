<div align="center">

<h1>🧠 Sentiment Analysis Ensemble Pipeline</h1>

<p>Production-grade multi-class sentiment classifier for tweets, reviews, and user feedback —<br>combining TF-IDF, SMOTE balancing, and a 4-model soft-voting ensemble.</p>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-latest-00BCD4?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-FF6600?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

| 🎯 Train Accuracy | ✅ Test Accuracy | 🤖 Ensemble Models | 📊 TF-IDF Features |
|:-:|:-:|:-:|:-:|
| **86%** | **75%** | **4** | **3,000** |

</div>

---

## 📌 Overview

This project presents a **production-grade Sentiment Analysis System** that classifies text into three categories:

- 🔴 **Negative**
- ⚪ **Neutral**
- 🟢 **Positive**

The pipeline is engineered for real-world robustness — handling class imbalance, controlling overfitting via regularization, and combining multiple models through soft-voting ensemble learning.

---

## ⚙️ Tech Stack

| Category | Libraries |
|---|---|
| Core ML | `scikit-learn`, `LightGBM`, `XGBoost` |
| Imbalance Handling | `imbalanced-learn (SMOTE)` |
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |

---

## 🏗️ Pipeline Architecture

```
Raw Text Data
     │
     ▼
┌─────────────────────┐
│  Text Preprocessing │
└──────────┬──────────┘
           │
     ▼
┌─────────────────────────────┐
│  TF-IDF Vectorization       │
│  max_features=3000          │
│  ngram_range=(1,2)          │
└──────────┬──────────────────┘
           │
     ▼
┌──────────────────────────────┐
│  SMOTE (Multi-class Balance) │
└──────────┬───────────────────┘
           │
     ▼
┌──────────────────────────────────────────────────────────┐
│  Model Training                                          │
│  ┌────────────┐  ┌─────────┐  ┌───────────┐  ┌───────┐ │
│  │  LightGBM  │  │ XGBoost │  │Rand Forest│  │Log Reg│ │
│  └────────────┘  └─────────┘  └───────────┘  └───────┘ │
└──────────────────────┬───────────────────────────────────┘
                       │
     ▼
┌──────────────────────────────────┐
│  Soft Voting Ensemble            │
│  (probability averaging)         │
└──────────────────────────────────┘
           │
     ▼
  Prediction + Evaluation
```

---

## 🤖 Models

| Model | Type | Strength | Key Config |
|---|---|---|---|
| **LightGBM** | Gradient Boosting | Fast & efficient | Leaf-wise splits, L1/L2 reg |
| **XGBoost** | Gradient Boosting | Regularized & robust | `reg_alpha`, `reg_lambda` |
| **Random Forest** | Bagging | Reduces variance | Depth constraints |
| **Logistic Regression** | Linear Model | Stable baseline | C-regularization |

> Final prediction uses **Soft Voting** — probabilities from all 4 models are averaged before classification.

---

## ✨ Features

- ✔️ TF-IDF with **bi-gram support** (`ngram_range=(1,2)`)
- ✔️ **Multi-class SMOTE** for handling label imbalance
- ✔️ **Soft-voting ensemble** across 4 diverse models
- ✔️ **Overfitting control** via regularization (`reg_alpha`, `reg_lambda`) and depth constraints
- ✔️ Stratified 80/20 train/test split
- ✔️ Comprehensive evaluation: Accuracy · F1-score (per class) · Confusion Matrix
- ✔️ **Custom text prediction** via a simple API call

---

## 📊 Performance

| Metric | Score |
|---|---|
| Training Accuracy | 86% |
| Test Accuracy | **75%** |

> The 11% train/test gap reflects **intentional regularization** — this ensemble is tuned for generalization, not leaderboard overfitting. A model that memorizes training data is not production-ready.

---

## 🚀 Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm xgboost matplotlib seaborn
```

### Run the Pipeline

```python
import pandas as pd
from sentiment_ensemble_pipeline import SentimentEnsemblePipeline

# Load dataset
df = pd.read_csv("tweet.csv")  # requires: clean_text, sentiment_label

# Initialize and run
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

Your dataset must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `clean_text` | `str` | Preprocessed input text |
| `sentiment_label` | `str` | Target label: `Negative`, `Neutral`, or `Positive` |

---

## 🔍 How It Works

**1. Data Splitting**
Stratified 80/20 train/test split to preserve class distribution across both sets.

**2. Feature Engineering**
TF-IDF vectorization with bi-gram support (`ngram_range=(1,2)`, `max_features=3000`) converts raw text into numerical feature vectors.

**3. Handling Class Imbalance**
SMOTE generates synthetic minority-class samples in the training set, preventing the model from being biased toward majority classes.

**4. Model Training**
Four models are trained independently, each with hyperparameters tuned to reduce overfitting via regularization and depth constraints.

**5. Ensemble Learning**
A `VotingClassifier` with `voting='soft'` averages the predicted probabilities from all four models, producing a more stable and generalizable final prediction.

**6. Evaluation**
Results are reported via confusion matrix, per-class F1-score, and overall accuracy.

---

## 📈 Roadmap

- [ ] 🔹 Fine-tune BERT / RoBERTa for deep contextual understanding
- [ ] 🔹 Automated hyperparameter optimization using Optuna
- [ ] 🔹 REST API deployment via FastAPI
- [ ] 🔹 Interactive real-time dashboard with Streamlit
- [ ] 🔹 Live feed integration with Twitter API / News API

---

## 💡 Why This Project Stands Out

| Challenge | Solution |
|---|---|
| Imbalanced classes | Multi-class SMOTE oversampling |
| Overfitting risk | L1/L2 regularization + depth limits |
| Single-model fragility | 4-model soft-voting ensemble |
| Poor generalization | Stratified split + controlled train/test gap |

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 🙌 Author

**Narottam Kumar** — Built as part of advanced Machine Learning and NLP practice, focusing on real-world robustness and scalability.

---

<div align="center">
<sub>If this helped you, consider giving it a ⭐</sub>
</div>
