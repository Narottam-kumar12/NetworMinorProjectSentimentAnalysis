# 🧠 Sentiment Analysis Ensemble Pipeline  
**Author:** Narottam Kumar  
**File:** `sentiment_ensemble_pipeline.py`  
**Description:** Multi-class sentiment classification pipeline using **TF-IDF vectorization**, **SMOTE balancing**, and an **ensemble of ML models** (LightGBM, XGBoost, Random Forest, Logistic Regression) with overfitting control.

---

## 📘 Overview  
This project implements a **Sentiment Analysis System** capable of classifying text (e.g., tweets or reviews) into **three sentiment categories: Negative, Neutral, and Positive**.

The pipeline performs:
1. Text preprocessing with **TF-IDF vectorization**
2. **Class imbalance handling** using multi-class SMOTE
3. **Ensemble learning** using multiple ML models
4. **Overfitting control** through hyperparameter regularization
5. Performance evaluation with **confusion matrix**, **F1-scores**, and **classification reports**

---

## ⚙️ Key Features  
- **TF-IDF Vectorization** with configurable `max_features` and `ngram_range`  
- **Multi-class SMOTE** balancing for imbalanced datasets  
- **Soft Voting Ensemble** combining:
  - LightGBM (`lgb.LGBMClassifier`)
  - XGBoost (`XGBClassifier`)
  - Random Forest
  - Logistic Regression  
- **Overfitting Control** with regularization (`reg_alpha`, `reg_lambda`) and depth limits  
- **Performance Evaluation**:
  - Accuracy
  - F1-score (per class)
  - Confusion Matrix (visualized)
- **Prediction for new text inputs**

---

## 🧩 Model Architecture  

| Model | Purpose | Key Parameters |
|--------|----------|----------------|
| **LightGBM** | Gradient boosting, fast and efficient | `num_leaves=31`, `max_depth=6`, `learning_rate=0.05` |
| **XGBoost** | Robust boosting with regularization | `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8` |
| **Random Forest** | Bagging-based ensemble | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=50` |
| **Logistic Regression** | Linear baseline for stability | `C=1.0`, `class_weight='balanced'` |

All models contribute **soft probabilities** in a **VotingClassifier** ensemble.

---

## 🧠 How the Pipeline Works

1. **Data Split:**  
   The dataset is divided into training (80%) and testing (20%) using `train_test_split()` with stratification.

2. **Vectorization:**  
   Text data is transformed into numeric features via TF-IDF (`max_features=3000`, bigram support).

3. **Balancing:**  
   Uses **SMOTE (Synthetic Minority Over-sampling Technique)** to equalize class distribution.

4. **Training:**  
   Each base learner is trained, and the ensemble combines their outputs using **soft voting**.

5. **Evaluation:**  
   Reports training/testing accuracy, F1-scores, and confusion matrix plots.

6. **Prediction:**  
   Accepts any new text (or list of texts) and returns predicted sentiment.

---

## 🧪 Example Usage

```python
import pandas as pd
from sentiment_ensemble_pipeline import SentimentEnsemblePipeline

# Load your dataset
df = pd.read_csv("tweet.csv")  # Must contain 'clean_text' and 'sentiment_label' columns

# Initialize and run
model = SentimentEnsemblePipeline()
model.run_pipeline(df)

# Predict for new input
print(model.predict_new_tweet("I love this new feature, it's awesome!"))

##Output Examples
Training Accuracy: 0.86
Test Accuracy: 0.75
