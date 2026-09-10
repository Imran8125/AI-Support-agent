"""Baseline Classifiers for Intent Classification.

Baseline A (Trivial):
  - Predicts the single majority class across all training data.

Baseline B (Simple):
  - TF-IDF Vectorizer (word & char n-grams) + Logistic Regression.
  - Trained on 2,500 weakly labeled historical tweets from historical_kb.jsonl.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
import random
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.taxonomy.taxonomy_def import INTENT_NAMES

random.seed(42)

HISTORICAL_PATH = "data/processed/historical_kb.jsonl"
GOLDEN_PATH = "golden_set/golden_eval_set.json"

class BaselineMajority:
    """Baseline A: Majority Class Predictor."""
    def __init__(self):
        self.majority_class = None

    def fit(self, X: list[str], y: list[str]):
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, X: list[str]) -> list[str]:
        return [self.majority_class] * len(X)

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return np.ones((len(X), len(INTENT_NAMES))) / len(INTENT_NAMES)


class BaselineTfidfLogReg:
    """Baseline B: TF-IDF (word 1-2 ngrams + char 3-5 ngrams) + Logistic Regression."""
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )),
            ('clf', LogisticRegression(
                max_iter=500,
                C=1.5,
                class_weight='balanced',
                random_state=42
            ))
        ])

    def fit(self, X: list[str], y: list[str]):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: list[str]) -> list[str]:
        return self.pipeline.predict(X).tolist()

    def predict_proba(self, X: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)


def create_weak_training_data(n_samples: int = 2500) -> tuple[list[str], list[str]]:
    """Builds a weak supervision training set from historical_kb.jsonl using keyword patterns."""
    from src.taxonomy.curate_golden_set import INTENT_FILTERS, clean_text
    
    X, y = [], []
    with open(HISTORICAL_PATH) as f:
        for line in f:
            if len(X) >= n_samples:
                break
            t = json.loads(line)
            clean_t = clean_text(t["customer_text"])
            if len(clean_t) < 20:
                continue
                
            matched = "other_unclear"
            for intent, patterns in INTENT_FILTERS.items():
                for pat in patterns:
                    if re.search(pat, clean_t, re.IGNORECASE):
                        matched = intent
                        break
                if matched != "other_unclear":
                    break
                    
            X.append(clean_t)
            y.append(matched)
            
    return X, y


def evaluate_baselines():
    print("Loading Golden Evaluation Set...")
    with open(GOLDEN_PATH) as f:
        golden_data = json.load(f)
        
    eval_texts = [item["customer_text"] for item in golden_data]
    eval_labels = [item["golden_intent"] for item in golden_data]
    
    print(f"Loaded {len(eval_texts)} golden evaluation examples.")
    
    # 1. Train models on historical training set
    print("\nCreating weak supervision training set from historical data...")
    X_train, y_train = create_weak_training_data(n_samples=2500)
    print(f"Training set size: {len(X_train)} examples across {len(set(y_train))} classes.")
    
    # Baseline A
    print("\n--- BASELINE A (Majority Class) ---")
    base_a = BaselineMajority().fit(X_train, y_train)
    pred_a = base_a.predict(eval_texts)
    acc_a = accuracy_score(eval_labels, pred_a)
    f1_a = f1_score(eval_labels, pred_a, average='macro', zero_division=0)
    print(f"Predicted Majority Class: '{base_a.majority_class}'")
    print(f"Accuracy: {acc_a * 100:.2f}%")
    print(f"Macro-F1: {f1_a * 100:.2f}%")
    
    # Baseline B
    print("\n--- BASELINE B (TF-IDF + Logistic Regression) ---")
    base_b = BaselineTfidfLogReg().fit(X_train, y_train)
    pred_b = base_b.predict(eval_texts)
    acc_b = accuracy_score(eval_labels, pred_b)
    f1_b = f1_score(eval_labels, pred_b, average='macro', zero_division=0)
    print(f"Accuracy: {acc_b * 100:.2f}%")
    print(f"Macro-F1: {f1_b * 100:.2f}%")
    
    print("\nDetailed Baseline B Classification Report:")
    print(classification_report(eval_labels, pred_b, zero_division=0))
    
    # Save baseline results
    results = {
        "baseline_a": {
            "name": "Majority Class Predictor",
            "majority_class": base_a.majority_class,
            "accuracy": round(acc_a, 4),
            "macro_f1": round(f1_a, 4)
        },
        "baseline_b": {
            "name": "TF-IDF + Logistic Regression",
            "accuracy": round(acc_b, 4),
            "macro_f1": round(f1_b, 4)
        }
    }
    
    os.makedirs("data/eval_results", exist_ok=True)
    with open("data/eval_results/baselines_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Baseline results saved to data/eval_results/baselines_results.json")
    
    return base_a, base_b

if __name__ == "__main__":
    evaluate_baselines()
