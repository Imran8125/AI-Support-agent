"""Evaluation Metrics Computation.

Computes:
  - Intent classification: Accuracy, Macro-F1, Precision, Recall per class
  - Escalation decision: Accuracy, Precision, Recall, F1
"""

from __future__ import annotations

import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    # Calculate exact per-class raw counts (TP, FP, FN, Support)
    classes = sorted(list(set(y_true) | set(y_pred)))
    per_class_counts = {}
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = sum(1 for yt in y_true if yt == c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / support if support > 0 else 0.0
        f1_val = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_counts[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "raw_correct": f"{tp}/{support}",
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1_val, 4)
        }
        
        # Enrich report dict with raw counts
        if c in report:
            report[c]["tp"] = tp
            report[c]["fp"] = fp
            report[c]["fn"] = fn
            report[c]["raw_correct"] = f"{tp}/{support}"
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class_counts": per_class_counts,
        "report": report
    }

def compute_escalation_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4)
    }
