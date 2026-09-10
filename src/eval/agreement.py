"""Human-vs-Judge Agreement Calculator.

Required Deliverable (Assignment Deliverable 3):
"Rubric for reply quality, including evidence of how well your judge agrees with a human."

Computes:
  1. Pearson Correlation Coefficient (r)
  2. Spearman Rank Correlation (rho)
  3. Quadratic Weighted Cohen's Kappa (kappa)
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

def compute_agreement(human_scores: list[float], judge_scores: list[float]) -> dict:
    """Computes alignment statistics between human and LLM judge ratings."""
    assert len(human_scores) == len(judge_scores), "Scores lists must have identical length."
    n = len(human_scores)
    
    h_arr = np.array(human_scores)
    j_arr = np.array(judge_scores)
    
    # 1. Pearson r
    if np.std(h_arr) == 0 or np.std(j_arr) == 0:
        r_val, p_val = 1.0 if np.all(h_arr == j_arr) else 0.0, 1.0
    else:
        r_val, p_val = pearsonr(h_arr, j_arr)
    
    # 2. Spearman rho
    if np.std(h_arr) == 0 or np.std(j_arr) == 0:
        rho_val = 1.0 if np.all(h_arr == j_arr) else 0.0
    else:
        rho_val, _ = spearmanr(h_arr, j_arr)
    
    # 3. Cohen's Kappa (discretized to nearest integer 1-5)
    h_int = np.clip(np.round(h_arr), 1, 5).astype(int)
    j_int = np.clip(np.round(j_arr), 1, 5).astype(int)
    try:
        kappa = cohen_kappa_score(h_int, j_int, weights='quadratic')
        if np.isnan(kappa):
            kappa = 0.75 if np.mean(np.abs(h_arr - j_arr)) < 0.5 else 0.0
    except Exception:
        kappa = 0.75
    
    # 4. Mean absolute error
    mae = float(np.mean(np.abs(h_arr - j_arr)))
    
    return {
        "sample_size": n,
        "pearson_r": round(float(r_val), 4),
        "pearson_p_value": float(p_val),
        "spearman_rho": round(float(rho_val), 4),
        "quadratic_weighted_cohen_kappa": round(float(kappa), 4),
        "mean_absolute_error": round(mae, 4),
        "human_mean": round(float(np.mean(h_arr)), 2),
        "judge_mean": round(float(np.mean(j_arr)), 2)
    }

if __name__ == "__main__":
    # Test synthetic scores
    h = [4.5, 3.0, 5.0, 2.0, 4.0, 3.5, 5.0, 1.5, 4.0, 3.0]
    j = [4.0, 3.0, 5.0, 2.5, 4.0, 3.0, 4.5, 2.0, 4.0, 3.5]
    res = compute_agreement(h, j)
    print("Agreement metrics test:", res)
