"""
MALLORN Evaluation Utilities
Model evaluation and threshold optimization.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Find the threshold that maximizes F1 score.
    Returns (optimal_threshold, best_f1_score).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def calculate_f1_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """Calculate F1 score at a given threshold."""
    preds_binary = (y_pred >= threshold).astype(int)
    return f1_score(y_true, preds_binary)
