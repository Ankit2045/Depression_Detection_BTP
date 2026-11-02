"""
Evaluation Metrics & Participant Aggregation Module.

Implements clinical classification metrics and leak-free prediction aggregation
according to Section 9 and Section 11 of the Project Master Specification.

Metrics computed at Participant Level (primary) and Window Level (secondary):
- Balanced Accuracy
- Sensitivity (Recall for Depression class)
- Specificity (Recall for Healthy Control class)
- Precision
- F1-Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- Confusion Matrix (TN, FP, FN, TP)
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calculate comprehensive clinical classification metrics for binary predictions.

    Parameters:
        y_true: 1D array of ground truth labels (0 or 1).
        y_prob: 1D array of predicted probabilities for positive class (label 1).
        threshold: Decision threshold for converting probabilities to binary labels.

    Returns:
        Dictionary of computed metrics and confusion counts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Apply decision threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Basic counts
    n_samples = len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Clinical sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Scikit-learn standard metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0

    # ROC-AUC calculation (valid only when both classes are present)
    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
    else:
        auc = np.nan

    return {
        "n_samples": int(n_samples),
        "threshold": float(threshold),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(auc) if not np.isnan(auc) else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def aggregate_participant_predictions(
    window_df: pd.DataFrame,
    pred_col: str = "pred_prob",
    true_col: str = "label",
    pid_col: str = "participant_id",
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aggregate window-level probabilities to participant-level diagnoses.

    Formula:
        P_participant = mean(P_window for all eligible windows of that participant)

    Parameters:
        window_df: DataFrame containing at least participant_id, true label, and window pred_prob.
        pred_col: Name of predicted probability column.
        true_col: Name of ground truth label column.
        pid_col: Name of participant identifier column.
        threshold: Decision threshold for participant diagnosis.

    Returns:
        participant_df: DataFrame with 1 row per participant.
        participant_metrics: Dictionary of participant-level evaluation metrics.
    """
    records = []

    for pid, group in window_df.groupby(pid_col):
        true_label = int(group[true_col].iloc[0])
        window_probs = group[pred_col].values
        participant_prob = float(np.mean(window_probs))
        participant_pred = int(participant_prob >= threshold)
        
        is_correct = bool(int(true_label) == int(participant_pred))
        records.append({
            "participant_id": pid,
            "true_label": true_label,
            "pred_prob": round(participant_prob, 5),
            "pred_label": participant_pred,
            "window_count": len(group),
            "is_correct": is_correct,
        })

    participant_df = pd.DataFrame(records)

    # Compute overall participant-level metrics
    metrics = calculate_classification_metrics(
        y_true=participant_df["true_label"].values,
        y_prob=participant_df["pred_prob"].values,
        threshold=threshold,
    )

    return participant_df, metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_target: str = "balanced_accuracy",
) -> Tuple[float, float]:
    """
    Find optimal classification threshold on training/validation predictions only.
    Strictly leakage-safe: NEVER run this on outer test predictions.

    Returns:
        best_threshold: Optimal probability cutoff in [0.1, 0.9]
        best_score: Highest validation metric achieved.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        m = calculate_classification_metrics(y_true, y_prob, threshold=t)
        score = m.get(metric_target, 0.0)
        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh, best_score
