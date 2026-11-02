"""
Stacking Meta-Classifier Module for Multimodal Actigraphy Fusion.

Implements second-level Stacking with Logistic Regression meta-learner:
- Base features: [P_statistical, P_deep]
- Meta-learner: LogisticRegression(class_weight='balanced')

Strict Leakage Prevention:
For each outer LOSO fold k, the meta-learner is fitted exclusively on the 54
training participants' out-of-fold probabilities.
The meta-classifier is NEVER trained on the outer test participant.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.metrics import calculate_classification_metrics


def run_stacking_fusion(
    p_stat_all: np.ndarray,
    p_deep_all: np.ndarray,
    y_true_all: np.ndarray,
    pids_all: np.ndarray,
    outer_folds: List[Dict],
    model_name: str,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run leak-free 55-fold LOSO Stacking using Logistic Regression meta-classifier.
    """
    n_participants = len(y_true_all)
    pid_to_idx = {pid: i for i, pid in enumerate(pids_all)}

    # 2D meta-feature matrix: [P_stat, P_deep]
    X_meta = np.column_stack([p_stat_all, p_deep_all])

    p_stacked_oof = np.zeros(n_participants, dtype=float)

    for fold in outer_folds:
        test_pid = fold["test_participant"]
        test_idx = pid_to_idx[test_pid]
        train_pids = fold["train_participants"]
        train_indices = [pid_to_idx[p] for p in train_pids]

        X_train_meta = X_meta[train_indices]
        y_train = y_true_all[train_indices]
        X_test_meta = X_meta[test_idx : test_idx + 1]

        # Fit meta-classifier strictly on 54 training subjects
        meta_clf = LogisticRegression(class_weight="balanced", random_state=random_state)
        meta_clf.fit(X_train_meta, y_train)

        # Predict probability for unseen test patient
        test_prob = meta_clf.predict_proba(X_test_meta)[0, 1]
        p_stacked_oof[test_idx] = float(test_prob)

    metrics = calculate_classification_metrics(y_true_all, p_stacked_oof, threshold=0.5)
    metrics["model"] = f"stat_and_{model_name}"
    metrics["fusion_strategy"] = "stacking_meta_learner"

    preds_df = pd.DataFrame({
        "participant_id": pids_all,
        "true_label": y_true_all,
        f"p_stacked_{model_name}": np.round(p_stacked_oof, 5),
    })

    return metrics, preds_df
