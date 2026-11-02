"""
Probability-Level Multimodal Fusion Module.

Implements:
1. Probability Averaging (Equal Weight: alpha = 0.5):
   P_fused = 0.5 * P_statistical + 0.5 * P_deep

2. Weighted Probability Fusion (alpha in [0, 1]):
   P_fused = alpha * P_statistical + (1 - alpha) * P_deep

Strict Leakage Prevention:
For weighted probability fusion, alpha is optimized strictly on the 54 training
subjects in each outer LOSO fold (maximizing training balanced accuracy).
Alpha is NEVER optimized against outer test labels.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.metrics import calculate_classification_metrics


def fuse_probability_averaging(
    p_stat: np.ndarray,
    p_deep: np.ndarray,
) -> np.ndarray:
    """Equal-weight probability averaging (alpha = 0.5)."""
    return 0.5 * p_stat + 0.5 * p_deep


def fuse_weighted_probability(
    p_stat: np.ndarray,
    p_deep: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Weighted probability fusion: alpha in [0, 1]."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * p_stat + (1.0 - alpha) * p_deep


def find_optimal_alpha_train(
    p_stat_train: np.ndarray,
    p_deep_train: np.ndarray,
    y_train: np.ndarray,
    metric_target: str = "balanced_accuracy",
    num_grid_steps: int = 41,
) -> float:
    """
    Find optimal fusion weight alpha in [0.0, 1.0] using training subjects only.
    """
    alpha_candidates = np.linspace(0.0, 1.0, num_grid_steps)
    best_alpha = 0.5
    best_score = -1.0

    for a in alpha_candidates:
        p_fused = fuse_weighted_probability(p_stat_train, p_deep_train, a)
        m = calculate_classification_metrics(y_train, p_fused, threshold=0.5)
        score = m.get(metric_target, 0.0)
        if score > best_score:
            best_score = score
            best_alpha = float(a)

    return best_alpha


def run_decision_level_fusion(
    p_stat_all: np.ndarray,
    p_deep_all: np.ndarray,
    y_true_all: np.ndarray,
    pids_all: np.ndarray,
    outer_folds: List[Dict],
    model_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Evaluate both Probability Averaging and Leak-Free Weighted Fusion across the 55 LOSO folds.

    Returns:
        metrics_avg: Metrics for equal-weight probability averaging
        metrics_weighted: Metrics for leak-free weighted fusion
        preds_df: DataFrame of per-participant fused probabilities
    """
    n_participants = len(y_true_all)
    pid_to_idx = {pid: i for i, pid in enumerate(pids_all)}

    # 1. Equal-weight averaging (no parameters to tune)
    p_avg_all = fuse_probability_averaging(p_stat_all, p_deep_all)
    metrics_avg = calculate_classification_metrics(y_true_all, p_avg_all, threshold=0.5)
    metrics_avg["model"] = f"stat_and_{model_name}"
    metrics_avg["fusion_strategy"] = "probability_averaging"
    metrics_avg["alpha"] = 0.5

    # 2. Leak-free weighted fusion (alpha tuned per outer fold on 54 train participants)
    p_weighted_oof = np.zeros(n_participants, dtype=float)
    alphas_chosen = np.zeros(n_participants, dtype=float)

    for fold in outer_folds:
        test_pid = fold["test_participant"]
        test_idx = pid_to_idx[test_pid]
        train_pids = fold["train_participants"]
        train_indices = [pid_to_idx[p] for p in train_pids]

        # Optimize alpha strictly on the 54 training participants
        best_alpha = find_optimal_alpha_train(
            p_stat_train=p_stat_all[train_indices],
            p_deep_train=p_deep_all[train_indices],
            y_train=y_true_all[train_indices],
            metric_target="balanced_accuracy",
        )

        alphas_chosen[test_idx] = best_alpha
        # Apply to unseen held-out test participant
        p_weighted_oof[test_idx] = fuse_weighted_probability(
            p_stat_all[test_idx], p_deep_all[test_idx], alpha=best_alpha
        )

    metrics_weighted = calculate_classification_metrics(y_true_all, p_weighted_oof, threshold=0.5)
    metrics_weighted["model"] = f"stat_and_{model_name}"
    metrics_weighted["fusion_strategy"] = "weighted_probability"
    metrics_weighted["mean_alpha"] = float(np.mean(alphas_chosen))

    preds_df = pd.DataFrame({
        "participant_id": pids_all,
        "true_label": y_true_all,
        f"p_avg_{model_name}": np.round(p_avg_all, 5),
        f"p_weighted_{model_name}": np.round(p_weighted_oof, 5),
        f"alpha_weighted_{model_name}": np.round(alphas_chosen, 3),
    })

    return metrics_avg, metrics_weighted, preds_df
