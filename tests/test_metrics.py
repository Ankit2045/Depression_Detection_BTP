"""
Unit tests for Evaluation Metrics and Participant-Level Aggregation.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    calculate_classification_metrics,
    aggregate_participant_predictions,
    find_optimal_threshold,
)


def test_perfect_classification():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.95, 0.80, 0.10, 0.20])
    m = calculate_classification_metrics(y_true, y_prob, threshold=0.5)

    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1_score"] == 1.0
    assert np.isclose(m["mcc"], 1.0)
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_imperfect_classification_counts():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.8, 0.7, 0.3, 0.2, 0.6, 0.1]) # TP=2, FN=1, TN=2, FP=1
    m = calculate_classification_metrics(y_true, y_prob, threshold=0.5)

    assert m["confusion_matrix"]["tp"] == 2
    assert m["confusion_matrix"]["fn"] == 1
    assert m["confusion_matrix"]["tn"] == 2
    assert m["confusion_matrix"]["fp"] == 1
    assert np.isclose(m["sensitivity"], 2 / 3)
    assert np.isclose(m["specificity"], 2 / 3)
    assert np.isclose(m["balanced_accuracy"], 2 / 3)


def test_participant_level_aggregation():
    # 2 participants:
    # condition_1 (label 1) with 3 windows: [0.6, 0.7, 0.8] -> mean 0.70 (Correct)
    # control_1 (label 0) with 2 windows: [0.2, 0.4] -> mean 0.30 (Correct)
    window_df = pd.DataFrame({
        "participant_id": ["condition_1", "condition_1", "condition_1", "control_1", "control_1"],
        "label": [1, 1, 1, 0, 0],
        "pred_prob": [0.6, 0.7, 0.8, 0.2, 0.4],
    })

    part_df, metrics = aggregate_participant_predictions(
        window_df=window_df,
        pred_col="pred_prob",
        true_col="label",
        pid_col="participant_id",
        threshold=0.5,
    )

    assert len(part_df) == 2
    c1_row = part_df[part_df["participant_id"] == "condition_1"].iloc[0]
    assert np.isclose(c1_row["pred_prob"], 0.70)
    assert c1_row["pred_label"] == 1
    assert c1_row["is_correct"] == True

    ctrl1_row = part_df[part_df["participant_id"] == "control_1"].iloc[0]
    assert np.isclose(ctrl1_row["pred_prob"], 0.30)
    assert ctrl1_row["pred_label"] == 0
    assert ctrl1_row["is_correct"] == True

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_find_optimal_threshold():
    # If positive samples have probs [0.4, 0.45] and negative [0.1, 0.2]
    # Default 0.5 threshold fails for positives, but threshold ~0.35 achieves 100%
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.42, 0.48, 0.15, 0.22])

    thresh, score = find_optimal_threshold(y_true, y_prob, metric_target="balanced_accuracy")
    assert thresh < 0.40
    assert score == 1.0
