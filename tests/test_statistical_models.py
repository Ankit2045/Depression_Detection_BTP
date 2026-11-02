"""
Unit and Integration Tests for Statistical Models and Predictions.
"""

import os
import pandas as pd
import numpy as np
import pytest

METRICS_PATH = "results/metrics/statistical_results_table.csv"
PART_PREDS_PATH = "results/predictions/statistical_participant_predictions.csv"
WIN_PREDS_PATH = "results/predictions/statistical_window_predictions.csv"


def test_statistical_results_table_exists():
    assert os.path.exists(METRICS_PATH), f"Results table missing at {METRICS_PATH}"
    df = pd.read_csv(METRICS_PATH)
    assert len(df) == 8, f"Expected 8 experiments, got {len(df)}"
    expected_cols = [
        "experiment_id", "model", "feature_set", "num_features",
        "balanced_accuracy", "accuracy", "sensitivity", "specificity",
        "precision", "f1_score", "mcc", "roc_auc", "tp", "tn", "fp", "fn"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in results table"

    # Verify confusion matrix math
    for _, row in df.iterrows():
        tp, tn, fp, fn = row["tp"], row["tn"], row["fp"], row["fn"]
        assert tp + tn + fp + fn == 55, "Confusion matrix sum != 55 participants"
        assert tp + fn == 23, "Depression condition count != 23"
        assert tn + fp == 32, "Control count != 32"
        expected_sens = tp / 23.0
        expected_spec = tn / 32.0
        assert np.isclose(row["sensitivity"], expected_sens)
        assert np.isclose(row["specificity"], expected_spec)
        assert np.isclose(row["balanced_accuracy"], (expected_sens + expected_spec) / 2.0)


def test_participant_predictions_file_integrity():
    assert os.path.exists(PART_PREDS_PATH), f"Predictions missing at {PART_PREDS_PATH}"
    df = pd.read_csv(PART_PREDS_PATH)
    assert len(df) == 55, f"Expected 55 participant rows, got {len(df)}"

    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    assert len(prob_cols) == 8
    assert len(pred_cols) == 8

    for pcol in prob_cols:
        assert (df[pcol] >= 0.0).all() and (df[pcol] <= 1.0).all()

    for prcol in pred_cols:
        assert set(df[prcol].unique()).issubset({0, 1})


def test_window_predictions_file_integrity():
    assert os.path.exists(WIN_PREDS_PATH), f"Window predictions missing at {WIN_PREDS_PATH}"
    df = pd.read_csv(WIN_PREDS_PATH)
    assert len(df) == 1331, f"Expected 1331 window rows, got {len(df)}"
    prob_cols = [c for c in df.columns if c.endswith("_prob")]
    assert len(prob_cols) == 8
    for pcol in prob_cols:
        assert (df[pcol] >= 0.0).all() and (df[pcol] <= 1.0).all()
