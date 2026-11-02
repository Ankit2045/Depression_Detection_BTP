"""
Unit tests for Deep Learning Benchmark Results and Predictions.
"""

import os
import numpy as np
import pandas as pd
import pytest

DEEP_METRICS_PATH = "results/metrics/deep_results_table.csv"
DEEP_PART_PREDS_PATH = "results/predictions/deep_participant_predictions.csv"
DEEP_WIN_PREDS_PATH = "results/predictions/deep_window_predictions.csv"


def test_deep_results_table_integrity():
    assert os.path.exists(DEEP_METRICS_PATH), f"Deep metrics missing at {DEEP_METRICS_PATH}"
    df = pd.read_csv(DEEP_METRICS_PATH)
    assert len(df) == 3, f"Expected 3 models (LSTM, GRU, CNN1D), got {len(df)}"

    expected_models = {"lstm", "gru", "cnn1d"}
    assert set(df["model"].unique()) == expected_models

    for _, row in df.iterrows():
        tp, tn, fp, fn = row["tp"], row["tn"], row["fp"], row["fn"]
        assert tp + tn + fp + fn == 55
        assert tp + fn == 23
        assert tn + fp == 32
        assert np.isclose(row["sensitivity"], tp / 23.0)
        assert np.isclose(row["specificity"], tn / 32.0)
        assert np.isclose(row["balanced_accuracy"], (row["sensitivity"] + row["specificity"]) / 2.0)


def test_deep_participant_predictions_integrity():
    assert os.path.exists(DEEP_PART_PREDS_PATH), f"Deep predictions missing at {DEEP_PART_PREDS_PATH}"
    df = pd.read_csv(DEEP_PART_PREDS_PATH)
    assert len(df) == 55

    for m in ["lstm", "gru", "cnn1d"]:
        assert f"{m}_prob" in df.columns
        assert f"{m}_pred" in df.columns
        assert (df[f"{m}_prob"] >= 0.0).all() and (df[f"{m}_prob"] <= 1.0).all()
        assert set(df[f"{m}_pred"].unique()).issubset({0, 1})


def test_deep_window_predictions_integrity():
    assert os.path.exists(DEEP_WIN_PREDS_PATH), f"Deep window predictions missing at {DEEP_WIN_PREDS_PATH}"
    df = pd.read_csv(DEEP_WIN_PREDS_PATH)
    assert len(df) == 1331
    for m in ["lstm", "gru", "cnn1d"]:
        assert f"{m}_prob" in df.columns
        assert (df[f"{m}_prob"] >= 0.0).all() and (df[f"{m}_prob"] <= 1.0).all()
