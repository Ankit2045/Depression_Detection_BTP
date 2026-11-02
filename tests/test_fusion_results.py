"""
Unit and Integration Tests for Multimodal Fusion Benchmark.
"""

import os
import numpy as np
import pandas as pd
import pytest

FUSION_METRICS_PATH = "results/metrics/fusion_results_table.csv"
FUSION_PREDS_PATH = "results/predictions/fusion_participant_predictions.csv"


def test_fusion_metrics_table_integrity():
    assert os.path.exists(FUSION_METRICS_PATH), f"Fusion table missing at {FUSION_METRICS_PATH}"
    df = pd.read_csv(FUSION_METRICS_PATH)
    assert len(df) in (10, 11), f"Expected 10 or 11 fusion experiments, got {len(df)}"

    expected_strategies = {
        "Probability Averaging",
        "Weighted Probability",
        "Stacking (Meta-Learner)",
        "Feature Concatenation (85D)",
        "Dual-Fusion Ensemble (Champion)",
    }
    assert set(df["fusion_type"].unique()).issubset(expected_strategies)

    for _, row in df.iterrows():
        tp, tn, fp, fn = row["tp"], row["tn"], row["fp"], row["fn"]
        assert tp + tn + fp + fn == 55
        assert tp + fn == 23
        assert tn + fp == 32
        assert np.isclose(row["sensitivity"], tp / 23.0)
        assert np.isclose(row["specificity"], tn / 32.0)
        assert np.isclose(row["balanced_accuracy"], (row["sensitivity"] + row["specificity"]) / 2.0)


def test_fusion_participant_predictions_integrity():
    assert os.path.exists(FUSION_PREDS_PATH), f"Fusion predictions missing at {FUSION_PREDS_PATH}"
    df = pd.read_csv(FUSION_PREDS_PATH)
    assert len(df) == 55

    prob_cols = [c for c in df.columns if c.startswith("p_")]
    for pcol in prob_cols:
        assert (df[pcol] >= 0.0).all() and (df[pcol] <= 1.0).all()


def test_fusion_outperforms_individual_branches():
    df_fusion = pd.read_csv(FUSION_METRICS_PATH)
    df_stat = pd.read_csv("results/metrics/statistical_results_table.csv")
    df_deep = pd.read_csv("results/metrics/deep_results_table.csv")

    stat_champ_acc = df_stat[df_stat["experiment_id"] == "logistic_regression_all_21"]["balanced_accuracy"].iloc[0]
    deep_champ_acc = df_deep[df_deep["model"] == "cnn1d"]["balanced_accuracy"].iloc[0]

    # Best fusion model
    best_fusion_acc = df_fusion["balanced_accuracy"].max()

    # Empirical test: Fusion must improve over individual branches
    assert best_fusion_acc > stat_champ_acc, f"Fusion {best_fusion_acc} did not beat Stat {stat_champ_acc}"
    assert best_fusion_acc > deep_champ_acc, f"Fusion {best_fusion_acc} did not beat Deep {deep_champ_acc}"
    assert best_fusion_acc >= 0.9096, f"Expected fusion balanced accuracy >= 90.96%, got {best_fusion_acc}"
    assert np.isclose(best_fusion_acc, 0.9253, atol=0.01)
