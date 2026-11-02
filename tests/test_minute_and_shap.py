"""
Unit and integration tests for High-Resolution (1-Minute) 1D-CNN and Explainable AI (SHAP).
"""

import os
import torch
import numpy as np
import pandas as pd
import pytest
from deep_learning.cnn1d_minute import ActigraphyCNN1DMinute


def test_cnn1d_minute_forward_and_latent_hook():
    """Verify 1D-CNN Minute architecture on (Batch, 1440, 1) and latent extraction."""
    model = ActigraphyCNN1DMinute(in_channels=1, dropout=0.2)
    x = torch.randn(8, 1440, 1)

    logits = model(x)
    assert logits.shape == (8,), f"Expected logits shape (8,), got {logits.shape}"

    latent = model.extract_features(x)
    assert latent.shape == (8, 64), f"Expected latent shape (8, 64), got {latent.shape}"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 20_000 < n_params < 100_000, f"Expected parameter count in [20k, 100k], got {n_params}"


def test_resolution_comparison_table_integrity():
    """Verify resolution comparison results table schema and values."""
    csv_path = "results/metrics/resolution_comparison_table.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist"

    df = pd.read_csv(csv_path)
    assert len(df) == 4, f"Expected 4 rows (hourly/minute x standalone/fusion), got {len(df)}"

    expected_cols = [
        "experiment", "resolution", "sequence_length", "accuracy",
        "balanced_accuracy", "sensitivity", "specificity", "f1_score", "mcc", "roc_auc", "tp", "tn"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in {csv_path}"

    for _, row in df.iterrows():
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["balanced_accuracy"] <= 1.0
        assert 0.0 <= row["sensitivity"] <= 1.0
        assert 0.0 <= row["specificity"] <= 1.0
        assert 0.0 <= row["roc_auc"] <= 1.0
        assert row["tp"] + row["tn"] <= 55


def test_minute_predictions_file_integrity():
    """Verify participant-level predictions for 1-minute model."""
    part_path = "results/predictions/minute_participant_predictions.csv"
    assert os.path.exists(part_path), f"File {part_path} does not exist"

    df = pd.read_csv(part_path)
    assert len(df) == 55, f"Expected 55 participant rows, got {len(df)}"
    assert not df["pred_prob"].isnull().any(), "Found NaN in pred_prob"
    assert (df["pred_prob"] >= 0.0).all() and (df["pred_prob"] <= 1.0).all()


def test_shap_feature_importance_table_integrity():
    """Verify SHAP biomarker ranking table schema, rankings, and statistical constraints."""
    csv_path = "results/metrics/shap_feature_importance.csv"
    assert os.path.exists(csv_path), f"File {csv_path} does not exist"

    df = pd.read_csv(csv_path)
    assert len(df) == 21, f"Expected 21 digital biomarkers, got {len(df)}"

    expected_cols = [
        "feature", "display_name", "family", "mean_abs_shap", "std_shap",
        "mean_coefficient", "odds_ratio", "direction", "depressed_mean",
        "control_mean", "cohens_d", "p_value", "clinical_role", "rank"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in {csv_path}"

    # Rankings should be strictly 1 to 21
    assert list(df["rank"]) == list(range(1, 22)), "Ranks are not strictly 1 to 21"
    # mean_abs_shap should be descending
    assert df["mean_abs_shap"].is_monotonic_decreasing, "mean_abs_shap is not monotonically decreasing"
    # All odds ratios must be positive
    assert (df["odds_ratio"] > 0).all(), "Found non-positive odds ratio"


def test_generated_figures_exist():
    """Verify Figure 8 (SHAP) and Figure 9 (Resolution Comparison) exist and are non-empty."""
    fig8 = "results/figures/figure8_shap_summary.png"
    fig9 = "results/figures/figure9_resolution_comparison.png"

    assert os.path.exists(fig8), f"Missing {fig8}"
    assert os.path.exists(fig9), f"Missing {fig9}"

    assert os.path.getsize(fig8) > 50_000, f"{fig8} is suspiciously small (<50KB)"
    assert os.path.getsize(fig9) > 50_000, f"{fig9} is suspiciously small (<50KB)"
