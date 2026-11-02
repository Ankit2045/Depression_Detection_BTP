"""
Unit and integration tests for the Baseline Reproduction (Arora et al., 2023).
"""

import os
import torch
import numpy as np
import pandas as pd
import pytest

from baseline.data_loader import load_baseline_data
from baseline.model import LSTMFeatureExtractor, HybridFeatureSpace, build_svm_classifier


def test_baseline_data_loader():
    """Verify data loader loads sequence data and 3 baseline features."""
    X_seq, X_stat, y, meta_df = load_baseline_data()
    assert X_seq.shape == (1331, 24, 1), f"Unexpected sequence shape: {X_seq.shape}"
    assert X_stat.shape == (1331, 3), f"Unexpected stats shape: {X_stat.shape}"
    assert len(y) == 1331, f"Unexpected labels count: {len(y)}"
    assert set(np.unique(y)) == {0, 1}, "Labels are not binary {0, 1}"


def test_baseline_model_architecture():
    """Verify LSTM feature extractor and hybrid feature space construction."""
    model = LSTMFeatureExtractor(input_dim=1, hidden_dim=64, num_layers=2)
    x = torch.randn(8, 24, 1)

    logits = model(x)
    assert logits.shape == (8,), f"Expected logits shape (8,), got {logits.shape}"

    latent = model.extract_features(x)
    assert latent.shape in [(8, 64), (8, 128), (8, 256)], f"Expected latent shape (8, 128/256), got {latent.shape}"

    stat_sample = np.random.randn(8, 3)
    hybrid = HybridFeatureSpace.construct(latent.detach().numpy(), stat_sample)
    assert hybrid.shape in [(8, 67), (8, 131), (8, 259)], f"Expected hybrid shape, got {hybrid.shape}"


def test_baseline_svm_instantiation():
    """Verify SVM classifier initialization."""
    svm = build_svm_classifier(kernel="rbf", C=10.0)
    assert svm.kernel == "rbf"
    assert svm.C == 10.0
    assert svm.probability is True


def test_baseline_results_integrity():
    """Verify baseline 10-fold results table exists and contains expected ablations."""
    results_path = "baseline/results/baseline_10fold_results.csv"
    assert os.path.exists(results_path), f"Missing {results_path}"

    df = pd.read_csv(results_path)
    assert len(df) == 3, f"Expected 3 ablation configurations, got {len(df)}"

    configs = list(df["model_configuration"].values)
    assert any("Statistical Features Alone" in c for c in configs)
    assert any("Deep LSTM Features Alone" in c for c in configs)
    assert any("Hybrid LSTM-SVM" in c for c in configs)

    for _, row in df.iterrows():
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["balanced_accuracy"] <= 1.0
        assert 0.0 <= row["roc_auc"] <= 1.0
        assert row["tp"] + row["tn"] + row["fp"] + row["fn"] == 1331


def test_baseline_figure_exists():
    """Verify baseline ROC curve figure exists and is publication-quality."""
    fig_path = "baseline/results/figures/baseline_roc_curve.png"
    assert os.path.exists(fig_path), f"Missing {fig_path}"
    assert os.path.getsize(fig_path) > 50_000, "Baseline ROC figure is suspiciously small"
