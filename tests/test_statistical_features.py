"""
Unit tests for Statistical Feature Extraction Module.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from statistical.extract_features import (
    extract_features_for_window,
    compute_m10_l5,
    FEATURE_CATEGORIES,
    BASELINE_3_FEATURES,
)

CSV_PATH = "data/processed/features_statistical.csv"
META_PATH = "data/processed/feature_metadata.json"


def test_compute_m10_l5_synthetic():
    # Synthetic 24-hour sequence: 10 hours of 100, 14 hours of 10
    seq = np.array([100.0] * 10 + [10.0] * 14)
    m10, l5, ra = compute_m10_l5(seq)
    assert np.isclose(m10, 100.0)
    assert np.isclose(l5, 10.0)
    expected_ra = (100.0 - 10.0) / (100.0 + 10.0 + 1e-5)
    assert np.isclose(ra, expected_ra)


def test_extracted_features_file_integrity():
    assert os.path.exists(CSV_PATH), f"Features CSV not found at {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    assert df.shape == (1331, 25), f"Expected shape (1331, 25), got {df.shape}"

    # Verify no missing or infinite values
    feature_cols = [
        col for col in df.columns if col not in ["window_id", "participant_id", "group", "label"]
    ]
    assert len(feature_cols) == 21
    assert df[feature_cols].isna().sum().sum() == 0, "Found NaNs in features"
    assert np.isinf(df[feature_cols].values).sum() == 0, "Found Infs in features"


def test_feature_ranges_and_constraints():
    df = pd.read_csv(CSV_PATH)

    # Ratios must be between 0 and 1
    for ratio_col in ["zero_ratio", "day_zero_ratio", "night_zero_ratio", "relative_amplitude"]:
        assert (df[ratio_col] >= 0.0).all(), f"{ratio_col} has values < 0"
        assert (df[ratio_col] <= 1.0 + 1e-5).all(), f"{ratio_col} has values > 1"

    # Activity metrics must be non-negative
    for non_neg_col in ["mean_activity", "std_activity", "median_activity", "q25_activity", "q75_activity", "max_activity", "m10", "l5"]:
        assert (df[non_neg_col] >= 0.0).all(), f"{non_neg_col} has negative values"

    # Autocorrelation must be in [-1, 1]
    assert (df["autocorr_lag1"] >= -1.0 - 1e-5).all()
    assert (df["autocorr_lag1"] <= 1.0 + 1e-5).all()


def test_baseline_3_features_exist():
    assert os.path.exists(META_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    assert meta["baseline_3_features"] == ["mean_activity", "std_activity", "zero_ratio"]
    for feat in BASELINE_3_FEATURES:
        assert feat in meta["feature_names_all"]
