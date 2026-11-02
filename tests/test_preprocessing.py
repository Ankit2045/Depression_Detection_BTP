"""
Unit and Integration Tests for Phase 2 Preprocessing Pipeline.

Tests:
1. Day truncation according to scores.csv.
2. Daylight Saving Time gap interpolation and imputation flags.
3. Window generation, shapes, 50% overlap, and participant boundary protection.
4. Leave-One-Subject-Out (LOSO) splitting integrity and leakage prevention.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from preprocessing.clean_data import clean_participant_series, load_participant_metadata
from preprocessing.create_windows import generate_windows_for_participant


METADATA_PATH = "data/processed/window_metadata.csv"
HOURLY_PATH = "data/processed/windows_hourly.npy"
MINUTE_PATH = "data/processed/windows_minute.npy"
SPLITS_PATH = "results/splits/loso_splits.json"


def test_clean_data_truncation():
    scores = load_participant_metadata("data/scores.csv")
    # Test condition_1 (days=11 -> 15840 minutes)
    c1_days = int(scores.loc["condition_1", "days"])
    df1, meta1 = clean_participant_series("data/condition/condition_1.csv", days_limit=c1_days)
    assert len(df1) == c1_days * 1440
    assert meta1["imputed_minutes"] == 0
    assert not df1["activity"].isna().any()


def test_dst_gap_interpolation():
    scores = load_participant_metadata("data/scores.csv")
    # control_6 contains the 60-min DST gap on 2003-03-30
    c6_days = int(scores.loc["control_6", "days"])
    df6, meta6 = clean_participant_series("data/control/control_6.csv", days_limit=c6_days)
    assert len(df6) == c6_days * 1440
    assert meta6["imputed_minutes"] == 60
    assert not df6["activity"].isna().any()
    assert (df6["activity"] >= 0).all()


def test_window_generation_shapes_and_counts():
    assert os.path.exists(HOURLY_PATH), f"Hourly windows file missing at {HOURLY_PATH}"
    assert os.path.exists(MINUTE_PATH), f"Minute windows file missing at {MINUTE_PATH}"
    assert os.path.exists(METADATA_PATH), f"Metadata file missing at {METADATA_PATH}"

    hourly = np.load(HOURLY_PATH)
    minute = np.load(MINUTE_PATH)
    meta = pd.read_csv(METADATA_PATH)

    assert hourly.shape == (1331, 24, 1), f"Expected (1331, 24, 1), got {hourly.shape}"
    assert minute.shape == (1331, 1440, 1), f"Expected (1331, 1440, 1), got {minute.shape}"
    assert len(meta) == 1331

    # Class balance check
    cond_count = (meta["label"] == 1).sum()
    ctrl_count = (meta["label"] == 0).sum()
    assert cond_count == 559, f"Expected 559 condition windows, got {cond_count}"
    assert ctrl_count == 772, f"Expected 772 control windows, got {ctrl_count}"


def test_no_cross_participant_windows():
    meta = pd.read_csv(METADATA_PATH)
    # Check that for each participant, the window indices are strictly contiguous
    for pid, group in meta.groupby("participant_id"):
        indices = group["window_index"].tolist()
        expected = list(range(len(indices)))
        assert indices == expected, f"Window indices non-contiguous for {pid}"


def test_loso_splits_leakage_safety():
    assert os.path.exists(SPLITS_PATH), f"Splits file missing at {SPLITS_PATH}"
    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)

    assert splits["total_outer_folds"] == 55
    outer_folds = splits["outer_folds"]
    seen_test_subjects = set()

    all_window_indices = set(range(splits["total_windows"]))

    for fold in outer_folds:
        test_pid = fold["test_participant"]
        assert test_pid not in seen_test_subjects, f"Duplicate test subject in LOSO: {test_pid}"
        seen_test_subjects.add(test_pid)

        train_pids = set(fold["train_participants"])
        assert test_pid not in train_pids, f"Leakage: {test_pid} is in both train and test!"
        assert len(train_pids) == 54

        test_win = set(fold["test_window_indices"])
        train_win = set(fold["train_window_indices"])

        # Disjoint windows
        assert len(test_win & train_win) == 0, f"Window leakage in fold {fold['fold_id']}"
        # Complete coverage
        assert (test_win | train_win) == all_window_indices

        # Check inner splits
        inner_splits = fold["inner_splits"]
        assert len(inner_splits) == 5

        for inner in inner_splits:
            val_p = set(inner["val_participants"])
            tr_inner_p = set(inner["train_participants"])
            assert test_pid not in val_p, "Outer test subject leaked to inner val!"
            assert test_pid not in tr_inner_p, "Outer test subject leaked to inner train!"
            assert len(val_p & tr_inner_p) == 0, "Inner train and val subjects overlap!"

            val_win = set(inner["val_window_indices"])
            tr_win = set(inner["train_window_indices"])
            assert len(val_win & tr_win) == 0, "Inner train and val windows overlap!"
            assert (val_win | tr_win) == train_win, "Inner splits do not partition train windows!"
