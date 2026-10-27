"""
Unit tests for data loading, schema integrity, and participant manifest.
"""

import os
import glob
import pandas as pd
import pytest


DATA_DIR = "data"
SCORES_FILE = os.path.join(DATA_DIR, "scores.csv")
MANIFEST_FILE = "results/audit/participant_manifest.csv"


def test_scores_file_existence_and_schema():
    assert os.path.exists(SCORES_FILE), f"Scores file not found at {SCORES_FILE}"
    df = pd.read_csv(SCORES_FILE)
    assert len(df) == 55, f"Expected 55 entries in scores.csv, got {len(df)}"
    expected_cols = [
        "number", "days", "gender", "age", "afftype",
        "melanch", "inpatient", "edu", "marriage", "work",
        "madrs1", "madrs2"
    ]
    assert list(df.columns) == expected_cols


def test_participant_file_counts():
    cond_files = glob.glob(os.path.join(DATA_DIR, "condition", "*.csv"))
    ctrl_files = glob.glob(os.path.join(DATA_DIR, "control", "*.csv"))
    assert len(cond_files) == 23, f"Expected 23 condition files, found {len(cond_files)}"
    assert len(ctrl_files) == 32, f"Expected 32 control files, found {len(ctrl_files)}"
    assert len(cond_files) + len(ctrl_files) == 55


def test_participant_id_matching():
    scores_df = pd.read_csv(SCORES_FILE)
    scores_ids = set(scores_df["number"].str.strip())

    cond_files = glob.glob(os.path.join(DATA_DIR, "condition", "*.csv"))
    ctrl_files = glob.glob(os.path.join(DATA_DIR, "control", "*.csv"))
    file_ids = set(
        [os.path.splitext(os.path.basename(f))[0] for f in cond_files + ctrl_files]
    )

    assert scores_ids == file_ids, (
        f"Mismatch between scores.csv IDs and file IDs: "
        f"Scores only: {scores_ids - file_ids}, Files only: {file_ids - scores_ids}"
    )


def test_manifest_integrity():
    assert os.path.exists(MANIFEST_FILE), f"Manifest not generated at {MANIFEST_FILE}"
    manifest = pd.read_csv(MANIFEST_FILE)
    assert len(manifest) == 55
    assert (manifest["label"].isin([0, 1])).all()
    assert (manifest[manifest["group"] == "condition"]["label"] == 1).all()
    assert (manifest[manifest["group"] == "control"]["label"] == 0).all()
    assert manifest["is_monotonic_increasing"].all(), "Not all participant timestamps are sorted"
    assert (manifest["duplicate_timestamps"] == 0).all(), "Duplicate timestamps detected"
    assert (manifest["negative_count"] == 0).all(), "Negative activity counts detected"


def test_sample_files_schema_and_values():
    sample_files = [
        os.path.join(DATA_DIR, "condition", "condition_1.csv"),
        os.path.join(DATA_DIR, "condition", "condition_23.csv"),
        os.path.join(DATA_DIR, "control", "control_1.csv"),
        os.path.join(DATA_DIR, "control", "control_32.csv"),
    ]
    for path in sample_files:
        assert os.path.exists(path), f"Sample file {path} missing"
        df = pd.read_csv(path)
        assert list(df.columns) == ["timestamp", "date", "activity"]
        assert len(df) > 1000
        assert df["activity"].dtype in ["int64", "float64"]
        assert (df["activity"] >= 0).all()
