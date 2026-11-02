"""
Leakage-Safe Participant Splitting Module.

Implements Leave-One-Subject-Out (LOSO) Outer Cross-Validation (55 folds):
- In every outer fold k, exactly 1 participant is held out as the test subject.
- All overlapping windows belonging to that test subject are placed strictly in the test set.
- All windows from the remaining 54 subjects form the outer training pool.
- Within the 54 training subjects, a 5-fold Stratified Group K-Fold split is pre-computed
  for hyperparameter selection, scaling, threshold tuning, and early stopping.
- Guaranteed ZERO subject-level and window-level overlap between train, validation, and test sets.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.clean_data import load_config, natural_sort_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_loso_splits(
    metadata_path: str = "data/processed/window_metadata.csv",
    config_path: str = "configs/data.yaml",
    output_dir: str = "results/splits",
) -> Dict[str, Any]:
    """
    Generate deterministic, leak-free Leave-One-Subject-Out (LOSO) splits.
    
    Returns:
        splits_dict: Structured dictionary containing all 55 outer folds and nested inner folds.
    """
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    meta_df = pd.read_csv(metadata_path)
    meta_df["window_index_global"] = meta_df.index.tolist()

    # Get ordered unique participants using natural numerical sort
    participants = sorted(meta_df["participant_id"].unique().tolist(), key=natural_sort_key)
    n_participants = len(participants)
    logger.info(f"Generating LOSO splits for {n_participants} participants across {len(meta_df)} windows.")

    participant_to_label = meta_df.groupby("participant_id")["label"].first().to_dict()
    inner_folds_count = config["validation"].get("inner_folds", 5)
    random_state = config["validation"].get("random_state", 42)

    outer_folds = []

    for fold_idx, test_pid in enumerate(participants):
        test_mask = (meta_df["participant_id"] == test_pid)
        train_mask = ~test_mask

        test_indices = meta_df.index[test_mask].tolist()
        train_indices = meta_df.index[train_mask].tolist()

        test_label = int(participant_to_label[test_pid])
        train_pids = sorted([p for p in participants if p != test_pid], key=natural_sort_key)

        # Train participant metadata for inner stratified group splitting
        train_meta_df = meta_df[train_mask].copy().reset_index() # keeps 'index' as original global idx
        train_groups = train_meta_df["participant_id"].values
        train_labels = train_meta_df["label"].values

        # Stratified Group K-Fold for inner validation
        sgkf = StratifiedGroupKFold(n_splits=inner_folds_count, shuffle=True, random_state=random_state)
        inner_splits = []

        for inner_fold_idx, (inner_train_rel_idx, inner_val_rel_idx) in enumerate(
            sgkf.split(train_meta_df, train_labels, groups=train_groups)
        ):
            inner_train_global_idx = train_meta_df.iloc[inner_train_rel_idx]["index"].tolist()
            inner_val_global_idx = train_meta_df.iloc[inner_val_rel_idx]["index"].tolist()

            inner_val_pids = sorted(train_meta_df.iloc[inner_val_rel_idx]["participant_id"].unique().tolist(), key=natural_sort_key)
            inner_train_pids = sorted(train_meta_df.iloc[inner_train_rel_idx]["participant_id"].unique().tolist(), key=natural_sort_key)

            # Verification of inner leakage protection
            assert len(set(inner_val_pids) & set(inner_train_pids)) == 0, "Inner participant leakage detected!"
            assert test_pid not in inner_val_pids, "Outer test PID leaked into inner validation!"
            assert test_pid not in inner_train_pids, "Outer test PID leaked into inner training!"

            inner_splits.append({
                "inner_fold_id": inner_fold_idx,
                "val_participants": inner_val_pids,
                "val_window_indices": inner_val_global_idx,
                "train_participants": inner_train_pids,
                "train_window_indices": inner_train_global_idx,
            })

        outer_folds.append({
            "fold_id": fold_idx,
            "test_participant": test_pid,
            "test_label": test_label,
            "test_window_count": len(test_indices),
            "test_window_indices": test_indices,
            "train_participant_count": len(train_pids),
            "train_window_count": len(train_indices),
            "train_participants": train_pids,
            "train_window_indices": train_indices,
            "inner_splits": inner_splits,
        })

    splits_data = {
        "cv_type": "leave_one_subject_out",
        "total_outer_folds": len(outer_folds),
        "total_windows": len(meta_df),
        "total_participants": n_participants,
        "inner_folds_per_outer_fold": inner_folds_count,
        "outer_folds": outer_folds,
    }

    # Save to disk
    json_path = os.path.join(output_dir, "loso_splits.json")
    with open(json_path, "w") as f:
        json.dump(splits_data, f, indent=2)

    # Save a lightweight summary table
    summary_records = []
    for f in outer_folds:
        summary_records.append({
            "fold_id": f["fold_id"],
            "test_participant": f["test_participant"],
            "test_label": f["test_label"],
            "test_windows": f["test_window_count"],
            "train_participants": f["train_participant_count"],
            "train_windows": f["train_window_count"],
        })
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(output_dir, "loso_splits_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    logger.info(
        f"Successfully generated and verified {len(outer_folds)} LOSO outer folds. "
        f"Splits saved to {json_path} and {summary_csv_path}."
    )
    return splits_data


if __name__ == "__main__":
    generate_loso_splits()
