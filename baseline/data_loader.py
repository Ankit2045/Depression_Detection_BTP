"""
Baseline Data Loader Module: Arora et al. (2023) Reproduction.

Paper Reference:
    A. Arora, P. Chakraborty, and M. P. S. Bhatia,
    "Identifying digital biomarkers in actigraph based sequential motor activity data for assessment of depression:
    a model evaluating SVM in LSTM extracted feature space,"
    International Journal of Information Technology (Springer), vol. 15, no. 2, pp. 797-802, Feb. 2023.

Loads 24-hour overlapping sliding windows (hourly downsampled, 12h hop, T=24)
and extracts the 3 baseline statistical digital biomarkers:
1. Mean Motor Activity
2. Standard Deviation of Motor Activity
3. Zero-Activity Ratio (Sedentary Immobility Fraction)
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def load_baseline_data(
    windows_path: str = "data/processed/windows_hourly.npy",
    features_csv: str = "data/processed/features_statistical.csv",
    metadata_csv: str = "data/processed/window_metadata.csv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load sequential actigraphy windows and 3 statistical features according to Arora et al. (2023).

    Returns:
        X_seq: np.ndarray of shape (N_windows, 24, 1) - sequential hourly activity.
        X_stat: np.ndarray of shape (N_windows, 3) - [mean_activity, std_activity, zero_ratio].
        y: np.ndarray of shape (N_windows,) - binary class labels (1 = depressed, 0 = control).
        meta_df: pd.DataFrame with window_id, participant_id, group, label.
    """
    if not os.path.exists(windows_path):
        raise FileNotFoundError(f"Missing sequences file at: {windows_path}")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Missing features file at: {features_csv}")

    X_seq = np.load(windows_path)  # (1331, 24, 1)
    df = pd.read_csv(features_csv)

    stat_cols = ["mean_activity", "std_activity", "zero_ratio"]
    for col in stat_cols:
        if col not in df.columns:
            raise KeyError(f"Expected statistical biomarker '{col}' in {features_csv}")

    X_stat = df[stat_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)

    meta_cols = ["window_id", "participant_id", "group", "label"]
    meta_df = df[meta_cols].copy()

    return X_seq, X_stat, y, meta_df
