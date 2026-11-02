"""
Window Generation Module for Depresjon Actigraphy Dataset.

Generates synchronized 24-hour windows with 12-hour hop size:
- Hourly downsampled sequences (24 steps x 1 feature, 1-hour mean activity)
- Minute-level sequences (1440 steps x 1 feature, 1-minute activity)
- Complete window-level metadata tracking subject IDs, diagnostic labels,
  timestamps, and missing/imputation flags.

Strictly preserves participant boundaries: windows NEVER span across subjects.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import numpy as np
import pandas as pd

from preprocessing.clean_data import clean_all_participants, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_windows_for_participant(
    pid: str,
    cleaned_df: pd.DataFrame,
    group: str,
    label: int,
    max_imputed_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate 24-hour windows with 12-hour hop for one participant.

    Parameters:
        pid: Participant ID (e.g. 'condition_1')
        cleaned_df: Clean continuous DataFrame with ['timestamp', 'activity', 'is_imputed']
        group: 'condition' or 'control'
        label: 1 for condition, 0 for control
        max_imputed_ratio: Maximum allowable proportion of imputed minutes in a window.

    Returns:
        hourly_windows: Array of shape (num_windows, 24, 1)
        minute_windows: Array of shape (num_windows, 1440, 1)
        metadata_list: List of dicts containing window metadata.
    """
    total_minutes = len(cleaned_df)
    window_minutes = 1440 # 24 hours
    hop_minutes = 720     # 12 hours

    activity_vals = cleaned_df["activity"].values.astype(np.float32)
    imputed_vals = cleaned_df["is_imputed"].values.astype(bool)
    timestamps = cleaned_df["timestamp"].values

    hourly_list = []
    minute_list = []
    metadata_list = []

    win_idx = 0
    start_pos = 0

    while start_pos + window_minutes <= total_minutes:
        end_pos = start_pos + window_minutes

        win_minute_act = activity_vals[start_pos:end_pos]
        win_imputed = imputed_vals[start_pos:end_pos]

        imputed_count = int(win_imputed.sum())
        imputed_ratio = imputed_count / float(window_minutes)
        is_valid = imputed_ratio <= max_imputed_ratio

        # Compute hourly downsampled representation: 24 chunks of 60 minutes
        # Reshape (1440,) -> (24, 60) -> mean along axis 1 -> (24,)
        hourly_chunk = win_minute_act.reshape(24, 60).mean(axis=1)

        win_id = f"{pid}_win_{win_idx:03d}"
        start_ts = str(timestamps[start_pos])
        end_ts = str(timestamps[end_pos - 1])

        meta = {
            "window_id": win_id,
            "participant_id": pid,
            "group": group,
            "label": label,
            "window_index": win_idx,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "total_points": window_minutes,
            "imputed_points": imputed_count,
            "imputed_ratio": round(imputed_ratio, 5),
            "has_imputed_data": bool(imputed_count > 0),
            "is_valid": is_valid,
        }

        hourly_list.append(hourly_chunk)
        minute_list.append(win_minute_act)
        metadata_list.append(meta)

        win_idx += 1
        start_pos += hop_minutes

    if len(hourly_list) == 0:
        return np.empty((0, 24, 1)), np.empty((0, 1440, 1)), []

    hourly_arr = np.array(hourly_list, dtype=np.float32)[:, :, np.newaxis]
    minute_arr = np.array(minute_list, dtype=np.float32)[:, :, np.newaxis]

    return hourly_arr, minute_arr, metadata_list


def create_all_windows(config_path: str = "configs/data.yaml"):
    """Generate and persist all windows and metadata across all 55 participants."""
    config = load_config(config_path)
    output_dir = config["dataset"]["cache_dir"]
    os.makedirs(output_dir, exist_ok=True)

    max_imputed_ratio = config["missing_data_policy"].get("max_missing_ratio_per_window", 0.05)
    mapping = config["labels"]["mapping"]

    logger.info("Cleaning participant time series...")
    cleaned_participants = clean_all_participants(config_path)

    all_hourly = []
    all_minute = []
    all_metadata = []

    for pid, cdf in cleaned_participants.items():
        group = "condition" if pid.startswith("condition") else "control"
        label = mapping[group]

        h_arr, m_arr, meta_list = generate_windows_for_participant(
            pid=pid,
            cleaned_df=cdf,
            group=group,
            label=label,
            max_imputed_ratio=max_imputed_ratio,
        )

        all_hourly.append(h_arr)
        all_minute.append(m_arr)
        all_metadata.extend(meta_list)

    final_hourly = np.concatenate(all_hourly, axis=0)
    final_minute = np.concatenate(all_minute, axis=0)
    meta_df = pd.DataFrame(all_metadata)

    # Validate output shapes and counts
    logger.info(f"Generated hourly windows array shape: {final_hourly.shape}")
    logger.info(f"Generated minute windows array shape: {final_minute.shape}")
    logger.info(f"Total window metadata records: {len(meta_df)}")
    logger.info(
        f"Class breakdown: Condition windows = {(meta_df['label'] == 1).sum()} "
        f"({(meta_df['label'] == 1).mean() * 100:.1f}%), "
        f"Control windows = {(meta_df['label'] == 0).sum()} "
        f"({(meta_df['label'] == 0).mean() * 100:.1f}%)"
    )
    logger.info(f"Windows with imputed minutes: {(meta_df['has_imputed_data']).sum()}")
    logger.info(f"Valid windows: {(meta_df['is_valid']).sum()} / {len(meta_df)}")

    # Save to disk
    hourly_path = os.path.join(output_dir, "windows_hourly.npy")
    minute_path = os.path.join(output_dir, "windows_minute.npy")
    meta_path = os.path.join(output_dir, "window_metadata.csv")
    summary_path = os.path.join(output_dir, "window_summary.json")

    np.save(hourly_path, final_hourly)
    np.save(minute_path, final_minute)
    meta_df.to_csv(meta_path, index=False)

    summary = {
        "total_windows": int(len(meta_df)),
        "condition_windows": int((meta_df["label"] == 1).sum()),
        "control_windows": int((meta_df["label"] == 0).sum()),
        "hourly_shape": list(final_hourly.shape),
        "minute_shape": list(final_minute.shape),
        "windows_with_imputed_data": int(meta_df["has_imputed_data"].sum()),
        "valid_windows": int(meta_df["is_valid"].sum()),
        "participants_count": int(meta_df["participant_id"].nunique()),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved processed windows and metadata to {output_dir}")
    return final_hourly, final_minute, meta_df


if __name__ == "__main__":
    create_all_windows("configs/data.yaml")
