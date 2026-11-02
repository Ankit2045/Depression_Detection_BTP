"""
Clean Data Module for Actigraphy Depresjon Dataset.

Responsibilities:
1. Load raw participant CSV files (timestamp, date, activity).
2. Truncate recording to the first N days specified in scores.csv (when configured).
3. Align to a continuous 1-minute time grid.
4. Detect missing intervals:
   - Interpolate short gaps (<= 65 minutes, e.g. Daylight Saving Time clock transition)
   - Track boolean indicator of imputed values.
   - Flag or reject sequences with unresolvable gaps (> 65 minutes).
5. Ensure zero-activity observations are strictly distinguished from missing data.
6. Validate monotonicity, length, and non-negativity.
"""

import os
import glob
import logging
from typing import Dict, Tuple, Optional
import yaml
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/data.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_participant_metadata(scores_path: str = "data/scores.csv") -> pd.DataFrame:
    """Load and normalize scores.csv."""
    df = pd.read_csv(scores_path)
    df["number"] = df["number"].astype(str).str.strip()
    return df.set_index("number")


def natural_sort_key(filepath_or_id: str) -> Tuple[int, int]:
    """
    Sort participant IDs or file paths numerically (natural sort):
    e.g. condition_1, condition_2, ..., condition_23, control_1, ..., control_32
    rather than alphabetical order (condition_1, condition_10, condition_2).
    """
    basename = os.path.splitext(os.path.basename(filepath_or_id))[0]
    parts = basename.split("_")
    group_order = 0 if parts[0] == "condition" else 1
    num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (group_order, num)


def clean_participant_series(
    filepath: str,
    days_limit: Optional[int] = None,
    max_gap_minutes: int = 65,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean and regularize a single participant's actigraphy time series.

    Returns:
        cleaned_df: DataFrame with columns ['timestamp', 'activity', 'is_imputed']
                    indexed by continuous 1-minute DatetimeIndex.
        meta: Dictionary of cleaning statistics and integrity checks.
    """
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    t0 = df["timestamp"].iloc[0]

    # Apply days truncation if specified (scores_days policy)
    if days_limit is not None and days_limit > 0:
        t_end_expected = t0 + pd.Timedelta(days=days_limit)
        # Filter raw rows within the [t0, t_end_expected) interval
        df = df[(df["timestamp"] >= t0) & (df["timestamp"] < t_end_expected)].copy()
        target_minutes = days_limit * 1440
        grid_end = t0 + pd.Timedelta(minutes=target_minutes - 1)
    else:
        grid_end = df["timestamp"].iloc[-1]
        target_minutes = int((grid_end - t0).total_seconds() / 60.0) + 1

    # Build continuous 1-minute time grid
    full_grid = pd.date_range(start=t0, end=grid_end, freq="1min")
    
    # Check for raw duplicates before reindexing
    dup_count = int(df["timestamp"].duplicated().sum())
    if dup_count > 0:
        logger.warning(f"Dropping {dup_count} duplicate timestamps in {filepath}")
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # Reindex to regular 1-minute grid
    df_indexed = df.set_index("timestamp").reindex(full_grid)
    
    missing_mask = df_indexed["activity"].isna()
    raw_missing_count = int(missing_mask.sum())

    # Check gap lengths
    if raw_missing_count > 0:
        # Find consecutive missing blocks
        missing_blocks = (~missing_mask).cumsum()[missing_mask]
        max_block_size = int(missing_blocks.value_counts().max()) if len(missing_blocks) > 0 else 0
        if max_block_size > max_gap_minutes:
            raise ValueError(
                f"File {filepath} contains a missing gap of {max_block_size} minutes, "
                f"exceeding maximum allowable threshold of {max_gap_minutes} minutes."
            )
        # Linear interpolation for acceptable gaps (e.g. DST jump)
        df_indexed["activity"] = df_indexed["activity"].interpolate(method="linear", limit_direction="both")
        # Ensure non-negative after interpolation
        df_indexed["activity"] = df_indexed["activity"].clip(lower=0.0)
    else:
        max_block_size = 0

    cleaned_df = pd.DataFrame({
        "timestamp": full_grid,
        "activity": df_indexed["activity"].values,
        "is_imputed": missing_mask.values,
    })

    cleaning_meta = {
        "start_timestamp": str(t0),
        "end_timestamp": str(grid_end),
        "total_minutes": len(cleaned_df),
        "expected_minutes": target_minutes,
        "raw_missing_minutes": raw_missing_count,
        "max_missing_gap_minutes": max_block_size,
        "imputed_minutes": raw_missing_count,
        "zero_activity_minutes": int((cleaned_df["activity"] == 0).sum()),
        "mean_activity": float(cleaned_df["activity"].mean()),
        "std_activity": float(cleaned_df["activity"].std()),
    }

    assert len(cleaned_df) == target_minutes, (
        f"Length mismatch: cleaned length {len(cleaned_df)} != expected {target_minutes}"
    )

    return cleaned_df, cleaning_meta


def clean_all_participants(config_path: str = "configs/data.yaml") -> Dict[str, pd.DataFrame]:
    """Clean all 55 participants and return a dict of participant DataFrames."""
    config = load_config(config_path)
    scores_meta = load_participant_metadata(config["dataset"]["scores_file"])

    use_scores_days = config.get("recording_duration_scope", "scores_days") == "scores_days"
    max_gap = config["missing_data_policy"].get("short_gap_threshold_minutes", 65)

    condition_files = sorted(
        glob.glob(os.path.join(config["dataset"]["condition_dir"], "*.csv")),
        key=natural_sort_key,
    )
    control_files = sorted(
        glob.glob(os.path.join(config["dataset"]["control_dir"], "*.csv")),
        key=natural_sort_key,
    )
    all_files = condition_files + control_files

    cleaned_participants = {}
    summaries = []

    for filepath in all_files:
        pid = os.path.splitext(os.path.basename(filepath))[0]
        days_limit = int(scores_meta.loc[pid, "days"]) if use_scores_days else None
        
        cleaned_df, meta = clean_participant_series(
            filepath=filepath,
            days_limit=days_limit,
            max_gap_minutes=max_gap,
        )
        cleaned_participants[pid] = cleaned_df
        meta["participant_id"] = pid
        summaries.append(meta)

    summary_df = pd.DataFrame(summaries)
    logger.info(
        f"Successfully cleaned {len(cleaned_participants)} participants. "
        f"Total minutes: {summary_df['total_minutes'].sum():,}. "
        f"Imputed minutes across all subjects: {summary_df['imputed_minutes'].sum()} "
        f"({summary_df['imputed_minutes'].sum() / summary_df['total_minutes'].sum() * 100:.4f}% of data)."
    )

    return cleaned_participants


if __name__ == "__main__":
    cleaned = clean_all_participants("configs/data.yaml")
    print(f"Cleaned {len(cleaned)} participant series successfully.")
