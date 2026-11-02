"""
Statistical Feature Extraction Module for Actigraphy Depression Detection.

Extracts 21 hand-crafted, clinically grounded actigraphy features for each 24-hour window:
1. Descriptive Statistics (9):
   - mean_activity, std_activity, median_activity, q25_activity, q75_activity,
     iqr_activity, max_activity, skewness, kurtosis
2. Activity Dynamics & Volatility (5):
   - zero_ratio, cv_activity, mean_abs_diff, rmssd, autocorr_lag1
3. Circadian & Diurnal Metrics (7):
   - day_mean (08:00-20:00), night_mean (20:00-08:00),
     day_zero_ratio, night_zero_ratio,
     m10 (10 most active hours), l5 (5 least active hours),
     relative_amplitude ((M10 - L5) / (M10 + L5))

Supports both:
- Full 21-feature set
- 3-feature baseline subset (mean, std, zero_ratio) matching Arora et al. (2023)
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.clean_data import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_CATEGORIES = {
    "descriptive": [
        "mean_activity",
        "std_activity",
        "median_activity",
        "q25_activity",
        "q75_activity",
        "iqr_activity",
        "max_activity",
        "skewness",
        "kurtosis",
    ],
    "variability": [
        "zero_ratio",
        "cv_activity",
        "mean_abs_diff",
        "rmssd",
        "autocorr_lag1",
    ],
    "circadian": [
        "day_mean",
        "night_mean",
        "day_zero_ratio",
        "night_zero_ratio",
        "m10",
        "l5",
        "relative_amplitude",
    ],
}

BASELINE_3_FEATURES = ["mean_activity", "std_activity", "zero_ratio"]


def compute_m10_l5(hourly_sequence: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute non-parametric circadian metrics (M10, L5, Relative Amplitude)
    from a 24-hour sequence of hourly averages.
    
    Parameters:
        hourly_sequence: 1D array of 24 hourly activity values.
    Returns:
        m10: Maximum average activity across 10 consecutive hours.
        l5: Minimum average activity across 5 consecutive hours.
        ra: Relative Amplitude (M10 - L5) / (M10 + L5 + 1e-5).
    """
    assert len(hourly_sequence) == 24, f"Expected 24 hourly points, got {len(hourly_sequence)}"
    # Circular wrap-around to allow intervals crossing midnight
    h_ext = np.concatenate([hourly_sequence, hourly_sequence[:10]])

    m10_candidates = [np.mean(h_ext[i : i + 10]) for i in range(24)]
    m10 = float(np.max(m10_candidates))

    l5_candidates = [np.mean(h_ext[i : i + 5]) for i in range(24)]
    l5 = float(np.min(l5_candidates))

    ra = float((m10 - l5) / (m10 + l5 + 1e-5))
    return m10, l5, ra


def extract_features_for_window(
    minute_vals: np.ndarray,
    hourly_vals: np.ndarray,
    start_ts_str: str,
) -> Dict[str, float]:
    """
    Extract all 21 statistical features for a single 24-hour window.
    """
    assert len(minute_vals) == 1440, f"Expected 1440 minutes, got {len(minute_vals)}"

    # 1. Descriptive Statistics
    mean_val = float(np.mean(minute_vals))
    std_val = float(np.std(minute_vals, ddof=1)) if len(minute_vals) > 1 else 0.0
    median_val = float(np.median(minute_vals))
    max_val = float(np.max(minute_vals))
    q25_val = float(np.percentile(minute_vals, 25))
    q75_val = float(np.percentile(minute_vals, 75))
    iqr_val = float(q75_val - q25_val)
    skew_val = float(stats.skew(minute_vals))
    kurt_val = float(stats.kurtosis(minute_vals))

    # 2. Activity & Variability Metrics
    zero_ratio = float(np.mean(minute_vals == 0))
    cv_val = float(std_val / (mean_val + 1e-5))
    diffs = np.diff(minute_vals)
    mean_abs_diff = float(np.mean(np.abs(diffs)))
    rmssd_val = float(np.sqrt(np.mean(diffs ** 2)))

    # Lag-1 Autocorrelation
    if std_val > 1e-5:
        autocorr_val = float(np.corrcoef(minute_vals[:-1], minute_vals[1:])[0, 1])
        if np.isnan(autocorr_val):
            autocorr_val = 0.0
    else:
        autocorr_val = 0.0

    # 3. Diurnal & Circadian Metrics
    start_ts = pd.to_datetime(start_ts_str)
    ts_range = pd.date_range(start=start_ts, periods=1440, freq="1min")
    hours = ts_range.hour

    day_mask = (hours >= 8) & (hours < 20)
    night_mask = ~day_mask

    day_vals = minute_vals[day_mask]
    night_vals = minute_vals[night_mask]

    day_mean = float(np.mean(day_vals)) if len(day_vals) > 0 else 0.0
    night_mean = float(np.mean(night_vals)) if len(night_vals) > 0 else 0.0
    day_zero_ratio = float(np.mean(day_vals == 0)) if len(day_vals) > 0 else 0.0
    night_zero_ratio = float(np.mean(night_vals == 0)) if len(night_vals) > 0 else 0.0

    m10_val, l5_val, ra_val = compute_m10_l5(hourly_vals)

    return {
        "mean_activity": mean_val,
        "std_activity": std_val,
        "median_activity": median_val,
        "q25_activity": q25_val,
        "q75_activity": q75_val,
        "iqr_activity": iqr_val,
        "max_activity": max_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "zero_ratio": zero_ratio,
        "cv_activity": cv_val,
        "mean_abs_diff": mean_abs_diff,
        "rmssd": rmssd_val,
        "autocorr_lag1": autocorr_val,
        "day_mean": day_mean,
        "night_mean": night_mean,
        "day_zero_ratio": day_zero_ratio,
        "night_zero_ratio": night_zero_ratio,
        "m10": m10_val,
        "l5": l5_val,
        "relative_amplitude": ra_val,
    }


def extract_all_statistical_features(
    minute_path: str = "data/processed/windows_minute.npy",
    hourly_path: str = "data/processed/windows_hourly.npy",
    metadata_path: str = "data/processed/window_metadata.csv",
    output_path: str = "data/processed/features_statistical.csv",
) -> pd.DataFrame:
    """
    Extract features for all 1,331 windows and persist to CSV.
    """
    logger.info("Loading preprocessed window arrays...")
    minute_arr = np.load(minute_path)[:, :, 0]
    hourly_arr = np.load(hourly_path)[:, :, 0]
    meta_df = pd.read_csv(metadata_path)

    n_windows = len(meta_df)
    assert len(minute_arr) == n_windows
    assert len(hourly_arr) == n_windows

    logger.info(f"Extracting 21 statistical features across {n_windows} windows...")

    records = []
    for i in range(n_windows):
        feats = extract_features_for_window(
            minute_vals=minute_arr[i],
            hourly_vals=hourly_arr[i],
            start_ts_str=meta_df.loc[i, "start_timestamp"],
        )
        # Attach metadata
        feats["window_id"] = meta_df.loc[i, "window_id"]
        feats["participant_id"] = meta_df.loc[i, "participant_id"]
        feats["group"] = meta_df.loc[i, "group"]
        feats["label"] = int(meta_df.loc[i, "label"])
        records.append(feats)

    features_df = pd.DataFrame(records)

    # Reorder columns: metadata first, then features
    all_feature_cols = (
        FEATURE_CATEGORIES["descriptive"]
        + FEATURE_CATEGORIES["variability"]
        + FEATURE_CATEGORIES["circadian"]
    )
    col_order = ["window_id", "participant_id", "group", "label"] + all_feature_cols
    features_df = features_df[col_order]

    # Verify no NaNs or Infs
    null_count = features_df[all_feature_cols].isna().sum().sum()
    inf_count = np.isinf(features_df[all_feature_cols].values).sum()
    assert null_count == 0, f"Detected {null_count} NaNs in extracted features"
    assert inf_count == 0, f"Detected {inf_count} Infs in extracted features"

    features_df.to_csv(output_path, index=False)
    logger.info(f"Saved extracted features table (shape: {features_df.shape}) to {output_path}")

    # Save feature names spec
    feature_meta_path = "data/processed/feature_metadata.json"
    with open(feature_meta_path, "w") as f:
        json.dump(
            {
                "total_features": len(all_feature_cols),
                "feature_names_all": all_feature_cols,
                "categories": FEATURE_CATEGORIES,
                "baseline_3_features": BASELINE_3_FEATURES,
            },
            f,
            indent=2,
        )

    return features_df


if __name__ == "__main__":
    df = extract_all_statistical_features()
    print("Features extracted successfully.")
