"""
Explainable AI (XAI) & Digital Biomarker Discovery Module.

Implements leak-free SHAP (SHapley Additive exPlanations) analysis for actigraphy-based
depression detection across the 55-fold Leave-One-Subject-Out (LOSO) cross-validation protocol.

Features analyzed: 21 clinically grounded digital biomarkers across 3 families:
1. Circadian Rhythmicity (Relative Amplitude, M10, L5)
2. Volatility & Regularity (CV, RMSSD, Mean Absolute Difference, Autocorrelation, Skewness, Kurtosis)
3. Descriptive Activity Statistics (Day Mean, Night Mean, Zero Ratios, Median, IQR, Max, etc.)

Outputs:
- results/metrics/shap_feature_importance.csv: Complete biomarker rankings, mean |SHAP|,
  odds ratios, direction of effect, and clinical interpretation.
- results/figures/figure8_shap_summary.png: Publication-ready 300 DPI multi-panel figure.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.clean_data import natural_sort_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BIOMARKER_METADATA = {
    # Circadian Rhythm Family
    "relative_amplitude": {
        "family": "Circadian Rhythm",
        "display_name": "Relative Amplitude (RA)",
        "clinical_role": "Circadian rhythm vigor: (M10 - L5) / (M10 + L5). Blunted in mood disorders.",
    },
    "m10": {
        "family": "Circadian Rhythm",
        "display_name": "M10 (Daytime Peak Activity)",
        "clinical_role": "Mean motor activity during the 10 most active hours. Reduced by psychomotor retardation.",
    },
    "l5": {
        "family": "Circadian Rhythm",
        "display_name": "L5 (Nocturnal Rest Activity)",
        "clinical_role": "Mean motor activity during the 5 least active hours. Elevated by insomnia and nocturnal restlessness.",
    },

    # Volatility & Regularity Family
    "cv_activity": {
        "family": "Volatility & Regularity",
        "display_name": "Coefficient of Variation (CV)",
        "clinical_role": "Ratio of standard deviation to mean. Measures relative behavioral unpredictability.",
    },
    "rmssd": {
        "family": "Volatility & Regularity",
        "display_name": "RMSSD",
        "clinical_role": "Root mean square of successive differences. Captures micro-scale motor fragmentation.",
    },
    "mean_abs_diff": {
        "family": "Volatility & Regularity",
        "display_name": "Mean Absolute Diff",
        "clinical_role": "Average epoch-to-epoch absolute activity jump.",
    },
    "autocorr_lag1": {
        "family": "Volatility & Regularity",
        "display_name": "Autocorrelation (Lag-1)",
        "clinical_role": "Behavioral inertia / temporal routine persistence minute-to-minute.",
    },
    "skewness": {
        "family": "Volatility & Regularity",
        "display_name": "Activity Skewness",
        "clinical_role": "Asymmetry of motor distribution; reflects burstiness of physical engagement.",
    },
    "kurtosis": {
        "family": "Volatility & Regularity",
        "display_name": "Activity Kurtosis",
        "clinical_role": "Heavy-tailedness of motor distribution; frequency of extreme activity spikes.",
    },

    # Descriptive Statistics Family
    "mean_activity": {
        "family": "Descriptive Statistics",
        "display_name": "Mean Activity",
        "clinical_role": "Global 24-hour average movement count.",
    },
    "std_activity": {
        "family": "Descriptive Statistics",
        "display_name": "Std Activity",
        "clinical_role": "Global 24-hour motor variability.",
    },
    "median_activity": {
        "family": "Descriptive Statistics",
        "display_name": "Median Activity",
        "clinical_role": "Baseline habitual motor intensity (50th percentile).",
    },
    "q25_activity": {
        "family": "Descriptive Statistics",
        "display_name": "25th Percentile Activity",
        "clinical_role": "Low-intensity sedentary threshold.",
    },
    "q75_activity": {
        "family": "Descriptive Statistics",
        "display_name": "75th Percentile Activity",
        "clinical_role": "Active engagement threshold.",
    },
    "iqr_activity": {
        "family": "Descriptive Statistics",
        "display_name": "IQR Activity",
        "clinical_role": "Mid-spread of daily physical activity.",
    },
    "max_activity": {
        "family": "Descriptive Statistics",
        "display_name": "Max Peak Activity",
        "clinical_role": "Highest recorded single-epoch motor surge in 24 hours.",
    },
    "zero_ratio": {
        "family": "Descriptive Statistics",
        "display_name": "Zero-Activity Ratio",
        "clinical_role": "Proportion of completely inactive minutes. Reflects sedentary immobility.",
    },
    "day_mean": {
        "family": "Descriptive Statistics",
        "display_name": "Diurnal Mean Activity",
        "clinical_role": "Average activity during daytime hours (08:00 - 20:00).",
    },
    "night_mean": {
        "family": "Descriptive Statistics",
        "display_name": "Nocturnal Mean Activity",
        "clinical_role": "Average activity during nocturnal hours (00:00 - 06:00).",
    },
    "day_zero_ratio": {
        "family": "Descriptive Statistics",
        "display_name": "Daytime Sedentary Ratio",
        "clinical_role": "Proportion of daytime minutes spent inactive.",
    },
    "night_zero_ratio": {
        "family": "Descriptive Statistics",
        "display_name": "Nighttime Sedentary Ratio",
        "clinical_role": "Proportion of nocturnal minutes spent inactive (sleep restfulness).",
    },
}


def compute_loso_shap_values(
    features_csv: str = "data/processed/features_statistical.csv",
    splits_json: str = "results/splits/loso_splits.json",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Compute out-of-fold SHAP values across all 55 LOSO folds for Logistic Regression.
    Guarantees zero data leakage: each participant's SHAP values are evaluated by
    a model trained strictly without that participant.
    """
    df = pd.read_csv(features_csv)
    with open(splits_json, "r") as f:
        splits = json.load(f)

    outer_folds = splits["outer_folds"]
    feature_cols = [c for c in df.columns if c not in ["window_id", "participant_id", "group", "label"]]

    # Participant-level feature aggregation (mean profile per participant)
    part_df = (
        df.groupby(["participant_id", "group", "label"])[feature_cols]
        .mean()
        .reset_index()
    )
    pids_sorted = sorted(part_df["participant_id"].unique(), key=natural_sort_key)
    part_df = part_df.set_index("participant_id").loc[pids_sorted].reset_index()

    N = len(part_df)
    D = len(feature_cols)

    oof_shap_matrix = np.zeros((N, D), dtype=np.float64)
    oof_standardized_X = np.zeros((N, D), dtype=np.float64)
    coef_list = []

    for fold in outer_folds:
        test_pid = fold["test_participant"]
        test_idx = part_df[part_df["participant_id"] == test_pid].index[0]
        train_mask = part_df["participant_id"] != test_pid

        X_train_raw = part_df.loc[train_mask, feature_cols].values
        y_train = part_df.loc[train_mask, "label"].values
        X_test_raw = part_df.loc[[test_idx], feature_cols].values

        # Training-only standardization
        mean_tr = X_train_raw.mean(axis=0)
        std_tr = X_train_raw.std(axis=0) + 1e-8

        X_train_norm = (X_train_raw - mean_tr) / std_tr
        X_test_norm = (X_test_raw - mean_tr) / std_tr

        oof_standardized_X[test_idx, :] = X_test_norm[0]

        # Fit Logistic Regression on training fold
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_norm, y_train)
        coef_list.append(clf.coef_[0])

        # Analytical Linear SHAP: SHAP_j(x) = beta_j * (x_j - E[X_j])
        # In standardized space where E[X_j] = 0, SHAP_j(x) = beta_j * x_j
        shap_explainer = shap.LinearExplainer(clf, X_train_norm)
        fold_shap = shap_explainer.shap_values(X_test_norm)
        oof_shap_matrix[test_idx, :] = fold_shap[0]

    mean_coefs = np.mean(coef_list, axis=0)
    return part_df, oof_shap_matrix, oof_standardized_X, feature_cols, mean_coefs


def generate_shap_report(
    part_df: pd.DataFrame,
    shap_matrix: np.ndarray,
    std_X: np.ndarray,
    feature_cols: List[str],
    mean_coefs: np.ndarray,
    output_csv: str = "results/metrics/shap_feature_importance.csv",
) -> pd.DataFrame:
    """
    Aggregate SHAP values, compute statistical significance and odds ratios,
    and save the comprehensive biomarker importance table.
    """
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    std_shap = np.std(shap_matrix, axis=0)

    depressed_mask = part_df["label"].values == 1
    control_mask = part_df["label"].values == 0

    records = []
    for j, f in enumerate(feature_cols):
        meta = BIOMARKER_METADATA.get(f, {"family": "Other", "display_name": f, "clinical_role": ""})
        vals_dep = part_df.loc[depressed_mask, f].values
        vals_ctrl = part_df.loc[control_mask, f].values

        # Two-sample Mann-Whitney U test (non-parametric)
        u_stat, p_val = stats.mannwhitneyu(vals_dep, vals_ctrl, alternative="two-sided")
        # Cohen's d effect size
        pooled_sd = np.sqrt(((len(vals_dep)-1)*np.var(vals_dep, ddof=1) + (len(vals_ctrl)-1)*np.var(vals_ctrl, ddof=1)) / (len(vals_dep)+len(vals_ctrl)-2))
        cohens_d = (np.mean(vals_dep) - np.mean(vals_ctrl)) / (pooled_sd + 1e-8)

        direction = "Positive (Risk Factor)" if mean_coefs[j] > 0 else "Negative (Protective Factor)"
        odds_ratio = float(np.exp(mean_coefs[j]))

        records.append({
            "feature": f,
            "display_name": meta["display_name"],
            "family": meta["family"],
            "mean_abs_shap": float(mean_abs_shap[j]),
            "std_shap": float(std_shap[j]),
            "mean_coefficient": float(mean_coefs[j]),
            "odds_ratio": odds_ratio,
            "direction": direction,
            "depressed_mean": float(np.mean(vals_dep)),
            "control_mean": float(np.mean(vals_ctrl)),
            "cohens_d": float(cohens_d),
            "p_value": float(p_val),
            "clinical_role": meta["clinical_role"],
        })

    report_df = pd.DataFrame(records)
    report_df = report_df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    report_df["rank"] = np.arange(1, len(report_df) + 1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    report_df.to_csv(output_csv, index=False)
    logger.info(f"Saved SHAP Feature Importance Table to {output_csv}")
    return report_df


def generate_shap_figures(
    report_df: pd.DataFrame,
    part_df: pd.DataFrame,
    shap_matrix: np.ndarray,
    std_X: np.ndarray,
    feature_cols: List[str],
    output_path: str = "results/figures/figure8_shap_summary.png",
):
    """
    Generate publication-ready 300 DPI multi-panel figure for Explainable AI.
    - Panel A: Global SHAP Feature Importance Bar Chart by Family
    - Panel B: Beeswarm Summary Plot showing directional impact
    - Panel C: Circadian Biomarker Violin / Box Distribution (Relative Amplitude, L5, M10)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = plt.figure(figsize=(15, 12), dpi=300)

    family_colors = {
        "Circadian Rhythm": "#2b5c8f",
        "Volatility & Regularity": "#2ca02c",
        "Descriptive Statistics": "#d95f02",
        "Other": "#7f7f7f",
    }

    # --- Panel A: Top 15 Biomarkers Bar Chart ---
    ax1 = fig.add_subplot(2, 2, 1)
    top15 = report_df.head(15).iloc[::-1]  # reverse for horizontal bar chart
    bar_colors = [family_colors.get(f, "#7f7f7f") for f in top15["family"]]

    y_pos = np.arange(len(top15))
    bars = ax1.barh(y_pos, top15["mean_abs_shap"], color=bar_colors, edgecolor="black", alpha=0.85, height=0.65)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top15["display_name"], fontsize=9, fontweight="bold")
    ax1.set_xlabel("Mean |SHAP Value| (Impact on Diagnostic Decision)", fontsize=10, fontweight="bold")
    ax1.set_title("(A) Top Digital Biomarkers by SHAP Importance", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    # Add custom legend for families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b5c8f", edgecolor="black", label="Circadian Rhythm"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Volatility & Regularity"),
        Patch(facecolor="#d95f02", edgecolor="black", label="Descriptive Statistics"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", fontsize=8)

    # --- Panel B: SHAP Beeswarm Summary Plot (Top 10 Features) ---
    ax2 = fig.add_subplot(2, 2, 2)
    top10_features = report_df.head(10)["feature"].tolist()
    top10_names = report_df.head(10)["display_name"].tolist()

    # Re-order top 10 for plotting (bottom to top)
    top10_features_rev = top10_features[::-1]
    top10_names_rev = top10_names[::-1]

    for idx, f in enumerate(top10_features_rev):
        col_idx = feature_cols.index(f)
        f_shap = shap_matrix[:, col_idx]
        f_val = std_X[:, col_idx]

        # Normalize feature values to [0, 1] for colormap
        norm_val = (f_val - np.min(f_val)) / (np.max(f_val) - np.min(f_val) + 1e-8)
        jitter = np.random.normal(0, 0.08, size=len(f_shap))
        scatter = ax2.scatter(
            f_shap,
            np.full_like(f_shap, idx) + jitter,
            c=norm_val,
            cmap="coolwarm",
            alpha=0.8,
            s=36,
            edgecolors="none",
        )

    ax2.axvline(0, color="black", linestyle="--", lw=1)
    ax2.set_yticks(np.arange(len(top10_names_rev)))
    ax2.set_yticklabels(top10_names_rev, fontsize=9, fontweight="bold")
    ax2.set_xlabel("SHAP Value (Log-Odds Impact: <0 Control, >0 Depressed)", fontsize=10, fontweight="bold")
    ax2.set_title("(B) Directional Impact of Top Biomarkers (Beeswarm)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax2, pad=0.02, shrink=0.8)
    cbar.set_label("Feature Value (Low: Blue -> High: Red)", fontsize=8, fontweight="bold")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    # --- Panel C: Circadian Biomarker Clinical Distributions ---
    ax3 = fig.add_subplot(2, 2, 3)
    ra_dep = part_df[part_df["label"] == 1]["relative_amplitude"].values
    ra_ctrl = part_df[part_df["label"] == 0]["relative_amplitude"].values

    box_data = [ra_ctrl, ra_dep]
    bplot1 = ax3.boxplot(box_data, patch_artist=True, widths=0.5,
                         boxprops=dict(facecolor="#2b5c8f", alpha=0.7),
                         medianprops=dict(color="black", lw=2))
    bplot1["boxes"][0].set_facecolor("#3274a1")
    bplot1["boxes"][1].set_facecolor("#e1812c")

    ax3.set_xticklabels(["Control\n(n=32)", "Depressed\n(n=23)"], fontsize=10, fontweight="bold")
    ax3.set_ylabel("Relative Amplitude (RA)", fontsize=10, fontweight="bold")
    ax3.set_title("(C) Circadian Relative Amplitude (RA)", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    # --- Panel D: Nocturnal Restlessness (L5) vs Daytime Vigor (M10) ---
    ax4 = fig.add_subplot(2, 2, 4)
    ctrl_mask = part_df["label"] == 0
    dep_mask = part_df["label"] == 1

    ax4.scatter(part_df.loc[ctrl_mask, "m10"], part_df.loc[ctrl_mask, "l5"],
                color="#3274a1", label="Control (Healthy)", s=55, alpha=0.85, edgecolors="black")
    ax4.scatter(part_df.loc[dep_mask, "m10"], part_df.loc[dep_mask, "l5"],
                color="#e1812c", label="Depressed (Condition)", s=65, marker="^", alpha=0.9, edgecolors="black")

    ax4.set_xlabel("M10: Daytime Peak Activity (mean counts/min)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("L5: Nocturnal Rest Activity (mean counts/min)", fontsize=10, fontweight="bold")
    ax4.set_title("(D) Diurnal Motor Vigor (M10) vs. Nocturnal Rest (L5)", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper right", frameon=True, facecolor="white")
    ax4.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Publication-Ready Explainable AI Figure to {output_path}")


def main():
    logger.info("=== Starting Explainable AI (SHAP) Biomarker Discovery ===")
    part_df, shap_mat, std_X, feats, mean_coefs = compute_loso_shap_values()
    report_df = generate_shap_report(part_df, shap_mat, std_X, feats, mean_coefs)
    generate_shap_figures(report_df, part_df, shap_mat, std_X, feats)

    print("\n" + "="*80)
    print("TOP 10 CLINICAL DIGITAL BIOMARKERS IDENTIFIED BY SHAP (55-FOLD LOSO):")
    print("="*80)
    print(report_df[["rank", "display_name", "family", "mean_abs_shap", "odds_ratio", "direction", "p_value"]].head(10).to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
