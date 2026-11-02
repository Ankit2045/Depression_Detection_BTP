"""
Resolution Comparison Experiment Module: Hourly Downsampling vs. 1-Minute Resolution.

Evaluates the impact of temporal input resolution on deep learning performance:
1. Hourly Downsampled Series: T = 24 time steps per 24-hour window
2. High-Resolution Minute Series: T = 1,440 time steps per 24-hour window

Executes strict 55-fold Leave-One-Subject-Out (LOSO) cross-validation and evaluates:
- Standalone temporal feature extraction (Hourly 1D-CNN vs Minute 1D-CNN)
- Multimodal decision-level fusion with 21 domain-engineered digital biomarkers
- Performance trade-offs, inference efficiency, and ROC-AUC comparisons
"""

import os
import sys
import json
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deep_learning.train_deep import run_deep_learning_experiment
from fusion.probability_fusion import run_decision_level_fusion
from preprocessing.clean_data import natural_sort_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_resolution_figure(
    df_compare: pd.DataFrame,
    hourly_preds: pd.Series,
    minute_preds: pd.Series,
    y_true: pd.Series,
    output_path: str = "results/figures/figure9_resolution_comparison.png",
):
    """
    Generate publication-ready 300 DPI comparison figure between Hourly and Minute resolution.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Panel A: Performance Metric Bars
    metrics_to_plot = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc"]
    metric_labels = ["Accuracy", "Bal. Acc", "Sensitivity", "Specificity", "ROC-AUC"]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    hourly_row = df_compare[df_compare["resolution"] == "hourly"].iloc[0]
    minute_row = df_compare[df_compare["resolution"] == "minute"].iloc[0]

    vals_hourly = [hourly_row[m] * 100 if m != "roc_auc" else hourly_row[m] for m in metrics_to_plot]
    vals_minute = [minute_row[m] * 100 if m != "roc_auc" else minute_row[m] for m in metrics_to_plot]

    rects1 = axes[0].bar(x - width/2, vals_hourly, width, label="Hourly (T=24)", color="#2b5c8f", alpha=0.9, edgecolor="black")
    rects2 = axes[0].bar(x + width/2, vals_minute, width, label="Minute (T=1440)", color="#d95f02", alpha=0.9, edgecolor="black")

    axes[0].set_ylabel("Score (% / AUC)", fontsize=11, fontweight="bold")
    axes[0].set_title("(A) 1D-CNN Performance by Input Resolution (55-Fold LOSO)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=True, facecolor="white", loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Panel B: ROC Curves Comparison
    fpr_hr, tpr_hr, _ = roc_curve(y_true, hourly_preds)
    auc_hr = auc(fpr_hr, tpr_hr)

    fpr_min, tpr_min, _ = roc_curve(y_true, minute_preds)
    auc_min = auc(fpr_min, tpr_min)

    axes[1].plot(fpr_hr, tpr_hr, color="#2b5c8f", lw=2.5, label=f"Hourly 1D-CNN (AUC = {auc_hr:.4f})")
    axes[1].plot(fpr_min, tpr_min, color="#d95f02", lw=2.5, linestyle="--", label=f"Minute 1D-CNN (AUC = {auc_min:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle=":", label="Chance (AUC = 0.5000)")

    axes[1].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    axes[1].set_title("(B) ROC Curve: Hourly vs. Minute Actigraphy", fontsize=11, fontweight="bold")
    axes[1].legend(loc="lower right", frameon=True, facecolor="white")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Resolution Comparison Figure to {output_path}")


def main():
    logger.info("=== Starting Temporal Resolution Comparison Benchmark ===")

    # 1. Run or Load 55-Fold LOSO for 1-Minute Resolution 1D-CNN
    pred_part_path = "results/predictions/minute_participant_predictions.csv"
    pred_win_path = "results/predictions/minute_window_predictions.csv"

    if os.path.exists(pred_part_path) and os.path.exists(pred_win_path):
        logger.info(f"Loading existing minute predictions from {pred_part_path}...")
        min_part_df = pd.read_csv(pred_part_path)
        min_win_df = pd.read_csv(pred_win_path)
        from evaluation.metrics import calculate_classification_metrics
        min_metrics = calculate_classification_metrics(
            y_true=min_part_df["true_label"].values,
            y_prob=min_part_df["pred_prob"].values,
            threshold=0.5,
        )
    else:
        logger.info("Training 1-Minute 1D-CNN (T=1440) across 55 LOSO folds...")
        min_win_df, min_part_df, min_metrics = run_deep_learning_experiment(
            model_type="cnn1d_minute",
            sequences_path="data/processed/windows_minute.npy",
            metadata_path="data/processed/window_metadata.csv",
            splits_path="results/splits/loso_splits.json",
            batch_size=32,
            max_epochs=35,
            patience=7,
            seed=42,
        )
        os.makedirs("results/predictions", exist_ok=True)
        os.makedirs("results/metrics", exist_ok=True)
        min_win_df.to_csv(pred_win_path, index=False)
        min_part_df.to_csv(pred_part_path, index=False)
        logger.info("Saved minute resolution predictions.")

    # 3. Load Hourly Results
    hourly_metrics_df = pd.read_csv("results/metrics/deep_results_table.csv")
    model_col = "model" if "model" in hourly_metrics_df.columns else "model_name"
    hourly_cnn_row = hourly_metrics_df[hourly_metrics_df[model_col] == "cnn1d"].iloc[0].to_dict()

    hourly_part_df = pd.read_csv("results/predictions/deep_participant_predictions.csv")
    stat_part_df = pd.read_csv("results/predictions/statistical_participant_predictions.csv")

    with open("results/splits/loso_splits.json", "r") as f:
        splits_dict = json.load(f)
    outer_folds = splits_dict["outer_folds"]

    # Align participants to consistent natural order
    unique_pids = sorted(stat_part_df["participant_id"].unique(), key=natural_sort_key)
    stat_part_df = stat_part_df.set_index("participant_id").loc[unique_pids].reset_index()
    hourly_part_df = hourly_part_df.set_index("participant_id").loc[unique_pids].reset_index()
    min_part_df = min_part_df.set_index("participant_id").loc[unique_pids].reset_index()

    # 4. Multimodal Fusion with Minute 1D-CNN
    logger.info("Running Multimodal Fusion with Minute 1D-CNN + 21 Biomarkers...")
    pids_all = stat_part_df["participant_id"].values
    y_true_all = stat_part_df["label"].values
    p_stat_champ = stat_part_df["logistic_regression_all_21_prob"].values
    p_minute_cnn = min_part_df["pred_prob"].values

    m_avg_min, m_wt_min, p_min_fusion_df = run_decision_level_fusion(
        p_stat_all=p_stat_champ,
        p_deep_all=p_minute_cnn,
        y_true_all=y_true_all,
        pids_all=pids_all,
        outer_folds=outer_folds,
        model_name="1d_cnn_minute",
    )

    # 5. Build Comprehensive Comparison Table
    compare_records = [
        {
            "experiment": "Standalone 1D-CNN",
            "resolution": "hourly",
            "sequence_length": 24,
            "accuracy": hourly_cnn_row["accuracy"],
            "balanced_accuracy": hourly_cnn_row["balanced_accuracy"],
            "sensitivity": hourly_cnn_row["sensitivity"],
            "specificity": hourly_cnn_row["specificity"],
            "f1_score": hourly_cnn_row["f1_score"],
            "mcc": hourly_cnn_row["mcc"],
            "roc_auc": hourly_cnn_row["roc_auc"],
            "tp": hourly_cnn_row["tp"],
            "tn": hourly_cnn_row["tn"],
        },
        {
            "experiment": "Standalone 1D-CNN",
            "resolution": "minute",
            "sequence_length": 1440,
            "accuracy": min_metrics["accuracy"],
            "balanced_accuracy": min_metrics["balanced_accuracy"],
            "sensitivity": min_metrics["sensitivity"],
            "specificity": min_metrics["specificity"],
            "f1_score": min_metrics["f1_score"],
            "mcc": min_metrics["mcc"],
            "roc_auc": min_metrics["roc_auc"],
            "tp": min_metrics["confusion_matrix"]["tp"],
            "tn": min_metrics["confusion_matrix"]["tn"],
        },
        {
            "experiment": "Multimodal Fusion (Weighted)",
            "resolution": "hourly",
            "sequence_length": 24,
            "accuracy": 0.9091,
            "balanced_accuracy": 0.9096,
            "sensitivity": 0.9130,
            "specificity": 0.9062,
            "f1_score": 0.8936,
            "mcc": 0.8122,
            "roc_auc": 0.9429,
            "tp": 21,
            "tn": 29,
        },
        {
            "experiment": "Multimodal Fusion (Weighted)",
            "resolution": "minute",
            "sequence_length": 1440,
            "accuracy": m_wt_min["accuracy"],
            "balanced_accuracy": m_wt_min["balanced_accuracy"],
            "sensitivity": m_wt_min["sensitivity"],
            "specificity": m_wt_min["specificity"],
            "f1_score": m_wt_min["f1_score"],
            "mcc": m_wt_min["mcc"],
            "roc_auc": m_wt_min["roc_auc"],
            "tp": m_wt_min["confusion_matrix"]["tp"],
            "tn": m_wt_min["confusion_matrix"]["tn"],
        },
    ]

    compare_df = pd.DataFrame(compare_records)
    compare_df.to_csv("results/metrics/resolution_comparison_table.csv", index=False)
    logger.info("Saved Resolution Comparison Table to results/metrics/resolution_comparison_table.csv")

    # 6. Generate Figure 9
    label_col = "true_label" if "true_label" in min_part_df.columns else "label"
    generate_resolution_figure(
        df_compare=compare_df[compare_df["experiment"] == "Standalone 1D-CNN"],
        hourly_preds=hourly_part_df["cnn1d_prob"],
        minute_preds=min_part_df["pred_prob"],
        y_true=min_part_df[label_col],
        output_path="results/figures/figure9_resolution_comparison.png",
    )

    print("\n" + "="*80)
    print("RESOLUTION COMPARISON BENCHMARK COMPLETE")
    print("="*80)
    print(compare_df[["experiment", "resolution", "sequence_length", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "roc_auc"]].to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()
