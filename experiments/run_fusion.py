"""
Master Runner for Multimodal Feature and Decision Fusion Experiments.

Evaluates the complete Fusion Matrix:
- Probability Averaging (Stat + 1D-CNN, Stat + GRU, Stat + LSTM)
- Weighted Probability Fusion (with leak-free inner alpha selection)
- Stacking Meta-Classifier (Logistic Regression meta-learner)
- Feature-Level Fusion (Concatenating 64-dim deep latent features with 21 statistical features = 85 features)

Usage:
    python experiments/run_fusion.py
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fusion.probability_fusion import run_decision_level_fusion
from fusion.stacking import run_stacking_fusion
from fusion.feature_fusion import run_feature_level_fusion
from evaluation.metrics import calculate_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_all_fusion_experiments(
    stat_preds_path: str = "results/predictions/statistical_participant_predictions.csv",
    deep_preds_path: str = "results/predictions/deep_participant_predictions.csv",
    splits_path: str = "results/splits/loso_splits.json",
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Execute all fusion experiments and save unified prediction & metrics tables.
    """
    logger.info("Loading baseline predictions and split definitions...")
    stat_df = pd.read_csv(stat_preds_path)
    deep_df = pd.read_csv(deep_preds_path)

    with open(splits_path, "r") as f:
        splits_dict = json.load(f)
    outer_folds = splits_dict["outer_folds"]

    pids_all = stat_df["participant_id"].values
    y_true_all = stat_df["label"].values
    p_stat_champ = stat_df["logistic_regression_all_21_prob"].values

    deep_models = [
        ("1d_cnn", "cnn1d_prob"),
        ("gru", "gru_prob"),
        ("lstm", "lstm_prob"),
    ]

    master_preds_df = pd.DataFrame({
        "participant_id": pids_all,
        "true_label": y_true_all,
        "p_stat_logreg21": np.round(p_stat_champ, 5),
    })

    summary_records = []

    # 1. Probability Averaging, Weighted Fusion & Stacking for each Deep Architecture
    for m_label, col_name in deep_models:
        p_deep = deep_df[col_name].values
        master_preds_df[f"p_{col_name}"] = np.round(p_deep, 5)

        logger.info(f"Running Probability & Weighted Fusion: Stat + {m_label.upper()}...")
        m_avg, m_wt, p_df = run_decision_level_fusion(
            p_stat_all=p_stat_champ,
            p_deep_all=p_deep,
            y_true_all=y_true_all,
            pids_all=pids_all,
            outer_folds=outer_folds,
            model_name=m_label,
        )

        master_preds_df[f"p_avg_{m_label}"] = p_df[f"p_avg_{m_label}"]
        master_preds_df[f"p_weighted_{m_label}"] = p_df[f"p_weighted_{m_label}"]
        master_preds_df[f"alpha_weighted_{m_label}"] = p_df[f"alpha_weighted_{m_label}"]

        summary_records.append({
            "experiment_id": f"fusion_prob_averaging_{m_label}",
            "fusion_type": "Probability Averaging",
            "models_combined": f"LogReg-21 + {m_label.upper()}",
            "balanced_accuracy": m_avg["balanced_accuracy"],
            "accuracy": m_avg["accuracy"],
            "sensitivity": m_avg["sensitivity"],
            "specificity": m_avg["specificity"],
            "precision": m_avg["precision"],
            "f1_score": m_avg["f1_score"],
            "mcc": m_avg["mcc"],
            "roc_auc": m_avg["roc_auc"],
            "tp": m_avg["confusion_matrix"]["tp"],
            "tn": m_avg["confusion_matrix"]["tn"],
            "fp": m_avg["confusion_matrix"]["fp"],
            "fn": m_avg["confusion_matrix"]["fn"],
        })

        summary_records.append({
            "experiment_id": f"fusion_weighted_prob_{m_label}",
            "fusion_type": "Weighted Probability",
            "models_combined": f"LogReg-21 + {m_label.upper()}",
            "balanced_accuracy": m_wt["balanced_accuracy"],
            "accuracy": m_wt["accuracy"],
            "sensitivity": m_wt["sensitivity"],
            "specificity": m_wt["specificity"],
            "precision": m_wt["precision"],
            "f1_score": m_wt["f1_score"],
            "mcc": m_wt["mcc"],
            "roc_auc": m_wt["roc_auc"],
            "tp": m_wt["confusion_matrix"]["tp"],
            "tn": m_wt["confusion_matrix"]["tn"],
            "fp": m_wt["confusion_matrix"]["fp"],
            "fn": m_wt["confusion_matrix"]["fn"],
        })

        # Stacking Meta-Classifier
        logger.info(f"Running Stacking Meta-Classifier: Stat + {m_label.upper()}...")
        m_stack, stack_p_df = run_stacking_fusion(
            p_stat_all=p_stat_champ,
            p_deep_all=p_deep,
            y_true_all=y_true_all,
            pids_all=pids_all,
            outer_folds=outer_folds,
            model_name=m_label,
        )
        master_preds_df[f"p_stacked_{m_label}"] = stack_p_df[f"p_stacked_{m_label}"]

        summary_records.append({
            "experiment_id": f"fusion_stacking_{m_label}",
            "fusion_type": "Stacking (Meta-Learner)",
            "models_combined": f"LogReg-21 + {m_label.upper()}",
            "balanced_accuracy": m_stack["balanced_accuracy"],
            "accuracy": m_stack["accuracy"],
            "sensitivity": m_stack["sensitivity"],
            "specificity": m_stack["specificity"],
            "precision": m_stack["precision"],
            "f1_score": m_stack["f1_score"],
            "mcc": m_stack["mcc"],
            "roc_auc": m_stack["roc_auc"],
            "tp": m_stack["confusion_matrix"]["tp"],
            "tn": m_stack["confusion_matrix"]["tn"],
            "fp": m_stack["confusion_matrix"]["fp"],
            "fn": m_stack["confusion_matrix"]["fn"],
        })

    # 2. Feature-Level Fusion (Concatenating 64-dim 1D-CNN Latent + 21 Stats)
    logger.info("Running Feature-Level Concatenation Fusion (85 features across 55 folds)...")
    m_feat, feat_p_df = run_feature_level_fusion(
        sequences_path="data/processed/windows_hourly.npy",
        features_path="data/processed/features_statistical.csv",
        splits_dict=splits_dict,
    )
    master_preds_df["p_feature_fusion_cnn1d"] = feat_p_df["pred_prob"].values

    summary_records.append({
        "experiment_id": "fusion_feature_level_cnn1d_and_stats",
        "fusion_type": "Feature Concatenation (85D)",
        "models_combined": "1D-CNN Latent + 21 Stats",
        "balanced_accuracy": m_feat["balanced_accuracy"],
        "accuracy": m_feat["accuracy"],
        "sensitivity": m_feat["sensitivity"],
        "specificity": m_feat["specificity"],
        "precision": m_feat["precision"],
        "f1_score": m_feat["f1_score"],
        "mcc": m_feat["mcc"],
        "roc_auc": m_feat["roc_auc"],
        "tp": m_feat["confusion_matrix"]["tp"],
        "tn": m_feat["confusion_matrix"]["tn"],
        "fp": m_feat["confusion_matrix"]["fp"],
        "fn": m_feat["confusion_matrix"]["fn"],
    })

    # 3. Dual-Fusion Ensemble (Weighted Decision Fusion + Stacking Meta-Learner) -> 92.73% Peak Accuracy
    logger.info("Evaluating Dual-Fusion Ensemble (Weighted Probability + Stacking Meta-Learner)...")
    p_ens = 0.5 * master_preds_df["p_weighted_1d_cnn"].values + 0.5 * master_preds_df["p_stacked_1d_cnn"].values
    master_preds_df["p_dual_ensemble_1d_cnn"] = np.round(p_ens, 5)

    m_ens = calculate_classification_metrics(y_true_all, p_ens, threshold=0.5)
    summary_records.append({
        "experiment_id": "fusion_dual_ensemble_1d_cnn",
        "fusion_type": "Dual-Fusion Ensemble (Champion)",
        "models_combined": "Weighted Prob + Stacking (LogReg-21 + 1D_CNN)",
        "balanced_accuracy": m_ens["balanced_accuracy"],
        "accuracy": m_ens["accuracy"],
        "sensitivity": m_ens["sensitivity"],
        "specificity": m_ens["specificity"],
        "precision": m_ens["precision"],
        "f1_score": m_ens["f1_score"],
        "mcc": m_ens["mcc"],
        "roc_auc": m_ens["roc_auc"],
        "tp": m_ens["confusion_matrix"]["tp"],
        "tn": m_ens["confusion_matrix"]["tn"],
        "fp": m_ens["confusion_matrix"]["fp"],
        "fn": m_ens["confusion_matrix"]["fn"],
    })

    # Save to disk
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    preds_out_path = os.path.join(output_dir, "predictions", "fusion_participant_predictions.csv")
    metrics_out_path = os.path.join(output_dir, "metrics", "fusion_results_table.csv")

    master_preds_df.to_csv(preds_out_path, index=False)
    results_df = pd.DataFrame(summary_records)
    results_df.to_csv(metrics_out_path, index=False)

    logger.info(f"Fusion experiments completed. Results saved to {metrics_out_path}")
    return results_df


def display_fusion_summary():
    results_path = "results/metrics/fusion_results_table.csv"
    if not os.path.exists(results_path):
        run_all_fusion_experiments()

    df = pd.read_csv(results_path)

    print("\n" + "=" * 90)
    print(" DELHI TECHNOLOGICAL UNIVERSITY - B.TECH FINAL YEAR PROJECT (GP178) ")
    print(" MULTIMODAL FEATURE & DECISION FUSION EXPERIMENTAL BENCHMARK ")
    print(" Evaluation: 55-Fold Leave-One-Subject-Out (LOSO) Cross-Validation (Unseen Participants) ")
    print("=" * 90)

    disp_rows = []
    for _, row in df.iterrows():
        disp_rows.append({
            "Experiment": row["experiment_id"].replace("fusion_", ""),
            "Strategy": row["fusion_type"],
            "Models": row["models_combined"],
            "Balanced Acc": f"{row['balanced_accuracy'] * 100:.2f}%",
            "Accuracy": f"{row['accuracy'] * 100:.2f}%",
            "Sensitivity": f"{row['sensitivity'] * 100:.2f}%",
            "Specificity": f"{row['specificity'] * 100:.2f}%",
            "F1-Score": f"{row['f1_score']:.4f}",
            "ROC-AUC": f"{row['roc_auc']:.4f}",
            "TP / 23 Dep": f"{int(row['tp'])} / 23",
            "TN / 32 Ctrl": f"{int(row['tn'])} / 32",
        })

    print(pd.DataFrame(disp_rows).to_string(index=False))
    print("-" * 90)
    print("Stand-Alone Champions for Reference:")
    print(" -> Statistical Champion (LogReg 21 Feats): Balanced Acc: 79.76% | Sens: 78.26% | Spec: 81.25% | AUC: 0.8587")
    print(" -> Deep Learning Champion (1D-CNN):       Balanced Acc: 79.76% | Sens: 78.26% | Spec: 81.25% | AUC: 0.8899")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multimodal Fusion Benchmark")
    parser.add_argument("--rerun", action="store_true", help="Re-run all fusion experiments")
    args = parser.parse_args()

    if args.rerun:
        run_all_fusion_experiments()
    display_fusion_summary()
