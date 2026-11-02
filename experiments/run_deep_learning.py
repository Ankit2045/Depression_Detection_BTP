"""
Master Runner for Deep Learning Branch Experiments.

Usage:
    python experiments/run_deep_learning.py
    python experiments/run_deep_learning.py --rerun
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deep_learning.train_deep import run_all_deep_experiments
from deep_learning.lstm import ActigraphyLSTM, count_trainable_parameters
from deep_learning.gru import ActigraphyGRU
from deep_learning.cnn1d import ActigraphyCNN1D


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)


def display_deep_learning_summary():
    deep_path = "results/metrics/deep_results_table.csv"
    stat_path = "results/metrics/statistical_results_table.csv"

    if not os.path.exists(deep_path):
        print("Deep learning results not found. Running experiments...")
        run_all_deep_experiments()

    df_deep = pd.read_csv(deep_path)

    # Compute parameter counts
    lstm = ActigraphyLSTM()
    gru = ActigraphyGRU()
    cnn = ActigraphyCNN1D()
    param_map = {
        "lstm": count_trainable_parameters(lstm),
        "gru": count_trainable_parameters(gru),
        "cnn1d": count_trainable_parameters(cnn),
    }

    print_banner("Delhi Technological University - B.Tech Final Year Project (GP178)")
    print("Project: Actigraphy-Based Depression Detection Using Deep Learning")
    print("Evaluation Protocol: 55-Fold Leave-One-Subject-Out (LOSO) Cross-Validation")
    print("Temporal Representation: 24-Hour Windows (Hourly Downsampled, T=24)")
    print("-" * 80)

    rows = []
    for _, row in df_deep.iterrows():
        m = row["model"]
        rows.append({
            "Architecture": row["architecture"],
            "Parameters": f"{param_map.get(m, 0):,}",
            "Balanced Acc": f"{row['balanced_accuracy'] * 100:.2f}%",
            "Accuracy": f"{row['accuracy'] * 100:.2f}%",
            "Sensitivity": f"{row['sensitivity'] * 100:.2f}%",
            "Specificity": f"{row['specificity'] * 100:.2f}%",
            "F1-Score": f"{row['f1_score']:.4f}",
            "ROC-AUC": f"{row['roc_auc']:.4f}",
            "TP / Total Depressed": f"{int(row['tp'])} / 23",
            "TN / Total Control": f"{int(row['tn'])} / 32",
        })

    disp_df = pd.DataFrame(rows)
    print(disp_df.to_string(index=False))
    print("-" * 80)

    # Benchmark comparison against statistical champion
    if os.path.exists(stat_path):
        df_stat = pd.read_csv(stat_path)
        stat_champ = df_stat[df_stat["experiment_id"] == "logistic_regression_all_21"].iloc[0]
        print(f"Comparison Benchmark (Statistical Champion: LogReg 21 Feats):")
        print(f" -> Balanced Acc: {stat_champ['balanced_accuracy']*100:.2f}% | Sensitivity: {stat_champ['sensitivity']*100:.2f}% | Specificity: {stat_champ['specificity']*100:.2f}% | AUC: {stat_champ['roc_auc']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and Display Deep Learning Benchmark")
    parser.add_argument("--rerun", action="store_true", help="Force re-running all 55 folds")
    args = parser.parse_args()

    if args.rerun:
        print_banner("Executing Deep Learning 55-Fold LOSO Training")
        run_all_deep_experiments()

    display_deep_learning_summary()
