"""
Publication-Quality Figure Generator for Research Paper.

Generates 300 DPI vector-compatible plots for the B.Tech CSE research paper:
1. Figure 1: Actigraphy Digital Biomarkers Comparison (Condition vs Control)
2. Figure 2: Statistical Classifiers Benchmark & Ablation (3 vs 21 features)
3. Figure 3: ROC Curves for Participant-Level Generalisation (55 LOSO Folds)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_figure1_digital_biomarkers():
    """Figure 1: Boxplots of Key Clinical Digital Biomarkers."""
    df = pd.read_csv("data/processed/features_statistical.csv")
    
    biomarkers = [
        ("mean_activity", "Mean Activity Count", "Counts / min"),
        ("day_zero_ratio", "Daytime Inactivity Ratio", "Proportion [0, 1]"),
        ("m10", "Peak 10-Hour Activity (M10)", "Hourly Mean Count"),
        ("relative_amplitude", "Circadian Relative Amplitude (RA)", "Index [0, 1]"),
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    colors = {"condition": "#d95f02", "control": "#1b9e77"} # Colorblind-safe palette
    
    for ax, (col, title, ylabel) in zip(axes, biomarkers):
        cond_data = df[df["group"] == "condition"][col]
        ctrl_data = df[df["group"] == "control"][col]
        
        bp = ax.boxplot(
            [cond_data, ctrl_data],
            tick_labels=["Depressed\n(N=23)", "Control\n(N=32)"],
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        
        # Color boxes
        bp["boxes"][0].set_facecolor(colors["condition"])
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(colors["control"])
        bp["boxes"][1].set_alpha(0.7)
        
        ax.set_title(title, weight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "figure1_digital_biomarkers.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 1: {out_path}")


def plot_figure2_statistical_comparison():
    """Figure 2: Model Performance Bar Chart (3 vs 21 features)."""
    df = pd.read_csv("results/metrics/statistical_results_table.csv")
    
    models = ["Logistic Reg.", "Linear SVM", "RBF SVM", "Random Forest"]
    model_keys = ["logistic_regression", "linear_svm", "rbf_svm", "random_forest"]
    
    b3_acc = [df[(df["model"] == m) & (df["feature_set"] == "baseline_3")]["balanced_accuracy"].iloc[0] * 100 for m in model_keys]
    a21_acc = [df[(df["model"] == m) & (df["feature_set"] == "all_21")]["balanced_accuracy"].iloc[0] * 100 for m in model_keys]
    
    b3_sens = [df[(df["model"] == m) & (df["feature_set"] == "baseline_3")]["sensitivity"].iloc[0] * 100 for m in model_keys]
    a21_sens = [df[(df["model"] == m) & (df["feature_set"] == "all_21")]["sensitivity"].iloc[0] * 100 for m in model_keys]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(models))
    width = 0.35
    
    # Balanced Accuracy Plot
    rects1 = ax1.bar(x - width/2, b3_acc, width, label="Baseline (3 features)", color="#7570b3", alpha=0.85, edgecolor="black")
    rects2 = ax1.bar(x + width/2, a21_acc, width, label="Our Biomarkers (21 features)", color="#d95f02", alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Balanced Accuracy (%)", weight="bold")
    ax1.set_title("Participant-Level Balanced Accuracy (55 LOSO Folds)", weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, weight="bold")
    ax1.set_ylim([45, 90])
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add values on top of bars
    for r in rects1:
        ax1.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width()/2, r.get_height() + 1), ha="center", fontsize=9)
    for r in rects2:
        ax1.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width()/2, r.get_height() + 1), ha="center", fontsize=9, weight="bold")
        
    # Depression Sensitivity Plot
    rects3 = ax2.bar(x - width/2, b3_sens, width, label="Baseline (3 features)", color="#7570b3", alpha=0.85, edgecolor="black")
    rects4 = ax2.bar(x + width/2, a21_sens, width, label="Our Biomarkers (21 features)", color="#1b9e77", alpha=0.85, edgecolor="black")
    ax2.set_ylabel("Depression Sensitivity / Recall (%)", weight="bold")
    ax2.set_title("Depression Detection Sensitivity (Unseen Patients)", weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, weight="bold")
    ax2.set_ylim([40, 90])
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    
    for r in rects3:
        ax2.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width()/2, r.get_height() + 1), ha="center", fontsize=9)
    for r in rects4:
        ax2.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width()/2, r.get_height() + 1), ha="center", fontsize=9, weight="bold")
        
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "figure2_statistical_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 2: {out_path}")


def plot_figure3_roc_curves():
    """Figure 3: Participant-Level ROC Curves across Models."""
    df_part = pd.read_csv("results/predictions/statistical_participant_predictions.csv")
    y_true = df_part["label"].values
    
    models = [
        ("logistic_regression_all_21_prob", "Logistic Reg. (21 Feats)", "#d95f02", 2.2),
        ("linear_svm_all_21_prob", "Linear SVM (21 Feats)", "#1b9e77", 1.8),
        ("random_forest_all_21_prob", "Random Forest (21 Feats)", "#7570b3", 1.8),
        ("logistic_regression_baseline_3_prob", "Logistic Reg. (3 Feats Baseline)", "#e7298a", 1.4),
        ("linear_svm_baseline_3_prob", "Linear SVM (3 Feats Baseline)", "#66a61e", 1.4),
    ]
    
    plt.figure(figsize=(7.5, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.5000)")
    
    for prob_col, label, color, lw in models:
        y_prob = df_part[prob_col].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=lw, label=f"{label} (AUC = {roc_auc:.4f})")
        
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", weight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", weight="bold")
    plt.title("Participant-Level ROC Curves (55-Fold LOSO Generalisation)", weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "figure3_roc_curves_statistical.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 3: {out_path}")


def plot_figure4_deep_vs_statistical_comparison():
    """Figure 4: Deep Learning vs Statistical Champion Comparison."""
    df_deep = pd.read_csv("results/metrics/deep_results_table.csv")
    df_stat = pd.read_csv("results/metrics/statistical_results_table.csv")

    stat_row = df_stat[df_stat["experiment_id"] == "logistic_regression_all_21"].iloc[0]

    models = ["Logistic Reg.\n(21 Biomarkers)", "LSTM", "GRU", "1D-CNN"]
    balanced_accs = [
        stat_row["balanced_accuracy"] * 100,
        df_deep[df_deep["model"] == "lstm"]["balanced_accuracy"].iloc[0] * 100,
        df_deep[df_deep["model"] == "gru"]["balanced_accuracy"].iloc[0] * 100,
        df_deep[df_deep["model"] == "cnn1d"]["balanced_accuracy"].iloc[0] * 100,
    ]
    aucs = [
        stat_row["roc_auc"] * 100,
        df_deep[df_deep["model"] == "lstm"]["roc_auc"].iloc[0] * 100,
        df_deep[df_deep["model"] == "gru"]["roc_auc"].iloc[0] * 100,
        df_deep[df_deep["model"] == "cnn1d"]["roc_auc"].iloc[0] * 100,
    ]
    sensitivities = [
        stat_row["sensitivity"] * 100,
        df_deep[df_deep["model"] == "lstm"]["sensitivity"].iloc[0] * 100,
        df_deep[df_deep["model"] == "gru"]["sensitivity"].iloc[0] * 100,
        df_deep[df_deep["model"] == "cnn1d"]["sensitivity"].iloc[0] * 100,
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.25

    rects1 = ax.bar(x - width, balanced_accs, width, label="Balanced Accuracy (%)", color="#1b9e77", alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x, sensitivities, width, label="Sensitivity / Recall (%)", color="#d95f02", alpha=0.85, edgecolor="black")
    rects3 = ax.bar(x + width, aucs, width, label="ROC-AUC (%)", color="#7570b3", alpha=0.85, edgecolor="black")

    ax.set_ylabel("Metric Value (%)", weight="bold")
    ax.set_title("Cross-Branch Comparison: Statistical Champion vs. Deep Learning (55 LOSO Folds)", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, weight="bold")
    ax.set_ylim([45, 95])
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rects in [rects1, rects2, rects3]:
        for r in rects:
            ax.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width()/2, r.get_height() + 0.8), ha="center", fontsize=8.5, weight="bold")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "figure4_deep_vs_statistical.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 4: {out_path}")


def plot_figure5_combined_roc_curves():
    """Figure 5: ROC Curves comparing 1D-CNN, GRU, LSTM, and Logistic Regression."""
    df_part_deep = pd.read_csv("results/predictions/deep_participant_predictions.csv")
    df_part_stat = pd.read_csv("results/predictions/statistical_participant_predictions.csv")
    y_true = df_part_deep["label"].values

    models = [
        (df_part_deep["cnn1d_prob"].values, "1D-CNN", "#1b9e77", 2.2),
        (df_part_deep["gru_prob"].values, "GRU", "#7570b3", 1.8),
        (df_part_deep["lstm_prob"].values, "LSTM", "#e7298a", 1.8),
        (df_part_stat["logistic_regression_all_21_prob"].values, "Logistic Reg. (21 Feats)", "#d95f02", 2.0),
    ]

    plt.figure(figsize=(7.5, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.5000)")

    for y_prob, label, color, lw in models:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=lw, label=f"{label} (AUC = {roc_auc:.4f})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", weight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", weight="bold")
    plt.title("Cross-Architecture ROC Curves (55-Fold LOSO Generalisation)", weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "figure5_cross_architecture_roc.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 5: {out_path}")


def plot_figure6_fusion_performance_boost():
    """Figure 6: Fusion Performance Boost Bar Chart."""
    df_fusion = pd.read_csv("results/metrics/fusion_results_table.csv")
    df_stat = pd.read_csv("results/metrics/statistical_results_table.csv")
    df_deep = pd.read_csv("results/metrics/deep_results_table.csv")

    stat_acc = df_stat[df_stat["experiment_id"] == "logistic_regression_all_21"]["balanced_accuracy"].iloc[0] * 100
    deep_acc = df_deep[df_deep["model"] == "cnn1d"]["balanced_accuracy"].iloc[0] * 100

    f_cnn_wt = df_fusion[df_fusion["experiment_id"] == "fusion_weighted_prob_1d_cnn"]["balanced_accuracy"].iloc[0] * 100
    f_cnn_avg = df_fusion[df_fusion["experiment_id"] == "fusion_prob_averaging_1d_cnn"]["balanced_accuracy"].iloc[0] * 100
    f_gru_wt = df_fusion[df_fusion["experiment_id"] == "fusion_weighted_prob_gru"]["balanced_accuracy"].iloc[0] * 100
    f_lstm_avg = df_fusion[df_fusion["experiment_id"] == "fusion_prob_averaging_lstm"]["balanced_accuracy"].iloc[0] * 100
    f_dual_ens = df_fusion[df_fusion["experiment_id"] == "fusion_dual_ensemble_1d_cnn"]["balanced_accuracy"].iloc[0] * 100

    models = [
        "Statistical Only\n(LogReg-21)",
        "Deep Learning Only\n(1D-CNN)",
        "Fused: Stat + LSTM\n(Avg)",
        "Fused: Stat + GRU\n(Weighted)",
        "Fused: Stat + 1D-CNN\n(Weighted)",
        "Dual-Fusion Ensemble\n(Weighted + Stacking - Champion)",
    ]
    accuracies = [stat_acc, deep_acc, f_lstm_avg, f_gru_wt, f_cnn_wt, f_dual_ens]
    colors = ["#7570b3", "#386cb0", "#f0027f", "#fdc086", "#1b9e77", "#d95f02"]

    plt.figure(figsize=(11.5, 5.2))
    bars = plt.bar(models, accuracies, color=colors, width=0.45, edgecolor="black", alpha=0.9)
    plt.ylabel("Participant Balanced Accuracy (%)", weight="bold")
    plt.title("Impact of Multimodal Fusion: Individual Branches vs. Fused Models (55 LOSO Folds)", weight="bold", pad=12)
    plt.ylim([65, 98])
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Highlight champion
    for i, b in enumerate(bars):
        h = b.get_height()
        fontweight = "bold" if i == 5 else "normal"
        plt.annotate(f"{h:.2f}%", (b.get_x() + b.get_width() / 2, h + 0.8), ha="center", fontsize=9.5, weight=fontweight)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "figure6_fusion_performance_boost.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 6: {out_path}")


def plot_figure7_fusion_roc_curves():
    """Figure 7: Master ROC Curves for Fusion vs Individual Champions."""
    stat = pd.read_csv("results/predictions/statistical_participant_predictions.csv")
    deep = pd.read_csv("results/predictions/deep_participant_predictions.csv")
    fuse = pd.read_csv("results/predictions/fusion_participant_predictions.csv")

    y_true = fuse["true_label"].values

    models = [
        (fuse["p_avg_1d_cnn"].values, "Fused: Stat + 1D-CNN (Prob Avg)", "#1b9e77", 2.4),
        (fuse["p_weighted_1d_cnn"].values, "Fused: Stat + 1D-CNN (Weighted)", "#d95f02", 2.2),
        (fuse["p_avg_gru"].values, "Fused: Stat + GRU (Prob Avg)", "#7570b3", 1.8),
        (deep["cnn1d_prob"].values, "Standalone 1D-CNN", "#386cb0", 1.6),
        (stat["logistic_regression_all_21_prob"].values, "Standalone LogReg (21 Feats)", "#e7298a", 1.6),
    ]

    plt.figure(figsize=(7.8, 6.2))
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.5000)")

    for y_prob, label, color, lw in models:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=lw, label=f"{label} (AUC = {roc_auc:.4f})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", weight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", weight="bold")
    plt.title("Master ROC Curves: Multimodal Fusion vs. Individual Branches", weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "figure7_fusion_roc_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated Figure 7: {out_path}")


if __name__ == "__main__":
    plot_figure1_digital_biomarkers()
    plot_figure2_statistical_comparison()
    plot_figure3_roc_curves()
    plot_figure4_deep_vs_statistical_comparison()
    plot_figure5_combined_roc_curves()
    plot_figure6_fusion_performance_boost()
    plot_figure7_fusion_roc_curves()
