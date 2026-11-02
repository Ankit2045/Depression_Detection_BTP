"""
Master Execution Pipeline for Clinical AI Extensions.

Module: ai_system.run_ai_system
Executes the modern AI suite:
1. Trains Temporal Multi-Head Self-Attention Transformer & extracts 24-hour attention profiles.
2. Generates publication-ready 300 DPI Figure 10 (Circadian Attention Heatmap).
3. Executes Generative AI Clinical Co-Pilot to synthesize DSM-5 psychiatric memos for all 55 patients.
4. Precompiles self-contained interactive web dataset (ai_system/dashboard/data.json).
5. Provides command to launch local clinician dashboard.

Usage:
    python ai_system/run_ai_system.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai_system.transformer_attention import (
    train_transformer_and_extract_attention,
    plot_figure10_circadian_attention,
)
from ai_system.clinical_agent import ClinicalDiagnosticCopilot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compile_dashboard_dataset(
    attention_df: pd.DataFrame,
    copilot: ClinicalDiagnosticCopilot,
    output_path: str = "ai_system/dashboard/data.json",
):
    """
    Precompile all patient waveforms, probabilities, z-scores, attention weights,
    and generated clinical notes into a single optimized JSON for the clinician dashboard.
    """
    logger.info("Compiling patient data for interactive clinician dashboard...")
    meta_df = pd.read_csv("data/processed/window_metadata.csv")
    scores_df = pd.read_csv("data/scores.csv")
    features_df = pd.read_csv("data/processed/features_statistical.csv")
    preds_df = pd.read_csv("results/predictions/fusion_participant_predictions.csv")
    sequences = np.load("data/processed/windows_hourly.npy")  # (1331, 24, 1)

    # 1. Compute control population average 24h waveform
    ctrl_pids = meta_df[meta_df["label"] == 0]["participant_id"].unique()
    ctrl_indices = meta_df[meta_df["participant_id"].isin(ctrl_pids)].index
    ctrl_waveform_avg = sequences[ctrl_indices].mean(axis=(0, 2)).tolist()
    ctrl_waveform_avg = [round(float(v), 2) for v in ctrl_waveform_avg]

    # Precompute healthy control norms
    ctrl_feat_mask = (features_df["label"] == 0)
    c_means = features_df[ctrl_feat_mask].mean(numeric_only=True)
    c_stds = features_df[ctrl_feat_mask].std(numeric_only=True) + 1e-8

    dashboard_records = []
    pids = preds_df["participant_id"].unique()

    for pid in pids:
        p_row = preds_df[preds_df["participant_id"] == pid].iloc[0]
        prob = float(p_row.get("p_dual_ensemble_1d_cnn", p_row.get("p_weighted_1d_cnn", 0.5)))
        label = int(p_row["true_label"])
        group = "Condition (Depression)" if label == 1 else "Control (Healthy)"

        # Demographics
        s_row = scores_df[scores_df["number"] == pid]
        if not s_row.empty:
            s_row = s_row.iloc[0]
            age = str(s_row.get("age", "50-54"))
            gender = "Female" if s_row.get("gender") == 2 else "Male" if s_row.get("gender") == 1 else "Unknown"
            madrs1 = float(s_row["madrs1"]) if pd.notna(s_row.get("madrs1")) else None
            madrs2 = float(s_row["madrs2"]) if pd.notna(s_row.get("madrs2")) else None
        else:
            age = "50-54"
            gender = "Control Subject"
            madrs1, madrs2 = None, None

        # 24h Patient Average Waveform
        p_win_idx = meta_df[meta_df["participant_id"] == pid].index
        p_waveform = sequences[p_win_idx].mean(axis=(0, 2)).tolist()
        p_waveform = [round(float(v), 2) for v in p_waveform]

        # Attention weights (24 hours)
        attn_row = attention_df[attention_df["participant_id"] == pid]
        if not attn_row.empty:
            attn_weights = [round(float(attn_row[f"hour_{h:02d}"].iloc[0]), 5) for h in range(24)]
        else:
            attn_weights = [round(1.0 / 24.0, 5)] * 24

        # Key Digital Biomarkers & Z-scores
        p_feats = features_df[features_df["participant_id"] == pid].mean(numeric_only=True)
        sed = float(p_feats.get("day_zero_ratio", p_feats.get("daytime_sedentary_ratio", 0.13)))
        sed_norm = float(c_means.get("day_zero_ratio", 0.13))
        sed_std = float(c_stds.get("day_zero_ratio", 0.08))
        sed_z = (sed - sed_norm) / (sed_std + 1e-8)

        m10 = float(p_feats.get("m10", p_feats.get("m10_vigor", 465.0)))
        m10_norm = float(c_means.get("m10", 465.0))
        m10_std = float(c_stds.get("m10", 130.0))
        m10_z = (m10 - m10_norm) / (m10_std + 1e-8)

        med_act = float(p_feats.get("median_activity", 114.0))
        med_norm = float(c_means.get("median_activity", 114.0))
        med_std = float(c_stds.get("median_activity", 40.0))
        med_z = (med_act - med_norm) / (med_std + 1e-8)

        noct = float(p_feats.get("night_mean", p_feats.get("nocturnal_mean", 40.0)))
        noct_norm = float(c_means.get("night_mean", 40.0))
        noct_std = float(c_stds.get("night_mean", 20.0))
        noct_z = (noct - noct_norm) / (noct_std + 1e-8)

        # Clinical Memo
        memo = copilot.generate_patient_memo(pid)

        dashboard_records.append({
            "participant_id": pid,
            "label": label,
            "group": group,
            "prob_ensemble": round(prob, 4),
            "age": age,
            "gender": gender,
            "madrs1": madrs1,
            "madrs2": madrs2,
            "waveform_24h": p_waveform,
            "control_benchmark_waveform": ctrl_waveform_avg,
            "attention_weights": attn_weights,
            "biomarkers": {
                "daytime_sedentary_ratio": round(sed, 4),
                "daytime_sedentary_ratio_z": round(sed_z, 2),
                "m10_vigor": round(m10, 2),
                "m10_vigor_z": round(m10_z, 2),
                "median_activity": round(med_act, 2),
                "median_activity_z": round(med_z, 2),
                "nocturnal_mean": round(noct, 2),
                "nocturnal_mean_z": round(noct_z, 2),
            },
            "clinical_memo": memo,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dashboard_records, f, indent=2)

    logger.info(f"Dashboard dataset successfully written to {output_path} ({len(dashboard_records)} participants).")


def main():
    print("=" * 85)
    print(" CLINICAL ARTIFICIAL INTELLIGENCE & TEMPORAL TRANSFORMER MODULE ")
    print(" Group GP178 | Delhi Technological University ")
    print(" Architecture: Multi-Head Self-Attention Transformer + Generative AI Clinical Co-Pilot ")
    print("=" * 85)

    # 1. Train Temporal Transformer and Extract 24-Hour Attention Weights
    logger.info("Executing Multi-Head Temporal Self-Attention Transformer...")
    model, attn_dep, attn_ctrl, attention_df = train_transformer_and_extract_attention(
        epochs=40,
        lr=0.001,
        seed=42,
    )
    attention_csv = "ai_system/results/attention_weights.csv"
    attention_df.to_csv(attention_csv, index=False)
    logger.info(f"Saved 24-hour attention weights to {attention_csv}")

    # 2. Generate Publication Figure 10
    fig_path = "ai_system/results/figures/figure10_circadian_attention_heatmap.png"
    plot_figure10_circadian_attention(attn_dep, attn_ctrl, output_path=fig_path)

    # 3. Generate Clinical AI Co-Pilot Memos for All Patients
    logger.info("Generating Generative AI Psychiatric Consultation Notes (55 Patients)...")
    copilot = ClinicalDiagnosticCopilot()
    memo_paths = copilot.generate_all_patient_memos()
    logger.info(f"Generated {len(memo_paths)} patient consultation notes in ai_system/results/clinical_notes/")

    # 4. Precompile Clinician Dashboard JSON
    compile_dashboard_dataset(attention_df=attention_df, copilot=copilot)

    print("\n" + "=" * 85)
    print(" CLINICAL AI SUITE GENERATION COMPLETE ")
    print("=" * 85)
    print(f"1. Temporal Transformer Attention Map:  {fig_path}")
    print(f"2. Hour-by-Hour Attention Weights:      {attention_csv}")
    print(f"3. Generative AI Patient Notes:         ai_system/results/clinical_notes/ (55 Memos)")
    print(f"4. Interactive Doctor Web Dashboard:    ai_system/dashboard/index.html")
    print("\nTo view the interactive clinical dashboard live:")
    print("  Run:  python -m http.server 8000 --directory ai_system/dashboard")
    print("  Open: http://localhost:8000 in your web browser.")
    print("=" * 85)


if __name__ == "__main__":
    main()
