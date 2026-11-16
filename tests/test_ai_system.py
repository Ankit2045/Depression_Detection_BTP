"""Tests for the standalone Clinical Artificial Intelligence and Transformer module (ai_system/)."""

import os
import json
import torch
import pytest
import numpy as np

from ai_system.transformer_attention import ActigraphyTemporalTransformer
from ai_system.clinical_agent import ClinicalDiagnosticCopilot


def test_transformer_architecture_and_forward():
    """Verify Temporal Transformer architecture shapes, attention extraction, and parameter budget."""
    batch_size = 4
    seq_len = 24  # 24 circadian hourly bins
    input_dim = 1

    model = ActigraphyTemporalTransformer(
        input_dim=input_dim,
        embed_dim=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.1,
    )

    model.eval()
    x = torch.randn(batch_size, seq_len, input_dim)
    with torch.no_grad():
        logits, attn_weights = model(x, return_attention=True)

    assert logits.shape == (batch_size,), f"Expected logits shape {(batch_size,)}, got {logits.shape}"
    assert attn_weights.shape == (batch_size, seq_len, seq_len), (
        f"Expected attention shape {(batch_size, seq_len, seq_len)}, got {attn_weights.shape}"
    )

    # Attention weights per row should sum to 1.0 (softmax)
    row_sums = attn_weights.sum(dim=-1).detach().cpu().numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-4), "Attention weights do not sum to 1 across sequence length."

    # Check parameter count is compact (< 25,000 parameters to prevent overfitting on N=55)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params < 25000, f"Transformer too large for clinical sample: {total_params} params"


def test_clinical_copilot_memo_generation():
    """Verify Generative AI Clinical Co-Pilot generates structured DSM-5 memos with correct clinical fields."""
    copilot = ClinicalDiagnosticCopilot()
    memo = copilot.generate_patient_memo("condition_1")

    assert isinstance(memo, str)
    assert len(memo) > 200
    assert "CLINICAL AI PSYCHIATRIC CONSULTATION MEMO" in memo
    assert "condition_1" in memo
    assert "Daytime Sedentary Ratio" in memo
    assert "M10 Daytime Peak Vigor" in memo
    assert "TARGETED CLINICAL" in memo


def test_ai_system_deliverables_exist():
    """Verify that all standalone AI system artifacts were generated and are valid."""
    dashboard_json = "ai_system/dashboard/data.json"
    heatmap_png = "ai_system/results/figures/figure10_circadian_attention_heatmap.png"
    attn_csv = "ai_system/results/attention_weights.csv"
    clinical_dir = "ai_system/results/clinical_notes"

    assert os.path.exists(dashboard_json), f"Missing {dashboard_json}"
    assert os.path.exists(heatmap_png), f"Missing {heatmap_png}"
    assert os.path.exists(attn_csv), f"Missing {attn_csv}"
    assert os.path.isdir(clinical_dir), f"Missing {clinical_dir}"

    with open(dashboard_json, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 55, f"Expected at least 55 participants in dashboard json, got {len(data)}"
    assert "participant_id" in data[0]
    assert "prob_ensemble" in data[0]

    notes = os.listdir(clinical_dir)
    assert len(notes) >= 55, f"Expected at least 55 clinical notes, found {len(notes)}"


def test_new_patient_inference_pipeline():
    """Verify end-to-end inference on raw external CSV actigraphy data."""
    from ai_system.predict_new_patient import NewPatientPredictor

    predictor = NewPatientPredictor()
    res = predictor.predict_patient(
        file_path="data/condition/condition_1.csv",
        patient_id="test_unit_patient",
        age="30-34",
        gender="Female",
        update_dashboard=False,
    )

    assert 0.0 <= res["prob_ensemble"] <= 1.0
    assert 0.0 <= res["prob_statistical"] <= 1.0
    assert 0.0 <= res["prob_deep_learning"] <= 1.0
    assert "biomarkers" in res
    assert "daytime_sedentary_ratio" in res["biomarkers"]
    assert "m10_vigor" in res["biomarkers"]
    assert "clinical_memo" in res
    assert "CLINICAL AI PSYCHIATRIC CONSULTATION MEMO" in res["clinical_memo"]
    assert os.path.exists("ai_system/results/clinical_notes/test_unit_patient_clinical_memo.md")
