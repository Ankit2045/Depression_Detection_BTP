"""
Production Model Exporter for Clinical Actigraphy Depression Detection (DTU-GP178).

Fits the champion models on the complete 55-subject reference cohort and saves:
1. StandardScaler for the 21 statistical biomarkers.
2. Production Logistic Regression model.
3. Production 1D-CNN model weights.
4. Healthy control baseline reference norms (means, stds, 24h curve) for Z-score generation.
5. Model configuration metadata.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from deep_learning.cnn1d import ActigraphyCNN1D

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_production_models(output_dir: str = "ai_system/models"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Preprocessed Data
    features_csv = "data/processed/features_statistical.csv"
    sequences_npy = "data/processed/windows_hourly.npy"
    window_meta_csv = "data/processed/window_metadata.csv"

    logger.info("Loading reference benchmark dataset (55 participants)...")
    feats_df = pd.read_csv(features_csv)
    meta_df = pd.read_csv(window_meta_csv)
    sequences = np.load(sequences_npy)

    # 21 feature column names
    meta_cols = ["window_id", "participant_id", "group", "label"]
    feature_names = [c for c in feats_df.columns if c not in meta_cols]
    assert len(feature_names) == 21, f"Expected 21 features, found {len(feature_names)}"

    X_stat = feats_df[feature_names].values
    y_win = meta_df["label"].values.astype(int)

    # 2. Fit and Export StandardScaler
    logger.info("Fitting and exporting production StandardScaler...")
    scaler = StandardScaler()
    X_stat_scaled = scaler.fit_transform(X_stat)
    joblib.dump(scaler, os.path.join(output_dir, "production_scaler.joblib"))

    # 3. Fit and Export Champion Logistic Regression
    logger.info("Fitting and exporting production Logistic Regression model...")
    lr_model = LogisticRegression(
        C=0.01,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    lr_model.fit(X_stat_scaled, y_win)
    joblib.dump(lr_model, os.path.join(output_dir, "production_logistic_regression.joblib"))

    # 4. Train and Export Champion 1D-CNN
    logger.info("Training and exporting production 1D-CNN...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cnn_model = ActigraphyCNN1D(in_channels=1, num_filters=(32, 64), kernel_size=3, dropout=0.2).to(device)

    # Calculate class weights for loss
    n_pos = np.sum(y_win == 1)
    n_neg = np.sum(y_win == 0)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)

    X_seq_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_win, dtype=torch.float32).to(device)

    cnn_model.train()
    dataset = torch.utils.data.TensorDataset(X_seq_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(40):
        for bx, by in loader:
            optimizer.zero_grad()
            out = cnn_model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    cnn_model.eval()
    torch.save(cnn_model.state_dict(), os.path.join(output_dir, "production_cnn1d.pt"))

    # 5. Compute Healthy Control Baseline Reference Norms
    logger.info("Computing healthy control reference norms...")
    ctrl_mask = (feats_df["label"] == 0)
    ctrl_feats = feats_df[ctrl_mask]
    
    ctrl_means = {k: float(v) for k, v in ctrl_feats[feature_names].mean().items()}
    ctrl_stds = {k: float(v) for k, v in ctrl_feats[feature_names].std().items()}

    # Average 24h waveform of healthy controls
    ctrl_win_idx = meta_df[meta_df["label"] == 0].index
    ctrl_24h_waveform = sequences[ctrl_win_idx].mean(axis=(0, 2)).tolist()
    ctrl_24h_waveform = [round(float(v), 2) for v in ctrl_24h_waveform]

    reference_norms = {
        "feature_names": feature_names,
        "control_means": ctrl_means,
        "control_stds": ctrl_stds,
        "control_benchmark_waveform": ctrl_24h_waveform,
        "weights": {
            "p_stat_weight": 0.50,
            "p_deep_weight": 0.50,
        },
        "scaler_means": scaler.mean_.tolist(),
        "scaler_scales": scaler.scale_.tolist(),
        "lr_coefs": lr_model.coef_[0].tolist(),
        "lr_intercept": float(lr_model.intercept_[0]),
    }

    with open(os.path.join(output_dir, "reference_norms.json"), "w") as f:
        json.dump(reference_norms, f, indent=2)

    dashboard_params_file = "ai_system/dashboard/model_params.json"
    with open(dashboard_params_file, "w") as f:
        json.dump(reference_norms, f, indent=2)

    logger.info(f"Production model bundle successfully saved to {output_dir}/ and {dashboard_params_file}")


if __name__ == "__main__":
    export_production_models()
