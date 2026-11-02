"""
Feature-Level Multimodal Fusion Module.

Concatenates high-level learned deep representations with domain-engineered statistical biomarkers:
- Deep Latent Vector (dim = 64) extracted from the trained 1D-CNN before the final classification head
- Statistical Feature Vector (dim = 21) containing clinical circadian and volatility biomarkers
- Total Fused Representation Dimension: 64 + 21 = 85 features per window

Strict Leakage Prevention:
In every outer LOSO fold:
- The deep feature extractor is trained strictly on the training subjects.
- StandardScaler is fitted strictly on the 85-dimensional training window vectors.
- A regularized Logistic Regression classification head is trained on the training windows.
- Predictions are evaluated on the held-out test participant and aggregated.
"""

import os
import sys
import copy
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deep_learning.datasets import ActigraphyDataset, compute_training_stats
from deep_learning.cnn1d import ActigraphyCNN1D
from deep_learning.train_deep import get_device, set_seed
from evaluation.metrics import calculate_classification_metrics, aggregate_participant_predictions

logger = logging.getLogger(__name__)


def run_feature_level_fusion(
    sequences_path: str = "data/processed/windows_hourly.npy",
    features_path: str = "data/processed/features_statistical.csv",
    splits_dict: Dict = None,
    output_dir: str = "results",
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run 55-fold LOSO Feature-Level Fusion (Concatenating 64-dim 1D-CNN latent + 21 statistical features).
    """
    device = get_device()
    sequences = np.load(sequences_path) # (1331, 24, 1)
    stat_df = pd.read_csv(features_path)

    # 21 statistical feature columns
    stat_cols = [c for c in stat_df.columns if c not in ["window_id", "participant_id", "group", "label"]]
    X_stat_all = stat_df[stat_cols].values.astype(np.float32) # (1331, 21)

    labels_all = stat_df["label"].values.astype(int)
    pids_all = stat_df["participant_id"].values
    win_ids_all = stat_df["window_id"].values
    n_windows = len(stat_df)

    outer_folds = splits_dict["outer_folds"]
    oof_window_probs = np.zeros(n_windows, dtype=np.float32)

    for fold in outer_folds:
        f_id = fold["fold_id"]
        test_pid = fold["test_participant"]
        test_idx = fold["test_window_indices"]
        train_idx = fold["train_window_indices"]

        set_seed(seed + f_id)

        # 1. Train 1D-CNN feature extractor on inner training split
        inner_split = fold["inner_splits"][0]
        inner_tr_idx = inner_split["train_window_indices"]
        inner_val_idx = inner_split["val_window_indices"]

        mean_val, std_val = compute_training_stats(sequences[inner_tr_idx])

        tr_dataset = ActigraphyDataset(sequences[inner_tr_idx], labels_all[inner_tr_idx], pids_all[inner_tr_idx], inner_tr_idx, mean=mean_val, std=std_val)
        val_dataset = ActigraphyDataset(sequences[inner_val_idx], labels_all[inner_val_idx], pids_all[inner_val_idx], inner_val_idx, mean=mean_val, std=std_val)

        tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

        cnn_model = ActigraphyCNN1D(in_channels=1, num_filters=(32, 64), kernel_size=3, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)

        pos_count = float(np.sum(labels_all[inner_tr_idx] == 1))
        neg_count = float(np.sum(labels_all[inner_tr_idx] == 0))
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))

        best_loss = float("inf")
        best_state = copy.deepcopy(cnn_model.state_dict())
        no_imp = 0

        for epoch in range(35):
            cnn_model.train()
            for bx, by, _, _ in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(cnn_model(bx), by)
                loss.backward()
                optimizer.step()

            cnn_model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vy, _, _ in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    val_losses.append(criterion(cnn_model(vx), vy).item())

            avg_vloss = float(np.mean(val_losses))
            if avg_vloss < best_loss:
                best_loss = avg_vloss
                best_state = copy.deepcopy(cnn_model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= 7:
                    break

        cnn_model.load_state_dict(best_state)
        cnn_model.eval()

        # 2. Extract 64-dim latent representations
        def extract_seq_features(idxs):
            ds = ActigraphyDataset(sequences[idxs], labels_all[idxs], pids_all[idxs], idxs, mean=mean_val, std=std_val)
            loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
            feats = []
            with torch.no_grad():
                for bx, _, _, _ in loader:
                    bx = bx.to(device)
                    lat = cnn_model.extract_features(bx).cpu().numpy()
                    feats.append(lat)
            return np.vstack(feats)

        latent_train = extract_seq_features(train_idx) # (len(train_idx), 64)
        latent_test = extract_seq_features(test_idx)   # (len(test_idx), 64)

        # 3. Concatenate with 21 statistical features
        X_train_fused = np.hstack([latent_train, X_stat_all[train_idx]]) # shape: (len(train_idx), 85)
        X_test_fused = np.hstack([latent_test, X_stat_all[test_idx]])    # shape: (len(test_idx), 85)

        # 4. Scale 85 features strictly on training windows
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train_fused)
        X_te_scaled = scaler.transform(X_test_fused)

        # 5. Fit regularized fusion classification head
        fusion_clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=seed)
        fusion_clf.fit(X_tr_scaled, labels_all[train_idx])

        # 6. Predict on held-out subject windows
        test_probs = fusion_clf.predict_proba(X_te_scaled)[:, 1]
        oof_window_probs[test_idx] = test_probs

    win_df = pd.DataFrame({
        "window_id": win_ids_all,
        "participant_id": pids_all,
        "label": labels_all,
        "pred_prob": np.round(oof_window_probs, 5),
    })

    part_df, metrics = aggregate_participant_predictions(
        window_df=win_df,
        pred_col="pred_prob",
        true_col="label",
        pid_col="participant_id",
        threshold=0.5,
    )

    metrics["model"] = "feature_fusion_cnn1d_and_stats"
    metrics["fusion_strategy"] = "feature_level_concatenation"
    metrics["fused_feature_dimension"] = 85

    return metrics, part_df
