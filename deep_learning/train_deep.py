"""
Deep Learning Training & Evaluation Module.

Evaluates LSTM, GRU, and 1D-CNN architectures on actigraphy sequences using
the 55-fold Leave-One-Subject-Out (LOSO) cross-validation protocol:
- Hardware acceleration: Apple Silicon Metal Performance Shaders (MPS) / CUDA / CPU
- Leakage-Safe Inner Early Stopping: Model checkpoints are selected strictly on
  inner validation subjects (holding out ~11 training subjects) to prevent overfitting.
- Training-Only Standardization: Normalization parameters (mean and std) are computed
  exclusively on the training sequences in each fold.
- Participant-Level Evaluation: Test predictions are aggregated per unseen participant.
"""

import os
import sys
import json
import logging
import random
import copy
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deep_learning.datasets import ActigraphyDataset, compute_training_stats
from deep_learning.lstm import ActigraphyLSTM
from deep_learning.gru import ActigraphyGRU
from deep_learning.cnn1d import ActigraphyCNN1D
from deep_learning.cnn1d_minute import ActigraphyCNN1DMinute
from evaluation.metrics import calculate_classification_metrics, aggregate_participant_predictions
from preprocessing.clean_data import natural_sort_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_model(model_type: str, input_dim: int = 1, hidden_dim: int = 64, dropout: float = 0.2) -> nn.Module:
    """Instantiate chosen deep learning architecture."""
    if model_type == "lstm":
        return ActigraphyLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=3, dropout=dropout, pooling="mean")
    elif model_type == "gru":
        return ActigraphyGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
    elif model_type == "cnn1d":
        return ActigraphyCNN1D(in_channels=input_dim, num_filters=(32, 64), kernel_size=3, dropout=dropout)
    elif model_type == "cnn1d_minute":
        return ActigraphyCNN1DMinute(in_channels=input_dim, num_filters=(32, 64, 64), dropout=dropout)
    else:
        raise ValueError(f"Unknown deep learning model type: {model_type}")


def train_and_eval_single_fold(
    model_type: str,
    fold_info: Dict,
    sequences_all: np.ndarray,
    labels_all: np.ndarray,
    pids_all: np.ndarray,
    win_indices_all: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    max_epochs: int = 35,
    patience: int = 7,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train and evaluate a single LOSO fold with leak-free inner early stopping.

    Returns:
        test_indices: Global indices of held-out test windows.
        test_probs: Predicted probabilities for the held-out test windows.
    """
    set_seed(seed + fold_info["fold_id"])

    test_idx = fold_info["test_window_indices"]

    # Use first inner split for early stopping validation (holds out ~11 training subjects)
    inner_split = fold_info["inner_splits"][0]
    inner_val_idx = inner_split["val_window_indices"]
    inner_tr_idx = inner_split["train_window_indices"]

    # 1. Compute standardization exclusively on inner training sequences
    X_inner_tr = sequences_all[inner_tr_idx]
    mean_val, std_val = compute_training_stats(X_inner_tr)

    # 2. Build Datasets
    train_dataset = ActigraphyDataset(
        sequences=sequences_all[inner_tr_idx],
        labels=labels_all[inner_tr_idx],
        pids=pids_all[inner_tr_idx],
        window_indices=win_indices_all[inner_tr_idx],
        mean=mean_val,
        std=std_val,
    )
    val_dataset = ActigraphyDataset(
        sequences=sequences_all[inner_val_idx],
        labels=labels_all[inner_val_idx],
        pids=pids_all[inner_val_idx],
        window_indices=win_indices_all[inner_val_idx],
        mean=mean_val,
        std=std_val,
    )
    test_dataset = ActigraphyDataset(
        sequences=sequences_all[test_idx],
        labels=labels_all[test_idx],
        pids=pids_all[test_idx],
        window_indices=win_indices_all[test_idx],
        mean=mean_val,
        std=std_val,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Class-weighted Loss
    y_tr_vals = labels_all[inner_tr_idx]
    pos_count = float(np.sum(y_tr_vals == 1))
    neg_count = float(np.sum(y_tr_vals == 0))
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # 4. Model and Optimizer
    model = create_model(model_type=model_type, input_dim=1, hidden_dim=64, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    # 5. Training Loop with Inner Early Stopping
    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y, _, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation Step
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_x, val_y, _, _ in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                v_logits = model(val_x)
                v_loss = criterion(v_logits, val_y)
                val_losses.append(v_loss.item())

        avg_val_loss = float(np.mean(val_losses))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # 6. Evaluate on Held-Out Test Participant
    model.load_state_dict(best_weights)
    model.eval()

    test_probs = []
    with torch.no_grad():
        for test_x, _, _, _ in test_loader:
            test_x = test_x.to(device)
            logits = model(test_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs.tolist())

    return np.array(test_idx), np.array(test_probs, dtype=np.float32)


def run_deep_learning_experiment(
    model_type: str,
    sequences_path: str = "data/processed/windows_hourly.npy",
    metadata_path: str = "data/processed/window_metadata.csv",
    splits_path: str = "results/splits/loso_splits.json",
    batch_size: int = 32,
    max_epochs: int = 35,
    patience: int = 7,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run 55-fold LOSO cross-validation for an individual deep learning architecture.
    """
    device = get_device()
    logger.info(f"Running 55-fold LOSO on {model_type.upper()} using device: {device}...")

    sequences = np.load(sequences_path) # (1331, 24, 1)
    meta_df = pd.read_csv(metadata_path)
    labels = meta_df["label"].values.astype(int)
    pids = meta_df["participant_id"].values
    win_indices = meta_df.index.values

    with open(splits_path, "r") as f:
        splits = json.load(f)

    outer_folds = splits["outer_folds"]
    n_windows = len(meta_df)
    oof_probs = np.zeros(n_windows, dtype=np.float32)

    for fold in outer_folds:
        f_id = fold["fold_id"]
        test_pid = fold["test_participant"]

        t_idx, t_probs = train_and_eval_single_fold(
            model_type=model_type,
            fold_info=fold,
            sequences_all=sequences,
            labels_all=labels,
            pids_all=pids,
            win_indices_all=win_indices,
            device=device,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
        )
        oof_probs[t_idx] = t_probs

    window_preds_df = pd.DataFrame({
        "window_id": meta_df["window_id"],
        "participant_id": meta_df["participant_id"],
        "group": meta_df["group"],
        "label": labels,
        "pred_prob": np.round(oof_probs, 5),
    })

    part_preds_df, metrics = aggregate_participant_predictions(
        window_df=window_preds_df,
        pred_col="pred_prob",
        true_col="label",
        pid_col="participant_id",
        threshold=0.5,
    )

    metrics["model_name"] = model_type
    metrics["input_resolution"] = "minute" if "minute" in sequences_path else "hourly"

    logger.info(
        f"Completed {model_type.upper()} LOSO -> Balanced Acc: {metrics['balanced_accuracy']*100:.2f}%, "
        f"Ordinary Acc: {metrics['accuracy']*100:.2f}%, Sens: {metrics['sensitivity']*100:.2f}%, "
        f"Spec: {metrics['specificity']*100:.2f}%, F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}"
    )

    return window_preds_df, part_preds_df, metrics


def run_all_deep_experiments(output_dir: str = "results") -> pd.DataFrame:
    """
    Run 55-fold LOSO cross-validation across LSTM, GRU, and 1D-CNN.
    """
    models = ["lstm", "gru", "cnn1d"]
    summary_list = []

    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    meta_df = pd.read_csv("data/processed/window_metadata.csv")
    unique_pids = sorted(meta_df["participant_id"].unique(), key=natural_sort_key)

    base_part_df = (
        meta_df[["participant_id", "group", "label"]]
        .drop_duplicates()
        .set_index("participant_id")
        .loc[unique_pids]
        .reset_index()
    )
    base_win_df = meta_df[["window_id", "participant_id", "group", "label"]].copy()

    for m in models:
        win_df, part_df, metrics = run_deep_learning_experiment(model_type=m)

        base_win_df[f"{m}_prob"] = win_df["pred_prob"]
        base_part_df[f"{m}_prob"] = part_df["pred_prob"].values
        base_part_df[f"{m}_pred"] = part_df["pred_label"].values

        summary_list.append({
            "model": m,
            "architecture": m.upper(),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1_score": metrics["f1_score"],
            "mcc": metrics["mcc"],
            "roc_auc": metrics["roc_auc"],
            "tp": metrics["confusion_matrix"]["tp"],
            "tn": metrics["confusion_matrix"]["tn"],
            "fp": metrics["confusion_matrix"]["fp"],
            "fn": metrics["confusion_matrix"]["fn"],
        })

    # Save outputs
    win_path = os.path.join(output_dir, "predictions", "deep_window_predictions.csv")
    part_path = os.path.join(output_dir, "predictions", "deep_participant_predictions.csv")
    table_path = os.path.join(output_dir, "metrics", "deep_results_table.csv")

    base_win_df.to_csv(win_path, index=False)
    base_part_df.to_csv(part_path, index=False)

    res_df = pd.DataFrame(summary_list)
    res_df.to_csv(table_path, index=False)

    logger.info(f"All deep learning experiments completed. Results saved to {table_path}")
    return res_df


if __name__ == "__main__":
    df_res = run_all_deep_experiments()
    print("\n=== FINAL DEEP LEARNING RESULTS (55-FOLD LOSO PARTICIPANT LEVEL) ===")
    print(df_res[["model", "balanced_accuracy", "accuracy", "sensitivity", "specificity", "f1_score", "roc_auc"]].to_string(index=False))
