"""
Statistical Branch Training & Evaluation Module.

Evaluates classical machine learning classifiers on actigraphy features using
the 55-fold Leave-One-Subject-Out (LOSO) cross-validation protocol:
1. Logistic Regression (L2 regularized)
2. Linear Support Vector Machine (Linear SVM)
3. RBF Support Vector Machine (RBF SVM)
4. Random Forest Classifier

Evaluated on two feature configurations (Ablation study):
- Baseline (3 features): mean, std, zero_ratio (matching Arora et al. 2023 / Garcia-Ceja et al. 2018)
- Full (21 features): All extracted clinical digital biomarkers

Strict Leakage Prevention:
- StandardScaler is fitted strictly on the 54 training subjects in each fold.
- Hyperparameter selection is performed strictly within the inner validation folds.
- Primary evaluation is strictly at the participant level (unseen held-out subjects).
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.metrics import calculate_classification_metrics, aggregate_participant_predictions
from preprocessing.clean_data import natural_sort_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_model_and_grid(model_name: str, random_state: int = 42):
    """Return model instance and hyperparameter grid for inner validation."""
    if model_name == "logistic_regression":
        base_model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=random_state)
        param_grid = [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    elif model_name == "linear_svm":
        base_model = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state)
        param_grid = [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}]
    elif model_name == "rbf_svm":
        base_model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)
        param_grid = [{"C": 0.1, "gamma": "scale"}, {"C": 1.0, "gamma": "scale"}, {"C": 10.0, "gamma": "scale"}]
    elif model_name == "random_forest":
        base_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1)
        param_grid = [
            {"max_depth": 3, "min_samples_split": 5},
            {"max_depth": 5, "min_samples_split": 5},
            {"max_depth": 8, "min_samples_split": 2},
            {"max_depth": None, "min_samples_split": 2},
        ]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return base_model, param_grid


def tune_hyperparameters_inner(
    model_name: str,
    param_grid: List[Dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_splits: List[Dict],
    random_state: int = 42,
) -> Dict:
    """
    Tune model hyperparameters strictly across inner validation folds.
    Returns the parameter configuration maximizing mean inner validation balanced accuracy.
    """
    if len(param_grid) <= 1:
        return param_grid[0]

    best_score = -1.0
    best_params = param_grid[0]

    for params in param_grid:
        inner_scores = []
        for inner_fold in inner_splits:
            val_idx = inner_fold["val_window_indices"]
            tr_idx = inner_fold["train_window_indices"]

            # Map global indices to relative positions in X_train
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            # Fit candidate model
            model, _ = get_model_and_grid(model_name, random_state)
            model.set_params(**params)
            model.fit(X_tr, y_tr)

            val_probs = model.predict_proba(X_val)[:, 1]
            m = calculate_classification_metrics(y_val, val_probs)
            inner_scores.append(m["balanced_accuracy"])

        mean_inner_score = float(np.mean(inner_scores))
        if mean_inner_score > best_score:
            best_score = mean_inner_score
            best_params = params

    return best_params


def evaluate_statistical_model_loso(
    model_name: str,
    feature_cols: List[str],
    features_df: pd.DataFrame,
    splits_dict: Dict,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run 55-fold Leave-One-Subject-Out cross-validation for a specific model & feature set.

    Returns:
        window_preds_df: Out-of-fold predictions for every window (1,331 rows).
        participant_preds_df: Aggregated predictions for each participant (55 rows).
        metrics: Participant-level clinical classification metrics.
    """
    X_all = features_df[feature_cols].values.astype(np.float32)
    y_all = features_df["label"].values.astype(int)
    pids_all = features_df["participant_id"].values
    win_ids_all = features_df["window_id"].values

    outer_folds = splits_dict["outer_folds"]
    n_folds = len(outer_folds)
    n_windows = len(features_df)

    oof_window_probs = np.zeros(n_windows, dtype=np.float32)
    fold_ids_recorded = np.zeros(n_windows, dtype=int)

    base_model, param_grid = get_model_and_grid(model_name, random_state)

    for fold_info in outer_folds:
        f_id = fold_info["fold_id"]
        test_pid = fold_info["test_participant"]
        test_idx = fold_info["test_window_indices"]
        train_idx = fold_info["train_window_indices"]

        # 1. Strict Train / Test Split
        X_train_raw = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test_raw = X_all[test_idx]

        # 2. Fit Scaler ONLY on Training Windows
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 3. Inner Hyperparameter Tuning (using inner relative mappings)
        # Map global train indices to relative train indices [0 .. len(train_idx)-1]
        global_to_rel = {g_idx: r_idx for r_idx, g_idx in enumerate(train_idx)}
        rel_inner_splits = []
        for inf in fold_info["inner_splits"]:
            rel_inner_splits.append({
                "val_window_indices": [global_to_rel[g] for g in inf["val_window_indices"]],
                "train_window_indices": [global_to_rel[g] for g in inf["train_window_indices"]],
            })

        best_params = tune_hyperparameters_inner(
            model_name=model_name,
            param_grid=param_grid,
            X_train=X_train_scaled,
            y_train=y_train,
            inner_splits=rel_inner_splits,
            random_state=random_state,
        )

        # 4. Fit Final Model on All 54 Training Subjects
        final_model, _ = get_model_and_grid(model_name, random_state)
        final_model.set_params(**best_params)
        final_model.fit(X_train_scaled, y_train)

        # 5. Predict on Unseen Held-Out Subject Windows
        test_probs = final_model.predict_proba(X_test_scaled)[:, 1]

        oof_window_probs[test_idx] = test_probs
        fold_ids_recorded[test_idx] = f_id

    # Assemble Window Predictions DataFrame
    window_preds_df = pd.DataFrame({
        "window_id": win_ids_all,
        "participant_id": pids_all,
        "fold_id": fold_ids_recorded,
        "true_label": y_all,
        "pred_prob": np.round(oof_window_probs, 5),
    })

    # Participant-level Aggregation
    participant_preds_df, participant_metrics = aggregate_participant_predictions(
        window_df=window_preds_df,
        pred_col="pred_prob",
        true_col="true_label",
        pid_col="participant_id",
        threshold=0.5,
    )

    participant_metrics["model_name"] = model_name
    participant_metrics["feature_set"] = "baseline_3" if len(feature_cols) == 3 else "all_21"
    participant_metrics["num_features"] = len(feature_cols)

    return window_preds_df, participant_preds_df, participant_metrics


def run_all_statistical_experiments(
    features_path: str = "data/processed/features_statistical.csv",
    splits_path: str = "results/splits/loso_splits.json",
    output_dir: str = "results",
) -> pd.DataFrame:
    """
    Run the complete matrix of classical classifiers across both feature sets.
    """
    logger.info("Loading extracted features and LOSO split definitions...")
    features_df = pd.read_csv(features_path)
    with open(splits_path, "r") as f:
        splits_dict = json.load(f)

    with open("data/processed/feature_metadata.json", "r") as f:
        feat_meta = json.load(f)

    baseline_feats = feat_meta["baseline_3_features"]
    all_feats = feat_meta["feature_names_all"]

    models_to_run = ["logistic_regression", "linear_svm", "rbf_svm", "random_forest"]
    feature_configs = [
        ("baseline_3", baseline_feats),
        ("all_21", all_feats),
    ]

    all_window_records = {}
    all_participant_records = {}
    summary_metrics_list = []

    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    for feat_name, f_cols in feature_configs:
        for m_name in models_to_run:
            exp_id = f"{m_name}_{feat_name}"
            logger.info(f"Running 55-fold LOSO CV: {exp_id} ({len(f_cols)} features)...")

            win_df, part_df, metrics = evaluate_statistical_model_loso(
                model_name=m_name,
                feature_cols=f_cols,
                features_df=features_df,
                splits_dict=splits_dict,
            )

            all_window_records[f"{exp_id}_prob"] = win_df["pred_prob"]
            all_participant_records[f"{exp_id}_prob"] = part_df["pred_prob"]
            all_participant_records[f"{exp_id}_pred"] = part_df["pred_label"]

            summary_metrics_list.append({
                "experiment_id": exp_id,
                "model": m_name,
                "feature_set": feat_name,
                "num_features": len(f_cols),
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

            logger.info(
                f"Completed {exp_id} -> Balanced Acc: {metrics['balanced_accuracy']*100:.2f}%, "
                f"Sens: {metrics['sensitivity']*100:.2f}%, Spec: {metrics['specificity']*100:.2f}%, "
                f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}"
            )

    # Compile master prediction files
    base_win_df = features_df[["window_id", "participant_id", "group", "label"]].copy()
    for col_name, s in all_window_records.items():
        base_win_df[col_name] = s
    win_out_path = os.path.join(output_dir, "predictions", "statistical_window_predictions.csv")
    base_win_df.to_csv(win_out_path, index=False)

    unique_pids = sorted(features_df["participant_id"].unique(), key=natural_sort_key)
    base_part_df = (
        features_df[["participant_id", "group", "label"]]
        .drop_duplicates()
        .set_index("participant_id")
        .loc[unique_pids]
        .reset_index()
    )
    for col_name, s in all_participant_records.items():
        base_part_df[col_name] = s.values
    part_out_path = os.path.join(output_dir, "predictions", "statistical_participant_predictions.csv")
    base_part_df.to_csv(part_out_path, index=False)

    results_table = pd.DataFrame(summary_metrics_list)
    table_out_path = os.path.join(output_dir, "metrics", "statistical_results_table.csv")
    results_table.to_csv(table_out_path, index=False)

    logger.info(f"All statistical experiments completed successfully. Results saved to {table_out_path}")
    return results_table


if __name__ == "__main__":
    df_res = run_all_statistical_experiments()
    print("\n=== FINAL STATISTICAL BRANCH RESULTS (PARTICIPANT-LEVEL 55-FOLD LOSO) ===")
    print(df_res[["experiment_id", "balanced_accuracy", "sensitivity", "specificity", "f1_score", "mcc", "roc_auc"]].to_string(index=False))
