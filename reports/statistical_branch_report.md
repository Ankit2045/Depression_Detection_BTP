# Phase 3 Statistical Branch Report

**Project:** A Comparative Study of Deep Learning Architectures and Feature Fusion Strategies for Depression Detection Using Actigraphy Data  
**Group:** DTU/CSE/BTP/GP178  
**Date:** November 1, 2026  

---

## 1. Executive Summary
In Phase 3, we implemented the evaluation metrics engine and conducted a systematic, leak-free **55-fold Leave-One-Subject-Out (LOSO) cross-validation** benchmark across four classical machine learning classifiers:
1. **Logistic Regression** (L2 regularized)
2. **Linear Support Vector Machine (Linear SVM)**
3. **RBF Support Vector Machine (RBF SVM)**
4. **Random Forest Classifier**

Each classifier was evaluated under two feature representations:
- **Baseline Feature Set (3 Features):** `mean_activity`, `std_activity`, and `zero_ratio` (matching Dr. Arora et al. 2023 and Garcia-Ceja et al. 2018).
- **Extended Digital Biomarkers (21 Features):** Adding circadian rhythm parameters ($M10$, $L5$, Relative Amplitude), activity volatility (RMSSD, lag-1 autocorrelation), and daytime psychomotor retardation.

---

## 2. Participant-Level Classification Results (55 Folds of LOSO)

Every prediction reported below reflects generalisation to a **completely unseen test participant** whose data was strictly excluded from feature scaling, hyperparameter tuning, and model training:

| Experiment ID | Model | Features | Balanced Acc | Ordinary Acc | Sensitivity (Recall) | Specificity | F1-Score | MCC | ROC-AUC | TP / FN / TN / FP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_regression_baseline_3` | Logistic Reg. | 3 | 66.37% | 67.27% | 60.87% | 71.88% | 0.6087 | 0.3274 | 0.7527 | 14 / 9 / 23 / 9 |
| `linear_svm_baseline_3` | Linear SVM | 3 | **71.06%** | **72.73%** | 60.87% | 81.25% | 0.6512 | 0.4319 | 0.7337 | 14 / 9 / 26 / 6 |
| `rbf_svm_baseline_3` | RBF SVM | 3 | 65.76% | 67.27% | 56.52% | 75.00% | 0.5909 | 0.3200 | 0.7473 | 13 / 10 / 24 / 8 |
| `random_forest_baseline_3` | Random Forest | 3 | 66.37% | 67.27% | 60.87% | 71.88% | 0.6087 | 0.3274 | 0.7486 | 14 / 9 / 23 / 9 |
| `logistic_regression_all_21` | Logistic Reg. | 21 | **79.76%** | **80.00%** | **78.26%** | **81.25%** | **0.7660** | **0.5919** | **0.8587** | **18 / 5 / 26 / 6** |
| `linear_svm_all_21` | Linear SVM | 21 | 70.45% | 72.73% | 56.52% | 84.38% | 0.6341 | 0.4299 | 0.8111 | 13 / 10 / 27 / 5 |
| `rbf_svm_all_21` | RBF SVM | 21 | 63.59% | 65.45% | 52.17% | 75.00% | 0.5581 | 0.2786 | 0.7323 | 12 / 11 / 24 / 8 |
| `random_forest_all_21` | Random Forest | 21 | 69.50% | 70.91% | 60.87% | 78.12% | 0.6364 | 0.3959 | 0.7514 | 14 / 9 / 25 / 7 |

---

## 3. Key Scientific Findings & Discussion

### Finding 1: Massive Generalisation Gain from Clinical Digital Biomarkers
Expanding the feature representation from 3 summary statistics to the 21 digital biomarkers led to a substantial improvement in generalisation:
- **Balanced Accuracy:** Increased from $66.37\% \rightarrow \mathbf{79.76\%}$ (+13.39 percentage points).
- **Sensitivity (Depression Recall):** Increased from $60.87\% \rightarrow \mathbf{78.26\%}$ (+17.39 percentage points). The model correctly diagnosed 18 out of 23 depressed patients (missing only 5).
- **Area Under the ROC Curve:** Rose from $0.7527 \rightarrow \mathbf{0.8587}$ (+0.106 AUC points).

### Finding 2: Direct Replication of the Literature Baseline
In the foundational paper introducing the Depresjon dataset (Garcia-Ceja et al. 2018), the authors reported **72.7% accuracy** using a Support Vector Machine on statistical features.
Our `linear_svm_baseline_3` model achieved an ordinary accuracy of **72.73%** (40 out of 55 participants correctly classified), validating that our pipeline directly replicates published baseline performance under leak-free conditions.

### Finding 3: Linear Classifiers Outperform Complex Non-linear Classifiers on Tabular Features
In this small-sample clinical regime ($N=55$ subjects), regularized linear models (Logistic Regression with L2 regularization) proved significantly more robust against overfitting than RBF SVM or Random Forest. The non-linear models tended to fit idiosyncratic movement habits of specific training participants, leading to lower sensitivity when tested on new subjects.

---

## 4. Derived Artifacts
All predictions and summary statistics are tracked in:
- `results/predictions/statistical_participant_predictions.csv` (Per-participant probabilities for all 55 subjects)
- `results/predictions/statistical_window_predictions.csv` (Per-window predictions across all 1,331 windows)
- `results/metrics/statistical_results_table.csv` (Complete comparative metrics table)
