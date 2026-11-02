# Phase 2 Preprocessing & Splitting Methodology

**Project:** A Comparative Study of Deep Learning Architectures and Feature Fusion Strategies for Depression Detection Using Actigraphy Data  
**Group:** DTU/CSE/BTP/GP178  
**Date:** October 31, 2026  

---

## 1. Overview and Objectives
Phase 2 establishes the shared, reproducible preprocessing and evaluation infrastructure. To guarantee research integrity and comparability, all experimental branches (Statistical Classifiers, Deep Learning Architectures, and Multi-modal Feature/Decision Fusion) load the exact same preprocessed window arrays and adhere to the exact same cross-validation splits.

---

## 2. Implemented Methodological Choices

### 2.1 Recording Duration Scope (Option B: Truncated by `scores.csv`)
- **Rationale:** To establish an empirical baseline comparable with prior literature on the Depresjon dataset (e.g., Garcia-Ceja et al. 2018, Dr. Arora et al. 2023), each participant's recording is truncated to the first $N$ days specified in the `days` column of `data/scores.csv`.
- **Verified Data Span:** 
  - 51 participants had continuously recorded minutes exactly matching $N \times 1,440$ minutes.
  - 4 participants had $N \times 1,440 - 60$ minutes due to a single 60-minute Daylight Saving Time transition on March 30, 2003.
  - Across all 55 participants, the active analysis spans exactly **693 days** ($997,920$ minutes).

### 2.2 Missing Data Policy & Regular Time Alignment
- **Regular Grid:** Each participant series was reindexed to a continuous 1-minute `DatetimeIndex` from $t_0$ to $t_0 + (N \times 1440 - 1) \text{ minutes}$.
- **Interpolation:** For the 4 participants affected by the European DST clock change (`control_6`, `control_7`, `control_31`, `control_32`), the 60 missing minutes were linearly interpolated.
- **Traceability:** Every interpolated minute was logged in an indicator array (`is_imputed = True`). Across the entire dataset, only **240 minutes** out of $997,920$ minutes ($0.0241\%$) were imputed.
- **Validity Rule:** Any window with $>5\%$ missing/imputed minutes is excluded. Because a 60-minute gap inside a 1,440-minute window accounts for only $4.17\%$, all generated windows are verified valid, with exactly 8 windows flagged with `has_imputed_data = True`.

### 2.3 Window Generation Protocol
- **Window Length:** 24 hours ($1,440$ minutes).
- **Hop Size:** 12 hours ($720$ minutes; 50% overlap).
- **Hourly Downsampling Representation:** Each 60 consecutive 1-minute points are averaged into an hourly mean activity count, yielding a 1D sequence of length **24** ($24 \times 1$).
- **Minute-Resolution Representation:** Preserves all **1,440** 1-minute values ($1440 \times 1$) for fine-grained temporal and statistical feature extraction.
- **Window Yield:**
  - Formula: For a participant with $N$ days, the number of overlapping windows is $2N - 1$.
  - Total Windows: $(2 \times 693) - 55 = \mathbf{1,331}$ windows.
  - Condition Windows: **559** ($42.0\%$).
  - Control Windows: **772** ($58.0\%$).
- **Participant Boundary Protection:** Windows never cross participant boundaries. Every window is strictly from one subject.

---

## 3. Evaluation Protocol: Leave-One-Subject-Out (LOSO) Cross-Validation

### 3.1 Unit of Generalization
The independent biological unit is the **participant**, not the individual window. Evaluating overlapping windows with standard K-Fold CV introduces severe data leakage, as slices of the same subject's 24-hour recordings appear in both train and test partitions.

### 3.2 Outer Cross-Validation (55 Folds)
- Exactly 55 outer folds are defined in `results/splits/loso_splits.json`.
- In fold $k \in \{0, \dots, 54\}$:
  - Test Set: All windows belonging to participant $P_k$ ($N_{\text{test}} = 1$).
  - Training Pool: All windows belonging to the remaining 54 participants ($N_{\text{train}} = 54$).
  - Overlap: Exactly **0** shared participants and **0** shared windows.

### 3.3 Nested Inner Validation (5 Folds per Outer Fold)
- Within the 54 training subjects, a 5-fold **Stratified Group K-Fold** cross-validation scheme is defined.
- Grouping is on `participant_id` and stratification is on diagnostic label.
- Inner folds are utilized exclusively for:
  - Scaling parameter computation (fitting Scaler on train folds only).
  - Feature selection.
  - Hyperparameter tuning and model selection.
  - Early stopping in neural network architectures.
  - Fusion weight ($\alpha$) and stacking meta-learner estimation.
- The outer test subject $P_k$ is completely isolated from all inner decisions.

---

## 4. Derived Artifacts Summary
The following reproducible files have been generated:
1. `data/processed/windows_hourly.npy` (Shape: `(1331, 24, 1)`, float32)
2. `data/processed/windows_minute.npy` (Shape: `(1331, 1440, 1)`, float32)
3. `data/processed/window_metadata.csv` (1,331 rows with full metadata)
4. `data/processed/window_summary.json` (Structured summary)
5. `results/splits/loso_splits.json` (Complete 55-fold outer and nested inner split indices)
6. `results/splits/loso_splits_summary.csv` (Fold-by-fold summary table)
