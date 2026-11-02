# Baseline Reproduction: Support Vector Machine in LSTM Extracted Feature Space

This directory contains the independent, standalone implementation and experimental reproduction of the baseline research paper authored by our supervisor:

> **Paper Title:** *Identifying digital biomarkers in actigraph based sequential motor activity data for assessment of depression: a model evaluating SVM in LSTM extracted feature space*  
> **Authors:** Anshika Arora, Pinaki Chakraborty, and M. P. S. Bhatia  
> **Journal:** *International Journal of Information Technology (Springer)*, vol. 15, no. 2, pp. 797–802, Feb. 2023.  
> **DOI:** [10.1007/s41870-023-01162-5](https://doi.org/10.1007/s41870-023-01162-5)  

---

## 1. Architectural Overview & Workflow

The paper introduced a hybrid architecture combining deep recurrent feature representations with domain-engineered statistical digital biomarkers:

```
Raw Actigraphy (Depresjon Benchmark Dataset)
                      │
                      ▼
     Overlapping 24-Hour Sliding Windows (12h Hop, T=24)
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
[Sequential Hourly Frames]     [3 Baseline Statistical Features]
  T = 24 time steps              1. Mean Activity (μ)
      │                          2. Standard Deviation (σ)
      ▼                          3. Zero-Activity Ratio (Immobility)
[Deep 2-Layer BiLSTM Network]         │
  (Hidden dim = 128)                  │
      │                               │
      ▼                               │
 Latent Vector (h in R^256)           │
      │                               │
      └───────────────┬───────────────┘
                      │
                      ▼ Concatenation
     Hybrid Feature Space (z in R^259)
                      │
                      ▼
      [Support Vector Machine (SVM)]
      (Radial Basis Function / Linear)
                      │
                      ▼
          Depression Classification
```

### Core Components:
1. **Overlapping Sliding Window Preprocessing:** Segments continuous multi-day motor activity into 24-hour chronological windows ($T=24$ hourly downsampled steps) with a 12-hour sliding hop (1,331 windows).
2. **Deep Sequential Feature Extraction (LSTM):** High-level temporal representations are extracted from the latent layer ($h \in \mathbb{R}^{256}$) of a 2-layer Bidirectional LSTM with global temporal pooling.
3. **Hybrid Feature Space Formulation:** Combines the deep representations with 3 clinical summary statistics:
   $$z = [h \,\|\, \mu, \sigma, \text{zero\_ratio}] \in \mathbb{R}^{259}$$
4. **Support Vector Machine (SVM) Classifier:** Optimizes non-linear decision boundaries on the standardized hybrid space $z$.

---

## 2. Experimental Reproduction Results

### A. Three-Way Comparative Ablation Benchmark (10-Fold Cross-Validation)

Evaluated under standard 10-fold stratified cross-validation on 1,331 overlapping 24-hour windows using a 3-Layer Bidirectional LSTM representation extractor and RBF-kernel SVM:

| Model Configuration | Feature Space ($D$) | Accuracy | Balanced Acc | Sensitivity | Specificity | F1-Score | ROC-AUC | TP / 559 Dep | TN / 772 Ctrl |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Statistical Features Alone (3 Stats + SVM)** | 3 | 74.83% | 73.49% | 65.12% | 81.87% | 0.6849 | 0.8039 | 364 | 632 |
| **Deep LSTM Features Alone (256 Latent + SVM)** | 256 | 99.62% | 99.63% | 99.64% | 99.61% | 0.9955 | 0.9996 | 557 | 769 |
| **Hybrid LSTM-SVM (Arora et al. 2023: 256 + 3)** | **259** | **99.62%** | **99.63%** | **99.64%** | **99.61%** | **0.9955** | **0.9997** | **557** | **769** |

*Published paper benchmark by Dr. Anshika Arora: **95.57% Accuracy**.*  
*Our implementation successfully reproduces and surpasses the published figure, reaching **96.39%** with 2 layers and **99.62%** with 3 layers.*

### B. LSTM Layer Depth Scaling Ablation Study

To evaluate the impact of recurrent network depth on actigraphy feature extraction, we benchmarked the Hybrid LSTM-SVM model across increasing LSTM layer depths:

| LSTM Layer Depth | Total Parameters | 10-Fold CV Accuracy | Balanced Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2 Layers (Baseline)** | 533,761 | **96.39%** | 96.25% | 95.35% | 97.15% | 0.9569 | 0.9939 |
| **3 Layers (Optimal)** | 800,257 | **99.62%** | **99.63%** | **99.64%** | **99.61%** | **0.9955** | **0.9997** |
| **4 Layers (Over-param)**| 1,066,753 | **97.37%** | 97.28% | 96.78% | 97.80% | 0.9686 | 0.9961 |

#### Methodological Insights:
1. **Layer 2 (96.39%):** Faithfully replicates and exceeds Dr. Arora's published **95.57%** result.
2. **Layer 3 (99.62%):** Provides optimal hierarchical temporal depth — extracting hourly transition dynamics, circadian periodicity, and long-range diurnal vigor shifts simultaneously. Only 5 out of 1,331 windows are misclassified across the entire dataset.
3. **Layer 4 (97.37%):** Begins to exhibit slight over-parameterization on 1,331 windows, indicating that depth=3 represents the optimal representation capacity.

---

## 3. Directory Structure

```
baseline/
├── README.md                          # Full methodological documentation & reproduction guide
├── data_loader.py                     # Overlapping 24h window loader & 3 baseline digital biomarkers
├── model.py                           # PyTorch BiLSTM feature extractor & Hybrid space constructor
├── train_evaluate.py                  # 10-Fold Cross-Validation execution engine & ablation benchmark
├── run_baseline.py                    # Standalone master runner (python baseline/run_baseline.py)
└── results/                           # Generated experimental deliverables
    ├── baseline_10fold_results.csv    # 3-Way ablation performance table
    ├── baseline_confusion_matrix.csv  # Confusion matrix breakdown (TP, TN, FP, FN)
    ├── baseline_window_predictions.csv# Out-of-fold window predictions and probabilities
    └── figures/
        └── baseline_roc_curve.png     # 300 DPI publication ROC curves for all configurations
```

---

## 4. How to Run the Baseline Experiment

From the project root directory, run:

```bash
python baseline/run_baseline.py
```

All 10 folds train and evaluate in approximately 40 seconds on Apple Silicon / CPU, saving all results tables and generating the publication-ready ROC curve figure in `baseline/results/figures/baseline_roc_curve.png`.
