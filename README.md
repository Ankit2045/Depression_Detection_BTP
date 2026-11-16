# Multimodal Feature Fusion of Statistical and Deep Learning Representations for Actigraphy-Based Depression Detection

**Delhi Technological University (DTU)**  
**Department of Computer Science and Engineering**  
**B.Tech Final Year Project (Project-I), AY 2026–27 | Group ID: `DTU/CSE/BTP/GP178`**

- **Team Members:** Ankit Singh (23/CS/064), Ankit (23/CS/059), Kumari Rashi (24/CS/253)
- **Project Supervisor:** Dr. Anshika Arora, Assistant Professor, Department of CSE, DTU

---

## 1. Project Overview & Research Contributions
This repository contains a complete, reproducible research pipeline investigating objective depression classification using continuous motor actigraphy from the benchmark **Depresjon dataset** (55 participants: 23 depressed, 32 healthy controls).

Unlike prior literature that relies on window-level splits susceptible to severe data leakage, our entire experimental framework is built upon a **leak-free 55-fold Leave-One-Subject-Out (LOSO) nested cross-validation protocol**.

### Key Empirical Findings:
1. **Digital Biomarkers Outperform Summary Statistics:**
   Expanding from 3 basic summary statistics (`mean`, `std`, `zero_ratio`) to **21 domain-engineered digital biomarkers** (circadian $M10, L5, RA$, daytime inactivity, volatility) boosted balanced accuracy from $66.37\% \rightarrow \mathbf{79.76\%}$ (+13.39%) and sensitivity from $60.87\% \rightarrow \mathbf{78.26\%}$ with regularized Logistic Regression.
2. **1D-CNN is the Deep Learning Champion:**
   A lightweight temporal 1D-CNN (only 6,593 parameters) outperformed complex recurrent architectures (LSTM, GRU), reaching an **ROC-AUC of 0.8899** and **79.76% balanced accuracy**.
3. **Multimodal Dual-Fusion Ensemble Delivers Breakthrough Synergy:**
   Fusing statistical digital biomarkers with deep temporal representations (**LogReg-21 + 1D-CNN**) via our **Dual-Fusion Ensemble** (averaging weighted linear probability with non-linear stacking) reached **92.73% Raw Accuracy (51/55 participants correct)**, **92.53% Balanced Accuracy**, **91.30% Sensitivity** (detecting 21 of 23 depressed patients), **93.75% Specificity** (clearing 30 of 32 healthy controls), and **0.9470 ROC-AUC** (+20.00% absolute boost over prior literature, $p = 0.0076$).
4. **Window-Level Baseline Reproduction & Scaling (Arora et al. 2023):**
   In an isolated module (`baseline/`), we faithfully reproduced Dr. Arora's 2023 BiLSTM-SVM architecture (achieving **96.39%**, beating the published 95.57%) and demonstrated that scaling to 3 bidirectional layers reaches **99.62% window accuracy (1,326/1,331 windows, ROC-AUC 0.9997)**.
5. **Clinical AI Co-Pilot & Interactive Web Dashboard (`ai_system/`):**
   Integrated an interpretable 4-head Temporal Self-Attention Transformer, a Generative AI Co-Pilot synthesizing DSM-5 aligned psychiatric consultation notes, and an interactive clinician dashboard with real-time inference on new smartwatch actigraphy CSVs.

---

## 2. Repository Structure

```text
BTP/
├── configs/
│   ├── data.yaml                   # Centralized dataset, windowing & LOSO CV configuration
│   └── models.yaml                 # PyTorch neural architectures & hyperparameters
├── data/
│   ├── condition/                  # 23 raw condition CSVs (immutable)
│   ├── control/                    # 32 raw control CSVs (immutable)
│   ├── scores.csv                  # Demographic & clinical metadata
│   └── processed/                  # Sliced 24h window arrays & feature tables
├── preprocessing/
│   ├── audit_data.py               # Dataset audit & participant manifest generator
│   ├── clean_data.py               # Time grid alignment & DST gap handling
│   ├── create_windows.py           # 24h overlapping window generation (hop: 12h)
│   └── create_splits.py            # 55-fold LOSO outer and 5-fold inner group splits
├── statistical/
│   ├── extract_features.py         # 21 clinical digital biomarkers extraction
│   └── train_statistical.py        # 55-fold LOSO training for classical ML models
├── deep_learning/
│   ├── datasets.py                 # PyTorch Dataset with training-only scaling
│   ├── lstm.py                     # 2-layer LSTM architecture
│   ├── gru.py                      # 2-layer GRU architecture
│   ├── cnn1d.py                    # Temporal 1D-CNN architecture
│   └── train_deep.py               # GPU/MPS training engine with leak-free early stopping
├── ai_system/                      # Clinical AI Co-Pilot & Web Dashboard
│   ├── transformer_attention.py    # 4-head Temporal Self-Attention Transformer
│   ├── clinical_agent.py           # Generative AI DSM-5 Consultation Memo engine
│   ├── predict_new_patient.py      # Turnkey inference CLI for external smartwatch CSVs
│   ├── dashboard_server.py         # Dedicated HTTP server with REST prediction API
│   └── dashboard/                  # Zero-dependency interactive clinician web UI
├── baseline/                       # Arora et al. (2023) Reproduction & Layer Scaling (99.62%)
│   ├── model.py                    # BiLSTM-SVM hybrid architecture with scalable depth
│   ├── train_evaluate.py           # 10-fold window-level CV engine
│   └── run_baseline.py             # Layer ablation runner (L = 2, 3, 4)
├── fusion/
│   ├── probability_fusion.py       # Probability averaging & leak-free weighted fusion
│   ├── stacking.py                 # Second-level Stacking with Logistic Regression
│   └── feature_fusion.py           # 85-dimensional feature-level concatenation
├── evaluation/
│   ├── metrics.py                  # Clinical metrics engine & participant aggregation
│   └── generate_figures.py         # Generates high-resolution 300 DPI figures
├── experiments/
│   ├── run_statistical.py          # Master CLI for statistical benchmark
│   ├── run_deep_learning.py        # Master CLI for deep learning benchmark
│   ├── run_fusion.py               # Master CLI for multimodal fusion benchmark
│   └── run_all.py                  # Master end-to-end pipeline execution
├── show_accuracy.py                # Instant terminal accuracy & benchmark summary
├── results/
│   ├── predictions/                # Complete out-of-fold predictions (participant & window)
│   ├── metrics/                    # Master CSV results tables
│   └── figures/                    # Publication-ready figures (PNG 300 DPI)
├── reports/
│   ├── dataset_audit_report.md     # Phase 1 dataset audit findings
│   ├── preprocessing_methodology.md# Phase 2 windowing and splitting specification
│   ├── statistical_branch_report.md# Phase 3 classical ML results and discussion
│   └── research_paper_draft.md     # Complete living draft for publication
└── tests/                          # 46 automated unit and integration tests
```

---

## 3. Quickstart & Execution

### Instant Benchmark & Accuracy Summary
To display the complete comparative benchmark table:
```bash
python show_accuracy.py
```

### Compile Publication-Ready Research Paper (PDF & HTML)
To build the complete 19-page publication-styled academic paper with embedded figures, tables, and mathematical formulas:
```bash
python scripts/compile_paper.py
```
- **PDF Document:** [`reports/Research_Paper_BTP_GP178.pdf`](reports/Research_Paper_BTP_GP178.pdf)
- **Interactive HTML:** [`reports/Research_Paper_BTP_GP178.html`](reports/Research_Paper_BTP_GP178.html)

### Clinician Web Dashboard
- **Live Online Dashboard:** [https://ankit2045.github.io/Depression_Detection_BTP/](https://ankit2045.github.io/Depression_Detection_BTP/)
- **Local Launch:**
```bash
python -m http.server 8000 --directory ai_system/dashboard
# or for full PyTorch inference server:
python ai_system/dashboard_server.py 8000
```
Open **`http://localhost:8000`** to browse all 65 cohort participants, inspect 24h waveforms, and test new CSV files.

### Run Automated Test Suite (46 Tests)
```bash
pytest tests/
```

---

## 4. Benchmark Performance Summary (55-Fold LOSO)

| Experiment Category | Model / Strategy | Balanced Acc | Ordinary Acc | Sensitivity | Specificity | F1-Score | ROC-AUC | Correct / 55 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Literature Baseline [1]** | Linear SVM (3 Feats) | 71.06% | 72.73% | 60.87% | 81.25% | 0.6512 | 0.7337 | 40 / 55 |
| **Statistical Champion** | Logistic Regression (21 Feats) | 79.76% | 80.00% | 78.26% | 81.25% | 0.7660 | 0.8587 | 44 / 55 |
| **Deep Learning Champion** | Temporal 1D-CNN | 79.76% | 80.00% | 78.26% | 81.25% | 0.7660 | 0.8899 | 44 / 55 |
| **Fusion (Probability Avg)** | LogReg-21 + 1D-CNN | 89.40% | 89.09% | **91.30%** | 87.50% | 0.8750 | **0.9524** | 49 / 55 |
| **Fusion (Weighted)** | LogReg-21 + 1D-CNN | 90.96% | 90.91% | **91.30%** | 90.62% | 0.8936 | 0.9429 | 50 / 55 |
| **Fusion (Stacking)** | LogReg-21 + 1D-CNN | 90.96% | 90.91% | **91.30%** | 90.62% | 0.8936 | 0.9429 | 50 / 55 |
| **★ Dual-Fusion Champion ★** | **Weighted + Stacking Ensemble** | **92.53%** | **92.73%** | **91.30%** | **93.75%** | **0.9130** | **0.9470** | **51 / 55** |

*(McNemar exact test between Literature Baseline and Fused Champion: $p = 0.0076 < 0.01$)*.

---

## 5. Research Deliverables
- **Academic Paper Draft:** [`reports/research_paper_draft.md`](reports/research_paper_draft.md)
- **High-Resolution Figures:** [`results/figures/`](results/figures/) (Figures 1 through 10 at 300 DPI)
- **Clinical AI Notes (55 Patients):** [`ai_system/results/clinical_notes/`](ai_system/results/clinical_notes/)
- **Out-of-Fold Predictions:** [`results/predictions/`](results/predictions/)
