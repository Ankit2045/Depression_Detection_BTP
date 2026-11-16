# PROJECT GP178 — TEAM ONBOARDING & INSTANT CONTEXT PROMPT

> **How to use:** Copy and paste the prompt below into a fresh Antigravity session. It will immediately ingest the complete project context, architecture, experimental results, and codebase without needing to re-run or re-explain anything.

---

```markdown
You are assisting with the Delhi Technological University (DTU) B.Tech Final Year Research Project (AY 2026–27).
Group ID: DTU/CSE/BTP/GP178
Supervisor: Dr. Anshika Arora, Assistant Professor, Department of Computer Science & Engineering, DTU.
Project Title: Multimodal Feature Fusion of Statistical and Deep Learning Representations for Actigraphy-Based Depression Detection
Team Members: Ankit (23/CS/059), Ankit Singh (23/CS/064), and Kumari Rashi (24/CS/253, Junior Member).

================================================================================
1. CORE RESEARCH PROBLEM & METHODOLOGICAL CONTRIBUTIONS
================================================================================
We investigate objective depression detection using continuous wearable motor actigraphy from the benchmark Depresjon dataset (55 subjects: 23 depressed, 32 healthy controls; 1.57M minute observations).

Key Breakthroughs & Findings:
1. Exposed the Window-Level Data Leakage Trap (Arora et al. 2023 Reproduction):
   - In `baseline/`, we faithfully reproduced Dr. Arora's 2023 BiLSTM-SVM architecture (achieving 96.39%, beating published 95.57%).
   - Scaled depth to 3 bidirectional layers (BiLSTM-3) reaching 99.62% window accuracy (1,326/1,331 windows, ROC-AUC 0.9997).
   - THE CATCH: We identified that 95-99% window-level accuracy in the literature is an artifact of 50% temporal overlap leakage across sliding windows (slices of the same patient bleed across train and test sets; models memorize biometric identity rather than depression).
2. The Literature LOSO Dilemma:
   - When evaluated under strict Leave-One-Subject-Out (LOSO) cross-validation on completely unseen patients, published models drop to 71%-73% (Garcia-Ceja 72.7%, Frogner 1D-CNN 71%, Linear SVM 71.06% balanced acc with 60.87% sensitivity).
3. Statistical Digital Biomarkers Champion:
   - Engineered 21 clinical digital biomarkers (circadian M10, L5, Relative Amplitude RA, daytime sedentary ratio, RMSSD, IQR) in `statistical/extract_features.py`.
   - Regularized Logistic Regression boosted balanced accuracy from 66.37% -> 79.76% (+13.39%) and sensitivity to 78.26%.
4. Deep Learning Champion:
   - A lightweight temporal 1D-CNN (`deep_learning/cnn1d.py`, only 6,593 parameters) beat complex recurrent architectures (LSTM: 70.45%, GRU: 74.46%), reaching 79.76% balanced accuracy and 0.8899 ROC-AUC.
   - Hourly downsampling (T=24) beat raw 1-minute sequences (T=1,440) by suppressing movement noise.
5. Champion Dual-Fusion Ensemble (92.73% Subject-Level Accuracy):
   - In `fusion/`, we fused domain biomarkers with 1D-CNN representations.
   - Probability Averaging: 89.40% Bal Acc, 0.9524 ROC-AUC.
   - Weighted Probability (alpha=0.55): 90.96% Bal Acc.
   - Stacking Meta-Learner (Logistic Regression): 90.96% Bal Acc.
   - ★ Dual-Fusion Champion Ensemble ★ (averaging Weighted + Stacking):
     • Raw Accuracy: 92.73% (51 out of 55 participants correct on unseen folds)
     • Balanced Accuracy: 92.53%
     • Clinical Sensitivity: 91.30% (detected 21 of 23 depressed patients)
     • Clinical Specificity: 93.75% (cleared 30 of 32 healthy controls)
     • ROC-AUC: 0.9470
     • McNemar exact test: p = 0.0076 < 0.01 (statistically significant gain over literature).
6. Explainable AI (XAI) & Clinical Decision Support:
   - Out-of-fold SHAP identified Daytime Sedentary Ratio (Odds Ratio = 3.34, p = 0.00016) as primary positive risk factor, and Nocturnal Mean Activity (OR = 0.31, p = 0.00008) and Relative Amplitude (RA) as protective markers.
   - Clinical error forensics on the 4 borderline cases (e.g. condition_2 was in clinical remission, MADRS dropping 24 -> 11).
   - Generative AI Clinical Co-Pilot (`ai_system/clinical_agent.py`) producing DSM-5 consultation notes.
   - Interactive Clinician Web Dashboard (`ai_system/dashboard/`).

================================================================================
2. REPOSITORY ARCHITECTURE & FILE LOCATIONS
================================================================================
- `configs/data.yaml`, `configs/models.yaml`: Centralized configuration.
- `data/condition/` (23 CSVs), `data/control/` (32 CSVs), `data/scores.csv`: Raw immutable dataset.
- `data/processed/`: Sliced 24h window arrays (1,331 windows, 559 condition, 772 control) & feature CSVs.
- `preprocessing/`: Audit (`audit_data.py`), cleaning/imputation (`clean_data.py`), windowing (`create_windows.py`), and 55-fold LOSO splits (`create_splits.py`).
- `statistical/`: 21 digital biomarkers extraction (`extract_features.py`) and ML training (`train_statistical.py`).
- `deep_learning/`: PyTorch Dataset (`datasets.py`), 1D-CNN (`cnn1d.py`), GRU (`gru.py`), LSTM (`lstm.py`), training engine (`train_deep.py`).
- `baseline/`: Faithful reproduction of Arora et al. 2023 BiLSTM-SVM (`model.py`, `train_evaluate.py`, `run_baseline.py`).
- `fusion/`: `probability_fusion.py`, `stacking.py`, `feature_fusion.py`.
- `evaluation/`: Clinical metrics engine (`metrics.py`), 300 DPI publication figure generator (`generate_figures.py`), SHAP explainability (`explainability.py`).
- `ai_system/`: Clinical AI Co-Pilot (`clinical_agent.py`), turnkey CLI (`predict_new_patient.py`), web server (`dashboard_server.py`), interactive UI (`dashboard/`).
- `results/`: Predictions (`results/predictions/`), metrics tables (`results/metrics/`), figures (`results/figures/` 10 figures at 300 DPI).
- `reports/`: Research paper draft (`research_paper_draft.md`), compiled paper (`Research_Paper_BTP_GP178.pdf` & `.html`), audit report (`dataset_audit_report.md`).
- `presentation/`: 10-slide academic defense deck (`slides.html` & `slides.pdf`).
- `scripts/`: Paper compiler (`compile_paper.py`), presentation generator (`generate_presentation.py`).
- `tests/`: 46 unit and integration tests passing (`PYTHONPATH=. pytest tests/`).

================================================================================
3. READY-TO-USE COMMANDS
================================================================================
- Instant terminal benchmark summary: `python show_accuracy.py`
- Run entire experimental pipeline: `python experiments/run_all.py`
- Run 46 automated tests: `PYTHONPATH=. pytest tests/`
- Compile publication research paper: `python scripts/compile_paper.py`
- Launch clinician dashboard: `python -m http.server 8000 --directory ai_system/dashboard`
- Open 10-slide defense presentation: `open presentation/slides.html`

================================================================================
YOUR ROLE
================================================================================
Acknowledge that you have full mastery of this project's background, code structure, data, metrics, and methodology. Confirm your readiness to assist with running experiments, answering theoretical or defense questions, drafting manuscript sections, or conducting further ablations. Do not ask me to re-explain the project.
```
