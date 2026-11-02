# Clinical Artificial Intelligence & Temporal Transformer Module (DTU-GP178)

**Project:** Multimodal Feature Fusion of Statistical and Deep Learning Representations for Actigraphy-Based Depression Detection  
**Institution:** Department of Computer Science & Engineering, Delhi Technological University (DTU)  
**Supervisor:** Dr. Anshika Arora, Assistant Professor  
**Code Isolation:** Standalone, modular AI subsystem isolated from the core research pipeline.

---

## 1. Overview & Clinical Motivation

While classical machine learning and standard convolutional networks excel at statistical discriminability, modern psychiatric decision support requires **model interpretability**, **temporal saliency extraction**, and **clinically actionable reporting**.

This self-contained `ai_system/` module introduces three next-generation AI components:
1. **Multi-Head Temporal Self-Attention Transformer (`transformer_attention.py`)**: A modern deep learning architecture tailored for 24-hour periodic circadian actigraphy sequences, capturing long-range diurnal-nocturnal dependencies and exposing interpretable attention matrices.
2. **Generative AI Clinical Diagnostic Co-Pilot (`clinical_agent.py`)**: An automated clinical memo generator synthesizing quantitative digital biomarkers (daytime sedentary ratio, M10 peak vigor, median motor activity, nocturnal fragmentation) into DSM-5 aligned psychiatric consultation notes.
3. **Interactive Medical Clinician Web Dashboard (`dashboard/`)**: A zero-dependency HTML5/CSS3/JavaScript clinical application enabling psychiatrists to inspect 24-hour motor waveforms, compare patient profiles against healthy benchmarks, visualize attention heatmaps, and review automated diagnostic memos during viva demonstrations.

---

## 2. Directory Architecture

```
ai_system/
├── transformer_attention.py   # Multi-Head Temporal Self-Attention Transformer & Circadian Saliency
├── clinical_agent.py          # Generative AI Clinical Decision Support Engine (DSM-5 memos)
├── run_ai_system.py           # Standalone execution pipeline compiling all deliverables
├── dashboard/                 # Zero-dependency Clinician Web Interface
│   ├── index.html             # Single-page responsive clinical application layout
│   ├── app.js                 # Dynamic rendering, interactive charts, and participant switching
│   └── data.json              # Compiled dataset containing 55 participants, waveforms & notes
├── results/
│   ├── attention_weights.csv  # Hour-by-hour self-attention saliency weights
│   ├── figures/
│   │   └── figure10_circadian_attention_heatmap.png  # Attention heatmap across 24 circadian hours
│   └── clinical_notes/        # 55 Individual DSM-5 aligned markdown consultation notes
│       ├── condition_1_clinical_memo.md
│       └── ...
└── README.md                  # This documentation guide
```

---

## 3. Component Details

### A. Temporal Self-Attention Transformer (`transformer_attention.py`)
- **Input Representation:** 24-hour diurnal actigraphy profiles with sinusoidal and learned periodic circadian positional embeddings.
- **Attention Mechanism:** 4-head scaled dot-product temporal self-attention ($d_k = 16$, $d_{\text{model}} = 64$) extracting bidirectional relationships between nocturnal rest intervals and diurnal active bouts.
- **Attentive Temporal Pooling Head:** Instead of unweighted global averaging, dynamic temporal attention pooling assigns higher weights to critical transitional periods (awakening at 08:00–10:00 and afternoon circadian dip at 14:00–17:00).
- **Circadian Saliency Map:** Outputs Figure 10 (`figure10_circadian_attention_heatmap.png`), showing where the network attends when classifying depression vs. control.

### B. Generative AI Clinical Co-Pilot (`clinical_agent.py`)
- Computes individualized digital biomarker deviations in standard deviations ($\sigma$) relative to healthy normative baselines:
  * **Daytime Sedentary Ratio ($Z_{\text{sed}}$)**: Quantifies daytime psychomotor retardation.
  * **M10 Daytime Peak Vigor ($Z_{M10}$)**: Quantifies diurnal amplitude flattening.
  * **Median Motor Activity ($Z_{\text{med}}$)**: Quantifies baseline spontaneous hypokinesia.
  * **Nocturnal Activity ($Z_{\text{noct}}$)**: Detects sleep fragmentation and restlessness.
- Outputs structured clinical consultation memos adhering to DSM-5 Section II criteria for Major Depressive Episode.

### C. Clinician Web Dashboard (`dashboard/index.html`)
- **Live Participant Selector:** Quickly switch between all 55 participants (Condition and Control cohorts) with instant risk stratification badges (Critical, Moderate, Mild, Healthy).
- **24-Hour Waveform Visualizer:** Displays the patient's continuous 24-hour motor profile overlaid against the healthy normative benchmark envelope.
- **Circadian Attention Bar Chart:** Interactively visualizes which hours of the day drove the AI classification decision.
- **Radar & Metric Cards:** Real-time Z-score deviation badges and clinical thresholds.
- **Diagnostic Note Viewer:** Renders the complete, customized psychiatric memo for direct clinical consultation.

---

## 4. How to Run the AI System

### Step 1: Run the Standalone Pipeline
Generate all attention weights, figures, 55 clinical memos, and precompiled dashboard data with a single command:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. .venv/bin/python ai_system/run_ai_system.py
```

### Step 2: Launch the Clinician Web Dashboard
Start a local HTTP server inside the dashboard directory:

```bash
python -m http.server 8000 --directory ai_system/dashboard
```

Then open your web browser and navigate to:
```
http://localhost:8000
```

### Step 3: Testing Any New Patient / External Actigraphy CSV
To test an entirely new, prospective subject who is not part of the original 55 participants:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. .venv/bin/python ai_system/predict_new_patient.py \
  --file path/to/actigraphy_recording.csv \
  --patient_id "Patient_101" \
  --age "35-39" \
  --gender "Female" \
  --update_dashboard
```

**What this automatically does:**
1. Ingests the raw minute-level recording and slices it into 24-hour circadian windows (1,440 minutes with 12-hour hops).
2. Extracts all 21 statistical biomarkers and 24-hour sequential profiles.
3. Passes the data through the trained **Dual-Fusion Ensemble (1D-CNN + Logistic Regression)**.
4. Computes normative Z-scores relative to healthy control baselines.
5. Generates a new DSM-5 psychiatric consultation memo in `ai_system/results/clinical_notes/Patient_101_clinical_memo.md`.
6. With `--update_dashboard`, injects the new patient at the top of `data.json` so the patient immediately appears in the web dashboard!

---

## 5. Architectural Isolation

This module is strictly isolated from the core statistical models (`statistical/`), convolutional networks (`deep_learning/`), and decision fusion ensembles (`fusion/`). Running or modifying `ai_system/` does not alter or re-run the 55-fold Leave-One-Subject-Out (LOSO) cross-validation pipeline of the primary research study.
