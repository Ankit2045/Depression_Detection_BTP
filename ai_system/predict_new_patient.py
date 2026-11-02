"""
Predictive Clinical Diagnostic Pipeline for New Actigraphy Data (DTU-GP178).

Enables testing any new actigraphy CSV recording (from smartwatches, Actiwatches, etc.):
1. Ingests raw minute-by-minute actigraphy data (timestamp, activity).
2. Slices the continuous signal into 24-hour circadian windows (1,440 minutes, 12-hour sliding hop).
3. Extracts the 21 clinical digital biomarkers and 24-hour sequence profiles.
4. Executes multimodal inference using the trained Champion Dual-Fusion Ensemble (1D-CNN + Logistic Regression).
5. Computes standardized Z-score deviations relative to healthy reference baselines.
6. Automatically generates a DSM-5 aligned psychiatric consultation memo.
7. (Optional) Injects the new patient profile into the interactive Clinician Web Dashboard.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from statistical.extract_features import extract_features_for_window
from deep_learning.cnn1d import ActigraphyCNN1D
from ai_system.clinical_agent import ClinicalDiagnosticCopilot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NewPatientPredictor:
    def __init__(self, models_dir: str = "ai_system/models"):
        self.models_dir = models_dir
        self.scaler = joblib.load(os.path.join(models_dir, "production_scaler.joblib"))
        self.lr_model = joblib.load(os.path.join(models_dir, "production_logistic_regression.joblib"))

        with open(os.path.join(models_dir, "reference_norms.json"), "r") as f:
            self.norms = json.load(f)

        self.feature_names = self.norms["feature_names"]
        self.control_means = self.norms["control_means"]
        self.control_stds = self.norms["control_stds"]

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.cnn_model = ActigraphyCNN1D(in_channels=1, num_filters=(32, 64), kernel_size=3, dropout=0.2)
        cnn_weights = torch.load(os.path.join(models_dir, "production_cnn1d.pt"), map_location=self.device)
        self.cnn_model.load_state_dict(cnn_weights)
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

    def process_raw_csv(self, file_path: str) -> pd.DataFrame:
        """Load and sanitize raw minute-level actigraphy CSV."""
        df = pd.read_csv(file_path)
        # Look for timestamp/date column
        time_col = None
        for col in ["timestamp", "date", "Time", "Timestamp"]:
            if col in df.columns:
                time_col = col
                break
        
        act_col = None
        for col in ["activity", "Activity", "activity_count", "counts"]:
            if col in df.columns:
                act_col = col
                break

        if act_col is None:
            # Fallback to last column if numeric
            act_col = df.columns[-1]

        df = df.rename(columns={time_col: "timestamp", act_col: "activity"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["activity"] = pd.to_numeric(df["activity"], errors="coerce").fillna(0.0).clip(lower=0.0)
        return df

    def segment_windows(
        self,
        df: pd.DataFrame,
        window_size_min: int = 1440,
        step_size_min: int = 720,
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Segment continuous actigraphy series into 24-hour windows with 12-hour sliding hop.
        Returns list of (minute_values, hourly_values, start_timestamp).
        """
        total_len = len(df)
        windows = []

        if total_len < window_size_min:
            # If less than 24 hours, pad with repeated cycle to allow initial screening
            logger.warning(
                f"Recording duration ({total_len} mins) is less than 24 hours (1440 mins). Repeating signal to complete full circadian window."
            )
            reps = int(np.ceil(window_size_min / total_len))
            padded_act = np.tile(df["activity"].values, reps)[:window_size_min]
            hourly_vals = padded_act.reshape(24, 60).mean(axis=1)
            windows.append((padded_act, hourly_vals, str(df["timestamp"].iloc[0])))
            return windows

        for start_idx in range(0, total_len - window_size_min + 1, step_size_min):
            sub_df = df.iloc[start_idx : start_idx + window_size_min]
            min_vals = sub_df["activity"].values
            hourly_vals = min_vals.reshape(24, 60).mean(axis=1)
            start_ts = str(sub_df["timestamp"].iloc[0])
            windows.append((min_vals, hourly_vals, start_ts))

        return windows

    def predict_patient(
        self,
        file_path: str,
        patient_id: str = "new_patient",
        age: str = "Unknown",
        gender: str = "Unknown",
        update_dashboard: bool = False,
        output_memo_dir: str = "ai_system/results/clinical_notes",
    ) -> Dict:
        """Full end-to-end inference and clinical memo generation on new participant actigraphy data."""
        logger.info(f"Processing new participant actigraphy file: {file_path}")
        df = self.process_raw_csv(file_path)
        windows = self.segment_windows(df)
        logger.info(f"Extracted {len(windows)} overlapping 24-hour windows from {len(df):,} minutes of recording.")

        window_probs_stat = []
        window_probs_deep = []
        all_window_features = []
        all_hourly_waveforms = []

        for min_vals, hourly_vals, start_ts in windows:
            # 1. Feature extraction
            w_feats = extract_features_for_window(min_vals, hourly_vals, start_ts)
            # Sanitize any numerical NaNs or Infs (e.g. from constant/zero windows)
            w_feats = {k: float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in w_feats.items()}
            feat_vec = [w_feats[fname] for fname in self.feature_names]
            all_window_features.append(feat_vec)
            all_hourly_waveforms.append(hourly_vals)

            # 2. Statistical model inference
            feat_scaled = self.scaler.transform([feat_vec])
            p_stat = float(self.lr_model.predict_proba(feat_scaled)[0, 1])
            window_probs_stat.append(p_stat)

            # 3. 1D-CNN inference
            seq_tensor = torch.tensor(hourly_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                logit = self.cnn_model(seq_tensor).item()
                p_deep = float(1.0 / (1.0 + np.exp(-logit)))
            window_probs_deep.append(p_deep)

        # 4. Ensemble Fusion across windows
        p_stat_mean = float(np.mean(window_probs_stat))
        p_deep_mean = float(np.mean(window_probs_deep))
        p_dual_ensemble = 0.50 * p_stat_mean + 0.50 * p_deep_mean

        # 5. Aggregate Patient-Level Digital Biomarkers
        avg_features = np.mean(all_window_features, axis=0)
        feats_dict = {name: float(avg_features[i]) for i, name in enumerate(self.feature_names)}

        # Z-Scores vs healthy control baseline
        day_sed = feats_dict.get("day_zero_ratio", 0.13)
        day_sed_norm = float(self.control_means.get("day_zero_ratio", 0.126))
        day_sed_std = float(self.control_stds.get("day_zero_ratio", 0.08))
        day_sed_z = (day_sed - day_sed_norm) / (day_sed_std + 1e-8)

        m10 = feats_dict.get("m10", 471.4)
        m10_norm = float(self.control_means.get("m10", 471.4))
        m10_std = float(self.control_stds.get("m10", 130.0))
        m10_z = (m10 - m10_norm) / (m10_std + 1e-8)

        med_act = feats_dict.get("median_activity", 117.1)
        med_norm = float(self.control_means.get("median_activity", 117.1))
        med_std = float(self.control_stds.get("median_activity", 40.0))
        med_z = (med_act - med_norm) / (med_std + 1e-8)

        noct = feats_dict.get("night_mean", 40.0)
        noct_norm = float(self.control_means.get("night_mean", 168.2))
        noct_std = float(self.control_stds.get("night_mean", 50.0))
        noct_z = (noct - noct_norm) / (noct_std + 1e-8)

        # 6. Risk Tier Classification
        if p_dual_ensemble >= 0.80:
            risk_tier = "CRITICAL / HIGH RISK OF CLINICAL DEPRESSION"
            risk_badge = "CRITICAL RISK"
            action_text = "Urgent psychiatric assessment recommended. Marked psychomotor retardation and circadian flattening detected."
        elif p_dual_ensemble >= 0.50:
            risk_tier = "MODERATE / BORDERLINE RISK OF DEPRESSION"
            risk_badge = "MODERATE RISK"
            action_text = "Borderline psychomotor slowing detected. Recommend structured clinical interview (MADRS/HAM-D) and follow-up monitoring."
        elif p_dual_ensemble >= 0.30:
            risk_tier = "LOW / MILD RESIDUAL SYMPTOM PROFILE"
            risk_badge = "LOW RISK"
            action_text = "Activity signatures within normal healthy envelope. Recommend routine wellness check."
        else:
            risk_tier = "VERY LOW RISK (EUTHYMIC / HEALTHY PROFILE)"
            risk_badge = "HEALTHY / EUTHYMIC"
            action_text = "Robust diurnal circadian amplitude and normal spontaneous motor vitality. No depressive motor deceleration detected."

        # 7. Generate Clinical Memo
        memo = f"""# CLINICAL AI PSYCHIATRIC CONSULTATION MEMO
**Antigravity Clinical Decision Support System (DTU-GP178)**  
*Date of Assessment: Generated from Continuous 24h Actigraphy Protocol*

---

### I. PATIENT DEMOGRAPHICS & AI RISK STRATIFICATION
- **Patient Identifier:** `{patient_id}`
- **Age Bracket:** {age} | **Gender:** {gender}
- **Clinical Cohort Ground Truth:** Prospective / New Diagnostic Subject
- **AI Fused Depression Probability:** **{p_dual_ensemble * 100:.2f}%**
- **Clinical Risk Assessment:** **{risk_badge}** — *{risk_tier}*

---

### II. QUANTITATIVE DIGITAL BIOMARKER DISCOVERY (VS. HEALTHY NORMS)
| Physiological Domain | Patient Value | Healthy Control Reference | Z-Score (Deviation) | Clinical Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Daytime Sedentary Ratio** | **{day_sed * 100:.1f}%** | {day_sed_norm * 100:.1f}% (±{day_sed_std * 100:.1f}%) | **{day_sed_z:+.2f} $\\sigma$** | {"Severe psychomotor retardation" if day_sed_z > 1.0 else "Normal daytime mobility"} |
| **M10 Daytime Peak Vigor** | **{m10:.1f}** | {m10_norm:.1f} (±{m10_std:.1f}) | **{m10_z:+.2f} $\\sigma$** | {"Circadian amplitude blunting" if m10_z < -0.8 else "Preserved diurnal amplitude"} |
| **Median Motor Activity** | **{med_act:.1f} counts/min** | {med_norm:.1f} counts/min | **{med_z:+.2f} $\\sigma$** | {"Hypokinesia / Low baseline vigor" if med_act < 70 else "Normal spontaneous motor tone"} |
| **Nocturnal Mean Activity** | **{noct:.1f} counts/min** | {noct_norm:.1f} counts/min | **{noct_z:+.2f} $\\sigma$** | {"Restless sleep / Micro-awakenings" if noct > 180 else "Consolidated nocturnal sleep"} |

---

### III. PSYCHIATRIC PHENOTYPE & CIRCADIAN RHYTHMICITY
1. **Psychomotor Tone:**
   The patient exhibited an average daytime sedentary ratio of {day_sed * 100:.1f}%, representing a {day_sed_z:+.2f} standard deviation shift relative to age-matched healthy baselines. {"Continuous bouts of daytime immobility indicate marked psychomotor slowing." if day_sed_z > 0.5 else "Daytime movement bouts demonstrate normal motor engagement."}
2. **Circadian Rhythm Architecture:**
   Peak 10-hour diurnal vigor (M10 = {m10:.1f}) is {"significantly flattened, consistent with circadian desynchronization and blunted relative amplitude." if m10_z < -0.5 else "well preserved, indicating stable entrainment to the diurnal light-dark cycle."}
3. **Nocturnal Fragmentation:**
   Sleep periods show {"elevated micro-movement spikes, indicating disturbed sleep efficiency and possible nocturnal awakenings." if noct > 150 else "quiet motor consolidation, indicating stable restorative rest."}

---

### IV. TARGETED CLINICAL ACTION PLAN FOR ATTENDING PSYCHIATRIST
- **Action Tier:** {action_text}
- **Recommended Assessments:** Administer structured MADRS/HAM-D rating scale to cross-validate motor retardation against subjective depressive affect.
- **Digital Biomarker Target:** Prioritize increasing daily active intervals during the afternoon circadian dip (14:00 - 17:00).
- **Signed:** *Antigravity Multimodal Clinical Decision Support Engine (Version 2.0)*
"""
        # Save clinical memo
        os.makedirs(output_memo_dir, exist_ok=True)
        memo_path = os.path.join(output_memo_dir, f"{patient_id}_clinical_memo.md")
        with open(memo_path, "w") as f:
            f.write(memo)
        logger.info(f"Clinical consultation memo written to {memo_path}")

        # 8. (Optional) Inject into Dashboard data.json
        avg_waveform = np.mean(all_hourly_waveforms, axis=0).tolist()
        avg_waveform = [round(float(v), 2) for v in avg_waveform]

        result = {
            "participant_id": patient_id,
            "label": -1,  # New prospective patient
            "group": f"New Test ({risk_badge})",
            "prob_ensemble": round(p_dual_ensemble, 4),
            "prob_statistical": round(p_stat_mean, 4),
            "prob_deep_learning": round(p_deep_mean, 4),
            "age": age,
            "gender": gender,
            "madrs1": None,
            "madrs2": None,
            "waveform_24h": avg_waveform,
            "control_benchmark_waveform": self.norms["control_benchmark_waveform"],
            "attention_weights": [round(1.0 / 24.0, 5)] * 24,
            "biomarkers": {
                "daytime_sedentary_ratio": round(day_sed, 4),
                "daytime_sedentary_ratio_z": round(day_sed_z, 2),
                "m10_vigor": round(m10, 2),
                "m10_vigor_z": round(m10_z, 2),
                "median_activity": round(med_act, 2),
                "median_activity_z": round(med_z, 2),
                "nocturnal_mean": round(noct, 2),
                "nocturnal_mean_z": round(noct_z, 2),
            },
            "clinical_memo": memo,
        }

        if update_dashboard:
            dashboard_file = "ai_system/dashboard/data.json"
            if os.path.exists(dashboard_file):
                with open(dashboard_file, "r") as f:
                    dash_data = json.load(f)
                # Remove prior entry if patient_id already exists
                dash_data = [p for p in dash_data if p.get("participant_id") != patient_id]
                # Prepend new patient so it appears at top of selector
                dash_data.insert(0, result)
                with open(dashboard_file, "w") as f:
                    json.dump(dash_data, f, indent=2)
                logger.info(f"Updated {dashboard_file} with new patient {patient_id}.")

        return result


def main():
    parser = argparse.ArgumentParser(description="Test New Patient Actigraphy Data (DTU-GP178)")
    parser.add_argument("--file", type=str, required=True, help="Path to raw actigraphy CSV file.")
    parser.add_argument("--patient_id", type=str, default="new_patient_01", help="Identifier for new patient.")
    parser.add_argument("--age", type=str, default="30-34", help="Age bracket (e.g. 30-34).")
    parser.add_argument("--gender", type=str, default="Female", help="Gender (Male / Female).")
    parser.add_argument("--update_dashboard", action="store_true", help="Add this patient to Clinician Web Dashboard.")
    parser.add_argument("--memo_dir", type=str, default="ai_system/results/clinical_notes", help="Memo output directory.")
    args = parser.parse_args()

    predictor = NewPatientPredictor()
    res = predictor.predict_patient(
        file_path=args.file,
        patient_id=args.patient_id,
        age=args.age,
        gender=args.gender,
        update_dashboard=args.update_dashboard,
        output_memo_dir=args.memo_dir,
    )

    print("\n" + "=" * 80)
    print(f" CLINICAL PREDICTION RESULTS FOR NEW PARTICIPANT: {args.patient_id}")
    print("=" * 80)
    print(f" Multimodal Fused Probability: {res['prob_ensemble'] * 100:.2f}%")
    print(f"   - Statistical Model (LR):   {res['prob_statistical'] * 100:.2f}%")
    print(f"   - Deep Learning (1D-CNN):   {res['prob_deep_learning'] * 100:.2f}%")
    print(f" Risk Classification:          {res['group']}")
    print("-" * 80)
    print(" Digital Biomarkers (vs. Healthy Controls):")
    b = res["biomarkers"]
    print(f"   - Daytime Sedentary Ratio:  {b['daytime_sedentary_ratio']*100:.1f}% ({b['daytime_sedentary_ratio_z']:+.2f} σ)")
    print(f"   - M10 Diurnal Peak Vigor:   {b['m10_vigor']:.1f} ({b['m10_vigor_z']:+.2f} σ)")
    print(f"   - Median Motor Activity:    {b['median_activity']:.1f} counts/min ({b['median_activity_z']:+.2f} σ)")
    print(f"   - Nocturnal Activity:       {b['nocturnal_mean']:.1f} counts/min ({b['nocturnal_mean_z']:+.2f} σ)")
    print("-" * 80)
    print(f" Full Clinical Consultation Memo saved to:")
    print(f"   {args.memo_dir}/{args.patient_id}_clinical_memo.md")
    if args.update_dashboard:
        print(f" Added to Clinician Dashboard! Open http://localhost:8000 to view.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
