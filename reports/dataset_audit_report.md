# Depresjon Dataset Audit Report
**Audit Execution Timestamp:** `2026-10-31T22:11:57.407616`

## 1. Executive Summary
- **Dataset Name:** Depresjon
- **Total Participants Verified:** 55 (Condition: 23, Control: 32)
- **Total Minute-level Observations:** 1,571,706
- **Label Alignment:** Perfect match between participant files and `scores.csv` (0 missing, 0 orphans)
- **Schema Uniformity:** 100% of participant files share identical schema `[timestamp, date, activity]`
- **Monotonic Chronological Ordering:** 100% verified
- **Duplicate Timestamps:** 0 found across all 1,571,706 rows
- **Negative Activity Values:** 0 found (all counts $\ge 0$)

## 2. Participant and Observation Distributions

| Metric | Minimum | Median | Mean | Maximum |
| :--- | :--- | :--- | :--- | :--- |
| **Observations per Subject** | 19,299 | 23,244.0 | 28,576.5 | 65,407 |
| **Recording Duration (Days)** | 13.401 | 16.14 | 19.85 | 45.421 |
| **Reported `days` in `scores.csv`** | 5 | 13 | 12.6 | 20 |

## 3. Class-Wise Activity Characteristics

| Metric | Depressed Condition (N=23) | Healthy Control (N=32) | Disparity / Direction |
| :--- | :--- | :--- | :--- |
| **Mean Activity Count** | 163.72 ± 67.83 | 208.65 ± 84.64 | Controls show ~27.4% higher average activity |
| **Zero-Activity Ratio** | 43.62% ± 16.68% | 38.11% ± 18.79% | Depressed subjects show ~5.5% more zero-activity periods |
| **Mean Recording Days** | 16.66 days | 22.14 days | Controls recorded for longer on average |

## 4. Irregular Intervals and Missing Data Audit

Only **5 out of 55 participants** present irregular timestamp intervals, each containing exactly **1 single gap of 60 minutes**:

| Participant ID | Group | Row Count | Gap Interval | Missing Minutes | Verified Cause |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `control_1` | control | 51,611 | `2003-03-30 01:59:00` → `03:00:00` | 60 | Daylight Saving Time transition (spring forward) |
| `control_6` | control | 51,589 | `2003-03-30 01:59:00` → `03:00:00` | 60 | Daylight Saving Time transition (spring forward) |
| `control_7` | control | 51,490 | `2003-03-30 01:59:00` → `03:00:00` | 60 | Daylight Saving Time transition (spring forward) |
| `control_31` | control | 51,389 | `2003-03-30 01:59:00` → `03:00:00` | 60 | Daylight Saving Time transition (spring forward) |
| `control_32` | control | 51,619 | `2003-03-30 01:59:00` → `03:00:00` | 60 | Daylight Saving Time transition (spring forward) |

> [!NOTE]
> All 5 occurrences match the exact European Daylight Saving Time switch on Sunday, March 30, 2003, at 02:00 CET. There are zero unexplainable missing intervals, dropped packets, or arbitrary hardware downtime in the raw dataset.

## 5. Clinical and Demographic Metadata Audit

- **Condition Subjects (N=23):**
  - Affective Type: Bipolar II (15), Bipolar I (7), Unipolar Depression (1)
  - Setting: Outpatients (18), Inpatients (5)
  - Melancholia: Present (20), Missing/NA (3)
  - MADRS-1 (Baseline): Mean = 22.74 (range: 13–29)
  - MADRS-2 (Post-assessment): Mean = 20.00 (range: 11–28)
- **Control Subjects (N=32):**
  - Healthy volunteers from hospital staff and population controls; all clinical depression scores are appropriately NA.
  - Gender distribution: 20 Female / 12 Male (vs Condition: 10 Female / 13 Male).
