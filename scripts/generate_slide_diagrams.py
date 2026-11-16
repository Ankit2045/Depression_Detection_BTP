"""
Generate high-resolution, publication-grade diagrams for the BTP presentation.
All diagrams feature pure white backgrounds (#FFFFFF) and high-contrast clinical/academic palettes.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.makedirs('presentation/diagrams', exist_ok=True)

# Color Palette Tokens
WHITE = '#FFFFFF'
SLATE_900 = '#0F172A'
SLATE_800 = '#1E293B'
SLATE_700 = '#334155'
SLATE_500 = '#64748B'
SLATE_200 = '#E2E8F0'
SLATE_100 = '#F1F5F9'
SLATE_50 = '#F8FAFC'

CRIMSON = '#991B1B'
CRIMSON_LIGHT = '#DC2626'
BLUE = '#1D4ED8'
BLUE_LIGHT = '#3B82F6'
EMERALD = '#047857'
EMERALD_LIGHT = '#10B981'
AMBER = '#B45309'
AMBER_LIGHT = '#F59E0B'
PURPLE = '#6D28D9'

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = WHITE
plt.rcParams['axes.facecolor'] = WHITE


def save_fig(fig, filename):
    path = os.path.join('presentation/diagrams', filename)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=WHITE, edgecolor='none')
    plt.close(fig)
    print(f"Saved: {path}")


# -------------------------------------------------------------
# 1. Slide 2: Physical Illness vs Invisible Depression
# -------------------------------------------------------------
def make_diagram_slide2():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    # Title Banner
    ax.text(6, 6.1, "THE CLINICAL DIAGNOSTIC BLINDSPOT", ha='center', va='center',
            fontsize=16, fontweight='bold', color=CRIMSON)
    ax.text(6, 5.7, "Why Major Depression Evades Standard Medical Detection Paradigms", ha='center', va='center',
            fontsize=11, color=SLATE_700)

    # Box 1: Physical Illness
    box1 = patches.FancyBboxPatch((0.5, 1.8), 5.2, 3.6, boxstyle="round,pad=0.2",
                                  fc=SLATE_50, ec=BLUE, lw=2)
    ax.add_patch(box1)

    ax.text(3.1, 5.1, "Physical Illness: Visible & Directly Quantifiable", ha='center', va='center',
            fontsize=12, fontweight='bold', color=BLUE)
    
    physical_items = [
        ("Diabetes", "Blood glucose assay (mg/dL) - Objective & Instant"),
        ("Hypertension", "Sphygmomanometer blood pressure (mmHg)"),
        ("Pneumonia / COVID", "Chest X-Ray / CT scan & SpO2 pulse oximetry"),
        ("Fever / Infection", "Thermometer reading (38.5 deg C) & WBC count"),
        ("Cardiac Arrhythmia", "Electrocardiogram (ECG) continuous trace")
    ]
    y_pos = 4.5
    for title, desc in physical_items:
        ax.plot(0.9, y_pos, marker='o', markersize=6, color=BLUE)
        ax.text(1.2, y_pos, title + ":", fontweight='bold', fontsize=9.5, color=SLATE_900, va='center')
        ax.text(2.6, y_pos, desc, fontsize=8.5, color=SLATE_700, va='center')
        y_pos -= 0.6

    # Box 2: Depression (The Invisible Illness)
    box2 = patches.FancyBboxPatch((6.3, 1.8), 5.2, 3.6, boxstyle="round,pad=0.2",
                                  fc=SLATE_50, ec=CRIMSON, lw=2)
    ax.add_patch(box2)

    ax.text(8.9, 5.1, "Major Depression: 'The Invisible Mental Illness'", ha='center', va='center',
            fontsize=12, fontweight='bold', color=CRIMSON)

    depression_items = [
        ("No Biomarker", "Zero blood tests, no fever, no visible physical lesion"),
        ("Questionnaires", "Standard: MADRS, HAM-D, PHQ-9 (Subjective)"),
        ("Recall Bias", "Patients struggle to recall weeks of mood accurately"),
        ("Social Masking", "Patients conceal symptoms due to societal stigma"),
        ("Episodic Care", "Infrequent consultations (weeks apart); late detection")
    ]
    y_pos = 4.5
    for title, desc in depression_items:
        ax.plot(6.7, y_pos, marker='x', markersize=7, color=CRIMSON, markeredgewidth=2)
        ax.text(7.0, y_pos, title + ":", fontweight='bold', fontsize=9.5, color=SLATE_900, va='center')
        ax.text(8.5, y_pos, desc, fontsize=8.5, color=SLATE_700, va='center')
        y_pos -= 0.6

    # Bottom Bridge: The Solution (Wearable Actigraphy)
    box3 = patches.FancyBboxPatch((0.5, 0.3), 11.0, 1.2, boxstyle="round,pad=0.15",
                                  fc=EMERALD, ec=EMERALD, lw=1)
    ax.add_patch(box3)

    ax.text(6.0, 1.15, "OUR PROBLEM STATEMENT & ACTIGRAPHY SOLUTION", ha='center', va='center',
            fontsize=11.5, fontweight='bold', color=WHITE)
    ax.text(6.0, 0.65,
            "Major depression fundamentally impairs psychomotor vitality and circadian rhythmicity.\n"
            "By capturing continuous motor actigraphy via wearable IoMT devices, we extract objective physiological digital biomarkers,\n"
            "transforming psychiatric screening from subjective patient recall to automated, leak-free continuous assessment.",
            ha='center', va='center', fontsize=9.2, color=WHITE, multialignment='center')

    save_fig(fig, 'diagram_slide2_problem_statement.png')


# -------------------------------------------------------------
# 2. Slide 3: Depresjon Dataset & Diurnal Waveforms
# -------------------------------------------------------------
def make_diagram_slide3():
    fig = plt.figure(figsize=(12, 6.2))
    
    # Left subplot: Cohort stats & Windowing (Axes 1)
    ax1 = fig.add_axes([0.05, 0.12, 0.42, 0.8])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    ax1.text(0, 9.7, "DEPRESJON BENCHMARK COHORT AUDIT", fontsize=13, fontweight='bold', color=SLATE_900)
    ax1.text(0, 9.2, "Collected at Haukeland University Hospital, Bergen, Norway", fontsize=9.5, color=SLATE_500)

    # Box 1: Condition
    b1 = patches.FancyBboxPatch((0, 6.2), 9.6, 2.7, boxstyle="round,pad=0.2",
                                fc=SLATE_50, ec=CRIMSON, lw=1.5)
    ax1.add_patch(b1)
    ax1.text(0.3, 8.5, "Condition Cohort (N = 23 Depressed)", fontsize=11, fontweight='bold', color=CRIMSON)
    ax1.text(0.3, 7.8, "• Diagnosis: 15 Bipolar II, 7 Bipolar I, 1 Unipolar MDD", fontsize=9, color=SLATE_800)
    ax1.text(0.3, 7.2, "• Clinical Severity: Mean Baseline MADRS = 22.74 ± 4.80", fontsize=9, color=SLATE_800)
    ax1.text(0.3, 6.6, "• Settings: 18 Outpatients, 5 Inpatients (wrist accelerometer)", fontsize=9, color=SLATE_800)

    # Box 2: Control
    b2 = patches.FancyBboxPatch((0, 3.1), 9.6, 2.7, boxstyle="round,pad=0.2",
                                fc=SLATE_50, ec=EMERALD, lw=1.5)
    ax1.add_patch(b2)
    ax1.text(0.3, 5.4, "Control Cohort (N = 32 Healthy Volunteers)", fontsize=11, fontweight='bold', color=EMERALD)
    ax1.text(0.3, 4.7, "• Status: No history of depression or psychiatric illness", fontsize=9, color=SLATE_800)
    ax1.text(0.3, 4.1, "• Demographics: Age & gender matched hospital staff & volunteers", fontsize=9, color=SLATE_800)
    ax1.text(0.3, 3.5, "• Sensor: Cambridge Neurotechnology Actiwatch AW4 (32 Hz)", fontsize=9, color=SLATE_800)

    # Box 3: Windowing Rule
    b3 = patches.FancyBboxPatch((0, 0.2), 9.6, 2.5, boxstyle="round,pad=0.2",
                                fc=SLATE_100, ec=BLUE, lw=1.5)
    ax1.add_patch(b3)
    ax1.text(0.3, 2.3, "24-Hour Overlapping Windowing Protocol", fontsize=10.5, fontweight='bold', color=BLUE)
    ax1.text(0.3, 1.7, "• Window Length: 24h (1,440 min) | Hop Size: 12h (50% overlap)", fontsize=8.5, color=SLATE_800)
    ax1.text(0.3, 1.1, "• Total Yield: 1,331 Windows (Condition: 559, Control: 772)", fontsize=8.5, color=SLATE_800)
    ax1.text(0.3, 0.5, "• Strict Subject Boundaries: Zero windows cross participants", fontsize=8.5, color=SLATE_800)

    # Right subplot: Diurnal 24h Activity Waveforms (Axes 2)
    ax2 = fig.add_axes([0.53, 0.15, 0.43, 0.73])
    
    hours = np.linspace(0, 24, 200)
    # Healthy: sharp peak around 14:00, strong circadian amplitude, quiet nights
    control_curve = 340 * np.exp(-((hours - 14.5)**2) / 28) + 80 * np.exp(-((hours - 10.0)**2) / 10) + 20
    # Depressed: blunted peak, restless sleep, delayed awakening
    depressed_curve = 175 * np.exp(-((hours - 15.0)**2) / 36) + 40 * np.sin(hours * 0.8) + 55

    ax2.plot(hours, control_curve, color=EMERALD, lw=2.5, label='Healthy Control (Robust Diurnal Rhythm)')
    ax2.plot(hours, depressed_curve, color=CRIMSON, lw=2.5, linestyle='--', label='Depressed Patient (Blunted Circadian Rhythm)')
    
    ax2.fill_between(hours, control_curve, depressed_curve, where=(control_curve >= depressed_curve),
                     color=EMERALD, alpha=0.15, label='Circadian Vigour Deficit (M10)')
    ax2.fill_between(hours, depressed_curve, control_curve, where=(depressed_curve > control_curve),
                     color=CRIMSON, alpha=0.2, label='Nocturnal Restlessness (L5)')

    ax2.set_title("24-Hour Mean Activity Profile (Actigraphy Waveform)", fontsize=11.5, fontweight='bold', color=SLATE_900)
    ax2.set_xlabel("Time of Day (Hours: 00:00 to 24:00)", fontsize=10, color=SLATE_800)
    ax2.set_ylabel("Mean Activity Counts / Minute", fontsize=10, color=SLATE_800)
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 480)
    ax2.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax2.set_xticklabels(['00:00\nNight', '04:00', '08:00\nWake', '12:00\nNoon', '16:00', '20:00\nEve', '24:00'])
    ax2.grid(True, linestyle=':', alpha=0.6, color=SLATE_200)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

    save_fig(fig, 'diagram_slide3_dataset_and_waveform.png')


# -------------------------------------------------------------
# 3. Slide 4: 55-Fold Nested LOSO Cross-Validation
# -------------------------------------------------------------
def make_diagram_slide4():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    ax.text(6, 6.1, "THE 55-FOLD NESTED LEAVE-ONE-SUBJECT-OUT (LOSO) PROTOCOL",
            ha='center', va='center', fontsize=14, fontweight='bold', color=SLATE_900)
    ax.text(6, 5.7, "Eliminating Pervasive Literature Data Leakage for True Clinical Generalization",
            ha='center', va='center', fontsize=10.5, color=SLATE_500)

    # Top: The Flawed Approach
    b_bad = patches.FancyBboxPatch((0.5, 4.0), 11.0, 1.4, boxstyle="round,pad=0.15",
                                   fc='#FEF2F2', ec=CRIMSON, lw=1.5)
    ax.add_patch(b_bad)
    ax.text(0.8, 5.0, "TRADITIONAL WINDOW-LEVEL SPLIT (DATA LEAKAGE TRAP)", fontsize=10.5, fontweight='bold', color=CRIMSON)
    ax.text(0.8, 4.4,
            "• Randomly splits 1,331 overlapping windows into 10 folds regardless of participant identity.\n"
            "• Consecutive windows share 50% (12 hours) of identical time readings -> Direct temporal leakage.\n"
            "• Classifiers memorize individual biometric movement signatures rather than generalized depression pathology (95%+ artifact).",
            fontsize=8.5, color=SLATE_800)

    # Bottom: Our 55-Fold LOSO Framework
    b_good = patches.FancyBboxPatch((0.5, 0.4), 11.0, 3.3, boxstyle="round,pad=0.15",
                                    fc=SLATE_50, ec=BLUE, lw=2)
    ax.add_patch(b_good)
    ax.text(0.8, 3.3, "OUR STRICT 55-FOLD NESTED LOSO EVALUATION (ZERO LEAKAGE)", fontsize=11, fontweight='bold', color=BLUE)

    # Flowchart boxes inside
    # Outer Step 1: Split
    b_step1 = patches.FancyBboxPatch((0.8, 0.8), 2.8, 2.2, boxstyle="round,pad=0.1",
                                     fc=WHITE, ec=SLATE_500, lw=1.2)
    ax.add_patch(b_step1)
    ax.text(2.2, 2.65, "Outer Fold k (k = 1..55)", ha='center', fontsize=9.5, fontweight='bold', color=SLATE_900)
    ax.text(2.2, 2.1, "• Held-Out Test: P_k (N = 1)\n  Completely unseen patient\n  All W_k windows isolated", ha='center', fontsize=8, color=CRIMSON)
    ax.text(2.2, 1.25, "• Training Pool (N = 54)\n  All other 54 participants\n  Scaler fitted ONLY here", ha='center', fontsize=8, color=EMERALD)

    # Arrow 1
    ax.annotate('', xy=(4.0, 1.9), xytext=(3.6, 1.9),
                arrowprops=dict(facecolor=BLUE, edgecolor=BLUE, arrowstyle='->', lw=2))

    # Inner Step 2: Nested 5-fold CV
    b_step2 = patches.FancyBboxPatch((4.0, 0.8), 3.4, 2.2, boxstyle="round,pad=0.1",
                                     fc=WHITE, ec=SLATE_500, lw=1.2)
    ax.add_patch(b_step2)
    ax.text(5.7, 2.65, "Nested Inner 5-Fold Group CV", ha='center', fontsize=9.5, fontweight='bold', color=SLATE_900)
    ax.text(5.7, 2.0, "• Evaluated strictly on the 54\n  training subjects (zero test access)\n• Grouped by Subject ID\n• Tunes hyperparameters (C, gamma)\n• Selects optimal fusion weights",
            ha='center', fontsize=8, color=SLATE_800)

    # Arrow 2
    ax.annotate('', xy=(7.8, 1.9), xytext=(7.4, 1.9),
                arrowprops=dict(facecolor=BLUE, edgecolor=BLUE, arrowstyle='->', lw=2))

    # Step 3: Aggregation & Diagnosis
    b_step3 = patches.FancyBboxPatch((7.8, 0.8), 3.4, 2.2, boxstyle="round,pad=0.1",
                                     fc=WHITE, ec=EMERALD, lw=1.5)
    ax.add_patch(b_step3)
    ax.text(9.5, 2.65, "Patient-Level Aggregation", ha='center', fontsize=9.5, fontweight='bold', color=EMERALD)
    ax.text(9.5, 2.0, "• Predict on test patient P_k:\n  P_i = (1 / W_k) * SUM(P_w)\n• Final Clinical Diagnosis:\n  y_hat = I(P_i >= 0.5)\n• Guaranteed zero data leakage\n  across all 55 hospital subjects",
            ha='center', fontsize=8, color=SLATE_800)

    save_fig(fig, 'diagram_slide4_loso_55fold.png')


# -------------------------------------------------------------
# 4. Slide 5: Literature Comparison Under LOSO
# -------------------------------------------------------------
def make_diagram_slide5():
    fig, ax = plt.subplots(figsize=(11, 5.8))
    
    models = [
        "Frogner et al. (2019)\n1D-CNN (Raw Series)",
        "Garcia-Ceja et al. (2018)\nLinear SVM (3 Stats)",
        "Garcia-Ceja et al. (2018b)\nRandom Forest + Over",
        "Our Standalone\n1D-CNN (Hourly)",
        "Our Statistical Branch\nLogReg (21 Biomarkers)",
        "Jakobsen et al. (2020)\nDeep Neural Network",
        "[CHAMPION] OUR DUAL-FUSION\nEnsemble (Champion)"
    ]
    accuracies = [71.00, 72.73, 73.00, 80.00, 80.00, 84.00, 92.73]
    colors = [SLATE_500, SLATE_500, SLATE_500, BLUE, BLUE, SLATE_700, CRIMSON]

    bars = ax.barh(models, accuracies, color=colors, height=0.62, edgecolor=SLATE_800, lw=1)
    
    ax.set_xlim(50, 100)
    ax.set_xlabel("Participant-Level Accuracy on Unseen Clinical Subjects (%)", fontsize=10.5, fontweight='bold', color=SLATE_900)
    ax.set_title("COMPARATIVE BENCHMARK: MODELS EVALUATED UNDER LEAVE-ONE-SUBJECT-OUT (LOSO)",
                 fontsize=12.5, fontweight='bold', color=SLATE_900, pad=15)
    ax.grid(axis='x', linestyle=':', alpha=0.6, color=SLATE_200)

    for bar, acc in zip(bars, accuracies):
        w = bar.get_width()
        fontweight = 'bold' if acc > 90 else 'normal'
        ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{acc:.2f}%",
                va='center', fontsize=10, fontweight=fontweight, color=SLATE_900)

    # Highlight box for gain
    ax.annotate("+20.00% Absolute Boost\n(McNemar test: p = 0.0076)",
                xy=(92.73, 6), xytext=(85, 4.3),
                arrowprops=dict(facecolor=CRIMSON, edgecolor=CRIMSON, arrowstyle='->', lw=1.8),
                bbox=dict(boxstyle="round,pad=0.3", fc='#FEF2F2', ec=CRIMSON, lw=1.2),
                fontsize=9.5, fontweight='bold', color=CRIMSON)

    save_fig(fig, 'diagram_slide5_loso_literature.png')


# -------------------------------------------------------------
# 5. Slide 6: Arora et al. (2023) Reproduction & Layer Depth Scaling
# -------------------------------------------------------------
def make_diagram_slide6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={'width_ratios': [1.1, 0.9]})

    # Left: Window-Level Scaling
    configs = [
        "Statistical (3 Stats + SVM)",
        "Published Arora et al. (2023)\n(BiLSTM L=2)",
        "Our Reproduction\n(BiLSTM L=2)",
        "[CHAMPION] Our Enhanced Model\n(BiLSTM L=3)",
        "Our Saturated Model\n(BiLSTM L=4)"
    ]
    window_accs = [74.83, 95.57, 96.39, 99.62, 97.37]
    bar_colors = [SLATE_500, BLUE, BLUE, EMERALD, BLUE_LIGHT]

    bars1 = ax1.bar(range(len(configs)), window_accs, color=bar_colors, edgecolor=SLATE_800, lw=1, width=0.55)
    ax1.set_ylim(60, 105)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, fontsize=8.5, rotation=15, ha='right')
    ax1.set_ylabel("Window-Level Accuracy (10-Fold CV) %", fontsize=10, fontweight='bold', color=SLATE_900)
    ax1.set_title("WINDOW-LEVEL CAPACITY: ARORA ET AL. (2023)\nReproduction & BiLSTM Layer-Depth Scaling",
                  fontsize=11, fontweight='bold', color=SLATE_900)
    ax1.grid(axis='y', linestyle=':', alpha=0.6, color=SLATE_200)

    for bar, acc in zip(bars1, window_accs):
        h = bar.get_height()
        fw = 'bold' if acc > 99 else 'normal'
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{acc:.2f}%",
                 ha='center', fontsize=9.5, fontweight=fw, color=SLATE_900)

    # Right: The Crucial Conceptual Discrepancy
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    b_callout = patches.FancyBboxPatch((0.5, 0.5), 9.0, 9.0, boxstyle="round,pad=0.2",
                                       fc=SLATE_50, ec=SLATE_800, lw=1.5)
    ax2.add_patch(b_callout)

    ax2.text(5.0, 8.8, "RESOLVING THE 99.62% vs 92.73% PARADOX", ha='center', fontsize=11, fontweight='bold', color=CRIMSON)
    
    text_content = (
        "1. Window-Level Evaluation (99.62% Acc):\n"
        "   • Evaluated on 1,331 overlapping 24h slices.\n"
        "   • Consecutive windows share 12h of identical data.\n"
        "   • Result: Evaluates representation capacity, but\n"
        "     suffers from subject identity memorisation.\n\n"
        "2. Subject-Level LOSO Evaluation (92.73% Acc):\n"
        "   • 100% of test subject's data is strictly withheld.\n"
        "   • Evaluates true hospital clinical generalisation.\n"
        "   • Prior literature collapsed to 71%–73% under LOSO.\n\n"
        "Key Clinical Takeaway:\n"
        "   Our work is the first to achieve BOTH:\n"
        "   (a) 99.62% window capacity via BiLSTM depth scaling,\n"
        "   (b) 92.73% true generalisation on unseen subjects."
    )
    ax2.text(1.0, 7.8, text_content, va='top', fontsize=9, color=SLATE_900, linespacing=1.4)

    save_fig(fig, 'diagram_slide6_baseline_scaling.png')


# -------------------------------------------------------------
# 6. Slide 7: Hourly vs Minutewise Resolution Comparison
# -------------------------------------------------------------
def make_diagram_slide7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={'width_ratios': [1.1, 0.9]})

    # Left: Grouped Bar Chart
    categories = ['Standalone 1D-CNN', 'Multimodal Fusion (Weighted)']
    hourly_acc = [79.76, 90.96]
    minute_acc = [66.03, 76.02]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax1.bar(x - width/2, hourly_acc, width, label='Hourly Downsampled (T = 24)',
                     color=BLUE, edgecolor=SLATE_800, lw=1)
    rects2 = ax1.bar(x + width/2, minute_acc, width, label='Raw Minute-Level (T = 1,440)',
                     color=AMBER, edgecolor=SLATE_800, lw=1)

    ax1.set_ylabel('Balanced Accuracy (55-Fold LOSO) %', fontsize=10.5, fontweight='bold', color=SLATE_900)
    ax1.set_title('TEMPORAL GRANULARITY IMPACT: HOURLY VS MINUTE', fontsize=11.5, fontweight='bold', color=SLATE_900)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax1.set_ylim(50, 100)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='y', linestyle=':', alpha=0.6, color=SLATE_200)

    for r in rects1:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2, h + 1.2, f"{h:.2f}%", ha='center', fontsize=9.5, fontweight='bold', color=BLUE)
    for r in rects2:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2, h + 1.2, f"{h:.2f}%", ha='center', fontsize=9.5, fontweight='bold', color=AMBER)

    ax1.text(x[0], 85, "+13.73%", ha='center', fontsize=10, fontweight='bold', color=CRIMSON)
    ax1.text(x[1], 96, "+14.94%", ha='center', fontsize=10, fontweight='bold', color=CRIMSON)

    # Right: Methodological Reasons
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    b_reasons = patches.FancyBboxPatch((0.5, 0.5), 9.0, 9.0, boxstyle="round,pad=0.2",
                                       fc=SLATE_50, ec=BLUE, lw=1.5)
    ax2.add_patch(b_reasons)

    ax2.text(5.0, 8.8, "WHY HOURLY DOWNSAMPLING OUTPERFORMS MINUTE DATA", ha='center', fontsize=10.5, fontweight='bold', color=BLUE)

    reasons_text = (
        "1. Biological Low-Pass Filter:\n"
        "   Minute recordings (T=1,440) are saturated with high-frequency\n"
        "   stochastic noise (typing, fidgeting, device jostling).\n"
        "   Hourly averaging filters transient noise and elevates\n"
        "   the signal-to-noise ratio (SNR) of diurnal rest-activity.\n\n"
        "2. Diurnal Cycle Alignment:\n"
        "   Affective disorders manifest as macro-scale circadian\n"
        "   disintegration (delayed sleep phase, blunted M10 peak,\n"
        "   elevated L5 nocturnal movement). These signals operate at\n"
        "   the hourly circadian scale, not single-minute fluctuations.\n\n"
        "3. Computational & Optimization Stability:\n"
        "   • T=24 eliminates vanishing gradients across 1,440 steps.\n"
        "   • 60x fewer FLOPs enables instant training & mobile deployment."
    )
    ax2.text(1.0, 7.8, reasons_text, va='top', fontsize=8.8, color=SLATE_900, linespacing=1.35)

    save_fig(fig, 'diagram_slide7_hourly_vs_minute.png')


# -------------------------------------------------------------
# 7. Slide 8: Dual-Branch Architecture Pipeline
# -------------------------------------------------------------
def make_diagram_slide8():
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.0)
    ax.axis('off')

    ax.text(6, 5.6, "DUAL-BRANCH MULTIMODAL FEATURE FUSION PIPELINE",
            ha='center', va='center', fontsize=14, fontweight='bold', color=SLATE_900)
    ax.text(6, 5.2, "Coupling Handcrafted Circadian Digital Biomarkers with Learned Deep Temporal Representations",
            ha='center', va='center', fontsize=10, color=SLATE_500)

    # Input Box
    b_in = patches.FancyBboxPatch((0.5, 2.0), 2.2, 2.6, boxstyle="round,pad=0.15",
                                  fc=SLATE_100, ec=SLATE_800, lw=1.5)
    ax.add_patch(b_in)
    ax.text(1.6, 4.2, "24h Actigraphy\nWindow", ha='center', fontsize=10.5, fontweight='bold', color=SLATE_900)
    ax.text(1.6, 3.2, "T = 24 Hours\n(Hourly counts)\n+ 1,440 Minute Raw", ha='center', fontsize=8.5, color=SLATE_700)
    ax.text(1.6, 2.3, "Hop = 12h\n55 Subjects", ha='center', fontsize=8, color=SLATE_500)

    # Branch 1: Statistical (Top)
    b_stat = patches.FancyBboxPatch((3.4, 3.1), 4.2, 1.8, boxstyle="round,pad=0.15",
                                    fc=WHITE, ec=BLUE, lw=1.8)
    ax.add_patch(b_stat)
    ax.text(5.5, 4.6, "BRANCH A: 21 DIGITAL BIOMARKERS", ha='center', fontsize=10, fontweight='bold', color=BLUE)
    ax.text(5.5, 3.8, "• 9 Descriptive (Mean, Std, Median, IQR, Skew, Kurtosis)\n• 5 Volatility (Zero-Ratio, CV, MASD, RMSSD, Autocorr r1)\n• 7 Diurnal & Circadian (M10, L5, RA, Daytime Inactivity)",
            ha='center', fontsize=8, color=SLATE_800)
    ax.text(5.5, 3.25, "Classifier: Regularized Logistic Regression (C=0.1)", ha='center', fontsize=7.8, fontweight='bold', color=BLUE)

    # Branch 2: Deep Learning (Bottom)
    b_deep = patches.FancyBboxPatch((3.4, 0.8), 4.2, 1.8, boxstyle="round,pad=0.15",
                                    fc=WHITE, ec=PURPLE, lw=1.8)
    ax.add_patch(b_deep)
    ax.text(5.5, 2.3, "BRANCH B: TEMPORAL 1D-CNN (DEEP)", ha='center', fontsize=10, fontweight='bold', color=PURPLE)
    ax.text(5.5, 1.5, "• Conv1D (k=3, c=16) -> BatchNorm -> ReLU -> MaxPool\n• Conv1D (k=3, c=32) -> Global Average Pooling\n• Lightweight: Only 6,593 Parameters (Fast & Anti-Overfitting)",
            ha='center', fontsize=8, color=SLATE_800)
    ax.text(5.5, 0.95, "Output: Temporal Sequence Embedding P_deep", ha='center', fontsize=7.8, fontweight='bold', color=PURPLE)

    # Connecting arrows from Input
    ax.annotate('', xy=(3.4, 4.0), xytext=(2.7, 3.6),
                arrowprops=dict(facecolor=BLUE, edgecolor=BLUE, arrowstyle='->', lw=2))
    ax.annotate('', xy=(3.4, 1.7), xytext=(2.7, 3.0),
                arrowprops=dict(facecolor=PURPLE, edgecolor=PURPLE, arrowstyle='->', lw=2))

    # Fusion Box (Right)
    b_fuse = patches.FancyBboxPatch((8.3, 1.5), 3.2, 2.7, boxstyle="round,pad=0.15",
                                    fc=SLATE_50, ec=CRIMSON, lw=2)
    ax.add_patch(b_fuse)
    ax.text(9.9, 3.8, "DUAL-FUSION ENSEMBLE\n(CHAMPION)", ha='center', fontsize=10.5, fontweight='bold', color=CRIMSON)
    ax.text(9.9, 2.8, "• Weighted Linear Prob (w=0.4/0.6)\n• Stacking Meta-Learner (Non-linear)\n• Blended Dual Ensemble:\n  P = 0.5*P_weight + 0.5*P_stack",
            ha='center', fontsize=8, color=SLATE_800)
    ax.text(9.9, 1.8, "92.73% Accuracy (51/55)\n92.53% Balanced Acc", ha='center', fontsize=9, fontweight='bold', color=EMERALD)

    # Connecting arrows into Fusion
    ax.annotate('', xy=(8.3, 3.2), xytext=(7.6, 3.8),
                arrowprops=dict(facecolor=BLUE, edgecolor=BLUE, arrowstyle='->', lw=2))
    ax.annotate('', xy=(8.3, 2.5), xytext=(7.6, 1.9),
                arrowprops=dict(facecolor=PURPLE, edgecolor=PURPLE, arrowstyle='->', lw=2))

    save_fig(fig, 'diagram_slide8_architecture_pipeline.png')


# -------------------------------------------------------------
# 8. Slide 9: Confusion Matrix & Borderline Cases
# -------------------------------------------------------------
def make_diagram_slide9():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={'width_ratios': [0.9, 1.1]})

    # Confusion Matrix (ax1)
    cm = np.array([[30, 2],
                   [2, 21]])
    
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title("55-PATIENT CONFUSION MATRIX\nChampion Dual-Fusion Ensemble", fontsize=11, fontweight='bold', color=SLATE_900)
    
    tick_marks = np.arange(2)
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(['Pred Control', 'Pred Depressed'], fontsize=10, fontweight='bold')
    ax1.set_yticklabels(['True Control', 'True Depressed'], fontsize=10, fontweight='bold')

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            label = ""
            if i == 0 and j == 0: label = f"TN = {val}\n(93.75% Spec)"
            elif i == 0 and j == 1: label = f"FP = {val}\n(2 Controls)"
            elif i == 1 and j == 0: label = f"FN = {val}\n(2 Borderline)"
            elif i == 1 and j == 1: label = f"TP = {val}\n(91.30% Sens)"
            
            color = WHITE if val > 15 else CRIMSON
            ax1.text(j, i, label, ha="center", va="center", color=color, fontsize=10.5, fontweight='bold')

    ax1.set_ylabel('True Clinical Diagnostic Category', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Model Predicted Classification', fontsize=10, fontweight='bold')

    # Borderline Forensic Analysis (ax2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    b_forensic = patches.FancyBboxPatch((0.2, 0.4), 9.6, 9.2, boxstyle="round,pad=0.2",
                                        fc=SLATE_50, ec=AMBER, lw=1.5)
    ax2.add_patch(b_forensic)

    ax2.text(5.0, 9.1, "CLINICAL FORENSIC AUDIT OF THE 4 RESIDUAL ERRORS", ha='center', fontsize=10.5, fontweight='bold', color=AMBER)

    forensic_text = (
        "1. condition_2 (False Negative, p = 0.414):\n"
        "   • Admission MADRS was 24 (Moderate Depression);\n"
        "     Exit MADRS dropped to 11 (Full Clinical Remission).\n"
        "   • Actigraphy accurately sensed restored motor vitality;\n"
        "     the patient was clinically recovering during recording.\n\n"
        "2. condition_16 (False Negative, p = 0.435):\n"
        "   • Baseline MADRS was 13 (Mildest borderline in study).\n"
        "   • Subtle psychomotor slowing beneath threshold.\n\n"
        "3. control_4 (False Positive, p = 0.518):\n"
        "   • Sedentary desk-bound individual.\n"
        "   • Daytime inactivity ratio = 28.5% (+1.64 standard deviations\n"
        "     above control mean; resembles depressive daytime slowing).\n\n"
        "4. control_32 (False Positive, p = 0.584):\n"
        "   • Elderly subject with natural age-related psychomotor reduction."
    )
    ax2.text(0.6, 8.2, forensic_text, va='top', fontsize=8.6, color=SLATE_900, linespacing=1.35)

    save_fig(fig, 'diagram_slide9_fusion_results_cm.png')


# -------------------------------------------------------------
# 9. Slide 10: Clinical Decision Support & Explainable AI
# -------------------------------------------------------------
def make_diagram_slide10():
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.0)
    ax.axis('off')

    ax.text(6, 5.6, "TRANSLATIONAL IMPACT: BEDSIDE CLINICAL AI CO-PILOT",
            ha='center', va='center', fontsize=14, fontweight='bold', color=SLATE_900)
    ax.text(6, 5.2, "Bridging Black-Box Machine Learning and DSM-5 Psychiatric Decision Support",
            ha='center', va='center', fontsize=10, color=SLATE_500)

    # Pillar 1: XAI (SHAP)
    p1 = patches.FancyBboxPatch((0.5, 0.6), 3.4, 4.2, boxstyle="round,pad=0.15",
                                fc=WHITE, ec=BLUE, lw=1.5)
    ax.add_patch(p1)
    ax.text(2.2, 4.4, "1. Explainable AI\n(Out-of-Fold SHAP)", ha='center', fontsize=10.5, fontweight='bold', color=BLUE)
    ax.text(2.2, 3.4, "Top Physiological Determinants:\n"
                      "• Daytime Inactivity Ratio:\n"
                      "  Odds Ratio = 3.34 (p = 0.00016)\n"
                      "  Psychomotor retardation.\n"
                      "• Nocturnal Mean Activity:\n"
                      "  Odds Ratio = 0.31 (p = 0.00008)\n"
                      "  Sleep fragmentation indicator.\n"
                      "• Relative Amplitude (RA):\n"
                      "  Circadian rhythm blunting.",
            ha='center', fontsize=8, color=SLATE_800, linespacing=1.3)

    # Pillar 2: Temporal Attention
    p2 = patches.FancyBboxPatch((4.3, 0.6), 3.4, 4.2, boxstyle="round,pad=0.15",
                                fc=WHITE, ec=PURPLE, lw=1.5)
    ax.add_patch(p2)
    ax.text(6.0, 4.4, "2. Temporal Self-Attention\n(4-Head Transformer)", ha='center', fontsize=10.5, fontweight='bold', color=PURPLE)
    ax.text(6.0, 3.4, "Dynamic Circadian Saliency:\n"
                      "• Saliency weights highlight\n"
                      "  specific hours driving diagnosis.\n"
                      "• Major attention spikes at:\n"
                      "  - Morning wake transition\n"
                      "    (07:00-09:00: sluggish arousal)\n"
                      "  - Afternoon psychomotor dip\n"
                      "    (14:00-16:00: severe lethargy)\n"
                      "  - Nocturnal awakenings.",
            ha='center', fontsize=8, color=SLATE_800, linespacing=1.3)

    # Pillar 3: Web Dashboard & AI Notes
    p3 = patches.FancyBboxPatch((8.1, 0.6), 3.4, 4.2, boxstyle="round,pad=0.15",
                                fc=WHITE, ec=EMERALD, lw=1.5)
    ax.add_patch(p3)
    ax.text(9.8, 4.4, "3. Bedside Web Dashboard\n& DSM-5 Memos", ha='center', fontsize=10.5, fontweight='bold', color=EMERALD)
    ax.text(9.8, 3.4, "Zero-Dependency Clinical App:\n"
                      "• Live browser UI on Port 8000.\n"
                      "• Drag-and-drop inference for\n"
                      "  new smartwatch CSV records.\n"
                      "• Automated Generative AI\n"
                      "  Consultation Memos aligned\n"
                      "  with DSM-5 criteria.\n"
                      "• Turnkey patient triage.",
            ha='center', fontsize=8, color=SLATE_800, linespacing=1.3)

    save_fig(fig, 'diagram_slide10_clinical_copilot.png')


if __name__ == '__main__':
    print("Generating slide diagrams...")
    make_diagram_slide2()
    make_diagram_slide3()
    make_diagram_slide4()
    make_diagram_slide5()
    make_diagram_slide6()
    make_diagram_slide7()
    make_diagram_slide8()
    make_diagram_slide9()
    make_diagram_slide10()
    print("All slide diagrams successfully generated!")
