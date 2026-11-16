"""
Generate a comprehensive, publication-grade diagram illustrating:
1. Minute-level raw actigraphy signal
2. Minute-to-Hour aggregation (60-minute mean pooling / biological low-pass filter)
3. 24-Hour window creation
4. 12-Hour hop size and 50% sliding window overlap mechanism
5. Subject boundary preservation rule
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import pandas as pd

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.makedirs('presentation/diagrams', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# Color Palette Tokens (White Background & High Contrast)
WHITE = '#FFFFFF'
SLATE_950 = '#020617'
SLATE_900 = '#0F172A'
SLATE_800 = '#1E293B'
SLATE_700 = '#334155'
SLATE_600 = '#475569'
SLATE_400 = '#94A3B8'
SLATE_200 = '#E2E8F0'
SLATE_100 = '#F1F5F9'
SLATE_50 = '#F8FAFC'

CRIMSON = '#991B1B'
CRIMSON_LIGHT = '#DC2626'
CRIMSON_BG = '#FEF2F2'

BLUE = '#1D4ED8'
BLUE_LIGHT = '#3B82F6'
BLUE_BG = '#EFF6FF'

EMERALD = '#047857'
EMERALD_LIGHT = '#10B981'
EMERALD_BG = '#ECFDF5'

AMBER = '#B45309'
AMBER_LIGHT = '#F59E0B'
AMBER_BG = '#FEF3C7'

PURPLE = '#6D28D9'
PURPLE_LIGHT = '#8B5CF6'
PURPLE_BG = '#F5F3FF'

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = WHITE
plt.rcParams['axes.facecolor'] = WHITE


def load_sample_actigraphy():
    """Load 48 hours (2,880 minutes) of actual actigraphy from control_1.csv."""
    path = 'data/control/control_1.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        act = df['activity'].values[:2880].astype(float)
    else:
        # Fallback synthetic realistic actigraphy
        t = np.linspace(0, 48, 2880)
        diurnal = 350 * np.exp(-((t % 24 - 14)**2)/18) + 20
        noise = np.random.exponential(scale=60, size=2880) * (np.random.rand(2880) > 0.3)
        act = np.clip(diurnal + noise, 0, 1500)
    return act


def make_minute_to_hour_window_overlap_diagram():
    act_raw = load_sample_actigraphy()
    # Downsample to 48 hours: reshape (48, 60) -> mean(axis=1)
    act_hourly = act_raw.reshape(48, 60).mean(axis=1)

    fig = plt.figure(figsize=(15, 10.5))
    
    # Grid definition:
    # Top banner & Title: y in [0.92, 0.98]
    # Tier 1: Minute-Level Actigraphy + Zoom Inset: y in [0.65, 0.90]
    # Tier 2: Hourly Downsampled Series: y in [0.42, 0.60]
    # Tier 3: 24h Window Creation & 50% Overlap: y in [0.08, 0.38]

    # Main Title Header
    fig.text(0.5, 0.97, "ACTIGRAPHY SIGNAL PROCESSING & WINDOW GENERATION PIPELINE",
             ha='center', va='top', fontsize=16, fontweight='bold', color=SLATE_900)
    fig.text(0.5, 0.945, "From Continuous 1-Minute Integrations to Hourly Anti-Aliasing and 24-Hour Overlapping Slices (50% Hop)",
             ha='center', va='top', fontsize=11, color=SLATE_600)

    # =========================================================================
    # TIER 1: RAW MINUTE-LEVEL ACTIGRAPHY (48 Hours = 2,880 Minutes)
    # =========================================================================
    ax_min = fig.add_axes([0.08, 0.70, 0.58, 0.20])
    
    minutes_x = np.arange(2880) / 60.0  # in hours (0 to 48)
    ax_min.plot(minutes_x, act_raw, color=SLATE_400, lw=0.6, alpha=0.85, label='Raw 1-Min Counts (High-Frequency Micro-Noise)')
    
    # Highlight a 1-hour sample chunk around Hour 14 (Hour 14 to 15, index 840 to 900)
    zoom_start_h = 14.0
    zoom_end_h = 15.0
    ax_min.axvspan(zoom_start_h, zoom_end_h, color=AMBER_LIGHT, alpha=0.35, ec=AMBER, lw=1.5, ls='--')
    ax_min.text(14.5, 950, "Sample 1h Block\n(60 min)", ha='center', va='bottom', fontsize=8, fontweight='bold', color=AMBER)

    ax_min.set_xlim(0, 48)
    ax_min.set_ylim(0, 1200)
    ax_min.set_ylabel("Activity / Min", fontsize=9.5, fontweight='bold', color=SLATE_800)
    ax_min.set_title("STAGE 1: Continuous Raw Actigraphy (Cambridge Neurotechnology Actiwatch AW4, 1-Min Epochs)",
                     fontsize=10.5, fontweight='bold', color=SLATE_900, loc='left', pad=6)
    ax_min.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    ax_min.set_xticklabels(['0h (Day 1)', '6h', '12h (Noon)', '18h', '24h (Day 2)', '30h', '36h (Noon)', '42h', '48h'], fontsize=8.5)
    ax_min.grid(True, linestyle=':', alpha=0.5, color=SLATE_200)

    # Inset Zoom Panel: 60 Minutes of Hour 14 (Right of Tier 1)
    ax_zoom = fig.add_axes([0.72, 0.70, 0.22, 0.20])
    min_chunk = act_raw[840:900]
    ax_zoom.bar(np.arange(1, 61), min_chunk, color=AMBER, width=0.85, edgecolor=SLATE_800, lw=0.5)
    mean_val = min_chunk.mean()
    ax_zoom.axhline(mean_val, color=CRIMSON, lw=2, linestyle='--', label=f'Hourly Mean ({mean_val:.1f})')
    
    ax_zoom.set_xlim(0, 61)
    ax_zoom.set_ylim(0, max(min_chunk.max()*1.15, 600))
    ax_zoom.set_title("1-Hour Zoom (m = 1..60)", fontsize=9.5, fontweight='bold', color=AMBER)
    ax_zoom.set_xlabel("Minute (1 to 60)", fontsize=8, color=SLATE_800)
    ax_zoom.set_ylabel("Counts", fontsize=8, color=SLATE_800)
    ax_zoom.tick_params(labelsize=7.5)
    ax_zoom.legend(loc='upper right', fontsize=7.5)
    ax_zoom.grid(True, linestyle=':', alpha=0.4, color=SLATE_200)

    # Aggregation Operator Indicator Banner
    box_agg = FancyBboxPatch((0.08, 0.635), 0.86, 0.045, boxstyle="round,pad=0.008",
                             fc=BLUE_BG, ec=BLUE, lw=1.2, transform=fig.transFigure)
    fig.patches.append(box_agg)
    fig.text(0.51, 0.658,
             r"$\mathbf{STAGE\ 2:\ MINUTE\text{-}TO\text{-}HOUR\ CONVERSION\ (BIOLOGICAL\ LOW\text{-}PASS\ FILTER)}\quad\longrightarrow\quad "
             r"\bar{h}_j = \frac{1}{60} \sum_{m=1}^{60} x_{(j-1)\cdot 60 + m}\quad \left[\text{60}\times\text{ Dimensionality Reduction: } 1,440 \rightarrow 24 \text{ steps/day}\right]$",
             ha='center', va='center', fontsize=9.5, fontweight='bold', color=BLUE)

    # =========================================================================
    # TIER 2: HOURLY DOWNSAMPLED ACTIGRAPHY (48 Hourly Bins)
    # =========================================================================
    ax_hr = fig.add_axes([0.08, 0.44, 0.86, 0.17])
    
    hours_x = np.arange(48) + 0.5  # centered bars
    bar_colors = [EMERALD if (6 <= h % 24 < 22) else SLATE_600 for h in range(48)]
    ax_hr.bar(hours_x, act_hourly, width=0.75, color=bar_colors, edgecolor=SLATE_900, lw=0.8,
              label='Hourly Mean Activity (Daytime Active: Green | Night Rest: Slate)')
    
    # Smooth spline / trend curve
    ax_hr.plot(hours_x, act_hourly, color=CRIMSON, lw=2.2, marker='o', markersize=4,
               label='Circadian Diurnal Trajectory (Preserved Amplitude)')

    ax_hr.set_xlim(0, 48)
    ax_hr.set_ylim(0, max(act_hourly.max()*1.2, 500))
    ax_hr.set_ylabel("Mean Counts / Hour", fontsize=9.5, fontweight='bold', color=SLATE_800)
    ax_hr.set_title("STAGE 3: Hourly Downsampled Series (Anti-Aliased Circadian Diurnal Macro-Rhythm)",
                    fontsize=10.5, fontweight='bold', color=SLATE_900, loc='left', pad=6)
    ax_hr.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    ax_hr.set_xticklabels(['00:00 (Day 1)', '06:00', '12:00 (Noon)', '18:00', '00:00 (Day 2)', '06:00', '12:00 (Noon)', '18:00', '24:00 (End)'], fontsize=8.5)
    ax_hr.grid(True, linestyle=':', alpha=0.5, color=SLATE_200)
    ax_hr.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

    # =========================================================================
    # TIER 3: 24-HOUR WINDOW CREATION & 50% OVERLAP MECHANISM
    # =========================================================================
    ax_win = fig.add_axes([0.08, 0.08, 0.86, 0.32])
    ax_win.set_xlim(0, 48)
    ax_win.set_ylim(0, 7.2)
    ax_win.axis('off')

    ax_win.text(0, 6.9, "STAGE 4: 24-HOUR WINDOW SLICING & 12-HOUR HOP (50% OVERLAP MECHANISM)",
                fontsize=11.5, fontweight='bold', color=SLATE_900)

    # Timeline Ruler at top of Tier 3 (y = 5.9)
    ax_win.plot([0, 48], [5.9, 5.9], color=SLATE_900, lw=2)
    for h in range(0, 49, 6):
        ax_win.plot([h, h], [5.75, 6.05], color=SLATE_900, lw=1.8)
        lbl = f"{h}h" if h % 24 != 0 else (f"{h}h\n(Day {h//24 + 1})" if h < 48 else "48h\n(End)")
        ax_win.text(h, 6.2, lbl, ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=SLATE_800)

    # -------------------------------------------------------------
    # Window 1: Hours 0 to 24 (Day 1)
    # -------------------------------------------------------------
    b_w1 = FancyBboxPatch((0, 4.3), 24, 1.1, boxstyle="round,pad=0.1",
                          fc=BLUE_BG, ec=BLUE, lw=2.2)
    ax_win.add_patch(b_w1)
    ax_win.text(12, 4.85, "WINDOW 1 (W1): Day 1 Complete 24h Cycle [00:00 to 24:00]\n"
                          "Length W = 24 Hours (1,440 Minutes) | Shape: (24, 1) Hourly or (1440, 1) Minute",
                ha='center', va='center', fontsize=9.2, fontweight='bold', color=BLUE)

    # -------------------------------------------------------------
    # Window 2: Hours 12 to 36 (Day 1 12:00 to Day 2 12:00)
    # -------------------------------------------------------------
    b_w2 = FancyBboxPatch((12, 2.9), 24, 1.1, boxstyle="round,pad=0.1",
                          fc=EMERALD_BG, ec=EMERALD, lw=2.2)
    ax_win.add_patch(b_w2)
    ax_win.text(24, 3.45, "WINDOW 2 (W2): Inter-Day 24h Cycle [Day 1 12:00 to Day 2 12:00]\n"
                          "Shifted by Hop Size H = 12 Hours (720 Minutes)",
                ha='center', va='center', fontsize=9.2, fontweight='bold', color=EMERALD)

    # -------------------------------------------------------------
    # Window 3: Hours 24 to 48 (Day 2)
    # -------------------------------------------------------------
    b_w3 = FancyBboxPatch((24, 1.5), 24, 1.1, boxstyle="round,pad=0.1",
                          fc=PURPLE_BG, ec=PURPLE, lw=2.2)
    ax_win.add_patch(b_w3)
    ax_win.text(36, 2.05, "WINDOW 3 (W3): Day 2 Complete 24h Cycle [00:00 to 24:00]\n"
                          "Shifted by Hop Size H = 12 Hours (720 Minutes)",
                ha='center', va='center', fontsize=9.2, fontweight='bold', color=PURPLE)

    # -------------------------------------------------------------
    # Highlight Overlap Zone 1: Hours 12 to 24 (Between W1 and W2)
    # -------------------------------------------------------------
    b_ov1 = FancyBboxPatch((12.1, 4.0), 11.8, 1.5, boxstyle="round,pad=0.05",
                           fc=AMBER_BG, ec=AMBER, lw=1.5, ls='--', alpha=0.7)
    ax_win.add_patch(b_ov1)
    ax_win.text(18, 4.0, "[50% OVERLAP ZONE: 12h / 720 min]\nShared Identical Data Between W1 & W2",
                ha='center', va='center', fontsize=8, fontweight='bold', color=AMBER)

    # -------------------------------------------------------------
    # Highlight Overlap Zone 2: Hours 24 to 36 (Between W2 and W3)
    # -------------------------------------------------------------
    b_ov2 = FancyBboxPatch((24.1, 2.6), 11.8, 1.5, boxstyle="round,pad=0.05",
                           fc=AMBER_BG, ec=AMBER, lw=1.5, ls='--', alpha=0.7)
    ax_win.add_patch(b_ov2)
    ax_win.text(30, 2.6, "[50% OVERLAP ZONE: 12h / 720 min]\nShared Identical Data Between W2 & W3",
                ha='center', va='center', fontsize=8, fontweight='bold', color=AMBER)

    # Hop Arrows
    ax_win.annotate('', xy=(12, 5.5), xytext=(0, 5.5),
                    arrowprops=dict(facecolor=AMBER, edgecolor=AMBER, arrowstyle='<->', lw=1.8))
    ax_win.text(6, 5.65, "Hop H = 12h", ha='center', fontsize=8, fontweight='bold', color=AMBER)

    ax_win.annotate('', xy=(24, 5.5), xytext=(12, 5.5),
                    arrowprops=dict(facecolor=AMBER, edgecolor=AMBER, arrowstyle='<->', lw=1.8))
    ax_win.text(18, 5.65, "Hop H = 12h", ha='center', fontsize=8, fontweight='bold', color=AMBER)

    # -------------------------------------------------------------
    # Bottom Summary Metadata Cards (Formulas & Guardrails)
    # -------------------------------------------------------------
    # Card 1: Math Parameters
    c1 = FancyBboxPatch((0, 0.0), 14.5, 1.2, boxstyle="round,pad=0.1",
                        fc=SLATE_50, ec=SLATE_200, lw=1.2)
    ax_win.add_patch(c1)
    ax_win.text(0.3, 0.95, "SLIDING WINDOW PARAMETERS", fontsize=8.5, fontweight='bold', color=SLATE_900)
    ax_win.text(0.3, 0.60, "• Window Length (W): 24h = 1,440 minutes\n• Hop Size (H): 12h = 720 minutes\n• Overlap Ratio: (W - H) / W = 12 / 24 = 50.0%",
                fontsize=7.8, color=SLATE_700, linespacing=1.25)

    # Card 2: Yield Formula
    c2 = FancyBboxPatch((15.5, 0.0), 15.5, 1.2, boxstyle="round,pad=0.1",
                        fc=SLATE_50, ec=SLATE_200, lw=1.2)
    ax_win.add_patch(c2)
    ax_win.text(15.8, 0.95, "COHORT WINDOW YIELD FORMULA", fontsize=8.5, fontweight='bold', color=BLUE)
    ax_win.text(15.8, 0.60, "• Per Subject with N_i Days: W_i = 2 * N_i - 1 windows\n• Depresjon Total: 693 days -> 1,331 Total Windows\n  (Condition: 559 windows, Control: 772 windows)",
                fontsize=7.8, color=SLATE_700, linespacing=1.25)

    # Card 3: Boundary Guardrail
    c3 = FancyBboxPatch((32.0, 0.0), 15.8, 1.2, boxstyle="round,pad=0.1",
                        fc=CRIMSON_BG, ec=CRIMSON, lw=1.2)
    ax_win.add_patch(c3)
    ax_win.text(32.3, 0.95, "SUBJECT BOUNDARY GUARDRAIL", fontsize=8.5, fontweight='bold', color=CRIMSON)
    ax_win.text(32.3, 0.60, "• Strict Isolation: Windows NEVER cross subject boundaries.\n• Prevents cross-subject temporal mixing.\n• Enables strict 55-Fold Leave-One-Subject-Out (LOSO).",
                fontsize=7.8, color=SLATE_800, linespacing=1.25)

    # Save to multiple locations
    output_png = 'presentation/diagrams/actigraphy_minute_to_hour_window_overlap.png'
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor=WHITE, edgecolor='none')
    print(f"Saved: {output_png}")

    output_pdf = 'presentation/diagrams/actigraphy_minute_to_hour_window_overlap.pdf'
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor=WHITE, edgecolor='none')
    print(f"Saved: {output_pdf}")

    output_results = 'results/figures/figure11_windowing_and_overlap_mechanism.png'
    fig.savefig(output_results, dpi=300, bbox_inches='tight', facecolor=WHITE, edgecolor='none')
    print(f"Saved: {output_results}")

    plt.close(fig)


if __name__ == '__main__':
    print("Generating minute-to-hour and window overlap diagram...")
    make_minute_to_hour_window_overlap_diagram()
    print("Done!")
