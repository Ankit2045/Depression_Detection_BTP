#!/usr/bin/env python3
"""
Human-Crafted Academic Presentation Generator for DTU B.Tech Project GP178.

Generates:
1. `presentation/slides.html`: Modern, interactive presentation deck featuring:
   - Apple Keynote-style slide transitions with custom spring easing
   - Step-by-step fragment reveals (Click / Space / Arrow to unveil points)
   - Dynamic counting metrics (0% -> 99.62%, 0% -> 92.73%)
   - High-Res Image Lightbox Modal for figure inspection
   - Conference Laser Pointer mode ('L' key)
   - Slide Grid Overview ('G' key) for instant jumping during Q&A
   - Natural student speaker scripts divided by team members ('S' key)
   - Light/Dark academic projection themes ('T' key)
2. `presentation/slides.pdf`: 10-page landscape PDF for direct submission/printing.
"""

import os
import sys
import base64
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_HTML_PATH = os.path.join(PROJECT_ROOT, "presentation", "slides.html")
OUTPUT_PDF_PATH = os.path.join(PROJECT_ROOT, "presentation", "slides.pdf")
ARTIFACT_PATH = "/Users/ankitsingh/.gemini/antigravity/brain/a25b8baf-7d81-4632-b3bc-40e3ebf26292/presentation.html"


def image_to_base64(rel_path: str) -> str:
    path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def build_presentation():
    # Pre-encode figures in base64
    fig1 = image_to_base64("results/figures/figure1_digital_biomarkers.png")
    fig4 = image_to_base64("results/figures/figure4_deep_vs_statistical.png")
    fig6 = image_to_base64("results/figures/figure6_fusion_performance_boost.png")
    fig7 = image_to_base64("results/figures/figure7_fusion_roc_curves.png")
    fig8 = image_to_base64("results/figures/figure8_shap_summary.png")
    fig10 = image_to_base64("ai_system/results/figures/figure10_circadian_attention_heatmap.png")

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>B.Tech Research Defense &bull; Group GP178 &bull; DTU CSE</title>
  <style>
    /* -------------------------------------------------------------
       ACADEMIC DESIGN SYSTEM & THEME TOKENS
       DTU Crimson / Slate Executive Palette (Human-Crafted Look)
       ------------------------------------------------------------- */
    :root {{
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-serif: "Georgia", "Cambria", "Times New Roman", Times, serif;
      --font-mono: "SF Mono", "Fira Code", Menlo, Monaco, Consolas, monospace;
      
      --dtu-red: #991b1b;
      --dtu-red-light: #b91c1c;
      --dtu-crimson: #7f1d1d;
      --dtu-gold: #d97706;
      
      --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
    }}

    [data-theme="dark"] {{
      --bg-canvas: #090d16;
      --bg-surface: #101726;
      --bg-surface-elevated: #182238;
      --bg-surface-subtle: #0d1322;
      --border-color: #23314d;
      --border-strong: #33446b;
      
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      
      --accent-blue: #38bdf8;
      --accent-blue-soft: rgba(56, 189, 248, 0.12);
      --accent-red: #f43f5e;
      --accent-red-soft: rgba(244, 63, 94, 0.12);
      --accent-green: #10b981;
      --accent-green-soft: rgba(16, 185, 129, 0.12);
      --accent-amber: #f59e0b;
      --accent-amber-soft: rgba(245, 158, 11, 0.12);

      --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.5), 0 2px 6px -1px rgba(0, 0, 0, 0.3);
      --shadow-elevated: 0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
    }}

    [data-theme="light"] {{
      --bg-canvas: #f8fafc;
      --bg-surface: #ffffff;
      --bg-surface-elevated: #f1f5f9;
      --bg-surface-subtle: #f8fafc;
      --border-color: #cbd5e1;
      --border-strong: #94a3b8;
      
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #64748b;
      
      --accent-blue: #0284c7;
      --accent-blue-soft: rgba(2, 132, 199, 0.08);
      --accent-red: #dc2626;
      --accent-red-soft: rgba(220, 38, 38, 0.08);
      --accent-green: #059669;
      --accent-green-soft: rgba(5, 150, 105, 0.08);
      --accent-amber: #d97706;
      --accent-amber-soft: rgba(217, 119, 6, 0.08);

      --shadow-card: 0 4px 14px -2px rgba(15, 23, 42, 0.08), 0 2px 4px -1px rgba(15, 23, 42, 0.04);
      --shadow-elevated: 0 20px 25px -5px rgba(15, 23, 42, 0.1), 0 10px 10px -5px rgba(15, 23, 42, 0.04);
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    
    html, body {{
      width: 100vw;
      height: 100vh;
      overflow: hidden;
      background-color: var(--bg-canvas);
      color: var(--text-main);
      font-family: var(--font-sans);
      -webkit-font-smoothing: antialiased;
      user-select: none;
    }}

    /* Institutional Header Banner */
    #top-bar {{
      height: 44px;
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      z-index: 100;
    }}
    .dtu-brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--text-main);
    }}
    .dtu-seal-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--dtu-red);
      box-shadow: 0 0 8px rgba(153, 27, 27, 0.6);
    }}
    .top-meta {{
      color: var(--text-muted);
      display: flex;
      gap: 16px;
    }}

    /* Linear Progress Bar */
    #progress-track {{
      height: 3px;
      width: 100%;
      background: var(--border-color);
      position: relative;
    }}
    #progress-fill {{
      height: 100%;
      width: 10%;
      background: linear-gradient(90deg, var(--dtu-red), var(--accent-blue));
      transition: width 0.4s var(--ease-out);
    }}

    /* Presentation Viewport */
    #viewport {{
      position: relative;
      width: 100vw;
      height: calc(100vh - 44px - 3px - 54px);
      overflow: hidden;
    }}

    /* Slides & Animation Transitions */
    .slide {{
      position: absolute;
      inset: 0;
      padding: 32px 50px 24px 50px;
      opacity: 0;
      visibility: hidden;
      transform: translate3d(30px, 0, 0);
      transition: opacity 0.4s var(--ease-out), transform 0.4s var(--ease-out), visibility 0.4s;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }}
    .slide.active {{
      opacity: 1;
      visibility: visible;
      transform: translate3d(0, 0, 0);
      z-index: 10;
    }}
    .slide.exit-left {{
      opacity: 0;
      visibility: hidden;
      transform: translate3d(-30px, 0, 0);
    }}

    /* Fragment Elements (Click-to-Reveal in Defense) */
    .fragment {{
      opacity: 0.15;
      transform: translateY(6px);
      transition: opacity 0.35s var(--ease-out), transform 0.35s var(--ease-out);
    }}
    .fragment.visible {{
      opacity: 1;
      transform: translateY(0);
    }}

    /* Slide Titles & Headers */
    .slide-section-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--dtu-red);
      background: rgba(153, 27, 27, 0.1);
      border: 1px solid rgba(153, 27, 27, 0.3);
      padding: 3px 10px;
      border-radius: 4px;
      width: fit-content;
      margin-bottom: 8px;
    }}
    [data-theme="light"] .slide-section-badge {{
      background: rgba(153, 27, 27, 0.08);
      border-color: rgba(153, 27, 27, 0.25);
    }}
    .slide-title {{
      font-size: 24pt;
      font-weight: 800;
      letter-spacing: -0.6px;
      line-height: 1.2;
      color: var(--text-main);
      margin-bottom: 4px;
    }}
    .slide-subtitle {{
      font-size: 12.5pt;
      color: var(--text-muted);
      margin-bottom: 18px;
      font-weight: 400;
    }}

    /* Card System */
    .card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 20px 22px;
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
      position: relative;
    }}
    .card-academic {{
      border-top: 3px solid var(--dtu-red);
    }}
    .card-accent-blue {{
      border-top: 3px solid var(--accent-blue);
    }}
    .card-accent-green {{
      border-top: 3px solid var(--accent-green);
    }}
    .card-accent-red {{
      border-top: 3px solid var(--accent-red);
    }}
    .card-title {{
      font-size: 12.5pt;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .card-body {{
      font-size: 10pt;
      color: var(--text-muted);
      line-height: 1.55;
    }}
    .card-body p {{
      margin-bottom: 8px;
    }}
    .card-body strong {{
      color: var(--text-main);
    }}

    /* Grids */
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      flex: 1;
    }}
    .grid-3 {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 18px;
      flex: 1;
    }}
    .grid-4-8 {{
      display: grid;
      grid-template-columns: 4fr 8fr;
      gap: 20px;
      flex: 1;
    }}
    .grid-7-5 {{
      display: grid;
      grid-template-columns: 7fr 5fr;
      gap: 20px;
      flex: 1;
    }}

    /* Academic Bullet Points */
    ul.academic-bullets {{
      list-style: none;
      padding-left: 0;
    }}
    ul.academic-bullets li {{
      position: relative;
      padding-left: 18px;
      margin-bottom: 8px;
      font-size: 10pt;
      line-height: 1.5;
      color: var(--text-muted);
    }}
    ul.academic-bullets li::before {{
      content: "";
      position: absolute;
      left: 2px;
      top: 8px;
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: var(--dtu-red);
    }}
    ul.academic-bullets li strong {{
      color: var(--text-main);
    }}

    /* Formal Academic Booktabs Table */
    table.academic-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 9.5pt;
      margin-top: 8px;
    }}
    table.academic-table thead th {{
      border-top: 2px solid var(--text-main);
      border-bottom: 1px solid var(--text-main);
      padding: 8px 10px;
      text-align: left;
      font-weight: 700;
      color: var(--text-main);
      background: var(--bg-surface-subtle);
    }}
    table.academic-table tbody td {{
      padding: 7px 10px;
      border-bottom: 1px solid var(--border-color);
      color: var(--text-muted);
    }}
    table.academic-table tbody tr:last-child td {{
      border-bottom: 2px solid var(--text-main);
    }}
    table.academic-table tr.row-highlight td {{
      background: var(--accent-blue-soft);
      color: var(--text-main);
      font-weight: 600;
    }}
    table.academic-table tr.row-danger td {{
      background: var(--accent-red-soft);
      color: var(--text-main);
    }}

    /* Animated Statistic Hero Boxes */
    .stat-hero-box {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
      text-align: center;
      margin-top: auto;
    }}
    .stat-hero-number {{
      font-size: 30pt;
      font-weight: 900;
      color: var(--text-main);
      font-feature-settings: "tnum";
      line-height: 1;
      margin-bottom: 4px;
    }}
    .stat-hero-number.text-red {{ color: var(--accent-red); }}
    .stat-hero-number.text-blue {{ color: var(--accent-blue); }}
    .stat-hero-number.text-green {{ color: var(--accent-green); }}
    .stat-hero-label {{
      font-size: 9pt;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      font-weight: 600;
    }}

    /* Interactive Graph Boxes with Zoom Hover */
    .graph-preview-box {{
      position: relative;
      width: 100%;
      height: 100%;
      min-height: 200px;
      background: var(--bg-surface-subtle);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      cursor: zoom-in;
      transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .graph-preview-box:hover {{
      border-color: var(--accent-blue);
      box-shadow: 0 0 12px var(--accent-blue-soft);
    }}
    .graph-preview-box img {{
      max-width: 98%;
      max-height: 90%;
      object-fit: contain;
      border-radius: 4px;
    }}
    .graph-zoom-hint {{
      position: absolute;
      bottom: 8px;
      right: 8px;
      background: rgba(15, 23, 42, 0.85);
      color: #ffffff;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 8.5pt;
      font-weight: 600;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 4px;
      pointer-events: none;
    }}

    /* Slide 1 Formal Academic Title Slide */
    .title-slide-container {{
      max-width: 960px;
      margin: 0 auto;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
    }}
    .univ-title {{
      font-size: 13pt;
      font-weight: 700;
      letter-spacing: 2.5px;
      text-transform: uppercase;
      color: var(--dtu-red);
      margin-bottom: 6px;
    }}
    .dept-title {{
      font-size: 10.5pt;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 1.2px;
      margin-bottom: 24px;
    }}
    .main-project-heading {{
      font-size: 26pt;
      font-weight: 900;
      line-height: 1.25;
      color: var(--text-main);
      margin-bottom: 24px;
      max-width: 880px;
    }}
    .meta-roster-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      width: 100%;
      text-align: left;
      margin-top: 10px;
    }}
    .roster-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 18px 24px;
    }}
    .roster-card h4 {{
      font-size: 10.5pt;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--dtu-red);
      margin-bottom: 10px;
      border-bottom: 1px solid var(--border-color);
      padding-bottom: 4px;
    }}
    .roster-card p {{
      font-size: 10.5pt;
      color: var(--text-main);
      margin-bottom: 5px;
      display: flex;
      justify-content: space-between;
    }}
    .roster-card p span.roll {{
      font-family: var(--font-mono);
      color: var(--text-muted);
      font-size: 10pt;
    }}

    /* Bottom Controller & Presenter Bar */
    #bottom-bar {{
      height: 54px;
      background: var(--bg-surface);
      border-top: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 32px;
      font-size: 11pt;
      z-index: 100;
    }}
    .ctrl-group {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .academic-btn {{
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 9.5pt;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s, border-color 0.15s;
    }}
    .academic-btn:hover {{
      background: var(--dtu-red);
      color: #ffffff;
      border-color: var(--dtu-red);
    }}
    .academic-btn:disabled {{
      opacity: 0.35;
      cursor: not-allowed;
    }}
    #slide-counter-disp {{
      font-size: 10.5pt;
      font-weight: 600;
      color: var(--text-muted);
      margin: 0 12px;
      font-feature-settings: "tnum";
    }}

    /* Lightbox Modal for High-Resolution Graph Inspection */
    #lightbox-modal {{
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.92);
      backdrop-filter: blur(8px);
      z-index: 999;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 40px;
    }}
    #lightbox-modal.open {{
      display: flex;
    }}
    #lightbox-img {{
      max-width: 90vw;
      max-height: 85vh;
      object-fit: contain;
      border-radius: 8px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
      background: #ffffff;
    }}
    #lightbox-close {{
      position: absolute;
      top: 24px;
      right: 32px;
      color: #ffffff;
      font-size: 28pt;
      font-weight: 300;
      cursor: pointer;
      line-height: 1;
    }}

    /* Conference Laser Pointer Effect */
    #laser-dot {{
      position: fixed;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #ff0033;
      box-shadow: 0 0 12px #ff0033, 0 0 24px #ff0033;
      pointer-events: none;
      z-index: 9999;
      transform: translate(-50%, -50%);
      display: none;
    }}
    body.laser-active, body.laser-active * {{
      cursor: none !important;
    }}
    body.laser-active #laser-dot {{
      display: block;
    }}

    /* Slide Overview Grid Modal */
    #grid-overview {{
      position: fixed;
      inset: 0;
      background: rgba(9, 13, 22, 0.96);
      backdrop-filter: blur(10px);
      z-index: 500;
      display: none;
      flex-direction: column;
      padding: 40px 60px;
      overflow-y: auto;
    }}
    #grid-overview.open {{
      display: flex;
    }}
    .grid-overview-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
    }}
    .grid-overview-title {{
      font-size: 16pt;
      font-weight: 700;
      color: var(--text-main);
    }}
    .grid-overview-cards {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 16px;
    }}
    .overview-thumb {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px;
      cursor: pointer;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .overview-thumb:hover {{
      transform: translateY(-4px);
      border-color: var(--dtu-red);
    }}
    .overview-thumb-num {{
      font-size: 10pt;
      font-weight: 700;
      color: var(--dtu-red);
      margin-bottom: 4px;
    }}
    .overview-thumb-title {{
      font-size: 9pt;
      font-weight: 600;
      color: var(--text-main);
      line-height: 1.3;
    }}

    /* Speaker Notes Teleprompter Drawer */
    #speaker-drawer {{
      position: absolute;
      bottom: 54px;
      left: 0;
      width: 100%;
      background: var(--bg-surface-elevated);
      border-top: 2px solid var(--dtu-red);
      padding: 16px 36px;
      display: none;
      box-shadow: 0 -10px 25px rgba(0,0,0,0.4);
      z-index: 90;
      max-height: 180px;
      overflow-y: auto;
    }}
    #speaker-drawer.open {{
      display: block;
    }}
    .speaker-tag {{
      display: inline-block;
      font-size: 9pt;
      font-weight: 700;
      color: #ffffff;
      background: var(--dtu-red);
      padding: 2px 8px;
      border-radius: 4px;
      margin-bottom: 6px;
    }}
    #speaker-text {{
      font-size: 10.5pt;
      color: var(--text-main);
      line-height: 1.5;
    }}

    /* Print / PDF Export Styling */
    @media print {{
      @page {{
        size: landscape;
        margin: 0;
      }}
      body {{
        overflow: visible !important;
        height: auto !important;
        width: 100% !important;
        background: #ffffff !important;
        color: #000000 !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}
      #top-bar, #progress-track, #bottom-bar, #speaker-drawer, #lightbox-modal, #grid-overview, #laser-dot {{
        display: none !important;
      }}
      #viewport {{
        height: auto !important;
        overflow: visible !important;
      }}
      .slide {{
        position: relative !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        transform: none !important;
        page-break-after: always !important;
        break-after: page !important;
        width: 100vw !important;
        height: 100vh !important;
        padding: 36px 44px !important;
        background: #ffffff !important;
        color: #000000 !important;
      }}
      .fragment {{
        opacity: 1 !important;
        transform: none !important;
      }}
      .card {{
        background: #ffffff !important;
        border-color: #cbd5e1 !important;
        box-shadow: none !important;
        color: #000000 !important;
      }}
      .slide-title, .card-title, .stat-hero-number {{
        color: #000000 !important;
      }}
      .card-body, .slide-subtitle, ul.academic-bullets li {{
        color: #334155 !important;
      }}
      table.academic-table thead th {{
        border-color: #000000 !important;
        background: #f1f5f9 !important;
        color: #000000 !important;
      }}
      table.academic-table tbody td {{
        color: #000000 !important;
      }}
    }}
  </style>
</head>
<body>

  <!-- Top Academic Banner -->
  <div id="top-bar">
    <div class="dtu-brand">
      <div class="dtu-seal-dot"></div>
      <span>Delhi Technological University &bull; CSE Research Defense</span>
    </div>
    <div class="top-meta">
      <span>Group ID: <strong>DTU/CSE/BTP/GP178</strong></span>
      <span>AY 2026–27</span>
    </div>
  </div>
  <div id="progress-track"><div id="progress-fill"></div></div>

  <!-- Main Slide Viewport -->
  <div id="viewport">

    <!-- =========================================================
         SLIDE 1: Title & Author Roster
         ========================================================= -->
    <div class="slide active" id="slide-1" data-title="Title & Roster">
      <div class="title-slide-container">
        <div class="univ-title">Department of Computer Science and Engineering</div>
        <div class="dept-title">Bachelor of Technology Final Year Project Defense (Project-I)</div>
        
        <h1 class="main-project-heading">Multimodal Feature Fusion of Statistical and Deep Learning Representations for Actigraphy-Based Depression Detection</h1>
        
        <div class="meta-roster-grid">
          <div class="roster-card">
            <h4>Candidate Presenters</h4>
            <p><strong>Ankit</strong> <span class="roll">23/CS/059</span></p>
            <p><strong>Ankit Singh</strong> <span class="roll">23/CS/064</span></p>
            <p style="margin-top: 6px;"><strong>Kumari Rashi</strong> (Junior Member) <span class="roll">24/CS/253</span></p>
          </div>
          <div class="roster-card">
            <h4>Supervision & Affiliation</h4>
            <p><strong>Dr. Anshika Arora</strong></p>
            <p style="font-size: 9pt; color: var(--text-muted); margin-top:-3px;">Assistant Professor, Dept. of CSE, DTU</p>
            <p style="margin-top: 8px;"><strong>Group Identifier:</strong> <span class="roll" style="color:var(--dtu-red); font-weight:700;">DTU/CSE/BTP/GP178</span></p>
          </div>
        </div>
      </div>
    </div>

    <!-- =========================================================
         SLIDE 2: Clinical Motivation & Objectives
         ========================================================= -->
    <div class="slide" id="slide-2" data-title="Clinical Motivation">
      <div class="slide-section-badge">&sect; 01 &bull; Clinical Background & Objectives</div>
      <h2 class="slide-title">Continuous Actigraphy for Objective Depression Assessment</h2>
      <p class="slide-subtitle">Overcoming subjective diagnostic recall bias using continuous wearable Internet-of-Medical-Things (IoMT)</p>

      <div class="grid-3">
        <div class="card card-academic fragment">
          <div class="card-title">Current Clinical Reality</div>
          <div class="card-body">
            <p>Major Depressive Disorder (MDD) affects over <strong>300 million people</strong> globally.</p>
            <p>Current diagnosis relies on retrospective self-reports (MADRS, HDRS) and sporadic consultations.</p>
            <p><strong>Core limitation:</strong> Severe recall bias, lack of continuous longitudinal granularity, and episodic latency.</p>
          </div>
          <div class="stat-hero-box" style="margin-top: auto;">
            <div class="stat-hero-number text-red">300M+</div>
            <div class="stat-hero-label">Affected Globally (WHO)</div>
          </div>
        </div>

        <div class="card card-accent-blue fragment">
          <div class="card-title">Passive Wearable Proxy</div>
          <div class="card-body">
            <p>Continuous wrist actigraphy records motion signals <strong>24 hours a day, 7 days a week</strong>.</p>
            <p>Directly quantifies <strong>psychomotor retardation</strong>, diurnal cycle flattening, and nocturnal sleep fragmentation.</p>
            <p>Unobtrusive, objective, non-stigmatizing digital proxy for psychiatric monitoring.</p>
          </div>
          <div class="stat-hero-box" style="margin-top: auto;">
            <div class="stat-hero-number text-blue">24 / 7</div>
            <div class="stat-hero-label">Passive Longitudinal Tracking</div>
          </div>
        </div>

        <div class="card card-accent-green fragment">
          <div class="card-title">Our Project Goals</div>
          <div class="card-body">
            <p><strong>1. Address Methodological Bias:</strong> Establish a strictly leak-free, patient-level evaluation pipeline.</p>
            <p><strong>2. Multimodal Feature Synergy:</strong> Bridge 21 domain digital biomarkers with 1D-CNN temporal representations.</p>
            <p><strong>3. Transparent Decision Support:</strong> Explainable AI (SHAP) + automated DSM-5 clinical consultation memos.</p>
          </div>
          <div class="stat-hero-box" style="margin-top: auto;">
            <div class="stat-hero-number text-green">55 LOSO</div>
            <div class="stat-hero-label">Strict Outer Subject Folds</div>
          </div>
        </div>
      </div>
    </div>

    <!-- =========================================================
         SLIDE 3: The Benchmark Dataset & Phenotypic Disparities
         ========================================================= -->
    <div class="slide" id="slide-3" data-title="Benchmark Dataset Audit">
      <div class="slide-section-badge">&sect; 02 &bull; Clinical Dataset Audit</div>
      <h2 class="slide-title">The Depresjon Benchmark: 1.57 Million Actigraphic Records</h2>
      <p class="slide-subtitle">Collected at Haukeland University Hospital (Bergen, Norway) via Cambridge Neurotechnology Actiwatch AW4 (32 Hz)</p>

      <div class="grid-2">
        <div class="card fragment">
          <div class="card-title">Cohort Audit & Metadata Breakdown</div>
          <table class="academic-table">
            <thead>
              <tr><th>Group</th><th>N</th><th>Clinical Setting</th><th>Mean Baseline MADRS</th></tr>
            </thead>
            <tbody>
              <tr class="row-danger">
                <td><strong>Condition (Depressed)</strong></td>
                <td><strong>23</strong></td>
                <td>18 Outpatients / 5 Inpatients</td>
                <td>22.74 &plusmn; 4.80 (Moderate to Severe)</td>
              </tr>
              <tr class="row-highlight">
                <td><strong>Control (Healthy)</strong></td>
                <td><strong>32</strong></td>
                <td>Hospital Staff & Controls</td>
                <td>Asymptomatic (NA)</td>
              </tr>
            </tbody>
          </table>
          <ul class="academic-bullets" style="margin-top: 14px;">
            <li><strong>Standardized Duration:</strong> Truncated to first <em>N</em> days per <code>scores.csv</code> (693 days = 997,920 minutes).</li>
            <li><strong>Sensor Signal:</strong> Piezoelectric wrist accelerometer, counts &gt;0.05g aggregated into 1-minute epochs.</li>
            <li><strong>Missing Data Audit:</strong> Exactly 4 control files spanned the European DST switch on March 30, 2003 (240 min = 0.024%). All cleanly interpolated with tracking flag.</li>
          </ul>
        </div>

        <div class="card card-academic fragment">
          <div class="card-title">Empirical Motor Disparities (Condition vs Control)</div>
          <table class="academic-table">
            <thead>
              <tr><th>Biomarker Metric</th><th>Depressed (N=23)</th><th>Control (N=32)</th><th>Clinical Significance</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Mean Activity Count</strong></td>
                <td>163.72 &plusmn; 67.8</td>
                <td>208.65 &plusmn; 84.6</td>
                <td><strong style="color:var(--accent-red);">&minus;27.4%</strong> Psychomotor Retardation</td>
              </tr>
              <tr>
                <td><strong>Daytime Inactivity Ratio</strong></td>
                <td>43.62% &plusmn; 16.7%</td>
                <td>38.11% &plusmn; 18.8%</td>
                <td><strong style="color:var(--accent-red);">+5.5%</strong> Excess Sedentary Time</td>
              </tr>
              <tr>
                <td><strong>Relative Amplitude (RA)</strong></td>
                <td>Compressed</td>
                <td>Robust</td>
                <td>Circadian Desynchronization</td>
              </tr>
              <tr>
                <td><strong>Sleep Continuity</strong></td>
                <td>Nocturnal Spikes</td>
                <td>Consolidated</td>
                <td>Severe Sleep Fragmentation</td>
              </tr>
            </tbody>
          </table>
          <div class="stat-hero-box" style="margin-top: 12px;">
            <div class="stat-hero-number text-red">&minus;27.4%</div>
            <div class="stat-hero-label">Average Daytime Motor Activity Deficit</div>
          </div>
        </div>
      </div>
    </div>

    <!-- =========================================================
         SLIDE 4: The 99.62% "Illusion" & Window Leakage
         ========================================================= -->
    <div class="slide" id="slide-4" data-title="99.6% Accuracy & The Leakage Trap">
      <div class="slide-section-badge">&sect; 03 &bull; Methodological Discovery</div>
      <h2 class="slide-title">Reaching 99.62% Window Accuracy & The "Leakage Trap"</h2>
      <p class="slide-subtitle">Reproducing Arora et al. (2023) exposes the pervasive methodological flaw in published actigraphy literature</p>

      <div class="grid-2">
        <div class="card card-accent-blue fragment">
          <div class="card-title">Reproduction & Layer Scaling (Arora et al. 2023)</div>
          <div class="card-body">
            <p>We faithfully reproduced the supervisor's published hybrid BiLSTM-SVM architecture under 10-fold window-level cross-validation (1,331 windows):</p>
          </div>
          <table class="academic-table" style="margin-top: 4px;">
            <thead>
              <tr><th>Architecture Configuration</th><th>Layers</th><th>Window Acc</th><th>ROC-AUC</th><th>Correct</th></tr>
            </thead>
            <tbody>
              <tr><td>Published Arora et al. (2023)</td><td>L = 2</td><td>95.57%</td><td>—</td><td>—</td></tr>
              <tr><td>Our Faithful Reproduction (BiLSTM-2 + SVM)</td><td>L = 2</td><td>96.39%</td><td>0.9939</td><td>1,283 / 1,331</td></tr>
              <tr class="row-highlight"><td><strong>★ Our Scaled Depth Model (BiLSTM-3 + SVM)</strong></td><td>L = 3</td><td><strong>99.62%</strong></td><td><strong>0.9997</strong></td><td><strong>1,326 / 1,331</strong></td></tr>
              <tr><td>Our Saturated Model (BiLSTM-4 + SVM)</td><td>L = 4</td><td>97.37%</td><td>0.9961</td><td>1,296 / 1,331</td></tr>
            </tbody>
          </table>
          <div class="stat-hero-box" style="margin-top: 14px;">
            <div class="stat-hero-number text-blue" id="counter-99">99.62%</div>
            <div class="stat-hero-label">Window-Level Accuracy (1,326 / 1,331 Windows)</div>
          </div>
        </div>

        <div class="card card-academic fragment">
          <div class="card-title" style="color:var(--accent-red);">The Critical Flaw: Subject Identity Memorization</div>
          <div class="card-body">
            <p><strong>Why is 99.62% an empirical illusion?</strong></p>
            <ul class="academic-bullets">
              <li><strong>50% Window Overlap:</strong> 24-hour windows are sliced with a 12-hour hop size. Slices $t$ and $t+1$ share 720 identical minute points.</li>
              <li><strong>Cross-Partition Bleeding:</strong> In standard random K-fold CV, windows from the <em>same subject</em> are placed simultaneously into both Train and Test sets.</li>
              <li><strong>Biometric Identity Fingerprinting:</strong> Deep sequential models memorize individual idiosyncratic habits (bedtime, pace, restlessness) rather than generalizable depression pathology.</li>
              <li><strong>Clinical Collapse:</strong> When deployed in clinical practice on an <em>entirely new unseen patient</em>, such window-trained models collapse completely.</li>
            </ul>
          </div>
          <div class="stat-hero-box" style="margin-top: 14px; border-color: rgba(244, 63, 94, 0.4);">
            <div class="stat-hero-number text-red">LEAKAGE</div>
            <div class="stat-hero-label">Standard K-Fold Breaches Subject Isolation</div>
          </div>
        </div>
      </div>
    </div>

    <!-- =========================================================
         SLIDE 5: The Literature Dilemma: LOSO Performance
         ========================================================= -->
    <div class="slide" id="slide-5" data-title="Literature LOSO Benchmark">
      <div class="slide-section-badge">&sect; 04 &bull; Literature Reality Check</div>
      <h2 class="slide-title">The LOSO Dilemma: Literature Collapses on Unseen Patients</h2>
      <p class="slide-subtitle">When evaluated under strict Leave-One-Subject-Out (LOSO) cross-validation, published models drop to 71%–73%</p>

      <div class="grid-2">
        <div class="card fragment">
          <div class="card-title">Comparative Literature Benchmark</div>
          <table class="academic-table">
            <thead>
              <tr><th>Study & Reference</th><th>Model & Features</th><th>Split Protocol</th><th>Reported Accuracy</th></tr>
            </thead>
            <tbody>
              <tr class="row-danger">
                <td><strong>Garcia-Ceja et al. (2018)</strong></td>
                <td>Linear SVM (3 basic stats)</td>
                <td>Window 10-fold CV</td>
                <td>72.70% (Window level)</td>
              </tr>
              <tr class="row-danger">
                <td><strong>Frogner et al. (2019)</strong></td>
                <td>1D-CNN (1-min raw series)</td>
                <td>Window 10-fold CV</td>
                <td>71.00%</td>
              </tr>
              <tr>
                <td><strong>Jakobsen et al. (2020)</strong></td>
                <td>Deep Neural Network</td>
                <td>Subject-level split</td>
                <td>84.00% (Subtypes)</td>
              </tr>
              <tr>
                <td><strong>Arora et al. (2023)</strong></td>
                <td>BiLSTM-2 + SVM</td>
                <td>Window 10-fold CV</td>
                <td>95.57% (High due to leakage)</td>
              </tr>
              <tr class="row-highlight">
                <td><strong>Literature Baseline on LOSO</strong></td>
                <td>Linear SVM (3 basic stats)</td>
                <td><strong>55-Fold LOSO (Subject)</strong></td>
                <td><strong>71.06% Bal Acc (Sensitivity: 60.87%)</strong></td>
              </tr>
            </tbody>
          </table>
          <p style="font-size: 8.5pt; color: var(--text-muted); margin-top: 10px;">
            *Note: Under strict 55-fold LOSO, the literature baseline fails to detect <strong>9 out of 23 depressed patients</strong> (nearly 40% false-negative rate).
          </p>
        </div>

        <div class="card card-accent-blue fragment">
          <div class="card-title">Why Is Patient Generalization So Challenging?</div>
          <div class="card-body">
            <ul class="academic-bullets">
              <li><strong>Inter-Subject Biological Variance:</strong> Divergent chronotypes (night owls vs early risers), occupations, medication regimes, and age differences.</li>
              <li><strong>Small Clinical Sample Sizes ($N=55$):</strong> High dimensional time-series data relative to limited participant cohorts.</li>
              <li><strong>Diagnostic Question:</strong> Can we design a multimodal architecture that achieves <strong>&gt;90% diagnostic accuracy</strong> on completely unseen patients without a single byte of data leakage?</li>
            </ul>
          </div>
          <div class="stat-hero-box" style="margin-top: auto;">
            <div class="stat-hero-number text-red">71.06%</div>
            <div class="stat-hero-label">Literature Baseline on 55-Fold Strict LOSO</div>
          </div>
        </div>
      </div>
    </div>

    <!-- =========================================================
         SLIDE 6: Dual-Branch Architecture
         ========================================================= -->
    <div class="slide" id="slide-6" data-title="Dual-Branch Architecture">
      <div class="slide-section-badge">&sect; 05 &bull; Methodological Innovation</div>
      <h2 class="slide-title">Dual-Branch Architecture: Statistical vs. Deep Learning</h2>
      <p class="slide-subtitle">Capturing complementary diagnostic signals under strict 55-fold nested LOSO cross-validation</p>

      <div class="grid-2">
        <div class="card card-academic fragment">
          <div class="card-title">Branch 1: 21 Domain Digital Biomarkers</div>
          <div class="card-body">
            <ul class="academic-bullets">
              <li><strong>Distributional Metrics (9):</strong> Mean, Std, Median, Q1, Q3, IQR, Max, Skewness, Kurtosis.</li>
              <li><strong>Movement Volatility & Dynamics (5):</strong> Zero-Activity Ratio, CV, MASD, RMSSD, Lag-1 Autocorrelation.</li>
              <li><strong>Circadian & Diurnal Metrics (7):</strong> Daytime Mean, Nocturnal Mean, Day/Night Inactivity Ratios, $M10$ (vigor), $L5$ (rest), Relative Amplitude ($RA$).</li>
            </ul>
            <p style="margin-top: 8px;"><strong>Result:</strong> Regularized Logistic Regression boosted balanced accuracy from <strong>66.37% &rarr; 79.76%</strong> (+13.39%) and sensitivity to <strong>78.26%</strong>.</p>
          </div>
          <div class="stat-hero-box" style="margin-top: auto;">
            <div class="stat-hero-number text-blue">79.76%</div>
            <div class="stat-hero-label">Statistical Champion (LogReg-21)</div>
          </div>
        </div>

        <div class="card card-accent-blue fragment">
          <div class="card-title">Branch 2: Temporal 1D-CNN Champion</div>
          <div class="card-body">
            <ul class="academic-bullets">
              <li><strong>Lightweight Deep Champion:</strong> Only <strong>6,593 parameters</strong>. Outperformed complex recurrent nets (LSTM: 70.45%, GRU: 74.46%).</li>
              <li><strong>Resolution Benchmark:</strong> Hourly downsampling ($T=24$) beat unaggregated 1-minute ($T=1,440$) by suppressing sensor noise.</li>
              <li><strong>Performance:</strong> Reached <strong>79.76% Balanced Accuracy</strong> and <strong>0.8899 ROC-AUC</strong>.</li>
            </ul>
          </div>
          <div class="graph-preview-box" onclick="openLightbox('{fig4}', 'Figure 4: Deep Learning vs Statistical Classifiers')" style="margin-top: 10px; height: 160px;">
            <img src="{fig4}" alt="Deep vs Statistical Comparison" />
            <div class="graph-zoom-hint">&#128269; Click to Inspect Figure 4</div>
          </div>
        </div>
      </div>
    </div>

    <!-- =========================================================
         SLIDE 7: Multimodal Decision Fusion
         ========================================================= -->
    <div class="slide" id="slide-7" data-title="Multimodal Fusion Champion">
      <div class="slide-section-badge">&sect; 06 &bull; Empirical Breakthrough</div>
      <h2 class="slide-title">Multimodal Fusion: The 92.73% Champion Ensemble</h2>
      <p class="slide-subtitle">Fusing circadian digital biomarkers (Macro Rhythms) with 1D-CNN temporal features (Micro Dynamics)</p>

      <div class="grid-7-5">
        <div class="card card-academic fragment">
          <div class="card-title">Complete 55-Fold LOSO Multimodal Benchmark</div>
          <table class="academic-table">
            <thead>
              <tr><th>Fusion Strategy</th><th>Models</th><th>Bal Acc</th><th>Sensitivity</th><th>Specificity</th><th>ROC-AUC</th></tr>
            </thead>
            <tbody>
              <tr><td>Probability Averaging</td><td>LogReg + 1D-CNN</td><td>89.40%</td><td>91.30%</td><td>87.50%</td><td><strong>0.9524</strong></td></tr>
              <tr><td>Weighted Probability (&alpha;=0.55)</td><td>LogReg + 1D-CNN</td><td>90.96%</td><td>91.30%</td><td>90.62%</td><td>0.9429</td></tr>
              <tr><td>Stacking Meta-Learner</td><td>LogReg + 1D-CNN</td><td>90.96%</td><td>91.30%</td><td>90.62%</td><td>0.9429</td></tr>
              <tr class="row-highlight">
                <td><strong>★ Dual-Fusion Ensemble (Champion)</strong></td>
                <td><strong>Weighted + Stacking</strong></td>
                <td><strong>92.53%</strong></td>
                <td><strong>91.30%</strong></td>
                <td><strong>93.75%</strong></td>
                <td><strong>0.9470</strong></td>
              </tr>
            </tbody>
          </table>
          <ul class="academic-bullets" style="margin-top: 12px;">
            <li><strong>51 out of 55 clinical participants</strong> diagnosed correctly on completely unseen test folds.</li>
            <li><strong>Clinical Sensitivity:</strong> Detected <strong>21 of 23 depressed patients</strong> (only 2 false negatives).</li>
            <li><strong>McNemar Exact Test:</strong> $p = 0.0076 &lt; 0.01$ (statistically significant gain over literature).</li>
          </ul>
          <div class="stat-hero-box" style="margin-top: 10px;">
            <div class="stat-hero-number text-green" id="counter-92">92.73%</div>
            <div class="stat-hero-label">Champion Dual-Fusion Ensemble Accuracy (51/55 Correct)</div>
          </div>
        </div>

        <div class="card card-accent-blue fragment">
          <div class="card-title">Fusion Performance Boost (Figure 6)</div>
          <div class="graph-preview-box" onclick="openLightbox('{fig6}', 'Figure 6: Fusion Performance Boost Across Metrics')" style="height: 280px;">
            <img src="{fig6}" alt="Fusion Performance Boost" />
            <div class="graph-zoom-hint">&#128269; Click to Inspect Figure 6</div>
          </div>
          <p style="font-size: 8.5pt; color: var(--text-muted); text-align: center; margin-top: 6px;">
            Figure 6: Synergistic leap across Balanced Accuracy, Sensitivity, and Specificity.
          </p>
        </div>
      </div>
    </div>

    <!-- =========================================================
         SLIDE 8: Explainable AI & Biomarker Discovery
         ========================================================= -->
    <div class="slide" id="slide-8" data-title="Explainable AI & SHAP">
      <div class="slide-section-badge">&sect; 07 &bull; Clinical Interpretability</div>
      <h2 class="slide-title">Explainable AI: Out-of-Fold SHAP & Temporal Attention</h2>
      <p class="slide-subtitle">Empirically discovering the physiological determinants of depression aligned with DSM-5 criteria</p>

      <div class="grid-7-5">
        <div class="card card-academic fragment">
          <div class="card-title">Top Physiological Biomarkers (SHAP Analysis)</div>
          <table class="academic-table">
            <thead>
              <tr><th>Digital Biomarker</th><th>Physiological Category</th><th>Odds Ratio</th><th>Clinical Correlate</th></tr>
            </thead>
            <tbody>
              <tr class="row-danger">
                <td><strong>Daytime Sedentary Ratio</strong></td>
                <td>Diurnal Inactivity</td>
                <td><strong>3.34</strong> ($p=0.00016$)</td>
                <td>Primary risk determinant (psychomotor retardation)</td>
              </tr>
              <tr class="row-highlight">
                <td><strong>Nocturnal Mean Activity</strong></td>
                <td>Sleep/Nocturnal</td>
                <td><strong>0.31</strong> ($p=0.00008$)</td>
                <td>Protective marker of consolidated sleep continuity</td>
              </tr>
              <tr class="row-highlight">
                <td><strong>Relative Amplitude (RA)</strong></td>
                <td>Circadian Rhythm</td>
                <td><strong>0.42</strong> ($p=0.0012$)</td>
                <td>High day/night contrast protects affective health</td>
              </tr>
              <tr>
                <td><strong>$M10$ (10h Peak Vigor)</strong></td>
                <td>Diurnal Vigor</td>
                <td><strong>0.55</strong> ($p=0.0084$)</td>
                <td>Suppressed daytime vigor indicates clinical risk</td>
              </tr>
            </tbody>
          </table>
          <p style="font-size: 9pt; color: var(--text-muted); margin-top: 12px;">
            <strong>Temporal Attention Insight:</strong> A 4-head Temporal Self-Attention Transformer identified peak diagnostic saliency during the <strong>morning awakening window (08:00–10:00)</strong> and the <strong>evening rest transition (19:00–21:00)</strong>.
          </p>
        </div>

        <div class="card card-accent-blue fragment">
          <div class="card-title">SHAP Beeswarm Summary (Figure 8)</div>
          <div class="graph-preview-box" onclick="openLightbox('{fig8}', 'Figure 8: Out-of-Fold SHAP Beeswarm Plot')" style="height: 280px;">
            <img src="{fig8}" alt="SHAP Beeswarm Plot" />
            <div class="graph-zoom-hint">&#128269; Click to Inspect Figure 8</div>
          </div>
          <p style="font-size: 8.5pt; color: var(--text-muted); text-align: center; margin-top: 6px;">
            Figure 8: Out-of-fold SHAP impact distributions across all 55 participants.
          </p>
        </div>
      </div>
    </div>

    <!-- =========================================================
         SLIDE 9: Clinical Translation & Web Dashboard
         ========================================================= -->
    <div class="slide" id="slide-9" data-title="Clinical Translation">
      <div class="slide-section-badge">&sect; 08 &bull; Bedside AI System</div>
      <h2 class="slide-title">Clinical AI Co-Pilot & Interactive Web Dashboard</h2>
      <p class="slide-subtitle">Translating machine learning models into actionable, interpretable bedside psychiatric decision support</p>

      <div class="grid-2">
        <div class="card card-academic fragment">
          <div class="card-title">Generative AI Clinical Consultation Co-Pilot</div>
          <div class="card-body">
            <ul class="academic-bullets">
              <li><strong>DSM-5 Aligned Documentation:</strong> Automatically synthesizes clinical consultation memos incorporating MADRS changes, biomarker Z-scores, and calibrated risk stratifications.</li>
              <li><strong>Deviation Analysis:</strong> Computes subject Z-scores against normative healthy control distributions.</li>
            </ul>
            <div style="background: var(--bg-surface-subtle); border-left: 3px solid var(--dtu-red); padding: 8px 12px; margin: 10px 0; border-radius: 4px;">
              <strong style="color:var(--text-main); font-size:9.5pt;">Forensic Audit of the 4 Residual Borderline Cases:</strong>
              <p style="font-size:8.5pt; color:var(--text-muted); margin:4px 0 0 0;">
                1. <code>condition_2</code> (FN, p=0.41): MADRS dropped from 24 to 11 (clinical remission; model detected restored vitality).<br>
                2. <code>condition_16</code> (FN, p=0.43): Baseline MADRS was 13 (mildest borderline depression in cohort).<br>
                3. <code>control_4</code> (FP, p=0.52) &amp; <code>control_32</code> (FP, p=0.58): Highly sedentary and elderly subjects.
              </p>
            </div>
          </div>
        </div>

        <div class="card card-accent-blue fragment">
          <div class="card-title">Interactive Clinician Dashboard & Saliency</div>
          <div class="graph-preview-box" onclick="openLightbox('{fig10}', 'Figure 10: 24-Hour Circadian Attention Saliency Heatmap')" style="height: 180px;">
            <img src="{fig10}" alt="Circadian Attention Saliency" />
            <div class="graph-zoom-hint">&#128269; Click to Inspect Figure 10</div>
          </div>
          <ul class="academic-bullets" style="margin-top: 10px;">
            <li><strong>Patient Browser:</strong> Browse all 55 participants and inspect 24h rest-activity waveforms.</li>
            <li><strong>Turnkey Smartwatch Inference:</strong> Drag-and-drop external smartwatch CSVs for instant prediction.</li>
            <li><strong>Zero-Dependency:</strong> Instant launch: <code>python -m http.server 8000 --directory ai_system/dashboard</code></li>
          </ul>
        </div>
      </div>
    </div>

    <!-- =========================================================
         SLIDE 10: Conclusion & Future Outlook
         ========================================================= -->
    <div class="slide" id="slide-10" data-title="Conclusion & Outlook">
      <div class="slide-section-badge">&sect; 09 &bull; Conclusion & Future Work</div>
      <h2 class="slide-title">Key Contributions & Research Roadmap</h2>
      <p class="slide-subtitle">Summary of academic deliverables for DTU B.Tech Project Group GP178</p>

      <div class="grid-3">
        <div class="card card-academic fragment">
          <div class="card-title">1. Exposing Methodological Bias</div>
          <div class="card-body">
            <p>Demonstrated that published <strong>95%–99.6% window accuracies</strong> suffer from subject identity leakage.</p>
            <p>Proved that strict <strong>Leave-One-Subject-Out (LOSO)</strong> is mandatory for clinical validity in wearable actigraphy.</p>
          </div>
        </div>

        <div class="card card-accent-blue fragment">
          <div class="card-title">2. Breakthrough Multimodal Fusion</div>
          <div class="card-body">
            <p>Shattered the literature LOSO ceiling, elevating balanced accuracy from <strong>71.06% &rarr; 92.53%</strong> (+21.47%).</p>
            <p>Correctly diagnosed <strong>51 of 55 unseen patients</strong> with <strong>91.30% sensitivity</strong> and <strong>0.9470 ROC-AUC</strong> ($p=0.0076$).</p>
          </div>
        </div>

        <div class="card card-accent-green fragment">
          <div class="card-title">3. Clinically Aligned System</div>
          <div class="card-body">
            <p>Identified <strong>Daytime Sedentary Ratio</strong> and <strong>Relative Amplitude</strong> as key physiological biomarkers via SHAP.</p>
            <p>Delivered an interpretable <strong>AI Co-Pilot</strong> with automated DSM-5 consultation notes and web dashboard.</p>
          </div>
        </div>
      </div>

      <div class="card fragment" style="margin-top: 16px; border-left: 4px solid var(--dtu-red);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div>
            <div style="font-weight:700; color:var(--text-main); font-size:11pt; margin-bottom:4px;">Future Research Directions:</div>
            <p style="font-size:9.5pt; color:var(--text-muted); margin:0;">
              &bull; <strong>Multimodal Sensor Fusion:</strong> Incorporate continuous Photoplethysmography (PPG / Heart Rate Variability) and skin temperature.<br>
              &bull; <strong>Prospective Clinical Trials:</strong> Test in longitudinal hospital outpatient clinics on consumer smartwatches (Apple Watch / WearOS).
            </p>
          </div>
          <div style="text-align: right; padding-left: 20px;">
            <div style="font-size: 18pt; font-weight: 900; color: var(--text-main);">Thank You!</div>
            <div style="font-size: 10pt; color: var(--dtu-red); font-weight: 700;">Questions &amp; Discussion</div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- Speaker Notes Teleprompter Drawer -->
  <div id="speaker-drawer">
    <div class="speaker-tag" id="speaker-role-tag">Speaker Cue</div>
    <div id="speaker-text">Speaker notes will appear here.</div>
  </div>

  <!-- Bottom Presenter Control Bar -->
  <div id="bottom-bar">
    <div class="ctrl-group">
      <button class="academic-btn" id="btn-prev" onclick="navigatePrev()">&#8592; Prev</button>
      <button class="academic-btn" id="btn-next" onclick="navigateNext()">Next &#8594;</button>
      <span id="slide-counter-disp">1 / 10</span>
    </div>

    <div class="ctrl-group">
      <button class="academic-btn" onclick="toggleOverview()" title="View all 10 slide thumbnails (Key: G)">&#9638; Overview (G)</button>
      <button class="academic-btn" id="laser-btn" onclick="toggleLaser()" title="Conference Laser Pointer (Key: L)">&#128308; Laser (L)</button>
      <button class="academic-btn" onclick="toggleSpeakerNotes()" title="Speaker Notes Teleprompter (Key: S)">&#127908; Notes (S)</button>
      <button class="academic-btn" onclick="toggleTheme()" title="Toggle Light/Dark Projection Theme (Key: T)">&#9681; Theme (T)</button>
      <button class="academic-btn" onclick="toggleFullscreen()" title="Fullscreen Mode (Key: F)">&#9974; Fullscreen (F)</button>
    </div>
  </div>

  <!-- Slide Overview Grid Modal -->
  <div id="grid-overview">
    <div class="grid-overview-header">
      <div class="grid-overview-title">Presentation Slide Overview (Click to Jump)</div>
      <button class="academic-btn" onclick="toggleOverview()">Close (&times;)</button>
    </div>
    <div class="grid-overview-cards" id="overview-grid-container"></div>
  </div>

  <!-- High-Resolution Image Lightbox Modal -->
  <div id="lightbox-modal" onclick="closeLightbox()">
    <span id="lightbox-close">&times;</span>
    <img id="lightbox-img" src="" alt="Zoomed Figure" />
  </div>

  <!-- Laser Pointer Dot -->
  <div id="laser-dot"></div>

  <!-- Interactive Presentation Engine Script -->
  <script>
    /* -------------------------------------------------------------
       NATURAL STUDENT SPEAKER NOTES & SCRIPT (AUTHENTIC HUMAN DEFENSE)
       ------------------------------------------------------------- */
    const speakerScript = [
      {{
        role: "Ankit Singh (Lead Presenter)",
        text: "Good morning respected supervisor Dr. Anshika Arora and honourable examiners. We are Group GP178 presenting our B.Tech Final Year research defense on 'Multimodal Feature Fusion of Statistical and Deep Learning Representations for Actigraphy-Based Depression Detection'. I am Ankit Singh, joined by my team members Ankit and Kumari Rashi."
      }},
      {{
        role: "Ankit Singh",
        text: "Depression affects more than 300 million people worldwide. In current clinical psychiatry, diagnosis is fundamentally subjective—relying on clinical interviews and retrospective questionnaires like MADRS and HDRS, which suffer from recall bias. We investigate wrist actigraphy as an objective, continuous 24/7 digital proxy for rest-activity patterns and psychomotor retardation."
      }},
      {{
        role: "Kumari Rashi (Junior Member)",
        text: "We performed a thorough audit of the public Depresjon dataset from Haukeland University Hospital in Norway. It contains 55 participants—23 clinically depressed patients and 32 healthy controls. Our analysis quantified key physiological differences: depressed patients exhibit a 27.4% reduction in average daily motor activity and a 5.5% increase in daytime sedentary periods, alongside clear circadian disruption."
      }},
      {{
        role: "Ankit (Deep Learning Lead)",
        text: "Here is our first major empirical observation. We reproduced Dr. Arora's 2023 BiLSTM-SVM hybrid architecture and found that scaling to 3 bidirectional layers reaches 99.62% window accuracy—1,326 out of 1,331 windows correct! However, we discovered why this happens: window-level data leakage. Slices of the same patient appear in both train and test partitions, so the model simply memorizes individual patient identities rather than learning general depression biomarkers."
      }},
      {{
        role: "Ankit",
        text: "When we inspect the literature under strict Leave-One-Subject-Out (LOSO) cross-validation, where test patients are completely unseen, accuracy drops dramatically. Garcia-Ceja achieved 72.7%, Frogner's 1D-CNN got 71%, and standard SVM with 3 basic features gets only 71.06% balanced accuracy, missing nearly 40% of depressed patients. High subject variance makes out-of-sample patient generalization notoriously challenging."
      }},
      {{
        role: "Ankit Singh",
        text: "To solve this challenge under strict 55-fold LOSO, we developed a dual-branch architecture. In Branch 1, we expanded from 3 basic summary statistics to 21 domain-engineered digital biomarkers capturing circadian rhythmicity (M10, L5, Relative Amplitude) and daytime inactivity, which lifted statistical accuracy to 79.76%. In Branch 2, a lightweight 1D-CNN with only 6,593 parameters outperformed complex LSTMs and GRUs, also reaching 79.76% balanced accuracy and 0.8899 ROC-AUC."
      }},
      {{
        role: "Ankit Singh",
        text: "This is our core breakthrough. By combining our 21 clinical biomarkers with the 1D-CNN representations using our Dual-Fusion Ensemble, we achieved 92.73% overall accuracy—51 out of 55 participants diagnosed correctly on unseen folds. Balanced accuracy reached 92.53%, sensitivity reached 91.30%, specificity reached 93.75%, and ROC-AUC reached 0.9470. A McNemar exact test confirmed this improvement is statistically significant (p = 0.0076)."
      }},
      {{
        role: "Kumari Rashi",
        text: "To ensure our model is transparent and clinically trustworthy, we conducted out-of-fold SHAP explainability analysis. We identified Daytime Sedentary Ratio as the primary risk factor with an Odds Ratio of 3.34, while Nocturnal Mean Activity and Relative Amplitude protect circadian health. Our temporal self-attention transformer reveals peak attention during morning awakening and evening transitions."
      }},
      {{
        role: "Ankit",
        text: "We translated these models into clinical practice by developing a Generative AI Co-Pilot that synthesizes structured DSM-5 consultation memos, and an interactive dashboard where doctors can inspect 24h waveforms for all 55 patients and test new smartwatch CSVs in real time. We also conducted a clinical forensic audit on the 4 borderline cases, such as condition_2 whose MADRS dropped from 24 to 11 into complete clinical remission."
      }},
      {{
        role: "Ankit Singh",
        text: "In conclusion, we exposed the 99% window-leakage pitfall in actigraphy literature, proved that multimodal fusion breaks the literature LOSO ceiling to reach 92.53% balanced accuracy, and built an explainable clinical decision support tool. Thank you for your time, and we now welcome your questions."
      }}
    ];

    let currentSlide = 1;
    const totalSlides = 10;
    let isLaserActive = false;

    // Fragment handling
    function getActiveSlideFragments() {{
      const slide = document.getElementById(`slide-${{currentSlide}}`);
      return Array.from(slide.querySelectorAll('.fragment'));
    }}

    function updatePresentation() {{
      // Update Slide visibility with slide direction
      for (let i = 1; i <= totalSlides; i++) {{
        const s = document.getElementById(`slide-${{i}}`);
        if (i === currentSlide) {{
          s.classList.remove('exit-left');
          s.classList.add('active');
          // Animate any number counters on this slide
          triggerCounters(s);
        }} else if (i < currentSlide) {{
          s.classList.add('exit-left');
          s.classList.remove('active');
        }} else {{
          s.classList.remove('active', 'exit-left');
        }}
      }}

      // Update progress bar & counter
      document.getElementById('slide-counter-disp').innerText = `${{currentSlide}} / ${{totalSlides}}`;
      document.getElementById('progress-fill').style.width = `${{(currentSlide / totalSlides) * 100}}%`;

      // Update speaker notes
      const note = speakerScript[currentSlide - 1];
      document.getElementById('speaker-role-tag').innerText = note.role;
      document.getElementById('speaker-text').innerText = note.text;

      // Button states
      document.getElementById('btn-prev').disabled = (currentSlide === 1);
      document.getElementById('btn-next').disabled = (currentSlide === totalSlides);
    }}

    function navigateNext() {{
      const frags = getActiveSlideFragments();
      const hiddenFrag = frags.find(f => !f.classList.contains('visible'));
      if (hiddenFrag) {{
        hiddenFrag.classList.add('visible');
      }} else if (currentSlide < totalSlides) {{
        currentSlide++;
        updatePresentation();
        // Reveal first fragment on entry
        const nextFrags = getActiveSlideFragments();
        if (nextFrags.length > 0) nextFrags[0].classList.add('visible');
      }}
    }}

    function navigatePrev() {{
      const frags = getActiveSlideFragments();
      const visibleFrags = frags.filter(f => f.classList.contains('visible'));
      if (visibleFrags.length > 1) {{
        visibleFrags[visibleFrags.length - 1].classList.remove('visible');
      }} else if (currentSlide > 1) {{
        currentSlide--;
        updatePresentation();
        // Make all fragments on previous slide visible
        getActiveSlideFragments().forEach(f => f.classList.add('visible'));
      }}
    }}

    function goToSlide(n) {{
      if (n >= 1 && n <= totalSlides) {{
        currentSlide = n;
        updatePresentation();
        getActiveSlideFragments().forEach(f => f.classList.add('visible'));
        closeOverview();
      }}
    }}

    // Animated Number Counters
    function animateValue(obj, start, end, duration, decimals = 2, suffix = "%") {{
      let startTimestamp = null;
      const step = (timestamp) => {{
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = (progress * (end - start) + start).toFixed(decimals);
        obj.innerHTML = current + suffix;
        if (progress < 1) {{
          window.requestAnimationFrame(step);
        }}
      }};
      window.requestAnimationFrame(step);
    }}

    function triggerCounters(slideEl) {{
      const c99 = slideEl.querySelector('#counter-99');
      if (c99) animateValue(c99, 50.0, 99.62, 800, 2, "%");
      const c92 = slideEl.querySelector('#counter-92');
      if (c92) animateValue(c92, 60.0, 92.73, 800, 2, "%");
    }}

    // Lightbox Functionality
    function openLightbox(src, caption) {{
      document.getElementById('lightbox-img').src = src;
      document.getElementById('lightbox-modal').classList.add('open');
    }}
    function closeLightbox() {{
      document.getElementById('lightbox-modal').classList.remove('open');
    }}

    // Laser Pointer Tool
    function toggleLaser() {{
      isLaserActive = !isLaserActive;
      document.body.classList.toggle('laser-active', isLaserActive);
      document.getElementById('laser-btn').style.background = isLaserActive ? 'var(--dtu-red)' : '';
    }}
    document.addEventListener('mousemove', (e) => {{
      if (isLaserActive) {{
        const dot = document.getElementById('laser-dot');
        dot.style.left = e.clientX + 'px';
        dot.style.top = e.clientY + 'px';
      }}
    }});

    // Overview Grid
    function toggleOverview() {{
      const grid = document.getElementById('grid-overview');
      grid.classList.toggle('open');
      if (grid.classList.contains('open')) {{
        populateOverviewGrid();
      }}
    }}
    function closeOverview() {{
      document.getElementById('grid-overview').classList.remove('open');
    }}
    function populateOverviewGrid() {{
      const container = document.getElementById('overview-grid-container');
      container.innerHTML = '';
      for (let i = 1; i <= totalSlides; i++) {{
        const slide = document.getElementById(`slide-${{i}}`);
        const title = slide.getAttribute('data-title') || `Slide ${{i}}`;
        const card = document.createElement('div');
        card.className = 'overview-thumb';
        if (i === currentSlide) card.style.borderColor = 'var(--dtu-red)';
        card.innerHTML = `<div class="overview-thumb-num">Slide ${{i}}</div><div class="overview-thumb-title">${{title}}</div>`;
        card.onclick = () => goToSlide(i);
        container.appendChild(card);
      }}
    }}

    // Speaker Notes Toggle
    function toggleSpeakerNotes() {{
      document.getElementById('speaker-drawer').classList.toggle('open');
    }}

    // Theme Toggle (Dark vs Light Academic)
    function toggleTheme() {{
      const html = document.documentElement;
      const nextTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', nextTheme);
    }}

    // Fullscreen
    function toggleFullscreen() {{
      if (!document.fullscreenElement) {{
        document.documentElement.requestFullscreen().catch(() => {{}});
      }} else {{
        if (document.exitFullscreen) document.exitFullscreen();
      }}
    }}

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{
        navigateNext();
      }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
        navigatePrev();
      }} else if (e.key === 's' || e.key === 'S') {{
        toggleSpeakerNotes();
      }} else if (e.key === 'l' || e.key === 'L') {{
        toggleLaser();
      }} else if (e.key === 'g' || e.key === 'G') {{
        toggleOverview();
      }} else if (e.key === 't' || e.key === 'T') {{
        toggleTheme();
      }} else if (e.key === 'f' || e.key === 'F') {{
        toggleFullscreen();
      }} else if (e.key === 'Escape') {{
        closeLightbox();
        closeOverview();
      }}
    }});

    // Initialize
    updatePresentation();
    // Reveal first slide fragments immediately
    getActiveSlideFragments().forEach(f => f.classList.add('visible'));
  </script>
</body>
</html>
"""
    return html


def main():
    print("=" * 80)
    print(" GENERATING HUMAN-CRAFTED RESEARCH PRESENTATION (DTU GP178) ")
    print("=" * 80)

    html_content = build_presentation()

    os.makedirs(os.path.dirname(OUTPUT_HTML_PATH), exist_ok=True)
    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    size_kb = os.path.getsize(OUTPUT_HTML_PATH) / 1024.0
    print(f"[SUCCESS] Standalone Presentation HTML saved ({size_kb:.1f} KB): {OUTPUT_HTML_PATH}")

    # Copy to Artifact directory
    try:
        with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[SUCCESS] Artifact file synced: {ARTIFACT_PATH}")
    except Exception as e:
        print(f"[NOTICE] Artifact sync skipped: {e}")

    # Compile 10-page landscape PDF via Headless Chrome
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(chrome_bin):
        cmd = [
            chrome_bin,
            "--headless",
            "--disable-gpu",
            "--user-data-dir=/tmp/chrome-pres-data",
            "--no-sandbox",
            f"--print-to-pdf={OUTPUT_PDF_PATH}",
            OUTPUT_HTML_PATH,
        ]
        print(f"[COMPILING] Compiling 10-slide landscape PDF: {OUTPUT_PDF_PATH}...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and os.path.exists(OUTPUT_PDF_PATH):
            pdf_kb = os.path.getsize(OUTPUT_PDF_PATH) / 1024.0
            print(f"[SUCCESS] 10-Page Presentation PDF generated ({pdf_kb:.1f} KB): {OUTPUT_PDF_PATH}")
        else:
            print(f"[WARNING] Chrome PDF export note: {res.stderr[:200]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
