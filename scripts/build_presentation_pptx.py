"""
Build a professional 10-slide PowerPoint presentation for BTP Group GP178.
Topic: Multimodal Feature Fusion of Statistical and Deep Learning Representations for Actigraphy-Based Depression Detection
Format: .pptx (16:9 Widescreen, White Background, High-Contrast Typography, Diagrams & Native Transitions/Animations)
Compatible: Cross-platform for macOS and Windows (PowerPoint, Keynote, Google Slides)
"""

import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml import parse_xml

# Color Definitions (Contrasting Modern Executive Palette on Pure White)
WHITE = RGBColor(255, 255, 255)
CARD_BG = RGBColor(248, 250, 252)       # Slate-50 ultra-light tint
CARD_BORDER = RGBColor(226, 232, 240)   # Slate-200 border
TEXT_TITLE = RGBColor(15, 23, 42)       # Slate-900 high contrast
TEXT_BODY = RGBColor(30, 41, 59)        # Slate-800
TEXT_MUTED = RGBColor(100, 116, 139)    # Slate-500
CRIMSON = RGBColor(153, 27, 27)         # DTU Crimson / Alert
CRIMSON_BG = RGBColor(254, 242, 242)    # Crimson-50 soft tint
BLUE = RGBColor(29, 78, 216)            # Royal Blue
BLUE_BG = RGBColor(239, 246, 255)       # Blue-50 soft tint
EMERALD = RGBColor(4, 120, 87)          # Emerald Champion
EMERALD_BG = RGBColor(236, 253, 245)    # Emerald-50 soft tint
AMBER = RGBColor(180, 83, 9)            # Amber Warning
AMBER_BG = RGBColor(254, 243, 199)      # Amber-50 soft tint
PURPLE = RGBColor(109, 40, 217)         # Deep Purple
PURPLE_BG = RGBColor(245, 243, 255)     # Purple-50 soft tint
SLATE_900 = TEXT_TITLE
SLATE_800 = TEXT_BODY
SLATE_500 = TEXT_MUTED


def set_white_background(slide):
    """Ensure the slide has a pure solid white background."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = WHITE


def add_slide_transition_and_animations(slide, shape_ids=None):
    """
    Applies smooth fade slide transition and sequential entrance animations
    conforming strictly to the OpenXML PresentationML schema.
    """
    sld = slide._element
    # Transition element
    trans_xml = '<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" spd="med"><p:fade/></p:transition>'
    trans = parse_xml(trans_xml)
    sld.append(trans)

    if not shape_ids:
        return

    pars = []
    blds = []
    node_id = 3
    for sp_id in shape_ids:
        blds.append(f'<p:bldP spid="{sp_id}" grpId="0" animBg="1"/>')
        par_item = f'''
        <p:par>
          <p:cTn id="{node_id}" fill="hold">
            <p:stCondLst><p:cond delay="0"/></p:stCondLst>
            <p:childTnLst>
              <p:par>
                <p:cTn id="{node_id+1}" fill="hold" nodeType="clickEffect">
                  <p:stCondLst><p:cond delay="0"/></p:stCondLst>
                  <p:childTnLst>
                    <p:set>
                      <p:cBhvr>
                        <p:cTn id="{node_id+2}" dur="1" fill="hold">
                          <p:stCondLst><p:cond delay="0"/></p:stCondLst>
                        </p:cTn>
                        <p:tgtEl><p:spTgt spid="{sp_id}"/></p:tgtEl>
                        <p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>
                      </p:cBhvr>
                      <p:to><p:strVal val="visible"/></p:to>
                    </p:set>
                    <p:animEffect transition="in" filter="fade">
                      <p:cBhvr>
                        <p:cTn id="{node_id+3}" dur="500">
                          <p:stCondLst><p:cond delay="0"/></p:stCondLst>
                        </p:cTn>
                        <p:tgtEl><p:spTgt spid="{sp_id}"/></p:tgtEl>
                      </p:cBhvr>
                    </p:animEffect>
                  </p:childTnLst>
                </p:cTn>
              </p:par>
            </p:childTnLst>
          </p:cTn>
        </p:par>'''
        pars.append(par_item)
        node_id += 4

    pars_str = '\n'.join(pars)
    blds_str = '\n'.join(blds)

    timing_xml = f'''<p:timing xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:tnLst>
    <p:par>
      <p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">
        <p:childTnLst>
          <p:seq concurrent="1" nextAc="seek">
            <p:cTn id="2" dur="indefinite" nodeType="mainSeq">
              <p:childTnLst>
                {pars_str}
              </p:childTnLst>
            </p:cTn>
          </p:seq>
        </p:childTnLst>
      </p:cTn>
    </p:par>
  </p:tnLst>
  <p:bldLst>
    {blds_str}
  </p:bldLst>
</p:timing>'''
    timing = parse_xml(timing_xml)
    sld.append(timing)


def add_slide_header(slide, badge_text, title_text, subtitle_text):
    """Standardized high-contrast slide header for Slides 2-10."""
    # Badge
    badge_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
    tf_b = badge_box.text_frame
    tf_b.word_wrap = True
    tf_b.margin_left = tf_b.margin_top = tf_b.margin_right = tf_b.margin_bottom = 0
    p_b = tf_b.paragraphs[0]
    p_b.text = badge_text.upper()
    p_b.font.name = "Arial"
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = CRIMSON

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.5))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.name = "Arial"
    p_t.font.size = Pt(20)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_TITLE

    # Subtitle
    sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.22), Inches(11.7), Inches(0.35))
    tf_s = sub_box.text_frame
    tf_s.word_wrap = True
    tf_s.margin_left = tf_s.margin_top = tf_s.margin_right = tf_s.margin_bottom = 0
    p_s = tf_s.paragraphs[0]
    p_s.text = subtitle_text
    p_s.font.name = "Arial"
    p_s.font.size = Pt(11)
    p_s.font.color.rgb = TEXT_MUTED


def add_container_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER, border_width=1):
    """Creates a clean rounded container card with subtle background and crisp border."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(border_width)
    else:
        shape.line.fill.background()
    return shape


# ==============================================================================
# SLIDE BUILDERS (10 SLIDES)
# ==============================================================================

def build_slide_1(prs):
    """Slide 1: Introduction / Title Page."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    # Academic Top Banner
    banner = slide.shapes.add_textbox(Inches(0.8), Inches(0.55), Inches(11.7), Inches(0.4))
    tf = banner.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "DELHI TECHNOLOGICAL UNIVERSITY • DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING • B.TECH PROJECT-I"
    p.font.name = "Arial"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = CRIMSON

    # Main Project Title
    tbox = slide.shapes.add_textbox(Inches(0.8), Inches(0.95), Inches(11.7), Inches(1.2))
    tf2 = tbox.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Multimodal Feature Fusion of Statistical and Deep Learning Representations for Actigraphy-Based Depression Detection"
    p2.font.name = "Arial"
    p2.font.size = Pt(24)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_TITLE

    # Tagline / Subtitle
    sbox = slide.shapes.add_textbox(Inches(0.8), Inches(2.25), Inches(11.7), Inches(0.45))
    tf3 = sbox.text_frame
    tf3.word_wrap = True
    p3 = tf3.paragraphs[0]
    p3.text = "Continuous Motor Actigraphy, Clinical Digital Biomarkers & Strict 55-Fold Leave-One-Subject-Out (LOSO) Cross-Validation"
    p3.font.name = "Arial"
    p3.font.size = Pt(12)
    p3.font.bold = True
    p3.font.color.rgb = BLUE

    # Left Card: Project Metadata & Team Roster
    c_meta = add_container_card(slide, Inches(0.8), Inches(2.9), Inches(5.6), Inches(4.0), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb_meta = slide.shapes.add_textbox(Inches(1.0), Inches(3.05), Inches(5.2), Inches(3.7))
    tf_m = tb_meta.text_frame
    tf_m.word_wrap = True

    p_m1 = tf_m.paragraphs[0]
    p_m1.text = "PROJECT METADATA & TEAM ROSTER"
    p_m1.font.name = "Arial"
    p_m1.font.size = Pt(11)
    p_m1.font.bold = True
    p_m1.font.color.rgb = TEXT_TITLE
    p_m1.space_after = Pt(8)

    meta_lines = [
        ("Group ID:", "DTU/CSE/BTP/GP178  (AY 2026–27)"),
        ("Supervisor:", "Dr. Anshika Arora, Assistant Professor, Dept. of CSE, DTU"),
        ("Team Members:", ""),
        ("  • Ankit Singh", "(Roll No: 23/CS/064)"),
        ("  • Ankit", "(Roll No: 23/CS/059)"),
        ("  • Kumari Rashi", "(Roll No: 24/CS/253)"),
        ("Institution:", "Delhi Technological University (formerly DCE), Delhi"),
        ("Benchmark:", "Depresjon Public Actigraphy Dataset (55 Subjects)")
    ]
    for lbl, val in meta_lines:
        p_item = tf_m.add_paragraph()
        p_item.text = f"{lbl} {val}"
        p_item.font.name = "Arial"
        p_item.font.size = Pt(9.5)
        p_item.font.color.rgb = TEXT_BODY
        if lbl in ["Group ID:", "Supervisor:", "Team Members:"]:
            p_item.font.bold = True
            p_item.font.color.rgb = TEXT_TITLE

    # Right Column: 3 Strategic Project Pillars
    pillars = [
        ("1. CLINICAL MOTIVATION (THE PROBLEM)",
         "Moving psychiatric depression evaluation from subjective, episodic recall questionnaires (MADRS, PHQ-9) to passive, objective continuous monitoring using wrist-worn IoMT accelerometers.",
         CRIMSON, CRIMSON_BG),
        ("2. METHODOLOGICAL RIGOR (ZERO LEAKAGE)",
         "Eliminating pervasive literature data leakage via a strict 55-Fold Nested Leave-One-Subject-Out (LOSO) cross-validation protocol to evaluate true clinical generalization on unseen patients.",
         BLUE, BLUE_BG),
        ("3. BREAKTHROUGH FINDING (92.73% CHAMPION)",
         "Coupling 21 circadian digital biomarkers with 1D-CNN temporal representations in our Dual-Fusion Ensemble achieves 92.73% Accuracy (51/55 correct) and 0.9470 ROC-AUC (+20.00% over baselines).",
         EMERALD, EMERALD_BG)
    ]
    y_card = 2.9
    right_cards = []
    for title, desc, col, bg_col in pillars:
        c_p = add_container_card(slide, Inches(6.6), Inches(y_card), Inches(5.9), Inches(1.22), bg_color=bg_col, border_color=col)
        right_cards.append(c_p.shape_id)
        tb_p = slide.shapes.add_textbox(Inches(6.8), Inches(y_card + 0.1), Inches(5.5), Inches(1.05))
        tf_p = tb_p.text_frame
        tf_p.word_wrap = True
        p_t = tf_p.paragraphs[0]
        p_t.text = title
        p_t.font.name = "Arial"
        p_t.font.size = Pt(10)
        p_t.font.bold = True
        p_t.font.color.rgb = col
        p_t.space_after = Pt(3)

        p_d = tf_p.add_paragraph()
        p_d.text = desc
        p_d.font.name = "Arial"
        p_d.font.size = Pt(8.5)
        p_d.font.color.rgb = TEXT_BODY
        y_card += 1.39

    add_slide_transition_and_animations(slide, [c_meta.shape_id] + right_cards)


def build_slide_2(prs):
    """Slide 2: Problem Statement & Clinical Motivation."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    add_slide_header(
        slide,
        "SECTION 01 • CLINICAL PROBLEM STATEMENT",
        "The Diagnostic Blindspot: Physical Illness vs. 'The Invisible Illness'",
        "Why major depression eludes traditional medical diagnostics and how continuous actigraphy bridges the gap"
    )

    # Left Column: Structured Cards (Width 5.5)
    c_left = add_container_card(slide, Inches(0.8), Inches(1.7), Inches(5.5), Inches(5.2), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.1), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True

    sections = [
        ("The Core Dilemma: Physical vs. Mental Illness",
         "• Any person with physical illness can be identified objectively:\n"
         "  - Diabetes has blood glucose assays; Hypertension has blood pressure;\n"
         "  - Pneumonia has chest X-rays; Cardiac disease has ECG traces.\n"
         "• But a person with depression CANNOT be detected physically:\n"
         "  - No blood test, no fever, no visible wound or physical swelling.\n"
         "  - MDD is 'The Invisible Illness' hiding behind outward appearances.",
         CRIMSON),
        ("The Flaw of Conventional Psychiatric Assessment",
         "• Clinical Standard: Subjective retrospective interviews (MADRS, PHQ-9).\n"
         "• Severe Recall Bias: Depressed patients struggle to accurately recall mood.\n"
         "• Social Stigma & Masking: Patients conceal symptoms during clinical visits.\n"
         "• Episodic Care: Consultations occur weeks apart; missed early relapses.",
         AMBER),
        ("Our Research Problem Statement",
         "• Can continuous, non-invasive wrist actigraphy capture the invisible behavioral footprint of depression passively?\n"
         "• Challenge: Build a leak-free machine learning framework that detects altered psychomotor vitality and circadian rhythmicity in unseen clinical patients.",
         BLUE)
    ]
    for i, (head, body, col) in enumerate(sections):
        p_h = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p_h.text = head
        p_h.font.name = "Arial"
        p_h.font.size = Pt(10)
        p_h.font.bold = True
        p_h.font.color.rgb = col
        p_h.space_after = Pt(2)
        if i > 0:
            p_h.space_before = Pt(8)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.name = "Arial"
        p_b.font.size = Pt(8.5)
        p_b.font.color.rgb = TEXT_BODY

    # Right Column: Diagram
    diag_shape = slide.shapes.add_picture('presentation/diagrams/diagram_slide2_problem_statement.png',
                                          Inches(6.5), Inches(1.7), Inches(6.0), Inches(5.2))

    add_slide_transition_and_animations(slide, [c_left.shape_id, diag_shape.shape_id])


def build_slide_3(prs):
    """Slide 3: Depresjon Benchmark Dataset."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    add_slide_header(
        slide,
        "SECTION 02 • CLINICAL BENCHMARK DATASET",
        "The Depresjon Benchmark: Cohort Audit & Actigraphic Protocol",
        "Continuous motor actigraphy from Haukeland University Hospital, Bergen, Norway"
    )

    # Left Column: Audit Cards
    c_left = add_container_card(slide, Inches(0.8), Inches(1.7), Inches(5.5), Inches(5.2), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.1), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True

    sections = [
        ("1. Clinical Cohort Audit (55 Total Subjects)",
         "• Condition Group (N = 23 Depressed Patients):\n"
         "  - 15 Bipolar II, 7 Bipolar I, 1 Unipolar Major Depressive Disorder;\n"
         "  - 18 Outpatients, 5 Inpatients; Mean baseline MADRS score = 22.74 ± 4.80.\n"
         "• Control Group (N = 32 Healthy Volunteers):\n"
         "  - Hospital staff & community volunteers; zero psychiatric history.",
         CRIMSON),
        ("2. Data Acquisition & Normalization (Option B)",
         "• Device: Cambridge Neurotechnology Actiwatch AW4 (wrist-worn, 32 Hz).\n"
         "• Standardized Active Days: Standardized to active days in scores.csv\n"
         "  (Mean 12.6 days/subject; Total 693 days = 997,920 minute readings).\n"
         "• DST Transition: Exactly 4 controls crossed European Daylight Saving Time\n"
         "  (March 30, 2003); only 240 missing minutes (0.024%) linearly interpolated.",
         BLUE),
        ("3. 24-Hour Overlapping Windowing Protocol",
         "• Window Length: W = 24 hours (1,440 minutes).\n"
         "• Hop Size: H = 12 hours (50% overlap) -> Yields exactly 1,331 windows.\n"
         "• Slicing Rule: Strict subject boundary enforcement; zero cross-subject windows.",
         EMERALD)
    ]
    for i, (head, body, col) in enumerate(sections):
        p_h = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p_h.text = head
        p_h.font.name = "Arial"
        p_h.font.size = Pt(10)
        p_h.font.bold = True
        p_h.font.color.rgb = col
        p_h.space_after = Pt(2)
        if i > 0:
            p_h.space_before = Pt(8)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.name = "Arial"
        p_b.font.size = Pt(8.5)
        p_b.font.color.rgb = TEXT_BODY

    # Right Column: Diagram (Minute-to-Hour Conversion, 24h Window Creation & 50% Overlap)
    diag_shape = slide.shapes.add_picture('presentation/diagrams/actigraphy_minute_to_hour_window_overlap.png',
                                          Inches(6.4), Inches(1.7), Inches(6.1), Inches(5.2))

    add_slide_transition_and_animations(slide, [c_left.shape_id, diag_shape.shape_id])


def build_slide_4(prs):
    """Slide 4: Leave-One-Subject-Out (LOSO) Validation."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    add_slide_header(
        slide,
        "SECTION 03 • METHODOLOGICAL RIGOR & CROSS-VALIDATION",
        "Evaluation Protocol: The 55-Fold Nested Leave-One-Subject-Out (LOSO)",
        "Eliminating pervasive literature data leakage to guarantee true hospital generalisation"
    )

    # Left Column: Why LOSO Matters
    c_left = add_container_card(slide, Inches(0.8), Inches(1.7), Inches(5.5), Inches(5.2), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.1), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True

    sections = [
        ("The Pervasive 'Leakage Trap' in Prior Literature",
         "• Most published studies randomly shuffle 1,331 overlapping windows into 10-fold CV.\n"
         "• Flaw 1 (Temporal Overlap): Consecutive windows share 12 hours (50%) of identical raw time readings across train and test folds.\n"
         "• Flaw 2 (Biometric Fingerprint Memorisation): Classifiers memorize individual walking cadences and circadian habits rather than general depression patterns, producing artificially inflated 95%+ accuracies.",
         CRIMSON),
        ("Our Gold Standard: 55-Fold Nested LOSO Architecture",
         "• Outer Cross-Validation (55 Folds): In each fold k, participant P_k is completely held out as an unseen test subject (N_test = 1). All other 54 participants form the training pool (N_train = 54).\n"
         "• Nested Inner 5-Fold Group CV: Hyperparameter tuning (C, gamma) and feature selection run strictly on the 54 training subjects grouped by ID.\n"
         "• Training-Only Preprocessing: StandardScaler is fitted ONLY on training subjects.\n"
         "• Patient-Level Aggregation: Window predictions for P_k are averaged: P_i = (1/W_i) * sum(P_w). Binary diagnosis: y_hat = I(P_i >= 0.5).",
         BLUE),
        ("Clinical Significance of Zero Leakage",
         "• The model is evaluated on a patient whose physical habits were never seen.\n"
         "• Validates real-world clinical deployment for incoming patients.",
         EMERALD)
    ]
    for i, (head, body, col) in enumerate(sections):
        p_h = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p_h.text = head
        p_h.font.name = "Arial"
        p_h.font.size = Pt(10)
        p_h.font.bold = True
        p_h.font.color.rgb = col
        p_h.space_after = Pt(2)
        if i > 0:
            p_h.space_before = Pt(8)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.name = "Arial"
        p_b.font.size = Pt(8.5)
        p_b.font.color.rgb = TEXT_BODY

    # Right Column: Diagram
    diag_shape = slide.shapes.add_picture('presentation/diagrams/diagram_slide4_loso_55fold.png',
                                          Inches(6.5), Inches(1.7), Inches(6.0), Inches(5.2))

    add_slide_transition_and_animations(slide, [c_left.shape_id, diag_shape.shape_id])


def build_slide_5(prs):
    """Slide 5: Literature Comparison of Models Under LOSO Split."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    add_slide_header(
        slide,
        "SECTION 04 • LITERATURE BENCHMARK & REALITY CHECK",
        "Cross-Study Benchmark: Models Evaluated Under Strict LOSO Splits",
        "Comparing published actigraphy models under leak-free subject-level validation on unseen patients"
    )

    # Top Half: Literature Comparison Table
    rows = 7
    cols = 6
    t_shape = slide.shapes.add_table(rows, cols, Inches(0.8), Inches(1.65), Inches(11.7), Inches(2.3))
    table = t_shape.table

    table_data = [
        ["Study & Model", "Input Representation", "Split Protocol", "Ordinary Acc", "Balanced Acc", "Sensitivity"],
        ["Garcia-Ceja et al. (2018)", "3 Stats (mean, std, zero_ratio)", "55-Fold LOSO", "72.73%", "71.06%", "60.87% (14/23)"],
        ["Garcia-Ceja et al. (2018b)", "Statistical + Oversampling", "10-Fold Subject", "73.00%", "~72.00%", "~69.00%"],
        ["Frogner et al. (2019)", "1D-CNN (Raw 1-min Series)", "Subject Split", "71.00%", "~70.50%", "~68.00%"],
        ["Our Statistical Branch", "21 Clinical Digital Biomarkers", "55-Fold Nested LOSO", "80.00%", "79.76%", "78.26% (18/23)"],
        ["Our Standalone 1D-CNN", "24-Step Hourly Downsampled", "55-Fold Nested LOSO", "80.00%", "79.76%", "78.26% (18/23)"],
        ["★ Our Dual-Fusion Champion ★", "LogReg-21 + 1D-CNN Ensemble", "55-Fold Nested LOSO", "92.73%", "92.53%", "91.30% (21/23)"]
    ]

    col_widths = [Inches(2.5), Inches(2.8), Inches(1.8), Inches(1.4), Inches(1.4), Inches(1.8)]
    for j, w in enumerate(col_widths):
        table.columns[j].width = w

    for i, row in enumerate(table_data):
        for j, cell_text in enumerate(row):
            cell = table.cell(i, j)
            cell.text = cell_text
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.alignment = PP_ALIGN.CENTER if j >= 3 else PP_ALIGN.LEFT
            
            if i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TEXT_TITLE
                p.font.bold = True
                p.font.size = Pt(9.5)
                p.font.color.rgb = WHITE
            elif i == 6:  # Champion
                cell.fill.solid()
                cell.fill.fore_color.rgb = CRIMSON_BG
                p.font.bold = True
                p.font.size = Pt(9.5)
                p.font.color.rgb = CRIMSON
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if i % 2 == 1 else CARD_BG
                p.font.size = Pt(8.5)
                p.font.color.rgb = TEXT_BODY
                if i >= 4:
                    p.font.bold = True

    # Bottom Half: Diagram + Strategic Insights
    diag_shape = slide.shapes.add_picture('presentation/diagrams/diagram_slide5_loso_literature.png',
                                          Inches(0.8), Inches(4.15), Inches(6.8), Inches(2.9))

    c_takeaways = add_container_card(slide, Inches(7.8), Inches(4.15), Inches(4.7), Inches(2.9), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb_t = slide.shapes.add_textbox(Inches(8.0), Inches(4.25), Inches(4.3), Inches(2.7))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True

    p_th = tf_t.paragraphs[0]
    p_th.text = "CRITICAL LITERATURE INSIGHTS"
    p_th.font.name = "Arial"
    p_th.font.size = Pt(10)
    p_th.font.bold = True
    p_th.font.color.rgb = CRIMSON
    p_th.space_after = Pt(4)

    bullets = [
        "• The ~72% Subject-Level Ceiling: Without window leakage, published models consistently achieve only 71%–73% on unseen patients.",
        "• Digital Biomarker Power: Our 21 handcrafted features alone achieve 80.00% (+13.39% balanced accuracy boost over 3 basic stats).",
        "• Synergistic Breakthrough: Combining LogReg-21 with 1D-CNN breaks through the ceiling to reach 92.73% Accuracy (51/55 correct).",
        "• Statistical Validation: +20.00% absolute jump over foundational baseline is statistically significant (McNemar exact test: p = 0.0076 < 0.01)."
    ]
    for b in bullets:
        p_b = tf_t.add_paragraph()
        p_b.text = b
        p_b.font.name = "Arial"
        p_b.font.size = Pt(8)
        p_b.font.color.rgb = TEXT_BODY
        p_b.space_after = Pt(2)

    add_slide_transition_and_animations(slide, [t_shape.shape_id, diag_shape.shape_id, c_takeaways.shape_id])


def build_slide_6(prs):
    """Slide 6: Comparison with Baseline Model & Layer Scaling."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    add_slide_header(
        slide,
        "SECTION 05 • BASELINE REPRODUCTION & LAYER SCALING",
        "Comparison with Baseline: Arora et al. (2023) Reproduction & Scaling",
        "Reproducing Dr. Arora's BiLSTM-SVM benchmark and resolving the 99.62% vs 92.73% accuracy paradox"
    )

    # Left Column: Structured Cards
    c_left = add_container_card(slide, Inches(0.8), Inches(1.7), Inches(5.5), Inches(5.2), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.1), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True

    sections = [
        ("1. The Published Baseline: Arora et al. (2023)",
         "• Architecture: Hybrid Bidirectional LSTM (L = 2 layers) + Linear SVM (C = 0.1).\n"
         "• Input: 1-hour downsampled 24h actigraphy + 3 statistical features.\n"
         "• Evaluation: 10-Fold Cross-Validation on 1,331 overlapping windows.\n"
         "• Published Benchmark: 95.57% Window-Level Accuracy.",
         BLUE),
        ("2. Faithful Reproduction & Layer-Depth Scaling Benchmark",
         "• Reproduction (BiLSTM-2 + SVM): Reached 96.39% Window Accuracy (ROC-AUC 0.9939, 1,283/1,331 correct windows) -> Beats published figure!\n"
         "• ★ Enhanced Model (BiLSTM-3 + SVM): Increasing bidirectional depth to L = 3 layers unlocks 99.62% Window Accuracy (ROC-AUC 0.9997, 1,326/1,331 correct windows!).\n"
         "• Saturated Model (BiLSTM-4 + SVM): Reaches 97.37% (parameter saturation).",
         EMERALD),
        ("3. Resolving the Discrepancy: 99.62% vs. 92.73%",
         "• 99.62% Window Accuracy evaluates temporal representation capacity when overlapping windows from the same subject appear across folds.\n"
         "• 92.73% LOSO Accuracy evaluates true clinical generalization to completely unseen hospital patients without biometric memorization.",
         CRIMSON)
    ]
    for i, (head, body, col) in enumerate(sections):
        p_h = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p_h.text = head
        p_h.font.name = "Arial"
        p_h.font.size = Pt(10)
        p_h.font.bold = True
        p_h.font.color.rgb = col
        p_h.space_after = Pt(2)
        if i > 0:
            p_h.space_before = Pt(8)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.name = "Arial"
        p_b.font.size = Pt(8.5)
        p_b.font.color.rgb = TEXT_BODY

    # Right Column: Diagram
    diag_shape = slide.shapes.add_picture('presentation/diagrams/diagram_slide6_baseline_scaling.png',
                                          Inches(6.5), Inches(1.7), Inches(6.0), Inches(5.2))

    add_slide_transition_and_animations(slide, [c_left.shape_id, diag_shape.shape_id])


def build_slide_7(prs):
    """Slide 7: Temporal Granularity Benchmark: Hourly vs Minutewise."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    add_slide_header(
        slide,
        "SECTION 06 • TEMPORAL RESOLUTION ANALYSIS",
        "Temporal Resolution Benchmark: Hourly ($T=24$) vs. Minute ($T=1,440$)",
        "Controlled empirical proof under 55-fold LOSO of why hourly aggregation acts as a biological low-pass filter"
    )

    # Top Half: Empirical Resolution Comparison Table
    rows = 5
    cols = 8
    t_shape = slide.shapes.add_table(rows, cols, Inches(0.8), Inches(1.65), Inches(11.7), Inches(1.9))
    table = t_shape.table

    table_data = [
        ["Model & Paradigm", "Resolution", "Seq Length", "Balanced Acc", "Ordinary Acc", "Sensitivity", "ROC-AUC", "Correct / 55"],
        ["Standalone 1D-CNN", "Hourly (Downsampled)", "T = 24", "79.76%", "80.00%", "78.26%", "0.8899", "44 / 55"],
        ["Standalone 1D-CNN", "Minute (Raw Integrations)", "T = 1,440", "66.03%", "65.45%", "69.57%", "0.7378", "36 / 55"],
        ["★ Multimodal Fusion (Weighted)", "Hourly (Downsampled)", "T = 24", "90.96%", "90.91%", "91.30%", "0.9429", "50 / 55"],
        ["Multimodal Fusion (Weighted)", "Minute (Raw Integrations)", "T = 1,440", "76.02%", "76.36%", "73.91%", "0.8505", "42 / 55"]
    ]

    col_widths = [Inches(2.4), Inches(1.8), Inches(1.1), Inches(1.3), Inches(1.3), Inches(1.2), Inches(1.2), Inches(1.4)]
    for j, w in enumerate(col_widths):
        table.columns[j].width = w

    for i, row in enumerate(table_data):
        for j, cell_text in enumerate(row):
            cell = table.cell(i, j)
            cell.text = cell_text
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.alignment = PP_ALIGN.CENTER if j >= 1 else PP_ALIGN.LEFT
            
            if i == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TEXT_TITLE
                p.font.bold = True
                p.font.size = Pt(9)
                p.font.color.rgb = WHITE
            elif i == 3:  # Hourly Fusion Champion
                cell.fill.solid()
                cell.fill.fore_color.rgb = EMERALD_BG
                p.font.bold = True
                p.font.size = Pt(9)
                p.font.color.rgb = EMERALD
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if i % 2 == 1 else CARD_BG
                p.font.size = Pt(8.5)
                p.font.color.rgb = TEXT_BODY
                if "Hourly" in row[1]:
                    p.font.bold = True

    # Bottom Half: Diagram + 3 Physiological Takeaways
    diag_shape = slide.shapes.add_picture('presentation/diagrams/diagram_slide7_hourly_vs_minute.png',
                                          Inches(0.8), Inches(3.75), Inches(7.0), Inches(3.3))

    c_takeaways = add_container_card(slide, Inches(8.0), Inches(3.75), Inches(4.5), Inches(3.3), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb_t = slide.shapes.add_textbox(Inches(8.2), Inches(3.85), Inches(4.1), Inches(3.1))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True

    p_th = tf_t.paragraphs[0]
    p_th.text = "WHY HOURLY BEATS MINUTE DATA"
    p_th.font.name = "Arial"
    p_th.font.size = Pt(10)
    p_th.font.bold = True
    p_th.font.color.rgb = BLUE
    p_th.space_after = Pt(4)

    reasons = [
        ("1. Biological Low-Pass Filter:",
         "Minute series (T=1,440) contain intense stochastic movement noise (typing, fidgeting, transport). Hourly downsampling acts as an anti-aliasing physiological filter.",
         SLATE_900),
        ("2. Circadian Phase Alignment:",
         "Affective depression is fundamentally a diurnal macro-phenomenon (blunted M10 peak, elevated L5 rest). Hourly bins capture diurnal amplitude with high SNR.",
         CRIMSON),
        ("3. Computational Stability:",
         "T=24 sequence steps eliminates vanishing gradients, requires 60x fewer FLOPs, and avoids parameter overfitting on clinical cohorts.",
         EMERALD)
    ]
    for r_head, r_body, r_col in reasons:
        p_rh = tf_t.add_paragraph()
        p_rh.text = r_head
        p_rh.font.name = "Arial"
        p_rh.font.size = Pt(8.5)
        p_rh.font.bold = True
        p_rh.font.color.rgb = r_col
        p_rh.space_after = Pt(1)

        p_rb = tf_t.add_paragraph()
        p_rb.text = r_body
        p_rb.font.name = "Arial"
        p_rb.font.size = Pt(8)
        p_rb.font.color.rgb = TEXT_BODY
        p_rb.space_after = Pt(4)

    add_slide_transition_and_animations(slide, [t_shape.shape_id, diag_shape.shape_id, c_takeaways.shape_id])


def build_slide_8(prs):
    """Slide 8: Feature Engineering & Dual-Branch Architecture."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    add_slide_header(
        slide,
        "SECTION 07 • FEATURE ENGINEERING & MODEL ARCHITECTURE",
        "Dual-Branch Pipeline: 21 Digital Biomarkers & Temporal 1D-CNN",
        "Complementary extraction of macro circadian rhythms and micro temporal dynamics"
    )

    # Left Column: Structured Cards
    c_left = add_container_card(slide, Inches(0.8), Inches(1.7), Inches(5.5), Inches(5.2), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.1), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True

    sections = [
        ("Branch A: 21 Domain Digital Biomarkers (Statistical)",
         "• 9 Descriptive Distributional Features:\n"
         "  Mean, Std Dev, Median, Q1, Q3, Interquartile Range (IQR), Max, Fisher Skewness, Kurtosis.\n"
         "• 5 Movement Volatility & Inertia Features:\n"
         "  Zero-Activity Ratio, Coefficient of Variation (CV), Mean Absolute Successive Difference (MASD), RMSSD, Lag-1 Autocorrelation (r1).\n"
         "• 7 Non-Parametric Circadian Rhythm Analysis (NPCRA):\n"
         "  Daytime Mean, Nighttime Mean, Daytime Inactivity (ZR_day), Nighttime Inactivity (ZR_night), M10 (10 most active hours), L5 (5 least active hours), Relative Amplitude (RA).\n"
         "• Classifier: L2 Regularized Logistic Regression (C = 0.1) -> 79.76% Bal Acc.",
         BLUE),
        ("Branch B: Deep Learning Representation (1D-CNN)",
         "• Architecture: Conv1D (k=3, c=16) -> BatchNorm -> ReLU -> MaxPool -> Conv1D (k=3, c=32) -> Global Average Pooling -> Dense.\n"
         "• Hyper-Lightweight: Exactly 6,593 Parameters (Fast training, zero vanishing gradient, robust anti-overfitting).\n"
         "• Outperformed Recurrent Nets: Outperformed LSTM (70.45%) and GRU (74.46%) under 55-fold LOSO, reaching 0.8899 ROC-AUC.",
         PURPLE)
    ]
    for i, (head, body, col) in enumerate(sections):
        p_h = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p_h.text = head
        p_h.font.name = "Arial"
        p_h.font.size = Pt(10)
        p_h.font.bold = True
        p_h.font.color.rgb = col
        p_h.space_after = Pt(2)
        if i > 0:
            p_h.space_before = Pt(8)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.name = "Arial"
        p_b.font.size = Pt(8.5)
        p_b.font.color.rgb = TEXT_BODY

    # Right Column: Diagram
    diag_shape = slide.shapes.add_picture('presentation/diagrams/diagram_slide8_architecture_pipeline.png',
                                          Inches(6.5), Inches(1.7), Inches(6.0), Inches(5.2))

    add_slide_transition_and_animations(slide, [c_left.shape_id, diag_shape.shape_id])


def build_slide_9(prs):
    """Slide 9: Multimodal Dual-Fusion & Benchmark Results."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    add_slide_header(
        slide,
        "SECTION 08 • EMPIRICAL BREAKTHROUGH & CLINICAL FORENSICS",
        "Multimodal Dual-Fusion: The 92.73% Champion Ensemble",
        "Coupling weighted linear probability with meta-learner stacking to achieve breakthrough diagnostic accuracy"
    )

    # Left Column: Performance Metrics & Forensic Audit
    c_left = add_container_card(slide, Inches(0.8), Inches(1.7), Inches(5.5), Inches(5.2), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.1), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True

    sections = [
        ("The Champion Dual-Fusion Ensemble Strategy",
         "• Fusion Formulation: Averages leak-free weighted linear probability (w=0.4/0.6) with non-linear Logistic Regression stacking:\n"
         "  P_ensemble = 0.5 * P_weighted + 0.5 * P_stacking\n"
         "• Fuses macro-circadian rhythms with micro-temporal deep dynamics.",
         CRIMSON),
        ("Master Clinical Performance (N = 55 Unseen Patients)",
         "• Raw Overall Accuracy: 92.73%  (51 out of 55 participants correct)\n"
         "• Balanced Accuracy: 92.53%  (Robust to 23 Condition vs 32 Control)\n"
         "• Clinical Sensitivity: 91.30%  (21 of 23 depressed patients detected)\n"
         "• Clinical Specificity: 93.75%  (30 of 32 healthy controls cleared)\n"
         "• F1-Score: 0.9130 | Matthews Corr (MCC): 0.8505 | ROC-AUC: 0.9470\n"
         "• Statistical Significance: p = 0.0076 < 0.01 vs literature baseline.",
         EMERALD),
        ("Forensic Audit of the 4 Residual Borderline Cases",
         "• condition_2 (FN): Admission MADRS 24 -> Exit MADRS 11 (Full remission!).\n"
         "• condition_16 (FN): Baseline MADRS was 13 (Mildest borderline case).\n"
         "• control_4 (FP): Highly sedentary desk worker (Daytime zero ratio +1.64σ).\n"
         "• control_32 (FP): Elderly subject with age-related psychomotor slowing.",
         AMBER)
    ]
    for i, (head, body, col) in enumerate(sections):
        p_h = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p_h.text = head
        p_h.font.name = "Arial"
        p_h.font.size = Pt(10)
        p_h.font.bold = True
        p_h.font.color.rgb = col
        p_h.space_after = Pt(2)
        if i > 0:
            p_h.space_before = Pt(8)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.name = "Arial"
        p_b.font.size = Pt(8.5)
        p_b.font.color.rgb = TEXT_BODY

    # Right Column: Diagram
    diag_shape = slide.shapes.add_picture('presentation/diagrams/diagram_slide9_fusion_results_cm.png',
                                          Inches(6.5), Inches(1.7), Inches(6.0), Inches(5.2))

    add_slide_transition_and_animations(slide, [c_left.shape_id, diag_shape.shape_id])


def build_slide_10(prs):
    """Slide 10: Clinical AI Co-Pilot, Explainability & Conclusion."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_white_background(slide)

    add_slide_header(
        slide,
        "SECTION 09 • TRANSLATIONAL IMPACT & CONCLUSION",
        "Bedside Clinical AI Co-Pilot, Explainable AI & Research Roadmap",
        "Translating machine learning into actionable, interpretable bedside psychiatric decision support"
    )

    # Left Column: Research Deliverables & Future Work
    c_left = add_container_card(slide, Inches(0.8), Inches(1.7), Inches(5.5), Inches(5.2), bg_color=CARD_BG, border_color=CARD_BORDER)
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.1), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True

    sections = [
        ("1. Explainable AI: Physiological Determinants (SHAP)",
         "• Daytime Sedentary Ratio (OR = 3.34, p = 0.00016): Dominant biomarker of psychomotor retardation.\n"
         "• Nocturnal Mean Activity (OR = 0.31, p = 0.00008): Reflects sleep continuity deficit.\n"
         "• 4-Head Temporal Attention: Saliency peaks at 07:00-09:00 (sluggish wake transition) and 14:00-16:00 (afternoon psychomotor drop).",
         BLUE),
        ("2. Bedside AI Co-Pilot & Web Dashboard (ai_system/)",
         "• Turnkey Clinician Web Dashboard on Port 8000.\n"
         "• Instant drag-and-drop CSV inference for new smartwatch actigraphy.\n"
         "• Automated DSM-5 aligned psychiatric consultation notes.",
         EMERALD),
        ("3. Academic Deliverables & Future Work",
         "• 19-Page Academic Manuscript ready for journal submission.\n"
         "• 46 Automated Unit & Integration Tests guaranteeing reproducibility.\n"
         "• Future Roadmap: Multi-center cohort validation & multi-sensor fusion (actigraphy + photoplethysmography HRV + skin temp).",
         CRIMSON)
    ]
    for i, (head, body, col) in enumerate(sections):
        p_h = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p_h.text = head
        p_h.font.name = "Arial"
        p_h.font.size = Pt(10)
        p_h.font.bold = True
        p_h.font.color.rgb = col
        p_h.space_after = Pt(2)
        if i > 0:
            p_h.space_before = Pt(8)

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.name = "Arial"
        p_b.font.size = Pt(8.5)
        p_b.font.color.rgb = TEXT_BODY

    # Right Column: Diagram
    diag_shape = slide.shapes.add_picture('presentation/diagrams/diagram_slide10_clinical_copilot.png',
                                          Inches(6.5), Inches(1.7), Inches(6.0), Inches(5.2))

    add_slide_transition_and_animations(slide, [c_left.shape_id, diag_shape.shape_id])


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    print("Building 10 professional slides...")
    build_slide_1(prs)
    build_slide_2(prs)
    build_slide_3(prs)
    build_slide_4(prs)
    build_slide_5(prs)
    build_slide_6(prs)
    build_slide_7(prs)
    build_slide_8(prs)
    build_slide_9(prs)
    build_slide_10(prs)

    output_path = "presentation/BTP_Final_Defense_GP178.pptx"
    prs.save(output_path)
    print(f"Presentation successfully created and saved to: {output_path}")

    # Also save a copy as presentation/presentation.pptx for convenience
    copy_path = "presentation/presentation.pptx"
    prs.save(copy_path)
    print(f"Convenience copy saved to: {copy_path}")


if __name__ == '__main__':
    main()
