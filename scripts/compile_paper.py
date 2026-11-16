#!/usr/bin/env python3
"""
Research Paper Compiler for DTU B.Tech Project GP178.

Converts `reports/research_paper_draft.md` into:
1. A publication-styled standalone academic HTML paper: `reports/Research_Paper_BTP_GP178.html`
2. A high-resolution print PDF via Headless Chrome: `reports/Research_Paper_BTP_GP178.pdf`
"""

import os
import re
import sys
import base64
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRAFT_MD_PATH = os.path.join(PROJECT_ROOT, "reports", "research_paper_draft.md")
OUTPUT_HTML_PATH = os.path.join(PROJECT_ROOT, "reports", "Research_Paper_BTP_GP178.html")
OUTPUT_PDF_PATH = os.path.join(PROJECT_ROOT, "reports", "Research_Paper_BTP_GP178.pdf")


def image_to_base64_data_uri(img_path: str) -> str:
    """Convert a local image to base64 Data URI for complete offline embedding."""
    if not os.path.isabs(img_path):
        img_path = os.path.join(PROJECT_ROOT, img_path)
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(img_path)[1].lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def markdown_to_academic_html(md_text: str) -> str:
    lines = md_text.splitlines()
    html_out = []
    in_table = False
    table_rows = []
    in_list = False
    in_abstract = False
    abstract_paras = []
    abstract_keywords = ""

    title = ""
    authors = ""
    supervisor = ""
    affiliation = ""
    group_id = ""
    working_paper = ""

    # Parse metadata header
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            i += 1
            continue
        if line.startswith("**Authors:**"):
            authors = line.replace("**Authors:**", "").strip()
            i += 1
            continue
        if line.startswith("**Supervisor:**"):
            supervisor = line.replace("**Supervisor:**", "").strip()
            i += 1
            continue
        if line.startswith("**Affiliation:**"):
            affiliation = line.replace("**Affiliation:**", "").strip()
            i += 1
            continue
        if line.startswith("**Group ID:**"):
            group_id = line.replace("**Group ID:**", "").strip()
            i += 1
            continue
        if line.startswith("**Working Paper Draft"):
            working_paper = line.strip("* ")
            i += 1
            continue
        if line == "---":
            i += 1
            break
        i += 1

    # Header block
    html_out.append(f"""
    <div class="paper-header">
      <h1 class="paper-title">{title}</h1>
      <div class="paper-meta">
        <p class="authors"><strong>Authors:</strong> {authors}</p>
        <p class="supervisor"><strong>Project Supervisor:</strong> {supervisor}</p>
        <p class="affiliation">{affiliation}</p>
        <p class="group-id"><span class="badge">Group ID: {group_id}</span> &bull; {working_paper}</p>
      </div>
    </div>
    """)

    # Function to format inline styles (math, bold, code, links)
    def inline_format(text: str) -> str:
        # Code backticks
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        # Bold + Italic
        text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"<strong><em>\1</em></strong>", text)
        # Bold
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        # Italic
        text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
        # Markdown links
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
        return text

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        t_html = ['<div class="table-container"><table>']
        # First row is header
        headers = [c.strip() for c in table_rows[0].split("|")[1:-1]]
        t_html.append("<thead><tr>" + "".join(f"<th>{inline_format(h)}</th>" for h in headers) + "</tr></thead>")
        t_html.append("<tbody>")
        for r in table_rows[2:]: # Skip separator row
            cells = [c.strip() for c in r.split("|")[1:-1]]
            if any(cells):
                t_html.append("<tr>" + "".join(f"<td>{inline_format(c)}</td>" for c in cells) + "</tr>")
        t_html.append("</tbody></table></div>")
        html_out.append("\n".join(t_html))
        table_rows = []
        in_table = False

    def flush_list():
        nonlocal in_list
        if in_list:
            html_out.append("</ul>")
            in_list = False

    # Process body lines
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        # Check for table
        if line.startswith("|") and line.endswith("|"):
            flush_list()
            in_table = True
            table_rows.append(line)
            i += 1
            continue
        elif in_table:
            flush_table()

        # Check for list items
        if line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html_out.append('<ul class="paper-list">')
                in_list = True
            item_text = line[2:].strip()
            html_out.append(f"<li>{inline_format(item_text)}</li>")
            i += 1
            continue
        elif in_list:
            flush_list()

        # Empty lines
        if not line:
            i += 1
            continue

        # Horizontal rule
        if line == "---":
            html_out.append('<hr class="paper-divider" />')
            i += 1
            continue

        # Headings
        if line.startswith("## Abstract"):
            # Enter abstract mode
            in_abstract = True
            i += 1
            continue

        if in_abstract:
            if line.startswith("## ") or line.startswith("# "):
                # End abstract
                in_abstract = False
                abs_content = "</p><p>".join(inline_format(p) for p in abstract_paras)
                html_out.append(f"""
                <div class="abstract-box">
                  <h3 class="abstract-title">Abstract</h3>
                  <div class="abstract-body"><p>{abs_content}</p></div>
                  {f'<p class="keywords"><strong>Keywords:</strong> {inline_format(abstract_keywords)}</p>' if abstract_keywords else ''}
                </div>
                """)
                # Fall through to heading handler
            elif line.startswith("**Keywords:**"):
                abstract_keywords = line.replace("**Keywords:**", "").strip()
                i += 1
                continue
            elif line == "---":
                i += 1
                continue
            else:
                abstract_paras.append(line)
                i += 1
                continue

        # Headings
        if line.startswith("## "):
            flush_list()
            heading_text = line[3:].strip()
            html_out.append(f'<h2 class="sec-heading">{inline_format(heading_text)}</h2>')
            i += 1
            continue
        elif line.startswith("### "):
            flush_list()
            heading_text = line[4:].strip()
            html_out.append(f'<h3 class="subsec-heading">{inline_format(heading_text)}</h3>')
            i += 1
            continue
        elif line.startswith("#### "):
            flush_list()
            heading_text = line[5:].strip()
            html_out.append(f'<h4 class="subsubsec-heading">{inline_format(heading_text)}</h4>')
            i += 1
            continue

        # Blockquote
        if line.startswith("> "):
            b_text = line[2:].strip()
            html_out.append(f'<blockquote class="paper-quote">{inline_format(b_text)}</blockquote>')
            i += 1
            continue

        # Display math $$ ... $$
        if line.startswith("$$") and line.endswith("$$") and len(line) > 2:
            html_out.append(f'<div class="display-math">{line}</div>')
            i += 1
            continue

        # Figure detection & image injection
        fig_match = re.search(r"See \*\*Figure (\d+)\*\* \(`([^`]+\.png)`\)", line)
        if fig_match:
            fig_num = fig_match.group(1)
            fig_path = fig_match.group(2)
            data_uri = image_to_base64_data_uri(fig_path)
            if data_uri:
                html_out.append(f"""
                <div class="figure-container">
                  <img src="{data_uri}" alt="Figure {fig_num}" class="paper-figure" />
                  <p class="figure-caption"><strong>Figure {fig_num}:</strong> {inline_format(line)}</p>
                </div>
                """)
                i += 1
                continue

        # Standard paragraph
        html_out.append(f'<p class="paper-paragraph">{inline_format(line)}</p>')
        i += 1

    if in_table:
        flush_table()
    if in_list:
        flush_list()

    body_html = "\n".join(html_out)

    complete_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <!-- MathJax Configuration -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    @page {{
      size: A4;
      margin: 20mm 18mm 20mm 18mm;
      @bottom-right {{
        content: counter(page);
      }}
    }}
    :root {{
      --text-color: #111827;
      --heading-color: #0f172a;
      --border-color: #cbd5e1;
      --link-color: #1d4ed8;
      --bg-color: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Times New Roman', Times, 'Cambria', Georgia, serif;
      font-size: 10.5pt;
      line-height: 1.5;
      color: var(--text-color);
      background-color: var(--bg-color);
      margin: 0 auto;
      max-width: 900px;
      padding: 30px 40px;
      text-align: justify;
      text-justify: inter-word;
    }}

    /* Paper Header */
    .paper-header {{
      text-align: center;
      margin-bottom: 25px;
      border-bottom: 1.5px solid #334155;
      padding-bottom: 20px;
    }}
    .paper-title {{
      font-size: 18pt;
      font-weight: 700;
      color: var(--heading-color);
      line-height: 1.25;
      margin: 0 0 15px 0;
      text-align: center;
    }}
    .paper-meta {{
      font-size: 10pt;
      color: #334155;
    }}
    .paper-meta p {{
      margin: 3px 0;
    }}
    .badge {{
      background-color: #e2e8f0;
      color: #0f172a;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 9pt;
      font-family: -apple-system, sans-serif;
    }}

    /* Abstract */
    .abstract-box {{
      background-color: #f8fafc;
      border: 1px solid var(--border-color);
      border-left: 4px solid #2563eb;
      padding: 14px 18px;
      margin: 20px 0 25px 0;
      border-radius: 2px;
    }}
    .abstract-title {{
      font-size: 11pt;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin: 0 0 8px 0;
      color: #1e3a8a;
    }}
    .abstract-body p {{
      margin: 0 0 8px 0;
      font-size: 9.8pt;
      line-height: 1.45;
    }}
    .keywords {{
      font-size: 9.5pt;
      margin-top: 8px;
      border-top: 1px dashed #cbd5e1;
      padding-top: 6px;
    }}

    /* Section Headings */
    h2.sec-heading {{
      font-size: 12.5pt;
      font-weight: 700;
      color: var(--heading-color);
      margin-top: 28px;
      margin-bottom: 10px;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 4px;
      page-break-after: avoid;
    }}
    h3.subsec-heading {{
      font-size: 11pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 18px;
      margin-bottom: 6px;
      page-break-after: avoid;
    }}
    h4.subsubsec-heading {{
      font-size: 10.5pt;
      font-weight: 600;
      font-style: italic;
      color: #334155;
      margin-top: 14px;
      margin-bottom: 4px;
      page-break-after: avoid;
    }}

    /* Paragraphs and Lists */
    .paper-paragraph {{
      margin: 0 0 10px 0;
      text-indent: 1.5em;
    }}
    .paper-list {{
      margin: 6px 0 12px 25px;
      padding-left: 10px;
    }}
    .paper-list li {{
      margin-bottom: 4px;
    }}
    .paper-divider {{
      border: 0;
      height: 1px;
      background: #e2e8f0;
      margin: 20px 0;
    }}
    .paper-quote {{
      border-left: 3px solid #94a3b8;
      margin: 10px 20px;
      padding: 6px 12px;
      background-color: #f1f5f9;
      font-style: italic;
    }}

    /* Tables */
    .table-container {{
      margin: 18px 0;
      overflow-x: auto;
      page-break-inside: avoid;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 9pt;
      margin: 0 auto;
      border-top: 2px solid #000;
      border-bottom: 2px solid #000;
    }}
    thead th {{
      border-bottom: 1px solid #000;
      padding: 6px 8px;
      text-align: left;
      font-weight: 700;
      background-color: #f8fafc;
    }}
    tbody td {{
      padding: 5px 8px;
      border-bottom: 1px solid #e2e8f0;
      vertical-align: top;
    }}
    tbody tr:last-child td {{
      border-bottom: none;
    }}

    /* Figures */
    .figure-container {{
      text-align: center;
      margin: 22px 0;
      page-break-inside: avoid;
    }}
    img.paper-figure {{
      max-width: 96%;
      height: auto;
      border: 1px solid #cbd5e1;
      border-radius: 3px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }}
    p.figure-caption {{
      font-size: 9pt;
      color: #334155;
      margin-top: 8px;
      font-style: italic;
    }}

    /* Math */
    .display-math {{
      margin: 12px 0;
      text-align: center;
      overflow-x: auto;
    }}
    code {{
      font-family: 'Courier New', Courier, monospace;
      font-size: 9.5pt;
      background-color: #f1f5f9;
      padding: 1px 4px;
      border-radius: 2px;
    }}

    /* Print Styles */
    @media print {{
      body {{
        padding: 0;
        max-width: 100%;
        font-size: 10pt;
      }}
      .paper-title {{ font-size: 16pt; }}
      a {{ color: #000; text-decoration: none; }}
      .badge {{ border: 1px solid #666; }}
    }}
  </style>
</head>
<body>
{body_html}
</body>
</html>
"""
    return complete_document


def compile_to_pdf(html_path: str, pdf_path: str):
    """Compiles the HTML document to PDF using Google Chrome Headless."""
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    chrome_bin = None
    for p in chrome_paths:
        if os.path.exists(p):
            chrome_bin = p
            break

    if not chrome_bin:
        print("[WARNING] Google Chrome binary not found. Skipping automatic PDF render.")
        print(f"You can open {html_path} in your browser and use 'Print to PDF' (Cmd+P).")
        return False

    cmd = [
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--user-data-dir=/tmp/chrome-pdf-user-data",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path,
    ]
    print(f"[COMPILING] Generating PDF using Headless Chrome: {pdf_path}...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0 and os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024.0
        print(f"[SUCCESS] Research Paper PDF successfully generated ({size_kb:.1f} KB): {pdf_path}")
        return True
    else:
        print(f"[WARNING] Chrome export returned code {res.returncode}. Output:\n{res.stderr}")
        return False


def main():
    print("=" * 80)
    print(" RESEARCH PAPER COMPILER (DTU B.TECH GP178) ")
    print("=" * 80)

    if not os.path.exists(DRAFT_MD_PATH):
        print(f"[ERROR] Markdown draft not found at {DRAFT_MD_PATH}")
        sys.exit(1)

    print(f"[READING] {DRAFT_MD_PATH}...")
    with open(DRAFT_MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    print("[PARSING] Converting Markdown to Academic Publication HTML...")
    html_content = markdown_to_academic_html(md_text)

    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    size_html_kb = os.path.getsize(OUTPUT_HTML_PATH) / 1024.0
    print(f"[SAVED] Academic HTML document written ({size_html_kb:.1f} KB): {OUTPUT_HTML_PATH}")

    # Compile to PDF
    compile_to_pdf(OUTPUT_HTML_PATH, OUTPUT_PDF_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()
