"""
Script to compile urop_research_progress_report_aug2026.md to a publication-quality PDF.
Uses headless Google Chrome to render HTML with MathJax, clean typography, and embedded figures.
"""

import os
import subprocess
import re

REPORT_MD = "/Users/chenyixin/.gemini/antigravity/brain/93c37d0e-e4f3-4103-b050-541a21acb302/urop_research_progress_report_aug2026.md"
REPORTS_DIR = "/Users/chenyixin/Documents/Independent Study/Swati's Summer Research/Hutch++/Matrix-vector_queries_estimation/reports"
OUT_PDF = os.path.join(REPORTS_DIR, "urop_research_progress_report_aug2026.pdf")
OUT_HTML = os.path.join(REPORTS_DIR, "urop_research_progress_report_aug2026.html")

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

with open(REPORT_MD, "r") as f:
    md_content = f.read()

# Convert markdown to clean HTML
html_body = md_content

# Replace markdown headers
html_body = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html_body, flags=re.M)
html_body = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html_body, flags=re.M)
html_body = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html_body, flags=re.M)
html_body = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html_body, flags=re.M)

# Replace horizontal rules
html_body = re.sub(r"^---$", r"<hr/>", html_body, flags=re.M)

# Replace images ![caption](path) with responsive figure cards
def replace_img(match):
    caption = match.group(1)
    path = match.group(2)
    return f"""
    <div class="figure-container">
        <img src="file://{path}" alt="{caption}"/>
        <div class="caption">{caption}</div>
    </div>
    """

html_body = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_img, html_body)

# Replace code blocks ``` ... ```
html_body = re.sub(r"```(.*?)\n(.*?)```", r"<pre><code>\2</code></pre>", html_body, flags=re.DOTALL)

# Replace bold and italics
html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
html_body = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html_body)

# Parse simple markdown tables
def parse_tables(text):
    lines = text.split("\n")
    in_table = False
    table_lines = []
    out = []
    
    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            in_table = True
            table_lines.append(line)
        else:
            if in_table:
                # process table
                if len(table_lines) >= 2:
                    header = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                    rows = [[c.strip() for c in r.split("|")[1:-1]] for r in table_lines[2:]]
                    t_html = "<table class='styled-table'><thead><tr>"
                    for h in header:
                        t_html += f"<th>{h}</th>"
                    t_html += "</tr></thead><tbody>"
                    for r in rows:
                        t_html += "<tr>"
                        for c in r:
                            t_html += f"<td>{c}</td>"
                        t_html += "</tr>"
                    t_html += "</tbody></table>"
                    out.append(t_html)
                table_lines = []
                in_table = False
            out.append(line)
    if in_table and len(table_lines) >= 2:
        header = [c.strip() for c in table_lines[0].split("|")[1:-1]]
        rows = [[c.strip() for c in r.split("|")[1:-1]] for r in table_lines[2:]]
        t_html = "<table class='styled-table'><thead><tr>"
        for h in header:
            t_html += f"<th>{h}</th>"
        t_html += "</tr></thead><tbody>"
        for r in rows:
            t_html += "<tr>"
            for c in r:
                t_html += f"<td>{c}</td>"
            t_html += "</tr>"
        t_html += "</tbody></table>"
        out.append(t_html)
    return "\n".join(out)

html_body = parse_tables(html_body)

# Replace paragraphs
paragraphs = html_body.split("\n\n")
processed_paragraphs = []
for p in paragraphs:
    p_strip = p.strip()
    if not p_strip:
        continue
    if p_strip.startswith("<h") or p_strip.startswith("<hr") or p_strip.startswith("<div") or p_strip.startswith("<pre") or p_strip.startswith("<table"):
        processed_paragraphs.append(p_strip)
    elif p_strip.startswith("- ") or p_strip.startswith("1. ") or p_strip.startswith("2. ") or p_strip.startswith("3. ") or p_strip.startswith("4. ") or p_strip.startswith("5. "):
        processed_paragraphs.append(f"<div class='list-block'>{p_strip.replace('\n', '<br/>')}</div>")
    else:
        processed_paragraphs.append(f"<p>{p_strip.replace('\n', ' ')}</p>")

final_body = "\n".join(processed_paragraphs)

html_template = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>UROP Research Progress Report: Adaptive & Certified Trace Estimation</title>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    @page {{
        size: letter;
        margin: 0.7in 0.7in 0.8in 0.7in;
        @bottom-right {{
            content: counter(page);
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        line-height: 1.55;
        font-size: 10.5pt;
        background: #ffffff;
    }}
    h1 {{
        color: #0f172a;
        font-size: 20pt;
        border-bottom: 2px solid #0284c7;
        padding-bottom: 6px;
        margin-top: 0;
        margin-bottom: 12px;
        font-weight: 700;
    }}
    h2 {{
        color: #0f172a;
        font-size: 14pt;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
        margin-top: 22px;
        margin-bottom: 10px;
        font-weight: 600;
        page-break-after: avoid;
    }}
    h3 {{
        color: #1e293b;
        font-size: 12pt;
        margin-top: 16px;
        margin-bottom: 8px;
        font-weight: 600;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 10px;
        text-align: justify;
    }}
    .figure-container {{
        text-align: center;
        margin: 16px 0;
        page-break-inside: avoid;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
    }}
    .figure-container img {{
        max-width: 96%;
        height: auto;
        border-radius: 4px;
    }}
    .caption {{
        font-size: 9pt;
        color: #475569;
        margin-top: 6px;
        font-style: italic;
        font-weight: 500;
    }}
    .styled-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9pt;
        page-break-inside: avoid;
    }}
    .styled-table th {{
        background-color: #0f172a;
        color: #ffffff;
        text-align: left;
        padding: 6px 10px;
        font-weight: 600;
    }}
    .styled-table td {{
        padding: 6px 10px;
        border-bottom: 1px solid #e2e8f0;
    }}
    .styled-table tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    pre {{
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 10px;
        font-size: 8.5pt;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        overflow-x: auto;
        page-break-inside: avoid;
        margin: 10px 0;
    }}
    .list-block {{
        margin-bottom: 10px;
        padding-left: 15px;
    }}
    hr {{
        border: 0;
        border-top: 1px solid #cbd5e1;
        margin: 16px 0;
    }}
    strong {{
        color: #0f172a;
    }}
</style>
</head>
<body>
{final_body}
</body>
</html>
"""

with open(OUT_HTML, "w") as f:
    f.write(html_template)

print(f"Wrote HTML to {OUT_HTML}")

# Compile HTML to PDF via headless Google Chrome
cmd = [
    CHROME_BIN,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--print-to-pdf-no-header",
    f"--print-to-pdf={OUT_PDF}",
    OUT_HTML
]

print("Running headless Chrome PDF generation...")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0:
    print(f"Successfully generated PDF: {OUT_PDF}")
else:
    print(f"Error: {res.stderr}")
