"""Create the ordered submission PDF, requiring real metadata for a final copy.

python scripts/build_submission.py --draft
python scripts/build_submission.py --config private_submission/config.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import textwrap
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from PIL import Image as PILImage, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    Preformatted, KeepTogether,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core import ROOT
from scripts.update_readme import run as update_readme

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 42
WIDTH = PAGE_WIDTH - 2 * MARGIN
NAVY = colors.HexColor("#17344a")
TEAL = colors.HexColor("#0e7490")
LIGHT = colors.HexColor("#edf5f8")


def get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=26, leading=31, textColor=NAVY, spaceAfter=18))
    styles.add(ParagraphStyle("SectionTitle", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=TEAL, spaceBefore=13, spaceAfter=9, keepWithNext=True))
    styles.add(ParagraphStyle("SubTitle", fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle("BodyCopy", fontName="Helvetica", fontSize=9, leading=13, spaceAfter=7, textColor=NAVY, splitLongWords=True))
    styles.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=7.2, leading=10, textColor=NAVY, splitLongWords=True))
    styles.add(ParagraphStyle("CellHead", fontName="Helvetica-Bold", fontSize=7.2, leading=10, textColor=colors.white))
    styles.add(ParagraphStyle("CodeCopy", fontName="Courier", fontSize=7.4, leading=10, backColor=LIGHT, borderPadding=7, spaceAfter=10))
    return styles


def inline(text):
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#0e7490">\1</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    # Plain URLs are clickable; do not process URLs already inside link markup.
    if "<link " not in text:
        text = re.sub(r"(https?://[^\s<>]+)", r'<link href="\1" color="#0e7490">\1</link>', text)
    return text


def table_flow(rows, styles):
    cols = len(rows[0])
    if cols == 7:
        widths = [109] + [(WIDTH - 109) / 6] * 6
    elif cols == 2:
        widths = [119, WIDTH - 119]
    else:
        widths = [WIDTH / cols] * cols
    cells = [[Paragraph(inline(cell), styles["CellHead"] if i == 0 else styles["Cell"]) for cell in row] for i, row in enumerate(rows)]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), .5, TEAL),
        ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#d5e5ee")),
    ]))
    return [table, Spacer(1, 9)]


def markdown_flows(text, styles):
    lines, flows, i = text.splitlines(), [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.extend(textwrap.wrap(lines[i], width=91, replace_whitespace=False, drop_whitespace=False) or [""])
                i += 1
            flows.append(Preformatted("\n".join(code), styles["CodeCopy"]))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [s.strip() for s in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                    rows.append(cells)
                i += 1
            flows.extend(table_flow(rows, styles))
            continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            path = ROOT / image_match.group(2)
            with PILImage.open(path) as img:
                w, h = img.size
            scale = min(WIDTH / w, 260 / h)
            flows.extend([Image(str(path), width=w * scale, height=h * scale),
                          Paragraph(inline(image_match.group(1)), styles["BodyCopy"]), Spacer(1, 8)])
            i += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("# ")
            style = "SectionTitle" if level <= 2 else "SubTitle"
            flows.append(Paragraph(inline(title), styles[style]))
            i += 1
            continue
        if line.startswith("- "):
            flows.append(Paragraph("- " + inline(line[2:]), styles["BodyCopy"]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("#", "|", "- ", "```", "![")):
            paragraph.append(lines[i].strip())
            i += 1
        flows.append(Paragraph(inline(" ".join(paragraph)), styles["BodyCopy"]))
    return flows


def validate_config(config):
    missing = [k for k in ["student_name", "bits_id", "github_url", "streamlit_url", "lab_screenshot"] if not config.get(k)]
    if missing:
        raise ValueError("Final submission is incomplete. Supply: " + ", ".join(missing) + ". Use --draft for a clearly marked preview.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["bits_id"]):
        raise ValueError("BITS ID must use letters, numbers, hyphens or underscores only.")
    github = urlparse(config["github_url"])
    if github.scheme != "https" or github.netloc != "github.com" or len(github.path.strip("/").split("/")) != 2:
        raise ValueError("Use the actual HTTPS GitHub owner/repository URL.")
    app = urlparse(config["streamlit_url"])
    if app.scheme != "https" or not app.netloc.endswith(".streamlit.app"):
        raise ValueError("Use the actual HTTPS public Streamlit Community Cloud app URL.")
    screenshot = Path(config["lab_screenshot"]).expanduser()
    if not screenshot.is_absolute():
        screenshot = ROOT / screenshot
    if not screenshot.is_file():
        raise ValueError("The provided lab screenshot file does not exist.")
    with PILImage.open(screenshot) as img:
        img.verify()
    return screenshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "submission_config.json"))
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--output", help="Optional PDF output path")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    screenshot = None
    if not args.draft:
        screenshot = validate_config(config)
    elif config.get("lab_screenshot"):
        screenshot = Path(config["lab_screenshot"]).expanduser()
        if not screenshot.is_absolute():
            screenshot = ROOT / screenshot
        if not screenshot.is_file():
            screenshot = None
    update_readme(args.config)
    output = Path(args.output) if args.output else ROOT / "private_submission" / ("Submission_Draft.pdf" if args.draft else config["bits_id"] + ".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = get_styles()
    story = [Paragraph("Machine Learning<br/>Assignment 2", styles["ReportTitle"]),
             Paragraph("SpamScope: Five-model classification comparison", styles["SectionTitle"])]
    if args.draft:
        required_labels = {
            "student_name": "student name", "bits_id": "BITS ID", "github_url": "GitHub link",
            "streamlit_url": "live app link", "lab_screenshot": "BITS Virtual Lab execution screenshot",
        }
        pending = [label for key, label in required_labels.items() if not config.get(key)]
        pending_text = "; ".join(pending) if pending else "final review and screenshot verification"
        story.append(Paragraph("<b>DRAFT - COMPLETE BEFORE SUBMISSION</b><br/>Pending: " + escape(pending_text) + ".", styles["BodyCopy"]))
    story.extend([Paragraph("Student: " + escape(config.get("student_name") or "Pending"), styles["BodyCopy"]),
                  Paragraph("BITS ID: " + escape(config.get("bits_id") or "Pending"), styles["BodyCopy"]), Spacer(1, 12)])
    for title, key in [("1. GitHub repository link", "github_url"), ("2. Live Streamlit application link", "streamlit_url")]:
        story.append(Paragraph(title, styles["SectionTitle"]))
        value = config.get(key)
        story.append(Paragraph(inline(value) if value else "<b>Pending - no live URL has been supplied.</b>", styles["BodyCopy"]))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Dataset: UCI Spambase - 4,601 rows and 57 input features. All five classifiers, six metrics, saved preprocessing/model pipelines and test CSV are included in the project.", styles["BodyCopy"]))
    story.append(Paragraph("Publication status is not inferred from the presence of code. Open and verify both real public links before submitting.", styles["BodyCopy"]))
    story.extend([PageBreak(), Paragraph("3. BITS Virtual Lab execution screenshot", styles["SectionTitle"])])
    if screenshot:
        with PILImage.open(screenshot) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            data = io.BytesIO()
            img.save(data, format="PNG")
            w, h = img.size
        data.seek(0)
        scale = min(WIDTH / w, 610 / h)
        story.append(Image(data, width=w * scale, height=h * scale))
        story.append(Paragraph("Screenshot supplied by the student as execution evidence.", styles["BodyCopy"]))
    else:
        story.append(Paragraph("<b>Pending student action.</b> Run the assignment in your BITS Virtual Lab session, capture a genuine execution screenshot, and supply its path in your private config. No lab screenshot has been generated or substituted.", styles["BodyCopy"]))
    story.extend([PageBreak(), Paragraph("4. Complete GitHub README content", styles["SectionTitle"])])
    story.extend(markdown_flows((ROOT / "README.md").read_text(), styles))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d5e5ee"))
        canvas.line(MARGIN, 36, PAGE_WIDTH - MARGIN, 36)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(MARGIN, 24, "Machine Learning Assignment 2" + (" | DRAFT" if args.draft else ""))
        canvas.drawRightString(PAGE_WIDTH - MARGIN, 24, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=MARGIN, leftMargin=MARGIN,
                            topMargin=42, bottomMargin=48, title="Machine Learning Assignment 2 - SpamScope",
                            author=config.get("student_name") or "")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {'draft' if args.draft else 'final'} PDF: {output}")
    if args.draft:
        print("Draft only: pending URLs or screenshot must be completed before submission.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
