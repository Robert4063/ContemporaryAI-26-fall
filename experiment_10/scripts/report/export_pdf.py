"""Export visible report cells of an executed .ipynb to a two-page A4 PDF.

Uses Notebook Markdown and PNG outputs as the sole content source. Code stays
editable in Jupyter; hidden computation cells do not enter the printed report.
Run with the bundled document Python, which includes ReportLab and pypdf.
"""
from pathlib import Path
import argparse
import base64
from io import BytesIO
import json
import re
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak)

ROOT = Path(__file__).resolve().parents[2]


def inline(text):
    text = escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def export(notebook, output):
    font = Path("C:/Windows/Fonts/simsun.ttc")
    if font.exists():
        pdfmetrics.registerFont(TTFont("ReportCN", str(font), subfontIndex=0))
        pdfmetrics.registerFontFamily("ReportCN", normal="ReportCN", bold="ReportCN", italic="ReportCN")
    else:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        pdfmetrics.registerFontFamily("ReportCN", normal="STSong-Light", bold="STSong-Light")
    normal = "ReportCN" if font.exists() else "STSong-Light"
    body = ParagraphStyle("body", fontName=normal, fontSize=9.4, leading=13.9,
                          wordWrap="CJK", spaceAfter=5, textColor=colors.HexColor("#223140"))
    title = ParagraphStyle("title", parent=body, fontSize=18, leading=23, spaceAfter=7)
    heading = ParagraphStyle("heading", parent=body, fontSize=11.5, leading=16,
                             spaceBefore=7, spaceAfter=4, keepWithNext=True,
                             textColor=colors.HexColor("#164573"))
    small = ParagraphStyle("table", parent=body, fontSize=8.1, leading=11,
                           spaceAfter=0, alignment=TA_LEFT)
    width = A4[0] - 80
    story = []

    def markdown(source):
        blocks = re.split(r"\n\s*\n", source.strip())
        for block in blocks:
            if block.startswith("# "):
                story.append(Paragraph(inline(block[2:]), title))
            elif block.startswith("## "):
                story.append(Paragraph(inline(block[3:]), heading))
            elif block.startswith("|"):
                lines = [s.strip() for s in block.splitlines() if s.strip()]
                rows = [[s.strip() for s in line.strip("|").split("|")] for line in lines]
                rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", s) for s in r)]
                data = [[Paragraph(inline(s), small) for s in row] for row in rows]
                n = len(rows[0])
                weights = {2: [.32, .68], 5: [.12, .28, .2, .2, .2],
                           7: [.07, .13, .27, .1325, .1325, .1325, .1325]}.get(n, [1/n]*n)
                table = Table(data, colWidths=[width*x for x in weights], repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff6")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                    ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#b6c9db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.extend([table, Spacer(1, 5)])
            else:
                story.append(Paragraph(inline(block.replace("\n", " ")), body))

    nb = json.loads(notebook.read_text(encoding="utf-8"))
    current_page = 1
    for cell in nb["cells"]:
        if "report-hidden" in cell.get("metadata", {}).get("tags", []):
            continue
        target_page = cell.get("metadata", {}).get("report_page", 1)
        if target_page != current_page:
            story.append(PageBreak())
            current_page = target_page
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if cell["cell_type"] == "markdown":
            markdown(source)
        elif cell["cell_type"] == "code":
            if cell.get("execution_count") is None:
                raise ValueError("Execute all Notebook cells before exporting")
            for item in cell.get("outputs", []):
                if item.get("output_type") == "error":
                    raise ValueError("Notebook contains an execution error")
                data = item.get("data", {})
                if "text/markdown" in data:
                    content = data["text/markdown"]
                    markdown("".join(content) if isinstance(content, list) else content)
                elif "image/png" in data:
                    content = data["image/png"]
                    binary = base64.b64decode("".join(content) if isinstance(content, list) else content)
                    with PILImage.open(BytesIO(binary)) as im:
                        aspect = im.height / im.width
                    story.extend([Image(BytesIO(binary), width=width, height=width*aspect), Spacer(1, 4)])

    def footer(canvas, doc):
        canvas.setFont(normal, 8)
        canvas.setFillColor(colors.HexColor("#66778b"))
        canvas.drawString(40, 25, "实验一 · TF-IDF文本分类")
        canvas.drawRightString(A4[0] - 40, 25, f"{doc.page} / 2")

    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = output.with_suffix(".pending.pdf")
    doc = SimpleDocTemplate(str(candidate), pagesize=A4, leftMargin=40, rightMargin=40,
                            topMargin=31, bottomMargin=40,
                            title="实验一：TF-IDF文本分类", author="")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(candidate).pages)
    if pages != 2:
        raise ValueError(f"Two-page requirement failed: {pages} pages. Adjust Notebook content/layout; inspect {candidate}")
    candidate.replace(output)
    print(f"Exported {pages} pages: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=ROOT / "reports/实验一报告.ipynb")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/实验一报告.pdf")
    args = parser.parse_args()
    export(args.notebook, args.output)
