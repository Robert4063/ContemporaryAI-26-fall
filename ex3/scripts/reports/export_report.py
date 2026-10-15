"""Export Markdown-only report notebook to matching Markdown and a compact PDF.

The notebook is the editable source. Supports its headings, paragraphs, tables,
Markdown images and explicit <!-- pagebreak -->; no notebook execution required.
"""
import argparse
import html
import json
import re
from itertools import groupby
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]


def inline(text, latin_font="ReportRoman"):
    def font_runs(value, font):
        # Keep Chinese and unsupported mathematical symbols (e.g. membership)
        # in Song/Hei, rather than emitting missing glyphs from the Latin font.
        cmap = pdfmetrics.getFont(font).face.charToGlyph
        return "".join(
            f'<font name="{font}">{html.escape("".join(run))}</font>'
            if supported else html.escape("".join(run))
            for supported, run in groupby(value, key=lambda c: ord(c) in cmap)
        )

    parts=[]
    for token in re.split(r"(\*\*.+?\*\*|`[^`]+`|<br\s*/?>)", text):
        if re.fullmatch(r"<br\s*/?>", token):
            parts.append("<br/>")
        elif token.startswith("**") and token.endswith("**"):
            parts.append('<font name="ReportHei">'+font_runs(token[2:-2], "ReportRomanBold")+'</font>')
        else:
            parts.append(font_runs(token[1:-1] if token.startswith("`") and token.endswith("`") else token, latin_font))
    return "".join(parts)


def export(notebook):
    data = json.loads(notebook.read_text(encoding="utf-8"))
    if data["nbformat"] != 4 or any(c["cell_type"] != "markdown" for c in data["cells"]):
        raise ValueError("Expected a Markdown-only nbformat 4 notebook")
    markdown = "\n\n".join("".join(cell["source"]) for cell in data["cells"]) + "\n"
    notebook.with_suffix(".md").write_text(markdown, encoding="utf-8")
    pdf_path = notebook.with_suffix(".pdf")
    pdfmetrics.registerFont(TTFont("ReportSong", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("ReportHei", "C:/Windows/Fonts/simhei.ttf"))
    pdfmetrics.registerFont(TTFont("ReportRoman", "C:/Windows/Fonts/times.ttf"))
    pdfmetrics.registerFont(TTFont("ReportRomanBold", "C:/Windows/Fonts/timesbd.ttf"))
    common = dict(fontName="ReportSong", fontSize=9.5, leading=13.2, wordWrap="CJK",
                  textColor=colors.black, alignment=TA_LEFT)
    body = ParagraphStyle("Body", **common, spaceAfter=4.0)
    title = ParagraphStyle("Title", **{**common,"fontName":"ReportHei","fontSize":16,"leading":22,"alignment":TA_CENTER},spaceAfter=9)
    heading = ParagraphStyle("Heading", **{**common,"fontName":"ReportHei","fontSize":11.1,"leading":15},spaceBefore=7,spaceAfter=4,keepWithNext=True)
    caption = ParagraphStyle("Caption", **{**common,"fontSize":8.1,"leading":10.8},spaceAfter=5)
    table_text = ParagraphStyle("Table", **{**common,"fontSize":8.3,"leading":11.2},spaceAfter=0)
    table_center = ParagraphStyle("TableCenter", parent=table_text, alignment=TA_CENTER)
    table_header = ParagraphStyle("TableHeader", parent=table_center, fontName="ReportHei")
    width = A4[0] - 72
    story=[]
    lines=markdown.splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=="<!-- pagebreak -->":story.append(PageBreak());i+=1;continue
        image_match=re.fullmatch(r"!\[(.*?)\]\((.+?)\)",line)
        if image_match:
            image_path=notebook.parent / image_match.group(2)
            with PILImage.open(image_path) as img:w,h=img.size
            story.append(Image(str(image_path),width=width,height=width*h/w))
            story.append(Spacer(1,2));i+=1;continue
        if line.startswith("# "):story.append(Paragraph(inline(line[2:], "ReportRomanBold"),title));i+=1;continue
        if line.startswith("## "):story.append(Paragraph(inline(line[3:], "ReportRomanBold"),heading));i+=1;continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                values=[s.strip() for s in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?",s) for s in values):rows.append(values)
                i+=1
            n=len(rows[0])
            col_widths=([50,93,70,70,70] if n==5 else [94,width-94] if n==2 else [72]+[(width-72)/(n-1)]*(n-1))
            if n==5:col_widths=[w*width/sum(col_widths) for w in col_widths]
            cells=[[Paragraph(inline(s, "ReportRomanBold" if row_index==0 else "ReportRoman"),
                              table_header if row_index==0 else table_text if n==2 else table_center)
                    for s in row] for row_index,row in enumerate(rows)]
            table=Table(cells,colWidths=col_widths,hAlign="LEFT",repeatRows=1)
            table.setStyle(TableStyle([("LINEABOVE",(0,0),(-1,0),.8,colors.black),
                ("LINEBELOW",(0,0),(-1,0),.45,colors.black),
                ("LINEBELOW",(0,-1),(-1,-1),.8,colors.black),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),
                ("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),3),
                ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
            story.extend([table,Spacer(1,5)]);continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(("#","|","![]","<!--")):
            para.append(lines[i].strip());i+=1
        text="".join(para)
        story.append(Paragraph(inline(text),caption if text.startswith(("图1","图2","图3","注：","环境：")) else body))

    def footer(canvas,document):
        canvas.saveState()
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(.35)
        canvas.line(36,30,A4[0]-36,30)
        canvas.setFont("ReportSong",8)
        canvas.setFillColor(colors.black)
        canvas.drawString(36,18,"实验一：文本分类 | 实验记录与分析")
        canvas.drawRightString(A4[0]-36,18,str(document.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(pdf_path),pagesize=A4,rightMargin=36,leftMargin=36,
                          topMargin=30,bottomMargin=39,title="实验一：文本分类",author="",allowSplitting=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(pdf_path)
    if len(reader.pages)>2:
        raise ValueError(f"Report exceeds 2 pages: {len(reader.pages)}. Revise content/layout before delivery.")
    print(f"Exported {len(reader.pages)} pages: {pdf_path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("notebook",nargs="?",type=Path,default=ROOT/"reports"/"实验一报告.ipynb")
    export(parser.parse_args().notebook.resolve())
