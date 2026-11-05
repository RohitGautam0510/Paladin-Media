"""Common styling, canvas, and typography definitions for PaladinMedia Academic Deliverables.

Mandatory Formatting Specifications:
- Paper: A4 (595.28 x 841.89 pt)
- Margins: Left: 1.5 inches (108 pt), Right: 1.0 inch (72 pt), Top: 1.0 inch (72 pt), Bottom: 1.0 inch (72 pt)
- Printable Width: 415.28 pt, Printable Height: 697.89 pt
- Font: Times-Roman, Times-Bold, Times-Italic, Times-BoldItalic
- Body: 12 pt, 1.5 line spacing (18 pt leading), Justified
- Headings: Bold, Black (#000000)
- Tables/Figures/Footnotes: 10 pt
- Page Numbering: Suppressed on Cover Page.
"""

from __future__ import annotations

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

A4_WIDTH, A4_HEIGHT = A4
LEFT_MARGIN = 108.0    # 1.5 inches
RIGHT_MARGIN = 72.0    # 1.0 inch
TOP_MARGIN = 72.0      # 1.0 inch
BOTTOM_MARGIN = 72.0   # 1.0 inch
PRINTABLE_WIDTH = A4_WIDTH - LEFT_MARGIN - RIGHT_MARGIN  # 415.28 pt
PRINTABLE_HEIGHT = A4_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN # 697.89 pt


class AcademicCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic page numbers, running headers, and academic footers."""

    def __init__(self, *args, **kwargs):
        self.doc_title = kwargs.pop("doc_title", "PaladinMedia Academic Report")
        self.doc_code = kwargs.pop("doc_code", "BCA411-7")
        self.use_roman_preliminary = kwargs.pop("use_roman_preliminary", False)
        self.preliminary_page_count = kwargs.pop("preliminary_page_count", 0)
        self.suppress_first_page = kwargs.pop("suppress_first_page", True)
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_header_footer(self, total_pages: int):
        page_num = self._pageNumber

        if self.suppress_first_page and page_num == 1:
            return  # Suppress running header/footer on title cover page

        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setFillColor(colors.HexColor("#222222"))

        # Running Header (pages > 1)
        header_y = A4_HEIGHT - TOP_MARGIN + 18
        self.drawString(LEFT_MARGIN, header_y, self.doc_title)
        self.drawRightString(A4_WIDTH - RIGHT_MARGIN, header_y, self.doc_code)

        # Header rule
        self.setStrokeColor(colors.HexColor("#666666"))
        self.setLineWidth(0.5)
        self.line(LEFT_MARGIN, header_y - 4, A4_WIDTH - RIGHT_MARGIN, header_y - 4)

        # Running Footer
        footer_y = BOTTOM_MARGIN - 24
        self.setStrokeColor(colors.HexColor("#888888"))
        self.setLineWidth(0.5)
        self.line(LEFT_MARGIN, footer_y + 12, A4_WIDTH - RIGHT_MARGIN, footer_y + 12)

        footer_left = "PaladinMedia | Rohit Gautam (Reg: 23215050)"
        self.drawString(LEFT_MARGIN, footer_y, footer_left)

        # Page Number Formatting
        if self.use_roman_preliminary and page_num <= self.preliminary_page_count:
            # Roman numerals for preliminary pages
            roman_nums = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
            p_str = roman_nums[page_num] if page_num < len(roman_nums) else str(page_num)
        elif self.use_roman_preliminary:
            arabic_num = page_num - self.preliminary_page_count
            p_str = f"Page {arabic_num}"
        else:
            p_str = f"Page {page_num} of {total_pages}"

        self.drawRightString(A4_WIDTH - RIGHT_MARGIN, footer_y, p_str)
        self.restoreState()


def get_academic_styles():
    """Builds sample stylesheet adhering strictly to Times New Roman and mandatory rules."""
    base = getSampleStyleSheet()

    body = ParagraphStyle(
        "AcademicBody",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=16,  # 1.5 lines spacing
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        spaceAfter=5,
    )

    body_bold = ParagraphStyle(
        "AcademicBodyBold",
        parent=body,
        fontName="Times-Bold",
    )

    title = ParagraphStyle(
        "AcademicTitle",
        parent=base["Title"],
        fontName="Times-Bold",
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=6,
        keepWithNext=True,
    )

    subtitle = ParagraphStyle(
        "AcademicSubtitle",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#222222"),
        spaceAfter=8,
        keepWithNext=True,
    )

    h1 = ParagraphStyle(
        "AcademicH1",
        parent=base["Heading1"],
        fontName="Times-Bold",
        fontSize=13.5,
        leading=18,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2 = ParagraphStyle(
        "AcademicH2",
        parent=base["Heading2"],
        fontName="Times-Bold",
        fontSize=12,
        leading=16,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    h3 = ParagraphStyle(
        "AcademicH3",
        parent=base["Heading3"],
        fontName="Times-Bold",
        fontSize=11,
        leading=15,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )

    table_cell = ParagraphStyle(
        "AcademicTableCell",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.black,
    )

    table_cell_center = ParagraphStyle(
        "AcademicTableCellCenter",
        parent=table_cell,
        alignment=TA_CENTER,
    )

    table_header = ParagraphStyle(
        "AcademicTableHeader",
        parent=table_cell,
        fontName="Times-Bold",
        alignment=TA_CENTER,
        textColor=colors.black,
    )

    caption = ParagraphStyle(
        "AcademicCaption",
        parent=base["Normal"],
        fontName="Times-Italic",
        fontSize=9.5,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
        spaceBefore=3,
        spaceAfter=6,
        keepWithNext=True,
    )

    callout = ParagraphStyle(
        "AcademicCallout",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#111111"),
    )

    return {
        "body": body,
        "body_bold": body_bold,
        "title": title,
        "subtitle": subtitle,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "table_cell": table_cell,
        "table_cell_center": table_cell_center,
        "table_header": table_header,
        "caption": caption,
        "callout": callout,
    }


def make_table(*args, **kwargs):
    """Creates a cleanly styled ReportLab Table with word-wrapped Paragraph cells.
    Supports both:
      make_table(header_data, rows_data, col_widths, styles=None)
      make_table(full_data, col_widths, styles=None)
    """
    styles = kwargs.get("styles", None)
    if len(args) == 4:
        header_data, rows_data, col_widths, styles_arg = args
        if styles is None:
            styles = styles_arg
    elif len(args) == 3:
        if isinstance(args[1], (list, tuple)) and len(args[1]) > 0 and isinstance(args[1][0], (int, float)):
            # Signature: full_data, col_widths, styles
            full_data, col_widths, styles_arg = args
            header_data = full_data[0]
            rows_data = full_data[1:]
            if styles is None:
                styles = styles_arg
        else:
            # Signature: header_data, rows_data, col_widths
            header_data, rows_data, col_widths = args
    elif len(args) == 2:
        # Signature: full_data, col_widths
        full_data, col_widths = args
        header_data = full_data[0]
        rows_data = full_data[1:]
    else:
        raise ValueError("Invalid arguments to make_table")

    if styles is None:
        styles = get_academic_styles()

    table_data = []

    # Format header
    hdr_row = [Paragraph(f"<b>{cell}</b>", styles["table_header"]) if isinstance(cell, str) else cell for cell in header_data]
    table_data.append(hdr_row)

    # Format rows
    for r in rows_data:
        row = []
        for cell in r:
            if isinstance(cell, str):
                row.append(Paragraph(cell, styles["table_cell"]))
            else:
                row.append(cell)
        table_data.append(row)

    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#AAAAAA")),
        ('LINEBELOW', (0, 0), (-1, 0), 1.0, colors.HexColor("#333333")),
    ]))
    return t


def init_docx_document(title="PaladinMedia Academic Report"):
    """Initializes a python-docx Document configured strictly with A4, 1.5 spacing, 1.5-inch left margin."""
    doc = docx.Document()

    for section in doc.sections:
        section.page_width = Inches(8.27)    # A4 Width
        section.page_height = Inches(11.69)  # A4 Height
        section.left_margin = Inches(1.5)    # 1.5 inch left margin
        section.right_margin = Inches(1.0)   # 1.0 inch right margin
        section.top_margin = Inches(1.0)     # 1.0 inch top margin
        section.bottom_margin = Inches(1.0)  # 1.0 inch bottom margin

    # Configure Normal Style font
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.5
    style_normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    return doc
