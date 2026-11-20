"""
Styling and formatting utilities for GreenTech Phase 3 Final Report docx generation.
Adheres strictly to the Report Writing Guidelines:
- Font: Times New Roman throughout (Consolas for code snippets)
- Font sizes: Chapter title 16 pt bold (centred, CAPITALS); section heading 14 pt bold;
  sub-section 12 pt bold; body text 12 pt
- Line spacing: 1.5 for body text; single spacing inside tables, captions, equations, and references
- Alignment: Body text justified
- Margins: Left 1.5 in (binding), Top 1.0 in, Bottom 1.0 in, Right 1.0 in
- Tables: Caption ABOVE table, numbered by chapter (e.g. Table 3.1)
- Figures: Caption BELOW figure, numbered by chapter (e.g. Figure 4.1)
- Equations: Numbered at right margin e.g. (4.1)
- Code snippets: <= 15 lines, Consolas 9.5 pt, shaded box
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

FONT_NAME = "Times New Roman"
COLOR_BLACK = RGBColor(0, 0, 0)
COLOR_TEXT = RGBColor(17, 24, 39)

def format_run(run, font_name=FONT_NAME, size_pt=12, bold=False, italic=False, color=COLOR_BLACK):
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color

def add_chapter_title(doc, title_text):
    """Add 16 pt bold centred CAPITALS chapter title with page break before."""
    p = doc.add_paragraph()
    p.paragraph_format.page_break_before = True
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(14)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title_text.upper())
    format_run(run, size_pt=16, bold=True, color=COLOR_BLACK)
    return p

def add_section_heading(doc, text):
    """Add 14 pt bold section heading (Heading 2)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    format_run(run, size_pt=14, bold=True, color=COLOR_BLACK)
    return p

def add_subsection_heading(doc, text):
    """Add 12 pt bold sub-section heading (Heading 3)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    format_run(run, size_pt=12, bold=True, color=COLOR_BLACK)
    return p

def add_body_p(doc, text, bold_prefix=None, space_after=6):
    """Add 12 pt body paragraph with 1.5 line spacing, justified."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        format_run(r_pre, size_pt=12, bold=True, color=COLOR_BLACK)
        
    run = p.add_run(text)
    format_run(run, size_pt=12, bold=False, color=COLOR_BLACK)
    return p

def add_bullet_p(doc, text, bold_prefix=None):
    """Add bullet paragraph with 1.5 line spacing and indent."""
    try:
        p = doc.add_paragraph(style='List Paragraph')
    except Exception:
        p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.left_indent = Inches(0.25)
    
    r_bullet = p.add_run("• ")
    format_run(r_bullet, size_pt=12, bold=True, color=COLOR_BLACK)
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        format_run(r_pre, size_pt=12, bold=True, color=COLOR_BLACK)
        
    run = p.add_run(text)
    format_run(run, size_pt=12, bold=False, color=COLOR_BLACK)
    return p

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    """Set inner padding for table cell in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    """Set background fill for table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    """Apply clean academic borders to table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_table_with_caption(doc, caption_text, headers, rows_data, col_widths=None):
    """Add a professional academic table with caption ABOVE the table."""
    # Caption ABOVE table
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_before = Pt(12)
    cp.paragraph_format.space_after = Pt(4)
    cp.paragraph_format.keep_with_next = True
    c_run = cp.add_run(caption_text)
    format_run(c_run, size_pt=10.5, bold=True, color=COLOR_BLACK)

    # Table creation
    table = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="999999", sz="6")

    # Header Row
    hdr_row = table.rows[0]
    hdr_row._tr.get_or_add_trPr().append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    for idx, text in enumerate(headers):
        cell = hdr_row.cells[idx]
        set_cell_shading(cell, "EAECEF")
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(text)
        format_run(r, size_pt=10, bold=True, color=COLOR_BLACK)

    # Data Rows
    for r_idx, r_data in enumerate(rows_data):
        row = table.rows[r_idx + 1]
        bg_col = "FAFAFA" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(r_data):
            cell = row.cells[c_idx]
            if bg_col != "FFFFFF":
                set_cell_shading(cell, bg_col)
            set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
            p = cell.paragraphs[0]
            # align center for short codes / numbers, left for descriptions
            val_str = str(val)
            if len(val_str) <= 12 and not any(ch.isalpha() and len(val_str.split()) > 1 for ch in val_str):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(val_str)
            format_run(r, size_pt=9.5, bold=False, color=COLOR_BLACK)

    # Apply column widths if provided
    if col_widths and len(col_widths) == len(headers):
        for row in table.rows:
            for c_idx, w in enumerate(col_widths):
                row.cells[c_idx].width = Inches(w)

    # Trailing spacing
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(6)
    return table

def add_figure_with_caption(doc, image_path, caption_text, width_inches=5.4):
    """Add figure image centered with caption BELOW the figure."""
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(12)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.keep_with_next = True
    
    run_img = p_img.add_run()
    run_img.add_picture(image_path, width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(12)
    run_cap = p_cap.add_run(caption_text)
    format_run(run_cap, size_pt=10, bold=True, color=COLOR_BLACK)
    return p_cap

def add_equation(doc, eq_latex_or_text, eq_number):
    """Add clean academic equation with right-aligned numbering e.g. (4.1)."""
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Col 0: Equation (centered)
    c0 = table.rows[0].cells[0]
    c0.width = Inches(4.8)
    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.paragraph_format.space_before = Pt(4)
    p0.paragraph_format.space_after = Pt(4)
    p0.paragraph_format.line_spacing = 1.0
    r0 = p0.add_run(eq_latex_or_text)
    format_run(r0, font_name="Times New Roman", size_pt=11, italic=True, color=COLOR_BLACK)

    # Col 1: Equation number (right aligned)
    c1 = table.rows[0].cells[1]
    c1.width = Inches(0.9)
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p1.paragraph_format.space_before = Pt(4)
    p1.paragraph_format.space_after = Pt(4)
    p1.paragraph_format.line_spacing = 1.0
    r1 = p1.add_run(f"({eq_number})")
    format_run(r1, font_name="Times New Roman", size_pt=11, bold=True, color=COLOR_BLACK)

    # Remove borders
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:insideH w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)

def add_code_listing(doc, caption_text, code_lines):
    """Add short code listing (<= 15 lines) in Consolas 9.5 pt with shaded box."""
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(10)
    p_cap.paragraph_format.space_after = Pt(2)
    p_cap.paragraph_format.keep_with_next = True
    r_cap = p_cap.add_run(caption_text)
    format_run(r_cap, size_pt=10, bold=True, color=COLOR_BLACK)

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Inches(5.7)
    set_cell_shading(cell, "F6F8FA")
    set_cell_margins(cell, top=120, bottom=120, left=160, right=160)

    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        f'<w:left w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    p_code = cell.paragraphs[0]
    p_code.paragraph_format.space_before = Pt(0)
    p_code.paragraph_format.space_after = Pt(0)
    p_code.paragraph_format.line_spacing = 1.05
    for idx, line in enumerate(code_lines):
        r_line = p_code.add_run(line + ("\n" if idx < len(code_lines) - 1 else ""))
        format_run(r_line, font_name="Consolas", size_pt=9.0, color=RGBColor(36, 41, 47))

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(6)
