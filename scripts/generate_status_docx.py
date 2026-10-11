"""
Generate PaladinMedia Project Status and Completion Report in Word (.docx) format.
Applies standard IMRaD font formatting and font size hierarchy:
- Title: 20pt Bold
- Metadata/Subtitle: 11pt
- Section Headings (H1): 14pt Bold
- Subsection Headings (H2): 12pt Bold
- Body Paragraphs: 11pt Regular, 1.15 line spacing
- Tables: 9.5pt (Headers Bold, Rows Regular)
- Captions/Footers: 8.5pt-9pt
Structure: Practical Engineering Project Status & Completion Report (Not academic IMRaD sections).
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, hex_color):
    """Set background color of a table cell."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    """Set padding for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """Set custom borders for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}/>')
    borders = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    for b_name, b_val in borders.items():
        if b_val:
            val, sz, space, color = b_val
            node = parse_xml(f'<w:{b_name} {nsdecls("w")} w:val="{val}" w:sz="{sz}" w:space="{space}" w:color="{color}"/>')
            tcBorders.append(node)
        else:
            node = parse_xml(f'<w:{b_name} {nsdecls("w")} w:val="none"/>')
            tcBorders.append(node)
    tcPr.append(tcBorders)

def add_callout(doc, text, title="KEY INVARIANT"):
    """Add a professional styled callout block."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.6)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    set_cell_borders(cell, left=("single", "24", "0", "1E3A8A"),
                           top=("none", "0", "0", "auto"),
                           bottom=("none", "0", "0", "auto"),
                           right=("none", "0", "0", "auto"))
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run_title = p.add_run(f"[{title}] ")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(10)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(30, 58, 138)
    
    run_text = p.add_run(text)
    run_text.font.name = "Calibri"
    run_text.font.size = Pt(10)
    run_text.font.italic = True
    run_text.font.color.rgb = RGBColor(51, 65, 85)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_status_docx(filename="PaladinMedia_Project_Status_Report.docx"):
    doc = docx.Document()
    
    # 0.85 in margins
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        
        # Header & Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("PALADINMEDIA | Project Status & Completion Report")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(148, 163, 184)
        
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("PaladinMedia Engineering Project Status Report — Internal & Academic Baseline")
        frun.font.name = "Calibri"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(148, 163, 184)

    navy = RGBColor(30, 58, 138)
    dark_slate = RGBColor(30, 41, 59)
    teal = RGBColor(15, 118, 110)
    muted_slate = RGBColor(100, 116, 139)

    # Document Title Block (20pt Bold)
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    run_tag = title_p.add_run("PALADINMEDIA ENGINEERING STATUS AUDIT\n")
    run_tag.font.name = "Calibri"
    run_tag.font.size = Pt(11)
    run_tag.font.bold = True
    run_tag.font.color.rgb = teal
    
    run_title = title_p.add_run("Project Status and Completion Report")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = navy
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(12)
    run_sub = sub_p.add_run("Detailed Engineering Status, Implementation Verification, and Milestone Completion Audit")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(11.5)
    run_sub.font.color.rgb = muted_slate
    
    # Metadata Block Table
    meta_table = doc.add_table(rows=4, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    col_widths = [Inches(1.5), Inches(1.8), Inches(1.5), Inches(1.8)]
    meta_data = [
        [("Project Name", True), ("PaladinMedia", False), ("Current Baseline", True), ("Foundation MVP 0 (Phase 2)", False)],
        [("Automated Tests", True), ("99 / 99 Passed (100%)", False), ("Sprint 1 Checks", True), ("24 / 24 Passed (100%)", False)],
        [("Report Date", True), ("October 2026", False), ("Software Revision", True), ("v0.2.0 (Phase 2 Baseline)", False)],
        [("Core Completion", True), ("100% (Phases 0–2)", False), ("Overall Progress", True), ("~45% (Full 6-Phase Scope)", False)]
    ]
    for r_idx, row in enumerate(meta_table.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths[c_idx]
            label, is_header = meta_data[r_idx][c_idx]
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            if is_header:
                set_cell_background(cell, "F1F5F9")
                set_cell_borders(cell, top=("single", "4", "0", "CBD5E1"), bottom=("single", "4", "0", "CBD5E1"))
            else:
                set_cell_background(cell, "FFFFFF")
                set_cell_borders(cell, top=("single", "4", "0", "E2E8F0"), bottom=("single", "4", "0", "CBD5E1"))
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(label)
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.font.bold = is_header
            r.font.color.rgb = navy if is_header else dark_slate
            
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Typography helpers
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = navy
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = teal
        return p

    def add_body(text, space_after=6):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(11)
        r.font.color.rgb = dark_slate
        return p

    # -------------------------------------------------------------
    # 1. Executive Summary & Project Objectives
    # -------------------------------------------------------------
    add_h1("1. Executive Summary & Project Objectives")
    add_body(
        "PaladinMedia is an advanced media security platform designed to conceal sensitive digital payloads "
        "(text, JSON configurations, high-entropy binaries, and raster images) inside benign-looking 512×512 RGB carrier images. "
        "The system solves the vulnerability of conventional encryption—which alerts network monitors by exposing high-entropy ciphertext—"
        "while simultaneously overcoming the fragility and lack of cryptographic security inherent in traditional steganography."
    )
    add_body(
        "The project enforces a core architectural rule: \"Security Before Concealment\". Sensitive content is not merely hidden; "
        "it is first framed with canonical wire headers and integrity checksums, encrypted using authenticated AES-256-GCM, fortified against "
        "channel noise using systematic Reed-Solomon GF(2^8) error correction, and only then embedded into the discrete wavelet domain (2D Haar DWT) "
        "of a carrier image. On the receiver side, a strictly symmetric extraction pipeline performs error correction, authenticates "
        "the cryptographic tag, and recovers the exact original payload with 100% byte fidelity."
    )
    add_callout(
        doc,
        "All claims and milestones in PaladinMedia are governed by the strict engineering mandate: "
        "'Hypothesis ≠ Implementation ≠ Experiment ≠ Validation'. Performance standards are never stated as completed without automated, "
        "deterministic test results.",
        title="ENGINEERING PRINCIPLE"
    )

    # -------------------------------------------------------------
    # 2. Current Project Implementation Status
    # -------------------------------------------------------------
    add_h1("2. Current Implementation Status")
    add_body(
        "The software architecture has been successfully engineered across three primary operational subsystems: "
        "the Secret Cryptographic Pipeline, the Transform-Domain Carrier Pipeline, and the Receiver Verification Pipeline."
    )
    
    add_h2("2.1 Cryptographic & Security Pipeline (Phase 1 Baseline)")
    add_body(
        "• Secret Serialization (src/core/serialization/serializer.py): Encapsulates payloads into canonical wire frames with "
        "magic bytes b'PLDS', format versioning, media type identifiers, payload length fields, and CRC-32 integrity trailers.\n"
        "• Authenticated Cryptography (src/security/crypto/provider.py): Utilizes AES-256-GCM authenticated cipher with 96-bit CSPRNG "
        "nonces, 128-bit GHASH authentication tags, and transcript-bound Additional Authenticated Data (AAD). Guarantees zero plaintext leakage "
        "upon tampering.\n"
        "• Galois Field Error Correction (src/security/ecc/encoder.py): Applies systematic Reed-Solomon GF(2^8) chunking with 2t=32 parity symbols "
        "per 255-byte block. Successfully corrects up to t=16 symbol errors (128 contiguous flipped bits) per block."
    )

    add_h2("2.2 Carrier Signal Processing & Embedding Pipeline (Phase 2 Baseline)")
    add_body(
        "• 2D Wavelet Decomposition (src/carrier/dwt.py): Performs one-level 2D Haar Discrete Wavelet Transform (DWT) in float64 precision, "
        "decomposing 512×512 RGB carrier images into LL, LH, HL, and HH subbands (3×256×256 shape) with zero numerical inversion loss.\n"
        "• Capacity Accounting & Hard Gating (src/carrier/capacity.py): A 512×512 carrier offers 65,536 bits (8,192 bytes) of raw space "
        "in the Red channel of the HH subband. Reserving 16 bytes for the carrier header leaves 8,176 bytes maximum protected payload "
        "(~7,088 bytes of secret data after RS, AES, and wire overheads). Payloads exceeding this threshold are immediately rejected by a hard gate.\n"
        "• Carrier Header & Deterministic Masking (src/carrier/header.py & mask.py): Injects a 16-byte fixed canonical header (b'PLDH') "
        "into the first 128 positions. Computes pseudo-random payload positions across the HH subband using a session-derived PRNG seed.\n"
        "• Wavelet Embedder & Extractor (src/carrier/embedding.py & extraction.py): Modulates integer parity on 2×HH coefficients with "
        "carrier pixel adjustments (±1 pixel), saving lossless 8-bit PNG stego containers that survive image roundtripping."
    )

    add_h2("2.3 Application & Observability Infrastructure")
    add_body(
        "• Backend Server & Event Bus (src/api/server.py & event_adapter.py): A lightweight FastAPI server exposing REST endpoints "
        "for conceal/extract actions and streaming server-sent events for pipeline telemetry.\n"
        "• Laboratory Frontend UI (frontend/): A React + TypeScript + Vite interactive laboratory interface scaffolded with Tailwind CSS, "
        "ready for coefficient visualization and interactive steganography analysis."
    )

    # -------------------------------------------------------------
    # 3. Verification & Empirical Testing Results
    # -------------------------------------------------------------
    add_h1("3. Verification & Empirical Testing Results")
    add_body(
        "The project has undergone rigorous automated testing across three independent evaluation harnesses:"
    )

    # Results Table (9.5pt table typography)
    res_table = doc.add_table(rows=9, cols=4)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_table.autofit = False
    r_widths = [Inches(1.8), Inches(1.3), Inches(1.3), Inches(2.2)]
    r_headers = ["Testing Area / Metric", "Measured Value", "Target Threshold", "Verification Result"]
    
    for c_idx, cell in enumerate(res_table.rows[0].cells):
        cell.width = r_widths[c_idx]
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(r_headers[c_idx])
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    r_data = [
        ["Automated Pytest Suite", "99 / 99 Passed (100%)", "100% Pass", "PASSED: Full unit & integration coverage across all modules (2.38s)."],
        ["Sprint 1 Invariant Verification", "24 / 24 Passed (100%)", "24 Checks", "PASSED: Multi-media reversibility, bounds, and fail-closed security."],
        ["DWT Mathematical Invertibility", "0.000000 max error", "0.000000", "PASSED: Float64 Haar transform exhibits zero numerical drift."],
        ["Peak Signal-to-Noise Ratio (PSNR)", "75.71 dB", "> 50.0 dB", "PASSED: Carrier distortion is invisible; exceeds perceptual stealth bar."],
        ["Structural Similarity (SSIM)", "1.00000", "> 0.999", "PASSED: Identical image structure preserved across all 3 color channels."],
        ["Pre-ECC Bit Error Rate (BER)", "0.000000", "< 0.01", "PASSED: Zero bit flips introduced through PNG carrier storage and reload."],
        ["Reed-Solomon Fault Tolerance", "Repairs t=16 symbols", "t ≤ 16 symbols", "PASSED: Corrects burst errors up to 128 consecutive flipped bits."],
        ["End-to-End Secret Recovery", "100.0% Byte Match", "Formal Equality", "PASSED: original_secret.data == recovered_secret.data is strictly TRUE."]
    ]
    for r_idx, row_data in enumerate(r_data):
        row = res_table.rows[r_idx + 1]
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, text in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.width = r_widths[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            set_cell_borders(cell, top=("single", "4", "0", "E2E8F0"), bottom=("single", "4", "0", "CBD5E1"))
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.font.bold = (c_idx == 0 or c_idx == 1)
            if c_idx == 1:
                r.font.color.rgb = teal
            else:
                r.font.color.rgb = dark_slate

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------
    # 4. Project Completion Status & Roadmap
    # -------------------------------------------------------------
    add_h1("4. Project Completion Status & Roadmap")
    add_body(
        "The PaladinMedia master architecture is structured into seven distinct phases (Phase 0 through Phase 6). "
        "The following completion matrix audits the exact delivery status of each phase:"
    )

    # Completion Table
    comp_table = doc.add_table(rows=8, cols=5)
    comp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    comp_table.autofit = False
    c_widths = [Inches(0.9), Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.7)]
    c_headers = ["Phase", "Milestone & Scope", "Target Baseline", "Completion", "Current Status & Deliverables"]
    
    for c_idx, cell in enumerate(comp_table.rows[0].cells):
        cell.width = c_widths[c_idx]
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(c_headers[c_idx])
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        
    c_data = [
        ["Phase 0", "Sprint-0 Readiness & Specs", "Governance Docs", "100%", "COMPLETE: Product Vision, Architecture, and IRS frozen."],
        ["Phase 1", "Cryptographic & ECC Core", "Sprint 1 Foundation", "100%", "COMPLETE: AES-256-GCM, RS GF(2^8), fail-closed security verified."],
        ["Phase 2", "Signal Processing & DWT Carrier", "Foundation MVP 0", "100%", "COMPLETE: Haar DWT/IDWT, mask, E03 verified (99 tests pass)."],
        ["Phase 3", "ACS-Sub-INN Neural Predictor", "Context Predictor CNN", "0%", "PENDING: Architecture specified; awaiting training harness."],
        ["Phase 4", "Model Training on DIV2K", "800 DIV2K Images", "0%", "PENDING: DIV2K dataset staged in workspace; ready for training."],
        ["Phase 5", "Edge Optimization & QAT", "ONNX Opset 14 / INT8", "0%", "PENDING: Quantization-aware training after neural model convergence."],
        ["Phase 6", "Laboratory GUI & Telemetry", "FastAPI + Vite + Energy", "35%", "IN PROGRESS: FastAPI server operational; Vite UI scaffolded."]
    ]
    for r_idx, row_data in enumerate(c_data):
        row = comp_table.rows[r_idx + 1]
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, text in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.width = c_widths[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            set_cell_borders(cell, top=("single", "4", "0", "E2E8F0"), bottom=("single", "4", "0", "CBD5E1"))
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.05
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.font.bold = (c_idx == 0 or c_idx == 3)
            if c_idx == 3:
                r.font.color.rgb = teal if "100%" in text else (RGBColor(180, 83, 9) if "35%" in text else muted_slate)
            else:
                r.font.color.rgb = dark_slate

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_h2("4.1 Completion Assessment Summary")
    add_body(
        "• Core Foundational Architecture (MVP 0 — Phases 0, 1, and 2): 100% COMPLETE AND VERIFIED. "
        "The end-to-end sender-to-receiver pipeline (serialization → AES-256-GCM encryption → Reed-Solomon encoding → "
        "2D Haar DWT → deterministic PRNG masking → carrier embedding → IDWT → lossless PNG storage → symmetric extraction → "
        "decryption → 100% exact byte recovery) is fully engineered, covered by 99 automated tests, and validated.\n"
        "• Overall Research Scope (Phases 0 through 6): ~45% COMPLETE. While the core steganographic and cryptographic engines are "
        "production-ready and verified, advanced neural learning (ACS-Sub-INN), edge deployment (ONNX INT8 QAT), and carbon telemetry "
        "represent the remaining forward research scope."
    )

    # -------------------------------------------------------------
    # 5. Observed Boundaries, Limitations & Next Actions
    # -------------------------------------------------------------
    add_h1("5. Observed Boundaries, Limitations & Next Actions")
    add_body(
        "• Lossless Container Boundary: The current baseline is validated across lossless PNG storage. It does not claim robustness against "
        "hostile lossy recompression (JPEG) or geometric scaling, which will be tackled during Phase 3 neural robustness studies.\n"
        "• Capacity Limit: Current baseline is constrained to ~7,088 bytes of secret data per 512×512 carrier image in 1 HH subband. "
        "Oversized payloads are safely rejected by the hard capacity gate without corrupting carrier coefficients.\n"
        "• Non-Claims: PaladinMedia is an academic research platform; it makes no claims of operational military certification or unconditional 'unbreakability'."
    )
    add_body(
        "Immediate Next Actions:\n"
        "1. Complete Vite UI connection to the FastAPI backend to provide live browser-based carrier inspection.\n"
        "2. Add CodeCarbon energy telemetry to track Joules and carbon emissions per embedding run.\n"
        "3. Launch Phase 3 PyTorch implementation of the ACS-Sub-INN context predictor on the DIV2K training set."
    )

    doc.save(filename)
    print(f"Successfully generated DOCX report: {filename}")

if __name__ == "__main__":
    build_status_docx("E:/Paladin_media/PaladinMedia_Project_Status_Report.docx")
    build_status_docx("E:/Paladin_media/PaladinMedia_Project_Status_Report_IMRaD.docx")
    os.makedirs("E:/Paladin_media/docs/reports", exist_ok=True)
    build_status_docx("E:/Paladin_media/docs/reports/PaladinMedia_Project_Status_Report.docx")
    build_status_docx("E:/Paladin_media/docs/reports/PaladinMedia_Project_Status_Report_IMRaD.docx")
