"""Generates PaladinMedia Week 7 Sprint-1 Testing & Review Document (.docx).

Individual project specification:
- Contributor / Owner: Rohit Gautam
- Dates: 7 – 9 Oct 2026
- Focus: Sprint-1 Feature Testing & Test Log, Individual Retrospective, Updated Backlog
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    """Set inner cell padding in twips."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_cell_shading(cell, color_hex):
    """Set background color of a cell."""
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))


def set_table_borders(table, color="D1D5DB", sz="4", val="single"):
    """Apply clean borders to a table."""
    tblPr = table._tbl.tblPr
    borders_xml = f"""
    <w:tblBorders {nsdecls("w")}>
        <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        <w:insideV w:val="none"/>
        <w:left w:val="none"/>
        <w:right w:val="none"/>
    </w:tblBorders>
    """
    tblPr.append(parse_xml(borders_xml))


def style_header_cell(cell, text, width=None):
    if width:
        cell.width = width
    set_cell_shading(cell, "1A365D")
    set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(9.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(255, 255, 255)


def style_data_cell(cell, text, width=None, bold=False, color="1F2937", bg=None, align=WD_ALIGN_PARAGRAPH.LEFT):
    if width:
        cell.width = width
    if bg:
        set_cell_shading(cell, bg)
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    p = cell.paragraphs[0]
    p.alignment = align
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(9)
    run.font.bold = bold
    r_val = int(color[0:2], 16)
    g_val = int(color[2:4], 16)
    b_val = int(color[4:6], 16)
    run.font.color.rgb = RGBColor(r_val, g_val, b_val)


def build_document():
    doc = docx.Document()

    # Set Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Header Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    run_title = title_p.add_run("PALADINMEDIA — WEEK 7 REPORT")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(26, 54, 93)  # #1A365D

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(14)
    run_sub = sub_p.add_run("Sprint-1 Feature Testing & Review (Test Log, Retrospective & Backlog Update)")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = RGBColor(71, 85, 105)  # #475569

    # Metadata Summary Box / Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(meta_table, color="CBD5E1")

    meta_data = [
        ("Project Title:", "PaladinMedia — Post-Quantum Media Steganography System"),
        ("Milestone Window:", "WEEK 7 | Sprint-1 Testing & Review | 7 – 9 October 2026"),
        ("Individual Contributor / Owner:", "Rohit Gautam (Individual Project)"),
        ("Current Milestone Status:", "Completed & 100% Verified (99/99 Pytest Suite Passing)"),
    ]

    for i, (k, v) in enumerate(meta_data):
        c0 = meta_table.cell(i, 0)
        c1 = meta_table.cell(i, 1)
        bg = "F1F5F9" if i % 2 == 0 else "FFFFFF"
        style_data_cell(c0, k, width=Inches(2.4), bold=True, color="1E293B", bg=bg)
        style_data_cell(c1, v, width=Inches(4.1), bold=(i == 3), color="0F172A" if i != 3 else "15803D", bg=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ----------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY
    # ----------------------------------------------------
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(4)
    r_h1 = h1.add_run("1. Executive Summary & Sprint-1 Testing Context")
    r_h1.font.name = "Calibri"
    r_h1.font.size = Pt(14)
    r_h1.font.bold = True
    r_h1.font.color.rgb = RGBColor(26, 54, 93)

    p1 = doc.add_paragraph()
    p1.paragraph_format.space_after = Pt(8)
    p1.paragraph_format.line_spacing = 1.15
    r_p1 = p1.add_run(
        "During Week 7 (7 – 9 October 2026), the scheduled testing and review cycle for Sprint-1 was "
        "executed. As an individual engineering project, the review and retrospective activities were "
        "conducted through a rigorous self-audit and automated test validation rather than team ceremonies. "
        "All core Sprint-1 deliverables—comprising the Secret Payload Serializer, AES-256-GCM Cryptographic Vault, "
        "Reed-Solomon Error Correction Encoder, and the initial Carrier Wavelet Integration—were subjected to "
        "exhaustive automated unit, integration, and security verification. A total of 99 automated test cases were "
        "executed with a 100% pass rate (0 failures, 0 regressions), successfully verifying complete fail-closed "
        "security, mathematical invertibility, and byte-exact media recovery."
    )
    r_p1.font.name = "Calibri"
    r_p1.font.size = Pt(10)
    r_p1.font.color.rgb = RGBColor(51, 65, 85)

    # ----------------------------------------------------
    # SECTION 2: SPRINT-1 FEATURE TESTING LOG
    # ----------------------------------------------------
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r_h2 = h2.add_run("2. Sprint-1 Feature Testing Log")
    r_h2.font.name = "Calibri"
    r_h2.font.size = Pt(14)
    r_h2.font.bold = True
    r_h2.font.color.rgb = RGBColor(26, 54, 93)

    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(6)
    r_p2 = p2.add_run(
        "The following test log details the formal verification scenarios executed against the Sprint-1 "
        "feature baseline, mapping test cases to their respective components, execution types, results, "
        "and empirical validation metrics:"
    )
    r_p2.font.name = "Calibri"
    r_p2.font.size = Pt(10)
    r_p2.font.color.rgb = RGBColor(51, 65, 85)

    test_log_table = doc.add_table(rows=1, cols=6)
    test_log_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(test_log_table, color="CBD5E1")

    # Header Row
    hdr_cells = test_log_table.rows[0].cells
    headers = [
        ("Test ID", Inches(0.8)),
        ("Feature / Module", Inches(1.3)),
        ("Test Scenario Description", Inches(2.0)),
        ("Type", Inches(0.7)),
        ("Result", Inches(0.6)),
        ("Verified Metric / Behavior", Inches(1.1)),
    ]
    for idx, (title, w) in enumerate(headers):
        style_header_cell(hdr_cells[idx], title, width=w)

    test_cases = [
        ("TL-01", "SecretSerializer", "Binary framing roundtrip with JSON metadata & CRC-32", "Unit", "PASS", "Exact byte match"),
        ("TL-02", "SecretSerializer", "CRC-32 checksum mismatch & corrupted header rejection", "Security", "PASS", "SerializationError"),
        ("TL-03", "AESGCMProvider", "NIST known answer vector encryption & decryption", "Unit", "PASS", "Exact plaintext"),
        ("TL-04", "AESGCMProvider", "Single-bit ciphertext tampering & tag modification", "Security", "PASS", "Fail-closed (0 leak)"),
        ("TL-05", "AESGCMProvider", "Mismatched Associated Authenticated Data (AAD)", "Security", "PASS", "AuthenticationError"),
        ("TL-06", "ReedSolomonEncoder", "GF(2^8) RS(255, 223) encoding with 32 parity bytes", "Unit", "PASS", "32B parity generated"),
        ("TL-07", "ReedSolomonEncoder", "Correct symbol errors within t=16 threshold", "Integration", "PASS", "100% error corrected"),
        ("TL-08", "ReedSolomonEncoder", "Catastrophic error injection (>16 symbols corrupted)", "Security", "PASS", "ECCDecodeError"),
        ("TL-09", "CapacityCalculator", "Hard-gate capacity verification at 8,176 B ceiling", "Unit", "PASS", "PayloadCapacityExceeded"),
        ("TL-10", "HaarDWT2D", "One-level 2D Haar DWT / IDWT floating-point invertibility", "Unit", "PASS", "Max error = 0.000000"),
        ("TL-11", "CarrierHeader", "16-byte fixed canonical framing codec (b'PLDH')", "Unit", "PASS", "Pack/unpack identity"),
        ("TL-12", "MediaStegoPipeline", "Full roundtrip: Secret -> Stego PNG -> Reload -> Recover", "E2E", "PASS", "PSNR: 75.89 dB, BER: 0.0"),
    ]

    for i, row in enumerate(test_cases):
        row_cells = test_log_table.add_row().cells
        bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
        style_data_cell(row_cells[0], row[0], width=Inches(0.8), bold=True, color="334155", bg=bg)
        style_data_cell(row_cells[1], row[1], width=Inches(1.3), bold=False, color="0F172A", bg=bg)
        style_data_cell(row_cells[2], row[2], width=Inches(2.0), bold=False, color="475569", bg=bg)
        style_data_cell(row_cells[3], row[3], width=Inches(0.7), bold=False, color="64748B", bg=bg)
        style_data_cell(row_cells[4], row[4], width=Inches(0.6), bold=True, color="16A34A", bg=bg)
        style_data_cell(row_cells[5], row[5], width=Inches(1.1), bold=False, color="0369A1", bg=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ----------------------------------------------------
    # SECTION 3: INDIVIDUAL SPRINT RETROSPECTIVE
    # ----------------------------------------------------
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(4)
    r_h3 = h3.add_run("3. Individual Sprint Retrospective (Self-Review)")
    r_h3.font.name = "Calibri"
    r_h3.font.size = Pt(14)
    r_h3.font.bold = True
    r_h3.font.color.rgb = RGBColor(26, 54, 93)

    p3 = doc.add_paragraph()
    p3.paragraph_format.space_after = Pt(6)
    r_p3 = p3.add_run(
        "As an individual project, the sprint review and retrospective is structured as a dedicated "
        "engineering self-assessment focusing on technical outcomes, obstacles overcome, and process refinements:"
    )
    r_p3.font.name = "Calibri"
    r_p3.font.size = Pt(10)
    r_p3.font.color.rgb = RGBColor(51, 65, 85)

    retro_table = doc.add_table(rows=1, cols=3)
    retro_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(retro_table, color="CBD5E1")

    retro_headers = [
        ("Retrospective Category", Inches(1.8)),
        ("Detailed Engineering Observations", Inches(3.2)),
        ("Action Items / Future Impact", Inches(1.5)),
    ]
    for idx, (title, w) in enumerate(retro_headers):
        style_header_cell(retro_table.rows[0].cells[idx], title, width=w)

    retro_entries = [
        (
            "What Went Well\n(Successes & Strengths)",
            "• Strict unidirectional pipeline (Serialize -> Encrypt -> ECC -> Embed) established without architectural leakage.\n"
            "• Absolute fail-closed cryptographic implementation verified: zero plaintext released under all tampering tests.\n"
            "• Zero bit errors (BER = 0.0) achieved across 8-bit RGB PNG disk reload using integer-parity modulation on Haar coefficients.\n"
            "• High carrier fidelity achieved: 75.89 dB PSNR and 1.000000 SSIM on baseline carrier images.",
            "Maintain frozen cryptographic contracts and preserve existing unit test suite across future phases."
        ),
        (
            "Challenges Faced\n(Obstacles & Resolutions)",
            "• Theoretical vs Usable Capacity Gap: Initial assumption of 8,192 B capacity collided with real protocol overheads. "
            "Resolved by modeling explicit capacity accounting (8,176 B protected, ~7,088 B usable secret).\n"
            "• Oversized Payload Handling: Large images (e.g. 791 KB) triggered capacity failures. "
            "Resolved by strictly enforcing the capacity hard-gate (PayloadCapacityExceededError) before transform execution.",
            "Incorporate adaptive lossless compression (zlib) and multi-carrier transport protocol (MCPP) for large payloads."
        ),
        (
            "Key Learnings &\nProcess Refinements",
            "• Automated test-driven verification eliminated regressions early.\n"
            "• Separating classical signal processing from future neural research prevented premature scope creep.\n"
            "• Disk I/O (PNG encode/decode) accounts for >75% of total roundtrip latency (~102 ms out of 134 ms total).",
            "Optimize image buffer operations in memory; prepare data tensors cleanly for Phase 3 neural engine integration."
        ),
    ]

    for i, row in enumerate(retro_entries):
        row_cells = retro_table.add_row().cells
        bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
        style_data_cell(row_cells[0], row[0], width=Inches(1.8), bold=True, color="1E293B", bg=bg)
        style_data_cell(row_cells[1], row[1], width=Inches(3.2), bold=False, color="334155", bg=bg)
        style_data_cell(row_cells[2], row[2], width=Inches(1.5), bold=False, color="0F766E", bg=bg)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ----------------------------------------------------
    # SECTION 4: UPDATED PROJECT BACKLOG
    # ----------------------------------------------------
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(12)
    h4.paragraph_format.space_after = Pt(4)
    r_h4 = h4.add_run("4. Updated Project Backlog")
    r_h4.font.name = "Calibri"
    r_h4.font.size = Pt(14)
    r_h4.font.bold = True
    r_h4.font.color.rgb = RGBColor(26, 54, 93)

    p4 = doc.add_paragraph()
    p4.paragraph_format.space_after = Pt(6)
    r_p4 = p4.add_run(
        "Following the Sprint-1 review and retrospective, the master project backlog has been formally updated. "
        "Sprint-1 work items are finalized as DONE, while upcoming Sprint-2 and Phase 3 items are groomed and prioritized:"
    )
    r_p4.font.name = "Calibri"
    r_p4.font.size = Pt(10)
    r_p4.font.color.rgb = RGBColor(51, 65, 85)

    backlog_table = doc.add_table(rows=1, cols=6)
    backlog_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(backlog_table, color="CBD5E1")

    backlog_headers = [
        ("Ticket ID", Inches(1.0)),
        ("Task / User Story", Inches(2.2)),
        ("Sprint / Phase", Inches(1.0)),
        ("Priority", Inches(0.7)),
        ("Status", Inches(0.7)),
        ("Resolution / Milestone Note", Inches(0.9)),
    ]
    for idx, (title, w) in enumerate(backlog_headers):
        style_header_cell(backlog_table.rows[0].cells[idx], title, width=w)

    backlog_items = [
        ("PDM-S1-001", "Freeze module contracts and wire specifications", "Sprint-1", "P0", "Done", "Contracts frozen"),
        ("PDM-S1-002", "Implement AESGCMProvider (AES-256-GCM)", "Sprint-1", "P0", "Done", "NIST vectors pass"),
        ("PDM-S1-003", "Fail-closed tamper handling and zero-leak verification", "Sprint-1", "P0", "Done", "All negative tests pass"),
        ("PDM-S1-004", "Implement ReedSolomonEncoder (RS GF(2^8))", "Sprint-1", "P0", "Done", "Corrects t=16 symbols"),
        ("PDM-S1-005", "Implement SecretSerializer with CRC-32 integrity", "Sprint-1", "P0", "Done", "b'PLDS' framing"),
        ("PDM-S1-006", "Sprint-1 Cryptographic integration test suite", "Sprint-1", "P0", "Done", "99/99 pytest green"),
        ("PDM-S2-001", "Haar DWT/IDWT 2D transform engine & capacity accounting", "Sprint-2", "P0", "Done", "Max error = 0.0"),
        ("PDM-S2-002", "16-byte fixed CarrierHeader codec (b'PLDH')", "Sprint-2", "P0", "Done", "Anchor positions 0..127"),
        ("PDM-S2-003", "Non-circular DeterministicMaskGenerator via SHA-256 PRNG", "Sprint-2", "P0", "Done", "Seed-reproducible"),
        ("PDM-S2-004", "Baseline Integer-Parity Embedding & Extraction engine", "Sprint-2", "P0", "Done", "BER = 0.0, PSNR = 75.89 dB"),
        ("PDM-S2-005", "Phase 2 Experimental Program Execution (E03 to E11)", "Sprint-2", "P0", "Done", "Capacity frontier mapped"),
        ("PDM-S3-001", "ACS-Sub-INN Neural Invertible Flow Architecture", "Sprint-3 (Phase 3)", "P1", "Planned", "Pending Phase 3 kickoff"),
        ("PDM-S3-002", "Context Predictor CNN & Adaptive Gain Network", "Sprint-3 (Phase 3)", "P1", "Planned", "Wavelet feature input"),
        ("PDM-S3-003", "Straight-Through Estimator (STE) Quantization Noise Layer", "Sprint-3 (Phase 3)", "P2", "Planned", "Lossy container training"),
    ]

    for i, row in enumerate(backlog_items):
        row_cells = backlog_table.add_row().cells
        bg = "F8FAFC" if i % 2 == 1 else "FFFFFF"
        style_data_cell(row_cells[0], row[0], width=Inches(1.0), bold=True, color="334155", bg=bg)
        style_data_cell(row_cells[1], row[1], width=Inches(2.2), bold=False, color="0F172A", bg=bg)
        style_data_cell(row_cells[2], row[2], width=Inches(1.0), bold=False, color="64748B", bg=bg)
        style_data_cell(row_cells[3], row[3], width=Inches(0.7), bold=True, color="B45309" if row[3] == "P0" else "475569", bg=bg)
        
        status_color = "15803D" if row[4] == "Done" else ("0369A1" if row[4] == "Planned" else "B45309")
        style_data_cell(row_cells[4], row[4], width=Inches(0.7), bold=True, color=status_color, bg=bg)
        style_data_cell(row_cells[5], row[5], width=Inches(0.9), bold=False, color="334155", bg=bg)

    # Output paths
    out_dir = os.path.join("docs", "project")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "PaladinMedia_Week7_Sprint1_Testing_and_Review.docx")
    doc.save(out_path)

    # Also save a copy in root for convenience
    root_path = "PaladinMedia_Week7_Sprint1_Testing_and_Review.docx"
    doc.save(root_path)

    print(f"Document successfully created:")
    print(f"  -> {out_path}")
    print(f"  -> {root_path}")
    return out_path


if __name__ == "__main__":
    build_document()
