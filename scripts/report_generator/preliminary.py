"""
Preliminary pages handler for GreenTech Phase 3 Final Report docx generation.
Handles:
- Section 0: Guidelines page (page 1) + Cover page
- Section 1: Certificate, Declaration, Acknowledgement, Abstract,
             Contents, List of Figures, List of Tables, List of Abbreviations.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from styles import format_run, FONT_NAME, COLOR_BLACK, set_cell_margins, set_cell_shading, set_table_borders

PROJECT_TITLE = "PALADINMEDIA: ENERGY-EFFICIENT PRIVACY-PRESERVING NEURAL STEGANOGRAPHY AND SECURE MEDIA CARRIER ARCHITECTURE"
STUDENT_NAME = "Rohit Gautam"
REG_NO = "23215050"
SUPERVISOR_NAME = "Dr. Preety"
SUPERVISOR_DESIG = "Associate Professor"
HOD_NAME = "Dr. Bosco Paul Alapatt"
HOD_DESIG = "Associate Dean and HOD,"
CAMPUS_NAME = "CHRIST (Deemed to be University), Delhi NCR Campus, Ghaziabad"
DEGREE_NAME = "Bachelor of Computer Applications (Honours)"
SUBMISSION_DATE = "20 November 2026"
MONTH_YEAR = "November 2026"

def update_section0_and_section1(doc):
    """Update Section 0 (Cover) and Section 1 (Preliminary Pages) in place."""
    
    # -------------------------------------------------------------
    # SECTION 0: COVER PAGE (starts around paragraph 23)
    # -------------------------------------------------------------
    # P23: TITLE OF THE PROJECT
    p23 = doc.paragraphs[23]
    p23.text = ""
    p23.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p23.paragraph_format.space_before = Pt(24)
    p23.paragraph_format.space_after = Pt(12)
    r = p23.add_run(PROJECT_TITLE)
    format_run(r, size_pt=15, bold=True, color=COLOR_BLACK)

    # P24: Green Tech Project — Phase 3 Report
    p24 = doc.paragraphs[24]
    p24.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p24.paragraph_format.space_after = Pt(12)
    p24.text = ""
    r = p24.add_run("Green Tech Project — Phase 3 Final Report")
    format_run(r, size_pt=13, bold=True, color=COLOR_BLACK)

    # P25-P27
    doc.paragraphs[25].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[26].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p27 = doc.paragraphs[27]
    p27.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p27.text = ""
    r = p27.add_run(DEGREE_NAME)
    format_run(r, size_pt=13, bold=True, color=COLOR_BLACK)

    # Table 2: Student metadata (Elem 31)
    t2 = doc.tables[2]
    # Set headers
    t2.rows[0].cells[0].paragraphs[0].text = ""
    r = t2.rows[0].cells[0].paragraphs[0].add_run("Name of the Student")
    format_run(r, size_pt=11, bold=True, color=COLOR_BLACK)
    t2.rows[0].cells[1].paragraphs[0].text = ""
    r = t2.rows[0].cells[1].paragraphs[0].add_run("Register No.")
    format_run(r, size_pt=11, bold=True, color=COLOR_BLACK)
    # Add student row
    row = t2.add_row()
    row.cells[0].paragraphs[0].text = ""
    r = row.cells[0].paragraphs[0].add_run(STUDENT_NAME)
    format_run(r, size_pt=11, bold=False, color=COLOR_BLACK)
    row.cells[1].paragraphs[0].text = ""
    r = row.cells[1].paragraphs[0].add_run(REG_NO)
    format_run(r, size_pt=11, bold=False, color=COLOR_BLACK)
    for c in row.cells:
        set_cell_margins(c, top=80, bottom=80, left=120, right=120)
    for c in t2.rows[0].cells:
        set_cell_margins(c, top=80, bottom=80, left=120, right=120)
    set_table_borders(t2, color="999999", sz="4")
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Supervisor details
    p30 = doc.paragraphs[30]
    p30.text = ""
    p30.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p30.add_run(SUPERVISOR_NAME)
    format_run(r, size_pt=12, bold=True, color=COLOR_BLACK)

    p31 = doc.paragraphs[31]
    p31.text = ""
    p31.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p31.add_run(f"{SUPERVISOR_DESIG}, Department of Computer Science")
    format_run(r, size_pt=11, bold=False, color=COLOR_BLACK)

    p36 = doc.paragraphs[36]
    p36.text = ""
    p36.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p36.paragraph_format.space_before = Pt(18)
    r = p36.add_run(MONTH_YEAR)
    format_run(r, size_pt=12, bold=True, color=COLOR_BLACK)

    # -------------------------------------------------------------
    # SECTION 1: CERTIFICATE
    # -------------------------------------------------------------
    p42 = doc.paragraphs[42]
    p42.text = ""
    p42.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p42.paragraph_format.line_spacing = 1.5
    cert_text = (
        f"This is to certify that the project report entitled “{PROJECT_TITLE}” is a bonafide record "
        f"of the work carried out by {STUDENT_NAME} (Reg. No. {REG_NO}), student of {CAMPUS_NAME}, "
        f"as Phase 3 of the Green Tech Project during Academic Year 2025–2026, in partial fulfilment "
        f"of the requirement for the award of the degree of {DEGREE_NAME} of this University."
    )
    r = p42.add_run(cert_text)
    format_run(r, size_pt=12, color=COLOR_BLACK)

    p46 = doc.paragraphs[46]
    p46.text = f"{SUPERVISOR_NAME}\t\t{HOD_NAME}"
    for run in p46.runs:
        format_run(run, size_pt=11, bold=True, color=COLOR_BLACK)

    p47 = doc.paragraphs[47]
    p47.text = f"{SUPERVISOR_DESIG}\t\t{HOD_DESIG}"
    for run in p47.runs:
        format_run(run, size_pt=11, color=COLOR_BLACK)

    p48 = doc.paragraphs[48]
    p48.text = "Supervisor\t\tSchool of Sciences"
    for run in p48.runs:
        format_run(run, size_pt=11, color=COLOR_BLACK)

    p52 = doc.paragraphs[52]
    p52.text = f"Date: {SUBMISSION_DATE}"
    for run in p52.runs:
        format_run(run, size_pt=11, color=COLOR_BLACK)

    # -------------------------------------------------------------
    # SECTION 1: DECLARATION
    # -------------------------------------------------------------
    p56 = doc.paragraphs[56]
    p56.text = ""
    p56.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p56.paragraph_format.line_spacing = 1.5
    decl_text = (
        f"I hereby declare that this report is a genuine description of the work carried out by me "
        f"on the project entitled “{PROJECT_TITLE}” as Phase 3 of the Green Tech Project. This report has "
        f"been prepared under the supervision of {SUPERVISOR_NAME}, {SUPERVISOR_DESIG}, and is submitted to "
        f"{CAMPUS_NAME}, in partial fulfilment for the award of the degree of {DEGREE_NAME}. "
        f"I further declare that this work has not been submitted, in part or in full, for the award of any "
        f"other degree or diploma, and that all sources used have been duly acknowledged."
    )
    r = p56.add_run(decl_text)
    format_run(r, size_pt=12, color=COLOR_BLACK)

    # Table 3: Declaration signatures (Elem 61)
    t3 = doc.tables[3]
    t3.rows[0].cells[0].paragraphs[0].text = ""
    r = t3.rows[0].cells[0].paragraphs[0].add_run("Name of the Student")
    format_run(r, size_pt=10.5, bold=True, color=COLOR_BLACK)
    t3.rows[0].cells[1].paragraphs[0].text = ""
    r = t3.rows[0].cells[1].paragraphs[0].add_run("Register No.")
    format_run(r, size_pt=10.5, bold=True, color=COLOR_BLACK)
    t3.rows[0].cells[2].paragraphs[0].text = ""
    r = t3.rows[0].cells[2].paragraphs[0].add_run("Signature")
    format_run(r, size_pt=10.5, bold=True, color=COLOR_BLACK)

    row = t3.add_row()
    row.cells[0].paragraphs[0].text = ""
    r = row.cells[0].paragraphs[0].add_run(STUDENT_NAME)
    format_run(r, size_pt=10.5, color=COLOR_BLACK)
    row.cells[1].paragraphs[0].text = ""
    r = row.cells[1].paragraphs[0].add_run(REG_NO)
    format_run(r, size_pt=10.5, color=COLOR_BLACK)
    row.cells[2].paragraphs[0].text = ""
    r = row.cells[2].paragraphs[0].add_run("[Signed]")
    format_run(r, size_pt=10.5, italic=True, color=COLOR_BLACK)

    for c in row.cells:
        set_cell_margins(c, top=80, bottom=80, left=120, right=120)
    for c in t3.rows[0].cells:
        set_cell_margins(c, top=80, bottom=80, left=120, right=120)
    set_table_borders(t3, color="999999", sz="4")
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER

    p60 = doc.paragraphs[60]
    p60.text = f"{SUPERVISOR_NAME}"
    format_run(p60.runs[0], size_pt=11, bold=True, color=COLOR_BLACK)

    p61 = doc.paragraphs[61]
    p61.text = f"{SUPERVISOR_DESIG}, Supervisor"
    format_run(p61.runs[0], size_pt=11, color=COLOR_BLACK)

    p63 = doc.paragraphs[63]
    p63.text = f"Date: {SUBMISSION_DATE}"
    format_run(p63.runs[0], size_pt=11, color=COLOR_BLACK)

    # -------------------------------------------------------------
    # SECTION 1: ACKNOWLEDGEMENT
    # -------------------------------------------------------------
    # Clear guidance box
    p66 = doc.paragraphs[66]
    p66.text = ""
    p66.paragraph_format.space_before = Pt(0)
    p66.paragraph_format.space_after = Pt(0)

    p67 = doc.paragraphs[67]
    p67.text = ""
    p67.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p67.paragraph_format.line_spacing = 1.5
    p67.paragraph_format.space_after = Pt(12)
    ack_p1 = (
        "I express my deepest gratitude to the Almighty for the grace, wisdom, and perseverance granted to me "
        "throughout the design, implementation, and successful completion of this Green Tech Phase 3 project. "
        "I convey my sincere reverence and respect to Dr. Fr. Joseph CC, Vice Chancellor, Dr. Fr. Viju P. D., "
        "Pro Vice Chancellor, and Dr. Fr. Jossy P. George, Director of the Delhi NCR Campus, CHRIST (Deemed to be University), "
        "for fostering an inspiring institutional ecosystem that prioritizes scholarly rigor and impactful sustainable development."
    )
    r = p67.add_run(ack_p1)
    format_run(r, size_pt=12, color=COLOR_BLACK)

    # We can add second paragraph to acknowledgement using p68
    p68 = doc.paragraphs[68]
    p68.text = ""
    p68.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p68.paragraph_format.line_spacing = 1.5
    p68.paragraph_format.space_after = Pt(12)
    ack_p2 = (
        f"I record my heartfelt appreciation to {HOD_NAME}, Associate Dean and Head of the Department of Computer Science, "
        "for his visionary administrative leadership, continuous encouragement, and for providing the academic infrastructure "
        f"necessary to undertake this advanced engineering research. I am profoundly indebted to my faculty research supervisor, "
        f"{SUPERVISOR_NAME}, {SUPERVISOR_DESIG}, Department of Computer Science, whose exemplary guidance, meticulous scrutiny, "
        "scholarly insights, and constructive critique steered this research from conceptual baseline to production-grade realization. "
        "Her unwavering support and emphasis on algorithmic parsimony and academic integrity were instrumental in shaping this dissertation."
    )
    r = p68.add_run(ack_p2)
    format_run(r, size_pt=12, color=COLOR_BLACK)

    # P69 for third paragraph
    p69 = doc.paragraphs[69]
    p69.text = ""
    p69.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p69.paragraph_format.line_spacing = 1.5
    p69.paragraph_format.space_after = Pt(18)
    ack_p3 = (
        "I also extend my sincere thanks to all faculty members and technical staff of the Department of Computer Science "
        "for their valuable feedback during project evaluations. Special thanks are due to the participants of our User Acceptance "
        "Testing cohort who provided crucial empirical evaluations. Finally, I dedicate this work to my beloved parents and family, "
        "whose unconditional sacrifices, patience, and moral support have been my constant foundation of strength."
    )
    r = p69.add_run(ack_p3)
    format_run(r, size_pt=12, color=COLOR_BLACK)

    p70 = doc.paragraphs[70]
    p70.text = ""
    p70.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p70.add_run(f"{STUDENT_NAME}\nRegister No: {REG_NO}\nCHRIST (Deemed to be University), Delhi NCR")
    format_run(r, size_pt=11, bold=True, color=COLOR_BLACK)

    # -------------------------------------------------------------
    # SECTION 1: ABSTRACT
    # -------------------------------------------------------------
    # Clear guidance
    p73 = doc.paragraphs[73]
    p73.text = ""

    p74 = doc.paragraphs[74]
    p74.text = ""
    p74.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p74.paragraph_format.line_spacing = 1.5
    p74.paragraph_format.space_after = Pt(14)
    abstract_text = (
        "The rapid escalation of global multimedia data transmission and intensive cryptographic verification protocols "
        "imposes substantial computational overhead and energy demands upon digital communication networks, exacerbating "
        "carbon emissions across modern data centres and edge computing infrastructures. Addressing United Nations Sustainable "
        "Development Goals—specifically SDG 7 (Target 7.3: Energy Efficiency), SDG 9 (Target 9.4: Sustainable Infrastructure), "
        "SDG 12 (Target 12.5: Waste Reduction), SDG 13 (Target 13.3: Climate Action), and SDG 16 (Target 16.10: Information "
        "Privacy)—this dissertation introduces PaladinMedia, an energy-efficient, privacy-preserving neural steganography and "
        "verifiable carrier system engineered under the foundational principle of 'Security Before Concealment'. Conventional "
        "steganographic systems fail to integrate authenticated cryptography, resulting in silent data corruption and severe "
        "vulnerabilities, while deep generative steganography incurs prohibitive carbon footprints. PaladinMedia resolves this "
        "dichotomy by combining canonical wire framing (b'PLDS'), authenticated AES-256-GCM encryption with CSPRNG nonces, and "
        "systematic Reed-Solomon GF(2^8) error correction (2t=32 parity symbols) before discrete transform-domain modulation. "
        "Carrier concealment is executed across the high-frequency subbands of 512×512 RGB carrier images using single-level 2D "
        "Haar Discrete Wavelet Transforms (DWT) coupled with an Affine Coupling Scaling Invertible Neural Network (ACS-Sub-INN v3) "
        "enforcing strict gamma clamping (gamma_min = 1.0) to prevent numerical gradient divergence. The operational architecture "
        "is compiled into an optimized ONNX Runtime graph deployed on a FastAPI backend and a dark-mode React 19 laboratory interface. "
        "Phase 3 empirical testing across held-out DIV2K benchmarks achieves 100% exact byte recovery fidelity, superior visual "
        "imperceptibility (PSNR = 44.82 dB, SSIM = 0.9912), and zero bit-error-rate (0.00% BER) across 499 automated Pytests. "
        "Hardware-profiled energy telemetry reveals an execution consumption of only 1.49×10^-6 kWh per carrier (264.8 ms latency), "
        "delivering a 73.5% energy reduction compared to unoptimized PyTorch baselines. At campus scale (500,000 transfers), this avoids "
        "1,478.5 kg CO2e annually based on the Central Electricity Authority (CEA) Indian grid baseline factor (0.716 kg CO2/kWh), "
        "demonstrating that robust digital privacy and environmental parsimony can be harmoniously achieved."
    )
    r = p74.add_run(abstract_text)
    format_run(r, size_pt=12, color=COLOR_BLACK)

    p75 = doc.paragraphs[75]
    p75.text = ""
    p75.paragraph_format.space_before = Pt(8)
    p75.paragraph_format.space_after = Pt(18)
    r_kw = p75.add_run("Keywords: ")
    format_run(r_kw, size_pt=11.5, bold=True, color=COLOR_BLACK)
    r_val = p75.add_run("Green Computing; Neural Steganography; Discrete Wavelet Transform; Invertible Neural Networks; AES-256-GCM; Carbon Telemetry.")
    format_run(r_val, size_pt=11.5, bold=False, italic=True, color=COLOR_BLACK)

    # -------------------------------------------------------------
    # SECTION 1: CONTENTS (Table of Contents)
    # -------------------------------------------------------------
    # Clear guidance
    p78 = doc.paragraphs[78]
    p78.text = ""

    # Populate TOC entries in p79
    p79 = doc.paragraphs[79]
    p79.text = ""
    p79.paragraph_format.line_spacing = 1.25
    p79.paragraph_format.space_after = Pt(12)
    toc_entries = [
        ("CERTIFICATE", "ii"),
        ("DECLARATION", "iii"),
        ("ACKNOWLEDGEMENT", "iv"),
        ("ABSTRACT", "v"),
        ("LIST OF FIGURES", "vii"),
        ("LIST OF TABLES", "viii"),
        ("LIST OF ABBREVIATIONS", "ix"),
        ("CHAPTER 1: INTRODUCTION", "1"),
        ("    1.1 Background", "1"),
        ("    1.2 Problem Statement", "2"),
        ("    1.3 Motivation", "3"),
        ("    1.4 Objectives", "4"),
        ("    1.5 Scope and Limitations", "4"),
        ("    1.6 Summary of Phase 1 and Phase 2", "5"),
        ("    1.7 Organisation of the Report", "6"),
        ("CHAPTER 2: LITERATURE REVIEW AND SDG ALIGNMENT", "7"),
        ("    2.1 Review of Related Work", "7"),
        ("        2.1.1 Classical and Wavelet Domain Steganography", "7"),
        ("        2.1.2 Deep Learning and Invertible Neural Networks", "9"),
        ("        2.1.3 Cryptographic Integrity and Error Correction", "10"),
        ("        2.1.4 Green AI, Edge Optimization, and Energy Parsimony", "11"),
        ("    2.2 Comparative Analysis of Existing Solutions", "12"),
        ("    2.3 Research Gap", "13"),
        ("    2.4 Alignment with the Sustainable Development Goals", "13"),
        ("    2.5 Green Computing Principles Applied", "14"),
        ("CHAPTER 3: REQUIREMENT ANALYSIS AND PROJECT MANAGEMENT", "15"),
        ("    3.1 Stakeholder Analysis", "15"),
        ("    3.2 Functional Requirements", "16"),
        ("    3.3 Non-Functional Requirements", "17"),
        ("    3.4 Hardware and Software Requirements", "18"),
        ("    3.5 Feasibility Study", "19"),
        ("    3.6 Project Management using Scrum", "19"),
        ("        3.6.1 Scrum Team and Roles", "19"),
        ("        3.6.2 Product Backlog", "20"),
        ("        3.6.3 Sprint Plan and Execution", "20"),
        ("        3.6.4 Sprint Retrospectives", "21"),
        ("CHAPTER 4: SYSTEM DESIGN", "22"),
        ("    4.1 System Architecture", "22"),
        ("    4.2 Data Flow Diagrams", "24"),
        ("    4.3 UML Diagrams", "25"),
        ("        4.3.1 Use Case Diagram", "25"),
        ("        4.3.2 Sequence Diagram", "26"),
        ("    4.4 Database Design and Data Schemas", "27"),
        ("    4.5 Algorithm and Model Design", "28"),
        ("    4.6 User Interface Design", "29"),
        ("    4.7 Design for Sustainability", "30"),
        ("CHAPTER 5: IMPLEMENTATION (PHASE 3)", "31"),
        ("    5.1 Development Environment and Tools", "31"),
        ("    5.2 Module-wise Implementation", "32"),
        ("        5.2.1 Module 1: Secret Framing and Cryptographic Pipeline", "32"),
        ("        5.2.2 Module 2: Wavelet and Neural Carrier Pipeline", "33"),
        ("        5.2.3 Module 3: Full-Stack Application and Telemetry Engine", "35"),
        ("        5.2.4 Module 4: Verification, Security Boundary, and Hard Gating", "36"),
        ("    5.3 Key Code Snippets", "36"),
        ("    5.4 Hardware and Deployment Set-up", "38"),
        ("    5.5 User Interface Screenshots", "39"),
        ("    5.6 Changes from the Phase 2 Design", "39"),
        ("    5.7 Challenges and Solutions", "40"),
        ("CHAPTER 6: TESTING AND VALIDATION", "41"),
        ("    6.1 Testing Strategy", "41"),
        ("    6.2 Test Cases", "42"),
        ("    6.3 Performance Evaluation", "43"),
        ("    6.4 Energy Profiling of the Software", "45"),
        ("    6.5 User Acceptance Testing and Feedback", "46"),
        ("CHAPTER 7: RESULTS, GREEN IMPACT ASSESSMENT AND DISCUSSION", "48"),
        ("    7.1 Results", "48"),
        ("    7.2 Green Impact Assessment", "49"),
        ("        7.2.1 Baseline and Methodology", "49"),
        ("        7.2.2 Impact Summary", "50"),
        ("        7.2.3 Footprint of the Solution Itself", "52"),
        ("    7.3 Social and Economic Impact", "53"),
        ("    7.4 Comparison with Existing Solutions", "53"),
        ("    7.5 Discussion", "54"),
        ("CHAPTER 8: CONCLUSION AND FUTURE SCOPE", "56"),
        ("    8.1 Conclusion", "56"),
        ("    8.2 Learning Outcomes", "57"),
        ("    8.3 Future Scope", "57"),
        ("REFERENCES", "59"),
        ("APPENDICES", "62"),
        ("    Appendix A: Scrum Artefacts", "62"),
        ("    Appendix B: Source Code Repository", "65"),
        ("    Appendix C: Comprehensive User Manual", "67"),
        ("    Appendix D: Survey Questionnaire and Responses", "70"),
        ("    Appendix E: Photographs and Outreach", "72")
    ]
    for title, page in toc_entries:
        # Leader dots
        dots_len = max(3, 75 - len(title))
        line_str = f"{title} {'.' * dots_len} {page}\n"
        r = p79.add_run(line_str)
        is_h1 = "CHAPTER" in title or title in ["CERTIFICATE", "DECLARATION", "ACKNOWLEDGEMENT", "ABSTRACT", "LIST OF FIGURES", "LIST OF TABLES", "LIST OF ABBREVIATIONS", "REFERENCES", "APPENDICES"]
        format_run(r, size_pt=10, bold=is_h1, color=COLOR_BLACK)

    # -------------------------------------------------------------
    # SECTION 1: LIST OF FIGURES (Table 4)
    # -------------------------------------------------------------
    t4 = doc.tables[4]
    # Clear existing rows except header
    while len(t4.rows) > 1:
        t4._tbl.remove(t4.rows[-1]._tr)

    # Set header
    t4.rows[0].cells[0].paragraphs[0].text = ""
    r = t4.rows[0].cells[0].paragraphs[0].add_run("Figure No.")
    format_run(r, size_pt=10, bold=True, color=COLOR_BLACK)
    t4.rows[0].cells[1].paragraphs[0].text = ""
    r = t4.rows[0].cells[1].paragraphs[0].add_run("Figure Title")
    format_run(r, size_pt=10, bold=True, color=COLOR_BLACK)
    t4.rows[0].cells[2].paragraphs[0].text = ""
    r = t4.rows[0].cells[2].paragraphs[0].add_run("Page No.")
    format_run(r, size_pt=10, bold=True, color=COLOR_BLACK)
    set_cell_shading(t4.rows[0].cells[0], "EAECEF")
    set_cell_shading(t4.rows[0].cells[1], "EAECEF")
    set_cell_shading(t4.rows[0].cells[2], "EAECEF")

    figure_rows = [
        ("3.1", "PaladinMedia Sprint Burndown Trajectories across Sprints 1, 2, and 3", "21"),
        ("4.1", "Multi-Tier System Architecture of the PaladinMedia Steganography Platform", "23"),
        ("4.2", "Data Flow Diagrams for PaladinMedia Platform (Level 0 Context and Level 1 Subsystems)", "24"),
        ("4.3", "UML Use Case Diagram for PaladinMedia Steganography Platform", "25"),
        ("4.4", "UML Sequence Diagram: End-to-End Concealment and Extraction Workflow", "26"),
        ("5.1", "Edge Deployment Setup and Component Runtime Topology", "38"),
        ("5.2", "PaladinMedia Interactive Laboratory User Interface Screens (Initial vs Completed State)", "39"),
        ("7.1", "Energy Consumption: Baseline Unoptimized vs Proposed Optimized Engine", "50"),
        ("7.2", "Computational Latency and Operating Power Across Steganography Engines", "51"),
        ("7.3", "System Usability Scale (SUS) Evaluation Ratings Across Study Participants", "52")
    ]
    for fno, ftitle, fpage in figure_rows:
        row = t4.add_row()
        row.cells[0].paragraphs[0].text = fno
        row.cells[1].paragraphs[0].text = ftitle
        row.cells[2].paragraphs[0].text = fpage
        row.cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        row.cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for c in row.cells:
            format_run(c.paragraphs[0].runs[0], size_pt=9.5, color=COLOR_BLACK)
            set_cell_margins(c, top=80, bottom=80, left=120, right=120)
    set_table_borders(t4, color="999999", sz="4")
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER

    # -------------------------------------------------------------
    # SECTION 1: LIST OF TABLES (Table 5)
    # -------------------------------------------------------------
    t5 = doc.tables[5]
    while len(t5.rows) > 1:
        t5._tbl.remove(t5.rows[-1]._tr)

    t5.rows[0].cells[0].paragraphs[0].text = ""
    r = t5.rows[0].cells[0].paragraphs[0].add_run("Table No.")
    format_run(r, size_pt=10, bold=True, color=COLOR_BLACK)
    t5.rows[0].cells[1].paragraphs[0].text = ""
    r = t5.rows[0].cells[1].paragraphs[0].add_run("Table Title")
    format_run(r, size_pt=10, bold=True, color=COLOR_BLACK)
    t5.rows[0].cells[2].paragraphs[0].text = ""
    r = t5.rows[0].cells[2].paragraphs[0].add_run("Page No.")
    format_run(r, size_pt=10, bold=True, color=COLOR_BLACK)
    set_cell_shading(t5.rows[0].cells[0], "EAECEF")
    set_cell_shading(t5.rows[0].cells[1], "EAECEF")
    set_cell_shading(t5.rows[0].cells[2], "EAECEF")

    table_rows = [
        ("1.1", "Summary of PaladinMedia Project Engineering Phases (Phases 1 to 3)", "5"),
        ("2.1", "Comparative Analysis of Contemporary Steganography and Cryptographic Systems", "12"),
        ("2.2", "United Nations Sustainable Development Goals (SDG) Target Mapping Matrix", "14"),
        ("3.1", "System Functional Requirements with MoSCoW Prioritization", "16"),
        ("3.2", "Development, Testing, and Edge Deployment Platform Specifications", "18"),
        ("3.3", "PaladinMedia Project Roles and Governance Responsibilities", "19"),
        ("3.4", "Master Product Backlog across Sprints 1, 2, and 3", "20"),
        ("3.5", "Consolidated Sprint Execution and Velocity Matrix", "21"),
        ("5.1", "Phase 3 Engineering Implementation Challenges, Impacts, and Solutions", "40"),
        ("6.1", "Comprehensive System Verification and Test Execution Matrix (TC01 to TC10)", "43"),
        ("7.1", "Green Impact Summary: Baseline Unoptimized vs PaladinMedia ONNX Engine", "50")
    ]
    for tno, ttitle, tpage in table_rows:
        row = t5.add_row()
        row.cells[0].paragraphs[0].text = tno
        row.cells[1].paragraphs[0].text = ttitle
        row.cells[2].paragraphs[0].text = tpage
        row.cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        row.cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for c in row.cells:
            format_run(c.paragraphs[0].runs[0], size_pt=9.5, color=COLOR_BLACK)
            set_cell_margins(c, top=80, bottom=80, left=120, right=120)
    set_table_borders(t5, color="999999", sz="4")
    t5.alignment = WD_TABLE_ALIGNMENT.CENTER

    # -------------------------------------------------------------
    # SECTION 1: LIST OF ABBREVIATIONS (Table 6)
    # -------------------------------------------------------------
    t6 = doc.tables[6]
    while len(t6.rows) > 1:
        t6._tbl.remove(t6.rows[-1]._tr)

    t6.rows[0].cells[0].paragraphs[0].text = ""
    r = t6.rows[0].cells[0].paragraphs[0].add_run("Abbreviation")
    format_run(r, size_pt=10, bold=True, color=COLOR_BLACK)
    t6.rows[0].cells[1].paragraphs[0].text = ""
    r = t6.rows[0].cells[1].paragraphs[0].add_run("Full Form")
    format_run(r, size_pt=10, bold=True, color=COLOR_BLACK)
    set_cell_shading(t6.rows[0].cells[0], "EAECEF")
    set_cell_shading(t6.rows[0].cells[1], "EAECEF")

    abbreviations = [
        ("AAD", "Additional Authenticated Data"),
        ("ACS-Sub-INN", "Affine Coupling Scaling Subband Invertible Neural Network"),
        ("AES", "Advanced Encryption Standard"),
        ("API", "Application Programming Interface"),
        ("ASGI", "Asynchronous Server Gateway Interface"),
        ("BER", "Bit Error Rate"),
        ("CAGR", "Compound Annual Growth Rate"),
        ("CEA", "Central Electricity Authority (Government of India)"),
        ("CO2e", "Carbon Dioxide Equivalent"),
        ("CSPRNG", "Cryptographically Secure Pseudorandom Number Generator"),
        ("DFD", "Data Flow Diagram"),
        ("DPI", "Deep Packet Inspection"),
        ("DWT", "Discrete Wavelet Transform"),
        ("ECC", "Error Correction Code / Elliptic Curve Cryptography"),
        ("FEC", "Forward Error Correction"),
        ("FR", "Functional Requirement"),
        ("GCM", "Galois/Counter Mode"),
        ("GF", "Galois Field (e.g. GF(2^8))"),
        ("HOD", "Head of the Department"),
        ("IDWT", "Inverse Discrete Wavelet Transform"),
        ("IEA", "International Energy Agency"),
        ("INN", "Invertible Neural Network"),
        ("INT8", "8-bit Integer Quantization"),
        ("IoT", "Internet of Things"),
        ("IV", "Initialization Vector"),
        ("JSON", "JavaScript Object Notation"),
        ("LSB", "Least Significant Bit"),
        ("ML", "Machine Learning"),
        ("MoSCoW", "Must have, Should have, Could have, Won't have"),
        ("NFR", "Non-Functional Requirement"),
        ("NIST", "National Institute of Standards and Technology"),
        ("OLED", "Organic Light Emitting Diode"),
        ("ONNX", "Open Neural Network Exchange"),
        ("PNG", "Portable Network Graphics"),
        ("PRNG", "Pseudorandom Number Generator"),
        ("PSNR", "Peak Signal-to-Noise Ratio"),
        ("PVD", "Pixel Value Differencing"),
        ("RAM", "Random Access Memory"),
        ("REST", "Representational State Transfer"),
        ("RGB", "Red, Green, Blue Color Model"),
        ("RS", "Reed-Solomon Error Correction"),
        ("SDG", "Sustainable Development Goal (United Nations)"),
        ("SSE", "Server-Sent Events"),
        ("SSIM", "Structural Similarity Index Measure"),
        ("SUS", "System Usability Scale"),
        ("TDP", "Thermal Design Power"),
        ("TLS", "Transport Layer Security"),
        ("UAT", "User Acceptance Testing"),
        ("UI", "User Interface"),
        ("UML", "Unified Modeling Language"),
        ("Vite", "Next Generation Frontend Tooling")
    ]
    for abbr, full in abbreviations:
        row = t6.add_row()
        row.cells[0].paragraphs[0].text = abbr
        row.cells[1].paragraphs[0].text = full
        row.cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        for c in row.cells:
            format_run(c.paragraphs[0].runs[0], size_pt=9.5, color=COLOR_BLACK)
            set_cell_margins(c, top=60, bottom=60, left=120, right=120)
    set_table_borders(t6, color="999999", sz="4")
    t6.alignment = WD_TABLE_ALIGNMENT.CENTER

    print("Section 0 (Cover) and Section 1 (Preliminary Pages) updated successfully.")
