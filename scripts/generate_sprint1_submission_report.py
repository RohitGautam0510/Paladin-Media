"""Generate PaladinMedia CIA-3 Sprint-1 Submission Report in DOCX and PDF formats.

Produces:
1. PaladinMedia_CIA3_Sprint1_Report.docx
2. PaladinMedia_CIA3_Sprint1_Report.pdf (Exactly 5 Pages)

Sections:
Page 1 - Project Overview and Sprint-1 Scope
Page 2 - Sprint Backlog and Feature Status
Page 3 - Technical Design & Architecture
Page 4 - Green Computing Practices & Test Log
Page 5 - Retrospective, Contributions & Next-Sprint Backlog
"""

from __future__ import annotations

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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

OUT_DIR = os.path.abspath(r"E:\Paladin_media\submission\CIA3_Sprint1")
os.makedirs(OUT_DIR, exist_ok=True)
PDF_PATH = os.path.join(OUT_DIR, "PaladinMedia_CIA3_Sprint1_Report.pdf")
DOCX_PATH = os.path.join(OUT_DIR, "PaladinMedia_CIA3_Sprint1_Report.docx")


# =============================================================================
# REPORTLAB PDF GENERATION (5 PAGES EXACT)
# =============================================================================

class Sprint1ReportCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic 'Page X of Y' footer and consistent academic header."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1E3A8A"))
        
        # Header (Top)
        self.drawString(36, 762, "PALADINMEDIA")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(104, 762, "|  CIA-3 Sprint-1 Submission Report  •  Academic Engineering Baseline")
        self.drawRightString(576, 762, "Course: BCA411-7  |  Reg No: 23215050")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 756, 576, 756)
        
        # Footer (Bottom)
        self.line(36, 32, 576, 32)
        self.setFont("Helvetica", 7.2)
        self.drawString(36, 21, "PaladinMedia Sprint-1 Audit Package  •  School of Sciences  •  Dr Preety (Supervisor)")
        self.drawRightString(576, 21, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf(filename=PDF_PATH):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=38
    )

    styles = getSampleStyleSheet()
    
    c_navy = colors.HexColor("#1E3A8A")
    c_slate = colors.HexColor("#0F172A")
    c_dark = colors.HexColor("#1E293B")
    c_muted = colors.HexColor("#475569")
    c_border = colors.HexColor("#CBD5E1")
    c_header_bg = colors.HexColor("#1E3A8A")
    c_row_alt = colors.HexColor("#F8FAFC")
    c_green = colors.HexColor("#047857")
    c_blue = colors.HexColor("#2563EB")
    c_amber = colors.HexColor("#D97706")
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12.5, leading=15, textColor=c_navy, spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.0, leading=10, textColor=c_muted, spaceAfter=4
    )
    h1_style = ParagraphStyle(
        'SecH1', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9.0, leading=11, textColor=c_navy, spaceBefore=4, spaceAfter=2
    )
    h2_style = ParagraphStyle(
        'SecH2', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=7.8, leading=9.5, textColor=c_slate, spaceBefore=3, spaceAfter=1.5
    )
    body_style = ParagraphStyle(
        'BodyDark', parent=styles['Normal'],
        fontName='Helvetica', fontSize=6.8, leading=8.6, textColor=c_dark, spaceAfter=2.5
    )
    body_bold = ParagraphStyle(
        'BodyBold', parent=body_style, fontName='Helvetica-Bold'
    )
    th_style = ParagraphStyle(
        'TH', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=6.2, leading=7.5, textColor=colors.white, alignment=0
    )
    td_style = ParagraphStyle(
        'TD', parent=styles['Normal'],
        fontName='Helvetica', fontSize=5.9, leading=7.2, textColor=c_dark
    )
    td_bold = ParagraphStyle(
        'TDBold', parent=td_style, fontName='Helvetica-Bold'
    )
    td_code = ParagraphStyle(
        'TDCode', parent=td_style, fontName='Courier', fontSize=5.6, leading=6.8
    )
    td_green = ParagraphStyle(
        'TDGreen', parent=td_style, fontName='Helvetica-Bold', textColor=c_green
    )
    td_amber = ParagraphStyle(
        'TDAmber', parent=td_style, fontName='Helvetica-Bold', textColor=c_amber
    )
    td_blue = ParagraphStyle(
        'TDBlue', parent=td_style, fontName='Helvetica-Bold', textColor=c_blue
    )
    callout_box_style = ParagraphStyle(
        'Callout', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=6.6, leading=8.4, textColor=c_navy
    )
    code_block_style = ParagraphStyle(
        'CodeBlock', parent=styles['Normal'],
        fontName='Courier', fontSize=5.8, leading=7.0, textColor=colors.HexColor("#0F172A")
    )

    story = []

    # =========================================================================
    # PAGE 1: PROJECT OVERVIEW & SPRINT-1 SCOPE
    # =========================================================================
    story.append(Paragraph("PALADINMEDIA: POST-QUANTUM MEDIA STEGANOGRAPHY & TRANSFORM CONCEALMENT", title_style))
    story.append(Paragraph("CIA-3 Sprint-1 Verification Report  •  Functional Prototype Audit & Baseline Handoff", subtitle_style))
    
    # Metadata Strip
    meta_table = [
        [
            Paragraph("<b>Contributor:</b> Rohit Gautam (Reg No: 23215050)", td_style),
            Paragraph("<b>Course / Dept:</b> BCA411-7 / School of Sciences", td_style),
            Paragraph("<b>Supervisor:</b> Dr Preety", td_style),
            Paragraph("<b>Deadline:</b> 11 Oct 2026, 11:59 PM", td_style)
        ],
        [
            Paragraph("<b>Current Baseline:</b> MVP 0 / Phase 2 Classical", td_style),
            Paragraph("<b>Pytest Suite:</b> 103 / 103 Passing (100%)", td_green),
            Paragraph("<b>Sprint-1 Checks:</b> 24 / 24 Passing (100%)", td_green),
            Paragraph("<b>Git Commits:</b> 9555996, a8c3f76 (Clean)", td_style)
        ]
    ]
    t_meta = Table(meta_table, colWidths=[135, 135, 135, 135])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.4, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. Abstract & Project Overview", h1_style))
    story.append(Paragraph(
        "PaladinMedia investigates secure, reversible transform-domain media steganography engineered as an integrated systems pipeline. "
        "The system combines cryptographic payload serialization, authenticated symmetric encryption (AES-256-GCM), Galois Field error-correction coding (Reed-Solomon RS(255, 223)), "
        "and 2D Discrete Wavelet Transform (1-level Haar DWT) integer-parity modulation into an uncompressed 8-bit RGB carrier container. "
        "The primary engineering objective is to eliminate steganographic payload corruption, establish mathematical invertibility, and guarantee absolute fail-closed security—ensuring "
        "that any transmission tampering, key error, or channel damage halts receiver execution without releasing unauthenticated plaintext data.",
        body_style
    ))

    story.append(Paragraph("2. Problem Statement & Proposed Engineering Solution", h1_style))
    story.append(Paragraph(
        "<b>The Problem:</b> Traditional spatial steganography (e.g., spatial LSB) suffers from high statistical detectability, zero transmission tamper protection, and complete lack of error tolerance. "
        "Conversely, naive transform-domain schemes frequently lack strict capacity budgeting, leading to catastrophic coefficient clipping or unrecoverable bit corruption upon image reload. "
        "Furthermore, unauthenticated steganography exposes receivers to malicious payload tampering without cryptographic nonces or message authentication tags.<br/>"
        "<b>The Proposed Solution:</b> PaladinMedia establishes a strict, unidirectional sender transformation pipeline: "
        "<i>Secret Media → SecretSerializer (b'PLDS' wire framing, CRC-32) → AESGCMProvider (AES-256-GCM, fresh 96-bit nonce, 128-bit GHASH tag) → ReedSolomonEncoder (GF(2^8), 2t=32 parity) → Carrier Capacity Gate → HaarDWT2D (Red-HH integer parity) → Lossless PNG</i>. "
        "The receiver applies exact symmetric reversal with constant-time cryptographic verification, guaranteeing zero plaintext leakage upon any tampering attempt.",
        body_style
    ))

    story.append(Paragraph("3. Sprint-1 Objectives & Scope", h1_style))
    story.append(Paragraph(
        "The scope of Sprint-1 encompasses the delivery and verification of the core cryptographic and mathematical foundation: "
        "<b>(1) Canonical Payload Framing:</b> Serialization of diverse media types (Text, JSON, Binary, Images, Audio) into a deterministic binary layout with header CRC-32 integrity; "
        "<b>(2) Authenticated Cryptography:</b> AES-256-GCM encryption with NIST SP 800-38D KAT compliance, unique 96-bit IV generation, and AAD context binding; "
        "<b>(3) Galois Field ECC:</b> Systematic chunked Reed-Solomon RS(255, 223) encoding capable of repairing up to <i>t=16</i> symbol errors per 255-byte block and 128-bit contiguous bursts; "
        "<b>(4) Carrier Hard-Gate Accounting:</b> Strict pre-transform capacity checking enforcing an 8,176-byte protected ceiling on 512×512 carriers; "
        "<b>(5) Baseline Wavelet Embedding:</b> Floating-point 2D Haar DWT detail subband integer-parity modulation achieving 0.000000 bit error rate (BER) through 8-bit PNG reload.",
        body_style
    ))

    story.append(Paragraph("4. Functional Prototype Status & Implementation State", h1_style))
    
    state_table_data = [
        [Paragraph("Subsystem / Architectural Component", th_style), Paragraph("Target Phase", th_style), Paragraph("Implementation Status", th_style), Paragraph("Verified Repository Evidence", th_style)],
        [
            Paragraph("Payload Serialization & CRC-32", td_bold), Paragraph("Sprint-1 / Phase 1", td_style),
            Paragraph("Completed & Verified", td_green), Paragraph("<code>src/core/serialization/serializer.py</code>; 14 unit tests pass", td_style)
        ],
        [
            Paragraph("AES-256-GCM Cryptographic Vault", td_bold), Paragraph("Sprint-1 / Phase 1", td_style),
            Paragraph("Completed & Verified", td_green), Paragraph("<code>src/security/crypto/provider.py</code>; 22 unit tests pass; NIST KATs", td_style)
        ],
        [
            Paragraph("Reed-Solomon RS(255, 223) ECC", td_bold), Paragraph("Sprint-1 / Phase 1", td_style),
            Paragraph("Completed & Verified", td_green), Paragraph("<code>src/security/ecc/encoder.py</code>; 23 unit tests pass; t=16 symbol repair", td_style)
        ],
        [
            Paragraph("Capacity Hard-Gate & 16B Header", td_bold), Paragraph("Sprint-1 / Phase 2", td_style),
            Paragraph("Completed & Verified", td_green), Paragraph("<code>src/carrier/capacity.py</code>, <code>header.py</code>; 8,176 B ceiling enforced", td_style)
        ],
        [
            Paragraph("Haar DWT & Integer-Parity Engine", td_bold), Paragraph("Sprint-1 / Phase 2", td_style),
            Paragraph("Completed & Verified", td_green), Paragraph("<code>src/carrier/dwt.py</code>, <code>embedding.py</code>; BER=0.0, PSNR>75 dB", td_style)
        ],
        [
            Paragraph("End-to-End Media Pipeline", td_bold), Paragraph("Sprint-1 / Phase 2", td_style),
            Paragraph("Completed & Verified", td_green), Paragraph("<code>src/core/media_pipeline.py</code>; <code>verify_sprint1.py</code> (24/24 pass)", td_style)
        ],
        [
            Paragraph("Observability API & Vite UI", td_bold), Paragraph("Phase 6 (Early)", td_style),
            Paragraph("Completed & Verified", td_green), Paragraph("<code>src/api/server.py</code> (FastAPI SSE); <code>frontend/src/App.tsx</code>", td_style)
        ],
        [
            Paragraph("Multi-Carrier Partitioning (E09)", td_bold), Paragraph("Phase 2 Experiment", td_style),
            Paragraph("Partially Completed", td_amber), Paragraph("<code>scripts/experiments/run_e09.py</code>; 15 KB partitioned across 3 carriers", td_style)
        ],
        [
            Paragraph("ACS-Sub-INN v3 Invertible Flow", td_bold), Paragraph("Phase 3 Neural", td_style),
            Paragraph("Planned (Zero Code)", td_blue), Paragraph("<code>Phase3_Acceptance_Criteria.md</code>; unedited frozen architecture", td_style)
        ],
        [
            Paragraph("CodeCarbon Energy Telemetry", td_bold), Paragraph("Phase 6 Telemetry", td_style),
            Paragraph("Planned (Roadmap)", td_blue), Paragraph("Architecture §14; scheduled for Phase 6 laboratory suite", td_style)
        ],
    ]
    t_state = Table(state_table_data, colWidths=[140, 75, 95, 230])
    t_state.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_header_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.35, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_row_alt]),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 3.5),
        ('RIGHTPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_state)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: SPRINT BACKLOG AND FEATURE STATUS
    # =========================================================================
    story.append(Paragraph("Sprint Backlog & Verifiable Feature Implementation Status", h1_style))
    story.append(Paragraph(
        "The following backlog reflects genuine Sprint-1 engineering tasks derived from repository source code, formal test fixtures, and verification scripts. "
        "In accordance with project records, all implemented Sprint-1 tasks were independently authored and verified by the sole individual developer (Rohit Gautam). "
        "Controlled status labels strictly reflect independently verified source evidence.",
        body_style
    ))

    backlog_table_data = [
        [Paragraph("ID", th_style), Paragraph("User Story", th_style), Paragraph("Tasks", th_style), Paragraph("Owner", th_style), Paragraph("Status", th_style), Paragraph("Evidence", th_style)],
        [
            Paragraph("PDM-01", td_code), Paragraph("Repository Architecture & Scaffolding", td_bold),
            Paragraph("Configure pyproject.toml, package hierarchy, and core exception taxonomy (PaladinSecurityError).", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/core/exceptions.py</code><br/><code>tests/unit/test_exceptions.py</code>", td_code)
        ],
        [
            Paragraph("PDM-02", td_code), Paragraph("Payload Framing & Serialization", td_bold),
            Paragraph("Implement SecretSerializer with canonical <code>b'PLDS'</code> header, sorted JSON metadata, and CRC-32 trailer.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/core/serialization/serializer.py</code><br/>14 unit tests pass", td_code)
        ],
        [
            Paragraph("PDM-03", td_code), Paragraph("Authenticated Cryptography Vault", td_bold),
            Paragraph("Develop AESGCMProvider enforcing AES-256-GCM, fresh 96-bit IVs, 128-bit GHASH tag, AAD context binding, and fail-closed.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/security/crypto/provider.py</code><br/>22 unit tests, NIST KAT pass", td_code)
        ],
        [
            Paragraph("PDM-04", td_code), Paragraph("Galois Field Error Correction Coding", td_bold),
            Paragraph("Implement ReedSolomonEncoder over GF(2^8) with systematic chunking, 2t=32 parity bytes, and 4B prefix framing.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/security/ecc/encoder.py</code><br/>23 unit tests, t=16 symbol repair", td_code)
        ],
        [
            Paragraph("PDM-05", td_code), Paragraph("Security Pipeline Orchestration", td_bold),
            Paragraph("Build SecurityPipeline enforcing Serialize → AES-GCM → Reed-Solomon unidirectional sequence and exact reversal.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/core/pipeline.py</code><br/><code>scripts/verify_sprint1.py</code> (24/24)", td_code)
        ],
        [
            Paragraph("PDM-06", td_code), Paragraph("Carrier Capacity Hard Gate", td_bold),
            Paragraph("Implement CapacityCalculator enforcing 8,176 B ceiling on 512×512 carriers, rejecting overflows before transform.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/carrier/capacity.py</code><br/><code>test_carrier_capacity.py</code>", td_code)
        ],
        [
            Paragraph("PDM-07", td_code), Paragraph("Wavelet Transform Signal Processing", td_bold),
            Paragraph("Implement 1-level 2D Haar DWT / IDWT in float64 precision decomposing 512×512 RGB images with zero numerical error.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/carrier/dwt.py</code><br/><code>test_carrier_dwt.py</code> (max err=0.0)", td_code)
        ],
        [
            Paragraph("PDM-08", td_code), Paragraph("Steganographic Framing & Deterministic Mask", td_bold),
            Paragraph("Build 16-byte fixed CarrierHeader (<code>b'PLDH'</code>) and non-circular DeterministicMaskGenerator via SHA-256 PRNG.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/carrier/header.py</code>, <code>mask.py</code><br/>Tests T-013, T-014 pass", td_code)
        ],
        [
            Paragraph("PDM-09", td_code), Paragraph("Transform-Domain Parity Modulation", td_bold),
            Paragraph("Implement BaselineEmbedder/Extractor modulating integer parity <code>(a-b-c+d)%2</code> on Red HH detail subband.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/carrier/embedding.py</code>, <code>extraction.py</code><br/>BER=0.000000 on PNG reload", td_code)
        ],
        [
            Paragraph("PDM-10", td_code), Paragraph("End-to-End Media Steganography Pipeline", td_bold),
            Paragraph("Build MediaStegoPipeline orchestrating full media-to-media roundtrip with PSNR > 75 dB and exact byte equality.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/core/media_pipeline.py</code><br/>Tests T-019, T-020 pass", td_code)
        ],
        [
            Paragraph("PDM-11", td_code), Paragraph("Observability API & Laboratory Frontend", td_bold),
            Paragraph("Develop FastAPI SSE server and Vite/React UI featuring 4-quadrant DWT visualizer, capacity meter, and live logs.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Completed", td_green), Paragraph("<code>src/api/server.py</code><br/><code>frontend/src/App.tsx</code>", td_code)
        ],
        [
            Paragraph("PDM-12", td_code), Paragraph("Multi-Carrier Partitioning Protocol", td_bold),
            Paragraph("Partition large payloads (>8 KB) across multiple distinct carrier covers with session framing and sequence headers.", td_style),
            Paragraph("Rohit Gautam", td_style), Paragraph("Partially Completed", td_amber), Paragraph("<code>scripts/experiments/run_e09.py</code><br/>15 KB split over 3 carriers verified", td_code)
        ],
        [
            Paragraph("PDM-13", td_code), Paragraph("ACS-Sub-INN v3 Invertible Flow Engine", td_bold),
            Paragraph("Continuous affine coupling block wavelet feature modulation to achieve lossy container (JPEG) robustness.", td_style),
            Paragraph("Unassigned", td_style), Paragraph("Planned", td_blue), Paragraph("<code>docs/specifications/</code><br/>Zero source code; Phase 3 research", td_code)
        ],
        [
            Paragraph("PDM-14", td_code), Paragraph("CodeCarbon Environmental Telemetry", td_bold),
            Paragraph("Instrument embedding/extraction pipelines with CodeCarbon to measure Joules, power draw, and CO2 emissions.", td_style),
            Paragraph("Unassigned", td_style), Paragraph("Planned", td_blue), Paragraph("Phase 6 specification roadmap;<br/>No runtime telemetry active", td_code)
        ],
    ]
    t_backlog = Table(backlog_table_data, colWidths=[36, 96, 175, 55, 62, 116])
    t_backlog.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_header_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.3, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_row_alt]),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_backlog)
    story.append(Spacer(1, 4))

    story.append(Paragraph("Completed vs. Planned Features Subsection", h2_style))
    story.append(Paragraph(
        "<b>Completed Features (Phases 0–2):</b> The repository contains a fully working, self-contained classical steganography system passing 103/103 tests in 2.36 s. "
        "The complete pipeline—from binary serialization to AES-256-GCM encryption, Reed-Solomon error correction, Haar DWT decomposition, integer-parity embedding, lossless PNG container generation, "
        "and receiver extraction—is 100% operational. It delivers high fidelity (PSNR = 75.92 dB, SSIM = 1.000000) and verbatim byte equality.<br/>"
        "<b>Planned Features (Phases 3–6):</b> The planned ACS-Sub-INN neural architecture, straight-through estimator (STE) quantization noise layers, ONNX Runtime INT8 deployment, "
        "and CodeCarbon energy telemetry are <i>unimplemented forward research items</i>. While detailed specifications exist in <code>docs/specifications/</code>, "
        "no neural training scripts, model weights, or telemetry hooks exist in <code>src/</code>. Furthermore, Experiment E14 conclusively proved that the classical baseline suffers 100% failure "
        "under lossy JPEG recompression, scientifically justifying the future transition to Phase 3 neural modeling.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: TECHNICAL DESIGN & ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("Technical Design, System Architecture & Module Interactions", h1_style))
    
    story.append(Paragraph("1. System Architecture & Transformation Pipeline", h2_style))
    arch_box_text = (
        "<b>SENDER PIPELINE:</b> Secret Media [Image/Text/JSON] → <b>SecretSerializer</b> (b'PLDS', CRC-32) → "
        "<b>AESGCMProvider</b> (AES-256-GCM, 96b IV, 128b Tag) → <b>ReedSolomonEncoder</b> (GF(2^8), 2t=32 parity) → "
        "<b>CapacityCalculator</b> (8,176 B Hard Gate) → <b>HaarDWT2D</b> (1-level Haar DWT → LL, LH, HL, HH) → "
        "<b>CarrierHeader</b> (16B b'PLDH') → <b>BaselineEmbedder</b> (2*HH parity on Red ch) → <b>IDWT</b> → Lossless 8-bit RGB PNG<br/>"
        "<b>RECEIVER RECOVERY PIPELINE:</b> Stego PNG → <b>HaarDWT2D</b> → <b>Extractor</b> (Extract 16B Header & Parity Bits) → "
        "<b>ReedSolomonEncoder.decode</b> (Repairs ≤16 symbol errors/block) → <b>AESGCMProvider.decrypt</b> (Constant-time GHASH check) → "
        "<b>SecretSerializer.deserialize</b> (CRC-32 verify) → <b>100% Exact Recovered Media</b>"
    )
    t_arch_box = Table([[Paragraph(arch_box_text, callout_box_style)]], colWidths=[540])
    t_arch_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 0.6, colors.HexColor("#3B82F6")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_arch_box)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Data-Flow & Byte Budget Transformation (512×512 Carrier)", h2_style))
    
    dfd_data = [
        [Paragraph("Pipeline Stage", th_style), Paragraph("Input Data", th_style), Paragraph("Transformation & Framing Overhead", th_style), Paragraph("Output Stream", th_style), Paragraph("Size (Small Fixture)", th_style)],
        [
            Paragraph("1. Serialization", td_bold), Paragraph("Secret Image / Text", td_style),
            Paragraph("Prepends 10B header (Magic <code>b'PLDS'</code>, version, media type), length (4B), JSON metadata, CRC-32 (4B)", td_style),
            Paragraph("Framed Bytes", td_code), Paragraph("152 B → 170 B (+18 B)", td_style)
        ],
        [
            Paragraph("2. Encryption", td_bold), Paragraph("Framed Bytes", td_style),
            Paragraph("AES-256-GCM encryption with fresh 12B IV and 16B GHASH authentication tag; binds optional AAD", td_style),
            Paragraph("Protected Ciphertext", td_code), Paragraph("170 B → 198 B (+28 B)", td_style)
        ],
        [
            Paragraph("3. ECC Encoding", td_bold), Paragraph("Protected Ciphertext", td_style),
            Paragraph("Systematic RS(255, 223) chunking; generates 32 parity bytes per block; prepends 4B big-endian length prefix", td_style),
            Paragraph("Protected RS Bitstream", td_code), Paragraph("198 B → 234 B (+36 B)", td_style)
        ],
        [
            Paragraph("4. Capacity Gate", td_bold), Paragraph("Protected RS Bitstream", td_style),
            Paragraph("Analytic capacity check: <code>len(stream) ≤ 8,176 B</code>. Aborts fail-closed before wavelet execution if exceeded", td_style),
            Paragraph("Validated Bitstream", td_code), Paragraph("234 B ≤ 8,176 B (PASS)", td_green)
        ],
        [
            Paragraph("5. Stego Embedding", td_bold), Paragraph("Cover Image (512×512)", td_style),
            Paragraph("1-level Haar DWT; embeds 16B CarrierHeader (128 bits) + 234B payload (1,872 bits) via Red-HH parity modulation", td_style),
            Paragraph("Modified Wavelet Coeffs", td_code), Paragraph("2,000 modified quads", td_style)
        ],
        [
            Paragraph("6. IDWT & PNG", td_bold), Paragraph("Modified Wavelet Coeffs", td_style),
            Paragraph("Inverse 2D Haar DWT reconstructing spatial domain; writes lossless 8-bit RGB PNG container to disk/buffer", td_style),
            Paragraph("Stego PNG Container", td_code), Paragraph("512×512 PNG (~380 KB)", td_style)
        ],
    ]
    t_dfd = Table(dfd_data, colWidths=[75, 80, 205, 95, 85])
    t_dfd.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_header_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.3, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_row_alt]),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_dfd)
    story.append(Spacer(1, 4))

    story.append(Paragraph("3. Major Architectural Components & Module Interactions", h2_style))
    story.append(Paragraph(
        "• <b>SecretSerializer (<code>src/core/serialization/serializer.py</code>):</b> Enforces canonical binary wire framing (<code>0x504C4453</code>), media enumeration, compact sorted JSON metadata serialization, and trailer CRC-32 verification.<br/>"
        "• <b>AESGCMProvider (<code>src/security/crypto/provider.py</code>):</b> Implements AES-256-GCM authenticated cipher with 12-byte cryptographically secure random nonces and 16-byte GHASH tags. Enforces fail-closed operation, raising <code>AuthenticationError</code> on any bit corruption.<br/>"
        "• <b>ReedSolomonEncoder (<code>src/security/ecc/encoder.py</code>):</b> Systematic Galois Field $GF(2^8)$ codec utilizing generator polynomial $(x+\alpha^0)\dots(x+\alpha^{31})$. Automatically splits streams into 223-byte chunks, appending 32 parity bytes to guarantee recovery of up to $t=16$ corrupted symbols per block.<br/>"
        "• <b>CapacityCalculator (<code>src/carrier/capacity.py</code>):</b> Evaluates carrier geometry in $O(1)$ time. Computes raw bit budget ($N^2/32$ bytes), deducts 16B header, and computes max RS-protected payload (8,176 B for 512×512; 32,752 B for 1024×1024). Raises <code>PayloadCapacityExceededError</code> on boundary overflow.<br/>"
        "• <b>HaarDWT2D & BaselineEmbedder (<code>src/carrier/</code>):</b> Decomposes 2D spatial RGB images into orthogonal subbands (LL, LH, HL, HH). Modulates integer parity $2\cdot\text{HH} = (a - b - c + d) \pmod 2$ in the Red channel by adjusting pixel $d$ by $\pm 1$. Extractor reconstructs bits directly from quad parities.",
        body_style
    ))

    story.append(Paragraph("4. Technology Stack & Architectural Rationale", h2_style))
    story.append(Paragraph(
        "• <b>Python 3.10.11:</b> Primary implementation language offering native type annotations, dataclasses, and rich scientific ecosystem.<br/>"
        "• <b>cryptography 49.0.0:</b> Standard OpenSSL-backed cryptographic provider providing constant-time AES-NI hardware-accelerated encryption.<br/>"
        "• <b>reedsolo 1.7.0:</b> Pure-Python Galois Field $GF(2^8)$ codec ensuring deterministic mathematical error correction without native C compilation.<br/>"
        "• <b>numpy 2.2.6 & Pillow 12.2.0:</b> Optimized vector/matrix arithmetic and lossless PNG codec ensuring zero quantization loss during carrier reload.<br/>"
        "• <b>FastAPI 0.141.1 & Vite/React 19:</b> Asynchronous REST/SSE streaming bridge and responsive laboratory UI for real-time subband visualization.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: GREEN COMPUTING AND TESTING
    # =========================================================================
    story.append(Paragraph("Green Computing Practices & Formal Verification Test Log", h1_style))
    
    story.append(Paragraph("1. Green Computing Implementation & Empirical Evaluation", h2_style))
    story.append(Paragraph(
        "PaladinMedia integrates green computing principles directly into its architectural design. In accordance with strict academic audit constraints, "
        "only verifiable practices implemented and measured in the codebase are marked as active; forward telemetry is designated as Planned.",
        body_style
    ))

    green_table_data = [
        [Paragraph("Practice", th_style), Paragraph("Implementation in Repository", th_style), Paragraph("Verified Evidence", th_style), Paragraph("Concrete Benefit", th_style), Paragraph("Limitation / Trade-off", th_style)],
        [
            Paragraph("In-Place Vectorized Processing", td_bold),
            Paragraph("NumPy array operations and direct memory views used for Haar DWT subbands and parity slicing.", td_style),
            Paragraph("<code>src/carrier/dwt.py</code><br/><code>src/carrier/embedding.py</code>", td_code),
            Paragraph("Minimizes heap allocations; RSS memory delta only <b>4.7–5.8 MB</b> per full roundtrip.", td_style),
            Paragraph("Intermediate float64 representations required for lossless DWT inversion.", td_style)
        ],
        [
            Paragraph("Minimalist Dependency Architecture", td_bold),
            Paragraph("Standard library + 4 lean packages (<code>cryptography</code>, <code>reedsolo</code>, <code>numpy</code>, <code>pillow</code>).", td_style),
            Paragraph("<code>pyproject.toml</code><br/>Zero heavyweight ML frameworks", td_code),
            Paragraph("Ultra-low disk/memory footprint; zero continuous GPU idle power drain.", td_style),
            Paragraph("Precludes deep learning feature extraction in the classical baseline.", td_style)
        ],
        [
            Paragraph("Fail-Closed $O(1)$ Capacity Gating", td_bold),
            Paragraph("Analytic byte calculation aborts oversized payloads before performing DWT or matrix manipulations.", td_style),
            Paragraph("<code>src/carrier/capacity.py</code><br/><code>test_carrier_capacity.py</code>", td_code),
            Paragraph("Eliminates 100% of wasted CPU cycles and memory allocations on invalid workloads.", td_style),
            Paragraph("Requires pre-computation of carrier dimensions before execution.", td_style)
        ],
        [
            Paragraph("Targeted, Deterministic Testing", td_bold),
            Paragraph("Exhaustive 103-test pytest suite executed in pure process isolation without persistent server daemons.", td_style),
            Paragraph("103 tests pass in <b>2.36 s</b><br/><code>verify_sprint1.py</code> (0.27 s)", td_code),
            Paragraph("Eliminates redundant training loops and long CI runs, conserving compute energy.", td_style),
            Paragraph("Does not simulate physical camera analog sensor noise.", td_style)
        ],
        [
            Paragraph("Runtime Execution Profiling", td_bold),
            Paragraph("Built-in high-resolution timers (<code>time.perf_counter</code>) and process memory tracking (<code>psutil</code>).", td_style),
            Paragraph("Observed latency: <b>92–106 ms</b><br/>RSS delta: <b>4.67 MB</b>", td_code),
            Paragraph("Provides verifiable empirical execution metrics for carbon modeling.", td_style),
            Paragraph("Host OS scheduling introduces ±5 ms variance across runs.", td_style)
        ],
        [
            Paragraph("CodeCarbon Telemetry", td_bold),
            Paragraph("Automated background emission and Joule tracking per steganographic execution.", td_style),
            Paragraph("Architecture §14 (Roadmap)<br/><b>Planned (Not Active)</b>", td_blue),
            Paragraph("Will provide standardized kWh and CO2e tracking across runs.", td_style),
            Paragraph("Currently unimplemented; scheduled for Phase 6 laboratory suite.", td_style)
        ],
    ]
    t_green = Table(green_table_data, colWidths=[90, 125, 95, 115, 115])
    t_green.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_header_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.3, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_row_alt]),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_green)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Sprint-1 Formal Test Log & Verification Results", h2_style))
    story.append(Paragraph(
        "Every recorded test case corresponds to a genuine automated test in <code>tests/</code> executed and verified under Python 3.10.11. "
        "The test suite achieved a 100% pass rate (103/103 passed in 2.36 s), verifying mathematical correctness and fail-closed security.",
        body_style
    ))

    test_table_data = [
        [Paragraph("Test Case Identifier & Name", th_style), Paragraph("Expected Result", th_style), Paragraph("Actual Observed Result", th_style), Paragraph("Status", th_style), Paragraph("Verified Test File & Evidence", th_style)],
        [
            Paragraph("<code>test_t019_full_media_to_media_mvp</code>", td_code),
            Paragraph("PSNR > 50 dB, SSIM > 0.999, exact recovered image (24×24)", td_style),
            Paragraph("PSNR = 75.89 dB, SSIM = 1.000000, verbatim byte equality", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/integration/test_media_pipeline.py</code>", td_code)
        ],
        [
            Paragraph("<code>test_t020_exact_byte_equality</code>", td_code),
            Paragraph("Verbatim match for payload data, media type, and metadata", td_style),
            Paragraph("100% exact equality; <code>exact_match == True</code>", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/integration/test_media_pipeline.py</code>", td_code)
        ],
        [
            Paragraph("<code>test_capacity_exceeded_hard_gate</code>", td_code),
            Paragraph("Oversized payload (10 KB) rejected before transform execution", td_style),
            Paragraph("<code>PayloadCapacityExceededError</code> raised fail-closed", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/integration/test_media_pipeline.py</code>", td_code)
        ],
        [
            Paragraph("<code>test_aes_256_gcm_encrypt_decrypt</code>", td_code),
            Paragraph("Clean roundtrip with unique 96b nonce and 128b GHASH tag", td_style),
            Paragraph("Exact plaintext recovered; nonces unique across runs", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/unit/test_crypto.py</code>", td_code)
        ],
        [
            Paragraph("<code>test_nist_sp800_38d_kat_vectors</code>", td_code),
            Paragraph("Cryptographic vectors match official NIST KAT values exactly", td_style),
            Paragraph("All NIST SP 800-38D test vectors validated", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/unit/test_crypto.py</code>", td_code)
        ],
        [
            Paragraph("<code>test_fail_closed_ciphertext_bit_flip</code>", td_code),
            Paragraph("Bit-flipped ciphertext halts with <code>AuthenticationError</code>, 0 leak", td_style),
            Paragraph("<code>AuthenticationError</code> raised; zero plaintext bytes leaked", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/unit/test_crypto.py</code>", td_code)
        ],
        [
            Paragraph("<code>test_symbol_error_correction_bound_t</code>", td_code),
            Paragraph("Repair exactly $t=16$ corrupted symbols per 255-byte block", td_style),
            Paragraph("All 16 corrupted bytes repaired; bitstream restored", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/unit/test_ecc.py</code>", td_code)
        ],
        [
            Paragraph("<code>test_negative_uncorrectable_errors</code>", td_code),
            Paragraph("Reject bitstream with $t+1=17$ corrupted symbols fail-closed", td_style),
            Paragraph("<code>ECCDecodeError</code> raised; execution halted", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/unit/test_ecc.py</code>", td_code)
        ],
        [
            Paragraph("<code>test_t018_png_save_reload_carrier</code>", td_code),
            Paragraph("Zero bit error rate through lossless PNG disk encode/reload", td_style),
            Paragraph("Bit Error Rate (BER) identically 0.000000", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/unit/test_carrier_embedding.py</code>", td_code)
        ],
        [
            Paragraph("<code>test_canonical_wire_framing_layout</code>", td_code),
            Paragraph("Magic bytes <code>b'PLDS'</code>, version 0x01, and CRC-32 validated", td_style),
            Paragraph("Deterministic binary layout verified down to exact offsets", td_style),
            Paragraph("PASS", td_green), Paragraph("<code>tests/unit/test_serialization.py</code>", td_code)
        ],
    ]
    t_test = Table(test_table_data, colWidths=[125, 125, 125, 35, 130])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_header_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.3, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_row_alt]),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_test)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: RETROSPECTIVE, CONTRIBUTIONS & NEXT-SPRINT BACKLOG
    # =========================================================================
    story.append(Paragraph("Sprint Retrospective, Author Contributions & Forward Roadmap", h1_style))
    
    story.append(Paragraph("1. Sprint-1 Retrospective & Engineering Assessment", h2_style))
    story.append(Paragraph(
        "<b>What Worked Well:</b> The implementation of the strict unidirectional pipeline (Serialize → Encrypt → ECC → Embed) was completely successful. "
        "The cryptographic vault achieved absolute fail-closed security with verified zero plaintext leakage under all adversarial tampering scenarios. "
        "Embedding integer parities into the Red-HH Haar subband achieved flawless mathematical invertibility with $\\text{BER} = 0.000000$ and $\\text{PSNR} > 75\\text{ dB}$ across lossless PNG containers.<br/>"
        "<b>What Did Not Work / Incomplete:</b> The classical integer-parity baseline suffers from severe vulnerability to lossy distribution channels. "
        "As established in Experiment E14 (220 empirical runs), JPEG recompression at <i>any</i> quality factor ($Q \\in [70, 100]$) and additive Gaussian noise ($\\sigma \\ge 0.5$) "
        "destroys spatial integer parity $(a - b - c + d) \\pmod 2$, producing $\\approx 50\\%$ raw parity bit error rates and $100\\%$ recovery failure. "
        "This empirical failure establishes the fundamental necessity for Phase 3 neural steganography.<br/>"
        "<b>Technical Challenges & Lessons Learned:</b> (1) Initial assumptions of 8,192 B capacity collided with real wire framing overheads; resolved by developing explicit mathematical capacity accounting down to the exact byte. "
        "(2) Disk I/O (PNG encode/decode) accounts for $>75\\%$ of total pipeline latency (~102 ms out of 134 ms total). "
        "(3) Separating the verified classical baseline from future neural designs prevented premature scope creep.",
        body_style
    ))

    story.append(Paragraph("2. Prioritized Next-Sprint Backlog (Sprint-2 / Phase 3)", h2_style))
    
    next_backlog_data = [
        [Paragraph("ID", th_style), Paragraph("Task Description", th_style), Paragraph("Target Phase", th_style), Paragraph("Priority", th_style), Paragraph("Technical Rationale", th_style)],
        [
            Paragraph("PDM-N01", td_code),
            Paragraph("Scaffold ACS-Sub-INN v3 Invertible Neural Network Architecture", td_bold),
            Paragraph("Phase 3 (Neural)", td_style), Paragraph("P0 (Highest)", td_green),
            Paragraph("Transition from discrete integer parity to continuous affine coupling blocks to resist channel noise.", td_style)
        ],
        [
            Paragraph("PDM-N02", td_code),
            Paragraph("Implement Differentiable JPEG Noise Simulation Layer", td_bold),
            Paragraph("Phase 3 (Robustness)", td_style), Paragraph("P0 (Highest)", td_green),
            Paragraph("Empirically proven necessity from E14: model DCT quantization noise during training to survive JPEG Q=80.", td_style)
        ],
        [
            Paragraph("PDM-N03", td_code),
            Paragraph("Multi-Carrier Transport Partitioning CLI & Auto-Reassembly", td_bold),
            Paragraph("Phase 2 Extension", td_style), Paragraph("P1 (High)", td_amber),
            Paragraph("Generalize E09 experiment script into production CLI for seamless multi-image secret reconstruction.", td_style)
        ],
        [
            Paragraph("PDM-N04", td_code),
            Paragraph("Steganalysis Resistance Evaluation against Spatial Rich Models (SRM)", td_bold),
            Paragraph("Phase 4 (Security)", td_style), Paragraph("P1 (High)", td_amber),
            Paragraph("Close Gap B by training an ensemble FLD classifier on 34,671 SRM submodel features to quantify detection error.", td_style)
        ],
        [
            Paragraph("PDM-N05", td_code),
            Paragraph("Integrate CodeCarbon Energy & Carbon Telemetry Telemetry", td_bold),
            Paragraph("Phase 6 (Telemetry)", td_style), Paragraph("P2 (Medium)", td_blue),
            Paragraph("Automate Joules and CO2e tracking per embedding session within the FastAPI observability stream.", td_style)
        ],
    ]
    t_next = Table(next_backlog_data, colWidths=[40, 160, 75, 55, 210])
    t_next.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_header_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.3, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_row_alt]),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_next)
    story.append(Spacer(1, 4))

    story.append(Paragraph("3. Individual Author Contributions & Verifiable Commit History", h2_style))
    story.append(Paragraph(
        "PaladinMedia is developed as an individual engineering project. In accordance with academic integrity guidelines, "
        "all contributions are attributed to the sole project developer without fabricating team members or commit statistics. "
        "Git repository metadata records two canonical baseline commits:",
        body_style
    ))

    contrib_table_data = [
        [Paragraph("Member", th_style), Paragraph("Tasks Done", th_style), Paragraph("Evidence", th_style), Paragraph("Commits", th_style)],
        [
            Paragraph("<b>Rohit Gautam</b><br/>(Reg No: 23215050)<br/><i>GitHub: RohitGautam0510</i><br/>rohit5.10gautam@gmail.com", td_style),
            Paragraph(
                "• Designed and implemented complete cryptographic vault (AES-256-GCM, fail-closed, 0 leak).<br/>"
                "• Implemented Reed-Solomon GF(2^8) error correction codec with systematic chunking.<br/>"
                "• Built 1-level 2D Haar DWT engine and integer-parity transform embedder/extractor.<br/>"
                "• Engineered carrier capacity accounting and pre-transform hard-gate enforcement.<br/>"
                "• Authored 103-test automated pytest suite and 24-check programmatic verification script.<br/>"
                "• Executed empirical experiments E03 through E14 (lossy channel degradation baseline).<br/>"
                "• Developed FastAPI SSE backend server and React/Vite laboratory user interface.",
                td_style
            ),
            Paragraph(
                "• <code>src/core/</code>, <code>src/security/</code>, <code>src/carrier/</code>, <code>src/api/</code><br/>"
                "• 103 passing pytest tests in <code>tests/</code><br/>"
                "• <code>scripts/verify_sprint1.py</code> (24/24 pass)<br/>"
                "• <code>docs/experiments/E03</code>–<code>E14</code> ledgers<br/>"
                "• <code>frontend/src/App.tsx</code>",
                td_code
            ),
            Paragraph(
                "<b>commit 9555996</b><br/><i>feat(audit): resolve P1 gate blockers with E14 lossy baseline, div2k split manifest, and acceptance criteria reconciliation</i><br/><br/>"
                "<b>commit a8c3f76</b><br/><i>baseline: phase 2 verified classical wavelet baseline and entry-gate audit</i>",
                td_code
            )
        ]
    ]
    t_contrib = Table(contrib_table_data, colWidths=[90, 195, 125, 130])
    t_contrib.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_header_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.3, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 3.5),
        ('RIGHTPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_contrib)

    # Build the PDF using NumberedCanvas
    doc.build(story, canvasmaker=Sprint1ReportCanvas)
    print(f"[OK] Report PDF successfully built at: {filename}")


# =============================================================================
# PYTHON-DOCX GENERATION (MATCHING CONTENT & STRUCTURE)
# =============================================================================

def set_cell_background(cell, hex_color):
    """Sets background color of a Word table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=60, bottom=60, left=100, right=100):
    """Sets internal padding for a Word table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def build_docx(filename=DOCX_PATH):
    doc = docx.Document()
    
    # Configure Margins (0.5 inch / 36 pt)
    sections = doc.sections
    for sec in sections:
        sec.top_margin = Inches(0.5)
        sec.bottom_margin = Inches(0.5)
        sec.left_margin = Inches(0.5)
        sec.right_margin = Inches(0.5)

    # Base Styles
    c_navy = RGBColor(30, 58, 138)
    c_dark = RGBColor(30, 41, 59)
    c_muted = RGBColor(71, 85, 105)

    # Title
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("PALADINMEDIA: POST-QUANTUM MEDIA STEGANOGRAPHY & TRANSFORM CONCEALMENT")
    r_title.bold = True
    r_title.font.size = Pt(14)
    r_title.font.color.rgb = c_navy
    p_title.paragraph_format.space_after = Pt(2)

    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("CIA-3 Sprint-1 Verification Report  •  Functional Prototype Audit & Baseline Handoff")
    r_sub.font.size = Pt(9.5)
    r_sub.font.color.rgb = c_muted
    p_sub.paragraph_format.space_after = Pt(6)

    # Metadata Table
    meta_data = [
        [
            "Contributor: Rohit Gautam (Reg No: 23215050)",
            "Course / Dept: BCA411-7 / School of Sciences",
            "Supervisor: Dr Preety",
            "Deadline: 11 Oct 2026, 11:59 PM"
        ],
        [
            "Current Baseline: MVP 0 / Phase 2 Classical",
            "Pytest Suite: 103 / 103 Passing (100%)",
            "Sprint-1 Checks: 24 / 24 Passing (100%)",
            "Git Commits: 9555996, a8c3f76 (Clean)"
        ]
    ]
    t_meta = doc.add_table(rows=2, cols=4)
    t_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(meta_data):
        for c_idx, val in enumerate(row):
            cell = t_meta.cell(r_idx, c_idx)
            cell.text = val
            set_cell_background(cell, "F1F5F9")
            set_cell_margins(cell, 40, 40, 60, 60)
            p = cell.paragraphs[0]
            p.runs[0].font.size = Pt(7.5)
            p.runs[0].font.color.rgb = c_dark
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(11)
        r.font.color.rgb = c_navy

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = c_dark

    def add_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(text)
        r.font.size = Pt(8.5)
        r.font.color.rgb = c_dark

    # Page 1 Content
    add_h1("1. Abstract & Project Overview")
    add_p(
        "PaladinMedia investigates secure, reversible transform-domain media steganography engineered as an integrated systems pipeline. "
        "The system combines cryptographic payload serialization, authenticated symmetric encryption (AES-256-GCM), Galois Field error-correction coding (Reed-Solomon RS(255, 223)), "
        "and 2D Discrete Wavelet Transform (1-level Haar DWT) integer-parity modulation into an uncompressed 8-bit RGB carrier container. "
        "The primary engineering objective is to eliminate steganographic payload corruption, establish mathematical invertibility, and guarantee absolute fail-closed security—ensuring "
        "that any transmission tampering, key error, or channel damage halts receiver execution without releasing unauthenticated plaintext data."
    )

    add_h1("2. Problem Statement & Proposed Engineering Solution")
    add_p(
        "The Problem: Traditional spatial steganography (e.g., spatial LSB) suffers from high statistical detectability, zero transmission tamper protection, and complete lack of error tolerance. "
        "Conversely, naive transform-domain schemes frequently lack strict capacity budgeting, leading to catastrophic coefficient clipping or unrecoverable bit corruption upon image reload. "
        "Furthermore, unauthenticated steganography exposes receivers to malicious payload tampering without cryptographic nonces or message authentication tags.\n"
        "The Proposed Solution: PaladinMedia establishes a strict, unidirectional sender transformation pipeline: "
        "Secret Media -> SecretSerializer (b'PLDS' wire framing, CRC-32) -> AESGCMProvider (AES-256-GCM, fresh 96-bit nonce, 128-bit GHASH tag) -> ReedSolomonEncoder (GF(2^8), 2t=32 parity) -> Carrier Capacity Gate -> HaarDWT2D (Red-HH integer parity) -> Lossless PNG. "
        "The receiver applies exact symmetric reversal with constant-time cryptographic verification, guaranteeing zero plaintext leakage upon any tampering attempt."
    )

    add_h1("3. Sprint-1 Objectives & Scope")
    add_p(
        "The scope of Sprint-1 encompasses the delivery and verification of the core cryptographic and mathematical foundation:\n"
        "1. Canonical Payload Framing: Serialization of diverse media types (Text, JSON, Binary, Images, Audio) into a deterministic binary layout with header CRC-32 integrity.\n"
        "2. Authenticated Cryptography: AES-256-GCM encryption with NIST SP 800-38D KAT compliance, unique 96-bit IV generation, and AAD context binding.\n"
        "3. Galois Field ECC: Systematic chunked Reed-Solomon RS(255, 223) encoding capable of repairing up to t=16 symbol errors per 255-byte block and 128-bit contiguous bursts.\n"
        "4. Carrier Hard-Gate Accounting: Strict pre-transform capacity checking enforcing an 8,176-byte protected ceiling on 512x512 carriers.\n"
        "5. Baseline Wavelet Embedding: Floating-point 2D Haar DWT detail subband integer-parity modulation achieving 0.000000 bit error rate (BER) through 8-bit PNG reload."
    )

    add_h1("4. Functional Prototype Status & Implementation State")
    state_rows = [
        ["Subsystem / Architectural Component", "Target Phase", "Implementation Status", "Verified Repository Evidence"],
        ["Payload Serialization & CRC-32", "Sprint-1 / Phase 1", "Completed & Verified", "src/core/serialization/serializer.py; 14 unit tests pass"],
        ["AES-256-GCM Cryptographic Vault", "Sprint-1 / Phase 1", "Completed & Verified", "src/security/crypto/provider.py; 22 unit tests pass; NIST KATs"],
        ["Reed-Solomon RS(255, 223) ECC", "Sprint-1 / Phase 1", "Completed & Verified", "src/security/ecc/encoder.py; 23 unit tests pass; t=16 symbol repair"],
        ["Capacity Hard-Gate & 16B Header", "Sprint-1 / Phase 2", "Completed & Verified", "src/carrier/capacity.py, header.py; 8,176 B ceiling enforced"],
        ["Haar DWT & Integer-Parity Engine", "Sprint-1 / Phase 2", "Completed & Verified", "src/carrier/dwt.py, embedding.py; BER=0.0, PSNR>75 dB"],
        ["End-to-End Media Pipeline", "Sprint-1 / Phase 2", "Completed & Verified", "src/core/media_pipeline.py; verify_sprint1.py (24/24 pass)"],
        ["Observability API & Vite UI", "Phase 6 (Early)", "Completed & Verified", "src/api/server.py (FastAPI SSE); frontend/src/App.tsx"],
        ["Multi-Carrier Partitioning (E09)", "Phase 2 Experiment", "Partially Completed", "scripts/experiments/run_e09.py; 15 KB partitioned across 3 carriers"],
        ["ACS-Sub-INN v3 Invertible Flow", "Phase 3 Neural", "Planned (Zero Code)", "Phase3_Acceptance_Criteria.md; unedited frozen architecture"],
        ["CodeCarbon Energy Telemetry", "Phase 6 Telemetry", "Planned (Roadmap)", "Architecture §14; scheduled for Phase 6 laboratory suite"],
    ]
    t_s = doc.add_table(rows=len(state_rows), cols=4)
    t_s.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_i, row in enumerate(state_rows):
        for c_i, val in enumerate(row):
            c = t_s.cell(r_i, c_i)
            c.text = val
            set_cell_margins(c, 40, 40, 50, 50)
            p = c.paragraphs[0]
            if r_i == 0:
                set_cell_background(c, "1E3A8A")
                p.runs[0].bold = True
                p.runs[0].font.size = Pt(7.5)
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_background(c, "F8FAFC" if r_i % 2 == 1 else "FFFFFF")
                p.runs[0].font.size = Pt(7.0)
                p.runs[0].font.color.rgb = c_dark

    doc.add_page_break()

    # Page 2 Content
    add_h1("Sprint Backlog & Verifiable Feature Implementation Status")
    add_p(
        "The following backlog reflects genuine Sprint-1 engineering tasks derived from repository source code, formal test fixtures, and verification scripts. "
        "In accordance with project records, all implemented Sprint-1 tasks were independently authored and verified by the sole individual developer (Rohit Gautam). "
        "Controlled status labels strictly reflect independently verified source evidence."
    )

    backlog_rows = [
        ["ID", "User Story", "Tasks", "Owner", "Status", "Evidence"],
        ["PDM-01", "Repository Architecture & Scaffolding", "Configure pyproject.toml, package hierarchy, and core exception taxonomy (PaladinSecurityError).", "Rohit Gautam", "Completed", "src/core/exceptions.py\ntests/unit/test_exceptions.py"],
        ["PDM-02", "Payload Framing & Serialization", "Implement SecretSerializer with canonical b'PLDS' header, sorted JSON metadata, and CRC-32 trailer.", "Rohit Gautam", "Completed", "src/core/serialization/serializer.py\n14 unit tests pass"],
        ["PDM-03", "Authenticated Cryptography Vault", "Develop AESGCMProvider enforcing AES-256-GCM, fresh 96-bit IVs, 128-bit GHASH tag, AAD context binding, and fail-closed.", "Rohit Gautam", "Completed", "src/security/crypto/provider.py\n22 unit tests, NIST KAT pass"],
        ["PDM-04", "Galois Field Error Correction Coding", "Implement ReedSolomonEncoder over GF(2^8) with systematic chunking, 2t=32 parity bytes, and 4B prefix framing.", "Rohit Gautam", "Completed", "src/security/ecc/encoder.py\n23 unit tests, t=16 symbol repair"],
        ["PDM-05", "Security Pipeline Orchestration", "Build SecurityPipeline enforcing Serialize -> AES-GCM -> Reed-Solomon unidirectional sequence and exact reversal.", "Rohit Gautam", "Completed", "src/core/pipeline.py\nscripts/verify_sprint1.py (24/24)"],
        ["PDM-06", "Carrier Capacity Hard Gate", "Implement CapacityCalculator enforcing 8,176 B ceiling on 512x512 carriers, rejecting overflows before transform.", "Rohit Gautam", "Completed", "src/carrier/capacity.py\ntest_carrier_capacity.py"],
        ["PDM-07", "Wavelet Transform Signal Processing", "Implement 1-level 2D Haar DWT / IDWT in float64 precision decomposing 512x512 RGB images with zero numerical error.", "Rohit Gautam", "Completed", "src/carrier/dwt.py\ntest_carrier_dwt.py (max err=0.0)"],
        ["PDM-08", "Steganographic Framing & Deterministic Mask", "Build 16-byte fixed CarrierHeader (b'PLDH') and non-circular DeterministicMaskGenerator via SHA-256 PRNG.", "Rohit Gautam", "Completed", "src/carrier/header.py, mask.py\nTests T-013, T-014 pass"],
        ["PDM-09", "Transform-Domain Parity Modulation", "Implement BaselineEmbedder/Extractor modulating integer parity (a-b-c+d)%2 on Red HH detail subband.", "Rohit Gautam", "Completed", "src/carrier/embedding.py, extraction.py\nBER=0.000000 on PNG reload"],
        ["PDM-10", "End-to-End Media Steganography Pipeline", "Build MediaStegoPipeline orchestrating full media-to-media roundtrip with PSNR > 75 dB and exact byte equality.", "Rohit Gautam", "Completed", "src/core/media_pipeline.py\nTests T-019, T-020 pass"],
        ["PDM-11", "Observability API & Laboratory Frontend", "Develop FastAPI SSE server and Vite/React UI featuring 4-quadrant DWT visualizer, capacity meter, and live logs.", "Rohit Gautam", "Completed", "src/api/server.py\nfrontend/src/App.tsx"],
        ["PDM-12", "Multi-Carrier Partitioning Protocol", "Partition large payloads (>8 KB) across multiple distinct carrier covers with session framing and sequence headers.", "Rohit Gautam", "Partially Completed", "scripts/experiments/run_e09.py\n15 KB split over 3 carriers verified"],
        ["PDM-13", "ACS-Sub-INN v3 Invertible Flow Engine", "Continuous affine coupling block wavelet feature modulation to achieve lossy container (JPEG) robustness.", "Unassigned", "Planned", "docs/specifications/\nZero source code; Phase 3 research"],
        ["PDM-14", "CodeCarbon Environmental Telemetry", "Instrument embedding/extraction pipelines with CodeCarbon to measure Joules, power draw, and CO2 emissions.", "Unassigned", "Planned", "Phase 6 specification roadmap;\nNo runtime telemetry active"],
    ]
    t_bl = doc.add_table(rows=len(backlog_rows), cols=6)
    t_bl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_i, row in enumerate(backlog_rows):
        for c_i, val in enumerate(row):
            c = t_bl.cell(r_i, c_i)
            c.text = val
            set_cell_margins(c, 30, 30, 40, 40)
            p = c.paragraphs[0]
            if r_i == 0:
                set_cell_background(c, "1E3A8A")
                p.runs[0].bold = True
                p.runs[0].font.size = Pt(7.0)
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_background(c, "F8FAFC" if r_i % 2 == 1 else "FFFFFF")
                p.runs[0].font.size = Pt(6.5)
                p.runs[0].font.color.rgb = c_dark

    add_h2("Completed vs. Planned Features Subsection")
    add_p(
        "Completed Features (Phases 0–2): The repository contains a fully working, self-contained classical steganography system passing 103/103 tests in 2.36 s. "
        "The complete pipeline—from binary serialization to AES-256-GCM encryption, Reed-Solomon error correction, Haar DWT decomposition, integer-parity embedding, lossless PNG container generation, "
        "and receiver extraction—is 100% operational. It delivers high fidelity (PSNR = 75.92 dB, SSIM = 1.000000) and verbatim byte equality.\n"
        "Planned Features (Phases 3–6): The planned ACS-Sub-INN neural architecture, straight-through estimator (STE) quantization noise layers, ONNX Runtime INT8 deployment, "
        "and CodeCarbon energy telemetry are unimplemented forward research items. While detailed specifications exist in docs/specifications/, "
        "no neural training scripts, model weights, or telemetry hooks exist in src/. Furthermore, Experiment E14 conclusively proved that the classical baseline suffers 100% failure "
        "under lossy JPEG recompression, scientifically justifying the future transition to Phase 3 neural modeling."
    )

    doc.add_page_break()

    # Page 3 Content
    add_h1("Technical Design, System Architecture & Module Interactions")
    add_h2("1. System Architecture & Transformation Pipeline")
    add_p(
        "SENDER PIPELINE: Secret Media [Image/Text/JSON] -> SecretSerializer (b'PLDS', CRC-32) -> "
        "AESGCMProvider (AES-256-GCM, 96b IV, 128b Tag) -> ReedSolomonEncoder (GF(2^8), 2t=32 parity) -> "
        "CapacityCalculator (8,176 B Hard Gate) -> HaarDWT2D (1-level Haar DWT -> LL, LH, HL, HH) -> "
        "CarrierHeader (16B b'PLDH') -> BaselineEmbedder (2*HH parity on Red ch) -> IDWT -> Lossless 8-bit RGB PNG\n\n"
        "RECEIVER RECOVERY PIPELINE: Stego PNG -> HaarDWT2D -> Extractor (Extract 16B Header & Parity Bits) -> "
        "ReedSolomonEncoder.decode (Repairs <=16 symbol errors/block) -> AESGCMProvider.decrypt (Constant-time GHASH check) -> "
        "SecretSerializer.deserialize (CRC-32 verify) -> 100% Exact Recovered Media"
    )

    add_h2("2. Data-Flow & Byte Budget Transformation (512x512 Carrier)")
    dfd_rows = [
        ["Pipeline Stage", "Input Data", "Transformation & Framing Overhead", "Output Stream", "Size (Small Fixture)"],
        ["1. Serialization", "Secret Image / Text", "Prepends 10B header (Magic b'PLDS', version, media type), length (4B), JSON metadata, CRC-32 (4B)", "Framed Bytes", "152 B -> 170 B (+18 B)"],
        ["2. Encryption", "Framed Bytes", "AES-256-GCM encryption with fresh 12B IV and 16B GHASH authentication tag; binds optional AAD", "Protected Ciphertext", "170 B -> 198 B (+28 B)"],
        ["3. ECC Encoding", "Protected Ciphertext", "Systematic RS(255, 223) chunking; generates 32 parity bytes per block; prepends 4B big-endian length prefix", "Protected RS Bitstream", "198 B -> 234 B (+36 B)"],
        ["4. Capacity Gate", "Protected RS Bitstream", "Analytic capacity check: len(stream) <= 8,176 B. Aborts fail-closed before wavelet execution if exceeded", "Validated Bitstream", "234 B <= 8,176 B (PASS)"],
        ["5. Stego Embedding", "Cover Image (512x512)", "1-level Haar DWT; embeds 16B CarrierHeader (128 bits) + 234B payload (1,872 bits) via Red-HH parity modulation", "Modified Wavelet Coeffs", "2,000 modified quads"],
        ["6. IDWT & PNG", "Modified Wavelet Coeffs", "Inverse 2D Haar DWT reconstructing spatial domain; writes lossless 8-bit RGB PNG container to disk/buffer", "Stego PNG Container", "512x512 PNG (~380 KB)"],
    ]
    t_df = doc.add_table(rows=len(dfd_rows), cols=5)
    t_df.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_i, row in enumerate(dfd_rows):
        for c_i, val in enumerate(row):
            c = t_df.cell(r_i, c_i)
            c.text = val
            set_cell_margins(c, 30, 30, 40, 40)
            p = c.paragraphs[0]
            if r_i == 0:
                set_cell_background(c, "1E3A8A")
                p.runs[0].bold = True
                p.runs[0].font.size = Pt(7.0)
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_background(c, "F8FAFC" if r_i % 2 == 1 else "FFFFFF")
                p.runs[0].font.size = Pt(6.5)
                p.runs[0].font.color.rgb = c_dark

    add_h2("3. Major Architectural Components & Module Interactions")
    add_p(
        "• SecretSerializer (src/core/serialization/serializer.py): Enforces canonical binary wire framing (0x504C4453), media enumeration, compact sorted JSON metadata serialization, and trailer CRC-32 verification.\n"
        "• AESGCMProvider (src/security/crypto/provider.py): Implements AES-256-GCM authenticated cipher with 12-byte cryptographically secure random nonces and 16-byte GHASH tags. Enforces fail-closed operation, raising AuthenticationError on any bit corruption.\n"
        "• ReedSolomonEncoder (src/security/ecc/encoder.py): Systematic Galois Field GF(2^8) codec utilizing generator polynomial (x+a^0)...(x+a^31). Automatically splits streams into 223-byte chunks, appending 32 parity bytes to guarantee recovery of up to t=16 corrupted symbols per block.\n"
        "• CapacityCalculator (src/carrier/capacity.py): Evaluates carrier geometry in O(1) time. Computes raw bit budget (N^2/32 bytes), deducts 16B header, and computes max RS-protected payload (8,176 B for 512x512; 32,752 B for 1024x1024). Raises PayloadCapacityExceededError on boundary overflow.\n"
        "• HaarDWT2D & BaselineEmbedder (src/carrier/): Decomposes 2D spatial RGB images into orthogonal subbands (LL, LH, HL, HH). Modulates integer parity 2*HH = (a - b - c + d) % 2 in the Red channel by adjusting pixel d by +/-1. Extractor reconstructs bits directly from quad parities."
    )

    add_h2("4. Technology Stack & Architectural Rationale")
    add_p(
        "• Python 3.10.11: Primary implementation language offering native type annotations, dataclasses, and rich scientific ecosystem.\n"
        "• cryptography 49.0.0: Standard OpenSSL-backed cryptographic provider providing constant-time AES-NI hardware-accelerated encryption.\n"
        "• reedsolo 1.7.0: Pure-Python Galois Field GF(2^8) codec ensuring deterministic mathematical error correction without native C compilation.\n"
        "• numpy 2.2.6 & Pillow 12.2.0: Optimized vector/matrix arithmetic and lossless PNG codec ensuring zero quantization loss during carrier reload.\n"
        "• FastAPI 0.141.1 & Vite/React 19: Asynchronous REST/SSE streaming bridge and responsive laboratory UI for real-time subband visualization."
    )

    doc.add_page_break()

    # Page 4 Content
    add_h1("Green Computing Practices & Formal Verification Test Log")
    add_h2("1. Green Computing Implementation & Empirical Evaluation")
    add_p(
        "PaladinMedia integrates green computing principles directly into its architectural design. In accordance with strict academic audit constraints, "
        "only verifiable practices implemented and measured in the codebase are marked as active; forward telemetry is designated as Planned."
    )

    green_rows = [
        ["Practice", "Implementation in Repository", "Verified Evidence", "Concrete Benefit", "Limitation / Trade-off"],
        ["In-Place Vectorized Processing", "NumPy array operations and direct memory views used for Haar DWT subbands and parity slicing.", "src/carrier/dwt.py\nsrc/carrier/embedding.py", "Minimizes heap allocations; RSS memory delta only 4.7–5.8 MB per full roundtrip.", "Intermediate float64 representations required for lossless DWT inversion."],
        ["Minimalist Dependency Architecture", "Standard library + 4 lean packages (cryptography, reedsolo, numpy, pillow).", "pyproject.toml\nZero heavyweight ML frameworks", "Ultra-low disk/memory footprint; zero continuous GPU idle power drain.", "Precludes deep learning feature extraction in the classical baseline."],
        ["Fail-Closed O(1) Capacity Gating", "Analytic byte calculation aborts oversized payloads before performing DWT or matrix manipulations.", "src/carrier/capacity.py\ntest_carrier_capacity.py", "Eliminates 100% of wasted CPU cycles and memory allocations on invalid workloads.", "Requires pre-computation of carrier dimensions before execution."],
        ["Targeted, Deterministic Testing", "Exhaustive 103-test pytest suite executed in pure process isolation without persistent server daemons.", "103 tests pass in 2.36 s\nverify_sprint1.py (0.27 s)", "Eliminates redundant training loops and long CI runs, conserving compute energy.", "Does not simulate physical camera analog sensor noise."],
        ["Runtime Execution Profiling", "Built-in high-resolution timers (time.perf_counter) and process memory tracking (psutil).", "Observed latency: 92–106 ms\nRSS delta: 4.67 MB", "Provides verifiable empirical execution metrics for carbon modeling.", "Host OS scheduling introduces +/-5 ms variance across runs."],
        ["CodeCarbon Telemetry", "Automated background emission and Joule tracking per steganographic execution.", "Architecture §14 (Roadmap)\nPlanned (Not Active)", "Will provide standardized kWh and CO2e tracking across runs.", "Currently unimplemented; scheduled for Phase 6 laboratory suite."],
    ]
    t_gr = doc.add_table(rows=len(green_rows), cols=5)
    t_gr.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_i, row in enumerate(green_rows):
        for c_i, val in enumerate(row):
            c = t_gr.cell(r_i, c_i)
            c.text = val
            set_cell_margins(c, 30, 30, 40, 40)
            p = c.paragraphs[0]
            if r_i == 0:
                set_cell_background(c, "1E3A8A")
                p.runs[0].bold = True
                p.runs[0].font.size = Pt(7.0)
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_background(c, "F8FAFC" if r_i % 2 == 1 else "FFFFFF")
                p.runs[0].font.size = Pt(6.5)
                p.runs[0].font.color.rgb = c_dark

    add_h2("2. Sprint-1 Formal Test Log & Verification Results")
    add_p(
        "Every recorded test case corresponds to a genuine automated test in tests/ executed and verified under Python 3.10.11. "
        "The test suite achieved a 100% pass rate (103/103 passed in 2.36 s), verifying mathematical correctness and fail-closed security."
    )

    test_rows = [
        ["Test Case Identifier & Name", "Expected Result", "Actual Observed Result", "Status", "Verified Test File & Evidence"],
        ["test_t019_full_media_to_media_mvp", "PSNR > 50 dB, SSIM > 0.999, exact recovered image (24x24)", "PSNR = 75.89 dB, SSIM = 1.000000, verbatim byte equality", "PASS", "tests/integration/test_media_pipeline.py"],
        ["test_t020_exact_byte_equality", "Verbatim match for payload data, media type, and metadata", "100% exact equality; exact_match == True", "PASS", "tests/integration/test_media_pipeline.py"],
        ["test_capacity_exceeded_hard_gate", "Oversized payload (10 KB) rejected before transform execution", "PayloadCapacityExceededError raised fail-closed", "PASS", "tests/integration/test_media_pipeline.py"],
        ["test_aes_256_gcm_encrypt_decrypt", "Clean roundtrip with unique 96b nonce and 128b GHASH tag", "Exact plaintext recovered; nonces unique across runs", "PASS", "tests/unit/test_crypto.py"],
        ["test_nist_sp800_38d_kat_vectors", "Cryptographic vectors match official NIST KAT values exactly", "All NIST SP 800-38D test vectors validated", "PASS", "tests/unit/test_crypto.py"],
        ["test_fail_closed_ciphertext_bit_flip", "Bit-flipped ciphertext halts with AuthenticationError, 0 leak", "AuthenticationError raised; zero plaintext bytes leaked", "PASS", "tests/unit/test_crypto.py"],
        ["test_symbol_error_correction_bound_t", "Repair exactly t=16 corrupted symbols per 255-byte block", "All 16 corrupted bytes repaired; bitstream restored", "PASS", "tests/unit/test_ecc.py"],
        ["test_negative_uncorrectable_errors", "Reject bitstream with t+1=17 corrupted symbols fail-closed", "ECCDecodeError raised; execution halted", "PASS", "tests/unit/test_ecc.py"],
        ["test_t018_png_save_reload_carrier", "Zero bit error rate through lossless PNG disk encode/reload", "Bit Error Rate (BER) identically 0.000000", "PASS", "tests/unit/test_carrier_embedding.py"],
        ["test_canonical_wire_framing_layout", "Magic bytes b'PLDS', version 0x01, and CRC-32 validated", "Deterministic binary layout verified down to exact offsets", "PASS", "tests/unit/test_serialization.py"],
    ]
    t_ts = doc.add_table(rows=len(test_rows), cols=5)
    t_ts.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_i, row in enumerate(test_rows):
        for c_i, val in enumerate(row):
            c = t_ts.cell(r_i, c_i)
            c.text = val
            set_cell_margins(c, 30, 30, 40, 40)
            p = c.paragraphs[0]
            if r_i == 0:
                set_cell_background(c, "1E3A8A")
                p.runs[0].bold = True
                p.runs[0].font.size = Pt(7.0)
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_background(c, "F8FAFC" if r_i % 2 == 1 else "FFFFFF")
                p.runs[0].font.size = Pt(6.5)
                p.runs[0].font.color.rgb = c_dark

    doc.add_page_break()

    # Page 5 Content
    add_h1("Sprint Retrospective, Author Contributions & Forward Roadmap")
    add_h2("1. Sprint-1 Retrospective & Engineering Assessment")
    add_p(
        "What Worked Well: The implementation of the strict unidirectional pipeline (Serialize -> Encrypt -> ECC -> Embed) was completely successful. "
        "The cryptographic vault achieved absolute fail-closed security with verified zero plaintext leakage under all adversarial tampering scenarios. "
        "Embedding integer parities into the Red-HH Haar subband achieved flawless mathematical invertibility with BER = 0.000000 and PSNR > 75 dB across lossless PNG containers.\n"
        "What Did Not Work / Incomplete: The classical integer-parity baseline suffers from severe vulnerability to lossy distribution channels. "
        "As established in Experiment E14 (220 empirical runs), JPEG recompression at any quality factor (Q in [70, 100]) and additive Gaussian noise (sigma >= 0.5) "
        "destroys spatial integer parity (a - b - c + d) % 2, producing ~50% raw parity bit error rates and 100% recovery failure. "
        "This empirical failure establishes the fundamental necessity for Phase 3 neural steganography.\n"
        "Technical Challenges & Lessons Learned: (1) Initial assumptions of 8,192 B capacity collided with real wire framing overheads; resolved by developing explicit mathematical capacity accounting down to the exact byte. "
        "(2) Disk I/O (PNG encode/decode) accounts for >75% of total pipeline latency (~102 ms out of 134 ms total). "
        "(3) Separating the verified classical baseline from future neural designs prevented premature scope creep."
    )

    add_h2("2. Prioritized Next-Sprint Backlog (Sprint-2 / Phase 3)")
    next_rows = [
        ["ID", "Task Description", "Target Phase", "Priority", "Technical Rationale"],
        ["PDM-N01", "Scaffold ACS-Sub-INN v3 Invertible Neural Network Architecture", "Phase 3 (Neural)", "P0 (Highest)", "Transition from discrete integer parity to continuous affine coupling blocks to resist channel noise."],
        ["PDM-N02", "Implement Differentiable JPEG Noise Simulation Layer", "Phase 3 (Robustness)", "P0 (Highest)", "Empirically proven necessity from E14: model DCT quantization noise during training to survive JPEG Q=80."],
        ["PDM-N03", "Multi-Carrier Transport Partitioning CLI & Auto-Reassembly", "Phase 2 Extension", "P1 (High)", "Generalize E09 experiment script into production CLI for seamless multi-image secret reconstruction."],
        ["PDM-N04", "Steganalysis Resistance Evaluation against Spatial Rich Models (SRM)", "Phase 4 (Security)", "P1 (High)", "Close Gap B by training an ensemble FLD classifier on 34,671 SRM submodel features to quantify detection error."],
        ["PDM-N05", "Integrate CodeCarbon Energy & Carbon Telemetry Telemetry", "Phase 6 (Telemetry)", "P2 (Medium)", "Automate Joules and CO2e tracking per embedding session within the FastAPI observability stream."],
    ]
    t_nx = doc.add_table(rows=len(next_rows), cols=5)
    t_nx.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_i, row in enumerate(next_rows):
        for c_i, val in enumerate(row):
            c = t_nx.cell(r_i, c_i)
            c.text = val
            set_cell_margins(c, 30, 30, 40, 40)
            p = c.paragraphs[0]
            if r_i == 0:
                set_cell_background(c, "1E3A8A")
                p.runs[0].bold = True
                p.runs[0].font.size = Pt(7.0)
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_background(c, "F8FAFC" if r_i % 2 == 1 else "FFFFFF")
                p.runs[0].font.size = Pt(6.5)
                p.runs[0].font.color.rgb = c_dark

    add_h2("3. Individual Author Contributions & Verifiable Commit History")
    add_p(
        "PaladinMedia is developed as an individual engineering project. In accordance with academic integrity guidelines, "
        "all contributions are attributed to the sole project developer without fabricating team members or commit statistics. "
        "Git repository metadata records two canonical baseline commits:"
    )

    contrib_rows = [
        ["Member", "Tasks Done", "Evidence", "Commits"],
        [
            "Rohit Gautam\n(Reg No: 23215050)\nGitHub: RohitGautam0510\nrohit5.10gautam@gmail.com",
            "• Designed and implemented complete cryptographic vault (AES-256-GCM, fail-closed, 0 leak).\n"
            "• Implemented Reed-Solomon GF(2^8) error correction codec with systematic chunking.\n"
            "• Built 1-level 2D Haar DWT engine and integer-parity transform embedder/extractor.\n"
            "• Engineered carrier capacity accounting and pre-transform hard-gate enforcement.\n"
            "• Authored 103-test automated pytest suite and 24-check programmatic verification script.\n"
            "• Executed empirical experiments E03 through E14 (lossy channel degradation baseline).\n"
            "• Developed FastAPI SSE backend server and React/Vite laboratory user interface.",
            "• src/core/, src/security/, src/carrier/, src/api/\n"
            "• 103 passing pytest tests in tests/\n"
            "• scripts/verify_sprint1.py (24/24 pass)\n"
            "• docs/experiments/E03–E14 ledgers\n"
            "• frontend/src/App.tsx",
            "commit 9555996:\nfeat(audit): resolve P1 gate blockers with E14 lossy baseline, div2k split manifest, and acceptance criteria reconciliation\n\n"
            "commit a8c3f76:\nbaseline: phase 2 verified classical wavelet baseline and entry-gate audit"
        ]
    ]
    t_ct = doc.add_table(rows=len(contrib_rows), cols=4)
    t_ct.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_i, row in enumerate(contrib_rows):
        for c_i, val in enumerate(row):
            c = t_ct.cell(r_i, c_i)
            c.text = val
            set_cell_margins(c, 30, 30, 40, 40)
            p = c.paragraphs[0]
            if r_i == 0:
                set_cell_background(c, "1E3A8A")
                p.runs[0].bold = True
                p.runs[0].font.size = Pt(7.0)
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_background(c, "FFFFFF")
                p.runs[0].font.size = Pt(6.5)
                p.runs[0].font.color.rgb = c_dark

    doc.save(filename)
    print(f"[OK] Report DOCX successfully built at: {filename}")


if __name__ == "__main__":
    build_pdf(PDF_PATH)
    build_docx(DOCX_PATH)
