"""Generate PaladinMedia CIA-3 Sprint-1 Demo Presentation.

Outputs:
1. PaladinMedia_CIA3_Sprint1_Demo.pptx (Exact 8 Slides with Speaker Notes)
2. PaladinMedia_CIA3_Sprint1_Demo.pdf (Exact 8 Slides Landscape 16:9)

Structure:
Slide 1: Title and problem
Slide 2: Innovation
Slide 3: Sprint-1 achievements
Slide 4: System architecture
Slide 5: Functional prototype
Slide 6: Testing and results
Slide 7: Green computing and technical trade-offs
Slide 8: Retrospective and next sprint
"""

from __future__ import annotations

import os
import sys

# Ensure custom target directory for python-pptx is on sys.path
sys.path.insert(0, r"E:\pip_packages")

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas

OUT_DIR = os.path.abspath(r"E:\Paladin_media\submission\CIA3_Sprint1")
os.makedirs(OUT_DIR, exist_ok=True)
PPTX_PATH = os.path.join(OUT_DIR, "PaladinMedia_CIA3_Sprint1_Demo.pptx")
PDF_PATH = os.path.join(OUT_DIR, "PaladinMedia_CIA3_Sprint1_Demo.pdf")


# =============================================================================
# SLIDE METADATA & CONTENT DEFINITION (8 SLIDES)
# =============================================================================

SLIDES_DATA = [
    {
        "slide_num": 1,
        "title": "PaladinMedia: Post-Quantum Media Steganography",
        "subtitle": "CIA-3 Sprint-1 Working Prototype Demonstration & Technical Audit",
        "category": "PROJECT OVERVIEW & PROBLEM STATEMENT",
        "cards": [
            {
                "header": "Core Problem Statement",
                "points": [
                    "Spatial Steganography Vulnerability: Naive spatial LSB embedding exhibits high statistical detectability (easily discovered by SRM/SPAM classifiers) and zero error tolerance.",
                    "Lack of Authenticated Cryptography: Unprotected steganographic payloads offer zero tamper detection; corrupted bits yield silent execution crashes or garbage plaintext.",
                    "Capacity Budgeting Collision: Ad-hoc transform embedding without strict capacity contracts causes coefficient overflow and catastrophic signal degradation."
                ]
            },
            {
                "header": "Proposed Engineering Solution",
                "points": [
                    "Integrated Systems Pipeline: Combines canonical binary serialization (CRC-32), authenticated encryption (AES-256-GCM), and Galois Field error correction (Reed-Solomon RS(255, 223)).",
                    "Transform-Domain Concealment: 1-level 2D Haar DWT embeds integer-parity modulated bits into the Red HH detail subband, preserving spatial perceptual transparency (PSNR > 75 dB).",
                    "Absolute Fail-Closed Guarantee: Constant-time authentication verifies ciphertext before releasing any bytes; zero plaintext is released upon tampering."
                ]
            }
        ],
        "meta": "Contributor: Rohit Gautam (Reg No: 23215050)  |  Course: BCA411-7  |  Supervisor: Dr Preety  |  Baseline: Phase 2 Classical",
        "notes": "Good morning/afternoon, evaluators. I am presenting PaladinMedia, a post-quantum secured transform-domain media steganography system. This presentation covers the verified CIA-3 Sprint-1 functional prototype. Traditional steganography often treats data embedding as an isolated trick, ignoring authenticated encryption, capacity hard-gating, and error correction. PaladinMedia resolves this as an integrated systems engineering problem, establishing unidirectional sender protection, Galois Field error correction, and 2D Haar wavelet parity embedding."
    },
    {
        "slide_num": 2,
        "title": "Architectural Innovation & Research Boundaries",
        "subtitle": "Distinguishing Implemented Capabilities from Forward Research Vision",
        "category": "INNOVATION & DESIGN DIFFERENTIATION",
        "cards": [
            {
                "header": "Verified Implemented Capabilities (Phases 0–2)",
                "points": [
                    "Cryptographic Framing & Tamper Proofing: SecretSerializer enforces canonical b'PLDS' framing, sorted JSON metadata, and CRC-32 trailer integrity.",
                    "NIST SP 800-38D AES-256-GCM Vault: Fresh 96-bit random nonces and 128-bit GHASH authentication tags ensure fail-closed security with 0 plaintext leakage.",
                    "Galois Field GF(2^8) Reed-Solomon Codec: Systematic RS(255, 223) chunking repairs up to t=16 symbol errors per block and 128-bit contiguous bursts.",
                    "Lossless Carrier Pipeline: Exact integer-parity modulation on 2*HH coefficients in the Red channel guarantees 0.000000 bit error rate across 8-bit PNG reload."
                ]
            },
            {
                "header": "Forward Research Vision (Phase 3+ Invertible Neural Flow)",
                "points": [
                    "ACS-Sub-INN v3 Architecture: Proposed continuous invertible neural network using affine coupling blocks to modulate continuous wavelet feature representations.",
                    "Differentiable JPEG Training: Simulated lossy DCT noise layers during training to force secret representations into invariant low/mid-frequency subspaces.",
                    "Edge Deployment Scaffolding: Quantization-aware training (QAT) targeted for ONNX Runtime INT8 deployment.",
                    "Scientific Boundary: ACS-Sub-INN v3 is an unedited research specification; zero neural code or trained weights are claimed as implemented."
                ]
            }
        ],
        "meta": "Core Innovation: Multi-layered defense uniting cryptographic integrity, algebraic ECC, and transform-domain signal concealment.",
        "notes": "Here we clearly distinguish what is implemented today from our forward research vision. Today, PaladinMedia implements a rock-solid classical steganographic foundation with AES-256-GCM, Reed-Solomon error correction, and 2D Haar wavelet integer parity embedding. In our future Phase 3, we plan to implement ACS-Sub-INN v3, an invertible neural network for lossy container robustness. However, in accordance with scientific integrity, we declare zero neural code as implemented today; Phase 3 remains an unedited research specification."
    },
    {
        "slide_num": 3,
        "title": "Sprint-1 Verified Engineering Achievements",
        "subtitle": "Empirical Verification Across Automated Pytest and Invariant Check Suites",
        "category": "VERIFIED ACHIEVEMENTS & MILESTONES",
        "cards": [
            {
                "header": "100% Invariant Verification (scripts/verify_sprint1.py)",
                "points": [
                    "24/24 Checks Passing: Validates unidirectional serialization -> encryption -> ECC sequence and exact reversal across all media types in 0.27 seconds.",
                    "Full Reversibility: Verbatim recovery verified for TEXT, JSON, RAW_BYTES, IMAGE, and AUDIO payloads.",
                    "Boundary Acceptance: Handles 0-byte (empty), 1-byte, ~7 KB full subband capacity, and 12 KB multi-block streams.",
                    "Fail-Closed Security: Verified zero plaintext leakage across tampered ciphertext, tampered tags, tampered nonces, mismatched AAD, and wrong keys."
                ]
            },
            {
                "header": "Quantitative Baseline Results (tests/ & experiments/)",
                "points": [
                    "103 / 103 Pytest Tests Passing: Full automated test suite passes in 2.36 s with 0 failures and 0 regressions.",
                    "High Carrier Signal Fidelity: PSNR = 75.92 dB (threshold > 50 dB), SSIM = 1.000000 (imperceptible modification).",
                    "Zero Bit Error Rate: BER = 0.000000 across lossless PNG disk save and reload operations.",
                    "Capacity Hard Gate: 8,176 B protected ceiling enforced in O(1) time; 10 KB oversized payloads abort fail-closed."
                ]
            }
        ],
        "meta": "Automated Testing Baseline: 103 / 103 Unit, Integration, Security, and Media Roundtrip Tests Green (100% Pass Rate).",
        "notes": "Slide 3 presents our verified achievements for Sprint-1. We have achieved 100% passing results across two rigorous verification suites: our standalone 24-check invariant verification script (which runs in 0.27 seconds) and our comprehensive 103-test pytest suite (which runs in 2.36 seconds). We observe exceptional carrier fidelity: 75.92 dB PSNR and 1.000000 SSIM on our reference 512x512 carriers, with an extracted bit error rate of identically 0.000000 under lossless PNG storage."
    },
    {
        "slide_num": 4,
        "title": "System Architecture & Layered Transformation Flow",
        "subtitle": "Unidirectional Cryptographic Protection and Exact Mathematical Invertibility",
        "category": "TECHNICAL DESIGN & DATA FLOW",
        "cards": [
            {
                "header": "End-to-End Pipeline Transformation Flow",
                "points": [
                    "1. Secret Serialization: Secret Media (152 B) prepends b'PLDS' magic, version, metadata, and CRC-32 -> 170 B Framed.",
                    "2. Authenticated Encryption: AES-256-GCM prepends 12B fresh nonce and appends 16B GHASH tag -> 198 B Protected Ciphertext.",
                    "3. Error Correction: Systematic RS(255, 223) chunking appends 32B parity per block and 4B length prefix -> 234 B RS Stream.",
                    "4. Capacity Hard Gate: O(1) pre-check validates len(stream) <= 8,176 B; aborts before wavelet transform if exceeded.",
                    "5. Wavelet Steganography: 1-level Haar DWT embeds 16B CarrierHeader + payload into Red-HH detail subband via integer parity.",
                    "6. Receiver Inversion: Inverse DWT -> PNG reload -> DWT -> RS Decode (repairs <=16 errors) -> AES-GCM verify -> Exact Media."
                ]
            },
            {
                "header": "Core Architectural Invariants",
                "points": [
                    "INV-01 (Unidirectional Sequence): Serialization -> AES-GCM -> Reed-Solomon -> Carrier Embedding.",
                    "INV-02 (Ciphertext Embedding): Wavelet transform strictly operates on authenticated ciphertext, never plaintext.",
                    "INV-03 (Fail-Closed Security): Any bit mutation in ciphertext or tag halts receiver with AuthenticationError; 0 plaintext leaked.",
                    "INV-04 (Capacity Hard Gate): PayloadCapacityExceededError raised before wavelet modulation if payload exceeds carrier allocation."
                ]
            }
        ],
        "meta": "Technology Stack: Python 3.10.11, OpenSSL AES-256-GCM (cryptography 49.0), Galois Field GF(2^8) (reedsolo 1.7), NumPy 2.2, Pillow 12.2.",
        "notes": "Slide 4 shows the system architecture and data-flow diagram. The system strictly enforces unidirectional transformation: secret media is first serialized into b'PLDS' framing, then encrypted with AES-256-GCM, and encoded with Reed-Solomon RS(255, 223). Only authenticated, error-correcting codewords enter the carrier domain. Our capacity calculator enforces an 8,176-byte ceiling before DWT execution. On the receiver side, symmetric reversal recovers the exact original bytes."
    },
    {
        "slide_num": 5,
        "title": "Functional Prototype & Observability Interface",
        "subtitle": "Verified Local Execution via Programmatic CLI and Interactive Web Interface",
        "category": "FUNCTIONAL PROTOTYPE DEMONSTRATION",
        "cards": [
            {
                "header": "Dual Execution Environments (100% Local / Offline)",
                "points": [
                    "Programmatic CLI & Verification Scripts: Runs directly via pytest -v and verify_sprint1.py; complete end-to-end media roundtrip executes in 92–106 ms.",
                    "FastAPI SSE Backend Server: Asynchronous streaming server (src/api/server.py, port 8000) emitting real-time Server-Sent Events across 14 pipeline stages.",
                    "Vite / React 19 Laboratory UI: Interactive interface (frontend/, port 5173) featuring 4-quadrant DWT visualizer (LL, LH, HL, HH), live capacity meter, and telemetry log.",
                    "Offline Operation: Prototype operates completely on local loopback (127.0.0.1); zero external APIs, credentials, or cloud hosting required."
                ]
            },
            {
                "header": "Demonstrated Pipeline Stages in Laboratory UI",
                "points": [
                    "Stage 1–4: Load Media -> Payload Serialization (CRC-32) -> AES-256-GCM Encryption -> Reed-Solomon ECC Encoding.",
                    "Stage 5–8: Capacity Hard-Gate Check -> 1-Level 2D Haar DWT -> Deterministic Parity Embedding -> IDWT PNG Synthesis.",
                    "Stage 9–11: Stego PNG Disk Reload -> Receiver Stego DWT -> Header & Payload Extraction.",
                    "Stage 12–14: Reed-Solomon Error Correction -> AES-GCM Tag Verification -> Exact Byte Equality Verification."
                ]
            }
        ],
        "meta": "Local Handoff Guide: Documented in LOCAL_PROTOTYPE_README.md with exact reproduction commands and verified outputs.",
        "notes": "Slide 5 covers our functional prototype. We provide two verified ways to demonstrate PaladinMedia: first, our lightning-fast programmatic CLI and test scripts, which execute full end-to-end media steganography in under 100 milliseconds; and second, our rich laboratory web UI powered by FastAPI and Vite/React. The UI provides real-time visibility into all 14 stages of the pipeline, displaying the 4-quadrant Haar DWT decomposition, live capacity meters, and byte-exact verification."
    },
    {
        "slide_num": 6,
        "title": "Experimental Validation & Results Ledger (E03–E14)",
        "subtitle": "Quantitative Findings Across 12 Empirical Experiments on DIV2K Dataset",
        "category": "TESTING & QUANTITATIVE RESULTS",
        "cards": [
            {
                "header": "Lossless Baseline Verification (E03, E12, E13)",
                "points": [
                    "E03 (Media Steganography): PSNR = 75.89 dB, SSIM = 1.000, BER = 0.000000 on 512x512 reference carrier.",
                    "E12 (Continuous Density Sweep): PSNR decays monotonically from 81.9 dB (10% payload) to 62.0 dB (100% capacity). 99.85% saturation verified on 512x512; 100.0% on 1024x1024.",
                    "E13 (Multi-Carrier Repeatability): 90 runs across 5 DIV2K HR images, 2 sizes (512, 1024), 3 payload tiers (10%, 50%, 90%), and 3 seeds. Standard deviation < 0.06 dB; 90/90 runs achieved 100% byte-exact recovery."
                ]
            },
            {
                "header": "Lossy Channel Degradation Boundary (Experiment E14)",
                "points": [
                    "220 Empirical Runs: Evaluated 1 lossless control, 7 JPEG quality factors (Q=70..100), and 3 Gaussian noise channels (sigma=0.5, 1.0, 2.0).",
                    "100% Catastrophic Failure Under Lossy Channels: 0.0% recovery (0/140 runs) under JPEG compression; raw parity BER ~ 49.9% due to DCT quantization destroying spatial parity.",
                    "Vulnerability to Micro-Noise: At sigma=0.5 (imperceptible spatial distortion, PSNR=52.8 dB), raw parity BER reaches 49.1%, overwhelming Reed-Solomon budget.",
                    "Empirical Justification for Phase 3: Scientifically proves classical parity cannot survive lossy channels, necessitating Phase 3 continuous neural steganography."
                ]
            }
        ],
        "meta": "Experimental Grounding: 12 formal experiment ledgers in docs/experiments/ backed by machine-readable JSON run records.",
        "notes": "Slide 6 highlights our quantitative experimental results. In lossless PNG environments, experiments E03, E12, and E13 prove flawless steganography: in E13 across 90 runs on DIV2K HR images, we achieved zero bit errors and 100% verbatim recovery with standard deviation under 0.06 dB. However, in Experiment E14, 220 empirical runs proved that the classical baseline suffers 100% failure under JPEG recompression and Gaussian noise. This is not a failure of our implementation, but a fundamental property of integer-parity embedding, providing the empirical justification for our Phase 3 neural research."
    },
    {
        "slide_num": 7,
        "title": "Green Computing Practices & Technical Trade-offs",
        "subtitle": "Software Efficiency, Resource Telemetry, and Concrete Engineering Decisions",
        "category": "GREEN COMPUTING & TRADE-OFFS",
        "cards": [
            {
                "header": "Verifiable Green Computing Practices (SDG GreenTech)",
                "points": [
                    "In-Place NumPy & Buffer Manipulation: Re-uses array buffers and memory slices; process RSS memory delta is only 4.7–5.8 MB during full media roundtrip.",
                    "Lean Dependency Footprint: Zero heavy GPU or deep learning framework overhead; core baseline runs on standard library + 4 lean packages, eliminating idle GPU draw.",
                    "Fail-Closed O(1) Capacity Gating: Analytic pre-check aborts oversized payloads before wavelet decomposition, eliminating 100% of wasted CPU cycles on invalid tasks.",
                    "Targeted Deterministic Testing: 103 tests complete in 2.36 seconds, avoiding redundant training loops and long CI execution workloads.",
                    "Telemetry Roadmap: CodeCarbon energy and emission tracking scheduled for Phase 6 laboratory suite (currently marked Planned)."
                ]
            },
            {
                "header": "Key Technical Trade-offs & Engineering Realities",
                "points": [
                    "Security vs. Capacity: 28B AES-GCM + 32B RS parity per block reduces plaintext capacity from 8,176 B to 7,090 B in exchange for tamper protection and error repair.",
                    "Fragility vs. Zero-Distortion: Integer-parity modulation guarantees 0.000000 BER and >75 dB PSNR in lossless containers, but lacks robustness against lossy DCT compression.",
                    "CPU vs. I/O Latency: Pipeline execution is compute-efficient (~15 ms for crypto/DWT), while image disk I/O accounts for >75% of total roundtrip latency (~102 ms)."
                ]
            }
        ],
        "meta": "Green Computing Metrics: 92–106 ms execution latency, 4.67 MB memory RSS delta, 2.36 s full test suite duration.",
        "notes": "Slide 7 addresses our Green Computing practices and engineering trade-offs. Aligned with SDG GreenTech principles, our architecture is optimized for minimal resource consumption: we utilize in-place NumPy vectorization, lean dependencies, and an O(1) capacity hard gate that immediately halts invalid workloads before expensive matrix transforms. Total execution consumes only ~5 MB of additional RSS memory and runs in under 110 ms. We also openly discuss our trade-offs, such as dedicating ~13% of our capacity to cryptographic and ECC overhead in exchange for verified tamper proofing."
    },
    {
        "slide_num": 8,
        "title": "Sprint Retrospective, Contributions & Next Steps",
        "subtitle": "Self-Assessment, Individual Author Traceability, and Prioritized Next-Sprint Backlog",
        "category": "RETROSPECTIVE & FORWARD ROADMAP",
        "cards": [
            {
                "header": "Sprint-1 Retrospective Summary",
                "points": [
                    "What Worked: Unidirectional pipeline proven; 100% mathematical invertibility achieved; fail-closed security verified; 103/103 tests green.",
                    "What Did Not Work / Incomplete: Classical baseline cannot survive lossy distribution channels (E14 proved 100% failure under JPEG recompression).",
                    "Key Learnings: Capacity accounting must be solved analytically down to the exact byte; disk I/O dominates end-to-end latency."
                ]
            },
            {
                "header": "Prioritized Next-Sprint Backlog (Sprint-2 / Phase 3)",
                "points": [
                    "P0 (Highest): Scaffold ACS-Sub-INN v3 Invertible Neural Network architecture with continuous affine coupling blocks.",
                    "P0 (Highest): Implement differentiable JPEG simulation layer to train lossy container robustness against DCT quantization.",
                    "P1 (High): Productionize Multi-Carrier Transport Partitioning (MCPP) CLI for large multi-image secret reconstruction (E09).",
                    "P1 (High): Conduct empirical steganalysis detectability baseline against Spatial Rich Models (SRM) to close Gap B.",
                    "P2 (Medium): Integrate CodeCarbon energy and carbon emission telemetry into FastAPI streaming adapter."
                ]
            }
        ],
        "meta": "Sole Project Author: Rohit Gautam (Reg No: 23215050) | GitHub: RohitGautam0510 (rohit5.10gautam@gmail.com) | Commits: 9555996, a8c3f76.",
        "notes": "To conclude on Slide 8: Sprint-1 has successfully delivered a rock-solid, mathematically proven, and fail-closed classical steganographic foundation passing 103 automated tests. All contributions are credited to the sole project author, Rohit Gautam, backed by clean Git commit history. Our next sprint will focus on scaffolding Phase 3: implementing the ACS-Sub-INN v3 invertible neural network with differentiable JPEG simulation to conquer lossy transmission channels. Thank you, and I look forward to your questions."
    }
]


# =============================================================================
# PPTX GENERATION (PYTHON-PPTX)
# =============================================================================

def build_pptx(filename=PPTX_PATH):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # Blank slide

    c_navy = RGBColor(30, 58, 138)       # #1E3A8A
    c_blue = RGBColor(37, 99, 235)       # #2563EB
    c_slate = RGBColor(15, 23, 42)       # #0F172A
    c_dark = RGBColor(30, 41, 59)        # #1E293B
    c_muted = RGBColor(100, 116, 139)    # #64748B
    c_card_bg = RGBColor(248, 250, 252)  # #F8FAFC
    c_card_border = RGBColor(203, 213, 225) # #CBD5E1
    c_accent_bg = RGBColor(239, 246, 255) # #EFF6FF

    for data in SLIDES_DATA:
        slide = prs.slides.add_slide(blank_layout)
        
        # 1. Top Category Pill / Banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = f"PALADINMEDIA CIA-3 SPRINT-1  •  SLIDE {data['slide_num']} OF 8  •  {data['category']}"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = c_blue

        # 2. Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.733), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = data["title"]
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = c_navy

        # 3. Subtitle
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.733), Inches(0.35))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = data["subtitle"]
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = c_muted

        # 4. Content Cards (Two columns: Left and Right)
        card_w = Inches(5.7)
        card_h = Inches(4.7)
        card_y = Inches(1.8)

        # Left Card
        card1_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), card_y, card_w, card_h)
        card1_shape.fill.solid()
        card1_shape.fill.fore_color.rgb = c_card_bg
        card1_shape.line.color.rgb = c_card_border
        card1_shape.line.width = Pt(1)

        tf_c1 = card1_shape.text_frame
        tf_c1.word_wrap = True
        tf_c1.margin_left = Inches(0.25)
        tf_c1.margin_right = Inches(0.25)
        tf_c1.margin_top = Inches(0.2)
        tf_c1.margin_bottom = Inches(0.2)
        
        p_c1_h = tf_c1.paragraphs[0]
        p_c1_h.text = data["cards"][0]["header"]
        p_c1_h.font.size = Pt(13)
        p_c1_h.font.bold = True
        p_c1_h.font.color.rgb = c_navy
        p_c1_h.space_after = Pt(8)

        for pt in data["cards"][0]["points"]:
            p_pt = tf_c1.add_paragraph()
            p_pt.text = f"• {pt}"
            p_pt.font.size = Pt(10)
            p_pt.font.color.rgb = c_dark
            p_pt.space_after = Pt(6)

        # Right Card
        card2_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.833), card_y, card_w, card_h)
        card2_shape.fill.solid()
        card2_shape.fill.fore_color.rgb = c_card_bg
        card2_shape.line.color.rgb = c_card_border
        card2_shape.line.width = Pt(1)

        tf_c2 = card2_shape.text_frame
        tf_c2.word_wrap = True
        tf_c2.margin_left = Inches(0.25)
        tf_c2.margin_right = Inches(0.25)
        tf_c2.margin_top = Inches(0.2)
        tf_c2.margin_bottom = Inches(0.2)
        
        p_c2_h = tf_c2.paragraphs[0]
        p_c2_h.text = data["cards"][1]["header"]
        p_c2_h.font.size = Pt(13)
        p_c2_h.font.bold = True
        p_c2_h.font.color.rgb = c_navy
        p_c2_h.space_after = Pt(8)

        for pt in data["cards"][1]["points"]:
            p_pt = tf_c2.add_paragraph()
            p_pt.text = f"• {pt}"
            p_pt.font.size = Pt(10)
            p_pt.font.color.rgb = c_dark
            p_pt.space_after = Pt(6)

        # 5. Bottom Metadata Strip
        meta_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.65), Inches(11.733), Inches(0.4))
        tf_meta = meta_box.text_frame
        tf_meta.word_wrap = True
        p_meta = tf_meta.paragraphs[0]
        p_meta.text = data["meta"]
        p_meta.font.size = Pt(9.5)
        p_meta.font.color.rgb = c_muted

        # 6. Speaker Notes
        if hasattr(slide, "notes_slide") and slide.notes_slide:
            notes_tf = slide.notes_slide.notes_text_frame
            notes_tf.text = data["notes"]

    prs.save(filename)
    print(f"[OK] Presentation PPTX successfully built at: {filename}")


# =============================================================================
# REPORTLAB PDF PRESENTATION GENERATION (MATCHING 8 SLIDES 16:9 LANDSCAPE)
# =============================================================================

class SlideDeckCanvas(canvas.Canvas):
    """Canvas for landscape presentation slides."""
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
            self.draw_slide_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_slide_decorations(self, page_count):
        self.saveState()
        # Top banner line
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(40, 500, 920, 500)
        
        # Bottom footer line
        self.line(40, 35, 920, 35)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(40, 22, "PaladinMedia CIA-3 Sprint-1 Demonstration Slide Deck  •  Academic Prototype Handoff")
        self.drawRightString(920, 22, f"Slide {self._pageNumber} of {page_count}")
        self.restoreState()


def build_presentation_pdf(filename=PDF_PATH):
    # 16:9 widescreen: 960 x 540 pt
    doc = SimpleDocTemplate(
        filename,
        pagesize=(960, 540),
        leftMargin=40,
        rightMargin=40,
        topMargin=25,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    
    c_navy = colors.HexColor("#1E3A8A")
    c_blue = colors.HexColor("#2563EB")
    c_dark = colors.HexColor("#1E293B")
    c_muted = colors.HexColor("#64748B")
    c_border = colors.HexColor("#CBD5E1")
    c_card_bg = colors.HexColor("#F8FAFC")

    cat_style = ParagraphStyle(
        'SlideCat', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8.5, leading=10, textColor=c_blue, spaceAfter=2
    )
    title_style = ParagraphStyle(
        'SlideTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=c_navy, spaceAfter=2
    )
    sub_style = ParagraphStyle(
        'SlideSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=13, textColor=c_muted, spaceAfter=8
    )
    card_h_style = ParagraphStyle(
        'CardH', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=c_navy, spaceAfter=5
    )
    card_pt_style = ParagraphStyle(
        'CardPt', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=c_dark, spaceAfter=4.5
    )
    meta_style = ParagraphStyle(
        'SlideMeta', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8.2, leading=10, textColor=c_muted
    )

    story = []

    for idx, data in enumerate(SLIDES_DATA):
        # Category Banner
        story.append(Paragraph(f"PALADINMEDIA CIA-3 SPRINT-1  •  SLIDE {data['slide_num']} OF 8  •  {data['category']}", cat_style))
        story.append(Paragraph(data["title"], title_style))
        story.append(Paragraph(data["subtitle"], sub_style))

        # Build 2-column card layout
        left_elements = [Paragraph(data["cards"][0]["header"], card_h_style)]
        for pt in data["cards"][0]["points"]:
            left_elements.append(Paragraph(f"• {pt}", card_pt_style))

        right_elements = [Paragraph(data["cards"][1]["header"], card_h_style)]
        for pt in data["cards"][1]["points"]:
            right_elements.append(Paragraph(f"• {pt}", card_pt_style))

        cards_table = Table([[left_elements, right_elements]], colWidths=[430, 430])
        cards_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), c_card_bg),
            ('BACKGROUND', (1,0), (1,0), c_card_bg),
            ('BOX', (0,0), (0,0), 0.6, c_border),
            ('BOX', (1,0), (1,0), 0.6, c_border),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(cards_table)
        story.append(Spacer(1, 8))

        # Bottom Metadata
        story.append(Paragraph(data["meta"], meta_style))

        if idx < len(SLIDES_DATA) - 1:
            story.append(PageBreak())

    doc.build(story, canvasmaker=SlideDeckCanvas)
    print(f"[OK] Presentation PDF successfully built at: {filename}")


if __name__ == "__main__":
    build_pptx(PPTX_PATH)
    build_presentation_pdf(PDF_PATH)
