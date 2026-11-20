"""
Chapter 3: Requirement Analysis and Project Management for GreenTech Phase 3 Final Report docx generation.
Suggested length: 6–7 pages (~2,400 words).
Covers:
3.1 Stakeholder Analysis
3.2 Functional Requirements (Table 3.1: FR1 to FR8 with MoSCoW prioritization)
3.3 Non-Functional Requirements (NFR1 to NFR6)
3.4 Hardware and Software Requirements (Table 3.2)
3.5 Feasibility Study (Technical, Operational, Economic, Environmental)
3.6 Project Management using Scrum:
    3.6.1 Scrum Team and Roles (Table 3.3)
    3.6.2 Product Backlog (Table 3.4: Master User Stories US-1 to US-8)
    3.6.3 Sprint Plan and Execution (Table 3.5 & Figure 3.1)
    3.6.4 Sprint Retrospectives (Sprint 1, 2, 3 retrospectives)
"""

import os
from styles import (
    add_chapter_title, add_section_heading, add_subsection_heading,
    add_body_p, add_bullet_p, add_table_with_caption, add_figure_with_caption
)

def build_chapter3(doc):
    add_chapter_title(doc, "CHAPTER 3: REQUIREMENT ANALYSIS AND PROJECT MANAGEMENT")

    # 3.1 Stakeholder Analysis
    add_section_heading(doc, "3.1 Stakeholder Analysis")
    add_body_p(
        doc,
        "A comprehensive requirement engineering process begins with identifying the operational stakeholders who interact with, evaluate, "
        "or are impacted by the deployed software system. The PaladinMedia platform recognizes three distinct tiers of stakeholders:"
    )
    add_bullet_p(
        doc,
        "These include investigative journalists, human rights whistleblowers, digital privacy advocates, and legal professionals operating "
        "in restrictive jurisdictional environments. Their primary requirement is absolute communication deniability combined with mathematical "
        "confidentiality. They demand an intuitive, client-side interface that conceals sensitive text documents, JSON records, or cryptographic "
        "credentials inside ordinary-looking carrier images without requiring remote cloud subscriptions that could leave forensic server trails.",
        bold_prefix="Primary Stakeholders (Privacy-Conscious End Users): "
    )
    add_bullet_p(
        doc,
        "These comprise edge application developers, corporate security officers, and systems integration engineers who incorporate steganographic "
        "and cryptographic routines into existing software workflows. Their requirements focus on API robustness, deterministic error handling, "
        "predictable execution latency, low memory footprint, and standards-compliant binary serialization formats.",
        bold_prefix="Secondary Stakeholders (Systems & Edge Engineers): "
    )
    add_bullet_p(
        doc,
        "These include faculty mentors, academic project evaluation committees at CHRIST (Deemed to be University), and sustainability compliance "
        "auditors monitoring adherence to the United Nations Sustainable Development Goals. Their expectations center upon empirical rigor, "
        "reproducible benchmark evidence, transparent Scrum documentation, and verifiable Green Computing telemetry.",
        bold_prefix="Tertiary Stakeholders (Academic & Sustainability Evaluators): "
    )

    # 3.2 Functional Requirements
    add_section_heading(doc, "3.2 Functional Requirements")
    add_body_p(
        doc,
        "Functional requirements define the core operational behaviors, input-output transformations, and algorithmic tasks the system must "
        "execute. To structure development across the agile lifecycle, requirements were categorized using the MoSCoW prioritization framework "
        "(Must have, Should have, Could have, Won't have), as detailed in Table 3.1."
    )

    headers_3_1 = ["Req ID", "Requirement Description", "Input / Trigger", "Output / Behavior", "Priority"]
    data_3_1 = [
        ["FR1", "Canonical Payload Framing & CRC-32 Validation", "Arbitrary bytes, media type",
         "Canonical b'PLDS' frame with version & CRC-32 trailer.", "Must"],
        ["FR2", "Authenticated AES-256-GCM Encryption", "Framed bytes, 256-bit key",
         "96-bit CSPRNG nonce, ciphertext, 128-bit GHASH tag.", "Must"],
        ["FR3", "Galois Field RS GF(2^8) Error Correction", "Ciphertext bitstream",
         "Systematic RS blocks with 2t=32 parity symbols.", "Must"],
        ["FR4", "2D Haar Wavelet Subband Decomposition", "512x512 RGB carrier",
         "Exact float64 LL, LH, HL, HH subbands (3x256x256).", "Must"],
        ["FR5", "O(1) Carrier Capacity Hard Gating", "Secret payload length",
         "Immediate rejection if payload > 8,176 protected bytes.", "Must"],
        ["FR6", "ACS-Sub-INN v3 Invertible Neural Modulation", "Protected bits + HH subband",
         "Bijective residual embedding with gamma_min >= 1.0.", "Must"],
        ["FR7", "Real-Time Green Hardware Telemetry Streaming", "Execution trigger / API call",
         "Live SSE events with Joules, latency, CPU%, and g CO2e.", "Should"],
        ["FR8", "Interactive Dark-Mode Laboratory Interface", "User file upload via browser",
         "Dual-pane carrier visualization, difference heatmaps.", "Should"]
    ]
    add_table_with_caption(
        doc,
        "Table 3.1: System Functional Requirements with MoSCoW Prioritization",
        headers_3_1, data_3_1, [0.8, 1.8, 1.3, 1.3, 0.6]
    )

    # 3.3 Non-Functional Requirements
    add_section_heading(doc, "3.3 Non-Functional Requirements")
    add_body_p(
        doc,
        "Non-functional requirements specify the operational quality attributes, performance constraints, and architectural invariants that "
        "govern system execution:"
    )
    add_bullet_p(
        doc,
        "The cryptographic subsystem must enforce fail-closed security. Any modification to ciphertext, authentication tags, or Additional "
        "Authenticated Data (AAD) must trigger an immediate authentication exception, zeroizing memory buffers and returning zero plaintext.",
        bold_prefix="NFR1 — Cryptographic Security & Fail-Closed Integrity: "
    )
    add_bullet_p(
        doc,
        "The steganographic carrier must preserve high perceptual fidelity. Across the standard DIV2K benchmark split, the system must achieve "
        "a Peak Signal-to-Noise Ratio (PSNR) greater than 42.0 dB and a Structural Similarity Index Measure (SSIM) greater than 0.980, ensuring "
        "zero humanly discernible distortion.",
        bold_prefix="NFR2 — Perceptual Imperceptibility: "
    )
    add_bullet_p(
        doc,
        "End-to-end embedding and extraction pipelines must execute with real-time responsiveness. Total computational latency on a standard "
        "consumer-grade multi-core edge CPU must not exceed 300 milliseconds per 512×512 carrier image.",
        bold_prefix="NFR3 — Computational Latency & Edge Responsiveness: "
    )
    add_bullet_p(
        doc,
        "Under lossless container transmission, the system must achieve 100% exact byte recovery fidelity, corresponding to a Bit Error Rate (BER) "
        "of exactly 0.00%. Furthermore, the system must tolerate burst errors up to 16 corrupted bytes per 255-byte block without data loss.",
        bold_prefix="NFR4 — Recovery Reliability & Error Resilience: "
    )
    add_bullet_p(
        doc,
        "Active resident memory (RAM RSS) during peak neural wavelet inference must remain bounded below 200 megabytes (MB), completely eliminating "
        "the requirement for dedicated desktop graphics processing hardware.",
        bold_prefix="NFR5 — Memory Boundedness & Resource Parsimony: "
    )
    add_bullet_p(
        doc,
        "The codebase must adhere to strict modularity and high testability, verified by an automated regression test suite achieving greater "
        "than 90% code coverage across all core modules.",
        bold_prefix="NFR6 — Code Quality & Test Verification: "
    )

    # 3.4 Hardware and Software Requirements
    add_section_heading(doc, "3.4 Hardware and Software Requirements")
    add_body_p(
        doc,
        "To ensure reproducible deployment across standard academic and edge environments, Table 3.2 details the precise hardware and software "
        "specifications utilized during development, verification, and target deployment."
    )

    headers_3_2 = ["Platform Tier", "Subsystem / Item", "Technical Specification / Version", "Operational Role"]
    data_3_2 = [
        ["Hardware Tier", "Processor (CPU)", "Intel Core i7-12700H (14 cores, 20 threads, up to 4.7 GHz)", "Primary development, profiling & benchmarking host."],
        ["Hardware Tier", "System Memory (RAM)", "16 GB DDR5 Dual-Channel 4800 MHz", "Buffer allocation, dataset caching, in-place transforms."],
        ["Hardware Tier", "Local Storage", "512 GB PCIe Gen4 NVMe SSD (read > 3,500 MB/s)", "Dataset storage (DIV2K HR), model ONNX weight caching."],
        ["Software Tier", "Operating System", "Microsoft Windows 11 Pro 64-bit (Build 22631)", "Primary development, test harness, edge host."],
        ["Software Tier", "Python Runtime", "Python 3.11.8 (64-bit CPython runtime)", "Core cryptographic, signal, and API server execution."],
        ["Software Tier", "Deep Learning Framework", "PyTorch 2.2.1 + TorchVision 0.17.1 (CPU build)", "Neural INN coupling block definition and training baseline."],
        ["Software Tier", "Inference Engine", "ONNX Runtime 1.16.3 (Optimized C++ execution)", "Static compiled graph execution, kernel fusion, pooling."],
        ["Software Tier", "Cryptography & FEC", "cryptography 42.0.5 + reedsolo 1.7.0", "AES-256-GCM cipher and Reed-Solomon GF(2^8) codecs."],
        ["Software Tier", "Application Backend", "FastAPI 0.104.1 + Uvicorn 0.24.0 (ASGI Server)", "RESTful endpoints, SSE telemetry bus, async middleware."],
        ["Software Tier", "Frontend Framework", "React 19 + TypeScript 5.3 + Vite 5.1 + Tailwind CSS 3.4", "Laboratory dark-mode UI, SSE stream consumer."]
    ]
    add_table_with_caption(
        doc,
        "Table 3.2: Development, Testing, and Edge Deployment Platform Specifications",
        headers_3_2, data_3_2, [1.1, 1.3, 1.8, 1.5]
    )

    # 3.5 Feasibility Study
    add_section_heading(doc, "3.5 Feasibility Study")
    add_body_p(
        doc,
        "Prior to committing implementation resources, a multi-dimensional feasibility study evaluated the viability of the proposed system:"
    )
    add_bullet_p(
        doc,
        "The mathematical foundation of 2D Haar wavelets and AES-256-GCM cryptography is thoroughly mature and standardized. Furthermore, the "
        "viability of deploying invertible neural networks via ONNX Runtime on consumer CPUs was demonstrated by achieving sub-300 ms latencies "
        "in early prototypes. The open-source availability of robust libraries (cryptography, reedsolo, onnxruntime) confirms high technical feasibility.",
        bold_prefix="Technical Feasibility: "
    )
    add_bullet_p(
        doc,
        "The platform operates as a self-contained local web service requiring no external cloud connections or third-party proprietary software. "
        "The modern browser-based laboratory interface provides clear drag-and-drop workflows, making the system readily usable by non-technical "
        "stakeholders, verifying high operational feasibility.",
        bold_prefix="Operational Feasibility: "
    )
    add_bullet_p(
        doc,
        "The entire software stack is built upon permissive open-source licenses (MIT, Apache 2.0, BSD). Deployment requires zero commercial software "
        "licenses or dedicated GPU hardware purchases, guaranteeing zero financial barrier to adoption.",
        bold_prefix="Economic Feasibility: "
    )
    add_bullet_p(
        doc,
        "The project directly reduces computational carbon intensity by substituting heavy GPU inference with compiled edge CPU execution, "
        "saving over 70% electrical power per transaction and demonstrating definitive positive environmental feasibility.",
        bold_prefix="Environmental Feasibility: "
    )

    # 3.6 Project Management using Scrum
    add_section_heading(doc, "3.6 Project Management using Scrum")
    add_body_p(
        doc,
        "To manage software complexity and ensure continuous verification across Phase 3, the project adopted an Agile Scrum project management "
        "methodology tailored to an academic engineering context. Development was divided into two-week sprints, governed by structured sprint "
        "backlogs, regular stand-ups, automated verification gates, and formal retrospectives."
    )

    # 3.6.1 Scrum Team and Roles
    add_subsection_heading(doc, "3.6.1 Scrum Team and Roles")
    add_body_p(
        doc,
        "In accordance with academic integrity guidelines, the project was executed by a solo student researcher under faculty supervision. "
        "Table 3.3 delineates the formal Scrum roles and governance allocations."
    )

    headers_3_3 = ["Team Member", "Scrum Role", "Academic Position", "Core Responsibilities"]
    data_3_3 = [
        ["Dr. Preety", "Product Owner &\nAcademic Mentor", "Associate Professor,\nDept of Computer Science",
         "Backlog approval, architectural review, acceptance criteria validation, academic milestone evaluation."],
        ["Rohit Gautam\n(Reg: 23215050)", "Scrum Master, Lead Dev\n& QA Engineer", "BCA (Honours) Student\nResearcher",
         "Sprint backlog planning, pipeline implementation (crypto, neural, UI), test automation, telemetry modeling."]
    ]
    add_table_with_caption(
        doc,
        "Table 3.3: PaladinMedia Project Roles and Governance Responsibilities",
        headers_3_3, data_3_3, [1.2, 1.3, 1.4, 1.8]
    )

    # 3.6.2 Product Backlog
    add_subsection_heading(doc, "3.6.2 Product Backlog")
    add_body_p(
        doc,
        "The Master Product Backlog captured user requirements decomposed into atomic, testable User Stories (US), assigned story points "
        "based on Fibonacci estimation (1, 2, 3, 5, 8, 13), as presented in Table 3.4."
    )

    headers_3_4 = ["Story ID", "User Story Description", "Points", "Sprint", "Verification Status"]
    data_3_4 = [
        ["US-1", "As a sender, I want my payload serialized with magic bytes and CRC-32 so that structural corruption is caught early.", "3", "Sprint 1", "Verified (Done)"],
        ["US-2", "As a sender, I want payload encryption using AES-256-GCM so that eavesdroppers cannot read ciphertext if extracted.", "5", "Sprint 1", "Verified (Done)"],
        ["US-3", "As a receiver, I want Reed-Solomon error correction so that minor bit errors in carrier transmission do not break decryption.", "5", "Sprint 1", "Verified (Done)"],
        ["US-4", "As a system, I want 2D Haar DWT decomposition so that data is embedded in high-frequency subbands with minimal visual artifact.", "5", "Sprint 1", "Verified (Done)"],
        ["US-5", "As a developer, I want an O(1) capacity hard gate so that oversized secrets are rejected before modifying carrier pixels.", "2", "Sprint 1", "Verified (Done)"],
        ["US-6", "As a researcher, I want an ACS-Sub-INN neural core with gamma clamping so that embedding is strictly mathematically invertible.", "8", "Sprint 2", "Verified (Done)"],
        ["US-7", "As an edge engineer, I want the INN compiled to ONNX Runtime so that inference executes under 300 ms on edge CPUs.", "8", "Sprint 2", "Verified (Done)"],
        ["US-8", "As an eco-auditor, I want real-time hardware energy telemetry streamed via SSE so that Joules and CO2e are monitored per run.", "5", "Sprint 3", "Verified (Done)"],
        ["US-9", "As an end user, I want a dark-mode React 19 UI so that I can visually inspect carriers while reducing OLED display power.", "5", "Sprint 3", "Verified (Done)"],
        ["US-10", "As a QA engineer, I want an end-to-end regression harness of >450 Pytests so that architectural contracts remain unbroken.", "5", "Sprint 3", "Verified (Done)"]
    ]
    add_table_with_caption(
        doc,
        "Table 3.4: Master Product Backlog across Sprints 1, 2, and 3",
        headers_3_4, data_3_4, [0.8, 2.5, 0.6, 0.8, 1.0]
    )

    # 3.6.3 Sprint Plan and Execution
    add_subsection_heading(doc, "3.6.3 Sprint Plan and Execution")
    add_body_p(
        doc,
        "Phase 3 execution was structured across three focused two-week sprints. Table 3.5 provides the consolidated sprint execution matrix, "
        "summarizing sprint durations, primary goals, planned versus completed story points, and review outcomes. Figure 3.1 illustrates the "
        "daily sprint burndown trajectories across all three sprints, demonstrating stable velocity and timely completion of all planned commitments."
    )

    headers_3_5 = ["Sprint", "Duration", "Sprint Goal", "Planned Pts", "Completed Pts", "Velocity / Review Outcome"]
    data_3_5 = [
        ["Sprint 1", "28/09/2026 – 06/10/2026", "Cryptographic foundation, RS-ECC, DWT float64 engine, capacity gating.", "20", "20", "100% velocity; 24 acceptance tests verified."],
        ["Sprint 2", "07/10/2026 – 16/10/2026", "ACS-Sub-INN v3 neural core, DIV2K dataset split, ONNX export & pooling.", "24", "24", "100% velocity; 440 regression tests verified."],
        ["Sprint 3", "17/10/2026 – 28/10/2026", "FastAPI SSE server, React 19 UI, Green AI telemetry, full-stack harness.", "22", "22", "100% velocity; 499 tests verified, 0 unhandled."]
    ]
    add_table_with_caption(
        doc,
        "Table 3.5: Consolidated Sprint Execution and Velocity Matrix",
        headers_3_5, data_3_5, [0.8, 1.4, 2.0, 0.7, 0.7, 1.1]
    )

    # Figure 3.1: Burndown chart
    fig_path = r"E:\Paladin_media\docs\reports\figures\fig_3_1_burndown.png"
    if os.path.exists(fig_path):
        add_figure_with_caption(
            doc, fig_path,
            "Figure 3.1: PaladinMedia Sprint Burndown Trajectories across Sprints 1, 2, and 3"
        )

    # 3.6.4 Sprint Retrospectives
    add_subsection_heading(doc, "3.6.4 Sprint Retrospectives")
    add_body_p(
        doc,
        "At the conclusion of each sprint, formal retrospectives evaluated process efficiency, highlighted technical roadblocks, and formulated "
        "adaptive process improvements:"
    )
    add_bullet_p(
        doc,
        "What went well: The cryptographic and error correction pipeline was completed ahead of schedule, passing all negative security test cases. "
        "What stalled: Initial experiments with float32 DWT introduced minor rounding discrepancies upon inverse transform. "
        "Adaptation: Switched all internal wavelet computation matrices strictly to IEEE-754 float64 precision, eliminating roundtrip inversion drift.",
        bold_prefix="Sprint 1 Retrospective (06 October 2026): "
    )
    add_bullet_p(
        doc,
        "What went well: ACS-Sub-INN coupling blocks achieved perfect mathematical invertibility. ONNX compilation reduced inference latency by 73%. "
        "What stalled: Deep coupling layers occasionally experienced gradient explosion when scaling parameters approached zero. "
        "Adaptation: Implemented hard clamping on the scaling network: s_clamp = exp(gamma * tanh(s_raw)) with gamma_min = 1.0, stabilizing all gradients.",
        bold_prefix="Sprint 2 Retrospective (16 October 2026): "
    )
    add_bullet_p(
        doc,
        "What went well: React 19 UI and FastAPI SSE telemetry streaming integrated smoothly. Full test suite reached 499 passing Pytests. "
        "What stalled: The optional codecarbon package caused environment installation conflicts on constrained edge platforms. "
        "Adaptation: Engineered an in-tree, hardware-profiled power estimation model (TDP CPU active scaling + RAM power density) enforcing strict transparency.",
        bold_prefix="Sprint 3 Retrospective (28 October 2026): "
    )

    print("Chapter 3 built successfully.")
