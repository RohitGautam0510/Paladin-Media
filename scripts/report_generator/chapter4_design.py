"""
Chapter 4: System Design for GreenTech Phase 3 Final Report docx generation.
Suggested length: 8–9 pages (~3,200 words).
Covers:
4.1 System Architecture (Multi-tier breakdown & Figure 4.1)
4.2 Data Flow Diagrams (DFD Level 0 & Level 1 & Figure 4.2)
4.3 UML Diagrams (4.3.1 Use Case Diagram & Figure 4.3; 4.3.2 Sequence Diagram & Figure 4.4)
4.4 Database Design, Schemas & Wire Protocol Layout
4.5 Algorithm / Model Design (Haar DWT, ACS-Sub-INN v3 math formulations, invertibility proof)
4.6 User Interface Design (Dark-mode OLED parsimony, side-by-side inspection)
4.7 Design for Sustainability (5 core design principles)
"""

import os
from styles import (
    add_chapter_title, add_section_heading, add_subsection_heading,
    add_body_p, add_bullet_p, add_table_with_caption, add_figure_with_caption,
    add_equation
)

def build_chapter4(doc):
    add_chapter_title(doc, "CHAPTER 4: SYSTEM DESIGN")

    # 4.1 System Architecture
    add_section_heading(doc, "4.1 System Architecture")
    add_body_p(
        doc,
        "The architecture of PaladinMedia is engineered as a decoupled, multi-tiered pipeline that strictly enforces the foundational doctrine: "
        "'Security Before Concealment'. Conventional steganographic platforms frequently blend cryptography, signal transformation, and UI logic "
        "into tightly coupled scripts, creating severe maintenance bottlenecks and security vulnerabilities. In contrast, PaladinMedia enforces "
        "strict separation of concerns across five cohesive operational layers: Presentation, Gateway, Cryptographic Core, Signal/Neural Core, "
        "and Telemetry Infrastructure, as illustrated in Figure 4.1."
    )

    fig_path = r"E:\Paladin_media\docs\reports\figures\fig_4_1_system_architecture.png"
    if os.path.exists(fig_path):
        add_figure_with_caption(
            doc, fig_path,
            "Figure 4.1: Multi-Tier System Architecture of the PaladinMedia Steganography Platform"
        )

    add_body_p(
        doc,
        "A detailed examination of each architectural tier reveals the underlying engineering mechanisms:"
    )
    add_bullet_p(
        doc,
        "Constructed using React 19, TypeScript 5.3, Vite 5.1, and Tailwind CSS 3.4. It executes entirely client-side within the user's web browser, "
        "providing drag-and-drop carrier image selection, raw secret text or binary file input, real-time capacity budget meters, interactive "
        "carrier inspection with difference amplification heatmaps, and a live Server-Sent Events (SSE) telemetry monitor. Crucially, the UI is styled "
        "in an OLED-optimized dark-mode palette (#0D1117 and #161B22) that turns off OLED display pixels, significantly conserving device battery power.",
        bold_prefix="1. Presentation Tier (Client Laboratory UI): "
    )
    add_bullet_p(
        doc,
        "Engineered with FastAPI 0.104 and Uvicorn on Python 3.11. It provides asynchronous RESTful endpoints (/api/v1/conceal and /api/v1/extract) "
        "and a streaming event route (/api/v1/telemetry/events). Incoming requests are filtered through custom ASGI middleware "
        "(RequestSizeLimitMiddleware) that enforces payload size gates and prevents denial-of-service memory exhaustion.",
        bold_prefix="2. Application & Gateway Tier: "
    )
    add_bullet_p(
        doc,
        "Encapsulates three sequential, strictly unidirectional security components: (a) SecretSerializer, which frames arbitrary payloads with "
        "canonical magic bytes (b'PLDS'), format versioning, media type identifiers, and CRC-32 integrity trailers; (b) CryptoProvider, which executes "
        "authenticated AES-256-GCM encryption with 96-bit CSPRNG nonces and 128-bit GHASH authentication tags; and (c) ECCEncoder, which chunks "
        "ciphertext into systematic Reed-Solomon GF(2^8) blocks with 2t=32 parity symbols per 255-byte block.",
        bold_prefix="3. Cryptographic & Data Integrity Core: "
    )
    add_bullet_p(
        doc,
        "Combines mathematical signal processing with deep learning: (a) 2D Haar DWT Engine, which decomposes 512×512 RGB images into LL, LH, HL, "
        "and HH subbands in float64 precision; (b) O(1) Capacity Hard Gate, which verifies that protected payloads do not exceed 8,176 bytes; and "
        "(c) ACS-Sub-INN v3 Neural Core, which uses invertible affine coupling blocks to modulate secret bits into the high-frequency Red-HH subband "
        "using Green-HH and Blue-HH as conditioning context.",
        bold_prefix="4. Signal Processing & Invertible Neural Core: "
    )
    add_bullet_p(
        doc,
        "Powered by ONNX Runtime 1.16 with session pooling, executing compiled static inference graphs locally on edge CPUs without GPU dependencies. "
        "Concurrently, the SustainabilityTelemetryTracker monitors execution latency (perf_counter), active CPU time, and RAM RSS memory footprint, "
        "computing dynamic power consumption and carbon emissions (g CO2e) per steganographic transaction.",
        bold_prefix="5. Hardware & Green Telemetry Tier: "
    )

    # 4.2 Data Flow Diagrams
    add_section_heading(doc, "4.2 Data Flow Diagrams")
    add_body_p(
        doc,
        "Data Flow Diagrams (DFDs) model the flow of information through the system, identifying external entities, processing functions, and "
        "data transformations. Figure 4.2 presents both the Level 0 Context DFD and the Level 1 Decomposed Subsystem DFD."
    )

    fig_path = r"E:\Paladin_media\docs\reports\figures\fig_4_2_dfd_levels.png"
    if os.path.exists(fig_path):
        add_figure_with_caption(
            doc, fig_path,
            "Figure 4.2: Data Flow Diagrams for PaladinMedia Platform (Level 0 Context and Level 1 Subsystems)"
        )

    add_body_p(
        doc,
        "In the Level 0 Context Diagram (Figure 4.2A), the entire system is represented as a single cohesive process (Process 0.0: PaladinMedia "
        "Core Platform). The Sender (Alice) provides a secret digital payload and an uncompressed 512×512 carrier image. The platform returns a visually "
        "indistinguishable, lossless stego PNG image accompanied by Green Computing telemetry. The Receiver (Bob) inputs the stego container, and the "
        "platform outputs the 100% verified, decrypted original secret."
    )
    add_body_p(
        doc,
        "The Level 1 Subsystem Flow Diagram (Figure 4.2B) decomposes Process 0.0 into its five sequential engineering stages: Process 1.0 (Framing) "
        "encapsulates the secret with canonical wire headers and CRC-32 trailers. Process 2.0 (Authenticated AES-256-GCM) encrypts the framed payload, "
        "producing ciphertext, CSPRNG nonces, and authentication tags. Process 3.0 (Reed-Solomon ECC Encoding) appends 32 parity bytes per block over "
        "GF(2^8). Process 4.0 (2D Haar DWT Decomposition) splits the 512×512 RGB carrier into spatial-frequency subbands. Process 5.0 (ACS-Sub-INN "
        "Modulation) embeds the protected bitstream into the Red-HH subband using invertible coupling operations, after which IDWT reconstructs the "
        "lossless stego PNG container."
    )

    # 4.3 UML Diagrams
    add_section_heading(doc, "4.3 UML Diagrams")
    add_body_p(
        doc,
        "Unified Modeling Language (UML) behavioral diagrams formalize the dynamic interactions between system actors, software components, and "
        "operational data flows."
    )

    # 4.3.1 Use Case Diagram
    add_subsection_heading(doc, "4.3.1 Use Case Diagram")
    add_body_p(
        doc,
        "The UML Use Case Diagram (Figure 4.3) identifies four primary actors and four core use cases within the PaladinMedia operational boundary:"
    )

    fig_path = r"E:\Paladin_media\docs\reports\figures\fig_4_3_use_case.png"
    if os.path.exists(fig_path):
        add_figure_with_caption(
            doc, fig_path,
            "Figure 4.3: UML Use Case Diagram for PaladinMedia Steganography Platform"
        )

    add_bullet_p(
        doc,
        "The Sender (Alice) initiates the 'Conceal Secret in Carrier' use case, selecting carrier images, defining secret payloads, and specifying "
        "encryption keys. Alice also triggers the 'Inspect Green Energy Telemetry' use case to verify the computational carbon footprint of the concealment.",
        bold_prefix="Sender Actor (Alice): "
    )
    add_bullet_p(
        doc,
        "The Receiver (Bob) executes the 'Authenticate & Decrypt Payload' use case by uploading a received stego PNG container. Bob's interaction "
        "verifies the Reed-Solomon parity, validates the 128-bit GHASH tag, and extracts the exact original payload.",
        bold_prefix="Receiver Actor (Bob): "
    )
    add_bullet_p(
        doc,
        "The Adversary (Eve) attempts to intercept or modify carrier image pixels in transit. Eve triggers the 'Detect Tampering & Fail-Closed Reject' "
        "use case, which guarantees that modified ciphertext or corrupted headers immediately halt extraction without leaking any plaintext.",
        bold_prefix="Adversary Actor (Eve): "
    )
    add_bullet_p(
        doc,
        "The Academic Auditor or Green Observer monitors the 'Inspect Green Energy Telemetry' use case to review live Joule consumption, "
        "CPU active utilization percentages, and CEA grid-referenced CO2e emissions.",
        bold_prefix="Auditor Actor (Green Observer): "
    )

    # 4.3.2 Sequence / Activity Diagram
    add_subsection_heading(doc, "4.3.2 Sequence / Activity Diagram")
    add_body_p(
        doc,
        "Figure 4.4 illustrates the end-to-end UML Sequence Diagram delineating the exact chronological message exchanges and invocation lifecycles "
        "across the User (Client), FastAPI Server, Cryptographic Core, and Wavelet/INN Engine."
    )

    fig_path = r"E:\Paladin_media\docs\reports\figures\fig_4_4_sequence.png"
    if os.path.exists(fig_path):
        add_figure_with_caption(
            doc, fig_path,
            "Figure 4.4: UML Sequence Diagram: End-to-End Concealment and Extraction Workflow"
        )

    add_body_p(
        doc,
        "During concealment, the client dispatches a multipart POST request (/api/v1/conceal). The FastAPI server coordinates the execution, "
        "calling frame_and_encrypt() on the CryptoCore, which returns authenticated ciphertext and Reed-Solomon parity blocks. The server passes "
        "the protected bitstream and carrier to the WaveletINN engine, which executes 2D Haar DWT, verifies the O(1) capacity hard gate, runs "
        "invertible neural embedding via ONNX Runtime, and performs IDWT reconstruction. The server packages the lossless stego PNG alongside "
        "real-time telemetry records, returning a 200 OK response. During extraction, the client uploads the stego PNG, the engine extracts the "
        "bitstream, corrects potential symbol errors via Reed-Solomon, validates the AES-GCM tag, and returns the 100% verified original payload."
    )

    # 4.4 Database Design and Data Schemas
    add_section_heading(doc, "4.4 Database Design and Data Schemas")
    add_body_p(
        doc,
        "To maximize privacy and eliminate forensic data traces, PaladinMedia deliberately eschews persistent disk-based relational databases. "
        "Instead, the system operates on a zero-persistence in-memory state model, where all secret payloads and carrier arrays exist strictly within "
        "transient process memory and are zeroized upon request completion. Operational data exchanges are formalized using strict Pydantic schemas "
        "and a canonical binary wire protocol layout."
    )
    add_body_p(
        doc,
        "Data exchange across the REST API is governed by four strongly-typed Pydantic schemas:"
    )
    add_bullet_p(
        doc,
        "Carries raw secret text, optional file bytes, secret media type (e.g. 'text/plain', 'application/json'), and a 256-bit passphrase.",
        bold_prefix="ConcealRequest Schema: "
    )
    add_bullet_p(
        doc,
        "Returns the base64-encoded lossless stego PNG, carrier dimensions, capacity utilization percentage, and embedded TelemetryRecord.",
        bold_prefix="ConcealResponse Schema: "
    )
    add_bullet_p(
        doc,
        "Uploads stego carrier bytes alongside the extraction passphrase.",
        bold_prefix="ExtractRequest Schema: "
    )
    add_bullet_p(
        doc,
        "Captures execution_time_ms, cpu_utilization_percent, memory_rss_mb, estimated_power_watts, energy_kwh, co2_emissions_grams, "
        "is_estimated (True), and telemetry_source ('estimated_hardware_model').",
        bold_prefix="TelemetryRecord Schema: "
    )
    add_body_p(
        doc,
        "Within the stego carrier image itself, data is organized using the Wire Protocol v2 binary header layout. The first 128 embedding "
        "positions (16 bytes) in the Red-HH subband are reserved for the canonical Carrier Header (b'PLDH'). This fixed-format binary header "
        "contains: (1) 4 bytes magic (b'PLDH'); (2) 1 byte protocol version (0x02); (3) 1 byte flag bitfield (indicating encryption and ECC mode); "
        "(4) 4 bytes big-endian unsigned integer encoding protected payload length; (5) 2 bytes reserved padding; and (6) 4 bytes CRC-32 checksum "
        "verifying header integrity. This design enables the extractor to validate carrier authenticity in O(1) time before attempting neural inversion."
    )

    # 4.5 Algorithm and Model Design
    add_section_heading(doc, "4.5 Algorithm and Model Design")
    add_body_p(
        doc,
        "The mathematical foundation of PaladinMedia integrates 2D Haar Discrete Wavelet Transforms with an Affine Coupling Scaling Subband "
        "Invertible Neural Network (ACS-Sub-INN v3). The mathematical formulations governing both components are rigorously defined below."
    )

    # 2D Haar DWT formulation
    add_subsection_heading(doc, "4.5.1 2D Haar Discrete Wavelet Transform Formulation")
    add_body_p(
        doc,
        "Given a 2D digital image channel I of dimensions H × W (where H = W = 512), single-level 2D Haar DWT decomposes I into four subbands "
        "of dimensions (H/2) × (W/2) = 256 × 256. For every non-overlapping 2×2 pixel quad [A, B; C, D] (where A = I[2i, 2j], B = I[2i, 2j+1], "
        "C = I[2i+1, 2j], and D = I[2i+1, 2j+1]), the forward transform equations are:"
    )

    add_equation(doc, "LL(i, j) = (A + B + C + D) / 2", "4.1")
    add_equation(doc, "LH(i, j) = (A + B - C - D) / 2", "4.2")
    add_equation(doc, "HL(i, j) = (A - B + C - D) / 2", "4.3")
    add_equation(doc, "HH(i, j) = (A - B - C + D) / 2", "4.4")

    add_body_p(
        doc,
        "Conversely, given the decomposed subbands, the exact spatial pixels are reconstructed without information loss via the Inverse Discrete "
        "Wavelet Transform (IDWT):"
    )

    add_equation(doc, "A = (LL + LH + HL + HH) / 2", "4.5")
    add_equation(doc, "B = (LL + LH - HL - HH) / 2", "4.6")
    add_equation(doc, "C = (LL - LH + HL - HH) / 2", "4.7")
    add_equation(doc, "D = (LL - LH - HL + HH) / 2", "4.8")

    add_body_p(
        doc,
        "By implementing Equations (4.1) through (4.8) using IEEE-754 float64 floating-point matrices, the transform achieves perfect mathematical "
        "invertibility: |I - IDWT(DWT(I))| < 10^-15, ensuring zero numerical distortion prior to steganographic modulation."
    )

    # ACS-Sub-INN v3 Formulation
    add_subsection_heading(doc, "4.5.2 ACS-Sub-INN v3 Neural Architecture and Invertibility Proof")
    add_body_p(
        doc,
        "The ACS-Sub-INN v3 architecture parameterizes a bijective mapping between carrier subband coefficients and embedded stego representations. "
        "The network operates on the high-frequency Red-HH subband (x_target in R^(1x256x256)), conditioned upon the context subbands formed by "
        "concatenating Green-HH and Blue-HH (x_context in R^(2x256x256))."
    )
    add_body_p(
        doc,
        "The core building block is an Affine Coupling Layer. The input feature x is split along channel or spatial dimensions into two halves, "
        "x1 and x2. The forward transformation (embedding) is defined as:"
    )

    add_equation(doc, "y_1 = x_1", "4.9")
    add_equation(doc, "y_2 = x_2 * exp(s(x_1, x_context)) + t(x_1, x_context)", "4.10")

    add_body_p(
        doc,
        "where s(.) is a neural scaling network and t(.) is a neural translation network, each parameterized by lightweight depthwise-separable "
        "convolutional blocks. Crucially, the backward transformation (extraction) is strictly mathematically invertible, regardless of the complexity "
        "or non-linearity of s(.) and t(.):"
    )

    add_equation(doc, "x_1 = y_1", "4.11")
    add_equation(doc, "x_2 = (y_2 - t(y_1, x_context)) * exp(-s(y_1, x_context))", "4.12")

    add_body_p(
        doc,
        "Mathematical Invertibility Proof: Substituting Equation (4.9) into (4.11) yields x1 = y1 = x1, which is an exact identity. Substituting "
        "y1 = x1 into Equation (4.12) gives: x2 = (y2 - t(x1, x_context)) * exp(-s(x1, x_context)). Substituting Equation (4.10) for y2 yields: "
        "x2 = ([x2 * exp(s(x1, x_context)) + t(x1, x_context)] - t(x1, x_context)) * exp(-s(x1, x_context)) = x2 * exp(s(x1, x_context)) * "
        "exp(-s(x1, x_context)) = x2 * 1 = x2. This proves that the inverse transformation recovers x2 exactly, with zero mathematical error."
    )
    add_body_p(
        doc,
        "To guarantee numerical stability during high-volume inference, the scaling network output s_raw is strictly bounded using the clamping formulation:"
    )

    add_equation(doc, "s_clamped = gamma_min * tanh(s_raw / gamma_min)", "4.13")

    add_body_p(
        doc,
        "Enforcing gamma_min = 1.0 bounds s_clamped within the interval [-1.0, +1.0], ensuring that exp(s_clamped) is strictly constrained between "
        "[1/e, e] approx [0.368, 2.718]. This completely eliminates exponential gradient explosions and division-by-zero singularities during edge execution."
    )

    # 4.6 User Interface Design
    add_section_heading(doc, "4.6 User Interface Design")
    add_body_p(
        doc,
        "The PaladinMedia laboratory interface was engineered to bridge complex steganographic algorithms with an intuitive, human-centered "
        "user experience. Key design decisions include:"
    )
    add_bullet_p(
        doc,
        "To address the energy waste of bright graphical interfaces, the application is styled in a deep dark-mode palette using zinc (#09090B), "
        "slate (#1E293B), and emerald (#10B981) accents. On modern OLED laptop and smartphone displays, black pixels draw zero electrical current, "
        "reducing client display power consumption by up to 38% compared to light-themed software.",
        bold_prefix="OLED Display Energy Conservation: "
    )
    add_bullet_p(
        doc,
        "The UI renders cover and stego images side-by-side with an interactive slider. A toggleable 'Difference Amplifier' multiplies pixel-wise "
        "luminance residuals by a factor of 20x, allowing users and evaluators to visually inspect the spatial distribution of steganographic "
        "modulations in high-texture carrier regions.",
        bold_prefix="Visual Carrier Inspection & Difference Amplification: "
    )
    add_bullet_p(
        doc,
        "As users type or paste secrets, a dynamic capacity progress bar calculates real-time payload consumption against the 7,088-byte budget, "
        "changing color from green (<70%) to amber (70-95%) and red (>95%), preventing accidental capacity gate rejections.",
        bold_prefix="Pre-Flight Capacity Budget Gauge: "
    )
    add_bullet_p(
        doc,
        "A dedicated telemetry card displays live metrics received via Server-Sent Events, including execution latency (ms), active CPU (%), "
        "power draw (W), and grams of CO2e avoided.",
        bold_prefix="Real-Time Telemetry Dashboard: "
    )

    # 4.7 Design for Sustainability
    add_section_heading(doc, "4.7 Design for Sustainability")
    add_body_p(
        doc,
        "Sustainability is not an afterthought in PaladinMedia; it is an architectural invariant embedded across the system lifecycle through "
        "five distinct engineering mechanisms:"
    )
    add_bullet_p(
        doc,
        "Modulating only high-frequency Red-HH wavelet coefficients limits data modifications to less than 8.3% of total carrier image coefficients, "
        "minimizing arithmetic operations and dynamic memory bit-flips.",
        bold_prefix="1. Frequency Domain Subband Sparsity: "
    )
    add_bullet_p(
        doc,
        "Exporting the ACS-Sub-INN model to static ONNX format enables aggressive operator fusion and dead-code elimination, allowing local edge CPU "
        "execution that eliminates the massive energy cost of transmitting high-resolution carrier images to remote cloud GPU clusters.",
        bold_prefix="2. Compiled Edge Graph Optimization: "
    )
    add_bullet_p(
        doc,
        "Array transformations in NumPy and ONNX utilize in-place buffer reuse and pre-allocated memory pools, avoiding frequent dynamic RAM "
        "reallocation and reducing memory controller energy consumption.",
        bold_prefix="3. In-Place Memory Allocation: "
    )
    add_bullet_p(
        doc,
        "The deployed production system operates strictly in inference mode with frozen weights (optimizer.step() disabled), completely avoiding "
        "the massive carbon footprint of iterative backpropagation in operational environments.",
        bold_prefix="4. Frozen Model Architecture: "
    )
    add_bullet_p(
        doc,
        "The system operates entirely on existing consumer laptops and desktop workstations, requiring no specialized hardware accelerators and "
        "diverting electronic waste (e-waste) by maximizing the functional lifespan of existing hardware.",
        bold_prefix="5. Hardware Longevity and E-Waste Diversion: "
    )

    print("Chapter 4 built successfully.")
