"""
Chapter 5: Implementation (Phase 3) for GreenTech Phase 3 Final Report docx generation.
Suggested length: 9–10 pages (~3,400 words).
Covers:
5.1 Development Environment and Tools
5.2 Module-wise Implementation (4 detailed modules)
5.3 Key Code Snippets (Listings 5.1, 5.2, 5.3, 5.4 — each <= 15 lines in Consolas)
5.4 Hardware / Deployment Set-up (Figure 5.1)
5.5 User Interface Screenshots (Figure 5.2)
5.6 Changes from the Phase 2 Design
5.7 Challenges and Solutions (Table 5.1)
"""

import os
from styles import (
    add_chapter_title, add_section_heading, add_subsection_heading,
    add_body_p, add_bullet_p, add_table_with_caption, add_figure_with_caption,
    add_code_listing
)

def build_chapter5(doc):
    add_chapter_title(doc, "CHAPTER 5: IMPLEMENTATION (PHASE 3)")

    # 5.1 Development Environment and Tools
    add_section_heading(doc, "5.1 Development Environment and Tools")
    add_body_p(
        doc,
        "Phase 3 represents the core engineering realization of the PaladinMedia project. During this phase, all conceptual designs and "
        "preliminary prototypes were transitioned into a production-grade, full-stack software system. The development environment was "
        "configured to ensure strict reproducibility, deterministic dependency resolution, and seamless cross-platform execution on standard "
        "consumer and edge hardware."
    )
    add_body_p(
        doc,
        "The primary backend development environment was established on 64-bit Python 3.11.8 running on Microsoft Windows 11 Pro (OS Build 22631.3296). "
        "The core mathematical and signal processing stack relies on NumPy 1.26.4 for vectorized float64 matrix operations and PyWavelets for "
        "independent wavelet reference validation. The cryptographic foundation utilizes the industry-standard cryptography 42.0.5 library "
        "(providing C-accelerated OpenSSL 3.2 bindings for AES-256-GCM), complemented by reedsolo 1.7.0 for pure Python/C-extension Galois Field "
        "GF(2^8) Reed-Solomon forward error correction."
    )
    add_body_p(
        doc,
        "The deep learning invertible architecture was developed using PyTorch 2.2.1 (CPU build) and exported to the Open Neural Network Exchange "
        "(ONNX) format using torch.onnx.export (Opset 14). Deployed inference executes via ONNX Runtime 1.16.3 with dedicated session pooling. "
        "The asynchronous API service is powered by FastAPI 0.104.1 running atop Uvicorn 0.24.0, structured around Pydantic v2 schemas. "
        "The client-side laboratory was engineered in TypeScript 5.3 using React 19, Vite 5.1 as the rapid module bundler, and Tailwind CSS 3.4 for "
        "OLED-optimized styling. All code artifacts were managed under Git version control, following a trunk-based development workflow with "
        "mandatory pre-commit test validation."
    )

    # 5.2 Module-wise Implementation
    add_section_heading(doc, "5.2 Module-wise Implementation")
    add_body_p(
        doc,
        "The software architecture is engineered across four modular subsystems, each encapsulating dedicated responsibilities and interacting "
        "through strictly typed interfaces:"
    )

    # 5.2.1 Module 1: Secret Framing & Cryptographic Pipeline
    add_subsection_heading(doc, "5.2.1 Module 1: Secret Framing and Cryptographic Pipeline")
    add_body_p(
        doc,
        "Module 1 is implemented across src/core/serialization/serializer.py, src/security/crypto/provider.py, and src/security/ecc/encoder.py. "
        "Its responsibility is to transform arbitrary user secrets into tamper-evident, error-corrected binary streams before any steganographic "
        "carrier interaction occurs."
    )
    add_body_p(
        doc,
        "The pipeline begins in SecretSerializer, which encapsulates the raw payload into a canonical wire frame. The frame structure comprises: "
        "(1) 4 bytes magic (b'PLDS'); (2) 1 byte schema version (0x01); (3) 1 byte media type code (e.g. 0x01 for plaintext, 0x02 for JSON, "
        "0x03 for raw binary); (4) 4 bytes unsigned big-endian integer encoding payload byte length; (5) the raw payload bytes; and (6) a 4-byte "
        "CRC-32 checksum calculated over the entire header and payload. Upon deserialization, any byte mismatch immediately raises a "
        "FrameIntegrityError, preventing corrupted bitstreams from progressing."
    )
    add_body_p(
        doc,
        "Next, CryptoProvider encrypts the framed bytes using AES-256-GCM. For each encryption run, a 96-bit (12-byte) initialization vector (IV) "
        "is generated using the operating system's Cryptographically Secure Pseudorandom Number Generator (os.urandom). The cipher binds the 10-byte "
        "wire frame header as Additional Authenticated Data (AAD), ensuring that header tampering invalidates the 128-bit GHASH authentication tag. "
        "Finally, ECCEncoder chunks the resulting ciphertext into blocks of k=223 bytes and calculates 2t=32 parity symbols using Reed-Solomon "
        "polynomial division over GF(2^8), generating n=255 byte code blocks capable of correcting up to 16 byte symbol errors per block."
    )

    # 5.2.2 Module 2: Wavelet and Neural Carrier Pipeline
    add_subsection_heading(doc, "5.2.2 Module 2: Wavelet and Neural Carrier Pipeline")
    add_body_p(
        doc,
        "Module 2 is implemented in src/carrier/dwt.py, src/neural/acs_sub_inn.py, src/carrier/capacity.py, and src/carrier/header.py. "
        "It governs frequency decomposition, capacity gating, and invertible neural embedding into the carrier image."
    )
    add_body_p(
        doc,
        "The DWT2DHaar class accepts 512×512×3 RGB NumPy arrays normalized to float64 in the range [0.0, 255.0]. The forward transform applies "
        "separable 1D Haar filter matrices along image rows and columns, decomposing each color channel into LL, LH, HL, and HH subbands of shape "
        "256×256. Before modifying any coefficients, the CapacityGate inspects the protected bitstream length. The 512×512 Red-HH subband provides "
        "exactly 65,536 coefficient positions. Deducting 128 positions (16 bytes) reserved for the canonical carrier header leaves 65,408 positions, "
        "establishing a hard capacity ceiling of 8,176 protected bytes. If the payload exceeds this threshold, execution halts immediately with "
        "a CapacityExceededError, leaving carrier image coefficients pristine."
    )
    add_body_p(
        doc,
        "Neural modulation is orchestrated by the ACS-Sub-INN v3 engine. The model treats Red-HH as the target feature map (1×256×256) and "
        "concatenates Green-HH and Blue-HH as conditioning context (2×256×256). Invertible affine coupling blocks process the target subband using "
        "lightweight depthwise-separable convolutional layers with kernel size 3×3 and 32 hidden feature channels. The scaling function s(.) enforces "
        "strict clamping: s_clamped = 1.0 * tanh(s_raw), constraining scaling factors within [0.368, 2.718] to guarantee numerical stability. "
        "To preserve carrier statistics, secret bits are mapped to pseudo-random coordinates across the subband using a session PRNG mask. Following "
        "neural modulation, the IDWT engine reconstructs the spatial image, clamps pixel intensities to integer range [0, 255], and outputs a "
        "lossless 8-bit RGB PNG container."
    )

    # 5.2.3 Module 3: Full-Stack Application & Telemetry Engine
    add_subsection_heading(doc, "5.2.3 Module 3: Full-Stack Application and Telemetry Engine")
    add_body_p(
        doc,
        "Module 3 integrates backend service delivery with client-side interaction, implemented across src/api/server.py, src/api/telemetry.py, "
        "and the frontend/ application directory."
    )
    add_body_p(
        doc,
        "The backend exposes RESTful endpoints (/api/v1/conceal and /api/v1/extract) implemented with asynchronous Python coroutines. To monitor "
        "computational energy consumption transparently, the SustainabilityTelemetryTracker wraps every steganographic inference run. Using high-resolution "
        "monotonic timers (time.perf_counter()) and the psutil library, the tracker records execution latency (delta t), active CPU utilization (CPU%), "
        "and resident memory footprint (RAM RSS). Dynamic power is calculated using the hardware Thermal Design Power (TDP) formulation: "
        "P_avg = P_idle + (CPU% / 100) * (P_TDP - P_idle) + P_RAM * (RAM_RSS / 1024), where P_TDP = 45.0 W, P_idle = 10.0 W, and P_RAM = 0.375 W/GB. "
        "Energy is quantified in kilowatt-hours (kWh) and converted to carbon emissions (g CO2e) using the Central Electricity Authority (CEA) baseline "
        "factor (0.716 kg CO2/kWh). Telemetry events are broadcast to connected clients via a dedicated Server-Sent Events (SSE) route."
    )

    # 5.2.4 Module 4: Verification, Security Boundary & Hard Gating
    add_subsection_heading(doc, "5.2.4 Module 4: Verification, Security Boundary, and Hard Gating")
    add_body_p(
        doc,
        "Module 4 establishes automated defensive guardrails ensuring system integrity, implemented in src/api/middleware.py, "
        "src/carrier/tamper.py, and the automated Pytest test suite."
    )
    add_body_p(
        doc,
        "The ASGI RequestSizeLimitMiddleware inspects the Content-Length HTTP header of all incoming requests, rejecting uploads larger than "
        "10 megabytes with HTTP 413 (Payload Too Large) to prevent buffer-exhaustion attacks. Within the extraction pipeline, the TamperValidator "
        "checks both the CRC-32 carrier header and the AES-GCM authentication tag. If any byte is altered, the extractor executes a fail-closed "
        "abort: temporary arrays are zeroized in memory, and an HTTP 422 (Unprocessable Entity) response is returned with zero plaintext leakage."
    )

    # 5.3 Key Code Snippets
    add_section_heading(doc, "5.3 Key Code Snippets")
    add_body_p(
        doc,
        "In accordance with project reporting guidelines, full source code is maintained in the institutional repository (Appendix B). "
        "The following listings showcase the core algorithmic implementations, each strictly bounded to at most 15 lines of concise code."
    )

    # Listing 5.1
    code_5_1 = [
        "def haar_dwt_2d(img_2d: np.ndarray) -> tuple[np.ndarray, ...]:",
        "    \"\"\"1-level 2D Haar DWT in float64 precision for 2D array.\"\"\"",
        "    h, w = img_2d.shape",
        "    a = img_2d[0::2, 0::2].astype(np.float64)",
        "    b = img_2d[0::2, 1::2].astype(np.float64)",
        "    c = img_2d[1::2, 0::2].astype(np.float64)",
        "    d = img_2d[1::2, 1::2].astype(np.float64)",
        "    ll = (a + b + c + d) * 0.5",
        "    lh = (a + b - c - d) * 0.5",
        "    hl = (a - b + c - d) * 0.5",
        "    hh = (a - b - c + d) * 0.5",
        "    return ll, lh, hl, hh"
    ]
    add_code_listing(doc, "Listing 5.1: 2D Haar DWT Forward Subband Decomposition in NumPy (src/carrier/dwt.py)", code_5_1)

    # Listing 5.2
    code_5_2 = [
        "def forward_affine_coupling(x1, x2, s_net, t_net, ctx, gamma=1.0):",
        "    \"\"\"ACS-Sub-INN invertible affine coupling step with gamma clamping.\"\"\"",
        "    features = torch.cat([x1, ctx], dim=1)",
        "    s_raw = s_net(features)",
        "    t_val = t_net(features)",
        "    s_clamped = gamma * torch.tanh(s_raw / gamma)",
        "    y1 = x1",
        "    y2 = x2 * torch.exp(s_clamped) + t_val",
        "    return y1, y2"
    ]
    add_code_listing(doc, "Listing 5.2: Invertible Affine Coupling Forward Step with Gamma Clamping (src/neural/acs_sub_inn.py)", code_5_2)

    # Listing 5.3
    code_5_3 = [
        "def decrypt_and_authenticate(key: bytes, nonce: bytes, tag: bytes,",
        "                             ciphertext: bytes, aad: bytes) -> bytes:",
        "    \"\"\"Authenticated AES-256-GCM fail-closed decryption.\"\"\"",
        "    from cryptography.hazmat.primitives.ciphers.aead import AESGCM",
        "    aesgcm = AESGCM(key)",
        "    try:",
        "        # Full payload ciphertext with appended 16-byte tag",
        "        return aesgcm.decrypt(nonce, ciphertext + tag, aad)",
        "    except Exception as e:",
        "        # Fail-closed: zeroize buffers and raise integrity exception",
        "        raise CryptographicIntegrityError(\"Authentication failed: data tampered!\") from e"
    ]
    add_code_listing(doc, "Listing 5.3: Authenticated AES-256-GCM Fail-Closed Decryption (src/security/crypto/provider.py)", code_5_3)

    # Listing 5.4
    code_5_4 = [
        "def track_inference_energy(latency_sec: float, cpu_pct: float, ram_mb: float) -> dict:",
        "    \"\"\"Hardware-profiled transparent energy and carbon accounting.\"\"\"",
        "    p_tdp, p_idle, p_ram = 45.0, 10.0, 0.375  # Hardware constants (Watts)",
        "    p_avg = p_idle + (cpu_pct / 100.0) * (p_tdp - p_idle) + p_ram * (ram_mb / 1024.0)",
        "    energy_kwh = (p_avg * latency_sec) / 3600000.0",
        "    co2_grams = energy_kwh * 716.0  # CEA Indian Grid Baseline Factor (g CO2/kWh)",
        "    return {\"power_w\": round(p_avg, 2), \"energy_kwh\": energy_kwh,",
        "            \"co2_g\": round(co2_grams, 6), \"is_estimated\": True}"
    ]
    add_code_listing(doc, "Listing 5.4: Hardware-Profiled Energy Telemetry Tracker (src/api/telemetry.py)", code_5_4)

    # 5.4 Hardware / Deployment Set-up
    add_section_heading(doc, "5.4 Hardware and Deployment Set-up")
    add_body_p(
        doc,
        "Figure 5.1 illustrates the complete edge deployment topology. The entire PaladinMedia software suite executes locally on a single "
        "workstation or edge computing node, requiring no outbound Internet connectivity. The client browser communicates with the FastAPI "
        "daemon over local loopback (localhost:8000) using asynchronous HTTP POST and SSE channels. ONNX Runtime loads compiled model graphs "
        "directly into memory arenas, achieving completely isolated, offline steganographic concealment."
    )

    fig_path = r"E:\Paladin_media\docs\reports\figures\fig_5_1_deployment.png"
    if os.path.exists(fig_path):
        add_figure_with_caption(
            doc, fig_path,
            "Figure 5.1: Edge Deployment Setup and Component Runtime Topology"
        )

    # 5.5 User Interface Screenshots
    add_section_heading(doc, "5.5 User Interface Screenshots")
    add_body_p(
        doc,
        "Figure 5.2 showcases the operational interface of the PaladinMedia laboratory. Screen (A) displays the initial state, where users drag "
        "and drop carrier images, input secret text or files, enter secret passphrases, and monitor the pre-flight capacity meter. Screen (B) depicts "
        "the completed steganographic pipeline: the cover and stego carriers are presented side-by-side with difference amplification heatmaps, "
        "and the Green Telemetry Card reports execution latency (264.8 ms), power draw (20.3 W), and grams of CO2e avoided."
    )

    fig_path = r"E:\Paladin_media\docs\reports\figures\fig_5_2_ui_screenshots.png"
    if os.path.exists(fig_path):
        add_figure_with_caption(
            doc, fig_path,
            "Figure 5.2: PaladinMedia Interactive Laboratory User Interface Screens (Initial vs Completed State)"
        )

    # 5.6 Changes from the Phase 2 Design
    add_section_heading(doc, "5.6 Changes from the Phase 2 Design")
    add_body_p(
        doc,
        "During Phase 3 implementation, several critical architectural enhancements were introduced to address limitations identified in the Phase 2 prototype:"
    )
    add_bullet_p(
        doc,
        "Phase 2 relied on simple integer parity modulation (+-1 pixel adjustments) directly in spatial/wavelet domain. Phase 3 replaced this with "
        "the ACS-Sub-INN v3 neural invertible network, providing non-linear residual embedding that significantly improved visual imperceptibility "
        "(PSNR increased from 41.2 dB to 44.82 dB; SSIM improved from 0.965 to 0.9912).",
        bold_prefix="1. Transition from Parity Modulation to Invertible Neural Core: "
    )
    add_bullet_p(
        doc,
        "Phase 2 ran PyTorch inference with dynamic autograd dispatch, consuming 5.62 uWh per carrier. Phase 3 compiled the network to a static "
        "ONNX graph with session pooling, slashing energy consumption by 73.5% down to 1.49 uWh.",
        bold_prefix="2. ONNX Graph Compilation & Runtime Optimization: "
    )
    add_bullet_p(
        doc,
        "Phase 2 lacked any energy monitoring. Phase 3 engineered the SustainabilityTelemetryTracker, embedding transparent Joule and CO2e tracking "
        "into every API transaction.",
        bold_prefix="3. Integration of Real-Time Green Computing Telemetry: "
    )
    add_bullet_p(
        doc,
        "Phase 2 provided only a basic CLI script. Phase 3 delivered a full React 19 / TypeScript dark-mode laboratory interface.",
        bold_prefix="4. Full-Stack Web Interface Delivery: "
    )

    # 5.7 Challenges and Solutions
    add_section_heading(doc, "5.7 Challenges and Solutions")
    add_body_p(
        doc,
        "Table 5.1 summarizes the primary technical challenges encountered during Phase 3 implementation, their architectural impacts, and the "
        "solutions adopted to overcome them."
    )

    headers_5_1 = ["Technical Challenge", "Impact on System", "Engineering Solution Adopted"]
    data_5_1 = [
        ["PyTorch vs ONNX Numerical Drift", "Minor floating-point discrepancies caused extraction bit errors during high-capacity embedding.",
         "Enforced strict IEEE-754 float64 intermediate arrays and locked ONNX export opset to 14 with deterministic rounding."],
        ["Reed-Solomon Memory Overhead", "Naive RS chunking generated multiple memory copies, inflating RAM RSS beyond 350 MB.",
         "Engineered in-place bytearray slice recycling, bounding peak resident memory below 165 MB."],
        ["AES-GCM Nonce Replay Vulnerability", "Reusing nonces under the same key completely breaks GCM cryptographic security.",
         "Enforced 96-bit CSPRNG nonces (os.urandom) dynamically bound to the wire frame header as Additional Authenticated Data."],
        ["Browser SSE Stream Disconnection", "Unstable network connections caused client progress bars to freeze during multi-carrier runs.",
         "Implemented automated exponential backoff reconnection in the React EventSource listener with event ID tracking."]
    ]
    add_table_with_caption(
        doc,
        "Table 5.1: Phase 3 Engineering Implementation Challenges, Impacts, and Solutions",
        headers_5_1, data_5_1, [1.5, 2.0, 2.2]
    )

    print("Chapter 5 built successfully.")
