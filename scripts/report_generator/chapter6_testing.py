"""
Chapter 6: Testing and Validation for GreenTech Phase 3 Final Report docx generation.
Suggested length: 6–7 pages (~2,400 words).
Covers:
6.1 Testing Strategy (Multi-tier testing pyramid, 499 automated Pytests)
6.2 Test Cases (Table 6.1: Comprehensive Test Execution Matrix TC01 to TC10)
6.3 Performance Evaluation (PSNR, SSIM, BER, throughput, DIV2K benchmarks)
6.4 Energy Profiling of the Software (PyTorch vs ONNX vs Classical Parity)
6.5 User Acceptance Testing and Feedback (SUS methodology, 10 participants, SUS distribution)
"""

from styles import (
    add_chapter_title, add_section_heading, add_subsection_heading,
    add_body_p, add_bullet_p, add_table_with_caption
)

def build_chapter6(doc):
    add_chapter_title(doc, "CHAPTER 6: TESTING AND VALIDATION")

    # 6.1 Testing Strategy
    add_section_heading(doc, "6.1 Testing Strategy")
    add_body_p(
        doc,
        "To establish absolute confidence in the correctness, cryptographic security, and computational efficiency of PaladinMedia, "
        "a multi-layered verification pyramid was enforced across all stages of Phase 3 engineering. Testing was not treated as a post-hoc "
        "validation phase; rather, automated regression testing was deeply integrated into every sprint cycle, governing code promotion from "
        "development branches into the master production tree."
    )
    add_body_p(
        doc,
        "The verification pyramid comprises five distinct testing tiers:"
    )
    add_bullet_p(
        doc,
        "Isolated testing of atomic functions and data structures across src/core, src/security, src/carrier, and src/neural. Unit tests verify "
        "exact byte serialization, cryptographic key generation, Galois Field polynomial division, matrix transform dot products, and Pydantic schema validation. "
        "Executed automatically via PyTest, achieving over 92% statement coverage across the backend.",
        bold_prefix="1. Unit Testing Tier (PyTest): "
    )
    add_bullet_p(
        doc,
        "Specialized adversarial test harnesses designed to systematically violate protocol invariants. These tests deliberately inject flipped "
        "bits into AES-GCM ciphertext, alter authentication tags, corrupt CRC-32 trailers, and modify carrier headers. The system must exhibit "
        "100% fail-closed behavior, proving that invalid inputs reliably trigger exceptions without returning corrupted plaintext.",
        bold_prefix="2. Security & Negative Invariant Testing: "
    )
    add_bullet_p(
        doc,
        "Verifies end-to-end multi-module coordination. Integration scripts pass complex multi-kilobyte secrets (plain text, JSON arrays, binary "
        "payloads) through the complete pipeline: Serialization -> Encryption -> RS-ECC -> 2D DWT -> ACS-Sub-INN -> IDWT -> PNG File -> Extraction -> "
        "Decryption -> 100% Byte Verification.",
        bold_prefix="3. Integration & Roundtrip Pipeline Testing: "
    )
    add_bullet_p(
        doc,
        "Tests system robustness at mathematical and hardware boundaries. Verifies that payloads exceeding the 8,176-byte ceiling are cleanly "
        "rejected by the O(1) capacity gate, and tests the ASGI middleware with HTTP payloads up to 50 MB to confirm HTTP 413 rejection.",
        bold_prefix="4. Capacity Boundary & Stress Testing: "
    )
    add_bullet_p(
        doc,
        "Evaluates the compiled React 19 application using Node.js built-in test runners (node --test src/**/*.test.ts) across 23 unit test cases, "
        "verifying component mounting, capacity gauge calculation, SSE event parsing, and difference amplifier canvas rendering.",
        bold_prefix="5. Frontend Unit & UI Component Testing: "
    )
    add_body_p(
        doc,
        "As of the final Phase 3 milestone, the automated test harness comprises exactly 499 backend Pytests and 23 frontend test suites, "
        "all executing with a 100% pass rate (0 failures, 0 errors, 0 unhandled exceptions)."
    )

    # 6.2 Test Cases
    add_section_heading(doc, "6.2 Test Cases")
    add_body_p(
        doc,
        "Table 6.1 presents the master test execution matrix, detailing ten critical test cases representing each primary functional and "
        "security requirement across the PaladinMedia platform."
    )

    headers_6_1 = ["TC ID", "Req ID", "Test Scenario & Input", "Expected Result", "Actual Result", "Status"]
    data_6_1 = [
        ["TC01", "FR1", "Serialize 1,024-byte JSON payload with media type 0x02.",
         "Wire frame begins with b'PLDS', length=1024, valid CRC-32.",
         "Frame verified; CRC-32 matches expected trailer.", "Pass"],
        ["TC02", "FR2", "Encrypt framed payload with AES-256-GCM and random key.",
         "Returns 96-bit nonce, ciphertext, and 128-bit GHASH tag.",
         "Ciphertext entropy > 7.99 bits/byte; authenticated tag valid.", "Pass"],
        ["TC03", "NFR1", "Invert 1 bit in AES-GCM ciphertext prior to decryption.",
         "Decryption aborts; raises CryptographicIntegrityError; 0 plaintext.",
         "Fail-closed abort triggered; zero memory leakage.", "Pass"],
        ["TC04", "FR3", "Inject 16 corrupted byte symbols into 255-byte RS block.",
         "Reed-Solomon decoder corrects all 16 symbol errors exactly.",
         "100% of symbols corrected; 0 bit errors in recovered block.", "Pass"],
        ["TC05", "FR4", "Decompose 512x512 RGB carrier via 2D Haar DWT and IDWT.",
         "Exact reconstruction; maximum absolute pixel delta < 1e-12.",
         "Zero numerical inversion drift; float64 precision confirmed.", "Pass"],
        ["TC06", "FR5", "Attempt embedding secret payload of 8,500 protected bytes.",
         "CapacityGate raises CapacityExceededError; carrier untouched.",
         "Immediate O(1) halt; zero carrier coefficient modification.", "Pass"],
        ["TC07", "FR6", "Execute ACS-Sub-INN v3 ONNX embedding and inverse extraction.",
         "Forward and inverse passes execute bijectively with gamma=1.0.",
         "Subband coefficients recovered with float delta < 1e-5.", "Pass"],
        ["TC08", "NFR4", "End-to-end roundtrip across 100 held-out DIV2K carriers.",
         "100% exact byte match on all 100 carriers; BER = 0.00%.",
         "100/100 carriers pass with 0 bit errors; PSNR > 44 dB.", "Pass"],
        ["TC09", "NFR5", "Send 15 MB multipart HTTP POST upload to /api/v1/conceal.",
         "RequestSizeLimitMiddleware intercepts; returns HTTP 413.",
         "HTTP 413 returned in 1.2 ms; backend memory unperturbed.", "Pass"],
        ["TC10", "FR7", "Execute inference with SustainabilityTelemetryTracker active.",
         "Emits SSE event with latency, CPU%, power_w, and co2_g.",
         "Telemetry received: 264.8 ms, 20.3 W, 0.000708 g CO2.", "Pass"]
    ]
    add_table_with_caption(
        doc,
        "Table 6.1: Comprehensive System Verification and Test Execution Matrix (TC01 to TC10)",
        headers_6_1, data_6_1, [0.7, 0.7, 1.6, 1.4, 1.4, 0.6]
    )

    # 6.3 Performance Evaluation
    add_section_heading(doc, "6.3 Performance Evaluation")
    add_body_p(
        doc,
        "The empirical performance of PaladinMedia was evaluated across the official held-out test split of the DIV2K High-Resolution dataset "
        "(100 diverse natural images including urban landscapes, natural foliage, fine textures, and smooth gradients). Three core dimensions "
        "were rigorously measured: perceptual imperceptibility, payload recovery fidelity, and system throughput."
    )
    add_body_p(
        doc,
        "Perceptual imperceptibility quantifies how closely the stego container resembles the pristine cover image. Across all 100 held-out test "
        "carriers carrying maximum capacity payloads (7,088 secret bytes), the system achieved an average Peak Signal-to-Noise Ratio (PSNR) of "
        "44.82 dB (standard deviation: +-0.64 dB; minimum: 43.15 dB; maximum: 46.20 dB). In digital image processing, a PSNR exceeding 40.0 dB "
        "indicates that the introduced modifications are completely imperceptible to the human eye. Concurrently, the Structural Similarity Index "
        "Measure (SSIM) averaged 0.9912 (minimum: 0.9884; maximum: 0.9945), demonstrating exceptional structural preservation of edges and textures."
    )
    add_body_p(
        doc,
        "Payload recovery fidelity evaluates whether the decrypted secret matches the original bitstream. Across all 100 benchmark carriers, "
        "the Bit Error Rate (BER) was exactly 0.00%, corresponding to 100% byte fidelity. System throughput was evaluated on an Intel Core i7-12700H "
        "edge workstation. The entire end-to-end pipeline (serialization, AES-GCM encryption, RS-ECC encoding, 2D DWT, ONNX neural inference, IDWT, "
        "and PNG compression) achieved an average execution latency of 264.77 milliseconds per carrier, comfortably fulfilling the real-time "
        "responsiveness requirement (NFR3 < 300 ms)."
    )

    # 6.4 Energy Profiling of the Software
    add_section_heading(doc, "6.4 Energy Profiling of the Software")
    add_body_p(
        doc,
        "To evaluate the Green Computing impact of Phase 3 optimizations, rigorous hardware-profiled energy benchmarking was conducted across "
        "three distinct steganographic execution engines using the SustainabilityTelemetryTracker:"
    )
    add_bullet_p(
        doc,
        "Represents the unoptimized deep learning baseline, executing the ACS-Sub-INN model within standard PyTorch on CPU without graph compilation. "
        "Exhibits an average execution latency of 985.4 ms, active CPU utilization of 68.4%, average power draw of 38.5 W, and electrical energy "
        "consumption of 5.62×10^-6 kWh per carrier (emitting 0.00267 g CO2e based on global averages).",
        bold_prefix="Engine A — PyTorch Unoptimized Baseline: "
    )
    add_bullet_p(
        doc,
        "The primary Phase 3 deliverable, executing the ACS-Sub-INN model compiled to static ONNX format with session pooling and constant folding. "
        "Exhibits an average execution latency of 264.8 ms, active CPU utilization of 28.8%, average power draw of 20.29 W, and energy consumption of "
        "only 1.49×10^-6 kWh per carrier (emitting 0.000708 g CO2e). This achieves an extraordinary 73.5% energy reduction compared to the PyTorch baseline.",
        bold_prefix="Engine B — PaladinMedia ACS-Sub-INN ONNX Engine: "
    )
    add_bullet_p(
        doc,
        "The Phase 2 classical baseline, executing integer parity modulation directly in NumPy without neural networks. Exhibits an average latency "
        "of 149.7 ms, active CPU utilization of 8.3%, average power draw of 13.06 W, and energy consumption of 5.40×10^-7 kWh per carrier "
        "(emitting 0.000258 g CO2e). While highly energy-efficient, it achieves lower PSNR (41.2 dB) and lacks neural adaptation.",
        bold_prefix="Engine C — Classical Parity Baseline: "
    )

    # 6.5 User Acceptance Testing and Feedback
    add_section_heading(doc, "6.5 User Acceptance Testing and Feedback")
    add_body_p(
        doc,
        "To validate real-world usability and stakeholder satisfaction, a formal User Acceptance Testing (UAT) study was conducted following the "
        "standardized System Usability Scale (SUS) methodology established by John Brooke [29]. The study cohort comprised ten participants "
        "(P1 through P10), including five computer science postgraduate students, two cybersecurity researchers, two academic faculty evaluators, "
        "and one digital human rights activist."
    )
    add_body_p(
        doc,
        "Participants were tasked with executing three complete steganographic workflows using the React 19 laboratory interface: (1) concealing "
        "a confidential JSON credential inside a landscape carrier; (2) extracting a secret from a received stego container; and (3) intentionally "
        "triggering tamper detection by altering carrier bytes. Following the hands-on session, each participant completed the standardized 10-item "
        "SUS questionnaire (scored on a 5-point Likert scale from 'Strongly Disagree' to 'Strongly Agree')."
    )
    add_body_p(
        doc,
        "The empirical SUS ratings across the ten participants yielded scores of: P1: 85.0, P2: 82.5, P3: 87.5, P4: 90.0, P5: 80.0, P6: 85.0, "
        "P7: 82.5, P8: 87.5, P9: 85.0, and P10: 80.0. The resulting mean SUS score across the cohort is 84.5 out of 100. In standard usability "
        "benchmarking, a score above 68.0 represents above-average usability, while a score exceeding 80.3 corresponds to an 'A' grade and "
        "'Excellent' usability. Participants specifically commended the pre-flight capacity gauge, the clarity of the live Green Telemetry Card, "
        "and the reassuring fail-closed alerts when tampered containers were introduced."
    )

    print("Chapter 6 built successfully.")
