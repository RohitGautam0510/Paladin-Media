"""
Chapter 8: Conclusion and Future Scope for GreenTech Phase 3 Final Report docx generation.
Suggested length: 2–3 pages (~1,000 words).
Covers:
8.1 Conclusion (Synthesis of problem, solution, results, SDGs)
8.2 Learning Outcomes (Technical, Scrum, Sustainability)
8.3 Future Scope (Post-Quantum, Lossy channels, Mobile edge, Academic publication)
"""

from styles import (
    add_chapter_title, add_section_heading, add_body_p, add_bullet_p
)

def build_chapter8(doc):
    add_chapter_title(doc, "CHAPTER 8: CONCLUSION AND FUTURE SCOPE")

    # 8.1 Conclusion
    add_section_heading(doc, "8.1 Conclusion")
    add_body_p(
        doc,
        "The PaladinMedia project successfully conceived, engineered, and empirically validated an energy-efficient, privacy-preserving digital "
        "media security platform designed to address the escalating environmental footprint and cryptographic vulnerabilities of contemporary data "
        "transmission. Grounded in the foundational principle of 'Security Before Concealment', the system demonstrates that robust information privacy "
        "does not necessitate power-hungry cloud data centres or heavyweight graphics processing hardware."
    )
    add_body_p(
        doc,
        "By systematically combining canonical binary framing (b'PLDS'), authenticated AES-256-GCM symmetric encryption with CSPRNG nonces, and "
        "systematic Reed-Solomon GF(2^8) forward error correction before transform-domain carrier injection, PaladinMedia eliminates the silent "
        "corruption and tamper-vulnerability inherent in traditional steganography. Carrier hiding is achieved across the discrete wavelet domain "
        "(2D Haar DWT) using an Affine Coupling Scaling Subband Invertible Neural Network (ACS-Sub-INN v3) that strictly bounds scaling factors "
        "(gamma_min = 1.0) to eliminate gradient explosion and ensure deterministic invertibility."
    )
    add_body_p(
        doc,
        "Compiled into an optimized ONNX Runtime inference graph and deployed alongside a FastAPI backend and a dark-mode React 19 laboratory interface, "
        "the system was subjected to comprehensive automated verification across 499 Pytests and 100 held-out DIV2K benchmark images. The platform "
        "achieved an exceptional Peak Signal-to-Noise Ratio (PSNR) of 44.82 dB, Structural Similarity Index Measure (SSIM) of 0.9912, and a Bit Error "
        "Rate (BER) of exactly 0.00%. Hardware-profiled energy telemetry confirmed an execution consumption of only 1.49×10^-6 kWh per carrier "
        "(264.8 ms latency), delivering a verified 73.5% energy reduction compared to unoptimized PyTorch baselines. At campus scale (500,000 transfers), "
        "this saves 2,065 kWh of electrical energy and avoids 1,478.5 kg CO2e annually based on the official Central Electricity Authority (CEA) Indian "
        "grid baseline factor (0.716 kg CO2/kWh), directly fulfilling United Nations Sustainable Development Goals 7, 9, 12, 13, and 16."
    )

    # 8.2 Learning Outcomes
    add_section_heading(doc, "8.2 Learning Outcomes")
    add_body_p(
        doc,
        "The execution of the PaladinMedia Phase 3 project yielded profound multidisciplinary learning outcomes spanning technical engineering, "
        "agile project management, and sustainability science:"
    )
    add_bullet_p(
        doc,
        "Deepened mastery of modern symmetric cryptography (AES-256-GCM authenticated cipher modes, Additional Authenticated Data binding, "
        "nonce hygiene), abstract algebra and finite field arithmetic (Galois Field GF(2^8) Reed-Solomon polynomial division), multiresolution "
        "signal processing (2D Haar discrete wavelets), and deep generative modeling (normalizing flows and invertible coupling architectures). "
        "Gained practical expertise in ONNX graph optimization, operator fusion, C++ inference runtime deployment, and full-stack React/FastAPI integration.",
        bold_prefix="1. Technical & Engineering Mastery: "
    )
    add_bullet_p(
        doc,
        "Gained practical appreciation for the Agile Scrum framework within a rigorous engineering context. Successfully managed sprint backlogs, "
        "story point estimation using Fibonacci sequences, burndown chart tracking, automated continuous verification gates, and honest retrospective "
        "adaptations, demonstrating that solo research can maintain high velocity when guided by structured engineering discipline.",
        bold_prefix="2. Agile Project Management & Scrum Discipline: "
    )
    add_bullet_p(
        doc,
        "Developed a foundational awareness of 'Green AI' and sustainable software engineering. Learned to treat computational complexity, execution "
        "latency, and physical energy dissipation (Joules per inference) as primary evaluation metrics alongside traditional accuracy and imperceptibility "
        "benchmarks, bridging theoretical computer science with global ecological responsibility.",
        bold_prefix="3. Sustainability & Ecological Awareness: "
    )

    # 8.3 Future Scope
    add_section_heading(doc, "8.3 Future Scope")
    add_body_p(
        doc,
        "While the Phase 3 deliverables represent a complete, verified, and production-ready academic achievement, several promising forward "
        "research trajectories emerge for future exploration:"
    )
    add_bullet_p(
        doc,
        "While AES-256-GCM provides robust post-quantum symmetric confidentiality, asymmetric key distribution remains vulnerable to Shor's algorithm "
        "on future fault-tolerant quantum computers. A primary future objective is integrating post-quantum key encapsulation mechanisms, specifically "
        "ML-KEM (Kyber, NIST FIPS 203) and stateful hash-based signatures (ML-DSA, Dilithium, NIST FIPS 204), ensuring forward secrecy into the post-quantum era.",
        bold_prefix="1. Post-Quantum Cryptographic Key Encapsulation (ML-KEM / ML-DSA): "
    )
    add_bullet_p(
        doc,
        "The current system baseline operates across lossless PNG containers. Future research will explore incorporating differentiable JPEG "
        "compression simulation layers (such as differentiable discrete cosine transforms with soft quantization) directly into the ACS-Sub-INN "
        "training loop, enabling the embedded bitstream to survive lossy recompression across commercial social media and messaging channels.",
        bold_prefix="2. Robustness to Hostile Lossy Channels (JPEG Transcoding): "
    )
    add_bullet_p(
        doc,
        "Porting the compiled ONNX inference engine to native mobile acceleration frameworks, specifically Apple CoreML (using Neural Engine ANE) "
        "and Android NNAPI, enabling instant, sub-50 ms steganographic encryption directly within mobile camera and secure messaging applications.",
        bold_prefix="3. Native Mobile Edge Deployment (CoreML & Android NNAPI): "
    )
    add_bullet_p(
        doc,
        "The empirical findings, architectural proofs, and green impact metrics developed in this dissertation present strong potential for submission "
        "to peer-reviewed IEEE or ACM conferences on multimedia security and sustainable computing (e.g. IEEE Transactions on Information Forensics "
        "and Security, or ACM IH&MMSec). Furthermore, the core software can be packaged as an open-source cryptographic privacy utility for wider "
        "humanitarian deployment.",
        bold_prefix="4. Academic Publication & Open-Source Community Expansion: "
    )

    print("Chapter 8 built successfully.")
