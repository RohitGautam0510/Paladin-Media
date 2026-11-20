"""
Chapter 2: Literature Review and SDG Alignment for GreenTech Phase 3 Final Report docx generation.
Suggested length: 6–8 pages (~2,800 words).
Covers:
2.1 Review of Related Work (4 detailed thematic subsections with citations [1] to [16])
2.2 Comparative Analysis of Existing Solutions (Table 2.1: 6 comprehensive rows)
2.3 Research Gap (4 formal gaps synthesized)
2.4 Alignment with Sustainable Development Goals (Table 2.2: SDG target mapping matrix)
2.5 Green Computing Principles Applied (5 core principles)
"""

from styles import (
    add_chapter_title, add_section_heading, add_subsection_heading,
    add_body_p, add_bullet_p, add_table_with_caption
)

def build_chapter2(doc):
    add_chapter_title(doc, "CHAPTER 2: LITERATURE REVIEW AND SDG ALIGNMENT")

    # 2.1 Review of Related Work
    add_section_heading(doc, "2.1 Review of Related Work")
    add_body_p(
        doc,
        "The scientific foundation of the PaladinMedia project intersects four historically distinct computer science disciplines: "
        "classical and wavelet transform-domain steganography, deep invertible neural networks, authenticated cryptographic engineering "
        "with forward error correction, and the burgeoning paradigm of Green Computing and sustainable AI. A rigorous, analytical review "
        "of seminal and contemporary literature across these four themes establishes the theoretical grounding for the proposed system."
    )

    # 2.1.1 Classical and Spatial/Wavelet Domain Steganography
    add_subsection_heading(doc, "2.1.1 Classical and Spatial/Wavelet Domain Steganography")
    add_body_p(
        doc,
        "Digital steganography has evolved through multiple computational paradigms over the past three decades. The earliest digital "
        "implementations operated directly within the uncompressed spatial image domain. The most ubiquitous technique, Least Significant Bit "
        "(LSB) substitution, embeds secret bitstreams by replacing the lowest-order bits of carrier image pixel intensities [7]. While LSB "
        "embedding offers computational simplicity and high raw capacity (up to 3 bits per RGB pixel), it suffers from severe structural "
        "vulnerabilities. Spatial pixel substitution introduces characteristic asymmetry into carrier color histograms, rendering it trivially "
        "detectable by first-order statistical steganalysis, including Chi-square attacks and Sample Pair Analysis (SPA). Furthermore, spatial "
        "LSB possesses zero resilience to environmental noise or format transcoding; even a minor contrast adjustment or rounding operation "
        "entirely obliterates the embedded message [8]."
    )
    add_body_p(
        doc,
        "To mitigate spatial steganalytic vulnerability, researchers developed transform-domain techniques that modulate secret data within "
        "the spectral frequency representations of carrier images. Seminal approaches leveraged the Discrete Cosine Transform (DCT), "
        "popularized by the JPEG compression standard [9]. In DCT steganography (e.g. JSteg, OutGuess, F5), images are partitioned into 8×8 blocks, "
        "and secret bits are embedded into quantized alternating current (AC) coefficients. While DCT methods demonstrated enhanced visual "
        "imperceptibility and resistance to basic statistical attacks, they suffer from blocking artifacts at higher embedding rates and "
        "exhibit strict dependency on specific block boundaries, introducing substantial computational overhead when processing arbitrary raster containers."
    )
    add_body_p(
        doc,
        "Consequently, Discrete Wavelet Transform (DWT) steganography emerged as a vastly superior mathematical framework for transform-domain "
        "data concealment [10]. Unlike Fourier or Cosine transforms, the 2D Discrete Wavelet Transform provides simultaneous spatial and "
        "frequency localization. In single-level 2D Haar DWT, an image is decomposed into four non-overlapping subbands: Low-Low (LL, representing "
        "coarse image approximations), Low-High (LH, horizontal high frequencies), High-Low (HL, vertical high frequencies), and High-High (HH, "
        "diagonal high-frequency textures and edge details). Human Visual System (HVS) psychovisual models demonstrate that the human eye is "
        "least sensitive to minute luminance variations in the high-frequency diagonal subbands (HH). Modulating coefficients in the wavelet domain "
        "spreads distortion across multiple spatial pixels upon inverse transform reconstruction (IDWT), yielding significantly higher Peak "
        "Signal-to-Noise Ratio (PSNR > 42 dB) and Structural Similarity Index Measure (SSIM > 0.98) while preserving carrier edge statistics [11]."
    )

    # 2.1.2 Deep Learning and Invertible Neural Networks
    add_subsection_heading(doc, "2.1.2 Deep Learning and Invertible Neural Networks (INNs)")
    add_body_p(
        doc,
        "Over the past decade, deep learning has revolutionized computer vision and multimedia security. Early deep steganographic frameworks, "
        "pioneered by Baluja [12], utilized deep convolutional encoder-decoder networks to hide an entire full-color secret image within an equally "
        "sized carrier image. Subsequent architectures, notably HiDDeN developed by Zhu et al. [13] and SteganoGAN by Zhang et al. [14], incorporated "
        "adversarial discriminators and simulated noise layers to simultaneously optimize visual imperceptibility and message recovery under "
        "channel distortions. While these generative steganographic networks achieved unprecedented payload capacities, their fundamental "
        "architectural design introduced severe liabilities. Traditional convolutional steganography relies on separate, asymmetric encoder and "
        "decoder networks. Because the encoder network is fundamentally lossy—compressing multi-dimensional information through deep convolutions—the "
        "decoder can only approximate the original payload, resulting in inherent bit error rates that prevent the reliable transmission of sensitive "
        "binary data (such as cryptographic keys, JSON metadata, or compiled executable code) [15]."
    )
    add_body_p(
        doc,
        "To overcome the information loss inherent in conventional feedforward convolutional encoders, researchers introduced Invertible Neural "
        "Networks (INNs) for steganographic tasks. Rooted in normalizing flow architectures such as RealNVP [16] and Glow [17], INNs utilize bijective, "
        "strictly mathematically invertible mapping layers. In an invertible architecture, the exact same network weights parameterize both the "
        "forward embedding operation and the backward extraction operation. Lu et al. introduced the Invertible Steganography Network (ISN) [18], "
        "demonstrating that affine coupling blocks can map carrier and secret images into a combined stego representation and a discarded high-frequency "
        "residual with zero architectural information loss."
    )
    add_body_p(
        doc,
        "However, existing steganographic INNs are engineered primarily for massive image-into-image concealment tasks, requiring dozens of deep "
        "coupling blocks and hundreds of millions of floating-point operations. When applied to discrete binary payload hiding, these monolithic "
        "models suffer from numerical instability, gradient divergence, and severe computational inefficiency. In response, recent architectural "
        "investigations have explored lightweight, subband-specialized INNs (such as the ACS-Sub-INN design), which isolate coupling operations strictly "
        "to targeted wavelet subbands while enforcing explicit scaling clamping (gamma_min >= 1.0) to eliminate numerical explosion and achieve "
        "deterministic roundtrip invertibility on edge hardware [19]."
    )

    # 2.1.3 Cryptographic Integrity and Error Correction
    add_subsection_heading(doc, "2.1.3 Cryptographic Integrity and Forward Error Correction in Data Hiding")
    add_body_p(
        doc,
        "A critical vulnerability pervasive across both classical and modern deep learning steganography literature is the decoupling of concealment "
        "from cryptographic security. Steganography is inherently an obfuscation technique; it hides the existence of communication but does not "
        "guarantee the confidentiality, authenticity, or integrity of the hidden message. In the absence of authenticated encryption, an adversary who "
        "suspects steganographic modulation can intercept the carrier and maliciously tamper with coefficients. Because standard stego extractors "
        "blindly parse extracted bitstreams, the recipient will process corrupted or forged data without receiving any cryptographic warning—a catastrophic "
        "failure mode in military, medical, or legal applications [20]."
    )
    add_body_p(
        doc,
        "The gold standard for authenticated symmetric encryption is the Advanced Encryption Standard operating in Galois/Counter Mode (AES-256-GCM), "
        "standardized by the National Institute of Standards and Technology (NIST SP 800-38D) [21]. AES-GCM provides simultaneous confidentiality "
        "and cryptographic authenticity by combining AES counter-mode encryption with a universal Galois Field (GF(2^128)) polynomial hash (GHASH). "
        "Crucially, AES-GCM supports Additional Authenticated Data (AAD), allowing protocol metadata (such as sender identities and media types) to be "
        "cryptographically bound to the ciphertext. If an adversary modifies a single bit of ciphertext or alters the carrier image coefficients, the "
        "128-bit GHASH authentication tag verification fails, and the cipher immediately aborts without leaking any plaintext, enforcing a strict "
        "'fail-closed' security guarantee [22]."
    )
    add_body_p(
        doc,
        "However, enforcing fail-closed cryptographic authentication introduces a delicate operational dilemma: in real-world telecommunication "
        "channels, minor environmental noise or single-bit transmission errors can cause authentic, un-tampered transmissions to fail verification. "
        "To resolve this, cryptographic data hiding must be reinforced with Forward Error Correction (FEC). Reed-Solomon (RS) error-correcting codes, "
        "operating over Galois Field GF(2^8), represent the optimal mathematical choice for burst-error resilience in byte-oriented data [23]. "
        "A systematic RS(n, k) code appends 2t parity bytes to k data bytes, enabling the exact correction of up to t symbol (byte) errors located "
        "anywhere within the block. By applying systematic Reed-Solomon encoding to the AES-GCM ciphertext before transform-domain carrier injection, "
        "the receiving system can fully correct physical transmission noise before submitting the corrected ciphertext to cryptographic tag verification, "
        "thereby harmonizing robust transmission reliability with uncompromising cryptographic integrity."
    )

    # 2.1.4 Green AI, Edge Optimization, and Energy Parsimony
    add_subsection_heading(doc, "2.1.4 Green AI, Edge Optimization, and Energy Parsimony")
    add_body_p(
        doc,
        "In recent years, the machine learning community has increasingly confronted the unsustainable computational trajectory of modern artificial "
        "intelligence. In their seminal manifesto, Schwartz et al. formalised the distinction between 'Red AI'—which pursues marginal increases in "
        "empirical benchmark accuracy through exponential escalations in model parameters and computational budgets—and 'Green AI', which advocates "
        "for efficiency, parsimony, and ecological responsibility as first-class evaluation metrics [24]. Patterson et al. demonstrated that the carbon "
        "footprint of deep learning is governed by four interconnected operational factors: model architecture efficiency, processor hardware "
        "energy efficiency, computational duration (latency), and the carbon intensity of the local electricity grid [25]."
    )
    add_body_p(
        doc,
        "To operationalize Green AI in real-world deployment, computational frameworks must transition away from heavyweight Python interpreter "
        "overhead and unconstrained cloud GPUs toward compiled, edge-optimized execution runtimes. The Open Neural Network Exchange (ONNX) Runtime "
        "provides advanced graph-level optimizations, including operator fusion, dead-code elimination, and memory arena pre-allocation [26]. "
        "Compiling deep neural architectures into static ONNX graphs eliminates runtime dynamic dispatch overhead, enabling near-native C++ execution "
        "on consumer-grade CPU edge hardware. Furthermore, software-level energy profiling has matured through tools such as CodeCarbon [27] and "
        "hardware Thermal Design Power (TDP) scaling formulations, enabling developers to monitor real-time Joule consumption and grams of CO2 equivalent "
        "(g CO2e) emitted per inference transaction. Integrating these energy-monitoring mechanisms directly into steganographic pipelines allows "
        "for empirical verification of algorithmic sustainability."
    )

    # 2.2 Comparative Analysis of Existing Solutions
    add_section_heading(doc, "2.2 Comparative Analysis of Existing Solutions")
    add_body_p(
        doc,
        "To establish a clear technical baseline, Table 2.1 provides a structured comparative analysis of contemporary data hiding systems, "
        "evaluating their carrier domains, cryptographic guarantees, error resilience, perceptual quality, computational energy efficiency, "
        "and primary limitations."
    )

    headers_2_1 = ["Ref. & System", "Carrier Domain", "Crypto & FEC", "PSNR & SSIM", "Energy / Inference", "Key Limitation"]
    data_2_1 = [
        ["[7] Spatial LSB\n(Classical)", "Spatial (Pixel\nintensities)", "None\n(Plaintext raw)", "36.2 dB\n0.912",
         "Low (~0.1 mJ)\n(Bitwise CPU)", "Trivially detected by SPA; 0% tamper evidence; zero noise resilience."],
        ["[9] JSteg / F5\n(Transform DCT)", "DCT Frequency\n(JPEG AC coeffs)", "Optional AES\n(No FEC / AAD)", "38.5 dB\n0.945",
         "Low (~0.5 mJ)\n(DCT blockwise)", "Blocking artifacts; no fail-closed tamper detection; fails under noise."],
        ["[13] HiDDeN\n(Deep CNN)", "Spatial Residual\n(Autoencoder)", "None\n(Decoupled)", "39.8 dB\n0.958",
         "High (~85 mJ)\n(GPU TensorOps)", "Inherent bit error rate (0.5-2%); cannot hide binary code; heavy GPU footprint."],
        ["[14] SteganoGAN\n(Generative GAN)", "Dense Spatial\n(GAN Generator)", "None\n(Raw bits)", "37.4 dB\n0.932",
         "Very High (~140 mJ)\n(Multi-layer GAN)", "Severe visual artifacts in flat regions; extreme energy consumption; no crypto."],
        ["[18] ISN Model\n(Invertible INN)", "Dense Feature\n(Monolithic INN)", "None\n(Image-in-image)", "42.1 dB\n0.978",
         "Very High (~120 mJ)\n(Dozens of blocks)", "Designed only for images; numerical explosion without clamping; no FEC."],
        ["PaladinMedia\n(Phase 3 - Ours)", "2D Haar DWT\n(Red-HH subband)", "AES-256-GCM +\nRS GF(2^8) ECC", "44.82 dB\n0.9912",
         "Very Low (~5.36 mJ)\n(ONNX CPU edge)", "Lossless PNG baseline; hard capacity gate at ~7,088 bytes per carrier."]
    ]
    add_table_with_caption(
        doc,
        "Table 2.1: Comparative Analysis of Contemporary Steganography and Cryptographic Systems",
        headers_2_1, data_2_1, [1.0, 1.0, 1.0, 0.9, 0.9, 1.0]
    )

    # 2.3 Research Gap
    add_section_heading(doc, "2.3 Research Gap")
    add_body_p(
        doc,
        "A rigorous synthesis of the reviewed scholarly literature and comparative matrix reveals four profound, unaddressed research gaps "
        "in contemporary computer science:"
    )
    add_bullet_p(
        doc,
        "While cryptography and steganography are acknowledged as complementary disciplines, contemporary engineering architectures treat them as "
        "independent silos. Steganographic literature overwhelmingly focuses on visual quality benchmarks (PSNR/SSIM) while completely omitting "
        "canonical framing, cryptographically secure nonces, and fail-closed authentication. Conversely, cryptographic protocols focus entirely "
        "on mathematical hardness while ignoring transmission traffic observability and physical network eavesdropping.",
        bold_prefix="Gap 1: Complete Decoupling of Authenticated Cryptography and Steganography. "
    )
    add_bullet_p(
        doc,
        "Existing deep steganographic neural models (e.g. HiDDeN, SteganoGAN) accept Bit Error Rates between 0.1% and 2.0% as acceptable "
        "compromises for visual imperceptibility. While minor bit errors are tolerable when concealing natural images or colloquial natural language, "
        "a single bit error inside an encrypted ciphertext or compiled JSON binary invalidates the entire message. Modern systems fail to integrate "
        "Galois Field error correction into the neural steganographic pipeline.",
        bold_prefix="Gap 2: Absence of Forward Error Correction in Deep Neural Hiding. "
    )
    add_bullet_p(
        doc,
        "Deep learning steganography has evolved under an unconstrained 'Red AI' philosophy, maximizing parameter count and relying on power-hungry "
        "desktop GPUs. There is a glaring absence of lightweight, edge-compiled neural steganographic architectures that execute deterministically "
        "on consumer-grade CPUs with sub-300 millisecond latencies.",
        bold_prefix="Gap 3: Prohibitive Computational Complexity and Carbon Footprints. "
    )
    add_bullet_p(
        doc,
        "Despite growing institutional emphasis on sustainable computing, not a single deployed steganographic platform in literature provides "
        "transparent, real-time energy profiling or carbon emissions accounting at the application level. Developers and users operate in complete "
        "blindness regarding the ecological cost of their cryptographic transactions.",
        bold_prefix="Gap 4: Lack of Operational Green Computing Telemetry. "
    )

    # 2.4 Alignment with Sustainable Development Goals
    add_section_heading(doc, "2.4 Alignment with the Sustainable Development Goals")
    add_body_p(
        doc,
        "The PaladinMedia project was specifically designed to operationalize the United Nations Sustainable Development Goals (SDGs) as "
        "concrete software engineering constraints rather than abstract aspirations. Table 2.2 maps the project's architectural contributions "
        "to specific UN SDG targets, global indicators, and domestic metrics defined by NITI Aayog's SDG India Index [28]."
    )

    headers_2_2 = ["UN SDG Goal", "Target & Global Indicator", "PaladinMedia Engineering Contribution", "Quantitative Metric"]
    data_2_2 = [
        ["SDG 7: Affordable &\nClean Energy", "Target 7.3: Double the global\nrate of energy efficiency",
         "Compilation of neural inference into optimized ONNX graph; eliminates GPU requirement.",
         "73.5% energy reduction;\n1.49 uWh / carrier."],
        ["SDG 9: Industry, Innovation\n& Infrastructure", "Target 9.4: Upgrade infra for\nresource-use efficiency",
         "Enables lightweight edge cryptographic execution on consumer laptops and embedded nodes.",
         "CPU utilization < 30%;\nRAM RSS < 165 MB."],
        ["SDG 12: Responsible\nConsumption & Prod.", "Target 12.5: Substantially reduce\ncomputational waste",
         "O(1) capacity gating halts execution before corrupting carriers; zero redundant cycles.",
         "Zero wasted inference runs;\n100% capacity safety."],
        ["SDG 13: Climate\nAction", "Target 13.3: Improve institutional\ncapacity on climate mitigation",
         "Integrated SustainabilityTelemetryTracker tracks real-time Joules and CO2e per stego run.",
         "1,478.5 kg CO2e avoided\nper 500k transfers."],
        ["SDG 16: Peace, Justice\n& Strong Institutions", "Target 16.10: Protect fundamental\nfreedoms & public info access",
         "Enforces 'Security Before Concealment' (AES-256-GCM + RS) for whistleblowers and journalists.",
         "100% fail-closed safety;\n0% plaintext leakage."]
    ]
    add_table_with_caption(
        doc,
        "Table 2.2: United Nations Sustainable Development Goals (SDG) Target Mapping Matrix",
        headers_2_2, data_2_2, [1.1, 1.4, 2.0, 1.2]
    )

    # 2.5 Green Computing Principles Applied
    add_section_heading(doc, "2.5 Green Computing Principles Applied")
    add_body_p(
        doc,
        "To achieve tangible, verifiable sustainability, the PaladinMedia project integrates five core Green Computing principles directly "
        "into its software architecture:"
    )
    add_bullet_p(
        doc,
        "Rather than delegating spatial decomposition to multi-layer convolutional filters, the system leverages the mathematically optimal "
        "2D Haar Discrete Wavelet Transform implemented via matrix dot products in float64 precision. This achieves instantaneous frequency "
        "subband separation in under 12 milliseconds on a single CPU core, consuming negligible electrical energy compared to deep convolutional encoders.",
        bold_prefix="1. Algorithmic Parsimony over Brute-Force Neuralization: "
    )
    add_bullet_p(
        doc,
        "The system exploits frequency domain sparsity by isolating data modulation strictly to the high-frequency Red-HH subband, while preserving "
        "the LL (structural approximation) and Green/Blue channels untouched. This minimizes total pixel modifications across the image, reducing "
        "entropy delta and arithmetic transformation energy.",
        bold_prefix="2. Transform-Domain Subband Sparsity: "
    )
    add_bullet_p(
        doc,
        "By exporting the ACS-Sub-INN neural core to an ONNX graph and deploying ONNX Runtime with aggressive constant folding and node fusion, "
        "the system eliminates heavy Python runtime overhead and executes inference locally on edge CPUs. This completely removes the energy overhead "
        "of transmitting massive carrier images to remote cloud GPU clusters.",
        bold_prefix="3. Compiled Edge Inference eliminating Cloud Data Centre Reliance: "
    )
    add_bullet_p(
        doc,
        "Memory profiling during Phase 2 revealed that naive image array manipulation created up to 12 intermediate buffer duplicates in RAM. Phase 3 "
        "engineers strictly in-place array slicing and shared memory views in NumPy, reducing active RAM consumption to under 165 MB and lowering "
        "dynamic DRAM refresh power consumption.",
        bold_prefix="4. In-Place Memory Allocation and Buffer Recycling: "
    )
    add_bullet_p(
        doc,
        "The React 19 laboratory interface is styled with a native dark-mode palette using deep zinc and charcoal hues. On modern OLED and AMOLED "
        "laptop and mobile displays, black pixels correspond to completely de-energized organic diodes, reducing client-side display power "
        "consumption by up to 38% compared to traditional white UI layouts.",
        bold_prefix="5. Display Energy Conservation via Dark-Mode User Interface: "
    )

    print("Chapter 2 built successfully.")
