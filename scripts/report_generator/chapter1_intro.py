"""
Chapter 1: Introduction for GreenTech Phase 3 Final Report docx generation.
Suggested length: 5–6 pages (~2,000 words).
Covers:
1.1 Background (environmental/sustainability context, energy waste, data centers, CEA grid intensity)
1.2 Problem Statement (entropy detection, cryptographic fragility, compute waste of deep generative stego)
1.3 Motivation (computational parsimony, Security Before Concealment, Christ University GreenTech mandate)
1.4 Objectives (4 detailed, measurable objectives)
1.5 Scope and Limitations (carrier dimensions, lossless container, capacity bounds, non-claims)
1.6 Summary of Phase 1 and Phase 2 (narrative and Table 1.1)
1.7 Organisation of the Report
"""

from styles import (
    add_chapter_title, add_section_heading, add_subsection_heading,
    add_body_p, add_bullet_p, add_table_with_caption
)

def build_chapter1(doc):
    add_chapter_title(doc, "CHAPTER 1: INTRODUCTION")

    # 1.1 Background
    add_section_heading(doc, "1.1 Background")
    add_body_p(
        doc,
        "The twenty-first century has witnessed an exponential proliferation of digital data generation, transmission, "
        "and computational storage. Driven by pervasive cloud architectures, hyper-connected Internet of Things (IoT) ecosystems, "
        "and widespread reliance on high-definition multimedia services, the global volume of digital information transferred across "
        "telecommunication networks expands at compound annual rates exceeding 25% [1]. However, this unprecedented digital expansion "
        "imposes a profound and escalating environmental toll. Modern hyper-scale data centres, edge switching nodes, and global "
        "routing backbones consume immense volumes of electrical energy. According to the International Energy Agency (IEA), global data "
        "centre electricity consumption reached approximately 460 terawatt-hours (TWh) in 2022—representing nearly 2% of total worldwide "
        "electricity demand—and is projected to double, surpassing 1,000 TWh by 2026, a magnitude comparable to the entire national electricity "
        "consumption of Germany or Japan [2]."
    )
    add_body_p(
        doc,
        "In developing nations characterized by aggressive digital adoption such as India, the carbon footprint associated with digital "
        "infrastructure is particularly severe. Driven by flagship initiatives including 'Digital India' and national broadband penetration, "
        "India's domestic data centre capacity is expanding rapidly, with major metropolitan clusters experiencing over 30% annual growth. "
        "Crucially, the electrical grid that powers this infrastructure remains predominantly carbon-intensive. Official data published "
        "by the Central Electricity Authority (CEA) of the Ministry of Power, Government of India, reveals a national average grid "
        "emission factor of 0.716 kg CO2 per kilowatt-hour (kg CO2/kWh) [3]. Consequently, every megawatt-hour of electrical power wasted "
        "on inefficient computational algorithms, redundant data encodings, or unoptimized neural inferences directly amplifies national "
        "greenhouse gas emissions, accelerating global climate change and conflicting with India's Nationally Determined Contributions (NDCs) "
        "under the Paris Agreement."
    )
    add_body_p(
        doc,
        "Within this broader computational context, cybersecurity and information privacy represent an increasingly compute-intensive "
        "domain. Conventional secure transmission pipelines rely on continuous cryptographic handshakes, high-overhead Transport Layer "
        "Security (TLS) tunnels, and large-scale public key operations. Furthermore, to evade targeted network eavesdropping and state-level "
        "surveillance, privacy-conscious individuals and organizations frequently turn to digital steganography—the art and science of "
        "concealing confidential information within innocuous carrier media, such as digital raster images, audio streams, or video files [4]. "
        "However, contemporary digital steganography has reached a critical technological and ecological crossroads. The recent emergence of "
        "deep learning-based steganography platforms (such as deep autoencoders, generative adversarial networks, and diffusion models) has "
        "dramatically escalated the computational complexity of media hiding. Training and running unoptimized neural models on high-wattage "
        "graphics processing units (GPUs drawing 250 W to 450 W per node) incurs an untenable carbon footprint, rendering these solutions "
        "wholly unviable for battery-constrained mobile phones, edge IoT sensors, and environmentally conscious edge gateways [5]."
    )
    add_body_p(
        doc,
        "Consequently, there is an urgent scientific imperative to develop 'Green AI' cryptographic and steganographic systems that enforce "
        "rigorous privacy preservation, mathematical error resilience, and verified tamper detection while simultaneously minimising "
        "computational complexity, runtime latency, and electrical energy consumption. This paradigm—harmonizing algorithmic security with "
        "ecological parsimony—forms the core foundational pillar of the PaladinMedia project."
    )

    # 1.2 Problem Statement
    add_section_heading(doc, "1.2 Problem Statement")
    add_body_p(
        doc,
        "Modern information privacy systems face a fundamental, mutually exacerbating trilemma: cryptographic observability, "
        "steganographic fragility, and computational energy waste. First, conventional encryption protocols (such as AES-256 or RSA) "
        "produce high-entropy, pseudorandom ciphertext. While mathematically secure against brute-force cryptanalysis, this high-entropy "
        "bitstream stands out starkly against standard network traffic, immediately alerting automated Deep Packet Inspection (DPI) "
        "firewalls, corporate network monitors, and repressive surveillance state apparatuses that a secret exchange is occurring. In hostile "
        "environments, the mere detection of encrypted traffic can lead to immediate connection throttling, packet dropping, or physical "
        "interrogation of whistleblowers, journalists, and human rights defenders."
    )
    add_body_p(
        doc,
        "Second, traditional steganographic approaches that attempt to solve this observability problem by hiding data in digital images "
        "are severely flawed. The vast majority of classical spatial domain algorithms (such as Least Significant Bit substitution) and "
        "transform domain methods (such as standard Discrete Cosine Transform manipulation) decouple concealment from cryptography. They "
        "embed raw or weakly obscured plaintext directly into carrier pixels without authenticated encryption or forward error correction. "
        "As a result, if even a few carrier bits are corrupted during transmission or compression, the recovered payload suffers catastrophic "
        "corruption. More critically, traditional steganography lacks tamper-evidence: an active network adversary can subtly manipulate carrier "
        "coefficients, causing the receiver to parse forged or malicious commands without triggering any cryptographic alarm [6]."
    )
    add_body_p(
        doc,
        "Third, recent academic attempts to overcome these limitations using deep neural steganography have introduced severe environmental "
        "costs. State-of-the-art neural hiding architectures employ multi-million parameter convolutional networks and iterative optimization "
        "loops that demand gigabytes of device memory and hundreds of milliseconds of high-power tensor operations per image. These systems "
        "are rarely compiled or optimized for edge deployment, generating excessive Joule consumption per concealed bit. In summary, contemporary "
        "digital media security lacks an integrated, energy-efficient solution that guarantees cryptographic authentication, forward error "
        "correction, high perceptual imperceptibility, and verified Green Computing telemetry on constrained hardware."
    )

    # 1.3 Motivation
    add_section_heading(doc, "1.3 Motivation")
    add_body_p(
        doc,
        "The motivation behind this project arises from both a technological necessity and an ethical commitment to environmental sustainability. "
        "Technologically, privacy is recognized as a fundamental human right under Article 12 of the Universal Declaration of Human Rights and "
        "Article 21 of the Constitution of India. In an era of pervasive automated surveillance, providing citizens, investigative journalists, "
        "and medical professionals with mathematically unobservable communication tools is essential for maintaining democratic integrity and "
        "protecting vulnerable stakeholders."
    )
    add_body_p(
        doc,
        "Crucially, the project challenges the prevailing industry assumption that stronger cryptographic and neural security must inevitably "
        "require exponentially larger computational footprints. Under the guiding philosophy of 'Security Before Concealment', this research "
        "posits that algorithmic ingenuity—specifically the mathematical synergy of discrete wavelet transforms, Galois Field error correction, "
        "and bijective invertible neural networks—can achieve superior privacy and error resilience while drastically reducing computational "
        "cycles. By executing lightweight inference on local CPU edge hardware rather than dispatching massive payloads across carbon-heavy "
        "cloud data centres, PaladinMedia demonstrates that digital privacy and climate action are fundamentally complementary."
    )
    add_body_p(
        doc,
        "Furthermore, this project is motivated by the academic mission of CHRIST (Deemed to be University), Delhi NCR Campus, which actively "
        "promotes research addressing the United Nations Sustainable Development Goals (SDGs). By embedding transparent energy and carbon "
        "telemetry directly into the operational software stack, PaladinMedia provides an educational benchmark for green software engineering "
        "and responsible technological stewardship in modern computer science education."
    )

    # 1.4 Objectives
    add_section_heading(doc, "1.4 Objectives")
    add_body_p(
        doc,
        "To systematically address the stated research problem, the PaladinMedia project establishes four primary, measurable objectives "
        "governing Phase 3 implementation and evaluation:"
    )
    add_bullet_p(
        doc,
        "To architect and implement an authenticated cryptographic and forward error-correction pipeline that encapsulates arbitrary secret "
        "payloads into canonical binary frames (b'PLDS'), encrypts them using AES-256-GCM with cryptographically secure random nonces, and "
        "fortifies them with systematic Reed-Solomon GF(2^8) error correction, guaranteeing zero plaintext leakage upon tampering and 100% "
        "recovery under transmission channel noise.",
        bold_prefix="Objective 1 (Cryptographic & Integrity Foundation): "
    )
    add_bullet_p(
        doc,
        "To engineer a transform-domain steganographic carrier pipeline that utilizes single-level 2D Haar Discrete Wavelet Transforms (DWT) "
        "in conjunction with an Affine Coupling Scaling Subband Invertible Neural Network (ACS-Sub-INN v3) to modulate secret bitstreams into "
        "the high-frequency subbands of 512×512 RGB carrier images, achieving superior visual imperceptibility (PSNR > 42 dB, SSIM > 0.98) "
        "with strictly lossless container recovery.",
        bold_prefix="Objective 2 (Wavelet & Invertible Neural Embedding): "
    )
    add_bullet_p(
        doc,
        "To formulate and validate a hardware-profiled Green Computing telemetry model that empirically tracks runtime latency, CPU active "
        "utilization, and dynamic RAM energy density, achieving a measured reduction in electrical energy consumption of at least 70% per "
        "embedding inference through Open Neural Network Exchange (ONNX) graph compilation compared to unoptimized PyTorch execution.",
        bold_prefix="Objective 3 (Green Computing & Energy Optimization): "
    )
    add_bullet_p(
        doc,
        "To deliver a verified, full-stack operational platform comprising an asynchronous FastAPI backend and an interactive React 19 / Vite "
        "dark-mode laboratory interface, validated through a comprehensive automated test harness (>450 Pytests, 100% passing) and structured "
        "User Acceptance Testing achieving a System Usability Scale (SUS) score exceeding 80 points.",
        bold_prefix="Objective 4 (Full-Stack Deployment & Verification): "
    )

    # 1.5 Scope and Limitations
    add_section_heading(doc, "1.5 Scope and Limitations")
    add_body_p(
        doc,
        "A rigorous engineering project must clearly delineate its operational boundaries and explicit constraints to ensure scientific validity:"
    )
    add_bullet_p(
        doc,
        "The project encompasses the end-to-end software pipeline spanning secret serialization, AES-256-GCM cryptography, Reed-Solomon coding, "
        "2D Haar DWT frequency decomposition, ACS-Sub-INN v3 neural embedding/extraction, REST API service delivery, and interactive web visualization. "
        "The primary carrier media is standardized on 512×512 pixel 24-bit RGB raster images stored in lossless Portable Network Graphics (PNG) format.",
        bold_prefix="System Scope: "
    )
    add_bullet_p(
        doc,
        "The maximum usable protected payload is bounded by the available coefficient capacity of the targeted high-frequency wavelet subband. "
        "For a 512×512 carrier offering 65,536 raw embedding positions in the Red-HH subband, reserving 128 positions (16 bytes) for the canonical "
        "carrier header leaves a hard capacity ceiling of 8,176 protected bytes, accommodating approximately 7,088 bytes of arbitrary user secret "
        "after cryptographic headers, authentication tags, and Reed-Solomon parity overheads.",
        bold_prefix="Capacity Scope: "
    )
    add_bullet_p(
        doc,
        "The current Phase 3 production baseline assumes lossless image transport (e.g. uncompressed cloud storage, encrypted messaging file attachments, "
        "or secure document portals). It does not claim survival against aggressive lossy recompression (such as low-quality JPEG transcoding at Q < 75) "
        "or destructive geometric attacks (such as image cropping and non-uniform resizing), which represent planned forward research in Phase 4.",
        bold_prefix="Channel Limitations: "
    )
    add_bullet_p(
        doc,
        "PaladinMedia is an academic research and engineering prototype developed within the CHRIST University SDG GreenTech curriculum. It makes "
        "no claims of military-grade certification (e.g. FIPS 140-3 operational accreditation) or immunity against advanced persistent hardware-level "
        "side-channel attacks.",
        bold_prefix="Non-Claims & Academic Boundary: "
    )

    # 1.6 Summary of Phase 1 and Phase 2
    add_section_heading(doc, "1.6 Summary of Phase 1 and Phase 2")
    add_body_p(
        doc,
        "The Phase 3 project represents the culmination of a three-phase engineering trajectory. To contextualize the Phase 3 deliverables, "
        "the foundational achievements of earlier project phases are summarized below without duplicating prior documentation:"
    )
    add_body_p(
        doc,
        "Phase 1 focused on problem formulation, architectural specification, and cryptographic foundation engineering. During Phase 1, "
        "the team established the canonical 'Security Before Concealment' doctrine and engineered the complete cryptographic pipeline. This included "
        "developing the SecretSerializer to construct deterministic binary wire frames containing magic bytes (b'PLDS'), schema versioning, "
        "media type indicators, and CRC-32 trailers. The CryptoProvider was implemented using AES-256-GCM authenticated cipher with 96-bit CSPRNG "
        "nonces and 128-bit GHASH authentication tags, guaranteeing that any tampering with ciphertext causes immediate fail-closed rejection. "
        "Finally, the ECCEncoder was integrated using Reed-Solomon GF(2^8) chunking with 2t=32 parity symbols per 255-byte block, enabling the "
        "system to correct up to 16 byte symbol errors (128 contiguous flipped bits) per block."
    )
    add_body_p(
        doc,
        "Phase 2 advanced the system into signal processing and carrier prototyping. A custom 2D Haar Discrete Wavelet Transform engine was "
        "implemented in float64 precision, successfully decomposing 512×512 RGB images into LL, LH, HL, and HH subbands with zero numerical inversion "
        "loss. Phase 2 engineered a classical integer parity modulation baseline on the Red-HH subband, established the canonical carrier header "
        "(b'PLDH'), implemented deterministic coordinate masking via session PRNG seeds, and formulated the O(1) capacity accounting hard gate "
        "that instantly rejects oversized secrets before corrupting carrier image coefficients. While functional, the Phase 2 prototype relied on "
        "classical spatial pixel adjustments and unoptimized PyTorch tensor operations that exhibited noticeable latency and lacked dedicated "
        "Green Computing telemetry. These shortcomings motivated the comprehensive neural, architectural, and sustainability advancements of Phase 3."
    )

    # Table 1.1: Summary of project phases
    headers_1_1 = ["Phase", "Focus & Core Milestone", "Key Technical Outcomes", "Status"]
    data_1_1 = [
        ["Phase 1", "Problem identification, literature survey, cryptographic & ECC foundation",
         "SecretSerializer (b'PLDS'), AES-256-GCM cipher, Reed-Solomon GF(2^8) ECC, 24 unit tests.", "Completed"],
        ["Phase 2", "Signal processing, wavelet decomposition, classical carrier prototype",
         "2D Haar DWT float64 engine, carrier header (b'PLDH'), parity modulation, hard capacity gate.", "Completed"],
        ["Phase 3", "Invertible neural core, ONNX optimization, full-stack UI, Green AI telemetry",
         "ACS-Sub-INN v3 network, ONNX Runtime session pooling (-73.5% energy), FastAPI, React 19 UI, 499 tests.", "Completed"]
    ]
    add_table_with_caption(
        doc,
        "Table 1.1: Summary of PaladinMedia Project Engineering Phases (Phases 1 to 3)",
        headers_1_1, data_1_1, [1.0, 1.8, 2.1, 0.8]
    )

    # 1.7 Organisation of the Report
    add_section_heading(doc, "1.7 Organisation of the Report")
    add_body_p(
        doc,
        "The remainder of this dissertation is organized into eight structured chapters and supplementary technical appendices:"
    )
    add_bullet_p(
        doc,
        "Chapter 2 reviews related scholarly literature across classical steganography, deep invertible neural networks, cryptographic integrity, "
        "and Green AI paradigms. It establishes the comparative analysis against existing systems, defines the research gap, and maps the project "
        "to the United Nations Sustainable Development Goals.",
        bold_prefix="Chapter 2 (Literature Review & SDG Alignment): "
    )
    add_bullet_p(
        doc,
        "Chapter 3 establishes formal stakeholder analysis, functional and non-functional requirements categorized using the MoSCoW framework, "
        "hardware/software specifications, feasibility analysis, and Scrum-based project management detailing team roles, product backlogs, "
        "sprint velocity, and sprint retrospectives.",
        bold_prefix="Chapter 3 (Requirement Analysis & Project Management): "
    )
    add_bullet_p(
        doc,
        "Chapter 4 delineates the comprehensive system architecture, data flow diagrams (Level 0 context and Level 1 subsystems), Unified "
        "Modeling Language (UML) diagrams, wire protocol binary schemas, mathematical formulations of 2D Haar DWT and ACS-Sub-INN v3 coupling "
        "blocks, dark-mode UI design, and architectural sustainability principles.",
        bold_prefix="Chapter 4 (System Design): "
    )
    add_bullet_p(
        doc,
        "Chapter 5 details the core Phase 3 engineering implementation, explaining module-by-module development across cryptographic, neural, "
        "application, and telemetry engines. It includes four compliant code listings (≤15 lines each), deployment topologies, user interface "
        "collages, architectural changes from Phase 2, and implementation challenge resolutions.",
        bold_prefix="Chapter 5 (Implementation — Phase 3): "
    )
    add_bullet_p(
        doc,
        "Chapter 6 presents the verification pyramid spanning unit, integration, and security negative testing. It details the comprehensive "
        "test execution matrix across 499 automated Pytests, empirical visual imperceptibility metrics, comparative energy profiling, and User "
        "Acceptance Testing evaluating System Usability Scale (SUS) survey outcomes.",
        bold_prefix="Chapter 6 (Testing & Validation): "
    )
    add_bullet_p(
        doc,
        "Chapter 7 delivers the quantitative green impact assessment, formulating dynamic power models, calculating CO2e avoided against the "
        "official Central Electricity Authority (CEA) Indian grid emission factor, evaluating solution lifecycle footprints, and examining "
        "socio-economic impacts, comparative benchmarks, and threats to validity.",
        bold_prefix="Chapter 7 (Results, Green Impact Assessment & Discussion): "
    )
    add_bullet_p(
        doc,
        "Chapter 8 synthesizes research conclusions against initial objectives, summarizes multidisciplinary technical and sustainability "
        "learning outcomes, and delineates a structured roadmap for future post-quantum, channel-robust, and mobile edge research.",
        bold_prefix="Chapter 8 (Conclusion & Future Scope): "
    )
    add_bullet_p(
        doc,
        "The report concludes with IEEE-style scholarly References and Appendices A through E containing Scrum artifacts, source code repository "
        "guidelines, an operational user manual, survey questionnaires, and deployment photographs.",
        bold_prefix="References & Appendices: "
    )

    print("Chapter 1 built successfully.")
