# PaladinMedia

**Transform-Domain Wavelet & Neural Invertible Media Steganography System**
*Authenticated AES-256-GCM Encryption, Reed-Solomon Error Correction, and Invertible Neural Concealment*

[![Test Suite](https://img.shields.io/badge/Tests-409%20Passed-brightgreen.svg)](#3-quickstart--verification)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Author](https://img.shields.io/badge/Author-RohitGautam0510-informational.svg)](https://github.com/RohitGautam0510)

PaladinMedia investigates secure, verifiable, and reversible media concealment as an integrated systems engineering problem. The architecture combines authenticated cryptographic payload protection (AES-256-GCM), Galois Field error correction (Reed-Solomon GF(2^8)), 2D discrete wavelet transforms (DWT), and an Invertible Neural Network (ACS-Sub-INN) into a reproducible sender-to-receiver pipeline.

---

## 1. System Architecture

The pipeline enforces unidirectional cryptographic and error-correction protection prior to carrier modulation, paired with symmetric fail-closed receiver recovery:

```text
SECRET PIPELINE (Phase 1 Cryptographic Foundation):
  Secret Media (Image, Text, JSON, Raw Binary)
    ↓
  SecretSerializer (b"PLDS" wire framing, metadata, CRC-32 integrity)
    ↓
  CryptoProvider (AES-256-GCM authenticated encryption, 96-bit nonce, 128-bit GHASH tag)
    ↓
  ECCEncoder (Reed-Solomon GF(2^8), systematic block chunking: qualified 2t=48 / legacy 2t=32)
    ↓
  Protected Payload Bitstream

CARRIER PIPELINE (Phase 2 Transform Domain + Phase 3/4/5 Neural Concealment):
  Cover Image (512×512, 1024×1024, or arbitrary even-dimension RGB)
    ↓
  HaarDWT2D (One-level 2D Haar DWT → LL, LH, HL, HH subbands)
    ↓
  CapacityCalculator (Hard capacity gate: pre-embedding allocation check)
    ↓
  CarrierHeader (16-byte fixed canonical framing: Magic b"PLDH", session ID, payload length, flags)
    ↓
  DeterministicMaskGenerator (PRNG permutation from session context)
    ↓
  Carrier Modulator:
    ├── Default Pipeline: Classical exact integer-parity modulation on Red 2*HH
    └── Qualified Deployment: Native ONNX Runtime spatial modulation (neural_spatial_embedding_opset14.onnx)
    ↓
  Inverse Haar DWT (IDWT)
    ↓
  8-bit RGB Stego PNG (Lossless container)

RECEIVER RECOVERY PIPELINE:
  Stego PNG
    ↓
  HaarDWT2D
    ↓
  Carrier Extractor (Fixed anchor extraction → mask reconstruction → protected bits)
    ↓
  ECCEncoder.decode (Reed-Solomon decoding: repairs up to t=24 errors/block in qualified deployment, t=16 in legacy)
    ↓
  CryptoProvider.decrypt (Constant-time GHASH tag verification, fail-closed on tamper)
    ↓
  SecretSerializer.deserialize (CRC-32 verification)
    ↓
  Recovered Secret Media (100% exact byte equality verified)
```

> [!NOTE]
> **Modulation Engine & ECC Profile Selection**:
> - Instantiating `MediaStegoPipeline()` directly without parameters defaults to the classical integer-parity modulation engine and the legacy Reed-Solomon $2t=32$ configuration ($t=16$ error correction).
> - The production FastAPI server initializes via `MediaStegoPipeline.create_qualified_deployment(backend="onnx")`. This selects native ONNX Runtime spatial models (`neural_spatial_embedding_opset14.onnx` and `neural_spatial_extraction_opset14.onnx`) and the qualified Reed-Solomon $2t=48$ profile ($t=24$ error correction capability).
> - Wire-level headers dynamically signal error correction profiles (`flags=0x01` for $2t=48$), allowing receiver instances to adapt automatically regardless of constructor defaults.
> - **Cryptographic Primitives**: Payload secrecy and tamper-resistance are provided by symmetric AES-256-GCM with 256-bit keys and Reed-Solomon error correction. The active runtime does not implement post-quantum asymmetric key exchange (such as ML-KEM/Kyber) or post-quantum signatures (such as ML-DSA/Dilithium).


---

## 2. Implementation Status & Phase Roadmap

| Phase | Milestone | Status | Key Deliverables & Validation |
| :--- | :--- | :---: | :--- |
| **Phase 1** | Cryptographic & ECC Core | **Complete** | AES-256-GCM, Reed-Solomon GF(2^8), NIST SP 800-38D KAT vectors, fail-closed security. |
| **Phase 2** | Transform-Domain Wavelet Engine | **Complete** | 2D Haar DWT/IDWT, Red-HH parity modulation, capacity gating, E03–E14 empirical ledgers. |
| **Phase 3** | ACS-Sub-INN Neural Architecture | **In progress** | Neural components and supporting tests are present; further training, integration and validation remain. |
| **Phase 4** | DIV2K Model Training | **Validation ongoing** | Training artifacts and telemetry are present; broader model validation and end-to-end integration remain. |
| **Phase 5** | ONNX Export & Hardening | **Validation ongoing** | ONNX model artifacts and API hardening work are present; deployment validation and INT8 quantisation work remain. |
| **UI** | Laboratory Web Interface | **Integration ongoing** | React/TypeScript interface and inspection components are present; end-to-end integration and refinement remain. |

---

## 3. Quickstart & Verification

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ & npm (for frontend laboratory UI)

### 3.1 Installation
```bash
# Clone repository
git clone https://github.com/RohitGautam0510/Paladin-Media.git
cd Paladin-Media

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install package in editable mode with development and test dependencies (including pytest)
pip install -e ".[dev]"
```

### 3.2 Running the Automated Test Suite
The repository includes an exhaustive test suite covering unit, integration, boundary, fail-closed security, Reed-Solomon error repair, and ONNX inference boundary tests:
```bash
pytest -v
```
- **Out-of-the-box (Synthetic Fixtures):** 409 tests pass immediately using embedded test fixtures and qualified ONNX models; 26 dataset-dependent tests cleanly skip with clear instructions.
- **With DIV2K Acquired:** Once the DIV2K dataset is downloaded into `dataset/DIV2K/` per [Section 4](#4-dataset-acquisition), all 435 tests execute and pass.

### 3.3 Running Programmatic Sprint Verification
To execute the standalone programmatic verification script (24 architectural checks):
```bash
python scripts/verify_sprint1.py
```

### 3.4 Launching Backend API & Laboratory UI
1. **Start Backend API (FastAPI on Port 8000):**
   ```bash
   python -m src.api.server
   ```
2. **Start Frontend Laboratory (Vite on Port 5173):**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser.

---

## 4. Dataset Acquisition

PaladinMedia relies on high-resolution carrier imagery for training and benchmarking. In accordance with open-source hygiene and data licensing, large image datasets are **not distributed directly** in Git history.

### 4.1 DIV2K High-Resolution Dataset
The project evaluates against the [DIV2K Dataset (CVPR 2017)](https://data.vision.ee.ethz.ch/cvl/DIV2K/):
1. Download `DIV2K_train_HR.zip` from the official [ETH Zurich Computer Vision Lab repository](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
2. Extract the 800 high-resolution `.png` images into:
   ```text
   dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/
   ```
3. Verify the canonical partition manifest against the dataset:
   ```bash
   python scripts/generate_div2k_split.py
   ```
   *The frozen manifest at [`docs/project/div2k_hr_split.json`](docs/project/div2k_hr_split.json) cryptographically verifies SHA-256 hashes and dimensions for 600 training, 100 validation, and 100 test images.*

### 4.2 MVP 0 Baseline Carriers
Exploratory carriers used in baseline experiments E03–E10 are located under `dataset/MVP 0 Dataset/`. Validate carrier integrity:
```bash
python scripts/preflight.py
```
*(The preflight script inspects `dataset/MVP 0 Dataset/` by default and accepts an optional custom directory path via command-line argument: `python scripts/preflight.py <path-to-dataset>`.)*

---

## 5. Supported Inputs & Operational Boundaries

### 5.1 Supported Carrier Specifications
- **Format:** 8-bit RGB lossless PNG.
- **Dimensions:** Any even spatial dimensions ($W, H \ge 64$, such that $W \bmod 2 = 0$ and $H \bmod 2 = 0$). Standard benchmark resolutions include $512\times512$ and $1024\times1024$.
- **Subband Allocation:** Modulation targets the high-frequency diagonal subband ($HH$) in the Red channel, using Green and Blue $HH$ subbands as spatial-domain context.

### 5.2 Supported Secret Payloads
- **Payload Types:** Plaintext (`TEXT`), structured JSON (`JSON`), raster images (`IMAGE`), acoustic audio (`AUDIO`), or arbitrary binary byte streams (`RAW_BYTES`).
- **Capacity Floor:**
  - $512\times512$ Carrier: Up to ~7,088 bytes of secret data (~8,176 bytes protected Reed-Solomon stream).
  - $1024\times1024$ Carrier: Up to ~28,498 bytes of secret data (~32,752 bytes protected stream).
- **Strict Capacity Gate:** Payloads exceeding carrier capacity are rejected fail-closed with `PayloadCapacityExceededError` before carrier coefficients are modified.

### 5.3 Observed Boundaries & Scientific Non-Claims
- **Lossless Channel Boundary:** The classical wavelet baseline is validated over lossless 8-bit RGB PNG containers. Under lossy JPEG compression ($Q \in [70, 100]$) or additive Gaussian noise ($\sigma \ge 0.5$), classical parity modulation fails completely (documented in [`docs/experiments/E14_lossy_channel_degradation.md`](docs/experiments/E14_lossy_channel_degradation.md)).
- **Neural Channel Resilience:** The Phase 3/4/5 ACS-Sub-INN model introduces learned context prediction to mitigate high-frequency artifacts, but requires lossless or noise-bounded channels for full byte recovery.
- **Academic Research Scope:** PaladinMedia is an experimental academic research system; it makes no claims of operational military certification or unconditional steganographic undetectability.

---

## 6. Repository Structure

```text
Paladin_media/
├── pyproject.toml                         # Python package configuration & dependencies
├── README.md                              # Primary project documentation & quickstart
├── .gitignore                             # Clean Git publication hygiene rules
│
├── src/                                   # Production backend packages
│   ├── api/                               # FastAPI application, diagnostics, and SSE adapter
│   ├── carrier/                           # Haar DWT/IDWT, embedding, extraction, capacity gate
│   ├── core/                              # End-to-end orchestration, serialization, exception hierarchy
│   ├── neural/                            # ACS-Sub-INN architecture, dataset loaders, ONNX export
│   └── security/                          # AES-256-GCM crypto provider, Reed-Solomon GF(2^8) ECC
│
├── frontend/                              # Vite + React 19 Laboratory UI
│   ├── src/components/                    # DWT subband inspector, modal views, terminal viewer
│   ├── src/services/                      # Backend SSE streaming adapter and workflow logic
│   └── package.json
│
├── tests/                                 # Comprehensive automated test suite (435 tests)
│   ├── fixtures/                          # Test vectors, NIST KATs, sample images
│   ├── integration/                       # End-to-end media roundtrip & security tests
│   └── unit/                              # Component unit tests (carrier, crypto, ecc, neural, onnx)
│
├── scripts/                               # Operational and verification tools
│   ├── experiments/                       # Experiment execution runners (E03–E14, Phase 4/5)
│   ├── generate_div2k_split.py            # DIV2K split generator & checksum validator
│   ├── preflight.py                       # Specification and carrier validator
│   └── verify_sprint1.py                  # Standalone 24-check acceptance script
│
├── checkpoints/                           # Model artifacts
│   ├── phase5a_onnx/                      # Exported ONNX inference models (opset 14)
│   │   ├── neural_spatial_*.onnx          # Active production runtime models (spatial domain)
│   │   └── acs_sub_inn_*.onnx             # Transform-domain evaluation and testing assets
│   └── phase4c_extended_30ep/             # Canonical trained PyTorch best model & visual fixture
│       ├── best_model.pt                  # Canonical 30-epoch PyTorch weights for reference & export
│       └── e2e_stego_carrier.png          # Visual verification stego carrier fixture
│
├── docs/                                  # Project governance & scientific ledgers
│   ├── specifications/                    # Authoritative system specifications
│   ├── experiments/                       # Formal experiment reports (E03–E14) & JSON records
│   ├── audits/                            # Gate audits, entry audits, security checks
│   └── project/                           # Dataset split manifests and provenance records
│
└── dataset/                               # Local/external carriers (git-ignored)
    ├── DIV2K/
    └── MVP 0 Dataset/
```

---

## 7. Citation & Project Governance

This codebase is maintained by [RohitGautam0510](https://github.com/RohitGautam0510) as part of academic research on verifiable media steganographic systems.

For questions, issues, or reproducible experiment replication, please open a GitHub Issue.
