# PaladinMedia (MVP 0 — Phase 2)

**Post-Quantum Media Steganography & Transform-Domain Concealment System**

PaladinMedia investigates secure, reversible media concealment as an integrated systems engineering problem. The product architecture combines cryptographic payload protection, Galois Field error correction, and 2D discrete wavelet transform carrier embedding into a verifiable sender-to-receiver pipeline.

---

## 1. Architecture Overview

The system strictly enforces unidirectional cryptographic protection before carrier embedding, and symmetric receiver recovery:

```text
SECRET PIPELINE (Phase 1 Cryptographic Foundation):
  Secret Media (Image, Text, JSON, Binary)
    ↓
  SecretSerializer (b"PLDS" wire framing, metadata, CRC-32)
    ↓
  CryptoProvider (AES-256-GCM authenticated encryption, 96-bit nonce, 128-bit GHASH tag)
    ↓
  ECCEncoder (Reed-Solomon GF(2^8), systematic block chunking, 2t=32 parity)
    ↓
  Protected Payload Bitstream

CARRIER PIPELINE (Phase 2 Signal Processing + Infrastructure):
  Cover Image (512×512 RGB)
    ↓
  HaarDWT2D (One-level 2D Haar DWT → LL, LH, HL, HH of shape (3, 256, 256))
    ↓
  CapacityCalculator (Hard-gate check: max 8,176 B protected payload in 1 HH channel)
    ↓
  CarrierHeader (16-byte fixed canonical framing: Magic b"PLDH", session, length)
    ↓
  DeterministicMaskGenerator (PRNG permutation from session context)
    ↓
  BaselineEmbedder (Exact integer-parity modulation on 2*HH in Red channel)
    ↓
  Inverse DWT (IDWT)
    ↓
  8-bit RGB Stego PNG (Lossless container)

RECEIVER RECOVERY PIPELINE:
  Stego PNG
    ↓
  HaarDWT2D
    ↓
  Extractor (Fixed header anchor extraction → mask reconstruction → payload bits)
    ↓
  ECCEncoder.decode (Reed-Solomon error correction, repairs up to t=16 errors/block)
    ↓
  CryptoProvider.decrypt (Constant-time GHASH tag verification, fail-closed)
    ↓
  SecretSerializer.deserialize (CRC-32 verification)
    ↓
  Recovered Secret Media (100% exact byte equality verified)
```

---

## 2. Repository Structure

```text
Paladin_media/
├── pyproject.toml                     # Python package metadata and test configuration
├── README.md                          # Project documentation and architecture guide
│
├── docs/                              # Project governance and documentation
│   ├── specifications/                # Authoritative project specifications
│   │   ├── Architecture_Specification.docx
│   │   ├── Product_Vision_Draft.docx
│   │   ├── Sprint-0_Implementation_Readiness_Specification_v0.1.docx
│   │   └── System_and_Research_Document_Draft.docx
│   ├── experiments/                   # Formal experiment ledgers
│   │   └── E03_media_steganography.md
│   ├── testing/                       # Testing methodology specifications
│   │   └── TEST_INFRA.md
│   └── project/                       # Project provenance and dataset manifests
│       ├── ORIGINAL_REQUEST.md
│       └── mvp0_manifest.json
│
├── src/                               # Python backend source code
│   ├── api/                           # FastAPI server and event observability adapter
│   │   ├── event_adapter.py
│   │   ├── server.py
│   │   └── __init__.py
│   ├── carrier/                       # Signal processing and transform-domain steganography
│   │   ├── capacity.py                # Carrier capacity accounting and hard gate
│   │   ├── dwt.py                     # 1-level 2D Haar DWT / IDWT & subband visualization
│   │   ├── embedding.py               # Baseline transform-domain embedder
│   │   ├── extraction.py              # Steganographic carrier extractor
│   │   ├── header.py                  # 16-byte fixed carrier header codec
│   │   ├── mask.py                    # Deterministic non-circular position generator
│   │   └── __init__.py
│   ├── core/                          # Core orchestration and serialization
│   │   ├── exceptions.py              # System exception hierarchy
│   │   ├── media_pipeline.py          # End-to-end media-to-media pipeline
│   │   ├── pipeline.py                # Sprint 1 cryptographic security pipeline
│   │   ├── serialization/             # Secret payload framing and serializer
│   │   │   ├── serializer.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   └── security/                      # Cryptography and error correction
│       ├── crypto/                    # AES-256-GCM provider
│       │   ├── provider.py
│       │   └── __init__.py
│       ├── ecc/                       # Reed-Solomon GF(2^8) codec
│       │   ├── encoder.py
│       │   └── __init__.py
│       └── __init__.py
│
├── frontend/                          # React + TypeScript + Vite Laboratory UI
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── public/                        # Static assets (favicons, icons)
│   └── src/                           # UI components, styles, and backend adapter
│       ├── App.tsx
│       ├── index.css
│       ├── main.tsx
│       └── services/
│           └── backendAdapter.ts
│
├── tests/                             # Comprehensive test suite (99 tests)
│   ├── conftest.py                    # Global fixtures and helpers
│   ├── fixtures/                      # Test vectors, KATs, and small image fixtures
│   │   ├── secret_small.png
│   │   ├── vectors.py
│   │   └── __init__.py
│   ├── unit/                          # Unit tests (T-001..T-004, T-011..T-018)
│   │   ├── test_carrier_capacity.py
│   │   ├── test_carrier_dwt.py
│   │   ├── test_carrier_embedding.py
│   │   ├── test_carrier_header.py
│   │   ├── test_carrier_mask.py
│   │   ├── test_crypto.py
│   │   ├── test_ecc.py
│   │   ├── test_exceptions.py
│   │   ├── test_serialization.py
│   │   └── __init__.py
│   └── integration/                   # Integration tests (T-019, T-020, fail-closed)
│       ├── test_fail_closed.py
│       ├── test_media_pipeline.py
│       ├── test_pipeline.py
│       └── __init__.py
│
└── scripts/                           # Operational and verification scripts
    ├── preflight.py                   # Specification validation and dataset manifest scanner
    ├── verify_sprint1.py              # Standalone programmatic Sprint 1 acceptance script
    └── __init__.py
```

---

## 3. Quickstart & Verification

### Running Automated Test Suite
The repository includes an exhaustive 99-test suite covering unit, boundary, fail-closed security, error-correction, and media roundtrip tests:
```bash
pytest -v
```

### Running Sprint 1 Verification
```bash
python scripts/verify_sprint1.py
```

### Running Dataset Preflight & Specification Validation
```bash
python scripts/preflight.py
```

### Running Backend API & Frontend UI
1. **Start Backend API (Port 8000):**
   ```bash
   python -m src.api.server
   ```
2. **Start Frontend Laboratory (Port 5173):**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Open `http://localhost:5173` to interact with the laboratory interface.
