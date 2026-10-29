# PaladinMedia MVP 0 Sprint 1: Test Infrastructure & Methodology Specification

**Document ID**: PDM-S1-TEST-INFRA-001
**Author**: Test Writer Subagent (`test_writer_1`)
**Target Milestone**: MVP 0 Sprint 1 (Cryptographic Foundation & Payload)
**Status**: ACTIVE TEST SUITE BASELINE

---

## 1. Overview & Architectural Testing Principles

This document specifies the opaque-box test infrastructure, test fixture definitions, error simulation models, and verification methodology for **PaladinMedia MVP 0 Sprint 1: Cryptographic Foundation & Payload**.

The test suite validates the core unidirectional sender transformation:
$$\text{Secret Media} \xrightarrow{\text{SecretSerializer}} \text{Framed Bytes} \xrightarrow{\text{CryptoProvider (AES-GCM)}} \text{Protected Bitstream} \xrightarrow{\text{ECCEncoder (Reed-Solomon)}} \text{RS Codewords}$$
and its symmetric receiver reversal:
$$\text{Channel Codewords} \xrightarrow{\text{ECC Decode}} \text{Protected Bitstream} \xrightarrow{\text{AES-GCM Decrypt}} \text{Framed Bytes} \xrightarrow{\text{SecretSerializer}} \text{Recovered Secret}$$

### Core Testing Mandates:
1. **Opaque-Box Design**: Tests interact exclusively through declared public API contracts in `src/core/serialization`, `src/security/crypto`, `src/security/ecc`, and `src/core/pipeline`.
2. **Authoritative Expected Outputs**: Tests verify outputs against official NIST SP 800-38D test vectors, canonical wire format byte layouts, and mathematical Galois Field $GF(2^8)$ code bounds.
3. **Fail-Closed Security Guarantee (INV-03)**: Any ciphertext bit flip, tag mutation, AAD discrepancy, or key mismatch must result in immediate exception raising (`AuthenticationError`) and leak zero unauthenticated plaintext.
4. **Symbol-Level ECC Evaluation (INV-06)**: Error correction capabilities are evaluated per $GF(2^8)$ byte symbols, proving resilience up to $t = \lfloor 2t/2 \rfloor = 16$ symbol errors per 255-byte block, burst error tolerance, and uncorrectable error detection.

---

## 2. 4-Tier Test Coverage Framework

The test suite is structured around a rigorous 4-tier testing methodology:

```
+-------------------------------------------------------------------------------+
| PALADINMEDIA 4-TIER SPRINT 1 TEST ARCHITECTURE                                |
+-------------------------------------------------------------------------------+
| Tier 1: Feature Coverage (Happy Path & Core Protocols)                       |
|   - Canonical wire format headers, magic bytes, versions, CRC-32 integrity   |
|   - Deterministic serialization & compact sorted-key JSON metadata           |
|   - AES-256-GCM encryption/decryption with 32B key, 12B nonce, 16B tag       |
|   - Official NIST SP 800-38D Known Answer Test (KAT) vectors                 |
|   - Reed-Solomon GF(2^8) chunking, length prefix framing, systematic parity   |
|   - Clean channel end-to-end pipeline recovery                               |
+-------------------------------------------------------------------------------+
| Tier 2: Boundary & Corner Cases                                               |
|   - 0-byte (empty) payloads across all layers                                 |
|   - 1-byte minimal payloads                                                   |
|   - Exact RS chunk boundaries: k-1 (222B), k (223B), k+1 (224B)               |
|   - Usable single sub-band steganographic capacity (~7,088 bytes)             |
|   - Multi-block payloads spanning dozens of RS blocks (10,000 bytes, 64KB)    |
|   - Complex metadata (Unicode, CJK, emojis, nested objects, None vs {})       |
+-------------------------------------------------------------------------------+
| Tier 3: Pairwise Combinations & Negative / Adversarial Security              |
|   - Tampered magic bytes, unsupported versions, truncated buffers             |
|   - Single-bit flips in ciphertext (head, middle, tail)                       |
|   - Single-bit flips in authentication tag (head, tail)                       |
|   - Tampered AAD, missing expected AAD, extraneous AAD                        |
|   - Incorrect 32-byte key rejection                                           |
|   - Strict proof of zero plaintext leakage across all failure modes           |
|   - Uncorrectable symbol errors: t+1 (17 errors) and t+5 (21 errors) per block|
+-------------------------------------------------------------------------------+
| Tier 4: Real-World Workloads & Integration Robustness                         |
|   - Realistic military/intelligence text directives                           |
|   - Synthetic 256-byte image pixel data                                       |
|   - Real-world structured JSON telemetry & mission payloads                   |
|   - Correctable channel symbol noise injection across single and multi-block  |
|   - Full pipeline fail-closed under post-ECC adversarial tampering            |
+-------------------------------------------------------------------------------+
```

---

## 3. Test Directory Layout & Artifact Inventory

```
tests/
├── __init__.py                     # Package marker
├── conftest.py                     # Global fixtures, sample keys, dynamic loaders
├── fixtures/
│   ├── __init__.py                 # Fixture exports
│   └── vectors.py                  # NIST KAT vectors, representative payloads, noise injectors
├── unit/
│   ├── __init__.py                 # Unit test marker
│   ├── test_exceptions.py          # Exception hierarchy inheritance & instantiation tests
│   ├── test_serialization.py       # T-001: SecretSerializer unit tests
│   ├── test_crypto.py              # T-002, T-003: AESGCMProvider unit tests
│   └── test_ecc.py                 # T-004: ReedSolomonEncoder unit tests
└── integration/
    ├── __init__.py                 # Integration test marker
    ├── test_pipeline.py            # End-to-end roundtrip & correctable noise tests
    └── test_fail_closed.py         # INV-03 fail-closed security & zero leakage tests
```

---

## 4. Test Execution & Automation Commands

### Full Test Suite Run
```bash
pytest -v
```

### Run Unit Tests Only
```bash
pytest tests/unit -v
```

### Run Integration Tests Only
```bash
pytest tests/integration -v
```

### Run Specific Test Modules
```bash
# Serialization tests (T-001)
pytest tests/unit/test_serialization.py -v

# Cryptographic provider tests (T-002, T-003)
pytest tests/unit/test_crypto.py -v

# Reed-Solomon error correction tests (T-004)
pytest tests/unit/test_ecc.py -v

# Pipeline integration tests
pytest tests/integration/test_pipeline.py -v

# Fail-closed security tests
pytest tests/integration/test_fail_closed.py -v
```

### Run with Coverage Reporting
```bash
pytest --cov=src --cov-report=term-missing tests/
```

---

## 5. Error Injection & Simulation Helpers

Module: `tests/fixtures/vectors.py`

| Helper Function | Signature | Purpose |
|---|---|---|
| `flip_bit` | `(data: bytes, byte_pos: int, bit_pos: int) -> bytes` | Inverts exactly one bit at a targeted byte and bit index. |
| `flip_random_bit` | `(data: bytes, rng=None) -> (bytes, int, int)` | Selects a random bit and inverts it; returns position metadata. |
| `corrupt_byte` | `(data: bytes, byte_pos: int, new_val=None) -> bytes` | Mutates an entire byte symbol to a guaranteed different value. |
| `inject_symbol_errors` | `(data, count, block_offset, block_length, seed) -> (bytes, list)` | Injects exactly `count` distinct symbol corruptions within an RS block. |
| `inject_burst_error` | `(data: bytes, start_byte: int, burst_byte_len: int) -> bytes` | Inverts all 8 bits across `burst_byte_len` consecutive bytes. |
| `truncate_data` | `(data: bytes, num_bytes: int) -> bytes` | Cuts `num_bytes` from buffer end to test truncation detection. |
| `append_garbage` | `(data: bytes, garbage: bytes) -> bytes` | Appends trailing rogue bytes to verify length-check strictness. |

---

## 6. Traceability to Requirements & Tickets

| Ticket / Scenario | Module Tested | Target Test Module | Test Coverage Summary |
|---|---|---|---|
| **PDM-S1-005** / **T-001** | `SecretSerializer` | `tests/unit/test_serialization.py` | Canonical framing, deterministic JSON metadata, 0B/1B/~7KB boundaries, CRC-32 integrity, corrupt magic/version/truncation rejection. |
| **PDM-S1-002** / **T-002** | `AESGCMProvider` | `tests/unit/test_crypto.py` | AES-256-GCM encryption/decryption, packaging format, NIST KAT vectors, nonce uniqueness across runs, AAD integrity binding. |
| **PDM-S1-003** / **T-003** | `AESGCMProvider` | `tests/unit/test_crypto.py` | Fail-closed security: single-bit ciphertext/tag flips, tampered AAD, wrong key, truncated payload; zero plaintext leakage proof. |
| **PDM-S1-004** / **T-004** | `ReedSolomonEncoder` | `tests/unit/test_ecc.py` | Systematic chunking over GF(2^8), exact symbol correction up to $t=16$, burst error tolerance (128 bits), uncorrectable error ($t+1$) rejection. |
| **PDM-S1-006** / **T-004+** | End-to-End Pipeline | `tests/integration/test_pipeline.py` | Full roundtrip: Serialize -> Encrypt -> ECC Encode -> Channel Noise -> ECC Decode -> Decrypt -> Deserialize. 100% exact recovery. |
| **INV-03 Fail-Closed** | Full Pipeline | `tests/integration/test_fail_closed.py` | Uncorrectable noise halts at ECC; post-ECC tampering halts at Crypto; zero plaintext returned across all failure scenarios. |
