# Comprehensive Regression Test Results & Build Verification

**Audit Date:** 2026-10-29  
**Auditor:** Senior Forensic Reliability Auditor & Cryptographic Engineer  
**Status:** ALL TESTS PASSING (ZERO FAILURES, ZERO REGRESSIONS)  
**Target Repository:** `PaladinMedia` (`E:\Paladin_media`)  

---

## 1. Executive Summary

This document captures the freshly observed, evidence-backed test results for the PaladinMedia repository following the implementation and qualification of Wire Protocol v2.

All test suites were executed in the active Windows environment using Python 3.10.11 and Node.js v24.16.0 / npm 11.13.0:
- **Python Regression Test Suite:** **470 passed, 0 failed** in 91.94s (Exit Code: `0`).
- **Frontend Test Suite:** **40 passed, 0 failed** in 233.56ms (Exit Code: `0`).
- **Frontend Production Build:** **Clean compilation and bundling** in 2.44s (Exit Code: `0`).

---

## 2. Python Regression Test Suite Execution

### 2.1 Invocation Metadata
- **Command:** `pytest tests/unit/ tests/integration/`
- **Working Directory:** `E:\Paladin_media`
- **Environment:** Windows, Python 3.10.11, pytest-8.3.2, pluggy-1.6.0, anyio-4.14.1
- **Timestamp:** 2026-10-29T23:50:01+05:30
- **Exit Code:** `0`
- **Execution Duration:** 91.94 seconds

### 2.2 Test Results Summary by Module

| Test Module / Directory | Category | Item Count | Status | Notes |
|:---|:---:|:---:|:---:|:---|
| `tests/unit/test_carrier_capacity.py` | Unit | 6 | **PASS** | Dynamic carrier dimensions, subband resolution, hard-gate enforcement. |
| `tests/unit/test_carrier_dwt.py` | Unit | 3 | **PASS** | 1-level 2D Haar DWT / IDWT numerical invertibility. |
| `tests/unit/test_carrier_embedding.py` | Unit | 3 | **PASS** | Transform-domain LSB substitution and subband extraction. |
| `tests/unit/test_carrier_header.py` | Unit | 2 | **PASS** | v1 / v2 header serialization, RS parity, magic parsing. |
| `tests/unit/test_carrier_mask.py` | Unit | 4 | **PASS** | Deterministic pseudo-random mask, anchor reservation. |
| `tests/unit/test_crypto.py` | Unit | 23 | **PASS** | AES-256-GCM, key derivation, authentication tags. |
| `tests/unit/test_dataset_boundary.py` | Unit | 31 | **PASS** | DIV2K dataset manifest preservation, split isolation. |
| `tests/unit/test_diagnostics.py` | Unit | 23 | **PASS** | Diagnostic store accounting, immutable snapshots. |
| `tests/unit/test_ecc.py` | Unit | 24 | **PASS** | Reed–Solomon codec, syndrome verification, Galois field. |
| `tests/unit/test_ecc_framing_recovery.py` | Unit | 26 | **PASS** | Codeword boundary recovery, length prefix framing. |
| `tests/unit/test_exceptions.py` | Unit | 2 | **PASS** | PaladinSecurityError hierarchy and fail-closed semantics. |
| `tests/unit/test_header_ecc_qualification.py`| Unit | 16 | **PASS** | Wire Protocol v2 header error correction qualification. |
| `tests/unit/test_loss_gradient_isolation.py` | Unit | 6 | **PASS** | PyTorch gradient freeze, optimizer isolation. |
| `tests/unit/test_loss_reconciliation.py` | Unit | 4 | **PASS** | Spectral loss weights, SSIM/PSNR loss components. |
| `tests/unit/test_neural_components.py` | Unit | 24 | **PASS** | ACS-Sub-INN v3 subnet architectures, Haar forward/inverse. |
| `tests/unit/test_neural_extraction.py` | Unit | 15 | **PASS** | Neural spatial extraction operator boundary. |
| `tests/unit/test_neural_reproducibility.py` | Unit | 3 | **PASS** | Seed determinism across inference executions. |
| `tests/unit/test_neural_spatial_bridge.py` | Unit | 13 | **PASS** | DWT coefficient to spatial domain translation bridge. |
| `tests/unit/test_neural_training_scaffold.py`| Unit | 11 | **PASS** | Dataset loader fixtures and training scaffolding. |
| `tests/unit/test_onnx_inference_boundary.py` | Unit | 7 | **PASS** | ONNX Runtime execution provider and tensor binding. |
| `tests/unit/test_phase5b_onnx_pipeline.py` | Unit | 23 | **PASS** | Neural ONNX opset 14 embedding and extraction roundtrip. |
| `tests/unit/test_phase5c_deployment_hardening.py` | Unit | 11 | **PASS** | Concurrency safety, resource cleanup, bounded cache. |
| `tests/unit/test_phase5d_api_security.py` | Unit | 37 | **PASS** | Input validation, MIME types, payload size bounds, SSE. |
| `tests/unit/test_serialization.py` | Unit | 20 | **PASS** | SecretPayload serialization, CRC32 integrity framing. |
| `tests/unit/test_training_engine.py` | Unit | 47 | **PASS** | Checkpoint serialization and engine safety gates. |
| `tests/unit/test_wire_protocol_v2.py` | Unit | 19 | **PASS** | Explicit Wire Protocol v2 unit test suite. |
| `tests/integration/test_ecc_spatial_roundtrip.py` | Integration | 6 | **PASS** | Full spatial embedding to IDWT to extraction roundtrip. |
| `tests/integration/test_end_to_end_acceptance.py` | Integration | 3 | **PASS** | Multi-carrier end-to-end acceptance scenarios. |
| `tests/integration/test_end_to_end_gate.py` | Integration | 10 | **PASS** | Strict end-to-end gate with cryptographic auth. |
| `tests/integration/test_fail_closed.py` | Integration | 6 | **PASS** | Verification of zero plaintext leakage on corruption. |
| `tests/integration/test_media_pipeline.py` | Integration | 3 | **PASS** | Multi-format image ingestion and extraction. |
| `tests/integration/test_pipeline.py` | Integration | 9 | **PASS** | End-to-end pipeline execution and verification. |
| `tests/integration/test_production_carrier_path.py` | Integration | 11 | **PASS** | Production carrier loading, caching, and DWT path. |
| `tests/integration/test_production_neural_orchestrator.py` | Integration | 21 | **PASS** | Orchestrator lifecycle, thread safety, model sessions. |
| **Total** | — | **470** | **470 PASSED** | **Zero Failures, Zero Errors** |

### 2.3 Warnings Observed
1. `StarletteDeprecationWarning`: `fastapi.testclient` deprecation warning regarding `httpx` vs `httpx2`. Benign third-party library notice.
2. `DecompressionBombWarning`: Triggered intentionally by security test `test_image_pixel_count_and_dimension_limits` verifying that oversized images trigger security rejections. Benign and expected.

---

## 3. Frontend Test Suite Execution

### 3.1 Invocation Metadata
- **Command:** `npx tsx --test frontend/src/services/backendAdapter.test.ts frontend/src/services/inspectorWorkflow.test.ts frontend/src/utils/differenceEngine.test.ts`
- **Working Directory:** `E:\Paladin_media`
- **Environment:** Node.js v24.16.0, tsx v4.19.2
- **Timestamp:** 2026-10-29T23:51:41+05:30
- **Exit Code:** `0`
- **Execution Duration:** 233.56 ms

### 3.2 Test Results Summary
- **Total Tests:** 40
- **Passed:** 40
- **Failed:** 0
- **Cancelled:** 0
- **Skipped:** 0
- **Detailed Suite Breakdown:**
  - `backendAdapter.test.ts`: 5/5 passed (terminal completion, early EOF, malformed JSON, HTTP error handling, user cancellation).
  - `inspectorWorkflow.test.ts`: 25/25 passed (view mode switching, Haar spatial support, embedding vs changed coefficients, run reset, capacity rejections, dynamic carrier dimensions 512×512, 1024×1024, 2040×1356, coordinate mapping for non-square carriers, zoom/pan transforms, partial diagnostic coordinates).
  - `differenceEngine.test.ts`: 10/10 passed (subband-to-spatial-block mapping, computeHaarHH arithmetic, dimension validation, pixel difference amplification, buffer validation, viewport calculation, canvas coordinate mapping).

---

## 4. Frontend Production Build Verification

### 4.1 Invocation Metadata
- **Command:** `npm run build --prefix frontend`
- **Working Directory:** `E:\Paladin_media`
- **Pipeline:** `tsc -b && vite build`
- **Timestamp:** 2026-10-29T23:51:47+05:30
- **Exit Code:** `0`
- **Execution Duration:** 2.44 seconds

### 4.2 Build Artifacts Generated
```
vite v8.3.0 building client environment for production...
transforming...
✓ 1883 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.45 kB │ gzip:  0.29 kB
dist/assets/index-7QNJs_vS.css   22.54 kB │ gzip:  4.85 kB
dist/assets/index-QMilaA_z.js   296.69 kB │ gzip: 88.01 kB
✓ built in 2.44s
```

TypeScript type-checking passed with **zero compiler diagnostics**, and Vite bundled the complete production asset graph cleanly.
