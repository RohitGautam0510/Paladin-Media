# PaladinMedia — Phase 5C Deployment Hardening Report
## Qualified Deployment Configuration and Operational Safeguards

**Repository:** `E:\Paladin_media`
**Phase:** 5 — Deployment Optimization
**Sprint:** Phase 5C — Qualified Deployment Configuration and Operational Safeguards
**Baseline Artifacts:**
- Accepted Phase 4C Extended Checkpoint: `checkpoints/phase4c_extended_30ep/best_model.pt`
- Accepted Phase 5A ONNX Opset 14 Graphs: `checkpoints/phase5a_onnx/*.onnx`
- Canonical Dataset Manifest: `docs/project/div2k_hr_split.json`
- Historical Phase 5B Evidence: `docs/experiments/phase5b_results.json`
**Evaluation Script:** `scripts/experiments/run_phase5c_assessment.py`
**Evaluation Artifact:** `docs/experiments/phase5c_results.json`
**Unit Test Suite:** `tests/unit/test_phase5c_deployment_hardening.py` (11 focused tests; **370/370 full suite passing in 44.99s**)
**Date:** 2026-10-24

---

## 1. Executive Summary and Final Verdict

Phase 5C hardens the deployment path of PaladinMedia by establishing the qualified Phase 5B configuration (`rs_2t_48` Reed–Solomon error correction backed by native ONNX Runtime inference sessions) as the default operational entry point, while enforcing strict carrier format safeguards and fail-closed security invariants:

1. **Operational Entry-Point Integration:** Connected the real operational application server (`src/api/server.py` and `src/api/event_adapter.py`) directly to `MediaStegoPipeline.create_qualified_deployment()`. Both streaming observability (`/api/stream`) and direct operational processing (`/api/embed`, `/api/extract`) execute with ONNX Runtime and `rs_2t_48` (RS(255, 207), $t=24$).
2. **Observable Configuration Surface:** Implemented `/api/config` and `get_deployment_configuration()`, exposing non-sensitive operational facts (inference backend, ECC profile, parity symbol count, carrier format policy, ONNX model verification status, and capacity ceilings) without ever leaking cryptographic keys, nonces, or secret plaintext.
3. **Carrier Format Hardening & Rejection:** The sender explicitly outputs lossless 8-bit RGB PNG carriers and annotates the stego result with format policies. The receiver validates carrier containers and explicitly rejects identifiable JPEG inputs with an actionable `UnsupportedCarrierFormatError`, warning that JPEG-transcoded carriers are not qualified for deployment. Unspecified or corrupted carriers fail closed via `CarrierHeaderError` without guessing format causes.
4. **Capacity Accounting Transparency:** Operational gates transparently separate raw application secret capacity ($M=0$: 6,578 B for `rs_2t_48`) from reference metadata capacity ($M=2$: 6,576 B for `rs_2t_48`). Oversized payloads are strictly rejected with `PayloadCapacityExceededError` before carrier pixel arrays are allocated or modified.
5. **Preserved Invariants:** Zero optimizer updates, zero retraining, zero modifications to ONNX artifacts, and 100% backward compatibility for legacy callers using `MediaStegoPipeline()` (which retains classical spatial parity modulation and `rs_2t_32`).

### Dynamically Computed Gate Statuses

| Gate | Scope | Status | Measured Empirical Finding |
| :--- | :--- | :---: | :--- |
| **G1: Preflight Integrity** | Model digest, ONNX files, manifest hashes | **`PASS`** | Checkpoint parameter digest `ec11dce7...` verified invariant. All 4 ONNX models intact with verified SHA-256 hashes. Canonical 600/100/100 DIV2K split preserved. |
| **G2: Operational Entry Point** | `server.py`, `event_adapter.py`, factories | **`PASS`** | Server and event adapter use `MediaStegoPipeline.create_qualified_deployment()` (`backend="onnx"`, `rs_2t_48`). Legacy constructor `MediaStegoPipeline()` retains `rs_2t_32` and `ParityModulationEngine`. Configuration observable via `/api/config`. |
| **G3: Carrier Format Hardening** | PNG acceptance, JPEG carrier rejection | **`PASS`** | Lossless 8-bit RGB PNG embeds and extracts with 100% exact recovery. Identifiable JPEG carriers explicitly rejected with `UnsupportedCarrierFormatError`. Corrupted headers fail closed without guessing format causes. |
| **G4: Capacity Hard-Gate Enforcement** | Ceilings, metadata overhead, preflight checks | **`PASS`** | Exact 6,578 B raw ceiling verified with 100% recovery; payloads exceeding limit (6,579 B) rejected with `PayloadCapacityExceededError` before carrier modification. Carrier pixels remain bit-for-bit identical. |
| **Overall Sprint Verdict** | Deployment Hardening & Operational Safeguards | **`PASS_WITH_LIMITATIONS`** | All critical deployment, security, and capacity requirements pass. Operating envelope remains bounded: lossy JPEG transcoded carriers and severe channel noise ($\sigma \ge 1.0$) remain unsupported and fail closed. |

---

## 2. Workstream A: Operational Entry-Point Integration

### 2.1 Operational Architecture

The repository's application entry point is the FastAPI service located in `src/api/server.py` and its streaming adapter `src/api/event_adapter.py`.

Prior to Phase 5C, `MediaEventAdapter` directly instantiated classical `SecurityPipeline()` (defaulting to `rs_2t_32`) and manually called classical `BaselineEmbedder` and `Extractor`. In Phase 5C:
- `src/api/server.py` pre-initializes `qualified_pipeline` using `MediaStegoPipeline.create_qualified_deployment(backend="onnx", onnx_dir=ONNX_DIR)`.
- `MediaEventAdapter` was updated to accept `pipeline: Optional[MediaStegoPipeline] = None`, defaulting to `MediaStegoPipeline.create_qualified_deployment()`.
- Both `/api/stream`, `/api/embed`, and `/api/extract` operate via the qualified pipeline.
- Session pooling is preserved: the initialized `InferenceSession` instances for embedding and extraction are reused across requests without re-allocating ONNX runtimes.

### 2.2 Observable Configuration Surface (`/api/config`)

Clients can query operational deployment facts via `GET /api/config` or programmatically via `get_deployment_configuration()`:
```json
{
  "status": "OPERATIONAL",
  "inference_backend": "onnx",
  "ecc_profile": "rs_2t_48",
  "ecc_parity_symbols": 48,
  "carrier_format_policy": "Lossless 8-bit RGB PNG only. Lossy transcoded carriers (e.g. JPEG) are strictly unsupported.",
  "carrier_dimensions": "512x512",
  "raw_secret_capacity_bytes": 6578,
  "reference_metadata_capacity_bytes": 6576,
  "max_protected_payload_bytes": 8176,
  "is_qualified_deployment": true,
  "onnx_models_verified": true,
  "onnx_models": {
    "acs_sub_inn_embedding_opset14.onnx": true,
    "acs_sub_inn_extraction_opset14.onnx": true,
    "neural_spatial_embedding_opset14.onnx": true,
    "neural_spatial_extraction_opset14.onnx": true
  },
  "sessions_initialized": true,
  "operational_qualification_note": "Primary qualified deployment profile rs_2t_48 with native ONNX Runtime backend. Reliable 100% exact recovery across 100B-5000B lossless payloads and robust up to Gaussian noise sigma<=0.5."
}
```

### 2.3 Legacy Compatibility Preservation

To ensure zero breakage of legacy Phase 2 workflows and tests:
- Calling `MediaStegoPipeline()` without arguments continues to instantiate `SecurityPipeline(ecc_encoder=ReedSolomonEncoder(parity_symbols=32))` and `ParityModulationEngine`.
- Calling `MediaEventAdapter(pipeline=MediaStegoPipeline())` retains legacy classical execution.
- Wire-level framing compatibility is preserved: `CarrierHeader.flags` dynamically communicates the profile (`0x00` = `rs_2t_32`, `0x01` = `rs_2t_48`, `0x02` = `rs_2t_64`). A receiver instantiated with `MediaStegoPipeline()` or `MediaStegoPipeline(backend="onnx")` inspects byte 5 of the header and dynamically configures a matching `ReedSolomonEncoder(parity_symbols=header.ecc_parity)`.

---

## 3. Workstream B: Carrier Format Hardening

### 3.1 Sender Format Guarantees

The sender produces strictly lossless 8-bit RGB PNG images. `MediaStegoResult` explicitly specifies:
- `carrier_format = "PNG"`
- `output_format_policy = "Lossless 8-bit RGB PNG only. Do not transcode or compress to JPEG."`

Downstream systems receive explicit metadata indicating that conversion or lossy compression of the stego PNG container destroys the high-frequency transform-domain modulation and causes fail-closed authentication rejection.

### 3.2 Receiver Carrier Format Validation & Error Handling

In `MediaStegoPipeline.extract_and_unprotect()`:
1. **Identifiable JPEG Input:**
   If `stego_carrier` is provided as bytes starting with JPEG SOI magic (`b"\xff\xd8\xff"`), a PIL image with `format in ("JPEG", "JPG")`, or a filepath ending in `.jpg`/`.jpeg`, the receiver immediately raises:
   ```text
   UnsupportedCarrierFormatError: JPEG-transcoded carriers are not qualified for the current deployment. PaladinMedia requires the original lossless 8-bit RGB PNG carrier.
   ```
2. **Unidentifiable / Damaged Input:**
   If the carrier is supplied as raw pixels (`np.ndarray`) or bytes without identifiable JPEG magic, the receiver does **not** infer that JPEG caused the error. If extraction fails, it raises `CarrierHeaderError`:
   ```text
   CarrierHeaderError: Invalid carrier header magic: expected b'PLDH', got b'...'. The carrier is invalid, corrupted, or stored in an unsupported format; please use the original lossless 8-bit RGB PNG carrier.
   ```
3. **Fail-Closed Security Property:**
   Any alteration to carrier pixels or headers results in `CarrierHeaderError`, `ECCDecodeError`, or `AuthenticationError`. In accordance with project security invariants, no unauthenticated plaintext is ever returned or leaked.


### 3.3 Independent Receiver Extraction and the `exact_match` Contract

1. **`exact_match` Contract:**
   In `MediaStegoPipeline.extract_and_unprotect(stego_carrier, key, aad=None, reference_payload=None)`, the return dataclass `MediaRecoveryResult.exact_match` is a boolean flag initialized to `False`:
   ```python
   exact_match = False
   if reference_payload is not None:
       exact_match = (recovered_payload == reference_payload)
   ```
   When `reference_payload=None` (the true operational receiver path), no reference comparison is performed, so `exact_match` is strictly `False`.
2. **Separation of Recovery Success from Reference Comparison:**
   Payload recovery success and reference comparison are separate concepts:
   - **Payload Recovery Success:** The payload is extracted from carrier wavelet coefficients, Reed–Solomon decoded, AES-256-GCM authenticated, CRC-32 verified, and deserialized into `recovered_payload` (`data` bytes, `metadata`, and `media_type`). Any corruption causes fail-closed exception raising.
   - **Reference Comparison (`exact_match`):** Purely an optional evaluation utility for automated test assertion. An independent operational receiver does not possess the sender's reference object; it operates with `reference_payload=None` and verifies recovery via successful completion without exception.
3. **Header-Driven Adaptation:**
   An independent receiver instance instantiated with default settings (`MediaStegoPipeline(backend="onnx")`, constructor default 32 ECC parity symbols) dynamically derives the sender's ECC profile (`rs_2t_48`, 48 parity symbols) exclusively from `CarrierHeader.flags` (`0x01`), requiring zero sender-side in-memory state or out-of-band coordination.

---

## 4. Workstream C: Payload Capacity and ECC Transparency

### 4.1 Documented Capacity Ceilings and Metadata Semantics

Capacity accounting for 512x512 carriers across all supported profiles:

| Configuration | ECC Parity ($2t$) | Code Rate | Raw Secret Ceiling ($M=0$) | Reference Metadata Ceiling ($M=2$) | Operational Status |
| :--- | :---: | :---: | :---:| :---: | :--- |
| **`rs_2t_48`** | **48 symbols** | **81.18%** | **6,578 B** | **6,576 B** | **Primary Qualified Deployment** |
| `rs_2t_32` | 32 symbols | 87.45% | 7,090 B | 7,088 B | Legacy Default (Poisson risk at limit) |
| `rs_2t_64` | 64 symbols | 74.90% | 6,066 B | 6,064 B | Maximum Resilience (Optional) |

#### Precise Semantics of `metadata_len` and $M$:
1. **Wire Framing Representation:**
   `SecretSerializer` prefixes serialized payloads with a 10-byte header containing a 16-bit big-endian field `metadata_len` ($M$), followed by $M$ bytes of compact UTF-8 JSON.
2. **Empty Mapping Representation:**
   When `SecretPayload.metadata` is `{}` (empty dictionary) or `None`, `SecretSerializer` writes `metadata_len = 0` and includes zero metadata bytes in the wire stream ($M = 0$). It does **not** serialize `{}` to `b"{}"` (2 bytes). Wire framing overhead is exactly 18 bytes.
3. **Raw Ceiling ($M=0$):**
   The documented raw ceiling of **6,578 bytes** corresponds to $M = 0$ (empty metadata dictionary `{}` or `None`).
4. **Reference Metadata Ceiling ($M=2$):**
   The **6,576 bytes** ceiling corresponds to $M = 2$ bytes, which is a theoretical baseline convention established in Sprint 0 / Architecture §10.2 (assuming a hypothetical 2-byte minimal JSON string `"{}"`).
5. **Actual Metadata Byte Length Overhead:**
   Usable capacity is calculated using the actual serialized metadata byte length $M$, with empty metadata handled according to the existing serializer contract ($M=0$). For non-empty metadata, $M = \text{len}(\text{compact UTF-8 JSON})$, giving usable secret capacity of exactly $6,578 - M$ bytes:
   - **Minimal 6-byte non-empty metadata:** A dictionary such as `{"": 0}` serializes to `b'{"":0}'` ($M = 6$ bytes), yielding a usable capacity of $6,578 - 6 = 6,572$ bytes. Tested empirically: 6,572 bytes embeds and extracts end-to-end with a fresh receiver (`reference_payload=None`); 6,573 bytes is rejected before carrier modification.
   - **Typical 7-byte metadata:** Standard single-character key/value dictionaries such as `{"a": 1}` serialize to `b'{"a":1}'` ($M = 7$ bytes), yielding a usable capacity of $6,578 - 7 = 6,571$ bytes. Tested empirically: 6,571 bytes embeds and extracts end-to-end with a fresh receiver; 6,572 bytes is rejected before carrier modification.

### 4.2 Fail-Closed Capacity Preflight

`MediaStegoPipeline.protect_and_embed()` enforces capacity before allocating buffers or calling neural modulation:
1. Calculates actual serialized metadata length: $M = \text{len}(\text{json\_metadata})$ if metadata is truthy, else $0$ (preserving the empty-dict contract).
2. Computes usable capacity: $\text{usable} = \text{CapacityCalculator.calculate\_usable\_secret\_capacity}(M, 256, 48)$.
3. If $\text{len}(\text{secret\_payload.data}) > \text{usable}$, raises `PayloadCapacityExceededError` immediately.
4. Empirically verified: given a payload exceeding the exact ceiling ($6,579\text{ B}$ for $M=0$, $6,573\text{ B}$ for $M=6$, or $6,572\text{ B}$ for $M=7$), the exception is raised before carrier mutation and the input carrier array remains byte-for-byte identical.

---

## 5. Workstream D: Regression Test Evidence

### 5.1 Focused Hardening Suite (`tests/unit/test_phase5c_deployment_hardening.py`)

All 11 targeted deployment hardening tests pass:

```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-8.3.2, pluggy-1.6.0
rootdir: E:\Paladin_media
configfile: pyproject.toml
plugins: anyio-4.14.1
collected 11 items

tests/unit/test_phase5c_deployment_hardening.py::test_operational_entry_point_selects_onnx_and_rs_2t_48 PASSED [  9%]
tests/unit/test_phase5c_deployment_hardening.py::test_legacy_default_constructor_still_uses_rs_2t_32 PASSED [ 18%]
tests/unit/test_phase5c_deployment_hardening.py::test_sender_emits_correct_profile_identifying_header_flags PASSED [ 27%]
tests/unit/test_phase5c_deployment_hardening.py::test_fresh_receiver_adapts_to_header_without_in_memory_state PASSED [ 36%]
tests/unit/test_phase5c_deployment_hardening.py::test_qualified_lossless_png_roundtrip_recovers_exact_payload PASSED [ 45%]
tests/unit/test_phase5c_deployment_hardening.py::test_jpeg_input_explicitly_rejected PASSED [ 54%]
tests/unit/test_phase5c_deployment_hardening.py::test_unreadable_or_corrupted_carrier_fails_closed PASSED [ 63%]
tests/unit/test_phase5c_deployment_hardening.py::test_exact_capacity_boundary_end_to_end_recovery_and_rejection PASSED [ 72%]
tests/unit/test_phase5c_deployment_hardening.py::test_onnx_session_reuse_remains_intact PASSED [ 81%]
tests/unit/test_phase5c_deployment_hardening.py::test_no_sensitive_payload_or_key_data_in_logs_or_errors PASSED [ 90%]
tests/unit/test_phase5c_deployment_hardening.py::test_api_server_endpoints PASSED [100%]

======================== 11 passed, 1 warning in 4.40s ========================
```

### 5.2 Phase 5B Unit Suite Non-Regression

`tests/unit/test_phase5b_onnx_pipeline.py`: **23 / 23 passed in 6.34s**.

### 5.3 Parity Evidence Consistency Check

`scripts/experiments/check_phase5b_parity_consistency.py`: **All consistency checks passed without error.**

### 5.4 Full Repository Test Suite

Command: `pytest`
Result: **370 passed in 44.99s** (`100% PASS`, 0 failures).

---

## 6. Preservation Checks

All frozen governance invariants were verified prior to sprint completion:
- **Phase 4C Checkpoint Digest:** `ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92` verified unchanged.
- **ONNX Model Hashes:** All four Phase 5A ONNX Opset 14 models verified unchanged:
  * `acs_sub_inn_embedding_opset14.onnx`: `3cd22ba35e55d4addfb89e59cdc695f1086383bef5d507f39a3c91dd4917c173`
  * `acs_sub_inn_extraction_opset14.onnx`: `ab696814635b39dec34b432b78e80bb949c5fcd68517a3cec5890102152ff8ba`
  * `neural_spatial_embedding_opset14.onnx`: `7cda5a3becb8283c6b84572e7adb6d64ea29ecbf9f1e667cf34dcd0324e2d784`
  * `neural_spatial_extraction_opset14.onnx`: `bf85d8404469d90c08aa50282db582e3c82b2c6ea2159293423ae6e846b82be2`
- **Canonical Dataset Splits:** 600 train / 100 val / 100 test verified.
- **Training Authorization:** Zero optimizer updates, zero epoch runs, zero parameter weight mutations.

---

## 7. Bounded Qualified Deployment Contract

1. **Qualified Operating Envelope:**
   - Inference Engine: ONNX Runtime (Opset 14) on CPU (`CPUExecutionProvider`).
   - Modulation: ACS-Sub-INN v3 transform-domain neural wavelet steganography.
   - ECC Profile: `rs_2t_48` (Reed–Solomon RS(255, 207), $t=24$).
   - Carrier Format: Lossless 8-bit RGB PNG (512x512).
   - Usable Payload: Up to **6,578 bytes raw** / **6,576 bytes reference**.
   - Channel Resilience: Lossless PNG (100% exact recovery) and additive Gaussian noise $\sigma \le 0.5$ (100% exact recovery).
2. **Unsupported / Non-Qualified Conditions:**
   - Lossy JPEG transcoding (Q=75, Q=85, Q=95) is strictly unsupported; identifiable inputs are explicitly rejected with `UnsupportedCarrierFormatError`.
   - Additive Gaussian noise $\sigma = 1.0$ is an operating boundary (partial recovery; remaining trials fail closed).
   - Additive Gaussian noise $\sigma \ge 2.0$ is unsupported and fails closed via authentication or ECC errors.
   - No plaintext is ever returned on damaged or tampered carriers.

---

## 8. Modified and Added Files

1. [`src/core/exceptions.py`](file:///e:/Paladin_media/src/core/exceptions.py): Added `UnsupportedCarrierFormatError`.
2. [`src/core/__init__.py`](file:///e:/Paladin_media/src/core/__init__.py): Exported `UnsupportedCarrierFormatError`.
3. [`src/carrier/header.py`](file:///e:/Paladin_media/src/carrier/header.py): Enhanced `CarrierHeader.unpack` error messaging to advise using original lossless PNG carriers.
4. [`src/core/media_pipeline.py`](file:///e:/Paladin_media/src/core/media_pipeline.py): Added carrier format metadata to `MediaStegoResult`, format detection and JPEG rejection in `extract_and_unprotect()`, and `get_capacity_limits()` query method.
5. [`src/api/server.py`](file:///e:/Paladin_media/src/api/server.py): Wired operational server to qualified deployment pipeline (`rs_2t_48` + ONNX), added `/api/config` observability endpoint, direct `/api/embed` and `/api/extract` routes with JPEG rejection, and non-sensitive startup logging.
6. [`src/api/event_adapter.py`](file:///e:/Paladin_media/src/api/event_adapter.py): Updated streaming observability adapter to use qualified deployment pipeline by default while preserving legacy injection.
7. [`tests/unit/test_phase5c_deployment_hardening.py`](file:///e:/Paladin_media/tests/unit/test_phase5c_deployment_hardening.py): Created 11-test regression suite covering all Phase 5C requirements.
8. [`scripts/experiments/run_phase5c_assessment.py`](file:///e:/Paladin_media/scripts/experiments/run_phase5c_assessment.py): Created standalone Phase 5C qualification and verification runner.
9. [`docs/experiments/phase5c_results.json`](file:///e:/Paladin_media/docs/experiments/phase5c_results.json): Recorded structured Phase 5C evaluation results.
10. [`docs/audits/PHASE5C_DEPLOYMENT_HARDENING_REPORT.md`](file:///e:/Paladin_media/docs/audits/PHASE5C_DEPLOYMENT_HARDENING_REPORT.md): Comprehensive Phase 5C audit report.
