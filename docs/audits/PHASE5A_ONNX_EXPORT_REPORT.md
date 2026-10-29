# PaladinMedia Phase 5A: ONNX Opset 14 Export and Numerical Parity Assessment Report

**Repository:** `E:\Paladin_media`
**Phase:** 5 — Deployment Optimization
**Sprint:** Phase 5A (ONNX Opset 14 Export & Parity Reconciliation)
**Date:** 2026-10-23
**Source Checkpoint:** `checkpoints/phase4c_extended_30ep/best_model.pt`
**Expected Checkpoint Parameter Digest:** `ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92`
**Execution Environment:** Windows 10, Python 3.10.11, PyTorch 2.5.1+cu124, ONNX 1.23.2, ONNX Runtime 1.23.2, Execution Provider `CPUExecutionProvider`
**Training Authorization:** NONE (Inference-only; zero optimizer steps, parameter updates, or loss weight modifications)
**Final Reconciled Sprint Verdict:** **`PASS_WITH_LIMITATIONS`**
- **Binary Demodulation Gate:** **`PASS`** (12/12 bit-exact, 0.0 error)
- **Strict Parity Gate (`rtol=1e-4, atol=1e-5`):** **`FAIL_STRICT_TOLERANCE`** (58/66 passed, 8 failed)
- **Adjusted Parity Gate (`rtol=1e-4, atol=2e-5`):** **`PASS`** (66/66 passed, 0 failed)
- **Adopted Verdict Criterion:** `adjusted_tolerance_with_strict_limitations_declared`

---

## 1. Executive Summary

Phase 5A was tasked with exporting the accepted `ACS-Sub-INN v3` neural inference path from the Phase 4C trained checkpoint to ONNX Opset 14, establishing structural validity, evaluating numerical parity against PyTorch on the canonical validation partition, and reconciling strict vs. adjusted numerical comparison gates.

In this final reconciliation execution:
1. **Source Checkpoint Safeguards (`PASS`):**
   - Verified fail-closed via `load_training_checkpoint()`.
   - Parameter digest `ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92`, schema `4.0`, architecture `ACS-Sub-INN-v3`, and canonical manifest hashes (`db605887...` / `d0f5f4ee...`) match with 100% precision.
   - Zero parameter mutations or optimizer steps were performed. Held-out test partition remained strictly untouched.
2. **ONNX Opset 14 Export Catalog & Structural Checks (`PASS`):**
   - Four production neural inference graphs were validated in `checkpoints/phase5a_onnx/`:
     - `acs_sub_inn_embedding_opset14.onnx` (385,954 bytes, SHA-256: `3cd22ba35e55d4addfb89e59cdc695f1086383bef5d507f39a3c91dd4917c173`)
     - `acs_sub_inn_extraction_opset14.onnx` (230,704 bytes, SHA-256: `ab696814635b39dec34b432b78e80bb949c5fcd68517a3cec5890102152ff8ba`)
     - `neural_spatial_embedding_opset14.onnx` (429,807 bytes, SHA-256: `7cda5a3becb8283c6b84572e7adb6d64ea29ecbf9f1e667cf34dcd0324e2d784`)
     - `neural_spatial_extraction_opset14.onnx` (234,628 bytes, SHA-256: `bf85d8404469d90c08aa50282db582e3c82b2c6ea2159293423ae6e846b82be2`)
   - All four models passed `onnx.checker.check_model()` with zero errors and were successfully reopened in ONNX Runtime `InferenceSession` (`CPUExecutionProvider`).
   - The parity reconciliation was performed directly on these existing serialized models without re-exporting.
3. **Bit-Exact Binary Demodulation Parity (`PASS`):**
   - In all 12 extraction checks across dynamic spatial resolutions and batch sizes, `extracted_binary` exhibited a maximum absolute error of **0.00000000e+00** (100% bit-exact identity between PyTorch and ONNX Runtime).
4. **Numerical Parity Reconciliation:**
   - **Strict Tolerance Gate (`rtol=1e-4, atol=1e-5`):** **`FAIL_STRICT_TOLERANCE`** (58 passed, 8 failed, 87.88% pass rate). Exactly 8 context-dependent comparisons on `val_image_1` ($512\times 512$) failed the strict $\text{atol}=10^{-5}$ threshold due to 2 pixels out of 65,536 (0.003%) exhibiting a discrepancy of up to $1.37\times 10^{-5}$ (vs. $1.19\times 10^{-5}$ allowed).
   - **Adjusted Tolerance Gate (`rtol=1e-4, atol=2e-5`):** **`PASS`** (66 passed, 0 failed, 100.00% pass rate). All 66 continuous and discrete tensor outputs pass completely.
   - **Adopted Verdict:** **`PASS_WITH_LIMITATIONS`** under the adjusted tolerance, retaining and declaring the strict-tolerance limitation.
5. **Independent Receiver Qualification (`PASS`):**
   - Confirmed `reference_payload` in `MediaStegoPipeline.extract_and_unprotect()` is strictly an optional assertion comparator.
   - Independent recovery on Phase 4C stego carrier with `reference_payload=None` recovered 1,000 / 1,000 secret bytes with **100% exact match** in 388.18 ms.
   - Preserved Phase 4C Codeword 4 Reed–Solomon correction limit observation ($t=16$).
6. **Deployment Qualification Boundary:**
   - Neural inference parity and independent receiver decryption are verified for lossless PNG roundtrips. Full lossy-channel degradation (JPEG, Gaussian noise) and end-to-end integration into production CLI pipelines are explicitly scoped for Phase 5B.

---

## 2. Source Checkpoint and Safeguards Verification

The Phase 4C trained checkpoint was verified fail-closed before and after parity evaluation:

| Invariant / Check | Expected Target | Measured / Observed | Status |
| :--- | :--- | :--- | :--- |
| **Checkpoint Path** | `checkpoints/phase4c_extended_30ep/best_model.pt` | File exists on disk (1.18 MB) | **PASS** |
| **Parameter Digest** | `ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92` | `ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92` | **PASS** |
| **Checkpoint Schema** | `4.0` | `4.0` | **PASS** |
| **Model Architecture**| `ACS-Sub-INN-v3` | `ACS-Sub-INN-v3` | **PASS** |
| **Gamma Min Constraint** | `1.0` | `1.0` | **PASS** |
| **Checkpoint Epoch/Step**| Epoch 30, Step 18000 | Epoch 30, Step 18000 | **PASS** |
| **Manifest File Hash** | `db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff` | Exact match | **PASS** |
| **Composite Dataset Hash**| `d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed` | Exact match | **PASS** |
| **Partition Integrity** | Train: 600, Val: 100, Test: 100 | Train: 600, Val: 100, Test: 100 | **PASS** |
| **Test Split Isolation**| 0 test split samples accessed | 0 test batches touched | **PASS** |

The checkpoint file and parameter weights remained strictly unmutated.

---

## 3. Verified ONNX Opset 14 Model Catalog

The four exported ONNX artifacts in `checkpoints/phase5a_onnx/` were verified for byte size, SHA-256 digest, and structural validity:

| Model Graph Name | Target File | File Size | SHA-256 Hash Digest | `onnx.checker` | ORT Session | Dynamic Axes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`acs_sub_inn_embedding`** | `acs_sub_inn_embedding_opset14.onnx` | 385,954 B | `3cd22ba35e55d4addfb89e59cdc695f1086383bef5d507f39a3c91dd4917c173` | **PASS** | **PASS** | Batch, Height, Width |
| **`acs_sub_inn_extraction`** | `acs_sub_inn_extraction_opset14.onnx` | 230,704 B | `ab696814635b39dec34b432b78e80bb949c5fcd68517a3cec5890102152ff8ba` | **PASS** | **PASS** | Batch, Height, Width |
| **`neural_spatial_embedding`** | `neural_spatial_embedding_opset14.onnx` | 429,807 B | `7cda5a3becb8283c6b84572e7adb6d64ea29ecbf9f1e667cf34dcd0324e2d784` | **PASS** | **PASS** | Batch, Spatial H/W, Subband H/W |
| **`neural_spatial_extraction`** | `neural_spatial_extraction_opset14.onnx` | 234,628 B | `bf85d8404469d90c08aa50282db582e3c82b2c6ea2159293423ae6e846b82be2` | **PASS** | **PASS** | Batch, Spatial H/W, Subband H/W |

---

## 4. Dual-Tolerance Numerical Parity Evidence

Numerical parity between PyTorch and ONNX Runtime was evaluated across representative canonical validation partition samples (`val_image_0` and `val_image_1`) across three dynamic shape configurations:
- `canonical_single_512` ($1\times 3\times 512\times 512$, subbands $256\times 256$)
- `scaled_single_256` ($1\times 3\times 256\times 256$, subbands $128\times 128$)
- `batched_pair_512` ($2\times 3\times 512\times 512$, subbands $256\times 256$)

A total of **66 output tensor comparisons** were evaluated under both strict and adjusted tolerance criteria:

### A. Dual Gate Summary

| Gate / Criterion | Configured Tolerance | Total Checks | Passed | Failed | Pass Rate | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strict Tolerance Gate** | $\text{rtol}=10^{-4}, \text{atol}=10^{-5}$ | 66 | 58 | 8 | 87.88% | **`FAIL_STRICT_TOLERANCE`** |
| **Adjusted Tolerance Gate** | $\text{rtol}=10^{-4}, \text{atol}=2\times 10^{-5}$ | 66 | 66 | 0 | 100.00% | **`PASS`** |
| **Binary Demodulation Gate**| Bit-exact ($\text{max\_abs\_err}=0.0$) | 12 | 12 | 0 | 100.00% | **`PASS`** |

### B. Output Breakdown Across Graph Families

| Output Tensor Name | Checked Runs | Max Absolute Error | Mean Absolute Error | Strict Gate ($\text{atol}=10^{-5}$) | Adjusted Gate ($\text{atol}=2\times 10^{-5}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `extracted_binary` (Demodulated Bits) | 12 | **0.00000000e+00** | **0.00000000e+00** | **12 / 12 PASS** | **12 / 12 PASS** |
| `gamma` (Adaptive Gain) | 12 | $4.76837158\times 10^{-6}$ | $1.28421034\times 10^{-6}$ | **12 / 12 PASS** | **12 / 12 PASS** |
| `stego_image` (Spatial Stego Carrier) | 6 | $1.52587891\times 10^{-5}$ | $1.51684920\times 10^{-6}$ | **6 / 6 PASS** | **6 / 6 PASS** |
| `delta_rec` (Modulated Subband Residual) | 12 | $3.43322754\times 10^{-5}$ | $1.96210386\times 10^{-6}$ | **12 / 12 PASS** | **12 / 12 PASS** |
| `hh_drop_stego` (Modulated Red-HH Subband) | 12 | $6.10351562\times 10^{-5}$ | $1.78241032\times 10^{-6}$ | **10 / 12 PASS (2 FAIL)** | **12 / 12 PASS** |
| `context` / `context_rec` (Predicted Context) | 24 | $6.10351562\times 10^{-5}$ | $1.58432104\times 10^{-6}$ | **18 / 24 PASS (6 FAIL)** | **24 / 24 PASS** |

### C. Detailed Analysis of the Strict-Tolerance Limitation

The 8 comparisons that failed the strict gate occurred exclusively on `val_image_1` at $512\times 512$ (`canonical_single_512` and `batched_pair_512`):
- **Elements failing strict tolerance:** Exactly **2 elements out of 65,536** ($0.003\%$ of elements in the tensor; **99.997% passing rate**):
  - Failing element 1 (idx `[0, 0, 175, 68]`): PyTorch = `0.00235668`, ONNX = `0.00234569`. Absolute discrepancy $\Delta = 1.0989\times 10^{-5}$ vs. allowed tolerance $1.0235\times 10^{-5}$ ($\Delta / \text{tol} = 1.074$).
  - Failing element 2 (idx `[0, 0, 239, 73]`): PyTorch = `-0.01904468`, ONNX = `-0.01905834`. Absolute discrepancy $\Delta = 1.3657\times 10^{-5}$ vs. allowed tolerance $1.1906\times 10^{-5}$ ($\Delta / \text{tol} = 1.147$).
- **Origin:** Standard floating-point SIMD vectorization and FMA accumulation order differences between Intel MKL-DNN / oneDNN (PyTorch) and Microsoft MLAS (ONNX Runtime) across the 8 sequential convolution layers in `ContextPredictorCNN`.
- **Impact on Modulation and Extraction:** Because the adaptive gain $\gamma \ge 1.0$ forces symbol separation to $|\delta_{\text{rec}}| \ge 1.0$, a discrepancy of $1.37\times 10^{-5}$ is over 70,000 times smaller than the decision boundary margin. All demodulated binary bits are 100% bit-exact across all shapes and samples.
- **Tolerances:** With $\text{atol}=2.0\times 10^{-5}$, 100% of elements across all 66 comparisons pass.

---

## 5. End-to-End Receiver Qualification Inspection

Section 5 required investigating whether `reference_payload` in `extract_and_unprotect()` participated in secret reconstruction:

1. **Source Code Inspection:**
   - Inspected [`MediaStegoPipeline.extract_and_unprotect()`](file:///e:/Paladin_media/src/core/media_pipeline.py#L207-L281).
   - Extraction derives bits strictly from the carrier subbands using the recovered header.
   - Decryption and Reed–Solomon decoding operate purely on `extracted_protected`.
   - `reference_payload` is used solely in line 273:
     ```python
     if reference_payload is not None:
         exact_match = (recovered_payload == reference_payload)
     ```
   - It is never passed to `unprotect()`, `RSCodec.decode()`, or `AESGCMProvider.decrypt()`.
2. **Empirical Independent Recovery Test:**
   - The Phase 4C stego carrier (`e2e_stego_carrier.png`) was decrypted with `reference_payload=None`:
     - Recovered payload size: 1,000 bytes.
     - Verbatim byte comparison with original secret: **100% exact match (`True`)**.
     - Receiver execution latency: **388.18 ms**.
     - Proves receiver operates with complete independence from plaintext inputs (zero plaintext leakage).
3. **Reed–Solomon Error Boundary Preservation:**
   - In Phase 4C, Codeword 4 suffered 16 symbol errors, reaching the exact theoretical limit $t=16$ of the RS$(255, 223)$ code.
   - All 5 codewords were repaired with zero uncorrectable errors, but Block 4 operated with zero safety margin.
   - **Recommendation for Phase 5B / Lossy Channel Deployment:** Increase ECC redundancy ($2t=48$, $t=24$) or introduce convolutional bit-interleaving to protect against burst errors in lossy environments.

---

## 6. Sprint Acceptance Matrix & Verdict

| Acceptance Gate | Expected Contract | Measured Evidence | Gate Status |
| :--- | :--- | :--- | :--- |
| **Checkpoint Integrity** | Digest `ec11dce7...`, Schema `4.0`, `gamma_min=1.0` | Exact match verified fail-closed before and after parity | **PASS** |
| **Manifest Immutability** | Canonical DIV2K HR splits (600 / 100 / 100) | Hashes identical, zero test split leakage | **PASS** |
| **Zero Training Invariant** | 0 optimizer steps, 0 epochs, parameters unmutated | Parameters strictly unchanged | **PASS** |
| **Inference Graph Formalization**| Modular deployable graphs with documented contracts | 4 graphs implemented in [`src/neural/onnx_export.py`](file:///e:/Paladin_media/src/neural/onnx_export.py) | **PASS** |
| **ONNX Opset 14 Serialization**| 4 valid ONNX models serialized and verified | All 4 models verified in `checkpoints/phase5a_onnx/` | **PASS** |
| **Structural Validity** | `onnx.checker.check_model()` | 4 / 4 models passed checker with zero errors | **PASS** |
| **ONNX Runtime Reopening** | Model loads in `InferenceSession` | 4 / 4 models loaded with `CPUExecutionProvider` | **PASS** |
| **Dynamic Shape Support** | Verified on dynamic batch and spatial axes | Tested on $512\times 512, 256\times 256, B=1, 2$ | **PASS** |
| **Binary Demodulation Gate** | `extracted_binary` identical to PyTorch | 12 / 12 checks bit-exact (0.0 error) | **PASS** |
| **Strict Numerical Parity Gate** | All comparisons pass at $\text{rtol}=10^{-4}, \text{atol}=10^{-5}$ | 58 passed, 8 failed (87.88% pass rate) | **`FAIL_STRICT_TOLERANCE`** |
| **Adjusted Numerical Parity Gate**| All comparisons pass at $\text{rtol}=10^{-4}, \text{atol}=2\times 10^{-5}$ | 66 passed, 0 failed (100.00% pass rate) | **`PASS`** |
| **Independent Receiver** | Operates without plaintext reference | 1,000 bytes exact match verified | **PASS** |
| **Focused Unit Test Suite** | Automated regression tests for inference boundary | [`tests/unit/test_onnx_inference_boundary.py`](file:///e:/Paladin_media/tests/unit/test_onnx_inference_boundary.py) (7/7 passed in 3.78s) | **PASS** |

### **FINAL SPRINT VERDICT: PASS_WITH_LIMITATIONS**
**Adopted Criterion:** `adjusted_tolerance_with_strict_limitations_declared`

---

## 7. Deliverables Summary

- **Verified ONNX Opset 14 Models:**
  - `checkpoints/phase5a_onnx/acs_sub_inn_embedding_opset14.onnx` (385,954 bytes, SHA-256: `3cd22ba3...`)
  - `checkpoints/phase5a_onnx/acs_sub_inn_extraction_opset14.onnx` (230,704 bytes, SHA-256: `ab696814...`)
  - `neural_spatial_embedding_opset14.onnx` (429,807 bytes, SHA-256: `7cda5a3b...`)
  - `neural_spatial_extraction_opset14.onnx` (234,628 bytes, SHA-256: `bf85d840...`)
- **Parity Telemetry & Reconciled JSON:**
  - [`docs/experiments/phase5a_onnx_parity_results.json`](file:///e:/Paladin_media/docs/experiments/phase5a_onnx_parity_results.json)
- **Automated Audit Runner (Parity-Only by Default):**
  - [`scripts/experiments/run_phase5a_onnx_export.py`](file:///e:/Paladin_media/scripts/experiments/run_phase5a_onnx_export.py)
- **Unit Regression Tests:**
  - [`tests/unit/test_onnx_inference_boundary.py`](file:///e:/Paladin_media/tests/unit/test_onnx_inference_boundary.py) (7 passed in 3.78s)
- **Source Checkpoint (Preserved):**
  - `checkpoints/phase4c_extended_30ep/best_model.pt` (SHA-256 digest: `ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92`)
