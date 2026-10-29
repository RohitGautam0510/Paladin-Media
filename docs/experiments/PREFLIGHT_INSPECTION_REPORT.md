# PaladinMedia — Phase 2 Preflight Inspection & Dataset Audit

**Document ID:** PDM-P2-PREFLIGHT-001
**Date:** 2026-10-09
**Status:** COMPLETE & VERIFIED
**Author:** PaladinMedia Senior Research & Validation Engineering

---

## 1. Executive Summary

Prior to executing the Phase 2 Experiment Program, a comprehensive preflight inspection of the local repository, datasets, cryptographic accounting model, and test suite was conducted in accordance with the governing Phase 2 specification.

Key preflight findings:
1. **MVP 0 Dataset (Dataset A):** Verified. Contains 60 images of exact dimension $512 \times 512$ in 8-bit RGBA/RGB format. Ready for E03 baseline validation.
2. **DIV2K Dataset (Dataset B):** Audited. The available local directory is `DIV2K_train_LR_bicubic_X2/DIV2K_train_LR_bicubic/X2/`, containing exactly 800 images in PNG RGB format.
   - Native width range: $558 \le W \le 1020$ pixels.
   - Native height range: $324 \le H \le 1020$ pixels.
   - Native minimum dimension $\min(W, H)$ range: $324 \le \min(W, H) \le 1020$ pixels.
   - **Crucial Constraint:** High-Resolution (HR) 2K source images are **not present** in the local environment. Because the maximum native dimension across all 800 images is 1020 pixels, **no image natively supports square crops $\ge 1024 \times 1024$**.
   - Per non-negotiable experimental constraints (Rule 2: *"Do not upscale LR images and describe them as native-resolution HR carriers"* and Rule 3: *"If a target resolution cannot be obtained without upscaling, mark that configuration as unavailable"*), all target carrier dimensions $\ge 1024 \times 1024$ (1024, 2048, 3072, 4096, 5120) for E04 will be documented as **UNAVAILABLE / SKIPPED**.
   - 785 out of 800 images natively support $512 \times 512$ square crops without resizing or interpolation.
3. **Capacity Hard Gate & Accounting:** Audited and verified. Usable secret plaintext capacity under the 1-channel HH baseline is strictly 7,088 bytes. The hard gate throws `PayloadCapacityExceededError` whenever protected payload exceeds 8,176 bytes. Oversized payloads fail closed before DWT or embedding.
4. **Test Suite:** 99 of 99 automated tests passing across `tests/unit/` and `tests/integration/`.

---

## 2. Dataset Inventory & Audit

### 2.1 Dataset A: MVP 0 Carrier Dataset
* **Path:** `e:\Paladin_media\MVP 0 Dataset`
* **File Count:** 60 images (`0001x2.png` through `0060x2.png`)
* **Format:** PNG
* **Dimensions:** $512 \times 512$ pixels (uniform across all 60 files)
* **Color Mode:** RGBA (converted to RGB uint8 for processing)
* **Value Range:** $[0, 255]$ uint8
* **Purpose:** Serves as the fixed reference baseline carrier dataset for **E03 Baseline End-to-End Validation**.

### 2.2 Dataset B: DIV2K Carrier Dataset
* **Path:** `e:\Paladin_media\DIV2K_train_LR_bicubic_X2\DIV2K_train_LR_bicubic\X2`
* **File Count:** 800 images (`0001x2.png` through `0800x2.png`)
* **Archive Present:** `DIV2K_train_LR_bicubic_X2.zip` (925,390,592 bytes)
* **Format:** PNG
* **Color Mode:** RGB (uniform across all 800 files)
* **Data Type:** `uint8`, value range $[0, 255]$
* **Dimensions Summary:**
  * Width: min = 558, max = 1020, median = 1020
  * Height: min = 324, max = 1020, median = 680
  * $\min(W, H)$: min = 324, max = 1020
* **Crop Availability:**
  * $\ge 512 \times 512$: **785 images** (98.1%) support native deterministic center crops.
  * $\ge 1024 \times 1024$: **0 images** (0.0%). Max dimension is 1020 px.
  * $\ge 2048 \times 2048$: **0 images** (0.0%).
* **Preflight Dataset Verdict for E04 (Spatial Scaling):**
  * Target $512 \times 512$: **SUPPORTED** via deterministic center crops from DIV2K images.
  * Target $1024 \times 1024$: **SKIPPED (UNAVAILABLE)** — native resolution capped at 1020 px; upscaling prohibited.
  * Target $2048 \times 2048$: **SKIPPED (UNAVAILABLE)** — HR dataset absent; upscaling prohibited.
  * Target $3072 \times 3072$: **SKIPPED (UNAVAILABLE)** — HR dataset absent; upscaling prohibited.
  * Target $4096 \times 4096$: **SKIPPED (UNAVAILABLE)** — HR dataset absent; upscaling prohibited.
  * Target $5120 \times 5120$: **SKIPPED (UNAVAILABLE)** — HR dataset absent; upscaling prohibited.

---

## 3. Capacity Accounting Model & Verification

### 3.1 Mathematical Formulation (Baseline 1 HH Channel)
For carrier dimension $H \times W = 512 \times 512$:
1. **DWT Decomposition:** 1-level 2D Haar DWT produces subbands of spatial dimension $\frac{H}{2} \times \frac{W}{2} = 256 \times 256$.
2. **Channel Allocation:** Red channel (Channel 0) of the diagonal detail subband ($\text{HH}$).
3. **Raw Embedding Positions:** $N_{\text{raw}} = 256 \times 256 = 65,536$ coefficients.
4. **Modulation Scheme:** 1 bit per coefficient position ($0.25\text{ bpp}$ relative to carrier image).
5. **Raw Embedding Capacity:**
   $$C_{\text{raw}} = \frac{65,536 \text{ bits}}{8 \text{ bits/byte}} = 8,192 \text{ bytes}$$
6. **Fixed Carrier Header:** 16 bytes (128 bits) occupied at fixed anchor indices $[0 \dots 127]$.
7. **Maximum Protected Payload Capacity:**
   $$C_{\text{protected}}^{\max} = C_{\text{raw}} - 16 = 8,176 \text{ bytes}$$
8. **Reed-Solomon Framing:** $n = 255, k = 223$ ($2t = 32$ parity bytes per 223 data bytes) plus a 4-byte stream length prefix.
   - For 32 RS blocks: $32 \times 223 = 7,136$ bytes maximum encrypted data.
   - Total RS output: $4 + 7,136 + (32 \times 32) = 8,164 \text{ bytes} \le 8,176 \text{ bytes}$.
9. **AES-256-GCM Overhead:** 12-byte IV/nonce + 16-byte GHASH authentication tag = 28 bytes.
   - Maximum serialized payload: $7,136 - 28 = 7,108 \text{ bytes}$.
10. **SecretSerializer Overhead:** 10-byte magic/version/media-type + 4-byte payload length + 4-byte CRC-32 = 18 bytes base framing.
11. **Usable Secret Plaintext Capacity:**
    $$C_{\text{secret}}^{\text{usable}} = 7,108 - 18 - \text{len(metadata)} \le 7,088 \text{ bytes (with standard metadata)}$$

### 3.2 Hard-Gate Enforcement Audit
* **Boundary Evaluation:**
  - $8,176\text{ B}$ protected payload: `fits = True`, `remaining = 0 B`.
  - $8,177\text{ B}$ protected payload: `fits = False`, raises `PayloadCapacityExceededError`.
* **Fail-Closed Verification:** Rejection occurs **strictly before** any transform, coefficient modification, or carrier allocation.
* **Oversized Input Handling:** Payloads larger than $8,176\text{ B}$ (e.g. 791,251 bytes from uncompressed large images) are rejected by hard gate with zero state corruption.

---

## 4. Test Infrastructure Preflight

The entire test suite was executed in the clean environment:
```
pytest tests
============================= 99 passed in 2.45s ==============================
```
All unit tests (`test_carrier_capacity.py`, `test_carrier_dwt.py`, `test_carrier_embedding.py`, `test_carrier_header.py`, `test_carrier_mask.py`, `test_crypto.py`, `test_ecc.py`, `test_exceptions.py`, `test_serialization.py`) and integration tests (`test_fail_closed.py`, `test_media_pipeline.py`, `test_pipeline.py`) pass without failure or warning.

---

## 5. Authorization for Experiment Execution

All preflight gates are satisfied:
- [x] Repository and directory structure verified.
- [x] Dataset variants, counts, and dimensions characterized.
- [x] Hard-gate capacity bounds mathematically and programmatically verified.
- [x] Baseline test suite fully green (99/99).
- [x] E04 native dimension restrictions identified and documented.

**Proceeding to Step 2: E03 — MVP0 Baseline End-to-End Validation.**
