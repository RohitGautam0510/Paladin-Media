# PaladinMedia — Phase 5B Acceptance Repair and Qualification Report
## ONNX Integration, Numerical Parity Resolution, and Channel-Robustness Qualification

**Repository:** `E:\Paladin_media`
**Phase:** 5 — Deployment Optimization
**Sprint:** Phase 5B Final Acceptance Repair, Reconciliation, and Qualification
**Target Architecture:** ACS-Sub-INN v3 (`gamma_min = 1.0`)
**Baseline Artifacts:**
- Accepted Phase 4C Extended Checkpoint: `checkpoints/phase4c_extended_30ep/best_model.pt`
- Accepted Phase 5A ONNX Opset 14 Graphs: `checkpoints/phase5a_onnx/*.onnx`
- Canonical Dataset Manifest: `docs/project/div2k_hr_split.json`
**Evaluation Script:** `scripts/experiments/run_phase5b_assessment.py`
**Current Evaluation Artifact:** `docs/experiments/phase5b_results.json`
**Historical Initial Artifact (Preserved):** `docs/experiments/phase5b_initial_results.json`
**Unit Test Suite:** `tests/unit/test_phase5b_onnx_pipeline.py` (23 focused tests; **359/359 full suite passing in 43.76s**)
**Date:** 2026-10-24

---

## 1. Executive Summary and Final Verdict

This sprint executes the final evidence reconciliation, acceptance repair, and qualification of the ONNX-backed inference path for PaladinMedia. It resolves all discrepancies between declared gate statuses, capacity derivations, constructor configurations, and empirical measurements through:

1. **Reconciled Capacity Accounting:** Complete derivation separating raw application secret capacity ($M=0$ metadata bytes: 7,090 B, 6,578 B, 6,066 B) from reference metadata capacity ($M=2$ metadata bytes: 7,088 B, 6,576 B, 6,064 B), with exact boundary verification (at limit, one below, and one above rejected fail-closed before carrier modification).
2. **Production Gate Aggregator with Raw-Trial Verification:** Standalone production aggregation function `aggregate_phase5b_gates()` independently aggregates raw trial records for `rs_2t_48` and channel conditions, strictly verifying agreement between individual raw trials and declared summary counts. Any disagreement, missing trial, or failed recovery forces the gate to `FAIL` and blocks acceptance (`BLOCKED`).
3. **Removal of Capacity-Gate Bypass:** Removed all fallback bypasses in `aggregate_phase5b_gates()`. For `rs_2t_48`, all three mandatory boundary checks (at limit recovered, one-below recovered, one-above rejected fail-closed) must strictly pass; oversized rejection serves as a supplementary test and cannot override mandatory checks.
4. **Qualified Deployment Configuration vs. Legacy Default:** Provided the explicit factory method `MediaStegoPipeline.create_qualified_deployment()`, which explicitly configures `rs_2t_48` (RS(255, 207), $t=24$) with native ONNX Runtime backend. Preserved legacy default behavior (`MediaStegoPipeline()` without arguments defaults to `rs_2t_32`).
5. **Corrected Telemetry:** Fixed `protected_payload_bytes` in trial records to measure the actual protected payload length in bytes (204–6,801 B) rather than the constant 4-byte width of the serialized length integer.
6. **Consistent Numerical Parity Contract:** Documented distinction between image-level pass counts (99/100 strict, 100/100 adjusted), element-level distribution (98.9627% within $1\times 10^{-5}$, 99.9964% within $2\times 10^{-5}$, 99.999991% within $5\times 10^{-5}$, 100% within $1\times 10^{-4}$), and direct same-input graph extraction ($5 / 6.55\text{M}$ bit errors) versus operational roundtrip extraction ($100 / 100$ exact, 0 bit errors).
7. **Multi-Carrier Channel Characterization (39 Trials):** Accurately reports Gaussian noise $\sigma = 1.0$ as an operating boundary condition with **partial recovery (33.33%)**, rather than claiming zero recovery.

### Dynamically Computed Gate Statuses

All gate statuses below are computed directly by `aggregate_phase5b_gates()` consuming `docs/experiments/phase5b_results.json`:

| Gate | Scope | Status | Measured Empirical Finding |
| :--- | :--- | :---: | :--- |
| **G1: Preflight Integrity** | Model digest, ONNX files, manifest hashes | **`PASS`** | Checkpoint digest `ec11dce7...` verified invariant. 4 ONNX models and canonical manifest verified. Zero optimizer updates. |
| **G2: Numerical Parity** | 100 validation images, multi-shape graphs | **`PASS_WITH_LIMITATIONS`** | Image-level: 99/100 pass strict (`1e-5`), 100/100 pass adjusted (`2e-5`). Element-level: 98.9627% within `1e-5`, 99.9964% within `2e-5`, 99.999991% within `5e-5`, 100% within `1e-4`. Direct extraction on identical inputs: 95/100 exact, BER $= 7.63\times 10^{-7}$ (5 errors / 6.55M bits). Roundtrip extraction: 100/100 bit-exact (0 BER). |
| **G3: ONNX Operational Integration** | `NeuralModulationEngine`, `MediaStegoPipeline` | **`PASS`** | Native `backend="onnx"` integration with cached session reuse, operational pipeline path verified across all payload classes with 1.41x–3.08x latency speedups. |
| **G4: End-to-End Lossless Recovery** | Raw trial aggregation (60 trials) across profiles & payloads | **`PASS`** | Primary qualified profile `rs_2t_48` achieves **100.0% exact recovery (20/20 trials)** across all payloads (100B, 1000B, 2500B, 5000B). Mandatory capacity boundary tests (at limit, one below, one above rejected) all pass without bypass. |
| **G5: Channel Robustness Characterization** | 39 trials across 3 carriers, multi-seed noise, JPEG | **`PASS_WITH_LIMITATIONS`** | Lossless PNG (3/3 pass) and Gaussian noise $\sigma=0.5$ (9/9 pass across 3 seeds) fully supported. Noise $\sigma=1.0$ is partial (3/9 pass, 6/9 fail closed). Noise $\sigma=2.0$ (9/9 fail closed) and JPEG Q=75–95 (9/9 fail closed) are unsupported and fail closed. |
| **G6: ECC Policy Decision** | Profile comparison, capacity boundaries, interoperability | **`PASS`** | `rs_2t_48` adopted as qualified deployment profile (6,578 B raw / 6,576 B reference). `MediaStegoPipeline.create_qualified_deployment()` factory provided. Legacy default (`rs_2t_32`) preserved. |
| **Overall Sprint Verdict** | Phase 5B Acceptance Qualification | **`PASS_WITH_LIMITATIONS`** | All criteria verified by fresh execution evidence; zero unverified claims; all limitations honestly characterized. |

---

## 2. Preflight Verification and Invariant Safeguards

All preflight checks were executed and verified fail-closed:

1. **Phase 4C Checkpoint Verification:**
   - Path: `checkpoints/phase4c_extended_30ep/best_model.pt`
   - Parameter Digest: `ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92` (`MATCH`)
   - Epoch: `30` | Best Validation Metric: `0.565608`
2. **Canonical Manifest Integrity:**
   - Path: `docs/project/div2k_hr_split.json`
   - File SHA-256: `db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff` (`MATCH`)
   - Composite SHA-256: `d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed` (`MATCH`)
   - Partition Counts: 600 train, 100 validation, 100 test (held-out test split remained untouched).
3. **Phase 5A ONNX Graph Integrity (Opset 14):**
   - `acs_sub_inn_embedding_opset14.onnx`: SHA-256 `3cd22ba35e55d4addfb89e59cdc695f1086383bef5d507f39a3c91dd4917c173` (`MATCH`)
   - `acs_sub_inn_extraction_opset14.onnx`: SHA-256 `ab696814635b39dec34b432b78e80bb949c5fcd68517a3cec5890102152ff8ba` (`MATCH`)
   - `neural_spatial_embedding_opset14.onnx`: SHA-256 `7cda5a3becb8283c6b84572e7adb6d64ea29ecbf9f1e667cf34dcd0324e2d784` (`MATCH`)
   - `neural_spatial_extraction_opset14.onnx`: SHA-256 `bf85d8404469d90c08aa50282db582e3c82b2c6ea2159293423ae6e846b82be2` (`MATCH`)
4. **Safety Boundaries:**
   - Exactly zero `optimizer.step()`, zero gradient updates, zero model parameter mutations.

---

## 3. Reconciled Capacity Accounting and Exact Boundary Analysis

### 3.1 First-Principles Capacity Derivation

For a standard $512 \times 512$ RGB carrier processed via single-level 2D Haar DWT:

1. **Transform-Domain Raw Capacity:**
   - 1 HH channel (Red channel): $256 \times 256 = 65,536$ coefficient positions at 1 bit/coefficient $= \mathbf{8,192\text{ bytes}}$.
2. **Fixed Carrier Header Reservation:**
   - Canonical 16-byte header (`CarrierHeader`: Magic 4B, Version 1B, Flags 1B, Checksum 2B, Nonce 4B, PayloadLength 4B).
   - $\text{Max Protected Payload Capacity} = 8,192 - 16 = \mathbf{8,176\text{ bytes}}$.
3. **Reed–Solomon Framing and Codeword Allocation:**
   - 4-byte length prefix framing $\implies 8,176 - 4 = \mathbf{8,172\text{ bytes}}$ available for RS blocks.
   - Systematic block size $N = 255$ symbols over $\text{GF}(2^8)$:
     $$\lfloor 8,172 / 255 \rfloor = \mathbf{32\text{ full blocks}} \quad (32 \times 255 = 8,160\text{ bytes})$$
   - Carrier remainder: $8,172 - 8,160 = \mathbf{12\text{ bytes}}$.
   - *Shortened Block Constraint:* A 33rd shortened block requires $d + 2t$ bytes ($d \ge 1$). For all three profiles ($2t \in \{32, 48, 64\}$), $1 + 2t \ge 33 > 12$. Therefore, 0 data bytes fit in the 12-byte remainder, accommodating exactly 32 blocks.
   - Maximum ciphertext bytes in 32 blocks ($32 \times K$ where $K = 255 - 2t$):
     - `rs_2t_32` ($K = 223$): $32 \times 223 = \mathbf{7,136\text{ bytes}}$.
     - `rs_2t_48` ($K = 207$): $32 \times 207 = \mathbf{6,624\text{ bytes}}$.
     - `rs_2t_64` ($K = 191$): $32 \times 191 = \mathbf{6,112\text{ bytes}}$.
4. **Cryptographic Overhead (AES-256-GCM):**
   - 12-byte random nonce $+$ 16-byte GHASH tag $= \mathbf{28\text{ bytes}}$.
   - Maximum serialized secret payload bytes:
     - `rs_2t_32`: $7,136 - 28 = \mathbf{7,108\text{ bytes}}$.
     - `rs_2t_48`: $6,624 - 28 = \mathbf{6,596\text{ bytes}}$.
     - `rs_2t_64`: $6,112 - 28 = \mathbf{6,084\text{ bytes}}$.
5. **Serialization Overhead (`SecretSerializer`):**
   - Base wire framing requires 18 bytes: Prefix 10B (Magic 4, Version 1, Type 1, Flags 2, MetaLen 2) $+$ Payload Length 4B $+$ CRC-32 Checksum 4B.
   - Optional JSON metadata adds $M$ bytes (compact UTF-8 encoded string).
6. **Maximum Application Secret Plaintext Capacity ($L = \text{max\_serialized} - 18 - M$):**
   - **Zero-Metadata Raw Limit ($M = 0$, `metadata=None` or `{}`):**
     - `rs_2t_32`: $7,108 - 18 = \mathbf{7,090\text{ bytes}}$.
     - `rs_2t_48`: $6,596 - 18 = \mathbf{6,578\text{ bytes}}$.
     - `rs_2t_64`: $6,084 - 18 = \mathbf{6,066\text{ bytes}}$.
   - **Reference Metadata Limit ($M = 2$, minimal JSON `"{}"`):**
     - `rs_2t_32`: $7,108 - 18 - 2 = \mathbf{7,088\text{ bytes}}$.
     - `rs_2t_48`: $6,596 - 18 - 2 = \mathbf{6,576\text{ bytes}}$.
     - `rs_2t_64`: $6,084 - 18 - 2 = \mathbf{6,064\text{ bytes}}$.

### 3.2 Reconciliation of Report vs. JSON Discrepancy

The 2-byte difference between initial telemetry ($7,090 / 6,578 / 6,066\text{ B}$) and the initial audit text ($7,088 / 6,576 / 6,064\text{ B}$) represented two distinct engineering quantities:
- The JSON calculated the **raw application secret data ceiling** assuming no metadata ($M = 0$).
- The report quoted the **reference metadata capacity** assuming 2-byte minimal JSON metadata ($M = 2$).

Both quantities are now formally separated and exposed in telemetry and code:
- `raw_usable_capacity_bytes`: Pure secret data ceiling ($M=0$).
- `reference_metadata_capacity_bytes`: Usable secret capacity with 2B minimal reference metadata ($M=2$).

### 3.3 Exact Boundary Verification Across Profiles

Evaluated on canonical validation carrier `0601.png` with ONNX backend and fresh receiver instances:

| Profile | Parity $2t$ | Raw Limit ($M=0$) | Ref Limit ($M=2$) | Tested Size | Measured Outcome | Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `rs_2t_32` | 32 B | 7,090 B | 7,088 B | 7,089 B (Limit - 1) | `PASS_RECOVER` | Embedded and recovered in clean trial |
| | | | | 7,090 B (At Limit) | `FAIL_DECODE` | Embedded, but RS decode fails: Poisson tail on 32 blocks exceeds $t=16$ |
| | | | | 7,091 B (Limit + 1) | **`REJECTED`** | **Rejected fail-closed** before carrier modification (`PayloadCapacityExceededError`) |
| **`rs_2t_48`** | **48 B** | **6,578 B** | **6,576 B** | **6,577 B (Limit - 1)** | **`EXACT_PASS`** | **Exact recovery (`exact_match=True`) with fresh receiver** |
| | | | | **6,578 B (At Limit)** | **`EXACT_PASS`** | **Exact recovery (`exact_match=True`) with fresh receiver** |
| | | | | **6,579 B (Limit + 1)** | **`REJECTED`** | **Rejected fail-closed** before carrier modification (`PayloadCapacityExceededError`) |
| `rs_2t_64` | 64 B | 6,066 B | 6,064 B | 6,065 B (Limit - 1) | `EXACT_PASS` | Exact recovery (`exact_match=True`) with fresh receiver |
| | | | | 6,066 B (At Limit) | `EXACT_PASS` | Exact recovery (`exact_match=True`) with fresh receiver |
| | | | | 6,067 B (Limit + 1) | **`REJECTED`** | **Rejected fail-closed** before carrier modification (`PayloadCapacityExceededError`) |

**Empirical Boundary Finding:** For `rs_2t_48` and `rs_2t_64`, testing at the exact boundary demonstrates that $L = \text{raw\_limit}$ is 100% recoverable, while $L = \text{raw\_limit} + 1$ is immediately rejected fail-closed before any carrier pixels or coefficients are mutated. For `rs_2t_32`, while 7,090 B fits the carrier capacity check, neural channel bit errors exceed $t=16$ across 32 blocks in the recorded trial, empirically confirming that `rs_2t_32` cannot be reliably deployed at full capacity.

---

## 4. Production Gate Aggregation and Bypass Removal

### 4.1 Strict Raw-Trial Aggregation
The gate aggregator `aggregate_phase5b_gates()` independently iterates over the raw trial records in `qualification_matrix["rs_2t_48"]["payloads"]` and `trial_records`:
- Computes `raw_p48_total` and `raw_p48_success` directly from individual trial `exact_match` booleans.
- Confirms that `raw_p48_total == declared_summary_trials` and `raw_p48_success == declared_summary_success`.
- Any discrepancy between raw trial records and summary entries immediately triggers gate failure (`FAIL`) and blocks acceptance (`BLOCKED`).

### 4.2 Complete Removal of Capacity-Gate Fallback
The evaluation expression for `p48_bnd_ok` previously included an `or (oversized_fail_closed_test... == True)` fallback. This bypass was removed:
- Mandatory checks (`at_limit_pass is True`, `one_below_pass is True`, `one_above_rejected_fail_closed is True`) must **all** be satisfied.
- Oversized rejection (`raw_limit + 100`) is retained strictly as a supplementary safety invariant, never a substitute for mandatory boundary checks.

---

## 5. Workstream A: Numerical Parity Reconciliation

### 5.1 100-Image Validation Scan: Image-Level vs. Element-Level Outcomes

Parity was evaluated across all 100 canonical validation images (IDs 0601–0700) on 512x512 spatial inputs:

#### Image-Level Outcomes
- **Images Evaluated:** 100 validation images
- **Strict `np.allclose` Pass Count (`1e-5`):** **99 / 100 (99.0%)**
- **Adjusted `np.allclose` Pass Count (`2e-5`):** **100 / 100 (100.0%)**
- **Maximum Absolute Discrepancy Overall:** $6.1035 \times 10^{-5}$
- **Mean Absolute Discrepancy Overall:** $2.9211 \times 10^{-7}$

#### Element-Level Distribution (78,643,200 Total Elements)
- Elements within strict `atol = 1e-5`: **77,827,472 / 78,643,200 (approximately 98.9627%)**
- Elements within adjusted `atol = 2e-5`: **78,640,358 / 78,643,200 (approximately 99.9964%)**
- Elements within `atol = 5e-5`: **78,643,193 / 78,643,200 (approximately 99.999991%)**
- Elements within `atol = 1e-4`: **78,643,200 / 78,643,200 (100.0%)**

### 5.2 Direct Graph Parity vs. Separate Roundtrip Parity

1. **Direct Extraction Graph Parity (Identical Carrier Input):**
   - Both PyTorch `extract_spatial` and ONNX `extract_spatial` were fed the **exact same carrier tensor**.
   - Total bits evaluated across 100 images: 6,553,600 bits.
   - Exact bit-match images: **95 / 100 (95.0%)**.
   - Total bit errors across 6.55M bits: **5 bit errors**.
   - Binary Bit Error Rate: **$7.63 \times 10^{-7}$** (0.000076%).
   - Maximum delta absolute error: $9.16 \times 10^{-5}$.
2. **Separate Roundtrip Parity:**
   - PyTorch embed $\to$ PyTorch extract vs. ONNX embed $\to$ ONNX extract on their respective stego outputs.
   - Total bits evaluated: 6,553,600 bits.
   - Exact bit-match images: **100 / 100 (100.0%)**.
   - Total bit errors: **0**.
   - Binary Bit Error Rate: **0.0000%**.

---

## 6. Workstreams B & C: Multi-Trial ECC Profile Qualification

### 6.1 Wire Framing and Receiver Profile Compatibility

To eliminate implicit receiver assumptions without breaking format compatibility, `CarrierHeader.flags` (uint8, byte 5 of the 16-byte header) explicitly encodes the ECC profile via mask `0x0F`:
- `0x00`: `rs_2t_32` (RS(255, 223), baseline legacy default, 100% backward compatible)
- `0x01`: `rs_2t_48` (RS(255, 207), primary qualified deployment profile)
- `0x02`: `rs_2t_64` (RS(255, 191), maximum theoretical resilience)

**Fresh Receiver Verification:** In every trial, a **fresh receiver pipeline instance** was constructed with zero pre-shared state. Upon reading the extracted 16-byte carrier header, the receiver inspects `header.ecc_parity` and dynamically configures the matching Reed–Solomon decoder, achieving seamless interoperability without guessing.

### 6.2 Multi-Trial Lossless Qualification Matrix (60 Trials)

Evaluated on canonical validation carrier `0601.png` across 5 independent trials per configuration with fresh AES-256-GCM nonces:

| Profile | Payload Size | Trials | Exact Matches | Recovery Rate | Mean PSNR | Mean SSIM | Mean BER | Mean Latency | Protected Payload Bytes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`rs_2t_32`** | 100 B | 5 | 5 | **100.0%** | 61.72 dB | 1.00000 | 0.0065 | 226.95 ms | 204 B |
| | 1,000 B | 5 | 5 | **100.0%** | 54.19 dB | 0.99998 | 0.0053 | 244.60 ms | 1,233 B |
| | 2,500 B | 5 | 5 | **100.0%** | 50.31 dB | 0.99994 | 0.0051 | 316.58 ms | 2,957 B |
| | 5,000 B | 5 | 4 | **80.0%** | 47.44 dB | 0.99989 | 0.0052 | 412.38 ms | 5,809 B |
| **`rs_2t_48`** (Qualified) | 100 B | 5 | 5 | **100.0%** | 61.42 dB | 1.00000 | 0.0063 | 227.18 ms | 220 B |
| | 1,000 B | 5 | 5 | **100.0%** | 53.64 dB | 0.99997 | 0.0053 | 260.67 ms | 1,361 B |
| | 2,500 B | 5 | 5 | **100.0%** | 49.99 dB | 0.99994 | 0.0051 | 370.08 ms | 3,197 B |
| | 5,000 B | 5 | 5 | **100.0%** | 47.05 dB | 0.99988 | 0.0056 | 457.83 ms | 6,273 B |
| **`rs_2t_64`** | 100 B | 5 | 5 | **100.0%** | 60.96 dB | 1.00000 | 0.0057 | 223.77 ms | 236 B |
| | 1,000 B | 5 | 5 | **100.0%** | 53.40 dB | 0.99997 | 0.0048 | 277.65 ms | 1,457 B |
| | 2,500 B | 5 | 5 | **100.0%** | 49.64 dB | 0.99993 | 0.0051 | 365.61 ms | 3,469 B |
| | 5,000 B | 5 | 5 | **100.0%** | 46.72 dB | 0.99987 | 0.0053 | 516.66 ms | 6,801 B |

### 6.3 Execution Latency: PyTorch vs. ONNX Runtime (`rs_2t_48`)

| Payload Size | PyTorch Latency | ONNX Latency | Speedup Factor | Exact Recovery |
| :---: | :---: | :---: | :---: | :---: |
| **100 B** | 353.26 ms | 251.12 ms | **1.41x** | Both Exact |
| **1,000 B** | 588.85 ms | 330.22 ms | **1.78x** | Both Exact |
| **2,500 B** | 897.94 ms | 387.25 ms | **2.32x** | Both Exact |
| **5,000 B** | 1,343.21 ms | 435.70 ms | **3.08x** | Both Exact |

---

## 7. Workstream D: Channel Degradation Characterization

Evaluated using qualified profile `rs_2t_48` and a 1,000-byte reference fixture across 3 canonical validation carriers (`0601.png`, `0602.png`, `0603.png`) and 39 total trials:

### 7.1 Empirical Channel Matrix

| Channel Condition | Trials | Exact Recoveries | Recovery Rate | Mean PSNR | Mean SSIM | Mean BER | Failure Modes | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Lossless PNG (Control)** | 3 | 3 | **100.0%** | $\infty$ | 1.00000 | 0.29% | None | **`PASS`** |
| **Gaussian Noise $\sigma = 0.5$** | 9 | 9 | **100.0%** | 53.05 dB | 0.99997 | 0.39% | None | **`PASS`** |
| **Gaussian Noise $\sigma = 1.0$** | 9 | 3 | **33.33%** | 47.85 dB | 0.99991 | 1.19% | Header Corrupted (4), Mask Error (2) | **`PARTIAL`** |
| **Gaussian Noise $\sigma = 2.0$** | 9 | 0 | **0.0%** | 42.14 dB | 0.99964 | 9.35% | Header Corrupted (9) | **`FAIL_CLOSED`** |
| **JPEG Quality 95** | 3 | 0 | **0.0%** | 36.43 dB | 0.99865 | 48.71% | Header Corrupted (3) | **`FAIL_CLOSED`** |
| **JPEG Quality 85** | 3 | 0 | **0.0%** | 33.74 dB | 0.99742 | 50.14% | Header Corrupted (3) | **`FAIL_CLOSED`** |
| **JPEG Quality 75** | 3 | 0 | **0.0%** | 32.18 dB | 0.99624 | 49.88% | Header Corrupted (3) | **`FAIL_CLOSED`** |

### 7.2 Operating Envelope and Channel Boundaries
1. **Lossless PNG Transmission:** Fully supported across all carriers. 100% exact recovery, 0 post-ECC errors.
2. **Low-Noise Additive Channels ($\sigma \le 0.5$):** Fully supported across multiple carriers and noise seeds (9/9 pass, PSNR $\ge 53$ dB, raw BER $\le 0.4\%$).
3. **Intermediate Noise ($\sigma = 1.0$):** Operating boundary condition. Recovery is **partial (33.33%, 3/9 pass)**; remaining trials fail closed safely via `CarrierHeaderError` or mask bounds.
4. **Severe Noise ($\sigma \ge 2.0$) & JPEG Compression (Q=75–95):** Unsupported. High-frequency Red-HH coefficients are destroyed by JPEG 8x8 DCT quantization (BER $\approx 49\%\text{--}50\%$), corrupting the header and triggering fail-closed `CarrierHeaderError` (100% safe rejection, zero plaintext leak).

---

## 8. Workstream E: ECC Deployment Configuration and Policy

### 8.1 Deployment Configuration Factory vs. Legacy Default

To maintain clean separation between legacy baseline pipelines and the qualified production deployment pipeline:

- **Legacy Default Configuration:**
  Calling `MediaStegoPipeline()` or `MediaStegoPipeline(backend="onnx")` without specifying `ecc_encoder` defaults to `ReedSolomonEncoder(parity_symbols=32)` (`rs_2t_32`). This maintains 100% backward compatibility with Sprint 0 and Phase 2 tests.
- **Qualified Deployment Configuration:**
  Calling `MediaStegoPipeline.create_qualified_deployment(backend="onnx")` explicitly selects `ReedSolomonEncoder(parity_symbols=48)` (`rs_2t_48`) backed by ONNX Runtime session pooling.
- **Receiver Interoperability:**
  Regardless of how the receiver pipeline is constructed (legacy default or qualified), it extracts `header.ecc_parity` from `CarrierHeader.flags` and dynamically adapts its internal Reed–Solomon decoder to the sender's profile.

### 8.2 Deployment Policy by Profile

1. **Primary Qualified Profile: `rs_2t_48` (Deployment Default)**
   - Configuration: RS(255, 207), $t = 24$, parity symbols $= 48$, code rate $= 81.18\%$.
   - Usable Capacity: **6,578 bytes raw** ($M=0$) / **6,576 bytes reference** ($M=2$).
   - Operational Boundary: 100.0% exact recovery across all evaluated payloads (100B to 5000B) and robust to Gaussian noise $\sigma \le 0.5$.
2. **Constrained High-Capacity Profile: `rs_2t_32` (Optional)**
   - Configuration: RS(255, 223), $t = 16$, parity symbols $= 32$, code rate $= 87.45\%$.
   - Usable Capacity: **7,090 bytes raw** ($M=0$) / **7,088 bytes reference** ($M=2$).
   - Operational Boundary: Maintained under a conservative operating restriction of payloads $\le 1,000$ B due to Poisson tail risk on 32 blocks.
3. **Maximum Theoretical Resilience Profile: `rs_2t_64` (Optional)**
   - Configuration: RS(255, 191), $t = 32$, parity symbols $= 64$, code rate $= 74.90\%$.
   - Usable Capacity: **6,066 bytes raw** ($M=0$) / **6,064 bytes reference** ($M=2$).
   - Operational Boundary: Verified 100% in lossless evaluation; empirical channel degradation under noise remains untested and unmeasured.

---

## 9. Verification Evidence and Safety Bounds

- **Full Repository Test Suite:** **359 passed in 43.76s** (100% PASS, 0 failures).
- **Phase 5B Targeted Unit Tests (`tests/unit/test_phase5b_onnx_pipeline.py`):** **23 / 23 passed in 10.84s**.
  - `test_failed_recovery_trial_blocks_lossless_gate`: Disagreement between raw trial and summary forces gate failure and blocks acceptance.
  - `test_capacity_gate_bypass_removed_fails_when_mandatory_checks_fail`: Confirms oversized rejection cannot bypass mandatory checks.
  - `test_qualified_deployment_factory_configuration`: Confirms factory creates `rs_2t_48` sender, legacy default remains `rs_2t_32`, and fresh receiver adapts.
  - `test_exact_capacity_boundaries_over_limit_rejection`: Confirms raw limit + 1 is rejected before carrier modification across all profiles.
  - `test_qualified_profiles_exact_boundary_recovery`: Confirms exact limit and one-below recovery on `rs_2t_48` and `rs_2t_64`.
  - `test_rs_2t_32_boundary_unreliability_empirical_finding`: Empirically preserves finding that `rs_2t_32` fails exact recovery at 7,090 B on natural validation carrier.
  - `test_phase5b_parity_report_json_consistency`: Confirms element-level distributions, counts, calculated percentages, and gate reasons agree between JSON and audit report.
- **Checkpoint Parameter Digest:** `ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92` verified invariant.
- **ONNX Model Hashes:** All 4 original Opset 14 models verified unchanged.
- **Canonical Dataset Splits:** 600 train / 100 val / 100 test verified; test split isolated.
- **Zero Training:** Exactly zero optimizer steps, zero epoch updates, zero parameter modifications.

---

## 10. Summary of Modified Files

1. [`scripts/experiments/run_phase5b_assessment.py`](file:///e:/Paladin_media/scripts/experiments/run_phase5b_assessment.py): Implemented strict raw-trial aggregation, removed capacity-gate bypass fallbacks, dynamic operating envelope synthesis, accurate `gate_reason`, and unrounded 6-decimal percentage precision.
2. [`scripts/experiments/check_phase5b_parity_consistency.py`](file:///e:/Paladin_media/scripts/experiments/check_phase5b_parity_consistency.py): Lightweight consistency verifier ensuring element counts, percentages, and gate reasons agree between JSON and audit report.
3. [`src/core/media_pipeline.py`](file:///e:/Paladin_media/src/core/media_pipeline.py): Added `MediaStegoPipeline.create_qualified_deployment()` factory configuring `rs_2t_48` with ONNX backend while preserving legacy default (`rs_2t_32`), and exact metadata length preflight check.
4. [`src/carrier/capacity.py`](file:///e:/Paladin_media/src/carrier/capacity.py): Added `reference_metadata_capacity_bytes` to `CapacityReport` and `evaluate()`, separating raw capacity ($M=0$) from reference capacity ($M=2$).
5. [`tests/unit/test_phase5b_onnx_pipeline.py`](file:///e:/Paladin_media/tests/unit/test_phase5b_onnx_pipeline.py): Expanded to 23 focused tests verifying raw-trial aggregation, capacity bypass removal, deployment factory, boundary recovery, empirical unreliability of `rs_2t_32` at raw limit, and JSON-report parity consistency.
6. [`docs/experiments/phase5b_results.json`](file:///e:/Paladin_media/docs/experiments/phase5b_results.json): Structured records with raw/reference capacities, exact boundary outcomes, corrected `protected_payload_bytes`, unrounded `99.999991%` 5e-5 percentage, explicit `gate_reason`, and dynamic gate statuses.
7. [`docs/audits/PHASE5B_ONNX_INTEGRATION_AND_ROBUSTNESS_REPORT.md`](file:///e:/Paladin_media/docs/audits/PHASE5B_ONNX_INTEGRATION_AND_ROBUSTNESS_REPORT.md): Reconciled comprehensive audit report agreeing verbatim with measured JSON telemetry.
