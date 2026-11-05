# Failure Case Analysis: 0715.png, 0774.png, and 0784.png

**Audit Date:** 2026-10-29  
**Auditor:** Senior Forensic Reliability Auditor & Cryptographic Engineer  
**Status:** FULLY RECONCILED & EMPIRICALLY GROUNDED  
**Target Repository:** `PaladinMedia` (`E:\Paladin_media`)  

---

## 1. Executive Summary

Across the canonical 100-carrier held-out test split (`docs/project/div2k_hr_split.json`), the Wire Protocol v2 evaluation achieved **97% exact byte-for-byte recovery** (97/100), improving upon the 95% baseline recovery under Wire Protocol v1.

Three carriers failed to achieve end-to-end recovery:
1. **`0715.png`** (Carrier ID: 715) — **Persistent Header Failure**
2. **`0774.png`** (Carrier ID: 774) — **Header Corrected / Payload RS Failure**
3. **`0784.png`** (Carrier ID: 784) — **Header Clean / Marginal Payload RS Failure**

This document provides a forensic post-mortem for each failure case, distinguishes recorded empirical facts from inferred hypotheses, and verifies that the pipeline adheres to the fail-closed security invariant under all failure modes.

---

## 2. Failure Case Post-Mortems

### 2.1 Carrier 0715.png — Uncorrectable Header Damage

- **Carrier Metadata:** `0715.png`, 512×512 RGB center-crop of DIV2K HR carrier 0715.
- **Session Counter:** `0x000002CB` (715)
- **Visual Fidelity:** PSNR: `53.61 dB`, SSIM: `0.999967`
- **Failure Stage:** `HEADER_UNPACK` ([`src/carrier/header.py:127-145`](file:///E:/Paladin_media/src/carrier/header.py#L127-L145))
- **Exception Raised:** `CarrierHeaderError: Reed-Solomon decoding failed: uncorrectable error`

#### Recorded Empirical Evidence
1. **Raw Header Error Count:** The extracted 192-bit header stream contained **11 bit errors** distributed across **10 distinct bytes** (out of 24 total codeword bytes).
2. **Codec Budget:** The shortened RS(24, 16) code over $\text{GF}(2^8)$ has $2t = 8$ parity symbols, conferring an error correction capability of $t = \lfloor 8 / 2 \rfloor = 4$ byte errors.
3. **Syndrome Evaluation:** The 10 corrupted symbol bytes far exceeded the 4-byte budget ($10 > 4$), triggering an uncorrectable syndrome in `RSCodec(8).decode()`.
4. **Baseline Comparison:** In the baseline v1 held-out evaluation (`held_out_test_results_20261029.json`), `0715.png` failed with `CarrierHeaderError: Unsupported ECC profile in carrier header flags: 0x05` due to corrupted flag bits.

#### Determinism Assessment
- **Status:** **100% Deterministic Failure.**
- **Rationale:** Header anchor coordinates `(0, 0)` through `(0, 191)` are static and non-randomized. The carrier image crop at `0715.png` contains steep diagonal high-frequency textures in the upper-left boundary, inducing severe transform-domain quantization and INN residual distortion. Every execution of `0715.png` with this carrier produces the identical 10-byte header corruption.

---

### 2.2 Carrier 0774.png — Header Successfully Corrected; Downstream Payload Failure

- **Carrier Metadata:** `0774.png`, 512×512 RGB center-crop of DIV2K HR carrier 0774.
- **Session Counter:** `0x00000306` (774)
- **Visual Fidelity:** PSNR: `54.89 dB`, SSIM: `0.999965`
- **Failure Stage:** `PAYLOAD_RS_DECODE` ([`src/carrier/extraction.py:175-189`](file:///E:/Paladin_media/src/carrier/extraction.py#L175-L189))
- **Exception Raised:** `CarrierExtractionError: Reed-Solomon decoding failed in block 5/6: uncorrectable symbol errors exceed capacity`

#### Recorded Empirical Evidence
1. **Header Error Correction:** The raw extracted header contained **7 bit errors distributed across 4 distinct bytes**.
2. **RS(24, 16) Header Recovery:** Because the 4 corrupted bytes were within the $t=4$ budget ($4 \le 4$), `RSCodec(8).decode()` successfully corrected all 4 bytes.
3. **Header Unpacking:** The header unpacked cleanly:
   - `magic = b"PLDH"`
   - `version = 0x02`
   - `session_counter = 774`
   - `payload_length = 1338` bytes
4. **Payload Decoding Failure:** Downstream, the 1,338-byte payload was partitioned into 6 Reed–Solomon RS(255, 207) blocks. Blocks 1, 2, 3, 4, and 6 decoded successfully, but block 5 accumulated more than 24 symbol errors ($> 24$), aborting extraction.
5. **Baseline Comparison:** In baseline v1, `0774.png` failed immediately at header unpack because the 7 uncorrected bit errors altered the `payload_length` field to an impossible value ($41,954,280$ bits), causing `MaskGenerationError`. Wire Protocol v2 successfully eliminated that header failure and allowed extraction to proceed into the payload.

#### Determinism Assessment
- **Status:** **Deterministic Header Recovery; Near-Deterministic Payload Failure.**
- **Rationale:** Header recovery is strictly deterministic. The payload failure occurs in block 5 where channel noise exceeds the $t=24$ RS capacity.

---

### 2.3 Carrier 0784.png — Clean Header; Marginal Payload RS Failure

- **Carrier Metadata:** `0784.png`, 512×512 RGB center-crop of DIV2K HR carrier 0784.
- **Session Counter:** `0x00000310` (784)
- **Visual Fidelity:** PSNR: `52.47 dB`, SSIM: `0.999957`
- **Failure Stage:** `PAYLOAD_RS_DECODE` ([`src/carrier/extraction.py:175-189`](file:///E:/Paladin_media/src/carrier/extraction.py#L175-L189))
- **Exception Raised:** `CarrierExtractionError: Reed-Solomon decoding failed in block 4/6: uncorrectable symbol errors exceed capacity`

#### Recorded Empirical Evidence
1. **Header Status:** The extracted header had **0 bit errors and 0 byte errors**. Header unpacking succeeded cleanly with `header_success: True`.
2. **Payload Decoding Failure:** During payload RS decoding, block 4 accumulated uncorrectable symbol errors ($> 24$), causing `CarrierExtractionError`.
3. **Baseline Comparison:**
   - In the baseline v1 held-out evaluation (`held_out_test_results_20261029.json`), `0784.png` had **passed** (`receiver_exact_byte_match: True`).
   - However, baseline records show that `0784.png` was on the verge of failure:
     - Baseline errors per block: `[10, 15, 6, 10, 12, 6]`
     - Max codeword error: **15 symbol errors** (out of max allowed 24).
     - Margin was only 9 symbols under baseline v1.

#### Root Cause of the v1 vs v2 Discrepancy on 0784.png
The difference between v1 passing and v2 failing on `0784.png` is explained by two concrete architectural differences:
1. **Payload Coordinate Pool Shift:** In v1, the header reserved 128 anchor bits (indices 0..127), and payload positions were sampled from `range(128, 65536)`. In v2, the header reserves 192 anchor bits (indices 0..191), and payload positions are sampled from `range(192, 65536)`. Because `random.Random(seed).sample()` operates over a different index space, completely different 2D coordinates were selected for the payload bits.
2. **AES-GCM Nonce Randomization:** In each test run, `os.urandom(12)` generates a fresh IV, causing different bit patterns to be embedded at each coordinate. On carrier `0784.png`, which exhibits high variance across spatial blocks, the v2 coordinates and ciphertext bit pattern pushed block 4 from 15 symbol errors to $> 24$.

#### Determinism Assessment
- **Status:** **Stochastically Marginal.**
- **Rationale:** Carrier `0784.png` is near the edge of the channel capacity boundary. Depending on the random AES-GCM IV and resulting ciphertext bit patterns, its worst-case RS block fluctuates around the 24-symbol threshold.

---

## 3. Fail-Closed Security Verification Across All Failure Modes

A core requirement of the PaladinMedia security architecture is the **fail-closed invariant**:
> Under no circumstance may the extraction pipeline return unauthenticated, partial, or corrupted plaintext to the caller.

| Failure Mode | Carrier | Stage Aborted | Exception Raised | Plaintext Leaked? | Cryptographic Verification |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Header Damage ($> 4$ bytes)** | `0715.png` | `HEADER_UNPACK` | `CarrierHeaderError` | **None** (Aborted before payload extraction) | Passed closed |
| **Payload Damage ($> 24$ symbols)** | `0774.png` | `PAYLOAD_RS_DECODE` | `CarrierExtractionError` | **None** (Aborted before AES decryption) | Passed closed |
| **Payload Damage ($> 24$ symbols)** | `0784.png` | `PAYLOAD_RS_DECODE` | `CarrierExtractionError` | **None** (Aborted before AES decryption) | Passed closed |

In all three failure cases:
1. `exact_match` was recorded as `False`.
2. `aes_decrypt_success` was recorded as `False`.
3. No raw or corrupted byte streams were delivered to application callers.
4. The system failed closed cleanly with strict exception propagation.
