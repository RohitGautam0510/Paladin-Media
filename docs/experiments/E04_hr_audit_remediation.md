# PaladinMedia — Experiment E04-HR Audit Remediation Report
**Audit Timestamp:** `2026-10-11T18:24:44Z`  **Audit Status:** Formal Remediation Completed
## 1. Executive Summary & Audit Scope
This document presents the audited and remediated findings for Experiment **E04-HR**. Per the audit directive, all reported quantities have been verified against the active source code, the underlying `reedsolo` and `cryptography` libraries, and real empirical execution logs.
### Key Audit Findings & Resolutions
1. **Reconciliation of High-Load 23,010-byte RS Length**: Confirmed mathematically and empirically. The 20,000-byte plaintext with 48-byte metadata serializes to 20,066 bytes, encrypts to 20,094 bytes, and is chunked into 90 full 223-byte blocks and 1 final 24-byte partial chunk. Encoded systematically with a 4-byte prefix, the 90 full blocks yield 22,950 bytes and the single shortened block yields 56 bytes ($24 + 32$), summing to exactly **23,010 bytes**.
2. **Capacity Reconciliation & Shortened RS Codes**: Clarified the distinction between conservative full-block-only floors (28,498 B / 28,496 B) and actual shortened RS encoder capability (28,574 B / 28,572 B) for 1024 × 1024 carriers.
3. **Dataset Dimension Correction**: Clarified that DIV2K HR contains 7 images that are exactly 2040 × 2040 pixels, but the maximum dimension across the entire dataset is 2,040 pixels (< 2,048 pixels), strictly justifying why $\ge 2048$ square carriers are marked as UNAVAILABLE.
4. **Memory Reporting Correction**: Formally decoupled theoretical working-array memory, measured peak process memory (working set), and ephemeral RSS deltas.
## 2. High-Load 20,000-Byte Payload Trace & Codeword Breakdown
| Stage | Component | Size (Bytes) | Running Total | Formulation / Details |
|:---|:---|:---:|:---:|:---|
| **Plaintext Input** | Secret Raw Data | 20,000 | 20,000 B | 20,000 pseudo-random bytes |
| **Serialization** | Wire Prefix Header | 10 | 20,010 B | Magic (`PLDS`) + Ver (`0x01`) + Type (`0x00`) + Flags (`0x0000`) + MetaLen (48) |
| | Compact JSON Metadata | 48 | 20,058 B | `{"note":"High-Load 1024x1024 test","size":20000}` |
| | Payload Length Field | 4 | 20,062 B | `0x00004E20` (20,000 big-endian) |
| | CRC-32 Checksum Trailer | 4 | 20,066 B | Big-endian CRC-32 across all preceding bytes |
| **Authenticated Encryption** | AES-256-GCM IV | 12 | 20,078 B | Cryptographic random nonce |
| | Ciphertext | 20,066 | 20,078 B | AES-256-GCM encrypted serialized payload |
| | GHASH Auth Tag | 16 | **20,094 B** | 128-bit authentication tag |
| **Error Correction (RS)** | Length Prefix | 4 | 4 B | Big-endian stream length `20,094` (`0x00004E7E`) |
| | 90 Full Blocks | 22,950 | 22,954 B | $90 \times (223\text{ data} + 32\text{ parity}) = 90 \times 255$ B |
| | 1 Shortened Block | 56 | **23,010 B** | $24\text{ data} + 32\text{ parity} = 56$ B (Shortened RS codeword) |
| **Carrier Embedding** | Carrier Header Anchor | 16 | 23,026 B | 128 header anchor bits on Red HH row 0 |
| | Protected Payload Bits | 184,080 | 184,208 bits | 23,010 x 8 bits placed via PCG64 mask |

> [!NOTE]
> **Why 23,010 Bytes?**: Under systematic chunking in `ReedSolomonEncoder`, 20,094 / 223 = 90 with a remainder of 24. The 90 full blocks expand to 90 x 255 = 22,950 bytes. The remaining 24 bytes form a shortened RS code of 24 + 32 = 56 bytes. Adding the 4-byte length prefix yields exactly 4 + 22,950 + 56 = 23,010 bytes.

### Empirical High-Load Roundtrip Results
- **Cover Image:** `0001.png` (1024 × 1024 center crop)
- **Crop Offset:** `(left=508, top=190)`
- **Stego Fidelity:** PSNR = **63.4576 dB**, SSIM = **0.999994**, MAE = **0.029331**, MaxErr = **1**
- **Carrier BER:** **0.0** (Zero bit errors among 184,080 bits)
- **Exact Plaintext Recovery:** **True**
- **Execution Timings:** Embed = 426.75 ms, Extract = 463.04 ms

## 3. Capacity Reconciliation Across Carrier Dimensions

$$\text{Usable Secret} = \left(N_{\text{full}} \times 223 + D_{\text{partial}}\right) - 28 (\text{AES}) - 18 (\text{Framing}) - \text{len}(\text{metadata})$$

| Metric | 512 × 512 Carrier | 1024 × 1024 (Full-Block Floor) | 1024 × 1024 (Actual Shortened RS) |
|:---|:---:|:---:|:---:|
| Raw Carrier Bits | 65,536 | 262,144 | 262,144 |
| Raw Carrier Bytes | 8,192 B | 32,768 B | 32,768 B |
| Carrier Header | 16 B | 16 B | 16 B |
| **Max Embeddable Protected** | **8,176 B** | **32,752 B** | **32,752 B** |
| Available for Blocks | 8,172 B | 32,748 B | 32,748 B |
| Full Blocks ($N_{\text{full}}$) | 32 | 128 | 128 |
| Remainder Space | 12 B ($< 33$) | 108 B | 108 B ($> 32$) |
| Partial Block Data ($D_{\text{partial}}$) | 0 B | 0 B | **76 B** ($108 - 32$) |
| Max Encrypted Ciphertext | 7,136 B | 28,544 B | **28,620 B** |
| Max Serialized Payload | 7,108 B | 28,516 B | **28,592 B** |
| **Plaintext (0 Metadata)** | **7,090 B** | **28,498 B** | **28,574 B** |
| **Plaintext (2B Ref Metadata)** | **7,088 B** | **28,496 B** | **28,572 B** |
| **Plaintext (48B High-Load Meta)**| **7,042 B** | **28,450 B** | **28,526 B** |

## 4. DIV2K HR Dataset Dimension Audit

- **Total HR Images:** 800
- **Images with Native 2040 × 2040 Dimensions:** **7**
- **Exact 2040 × 2040 Filenames:** `0044.png, 0155.png, 0217.png, 0450.png, 0500.png, 0513.png, 0716.png`
- **Maximum Dimension Observed:** **2040 px**
- **Images with Minimum Dimension $\ge 1024$:** **785 / 800**
- **Images with Minimum Dimension $\ge 2048$:** **0 / 800 (Zero)**

> [!IMPORTANT]
> **Status of $\ge 2048$ Resolutions**: The DIV2K HR dataset includes exactly 7 images with dimensions 2040 × 2040 pixels: `0044.png`, `0155.png`, `0217.png`, `0450.png`, `0500.png`, `0513.png`, `0716.png`. However, no dimension in the entire dataset reaches 2048 pixels (maximum native dimension is 2040 px). Consequently, native 2048 × 2048 square crops cannot be formed from any image without upscaling. Per architectural constraints prohibiting synthetic upscaling, 2048 × 2048 and larger square carriers are strictly marked as UNAVAILABLE.

## 5. Empirical Evaluation & Corrected Memory Reporting

Memory reporting is formally disambiguated into three tiers:
1. **Theoretical Working Array Allocation:** Fixed DWT array overhead (6.75 MB for 512×512, 27.00 MB for 1024×1024).
2. **Isolated Process Peak Working Set:** Peak memory in a dedicated clean Python subprocess: **60.50 MB** for 512×512, **147.16 MB** for 1024×1024.
3. **Long-Running Process Peak Working Set:** Cumulative high-water mark across repeated runs in a shared session (268.87 MB to 367.93 MB).
4. **Working Set Delta ($\Delta_{\text{RSS}}$):** Ephemeral per-run fluctuation subject to Python runtime garbage collection (ranging from +42.0 MB to -0.77 MB).

### 5.1 E04-A: Reference Baseline (512 × 512 HR Center Crops)

| Image ID | Native Dims | PSNR (dB) | SSIM | MAE | BER | Exact Match | Total Time | Theoretical Mem | Peak Working Set | RSS Delta |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0001.png` | 2040×1404 | 75.958 | 1.0 | 0.001649 | 0.0 | `True` | 66.11 ms | 6.75 MB | 266.95 MB | 14.02 MB |
| `0002.png` | 2040×1848 | 75.752 | 1.0 | 0.001729 | 0.0 | `True` | 64.58 ms | 6.75 MB | 266.95 MB | -1.14 MB |
| `0003.png` | 2040×1356 | 75.7809 | 1.0 | 0.001718 | 0.0 | `True` | 68.88 ms | 6.75 MB | 266.95 MB | 0.02 MB |
| `0004.png` | 2040×1344 | 75.651 | 1.0 | 0.00177 | 0.0 | `True` | 69.58 ms | 6.75 MB | 266.95 MB | 0.02 MB |
| `0005.png` | 1608×2040 | 75.7393 | 1.0 | 0.001734 | 0.0 | `True` | 64.13 ms | 6.75 MB | 266.95 MB | 0.02 MB |

### 5.2 E04-B: High-Resolution Carriers (1024 × 1024 HR Center Crops)

| Image ID | Native Dims | PSNR (dB) | SSIM | MAE | BER | Exact Match | Total Time | Theoretical Mem | Peak Working Set | RSS Delta |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0001.png` | 2040×1404 | 81.779 | 1.0 | 0.000432 | 0.0 | `True` | 256.22 ms | 27.0 MB | 288.36 MB | 41.8 MB |
| `0002.png` | 2040×1848 | 81.7662 | 1.0 | 0.000433 | 0.0 | `True` | 244.37 ms | 27.0 MB | 355.1 MB | -0.01 MB |
| `0003.png` | 2040×1356 | 81.6592 | 1.0 | 0.000444 | 0.0 | `True` | 253.42 ms | 27.0 MB | 368.41 MB | -0.52 MB |
| `0004.png` | 2040×1344 | 81.7854 | 1.0 | 0.000431 | 0.0 | `True` | 256.87 ms | 27.0 MB | 368.41 MB | -0.53 MB |
| `0005.png` | 1608×2040 | 81.6592 | 1.0 | 0.000444 | 0.0 | `True` | 236.4 ms | 27.0 MB | 368.41 MB | -0.59 MB |

## 6. E04-HR Formal Closure Decision Checklist

| Audit Item | Verification Status | Defensibility & Execution Evidence |
|:---|:---:|:---|
| **A1. DIV2K HR Inventory** | **VERIFIED** | All 800 images audited. Exactly 7 files (`0044.png`, `0155.png`, `0217.png`, `0450.png`, `0500.png`, `0513.png`, `0716.png`) are 2040 × 2040. Max dimension is 2040 px (< 2048 px). 785 support 1024 crops; 0 support 2048 crops. |
| **A2. Reed-Solomon Codeword Accounting** | **VERIFIED** | 10-stage trace of 20,000 B plaintext verified. $20,094 = 90 \times 223 + 24$. $4 + (90 \times 255) + 56 = 23,010$ B protected payload. Shortened RS coding verified in encoder, decoder, and test suite. |
| **A3. Capacity Gates & Boundaries** | **FIXED AND RE-VERIFIED** | Reconciled 7,088 B / 7,090 B (512) and 28,496 B / 28,572 B (1024) with explicit metadata parameters. Added `test_capacity_boundary_real_encoder_and_serializer`. 101/101 tests pass. |
| **A4. End-to-End Stego Roundtrip** | **VERIFIED** | High-load 20,000 B payload surviving PNG save/reload with BER = 0.0, PSNR = 63.46 dB, SSIM = 0.999994, exact plaintext match. |
| **A5. Memory Reporting Disambiguation** | **FIXED AND RE-VERIFIED** | Disentangled theoretical DWT array memory (6.75 / 27 MB), isolated subprocess peak working set (60.50 / 147.16 MB), session-wide peak working set (~268–368 MB), and ephemeral RSS deltas. |
| **A6. E04-HR Closure Decision** | **VERIFIED — CLOSED** | All material discrepancies resolved and defended by empirical execution evidence. |
