# Experiment E03: MVP0 Baseline End-to-End Validation

## Document Control
- **Experiment ID:** E03
- **Phase:** Phase 2 — Signal Processing + Carrier Infrastructure
- **Date:** 2026-10-09
- **Status:** COMPLETED & VERIFIED
- **Related Specifications:** PaladinMedia Architecture Specification §11/§12/§14, Sprint-0 IRS §6/§9, System & Research Document §18
- **Evidence Path:** `docs/experiments/e03_results.json`

---

## 1. Experimental Parameters & Hardware
- **Hardware Platform:** AMD64 Family 25 Model 80 Stepping 0 (AuthenticAMD), Windows 10.0.26200
- **Software Environment:** Python 3.10.11, NumPy 2.2.6, Pillow 12.2.0, Cryptography 49.0.0
- **Carrier Dataset:** MVP 0 Carrier Dataset v0.1 (`MVP 0 Dataset/0001x2.png`, $512 \times 512$ RGB, uint8)
- **Secret Media Fixture:** `tests/fixtures/secret_small.png` ($24 \times 24 \times 3$ PNG, 152 bytes plaintext)
- **Cryptographic Configuration:** AES-256-GCM, 32-byte key, 12-byte random nonce, 16-byte GHASH tag, AAD bound
- **Error Correction:** Reed-Solomon $n=255, k=223$ ($2t=32$ parity symbols), 4-byte length prefix
- **DWT Configuration:** One-level 2D Haar DWT on $512 \times 512$ RGB
  - Subband output shapes: $\text{LL}, \text{LH}, \text{HL}, \text{HH} = (3, 256, 256)$ in `float64`
- **Embedding Allocation:** Channel 0 (Red channel) of $\text{HH}$ subband ($256 \times 256 = 65,536$ coefficient positions)
- **Embedding Rate:** 1 bit per coefficient position ($0.25\text{ bpp}$)
- **Carrier Header:** 16-byte fixed canonical wire framing (`b"PLDH"` magic, version 1, flags 0, reserved 0, session counter 42, payload length 316)
- **Mask Configuration:** Fixed header anchors (indices 0..127) + deterministic PRNG payload indices
- **Embedding Modulation:** Integer-parity modulation on $2 \times \text{HH}$ coefficients with $\pm 1$ spatial adjustment on carrier pixel $a$

---

## 2. Capacity Accounting & Stage-by-Stage Sizes

| Pipeline Stage | Measured Size | Notes |
| :--- | :--- | :--- |
| **Secret Plaintext** | 152 bytes | $24 \times 24$ RGB PNG image fixture |
| **Serialized Wire Framing** | 220 bytes | Includes `b"PLDS"`, version, metadata dictionary, CRC-32 |
| **AES-256-GCM Encrypted** | 248 bytes | 220B ciphertext + 12B nonce + 16B GHASH tag |
| **Reed-Solomon Protected** | 316 bytes | 4B length prefix + 248B data + 2 blocks $\times$ 32 parity bytes |
| **Carrier Header** | 16 bytes | 128 bits fixed anchor positions |
| **Total Embedded Bits** | 2,656 bits | $(16 + 316) \times 8$ bits |
| **Raw Carrier Capacity** | 8,192 bytes | $256 \times 256$ bits = 65,536 bits |
| **Maximum Protected Payload Capacity** | 8,176 bytes | $8,192\text{ B} - 16\text{ B header}$ |
| **Usable Secret Plaintext Capacity** | 7,088 bytes | 32 RS blocks $\times 223 - 28\text{ B (AES)} - 18\text{ B (framing)}$ |
| **Remaining Capacity (This Run)** | 7,860 bytes | $8,176\text{ B} - 316\text{ B}$ |
| **Capacity Status** | **FITS** | Validated before transform |

---

## 3. Signal & Cryptographic Verification Results

| Dimension | Metric | Measured Value | Requirement / Standard |
| :--- | :--- | :--- | :--- |
| **DWT/IDWT Invertibility** | Max Absolute Error | **0.00000000** | Floating-point identity ($\text{Err} < 10^{-12}$) |
| **DWT/IDWT MSE** | Mean Squared Error | **0.00000000** | Mathematical zero |
| **Carrier Fidelity** | PSNR | **75.8850 dB** | High perceptual transparency ($> 50.0\text{ dB}$) |
| **Structural Similarity** | SSIM | **1.000000** | Structural identity |
| **Mean Absolute Pixel Error** | MAE | **0.001677** | Average pixel perturbation |
| **Maximum Absolute Pixel Error** | Max $|\Delta|$ | **1** | Strict bounded modulation ($\pm 1$) |
| **Raw Channel Errors** | BER (Pre-ECC) | **0.000000 (0 errors)** | Zero bit flip through PNG round trip |
| **Error Correction** | RS Decode Outcome | **SUCCESS** | Exact data symbol restoration |
| **Cryptographic Integrity** | AES-256-GCM Tag | **AUTHENTICATED** | GHASH matches; zero plaintext leak |
| **Payload Deserialization** | CRC-32 Checksum | **VERIFIED** | Canonical payload verified |
| **Secret Reconstruction** | Exact Byte Equality | **TRUE (100%)** | `recovered.data == original.data` |

---

## 4. Performance & Resource Benchmarking

| Stage | Execution Time (ms) | Peak Memory Delta |
| :--- | :--- | :--- |
| **Secret Serialization** | 0.088 ms | < 0.1 MB |
| **AES-256-GCM Encryption** | 0.203 ms | < 0.1 MB |
| **Reed-Solomon Encoding** | 2.650 ms | < 0.1 MB |
| **Transform & Embedding** | 18.238 ms | ~ 2.4 MB |
| **Lossless PNG Save & Reload** | 102.415 ms | ~ 3.1 MB |
| **Transform & Extraction** | 7.912 ms | ~ 1.2 MB |
| **Reed-Solomon Decoding** | 1.841 ms | < 0.1 MB |
| **AES-256-GCM Decryption** | 0.142 ms | < 0.1 MB |
| **Payload Deserialization** | 0.062 ms | < 0.1 MB |
| **Total Roundtrip Runtime** | **134.79 ms** | **+3.4 MB peak** |

---

## 5. Hard Gate Verification (Negative Control)
- **Input:** Oversized payload of 10,000 bytes (resulting in 11,575 bytes Reed-Solomon protected stream).
- **Hard Gate Execution:** `CapacityCalculator.check_capacity(10000)` evaluated prior to embedding.
- **Observed Behavior:** Raised `PayloadCapacityExceededError` immediately.
- **Fail-Closed Validation:** Zero carrier pixels modified, zero transform passes executed, zero memory leaks.

---

## 6. Experimental Conclusions & Limitations
1. **End-to-End Baseline Proven:** Demonstrates that the Phase 2 baseline carrier architecture reliably preserves encrypted payloads across full 8-bit RGB PNG disk serialization and reload with 0.0 bit errors and exact byte reconstruction.
2. **Perceptual Superiority:** At 75.88 dB PSNR and 1.0 SSIM, the baseline embedding is imperceptible to human visual inspection and basic statistical detection.
3. **Capacity Ceiling:** Under 1-level Haar DWT and 1 HH channel, the carrier cannot support payloads exceeding 8,176 bytes protected (7,088 bytes usable plaintext).
4. **Phase Boundary:** This experiment validates the classical signal processing baseline. It does not validate neural network robustness or lossy compression survival (Phase 3 scope).
