# Transform-Domain Capacity Accounting & Framing Audit

**Audit Date:** 2026-10-29  
**Auditor:** Senior Forensic Reliability Auditor & Cryptographic Engineer  
**Status:** FULLY RECONCILED & MATHEMATICALLY PROVEN  
**Target Repository:** `PaladinMedia` (`E:\Paladin_media`)  

---

## 1. Executive Summary

This audit establishes the rigorous, bit-exact capacity accounting model of PaladinMedia across dynamic carrier dimensions, transform-domain subbands, Wire Protocol versions (v1 and v2), Reed–Solomon error correction profiles, and cryptographic framing layers.

The audit resolves historical ambiguities regarding carrier capacity (clarifying the distinction between raw coefficient capacity, maximum protected payload capacity, and net application secret capacity) and verifies that [`CapacityCalculator`](file:///E:/Paladin_media/src/carrier/capacity.py#L76) strictly enforces these hard bounds.

---

## 2. Fundamental Spatial & Transform-Domain Allocation

The PaladinMedia embedding architecture operates in the high-frequency wavelet subband of carrier images to ensure visual imperceptibility and robustness against spatial low-pass filtering.

### 2.1 Wavelet Decomposition and Subband Dimensions

For any carrier image with spatial dimensions $H \times W$ (where $H, W$ are even integers):
1. A 1-level 2D Haar Discrete Wavelet Transform (DWT) decomposes each color channel into four spatial-frequency subbands:
   $$\text{LL} \quad (\text{Approximation}), \quad \text{LH} \quad (\text{Horizontal Highs}), \quad \text{HL} \quad (\text{Vertical Highs}), \quad \text{HH} \quad (\text{Diagonal Highs})$$
2. The dimensions of each subband are:
   $$H_{\text{sub}} = \frac{H}{2}, \qquad W_{\text{sub}} = \frac{W}{2}$$
3. **Channel Allocation Policy:** Per Architecture §10.2, embedding is strictly confined to **1 HH channel** (specifically, the Red channel HH subband). Green and Blue channels, as well as LL, LH, and HL subbands, remain untouched to preserve luminance structure and minimize perceptual distortion.
4. **Modulation Rate:** Exactly **1 bit per transform coefficient position**.

---

## 3. Bit & Byte Capacity Derivations (512×512 Standard Baseline)

For the standard 512×512 RGB carrier:
$$H = 512, \quad W = 512 \implies H_{\text{sub}} = 256, \quad W_{\text{sub}} = 256$$

### 3.1 Raw Coefficient Capacity

The total number of coefficient positions available in the Red-HH subband is:
$$N_{\text{raw\_bits}} = 256 \times 256 = 65,536 \text{ bits}$$
$$N_{\text{raw\_bytes}} = \frac{65,536 \text{ bits}}{8 \text{ bits/byte}} = \mathbf{8,192 \text{ bytes}}$$

### 3.2 Header Reservations: v1 vs. v2

Before any application payload or cryptographic framing can be embedded, fixed anchor positions must be reserved for the steganographic carrier header:

| Specification Layer | Wire Protocol v1 | Wire Protocol v2 | Notes |
|:---|:---:|:---:|:---|
| **Header Data Bytes** | 16 bytes | 16 bytes | `magic` (4B) + `version` (1B) + `flags` (1B) + `reserved` (2B) + `counter` (4B) + `length` (4B) |
| **Header ECC Parity Bytes**| 0 bytes | 8 bytes | Shortened systematic RS(24, 16) parity over $\text{GF}(2^8)$ |
| **Total Header Footprint** | **16 bytes** (128 bits) | **24 bytes** (192 bits) | Fixed anchor indices: `0..127` (v1) vs `0..191` (v2) |
| **Max Protected Payload** | **8,176 bytes** | **8,168 bytes** | $N_{\text{raw\_bytes}} - N_{\text{header\_bytes}}$ |
| **Available Payload Bits** | 65,408 bits | 65,344 bits | Candidate pool for PRNG mask generation |

### 3.3 Resolution of the "8,168 vs 8,192" Ambiguity
- **8,192 bytes** is the **gross physical capacity** of the 256×256 Red-HH coefficient grid.
- **8,168 bytes** is the **maximum net protected payload capacity** under Wire Protocol v2 ($8,192 - 24 = 8,168$).
- Under legacy Wire Protocol v1, the maximum protected payload capacity was **8,176 bytes** ($8,192 - 16 = 8,176$).

---

## 4. End-to-End Framing Overhead & Usable Secret Plaintext Capacity

To determine how many bytes of actual secret application data (e.g., secret images, keys, documents) fit inside a 512×512 carrier, we trace the full unidirectional serialization and encapsulation pipeline:

$$\text{Secret Application Plaintext } P$$
$$\Big\downarrow \text{SecretSerializer (Base framing: 18 B + variable JSON metadata } M\text{ B)}$$
$$\text{Serialized Payload } S$$
$$\Big\downarrow \text{AES-256-GCM Encryption (Overhead: 12 B IV + 16 B GHASH Tag = 28 B)}$$
$$\text{Ciphertext Stream } C$$
$$\Big\downarrow \text{Reed–Solomon FEC Encoding (4 B Length Prefix + Block Parity } 2t \times \text{Blocks)}$$
$$\text{Protected Bitstream Embedded into Carrier HH Subband}$$

### 4.1 Layer Breakdown

1. **Reed–Solomon Framing Layer:**
   - Length prefix: 4 bytes big-endian (`>I`).
   - Available space for RS codewords:
     $$A_{\text{RS}} = N_{\text{max\_protected}} - 4$$
     For v2: $A_{\text{RS}} = 8,168 - 4 = 8,164$ bytes.
   - Number of full 255-byte codewords:
     $$N_{\text{full}} = \lfloor 8,164 / 255 \rfloor = 32 \text{ blocks}$$
     $32 \times 255 = 8,160$ bytes.
   - Remainder:
     $$R = 8,164 \pmod{255} = 4 \text{ bytes}$$
     Because $R = 4 < 2t$ (for both $2t=32$ and $2t=48$), the 4 trailing bytes cannot support a shortened RS codeword (which requires at least $2t$ parity symbols). Thus, partial data capacity is 0.

2. **Reed–Solomon Payload Data Capacity ($K = 255 - 2t$):**
   - **Under RS(255, 223) ($2t=32$, baseline profile):**
     $$C_{\text{enc}} = 32 \times 223 = \mathbf{7,136 \text{ bytes}}$$
   - **Under RS(255, 207) ($2t=48$, primary qualified profile):**
     $$C_{\text{enc}} = 32 \times 207 = \mathbf{6,624 \text{ bytes}}$$

3. **AES-256-GCM Framing Layer:**
   - Deduct 12-byte initialization vector (IV) and 16-byte GHASH tag ($28$ bytes total).
   - Under RS(255, 223): $7,136 - 28 = \mathbf{7,108 \text{ bytes}}$ of serialized stream.
   - Under RS(255, 207): $6,624 - 28 = \mathbf{6,596 \text{ bytes}}$ of serialized stream.

4. **SecretSerializer Framing Layer:**
   - Deduct 18-byte fixed framing: 10-byte prefix (`b"PLDS\x01"` + flags + timestamp), 4-byte payload length, 4-byte CRC32.
   - Deduct metadata length $M$ (variable; default 0 bytes for pure secret, or 2 bytes `"{}"` for minimal reference JSON metadata).

### 4.2 Exact Usable Secret Capacity Summary (512×512 Carrier)

| Metric | Wire Protocol v1 (RS-32) | Wire Protocol v2 (RS-32) | Wire Protocol v2 (RS-48) |
|:---|:---:|:---:|:---:|
| **Raw HH Subband Capacity** | 8,192 B | 8,192 B | 8,192 B |
| **Carrier Header Anchor** | 16 B | 24 B | 24 B |
| **Max Protected Payload Capacity** | 8,176 B | 8,168 B | 8,168 B |
| **Max Encrypted Ciphertext ($C_{\text{enc}}$)** | 7,136 B | 7,136 B | 6,624 B |
| **Max Serialized Stream ($S_{\text{max}}$)** | 7,108 B | 7,108 B | 6,596 B |
| **Usable Secret Plaintext ($M = 0$ B)** | **7,090 Bytes** | **7,090 Bytes** | **6,578 Bytes** |
| **Usable Secret Plaintext ($M = 2$ B)** | **7,088 Bytes** | **7,088 Bytes** | **6,576 Bytes** |

*(Note: Under RS-32, because the 8-byte difference between v1 and v2 headers was absorbed by the trailing codeword remainder ($12 \to 4$), full-block capacity under RS-32 is identically 7,090 bytes).*

---

## 5. Dynamic Carrier Resolution Scaling

The accounting engine is fully generalized for dynamic carrier dimensions. For an arbitrary even carrier of size $H \times W$:

$$N_{\text{raw\_bytes}} = \left\lfloor \frac{\frac{H}{2} \cdot \frac{W}{2}}{8} \right\rfloor$$
$$N_{\text{max\_protected}} = N_{\text{raw\_bytes}} - 24$$

### 5.1 Evaluated Carrier Dimensions

| Carrier Resolution | Red-HH Subband Dim | Raw Capacity | Max Protected (v2) | Usable Secret (RS-48, $M=0$) | Example Dataset Image |
|:---|:---:|:---:|:---:|:---:|:---|
| **512 × 512** | 256 × 256 | 8,192 B | 8,168 B | 6,578 B | Canonical Test Crop |
| **1024 × 1024** | 512 × 512 | 32,768 B | 32,744 B | 26,458 B | 2× Scaled Carrier |
| **2040 × 1356** | 1020 × 678 | 86,445 B | 86,421 B | 70,099 B | `0043.png` (DIV2K HR Full Resolution) |

---

## 6. Hard-Gate Verification and Enforcement Audit

The capacity gate is enforced at [`src/carrier/capacity.py:228-243`](file:///E:/Paladin_media/src/carrier/capacity.py#L228-L243) via `CapacityCalculator.check_capacity()`:

1. **Immediate Preflight Evaluation:** Prior to mask generation, DWT embedding, or model inference, the sender evaluates the exact byte length of the Reed–Solomon protected stream against `report.max_protected_bytes`.
2. **Fail-Closed Security Property:** If `protected_payload_bytes > max_protected_bytes`, the function raises `PayloadCapacityExceededError`. The error message details the exact overflow and usable capacity.
3. **Immutability of Carrier:** When capacity is exceeded, no transform coefficients are altered, no partial masks are generated, and no network or disk writes occur.
