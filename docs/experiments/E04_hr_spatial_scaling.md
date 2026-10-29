# PaladinMedia — Experiment E04-HR: Carrier Spatial Scaling & High-Resolution DIV2K Validation
**Execution Timestamp:** `2026-10-11T17:48:06Z`  **Dataset Source:** `dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR` (800 HR Images)
## 1. Executive Summary
Experiment **E04-HR** empirically validates the carrier spatial scaling behavior of PaladinMedia's transform-domain steganography engine across native High-Resolution (HR) imagery from the DIV2K dataset. The evaluation encompasses:
- **E04-A (512x512 Reference)**: Center crops from 5 representative HR images establishing baseline performance.
- **E04-B (1024x1024 High-Resolution)**: Native 1024x1024 center crops from the same 5 HR images, demonstrating 4x carrier capacity expansion.
- **High-Load Stress Test**: Successful embedding and verbatim recovery of a 20,000-byte payload into a 1024x1024 carrier (far exceeding the 512x512 capacity limit).
- **Exact Cryptographic Verification**: Verbatim byte recovery (`ExactMatch = True`, `BER = 0.0`) verified across all runs through lossless 8-bit RGB PNG serialization and reload.
- **Negative Hard-Gate Verification**: Proven fail-closed rejection of oversized payloads before transform or carrier allocation.
## 2. Capacity Accounting & Mathematical Reconciliation
A formal reconciliation of usable secret capacity establishes the relationship between raw transform coefficients and delivered secret plaintext:

$$\text{Usable Secret} = \left\lfloor \frac{C_{\text{max\_prot}} - 4}{255} \right\rfloor \times 223 - 28 (\text{AES}) - 18 (\text{Framing}) - \text{len}(\text{metadata})$$

| Carrier Dimension | Subband Dim | Raw Capacity | Max Protected | RS Blocks | Max Encrypted | Usable (0 Meta) | Usable (2B Ref Meta) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **512 × 512** | 256 × 256 | 8,192 B | 8,176 B | 32 | 7,136 B | **7,090 B** | **7,088 B** |
| **1024 × 1024** | 512 × 512 | 32,768 B | 32,752 B | 128 | 28,544 B | **28,498 B** | **28,496 B** |

> [!NOTE]
> **Capacity Discrepancy Resolution**: The previous 7,088 vs 7,090 bytes discrepancy is entirely explained by metadata framing. With zero metadata bytes, the mathematical ceiling is strictly **7,090 bytes**. With standard 2-byte reference metadata (e.g., minimal JSON `{}`), usable plaintext is strictly **7,088 bytes**. Both values are mathematically exact and reconciled.
## 3. Spatial Scaling Analysis Table
| Carrier Dimension | Subband Dim | Raw Positions | Raw Capacity | Max Protected | Usable Plaintext (0 Meta) | DWT Est Mem | Execution Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `512x512` | `256x256` | 65,536 | 8,192 B | 8,176 B | 7,090 B | 6.75 MB | **EXECUTED (E04-A Reference Crop)** |
| `1024x1024` | `512x512` | 262,144 | 32,768 B | 32,752 B | 28,498 B | 27.0 MB | **EXECUTED (E04-B Native HR Crop)** |
| `2048x2048` | `1024x1024` | 1,048,576 | 131,072 B | 131,056 B | 114,353 B | 108.0 MB | **SKIPPED (Native HR Dim < 2048)** |
| `3072x3072` | `1536x1536` | 2,359,296 | 294,912 B | 294,896 B | 257,742 B | 243.0 MB | **SKIPPED (Native HR Dim < 2048)** |
| `4096x4096` | `2048x2048` | 4,194,304 | 524,288 B | 524,272 B | 458,219 B | 432.0 MB | **SKIPPED (Native HR Dim < 2048)** |
| `5120x5120` | `2560x2560` | 6,553,600 | 819,200 B | 819,184 B | 716,230 B | 675.0 MB | **SKIPPED (Native HR Dim < 2048)** |

> [!IMPORTANT]
> **Availability of >= 2048 Resolutions**: In DIV2K HR, the maximum dimensions are 2040x1848 pixels (e.g., `0002.png`). No images have a minimum dimension >= 2048. Consequently, 2048x2048, 3072x3072, 4096x4096, and 5120x5120 are strictly marked as **SKIPPED / UNAVAILABLE** because synthetic upscaling is prohibited by architectural standards.

## 4. Empirical Evaluation Results

### 4.1 E04-A: Reference Baseline (512 × 512 HR Center Crops)

| Image ID | Native Dimensions | Crop Offset (L, T) | PSNR (dB) | SSIM | MAE | MaxErr | BER | Exact Match | Total Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0001.png` | 2040×1404 | (764, 446) | 75.8554 | 1.0 | 0.001689 | 1 | 0.0 | `True` | 81.5 ms |
| `0002.png` | 2040×1848 | (764, 668) | 75.918 | 1.0 | 0.001664 | 1 | 0.0 | `True` | 76.11 ms |
| `0003.png` | 2040×1356 | (764, 422) | 75.8034 | 1.0 | 0.001709 | 1 | 0.0 | `True` | 86.12 ms |
| `0004.png` | 2040×1344 | (764, 416) | 75.8034 | 1.0 | 0.001709 | 1 | 0.0 | `True` | 83.64 ms |
| `0005.png` | 1608×2040 | (548, 764) | 75.8784 | 1.0 | 0.00168 | 1 | 0.0 | `True` | 75.28 ms |

### 4.2 E04-B: High-Resolution Carriers (1024 × 1024 HR Center Crops)

| Image ID | Native Dimensions | Crop Offset (L, T) | PSNR (dB) | SSIM | MAE | MaxErr | BER | Exact Match | Total Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0001.png` | 2040×1404 | (508, 190) | 81.8337 | 1.0 | 0.000426 | 1 | 0.0 | `True` | 305.72 ms |
| `0002.png` | 2040×1848 | (508, 412) | 81.6654 | 1.0 | 0.000443 | 1 | 0.0 | `True` | 291.73 ms |
| `0003.png` | 2040×1356 | (508, 166) | 81.7376 | 1.0 | 0.000436 | 1 | 0.0 | `True` | 263.2 ms |
| `0004.png` | 2040×1344 | (508, 160) | 81.9023 | 1.0 | 0.00042 | 1 | 0.0 | `True` | 278.95 ms |
| `0005.png` | 1608×2040 | (292, 508) | 81.779 | 1.0 | 0.000432 | 1 | 0.0 | `True` | 294.94 ms |

### 4.3 High-Load 1024 × 1024 Capacity Scaling Run

- **Carrier Image:** `0001.png` (1024 × 1024)
- **Secret Plaintext Size:** `20,000 bytes` (Exceeds 512×512 ceiling by ~2.8×)
- **Protected Payload Size:** `23,010 bytes`
- **Stego Fidelity:** PSNR = **63.435 dB**, SSIM = **0.999994**, MAE = **0.029483**
- **Carrier BER:** **0.0**
- **Exact Recovery:** **True**
- **Timings:** Embed = 476.42 ms, Extract = 433.69 ms

## 5. Negative Hard-Gate Testing

| Carrier Resolution | Subband Dim | Max Protected | Injected Payload | Fail-Closed Rejection | Exception Type |
|:---:|:---:|:---:|:---:|:---:|:---|
| `512x512` | `256×256` | 8,176 B | 8,177 B | **True** | `PayloadCapacityExceededError` |
| `1024x1024` | `512×512` | 32,752 B | 32,753 B | **True** | `PayloadCapacityExceededError` |

Both tests strictly raised `PayloadCapacityExceededError` before any wavelet transform or carrier modification occurred.

## 6. Architecture & Implementation Invariants

1. **Preservation of Core Sequencing**: Unidirectional pipeline flow (`Serialization -> AES-256-GCM -> Reed-Solomon -> Transform Embed`) preserved intact.
2. **Zero Plaintext Leakage**: All transform and embedding operations execute exclusively on authenticated ciphertext.
3. **Invertibility**: Haar wavelet decomposition is strictly unitary/lossless; integer-parity modulation on 2*HH preserves parity through PNG encoding with zero bit error.
4. **Fail-Closed Security**: Payload size over-allocation halts instantly at the capacity gate.
