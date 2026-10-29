# PaladinMedia — Experiment E13: Multi-Carrier Repeatability & Generalization Across DIV2K HR Images

**Experiment ID:** `E13`
**Execution Timestamp:** `2026-10-11T18:45:55Z`
**Cover Carriers:** `0001.png`, `0002.png`, `0003.png`, `0004.png`, `0005.png` (DIV2K HR)
**Total Executions:** 90 runs (5 images × 2 resolutions × 3 payload densities × 3 seeds)
**Seeds Evaluated:** [42, 1337, 2026]
**Modulation Scheme:** 1-level 2D Haar DWT, Red HH Channel 0, 1 bit/coefficient integer-parity

## 1. Executive Summary & Experimental Objectives

Experiment E13 rigorously benchmarks the statistical stability, fidelity variance, bit error rate consistency, and execution latency of the Phase 2 classical baseline steganographic engine. By sweeping across multiple distinct natural images with diverse textures and frequency profiles (from smooth gradients to high-frequency foliage and brickwork), E13 establishes whether the 1 bpc integer-parity embedding model generalizes robustly or exhibits image-specific degradation.

## 2. Statistical Aggregation Summary

| Carrier Geometry | Payload Tier | Plaintext | Samples | Mean PSNR (dB) | Std Dev | Min PSNR | Max PSNR | Mean SSIM | Max BER | Exact Recovery | Mean Embed Time | Mean Extract Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `512×512` | 10% Load | 709 B | 15 | **71.51** | ±0.048 | 71.43 | 71.60 | 0.999999 | 0.000000 | `True` | 43.1 ms | 15.9 ms |
| `512×512` | 50% Load | 3,545 B | 15 | **64.88** | ±0.019 | 64.85 | 64.92 | 0.999996 | 0.000000 | `True` | 75.9 ms | 66.7 ms |
| `512×512` | 90% Load | 6,381 B | 15 | **62.39** | ±0.015 | 62.37 | 62.43 | 0.999993 | 0.000000 | `True` | 113.2 ms | 114.5 ms |
| `1024×1024` | 10% Load | 2,857 B | 15 | **71.81** | ±0.023 | 71.76 | 71.83 | 0.999999 | 0.000000 | `True` | 192.3 ms | 68.2 ms |
| `1024×1024` | 50% Load | 14,287 B | 15 | **64.93** | ±0.013 | 64.90 | 64.95 | 0.999996 | 0.000000 | `True` | 383.9 ms | 326.9 ms |
| `1024×1024` | 90% Load | 25,716 B | 15 | **62.39** | ±0.008 | 62.38 | 62.41 | 0.999993 | 0.000000 | `True` | 582.5 ms | 552.9 ms |

## 3. Detailed Results Across Carrier Images

### 3.1 512 × 512 Carrier Crops Across Images (Averaged Over Seeds)

| Image | Native Dimensions | 10% Load PSNR | 50% Load PSNR | 90% Load PSNR | Mean SSIM | Mean BER | Exact Match |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0001.png` | 2040 × 1404 px | 71.51 dB | 64.89 dB | 62.38 dB | 0.999996 | 0.000000 | `True` |
| `0002.png` | 2040 × 1848 px | 71.50 dB | 64.89 dB | 62.38 dB | 0.999995 | 0.000000 | `True` |
| `0003.png` | 2040 × 1356 px | 71.46 dB | 64.87 dB | 62.39 dB | 0.999996 | 0.000000 | `True` |
| `0004.png` | 2040 × 1344 px | 71.50 dB | 64.87 dB | 62.41 dB | 0.999995 | 0.000000 | `True` |
| `0005.png` | 1608 × 2040 px | 71.57 dB | 64.87 dB | 62.38 dB | 0.999997 | 0.000000 | `True` |

### 3.2 1024 × 1024 Carrier Crops Across Images (Averaged Over Seeds)

| Image | Native Dimensions | 10% Load PSNR | 50% Load PSNR | 90% Load PSNR | Mean SSIM | Mean BER | Exact Match |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0001.png` | 2040 × 1404 px | 71.82 dB | 64.93 dB | 62.39 dB | 0.999996 | 0.000000 | `True` |
| `0002.png` | 2040 × 1848 px | 71.79 dB | 64.93 dB | 62.39 dB | 0.999995 | 0.000000 | `True` |
| `0003.png` | 2040 × 1356 px | 71.82 dB | 64.94 dB | 62.38 dB | 0.999997 | 0.000000 | `True` |
| `0004.png` | 2040 × 1344 px | 71.82 dB | 64.93 dB | 62.38 dB | 0.999997 | 0.000000 | `True` |
| `0005.png` | 1608 × 2040 px | 71.81 dB | 64.92 dB | 62.39 dB | 0.999997 | 0.000000 | `True` |

## 4. Analysis & Key Experimental Findings

1. **Remarkable Cross-Carrier Repeatability:** Across all 90 runs, PSNR exhibits negligible variance (standard deviation < 0.08 dB within any payload tier). Because integer-parity modulation operates purely on algebraic parities ($2\text{HH} = a - b - c + d$), the spatial distortion metric depends almost exclusively on the number of modified coefficients rather than carrier image content.
2. **Strict Universal Bit Invariance (BER = 0.000000):** Across all 5 natural images, 2 spatial resolutions, 3 density tiers, and 3 distinct random seeds (90/90 runs), the extracted bit error rate is identically **0.000000**, and 100% of payloads recovered with perfect cryptographic authentication and byte equality (`exact_match = True`).
3. **Perceptual Invisibility Guarantee:** The worst-case PSNR observed across all 90 tests is **62.37 dB** (at 90% load on 512×512), and the worst-case SSIM is **0.999991**, maintaining > 12 dB margin above the visual degradation threshold of 50 dB under lossless PNG container storage.
4. **Predictable Linear Execution Latency:** End-to-end embedding and extraction latency scale linearly with payload length and carrier resolution, showing no divergence or pathological stalls across diverse image textures.
