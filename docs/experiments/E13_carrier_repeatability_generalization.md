# PaladinMedia — Experiment E13: Multi-Carrier Repeatability & Generalization Across DIV2K HR Images

**Experiment ID:** `E13`
**Execution Timestamp:** `2026-10-11T18:57:21Z`
**Cover Carriers:** `0001.png`, `0002.png`, `0003.png`, `0004.png`, `0005.png` (DIV2K HR)
**Total Executions:** 90 runs (5 images × 2 resolutions × 3 payload densities × 3 seeds)
**Seeds Evaluated:** [42, 1337, 2026]
**Modulation Scheme:** 1-level 2D Haar DWT, Red HH Channel 0, 1 bit/coefficient integer-parity

## 1. Executive Summary & Experimental Scope

Experiment E13 benchmarks the statistical stability, fidelity variance, bit error rate consistency, and execution latency of the Phase 2 classical baseline steganographic engine. The empirical findings in this report are strictly scoped to the 90 evaluated experimental runs across 5 natural DIV2K HR images, 2 spatial resolutions (512×512 and 1024×1024), 3 payload density tiers (10%, 50%, and 90%), and 3 random seeds (42, 1337, 2026), stored and recovered under lossless 8-bit RGB PNG container format.

## 2. Statistical Aggregation Summary

| Carrier Geometry | Payload Tier | Plaintext Density ($P / P_{\text{max}}$) | Protected Utilization ($L_{\text{prot}} / L_{\text{max}}$) | Samples | Mean PSNR (dB) | Std Dev | Min PSNR | Max PSNR | Mean SSIM | Max BER | Exact Recovery | Mean Embed Time | Mean Extract Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `512×512` | 10% Load | 709 B (10.0%) | 887 B (10.85%) | 15 | **71.53** | ±0.058 | 71.43 | 71.61 | 0.999999 | 0.000000 | `True` | 43.1 ms | 16.1 ms |
| `512×512` | 50% Load | 3,545 B (50.0%) | 4,139 B (50.62%) | 15 | **64.88** | ±0.024 | 64.85 | 64.94 | 0.999996 | 0.000000 | `True` | 77.4 ms | 66.2 ms |
| `512×512` | 90% Load | 6,381 B (90.0%) | 7,359 B (90.01%) | 15 | **62.39** | ±0.017 | 62.36 | 62.42 | 0.999993 | 0.000000 | `True` | 111.7 ms | 115.5 ms |
| `1024×1024` | 10% Load | 2,857 B (10.0%) | 3,355 B (10.24%) | 15 | **71.81** | ±0.030 | 71.77 | 71.87 | 0.999999 | 0.000000 | `True` | 189.8 ms | 70.6 ms |
| `1024×1024` | 50% Load | 14,287 B (50.0%) | 16,417 B (50.13%) | 15 | **64.93** | ±0.012 | 64.91 | 64.95 | 0.999996 | 0.000000 | `True` | 368.4 ms | 320.7 ms |
| `1024×1024` | 90% Load | 25,716 B (90.0%) | 29,478 B (90.0%) | 15 | **62.39** | ±0.012 | 62.36 | 62.41 | 0.999993 | 0.000000 | `True` | 550.3 ms | 546.5 ms |

## 3. Detailed Results Across Carrier Images

### 3.1 512 × 512 Carrier Crops Across Images (Averaged Over Seeds)

| Image | Native Dimensions | 10% Load PSNR | 50% Load PSNR | 90% Load PSNR | Mean SSIM | Max BER | Exact Match |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0001.png` | 2040 × 1404 px | 71.46 dB | 64.88 dB | 62.39 dB | 0.999996 | 0.000000 | `True` |
| `0002.png` | 2040 × 1848 px | 71.59 dB | 64.86 dB | 62.38 dB | 0.999995 | 0.000000 | `True` |
| `0003.png` | 2040 × 1356 px | 71.56 dB | 64.89 dB | 62.39 dB | 0.999996 | 0.000000 | `True` |
| `0004.png` | 2040 × 1344 px | 71.53 dB | 64.90 dB | 62.39 dB | 0.999995 | 0.000000 | `True` |
| `0005.png` | 1608 × 2040 px | 71.53 dB | 64.87 dB | 62.38 dB | 0.999997 | 0.000000 | `True` |

### 3.2 1024 × 1024 Carrier Crops Across Images (Averaged Over Seeds)

| Image | Native Dimensions | 10% Load PSNR | 50% Load PSNR | 90% Load PSNR | Mean SSIM | Max BER | Exact Match |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0001.png` | 2040 × 1404 px | 71.83 dB | 64.94 dB | 62.39 dB | 0.999996 | 0.000000 | `True` |
| `0002.png` | 2040 × 1848 px | 71.80 dB | 64.93 dB | 62.39 dB | 0.999995 | 0.000000 | `True` |
| `0003.png` | 2040 × 1356 px | 71.82 dB | 64.93 dB | 62.39 dB | 0.999997 | 0.000000 | `True` |
| `0004.png` | 2040 × 1344 px | 71.80 dB | 64.94 dB | 62.39 dB | 0.999997 | 0.000000 | `True` |
| `0005.png` | 1608 × 2040 px | 71.81 dB | 64.93 dB | 62.39 dB | 0.999997 | 0.000000 | `True` |

## 4. Analysis & Defensible Scientific Findings

1. **Cross-Carrier Repeatability:** Across the 90 evaluated runs, PSNR variation was low within each resolution and payload tier, with reported standard deviations below 0.06 dB. This indicates consistent fidelity under the tested carrier images, payload sizes, seeds, and lossless PNG workflow. The results do not establish universal content-invariant distortion; broader claims require evaluation across a larger and more diverse carrier set.
2. **Observed Zero BER and Exact Recovery Across 90 Runs:** In all 90 evaluated runs, the extracted carrier bit error rate was observed to be identically **0.000000**, and 100% of payloads were recovered with successful cryptographic authentication and bit-exact equality (`exact_match = True`). These results represent observed empirical performance under lossless 8-bit RGB PNG container conditions, rather than a universal guarantee across lossy or adversarial environments.
3. **Objective Fidelity Metrics vs. Steganographic Concealment:** The lowest observed PSNR across all 90 runs was **62.36 dB** (at 90% load on 512×512), and the lowest SSIM was **0.999991**. While these values establish high numerical signal fidelity relative to the unmodified cover carrier, PSNR and SSIM do not establish steganographic undetectability.
4. **Latency:** Mean embedding latency increased across the three tested payload tiers at both resolutions. These measurements characterize the observed workload; they do not establish an asymptotic complexity bound or guarantee the same scaling under other configurations.
5. **Disambiguation of Capacity Accounting Metrics:**
   - **Total Raw Carrier Capacity:** Defined as `(size * size // 32)` bytes (8,192 B for 512×512; 32,768 B for 1024×1024).
   - **Usable Protected Capacity ($L_{\text{max}}$):** Defined as `(size * size // 32) - 16` bytes (8,176 B for 512×512; 32,752 B for 1024×1024). This serves as the denominator for **Protected Capacity Utilization** ($L_{\text{prot}} / L_{\text{max}}$).
   - **Maximum Usable Plaintext Capacity ($P_{\text{max}}$):** Defined after accounting for Reed-Solomon parity (32 bytes per 223 bytes), AES-256-GCM overhead (28 bytes), and framing (18 bytes). For $M=0$, $P_{\text{max}} = 7,090\text{ B}$ for 512×512 and $28,574\text{ B}$ for 1024×1024. This serves as the denominator for **Plaintext Density** ($P / P_{\text{max}}$).

## 5. Scope, Constraints & Baseline Limitations

The conclusions of Experiment E13 are subject to the following explicit boundaries:
- **Empirical Scope:** The empirical findings cover the five tested DIV2K HR images (`0001.png` through `0005.png`), two carrier resolutions (512×512 and 1024×1024), three payload tiers (10%, 50%, and 90%), and three random seeds (42, 1337, 2026).
- **Container Workflow Scope:** The reported recovery outcomes apply to the tested lossless 8-bit RGB PNG workflow.
- **Steganographic Undetectability:** PSNR and SSIM do not establish steganographic undetectability. Resistance to statistical or neural steganalysis was not established by E13.
- **Transformation Robustness:** Robustness to JPEG recompression, resizing, noise, and other transformations was not established by E13.
