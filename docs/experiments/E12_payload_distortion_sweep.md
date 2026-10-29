# PaladinMedia — Experiment E12: Continuous Payload-Density & Capacity vs. Distortion Sweep

**Experiment ID:** `E12`
**Execution Timestamp:** `2026-10-11T18:41:00Z`
**Cover Source:** `DIV2K HR 0001.png` (2040 × 1404 px native)
**Modulation Scheme:** 1-level 2D Haar DWT, Red HH Channel 0, 1 bit/coefficient integer-parity

## 1. Research Question & Theoretical Hypothesis

**Research Question:** How do carrier fidelity (PSNR, SSIM, MAE) and execution runtime behave as payload density scales continuously from minimal loads (~7%) up to 100% saturation and boundary overflow, across both 512×512 and 1024×1024 carriers?

**Theoretical Model:**
Under 1 bpc integer-parity embedding, each bit modulates $2\text{HH} = a - b - c + d$ by perturbing pixel $a \leftarrow a \pm 1$. Assuming uniformly distributed secret ciphertext bits, approximately 50% of coefficients require modulation ($\Delta = \pm 1$). Therefore, total spatial Mean Squared Error (MSE) is strictly linear in the number of embedded payload bytes $L$:

$$\text{MSE}(L) \approx \frac{L \times 8 \times 0.5}{H \times W \times 3} = \frac{4L}{3HW}$$

which produces a logarithmic PSNR decay frontier:

$$\text{PSNR}(L) \approx 10 \log_{10}\left(\frac{255^2}{\text{MSE}(L)}\right) = 10 \log_{10}\left(\frac{3 \times 255^2 \times HW}{4L}\right)$$

## 2. Empirical Results: 512 × 512 Reference Carrier

| Test Point | Plaintext | Protected Stream | Capacity Utilization | PSNR (dB) | SSIM | MAE | MaxErr | BER | Exact Match | Embed Time | Extract Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `500 B (~7% Load)` | 500 B | 646 B | 7.9% | **72.9561** | 0.999999 | 0.003292 | 1 | 0.0 | `True` | 38.84 ms | 11.15 ms |
| `1,500 B (~21% Load)` | 1,500 B | 1,774 B | 21.7% | **68.4691** | 0.999998 | 0.009251 | 1 | 0.0 | `True` | 54.38 ms | 33.04 ms |
| `3,500 B (~49% Load)` | 3,500 B | 4,062 B | 49.68% | **64.9445** | 0.999996 | 0.020827 | 1 | 0.0 | `True` | 88.93 ms | 61.26 ms |
| `5,500 B (~78% Load)` | 5,500 B | 6,350 B | 77.67% | **63.044** | 0.999994 | 0.032261 | 1 | 0.0 | `True` | 112.49 ms | 97.18 ms |
| `7,000 B (~99% Load)` | 7,000 B | 8,074 B | 98.75% | **61.9495** | 0.999992 | 0.041508 | 1 | 0.0 | `True` | 127.51 ms | 123.44 ms |
| `7,090 B (99.85% Boundary)` | 7,090 B | 8,164 B | 99.85% | **61.9374** | 0.999992 | 0.041623 | 1 | 0.0 | `True` | 129.05 ms | 142.91 ms |
| **7,091 B (Boundary + 1)** | 7,091 B | 8,197 B | > 100% (Overflow) | **BLOCKED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 3. Empirical Results: 1024 × 1024 High-Resolution Carrier

| Test Point | Plaintext | Protected Stream | Capacity Utilization | PSNR (dB) | SSIM | MAE | MaxErr | BER | Exact Match | Embed Time | Extract Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `2,000 B (~7% Load)` | 2,000 B | 2,370 B | 7.24% | **73.3144** | 0.999999 | 0.003031 | 1 | 0.0 | `True` | 164.44 ms | 45.22 ms |
| `7,000 B (~24% Load)` | 7,000 B | 8,074 B | 24.65% | **67.9818** | 0.999998 | 0.010349 | 1 | 0.0 | `True` | 269.95 ms | 174.96 ms |
| `14,000 B (~49% Load)` | 14,000 B | 16,066 B | 49.05% | **65.0322** | 0.999996 | 0.020411 | 1 | 0.0 | `True` | 405.47 ms | 325.21 ms |
| `21,000 B (~73% Load)` | 21,000 B | 24,090 B | 73.55% | **63.256** | 0.999994 | 0.030725 | 1 | 0.0 | `True` | 517.82 ms | 468.95 ms |
| `28,000 B (~98% Load)` | 28,000 B | 32,082 B | 97.95% | **62.0148** | 0.999992 | 0.040888 | 1 | 0.0 | `True` | 633.19 ms | 592.59 ms |
| `28,574 B (100.0% Boundary)` | 28,574 B | 32,752 B | 100.0% | **61.9264** | 0.999991 | 0.04173 | 1 | 0.0 | `True` | 657.63 ms | 600.25 ms |
| **28,575 B (Boundary + 1)** | 28,575 B | 32,753 B | > 100% (Overflow) | **BLOCKED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 4. Analysis & Key Experimental Findings

1. **Logarithmic Fidelity Decay Confirmed:** As predicted by theory, PSNR decays monotonically with payload size. At 512×512, PSNR transitions from **72.96 dB** (~7% load) down to **61.94 dB** (99.85% utilization). At 1024×1024, PSNR transitions from **73.31 dB** (~7% load) down to **61.93 dB** (100.0% utilization).
2. **Capacity Boundary Characteristics:** At 512×512, maximum plaintext capacity is 7,090 B (producing an 8,164 B protected stream, or 99.85% of the 8,176 B ceiling due to 12 remainder bytes unable to host a 33rd RS block). At 1024×1024, maximum plaintext capacity is 28,574 B (producing a 32,752 B protected stream, or exact 100.0% utilization via shortened RS block capability).
3. **Zero Perceptual Degradation Under Tested Lossless Container Conditions:** Under lossless 8-bit RGB PNG serialization and reload, PSNR remains > 61.9 dB across all valid operating densities, which is ~12–17 dB above standard human visual perceptibility thresholds (45–50 dB). SSIM remains strictly **>= 0.999991**.
4. **Zero Bit Error Rate Under Lossless Container Conditions:** Carrier BER is identically **0.000000** across all evaluated density points, confirming that integer-parity modulation produces zero inter-coefficient interference or roundtrip drift when saved and reloaded as lossless PNG.
5. **Fail-Closed Hard Gate Rejection:** At both 512×512 and 1024×1024, exceeding the usable plaintext capacity by even 1 byte (7,091 B and 28,575 B, respectively) causes `CapacityCalculator` to throw `PayloadCapacityExceededError`, terminating transform execution and preventing any plaintext leakage.
