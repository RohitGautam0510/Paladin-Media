# Experiment E14: Classical Channel Degradation & Lossy Compression Baseline

- **Document ID:** `PDM-EXP-E14-20261012`
- **Experiment Date:** 2026-10-12
- **Target Engine:** Phase 2 Classical Baseline (1-Level 2D Haar DWT, Red-HH Integer-Parity Modulation)
- **Dataset:** DIV2K High-Resolution Dataset (`0001.png` - `0005.png`)
- **Total Experimental Runs:** 220
- **Status:** COMPLETED & VERIFIED

---

## 1. Executive Summary & Core Scientific Conclusion

Experiment E14 establishes the empirical channel degradation boundary of the classical wavelet integer-parity steganographic engine. The experiment subjected 20 distinct carrier-payload configurations across 5 DIV2K HR images, 2 spatial resolutions (512x512 and 1024x1024), and 2 payload load tiers to 11 channel environments: 1 lossless PNG control, 7 JPEG recompression quality factors ($Q \in \{100, 95, 90, 85, 80, 75, 70\}$), and 3 additive Gaussian noise channels ($\sigma \in \{0.5, 1.0, 2.0\}$).

### Key Findings:
1. **Flawless Lossless Operation:** Under the lossless PNG container channel, the classical baseline achieved a **100% recovery success rate** (20/20 runs), $\text{BER} = 0.000000$, and zero header or GHASH authentication errors.
2. **Total Catastrophic Failure Under JPEG:** Under JPEG compression at **every tested quality factor** ($Q = 100$ down to $Q = 70$), the classical baseline suffered a **0.0% recovery success rate** (0/140 runs). The raw subband parity bit error rate hovered at approximately **50.0%** (pure statistical noise) due to high-frequency DCT quantization destroying spatial integer parity relationships $(a - b - c + d) \pmod 2$. In 100% of JPEG runs, the 16-byte carrier header magic bytes were immediately corrupted, preventing even framing extraction.
3. **Vulnerability to Micro-Noise:** Under additive Gaussian noise with standard deviation as small as $\sigma = 0.5$ (imperceptible spatial distortion, $\text{PSNR} \approx 52.8\text{ dB}$), the raw parity BER reached **49.1%**, completely overwhelming the Reed-Solomon RS(255, 223) error correction budget (which tolerates at most $\approx 3.0\%$ random bit errors). At $\sigma \ge 1.0$, raw parity BER reached $50.2\%$, resulting in $0.0\%$ recovery.
4. **Empirical Justification for Phase 3 (ACS-Sub-INN v3):** These empirical findings conclusively prove that classical integer-parity wavelet steganography cannot operate across lossy channels. The transition to the continuous-domain invertible neural network (ACS-Sub-INN v3) with channel noise modeling is scientifically justified.

---

## 2. Channel Performance Summary Table

| Channel Environment | Total Runs | Header Valid Rate | ECC Decode Success | Verbatim Payload Recovery | Mean Raw Parity BER | Mean Received PSNR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PNG (Lossless Control)** | 20 | 100.0% | 100.0% | **100.0%** | 0.00% | 67.56 dB |
| **JPEG (Q=100)** | 20 | 0.0% | 0.0% | **0.0%** | 47.59% | 41.96 dB |
| **JPEG (Q=95)** | 20 | 0.0% | 0.0% | **0.0%** | 49.87% | 38.63 dB |
| **JPEG (Q=90)** | 20 | 0.0% | 0.0% | **0.0%** | 49.94% | 36.33 dB |
| **JPEG (Q=85)** | 20 | 0.0% | 0.0% | **0.0%** | 49.93% | 34.88 dB |
| **JPEG (Q=80)** | 20 | 0.0% | 0.0% | **0.0%** | 49.91% | 33.88 dB |
| **JPEG (Q=75)** | 20 | 0.0% | 0.0% | **0.0%** | 49.83% | 33.13 dB |
| **JPEG (Q=70)** | 20 | 0.0% | 0.0% | **0.0%** | 49.84% | 32.58 dB |
| **Gaussian Noise (sigma=0.5)** | 20 | 0.0% | 0.0% | **0.0%** | 49.10% | 52.83 dB |
| **Gaussian Noise (sigma=1.0)** | 20 | 0.0% | 0.0% | **0.0%** | 50.20% | 47.73 dB |
| **Gaussian Noise (sigma=2.0)** | 20 | 0.0% | 0.0% | **0.0%** | 49.92% | 42.01 dB |

---

## 3. Mathematical & Algorithmic Failure Analysis

### 3.1 Why Integer-Parity Fails Under JPEG
The classical baseline modulates secret bits into the 2D Haar detail subband using integer parity:
$$2\text{HH} = a - b - c + d$$
where $a, b, c, d$ are 2x2 spatial quad pixels. A secret bit $b_i$ is embedded by adjusting spatial pixel $d$ such that $(a - b - c + d) \pmod 2 = b_i$.

JPEG compression operates via an 8x8 block Discrete Cosine Transform (DCT):
1. Spatial RGB quads are transformed to $YC_bC_r$ and decomposed into 64 frequency DCT basis functions.
2. High-frequency DCT coefficients are divided by a quantization matrix $Q_{uv}$ and rounded to nearest integers.
3. Inverse DCT and integer rounding reconstruct the spatial pixels.

Even when $Q=100$, high-frequency truncation alters individual pixel values by $\pm 1$. Because parity is evaluated modulo 2, a single-pixel delta flips the parity bit: $(d \pm 1) \pmod 2 = (d \pmod 2) \oplus 1$. Across an entire $256\times256$ subband, independent pixel rounding flips approximately half of all coefficient parity values, producing a uniform binomial error distribution with $p \approx 0.50$.

### 3.2 Reed-Solomon Correction Limits
PaladinMedia uses systematic Reed-Solomon $\text{RS}(255, 223)$, which provides $2t = 32$ parity symbols per block and can correct at most $t = 16$ symbol errors per 255 bytes (a maximum symbol error rate of $16 / 255 \approx 6.27\%$). Under random bit errors with error rate $p$, the byte symbol error rate is:
$$P_s = 1 - (1 - p)^8$$
When $p = 0.50$, $P_s = 1 - (0.5)^8 = 99.61\%$, overwhelmingly exceeding the Reed-Solomon correction capability.

---

## 4. Phase 3 Architecture Implications

1. **Continuous Feature Invariance:** Phase 3's ACS-Sub-INN v3 must modulate continuous wavelet features through affine coupling blocks rather than discrete integer parities.
2. **Differentiable Channel Simulation:** Training must incorporate simulated JPEG layers (e.g., Shin & Song differentiable JPEG or Fourier-domain approximation) to force INN latents into low/mid-frequency invariant subspaces.
3. **Robust Carrier Framing:** The carrier header must either be redundantly encoded with high-gain convolutional/repetition codes or inferred implicitly from carrier dimensions.

---

## 5. Traceability & Reproduction Command

- Runner: `scripts/experiments/run_e14.py`
- Raw JSON: `docs/experiments/e14_results.json`
- Reproduction:
  ```powershell
  python scripts/experiments/run_e14.py
  ```
