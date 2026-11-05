# Experiment E14: Classical Channel Degradation & Lossy Compression Baseline

- **Document ID:** `PDM-EXP-E14-20261012`
- **Experiment Date:** 2026-10-12
- **Target Engine:** Phase 2 Classical Baseline (1-Level 2D Haar DWT, Red-HH Integer-Parity Modulation, AES-256-GCM, Reed-Solomon RS(255, 223))
- **Dataset:** DIV2K High-Resolution Dataset (`0001.png` – `0005.png`)
- **Carrier Dimensions:** $512 \times 512$ and $1024 \times 1024$ native center crops
- **Payload Tiers:** Low (1,000 B on 512, 4,000 B on 1024) and Medium (3,500 B on 512, 14,000 B on 1024)
- **Evaluated Channels:** 11 conditions (Lossless PNG control, JPEG $Q \in \{100, 95, 90, 85, 80, 75, 70\}$, Additive Gaussian Noise $\sigma \in \{0.5, 1.0, 2.0\}$ with fixed seed 42)
- **Total Experimental Runs:** 220 (20 runs per channel condition across 20 carrier-payload configurations)
- **Status:** COMPLETED, VERIFIED & SCIENTIFIC LIMITATIONS RECONCILED

---

## 1. Executive Summary & Core Scientific Conclusion

Experiment E14 establishes the empirical channel degradation boundary of the Phase 2 classical wavelet integer-parity steganographic engine. The experiment subjected 20 distinct carrier-payload configurations across 5 DIV2K HR images, 2 spatial resolutions ($512 \times 512$ and $1024 \times 1024$), and 2 payload tiers (Low and Medium) to 11 channel environments: 1 lossless PNG control, 7 JPEG recompression quality factors ($Q \in \{100, 95, 90, 85, 80, 75, 70\}$), and 3 additive Gaussian noise channels ($\sigma \in \{0.5, 1.0, 2.0\}$).

### Core Scientific Conclusion:
> **The tested classical integer-parity baseline fails under the specific JPEG recompression and Gaussian-noise conditions evaluated in E14.**

This conclusion is strictly bounded to the evaluated parameter space and must not be generalized to every possible lossy channel, every JPEG codec, every noise distribution, or all steganographic techniques.

### Empirical Findings:
1. **Flawless Lossless Operation:** Under the lossless PNG container channel, the classical baseline achieved a **100% recovery success rate** (20/20 runs), $\text{BER} = 0.000000$, valid carrier headers, and zero AES-GCM GHASH authentication errors.
2. **Catastrophic Failure Under JPEG:** Under JPEG compression at all tested quality factors ($Q = 100$ down to $Q = 70$), the classical baseline exhibited a **0.0% recovery success rate** (0/140 runs). The raw subband parity bit error rate hovered at approximately $47.59\% - 49.94\%$ (pure statistical noise) due to high-frequency 8x8 DCT quantization disrupting spatial integer parity relationships $(a - b - c + d) \pmod 2$. In 100% of JPEG runs (140/140), extraction halted at the 16-byte carrier header anchor (`CarrierHeaderError`), blocking downstream ECC decoding and decryption.
3. **Catastrophic Failure Under Additive Gaussian Noise:** Under additive zero-mean Gaussian noise (generated via a deterministic pseudo-random seed `np.random.RandomState(42)`), the classical baseline exhibited a **0.0% recovery success rate** across all tested noise levels (0/60 runs). Raw subband parity BER reached $49.10\%$ at $\sigma = 0.5$, $50.20\%$ at $\sigma = 1.0$, and $49.92\%$ at $\sigma = 2.0$. In 100% of Gaussian noise runs (60/60), extraction halted at the 16-byte carrier header (`CarrierHeaderError`).
4. **Perceptual Noise vs. Parity Sensitivity:** While $\sigma = 0.5$ produced an average received-image PSNR of $52.83\text{ dB}$ relative to the cover ($53.01\text{ dB}$ relative to the stego image), objective fidelity does not equate to universal imperceptibility. Perceptual visibility depends on image content, display conditions, and perturbation setup. Even micro-perturbations that preserve high PSNR disrupt modulo-2 spatial parity.
5. **Phase 3 Motivation (Conservative Framing):** E14 establishes a failure baseline for the tested classical integer-parity carrier method under the specified JPEG recompression and Gaussian-noise conditions. These results motivate evaluating robustness-aware alternatives, including the planned ACS-Sub-INN v3 architecture. Any improvement must be demonstrated through controlled experiments against the same baseline and channels. Continuous-domain embedding alone does not guarantee robustness, and neural performance remains an untested hypothesis until Phase 3 is implemented and empirically evaluated.

---

## 2. Channel Performance Summary Table

The table below summarizes all 220 executed experimental runs grouped by channel environment, transcribed directly from `docs/experiments/e14_results.json`:

| Channel Environment | Total Runs | Header Valid Rate | ECC Decode Success* | Verbatim Payload Recovery | Mean Raw Parity BER | Mean Received PSNR (vs. Cover) | Mean Channel Distortion PSNR (vs. Stego) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PNG (Lossless Control)** | 20 | 100.0% | 100.0% | **100.0%** | 0.0000% | 67.56 dB | $\infty$ dB |
| **JPEG (Q=100)** | 20 | 0.0% | 0.0% | **0.0%** | 47.5935% | 41.96 dB | 41.96 dB |
| **JPEG (Q=95)** | 20 | 0.0% | 0.0% | **0.0%** | 49.8676% | 38.63 dB | 38.63 dB |
| **JPEG (Q=90)** | 20 | 0.0% | 0.0% | **0.0%** | 49.9364% | 36.33 dB | 36.33 dB |
| **JPEG (Q=85)** | 20 | 0.0% | 0.0% | **0.0%** | 49.9314% | 34.88 dB | 34.88 dB |
| **JPEG (Q=80)** | 20 | 0.0% | 0.0% | **0.0%** | 49.9062% | 33.88 dB | 33.88 dB |
| **JPEG (Q=75)** | 20 | 0.0% | 0.0% | **0.0%** | 49.8275% | 33.13 dB | 33.13 dB |
| **JPEG (Q=70)** | 20 | 0.0% | 0.0% | **0.0%** | 49.8436% | 32.58 dB | 32.58 dB |
| **Gaussian Noise ($\sigma=0.5$)** | 20 | 0.0% | 0.0% | **0.0%** | 49.1016% | 52.83 dB | 53.01 dB |
| **Gaussian Noise ($\sigma=1.0$)** | 20 | 0.0% | 0.0% | **0.0%** | 50.1975% | 47.73 dB | 47.78 dB |
| **Gaussian Noise ($\sigma=2.0$)** | 20 | 0.0% | 0.0% | **0.0%** | 49.9234% | 42.01 dB | 42.02 dB |

*\*Note on ECC Decode Success: In all 200 lossy/noisy runs, extraction halted at the carrier header stage. Downstream Reed-Solomon decoding was never invoked on extracted payload data; the 0.0% rate reflects full-pipeline recovery failure at the header anchor.*

### Metric Definitions and Separation:
- **Stego-Image Fidelity Pre-Channel (`stego_psnr_pre_channel`):** Fidelity of the stego carrier relative to the original cover immediately after embedding ($64.88 - 70.29\text{ dB}$, mean $67.56\text{ dB}$).
- **Received-Image Fidelity Post-Channel (`stego_psnr_post_channel`):** Fidelity of the received image relative to the original cover after channel transmission.
- **Channel Distortion Relative to Stego (`channel_distortion_psnr`):** Distortion introduced solely by channel degradation relative to the stego image ($\infty\text{ dB}$ for lossless PNG; $32.58 - 41.96\text{ dB}$ for JPEG; $42.02 - 53.01\text{ dB}$ for Gaussian noise).
- **Raw Subband Parity BER (`raw_subband_parity_ber`):** The empirical fraction of inverted parity bits across all $H_{\text{sub}} \times W_{\text{sub}}$ coefficients of the Red-HH subband:
  $$\text{BER}_{\text{raw}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}_{\{ (2\text{HH}_{\text{stego}} \pmod 2) \ne (2\text{HH}_{\text{rec}} \pmod 2) \}}$$
- **Verbatim Payload Recovery (`exact_match`):** Binary boolean indicating whether the deserialized recovered payload exactly matches the input secret bytes (`True` in 20/20 PNG runs; `False` in 0/200 lossy runs).

---

## 3. Subsystem Failure Isolation & Root Cause Analysis

### 3.1 Pipeline Sequencing and Header-Failure Limitation
The PaladinMedia extraction pipeline executes sequentially:
$$\text{Received Image} \xrightarrow{\text{2D Haar DWT}} \text{Red-HH Subband} \xrightarrow{\text{Header Extract}} \text{Payload Extract} \xrightarrow{\text{RS Decode}} \text{AES-GCM Decrypt} \xrightarrow{\text{Deserialize}} \text{Secret}$$

The carrier framing reserves the first 16 bytes (128 bits) of the Red-HH subband for header metadata (magic bytes, version, payload length, flags, session counter). In the current classical baseline implementation, this carrier header is embedded directly via integer-parity modulation **without independent error-correcting codes**.

**Empirical Subsystem Finding:**
- In all 140 JPEG runs and all 60 Gaussian noise runs, the run records in `docs/experiments/e14_results.json` document `failure_mode: "HEADER_CORRUPTED"` and `header_valid: False`.
- Because header extraction failed, extraction halted before payload boundaries could be parsed.
- **Critical Subsystem Distinction:** Downstream Reed-Solomon decoding and AES-256-GCM authenticated decryption were **never exercised on an extracted, corrupted payload ciphertext** during any of the 200 lossy runs.
- E14 demonstrates the failure of the **complete end-to-end recovery pipeline**, with uncorrected carrier header corruption as the proximate point of failure.

### 3.2 Mathematical Mechanism of Integer-Parity Degradation
The classical baseline embeds secret bits into 2x2 spatial pixel quads $[a, b; c, d]$ via the 2D Haar HH subband:
$$2\text{HH} = a - b - c + d$$
Embedding adjusts the bottom-right pixel $d$ such that $(a - b - c + d) \pmod 2 = \text{bit}_i$.

Under JPEG compression:
1. Spatial blocks ($8 \times 8$) are transformed via DCT into frequency coefficients.
2. High-frequency coefficients are quantized by dividing by $Q_{uv}$ and rounding to integers.
3. Inverse DCT and spatial integer clamping $(\text{round}(x))$ reconstruct spatial pixels.

Even at $Q=100$, quantization roundoff perturbs individual spatial pixel values by $\pm 1$. Because parity is evaluated modulo 2:
$$(d \pm 1) \pmod 2 = (d \pmod 2) \oplus 1$$
Every odd perturbation in any of the four quad pixels flips the extracted parity bit. Across an entire subband, independent quantization noise yields a raw parity bit error rate of $p \approx 0.50$ (empirically $47.59\% - 49.94\%$).

The probability of all 128 uncorrected carrier header bits surviving intact under an independent bit error rate $p$ is:
$$P(\text{Header Intact}) = (1 - p)^{128}$$
For $p = 0.48$, $P(\text{Header Intact}) \approx (0.52)^{128} \approx 6.4 \times 10^{-37}$. Consequently, carrier header survival is statistically impossible under uncorrected integer-parity modulation in lossy channels.

### 3.3 Theoretical Reed-Solomon Error-Correction Limits
The payload protection layer employs systematic Reed-Solomon $\text{RS}(255, 223)$, which appends $2t = 32$ parity symbols per 255-byte block.

**Exact Theoretical Boundaries:**
- **Symbol Correction Budget:** $\text{RS}(255, 223)$ can correct at most $t = 16$ erroneous 8-bit symbols per 255-byte block.
- **Symbol Error Rate Threshold:** Maximum correctable symbol error rate is $t / n = 16 / 255 \approx 6.27\%$.
- **Distinction Between BER and Symbol Error Rate:** Bit error rate (BER) measures bit-level inversions. Byte-symbol error rate ($P_s$) measures the probability that an 8-bit byte contains at least one bit error. A single bit error corrupts the entire 8-bit symbol.
- **Theoretical Error Model:** Assuming independent and identically distributed random bit errors at error rate $p$, the theoretical byte-symbol error rate is modeled by:
  $$P_s = 1 - (1 - p)^8$$
  * At $p = 0.001$ ($0.1\%$ BER), $P_s \approx 0.797\% < 6.27\%$ (correctable).
  * At $p = 0.008$ ($0.8\%$ BER), $P_s \approx 6.22\% \approx 6.27\%$ (correction boundary).
  * At $p = 0.500$ ($50.0\%$ BER, as observed under JPEG and noise), theoretical $P_s = 1 - (0.5)^8 = 99.61\%$, overwhelmingly exceeding the $6.27\%$ symbol capacity.

*Note:* This relationship is a theoretical mathematical model for independent random bit errors. It is not an empirical measurement of payload symbols in E14, because lossy extraction halted at the carrier header before payload symbols were decoded. Furthermore, RS(255, 223) does not have an intrinsic "3% bit error limit"; its mathematical bound is strictly 16 symbol errors per 255-byte codeword.

---

## 4. Distinction Between Fidelity, Robustness, and Steganalysis

E14 evaluates channel robustness under specific lossy channels, **not steganographic detectability**.

1. **Fidelity vs. Undetectability:** High PSNR ($> 64\text{ dB}$) and SSIM ($> 0.9999$) demonstrate low mathematical distortion relative to the cover image. They do not constitute evidence of steganographic security or resistance to steganalysis.
2. **Steganalysis Not Evaluated:** E14 did not perform cover/stego classification, feature extraction (e.g., Spatial Rich Models / SRM, SPAM), or residual histogram analysis. Steganographic undetectability remains uncharacterized.
3. **No Comparative Claims for Neural Models:** E14 does not demonstrate or prove that ACS-Sub-INN v3 is more secure, less detectable, or more robust than the classical baseline. Such claims require independent empirical testing following neural implementation and training.

---

## 5. Untested Transformations (NOT TESTED IN E14)

The following transformations and evaluation vectors were **NOT TESTED IN E14**:
- **Spatial Resizing & Scaling:** Downscaling, upscaling, or interpolation (bilinear, bicubic, Lanczos).
- **Geometric Transformations:** Cropping, rotation, shearing, translation, or aspect ratio modification.
- **Display & Capture Channels:** Screen capture, print-and-scan, or optical camera capture.
- **Social Media Pipelines:** Transcoding, aggressive re-compression, or container stripping by social platforms.
- **Alternative Codecs:** WebP, AVIF, JPEG-XL, or non-Pillow JPEG encoder implementations.
- **Statistical Steganalysis:** SRM, maxSRM, SPAM, or deep-learning steganalysers.
- **Neural Model Robustness:** ACS-Sub-INN v3 performance under channel degradation.

These areas represent unexecuted experimental gaps that must be formally evaluated in subsequent phases.

---

## 6. Phase 3 Architecture Implications

The empirical failure of the classical integer-parity baseline under the tested JPEG and noise conditions provides the primary scientific motivation for Phase 3:

1. **Failure Baseline Established:** E14 establishes an empirical negative baseline: classical 1-level Haar Red-HH integer-parity modulation cannot survive JPEG compression ($Q \le 100$) or additive Gaussian noise ($\sigma \ge 0.5$) under the evaluated setup.
2. **Motivation for Continuous Flow:** Robustness requires modulating continuous wavelet features via invertible coupling blocks rather than fragile discrete integer parities.
3. **Mandatory Header Protection:** Future carrier framing must employ robust, error-tolerant synchronization (e.g., high-redundancy repetition coding, DWT low-frequency anchor, or implicit spatial dimensioning) to avoid single-point header failure.
4. **Controlled Evaluation Requirement:** Any claim of neural superiority in Phase 3 must be validated against the exact channel conditions and baseline figures established in E14.

---

## 7. Traceability & Reproduction

- **Canonical Runner:** [`scripts/experiments/run_e14.py`](file:///E:/Paladin_media/scripts/experiments/run_e14.py)
- **Machine-Readable Run Records:** [`docs/experiments/e14_results.json`](file:///E:/Paladin_media/docs/experiments/e14_results.json)
- **Dataset Evaluated:** DIV2K High-Resolution (`0001.png` – `0005.png`)
- **Execution Command (Historical):**
  ```powershell
  python scripts/experiments/run_e14.py
  ```
