# PaladinMedia — Phase 3 Evidence Gate & Acceptance Criteria Specification

**Document ID:** PDM-P3-GATE-001  
**Status:** DRAFT / PROVISIONAL ACCEPTANCE CRITERIA  
**Target Phase:** Phase 3 — Neural Wavelet Steganography (ACS-Sub-INN v3 Engine)  
**Governing Architecture:** ACS-Sub-INN Frozen Architecture Specification v3  

---

## 1. Architectural Scope & Invariant Constraints

Phase 3 introduces the **ACS-Sub-INN** (Adaptive Coefficient Steganography Subband Invertible Neural Network) to modulate continuous wavelet coefficients.

### Invariant Architectural Constraints
1. **Preserve Cryptographic Pipeline:** The strict unidirectional sequence `Serialize -> Encrypt (AES-256-GCM) -> Error Correction (Reed-Solomon RS(255, 223))` remains non-negotiable. The neural network never encounters unauthenticated plaintext.
2. **Frozen Architecture Maintenance:** 
   - **No Secret-Side DWT:** Secret media is serialized and protected into an opaque bitstream; secret images are never decomposed via DWT.
   - **No Discarded Latent-Z:** Full invertible bidirectional flow without unrecoverable latent discarding.
   - **No Carrier Redistribution Redesign:** Modulation operates strictly in the transform domain.
3. **Lossless Recovery Priority:** Verbatim byte recovery (`ExactMatch = True`, $\text{BER} = 0.0$) through the carrier container must not be compromised by neural approximations.

---

## 2. Classical Benchmark Baseline

The ACS-Sub-INN neural model will be benchmarked directly against the audited Phase 2 classical baseline:

| Dimension / Parameter | Classical Phase 2 Baseline Standard | ACS-Sub-INN Neural Target |
|:---|:---|:---|
| **Transform** | 1-level 2D Haar DWT | 1-level 2D Haar DWT |
| **Carrier Subband** | Red channel of HH detail subband | 3-channel (RGB) HH detail subbands |
| **Modulation Scheme** | Integer-parity modulation on $2\text{HH} = a - b - c + d$ | Continuous invertible affine coupling blocks |
| **Capacity (512 × 512)** | 7,090 B usable (1 Ch), 21,362 B usable (3 Ch) | Matched capacity ($\ge 21,362\text{ B}$) |
| **Capacity (1024 × 1024)** | 28,574 B usable (1 Ch), 85,570 B usable (3 Ch) | Matched capacity ($\ge 85,570\text{ B}$) |
| **Carrier Container** | Lossless 8-bit RGB PNG | Lossless 8-bit RGB PNG & Robust JPEG ($Q \ge 75$) |
| **Baseline Fidelity** | PSNR: $62.0 - 75.9\text{ dB}$, SSIM: $> 0.9999$ | Matched or superior perceptual metrics |

---

## 3. Matched-Capacity & Matched-Distortion Comparison Protocol

Evaluation must follow rigorous scientific dual-frontier benchmarking:

### Protocol A: Matched-Capacity Evaluation
- Fix the protected payload size $L$ identically across Classical Baseline and Neural Model:
  - Low load: 1,500 bytes
  - Medium load: 7,000 bytes
  - High load (1024): 20,000 bytes
  - Near-capacity boundary: 100% of allocated capacity
- Measure and compare:
  - Stego visual fidelity: PSNR (dB), SSIM, LPIPS (Learned Perceptual Image Patch Similarity).
  - High-frequency residual energy across unmodulated subbands ($\text{LL}, \text{LH}, \text{HL}$).
  - Steganalysis detection resistance (e.g., against spatial/frequency domain classifiers).

### Protocol B: Matched-Distortion Evaluation
- Fix the target distortion threshold (e.g., $\text{PSNR} = 65.0\text{ dB}$ or $\text{PSNR} = 60.0\text{ dB}$).
- Measure the maximum embeddable secret payload (in bits per pixel / bytes) achievable while meeting the target distortion.

---

## 4. Rigorous Acceptance Criteria (Provisional Gates)

To justify the computational and deployment overhead of a neural network over the classical integer-parity baseline, the ACS-Sub-INN must meet the following provisional gates:

### Gate 1: Cryptographic Integrity & Verbatim Recovery (P0 - Hard Requirement)
- **Carrier Bit Error Rate (BER):** Must be strictly $\mathbf{0.000000}$ through 8-bit RGB PNG serialization and reload.
- **RS Decoding & AES-GCM Authentication:** 100% valid authentication tags; zero GHASH failures.
- **Plaintext Equality:** Verbatim byte match in 100% of non-corrupted test runs:
  $$\text{recovered\_payload.data} == \text{original\_secret.data}$$

### Gate 2: Stego Visual Fidelity & Perceptual Quality
- **PSNR:** $\ge 60.0\text{ dB}$ across all test images at 100% capacity load.
- **SSIM:** $\ge 0.9995$ across all test images.
- **Pixel Delta Bound:** Mean Absolute Error (MAE) $\le 0.05$; Maximum absolute pixel error $\le 2$ in spatial domain.

### Gate 3: Channel Robustness (The Justification Gate)
*Classical integer-parity modulation fails under lossy channels (JPEG, noise, spatial scaling). Neural steganography is justified only if it confers demonstrable robustness:*
- **JPEG Resistance (Provisional Target):** Survived payload recovery through JPEG compression down to Quality factor $Q=80$ (with RS ECC correction).
- If JPEG robustness is not achieved, the neural approach must demonstrate a statistically significant ($p < 0.01$) advantage in steganalytic undetectability (e.g., lower ROC-AUC under SRM / StegExpose classifiers) at matched capacity.

### Gate 4: Capacity & Hard Gate Compliance
- Strict enforcement of `PayloadCapacityExceededError` whenever $L > C_{\text{allocated}}$.
- Zero state corruption or unauthenticated plaintext leakage on capacity overflow.

### Gate 5: Computational & Memory Budgets
- **Inference Latency:** Total forward embedding (DWT + INN + IDWT) $\le 250\text{ ms}$ on target desktop GPU (or $\le 1000\text{ ms}$ on target multi-core CPU).
- **Peak Process Memory:** $\le 512\text{ MB}$ working set during 1024 × 1024 carrier processing.

---

## 5. Dataset Manifest & Partitioning Protocol

Neural training and validation must strictly prevent carrier leakage and data snooping:

- **Source Dataset:** DIV2K High-Resolution dataset (`dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR`, 800 images).
- **Split Standard:**
  - **Training Set:** 600 images (Images `0001.png` through `0600.png`).
  - **Validation Set:** 100 images (Images `0601.png` through `0700.png`).
  - **Held-Out Test Set:** 100 images (Images `0701.png` through `0800.png`).
- **Crop Policy:** Deterministic center crop or fixed-seed pseudo-random cropping during training; strictly deterministic center crops during evaluation.
- **Data Integrity:** Test set images must never be seen during training, hyperparameter tuning, or Context Predictor calibration.

---

## 6. Revision and Rejection Criteria

The Phase 3 neural approach shall be **revised or rejected** if any of the following occur:
1. **Uncorrectable Bit Errors in Lossless Container:** If floating-point quantization in the INN produces non-zero BER through lossless 8-bit PNG that cannot be repaired within the RS(255, 223) error budget.
2. **Negative Distortion Trade-off:** If ACS-Sub-INN achieves worse PSNR ($< 60\text{ dB}$) than the classical baseline ($61.9 - 75.9\text{ dB}$) without providing lossy channel robustness.
3. **Steganalytic Inferiority:** If neural coefficient modifications introduce recognizable statistical artifacts that increase detectability compared to the sparse localized modifications of the classical baseline.
4. **Catastrophic Latency Overhead:** If inference exceeds $5\times$ the classical roundtrip time without meeting channel robustness objectives.
