# PaladinMedia — Phase 3 Evidence Gate & Acceptance Criteria Specification

**Document ID:** PDM-P3-GATE-001
**Status:** RECONCILED POST-AUDIT SPECIFICATION
**Target Phase:** Phase 3 — Neural Wavelet Steganography (ACS-Sub-INN v3 Engine)
**Governing Architecture:** ACS-Sub-INN Frozen Architecture Specification v3
**Audit Reference:** `PDM-AUDIT-P3-GATE-20261012`

---

## 1. Architectural Scope & Invariant Constraints

Phase 3 introduces the **ACS-Sub-INN** (Adaptive Coefficient Steganography Subband Invertible Neural Network) to modulate continuous wavelet coefficients in high-frequency detail subbands.

### Invariant Architectural Constraints
1. **Preserve Cryptographic Pipeline:** The strict unidirectional sequence `Serialize -> Encrypt (AES-256-GCM) -> Error Correction (Reed-Solomon RS(255, 223))` remains non-negotiable. The neural network operates strictly on the protected bitstream and never encounters unauthenticated plaintext.
2. **Frozen Architecture Maintenance:**
   - **No Secret-Side DWT:** Secret media is serialized and protected into an opaque bitstream; secret images are never decomposed via DWT.
   - **No Discarded Latent-Z:** Full invertible bidirectional flow without unrecoverable latent discarding.
   - **Transform-Domain Only:** Modulation operates strictly in the wavelet transform domain.
3. **Lossless Recovery Priority:** Verbatim byte recovery (`ExactMatch = True`, post-ECC $\text{BER} = 0.0$) through the carrier container must not be compromised by neural approximations.

---

## 2. Classical Benchmark Baseline

The ACS-Sub-INN neural model will be benchmarked directly against the audited Phase 2 classical baseline:

| Dimension / Parameter | Classical Phase 2 Baseline Standard | ACS-Sub-INN Neural Target |
|:---|:---|:---|
| **Transform** | 1-level 2D Haar DWT | 1-level 2D Haar DWT |
| **Carrier Subband** | Red channel of HH detail subband | 3-channel (RGB) HH detail subbands (Base), or all HF subbands (Ext) |
| **Modulation Scheme** | Integer-parity modulation on $2\text{HH} = a - b - c + d$ | Continuous invertible affine coupling blocks |
| **Capacity (512 × 512)** | 7,090 B usable (1 Ch), 21,362 B usable (3 Ch) | Base Gate: $\ge 21,362\text{ B}$; High-Capacity Gate: $\ge 49,152\text{ B}$ |
| **Capacity (1024 × 1024)** | 28,574 B usable (1 Ch), 85,570 B usable (3 Ch) | Base Gate: $\ge 85,570\text{ B}$; High-Capacity Gate: $\ge 196,608\text{ B}$ |
| **Spatial Payload Density** | $0.218\text{ bpp}_{\text{spatial}}$ (1 Ch), $0.654\text{ bpp}_{\text{spatial}}$ (3 Ch) | Base: $\ge 0.654\text{ bpp}_{\text{spatial}}$; High-Capacity: $\ge 1.5 - 3.0\text{ bpp}_{\text{spatial}}$ |
| **Carrier Container** | Lossless 8-bit RGB PNG only | Dual-mode: Lossless 8-bit PNG & Robust JPEG ($Q \ge 80$) |
| **Baseline Fidelity** | PSNR: $62.0 - 75.9\text{ dB}$, SSIM: $> 0.9999$ | PNG: $\ge 60.0\text{ dB}$; JPEG: $\ge 42.0\text{ dB}$ |

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

## 4. Rigorous Acceptance Criteria (Audited Gated Metrics)

To justify the computational and deployment overhead of a neural network over the classical integer-parity baseline, the ACS-Sub-INN must satisfy the following gated criteria:

### Gate 1: Cryptographic Integrity & Verbatim Recovery (P0 - Hard Requirement)
- **Pre-ECC Inversion Bit Error Rate (BER):**
  - *Lossless PNG Container:* $\text{BER}_{\text{pre-ECC}} \le 1.0 \times 10^{-3}$ ($0.1\%$).
  - *Lossy JPEG Channel ($Q \ge 80$):* $\text{BER}_{\text{pre-ECC}} \le 3.0 \times 10^{-2}$ ($3.0\%$, safely within Reed-Solomon RS(255, 223) correction capacity of $t=16$ symbols/block).
- **Post-ECC Final Payload Recovery BER:** Must be strictly $\mathbf{0.000000}$ across 100% of test runs through container transmission.
- **RS Decoding & AES-GCM Authentication:** 100% valid authentication tags; zero GHASH tag verification failures.
- **Plaintext Equality:** Verbatim byte match in 100% of non-tampered test runs:
  $$\text{recovered\_payload.data} == \text{original\_secret.data}$$

### Gate 2: Stego Visual Fidelity & Perceptual Quality (P0 / P1)
- **Lossless Container Mode (PNG):**
  - **PSNR:** $\ge 60.0\text{ dB}$ across all test images at 100% capacity load (matching classical baseline fidelity).
  - **SSIM:** $\ge 0.9995$ across all test images.
  - **Spatial MAE:** $\le 0.05$; Maximum absolute pixel error $\le 2$.
- **Robust Lossy Channel Mode (JPEG $Q \ge 80$):**
  - **Stego vs. Cover PSNR (pre-channel):** $\ge 42.0\text{ dB}$.
  - **SSIM (pre-channel):** $\ge 0.9850$.
  - **LPIPS (AlexNet/VGG):** $\le 0.050$.

### Gate 3: Channel Robustness (The Justification Gate - P1)
*The classical integer-parity baseline fails completely under the lossy channels evaluated in E14 (JPEG compression at quality factors $Q \in [70, 100]$ and additive Gaussian noise $\sigma \ge 0.5$; spatial downsampling remains untested). Neural steganography is justified only if it confers demonstrable channel resilience or statistical undetectability:*
- **JPEG Resistance (Primary Target):** Verbatim payload recovery ($\text{BER}_{\text{post-ECC}} = 0.0$) through lossy JPEG recompression down to Quality factor $Q = 80$.
- **Statistical Undetectability (Secondary / Dual Target):** If evaluated under uncompressed/lossless channels, the neural approach must demonstrate a statistically significant ($p < 0.01$) advantage in steganalytic undetectability (e.g., lower ROC-AUC under Spatial Rich Models / SRM or SPAM classifiers) compared to the classical baseline at matched payload capacity.

### Gate 4: Capacity & Hard Gate Compliance (P0)
- **Metric Definitions & Denominators:**
  - *Spatial Payload Density:* $\text{bpp}_{\text{spatial}} = \frac{L_{\text{bits}}}{H \times W}$, where $H \times W$ is the spatial pixel dimension of the RGB carrier.
  - *Transform Coefficient Density:* $\text{bpp}_{\text{coeff}} = \frac{L_{\text{bits}}}{H_{\text{sub}} \times W_{\text{sub}} \times C_{\text{sub}}}$, where $H_{\text{sub}} \times W_{\text{sub}}$ is the subband dimension ($H/2 \times W/2$) and $C_{\text{sub}}$ is the number of modulated subbands/channels.
- **Base Parity Gate:** Achieve $\ge 0.654\text{ bpp}_{\text{spatial}}$ (matching 3-channel classical capacity: $21,362\text{ B}$ on $512\times512$, $85,570\text{ B}$ on $1024\times1024$).
- **High-Capacity Neural Gate:** Scalable to $\ge 1.5 - 3.0\text{ bpp}_{\text{spatial}}$ when modulating all high-frequency subbands ($\text{LH, HL, HH}$) with multi-bit continuous modulation.
- **Fail-Closed Hard Gate:** Strict enforcement of `PayloadCapacityExceededError` whenever $L > C_{\text{allocated}}$. Zero unauthenticated plaintext leakage on capacity overflow.

### Gate 5: Computational & Hardware Budgets (Separated Research vs. Deployment)
- **Phase 3 Research Gate (FP32 PyTorch):**
  - **Inference Latency:** Total forward embedding (DWT + INN + IDWT) $\le 250\text{ ms}$ on target desktop GPU (or $\le 1000\text{ ms}$ on 8-core CPU).
  - **Peak Process Memory:** $\le 512\text{ MB}$ working set during $1024\times1024$ carrier inference.
- **Downstream Edge / Deployment Gate (Post-Phase 3):**
  - **Format:** Exportable to ONNX Runtime / TensorRT.
  - **Quantization:** Support FP16 or INT8 quantization without exceeding the RS(255, 223) error correction budget.
  - **Mobile/Edge Latency:** $\le 250\text{ ms}$ on mobile CPU/NPU.

---

## 5. Dataset Manifest & Partitioning Protocol

Neural training, validation, and evaluation must strictly prevent carrier leakage and data snooping:

- **Source Dataset:** DIV2K High-Resolution dataset (`dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR`, 800 images).
- **Canonical Split Manifest:** [`docs/project/div2k_hr_split.json`](file:///E:/Paladin_media/docs/project/div2k_hr_split.json) (Composite SHA-256: `d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed`).
- **Partition Policy:**
  - **Training Set:** 600 images (Images `0001.png` through `0600.png`).
  - **Validation Set:** 100 images (Images `0601.png` through `0700.png`).
  - **Held-Out Test Set:** 100 images (Images `0701.png` through `0800.png`).
- **Crop Policy:** Deterministic center crop or fixed-seed pseudo-random cropping during training; strictly deterministic center crops during evaluation.
- **Data Integrity:** Test set images must never be seen during training, hyperparameter tuning, or Context Predictor calibration.

---

## 6. Revision and Rejection Criteria

The Phase 3 neural approach shall be **revised or rejected** if any of the following occur:
1. **Uncorrectable Bit Errors in Lossless Container:** If floating-point quantization in the INN produces $\text{BER}_{\text{pre-ECC}} > 1.0 \times 10^{-3}$ through lossless 8-bit PNG that cannot be repaired within the RS(255, 223) error budget.
2. **Negative Distortion Trade-off:** If ACS-Sub-INN achieves worse PSNR ($< 60\text{ dB}$) than the classical baseline ($62.0 - 75.9\text{ dB}$) in lossless container mode without providing lossy channel robustness.
3. **Steganalytic Inferiority:** If neural coefficient modifications introduce recognizable statistical artifacts that increase detectability compared to the classical baseline without a commensurate increase in capacity or robustness.
4. **Catastrophic Latency Overhead:** If inference exceeds $5\times$ the classical roundtrip time without meeting channel robustness objectives.
