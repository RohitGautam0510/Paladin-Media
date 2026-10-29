# Experiment E04: Carrier Spatial Scaling & Content Analysis

## Document Control
- **Experiment ID:** E04
- **Phase:** Phase 2 — Signal Processing + Carrier Infrastructure
- **Date:** 2026-10-09
- **Status:** COMPLETED & VERIFIED
- **Related Specifications:** PaladinMedia Architecture Specification §10/§11/§12, IRS §6/§9
- **Evidence Path:** `docs/experiments/e04_results.json`

---

## 1. Objective & Hypothesis
* **Objective:** Characterize the capacity, image fidelity, and computational overhead of carrier spatial scaling, and evaluate the effect of carrier content diversity using the DIV2K dataset.
* **Hypothesis:** Theoretical capacity scales quadratically ($O(N^2)$) with carrier side length $N$. With integer-parity modulation on Haar coefficients, high carrier fidelity (PSNR $> 75\text{ dB}$, $\text{SSIM} = 1.0$) and exact byte recovery are invariant across heterogeneous carrier textures.

---

## 2. Dataset Preflight & Native Availability
* **Source Dataset:** `DIV2K_train_LR_bicubic_X2/DIV2K_train_LR_bicubic/X2/` (800 PNG images, 8-bit RGB).
* **Observed Dimensions:** Maximum width and height across all 800 images are $1020$ pixels.
* **Native Crop Restrictions:** Per experimental design rules, synthetic upscaling is strictly prohibited. Configurations requiring dimensions $\ge 1024 \times 1024$ are unavailable natively and marked as skipped.

---

## 3. Spatial Resolution & Capacity Scaling Frontier

| Target Dimension | Subband Dimension | Raw Positions | Raw Capacity (B) | Max Protected (B) | Usable Secret (B) | Status | Execution / Skip Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **512 × 512** | $256 \times 256$ | 65,536 | 8,192 B | 8,176 B | 7,090 B | **EXECUTED** | Native center crops from DIV2K images |
| **1024 × 1024** | $512 \times 512$ | 262,144 | 32,768 B | 32,752 B | 28,498 B | **SKIPPED** | Max native dimension is 1020 px; upscaling prohibited |
| **2048 × 2048** | $1024 \times 1024$ | 1,048,576 | 131,072 B | 131,056 B | 114,353 B | **SKIPPED** | Native HR dataset absent; upscaling prohibited |
| **3072 × 3072** | $1536 \times 1536$ | 2,359,296 | 294,912 B | 294,896 B | 257,742 B | **SKIPPED** | Native HR dataset absent; upscaling prohibited |
| **4096 × 4096** | $2048 \times 2048$ | 4,194,304 | 524,288 B | 524,272 B | 458,219 B | **SKIPPED** | Native HR dataset absent; upscaling prohibited |
| **5120 × 5120** | $2560 \times 2560$ | 6,553,600 | 819,200 B | 819,184 B | 716,230 B | **SKIPPED** | Native HR dataset absent; upscaling prohibited |

---

## 4. Empirical DIV2K Evaluations ($512 \times 512$ Native Center Crops)

Evaluated across representative DIV2K images with diverse textures (smooth gradients, high-frequency edges, complex foliage):

| Image ID | Original Size | Crop Box ($x_0, y_0, w, h$) | PSNR (dB) | SSIM | MAE | BER | Exact Match | Total Pipeline Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0001x2.png` | $1020 \times 678$ | $(254, 83, 512, 512)$ | 75.7712 dB | 1.000000 | 0.001722 | 0.0 | **TRUE (100%)** | 129.4 ms |
| `0002x2.png` | $1020 \times 680$ | $(254, 84, 512, 512)$ | 75.8850 dB | 1.000000 | 0.001677 | 0.0 | **TRUE (100%)** | 131.2 ms |
| `0003x2.png` | $1020 \times 678$ | $(254, 83, 512, 512)$ | 75.7520 dB | 1.000000 | 0.001729 | 0.0 | **TRUE (100%)** | 128.7 ms |
| `0004x2.png` | $1020 \times 678$ | $(254, 83, 512, 512)$ | 75.9883 dB | 1.000000 | 0.001638 | 0.0 | **TRUE (100%)** | 130.5 ms |
| `0005x2.png` | $1020 \times 678$ | $(254, 83, 512, 512)$ | 75.7744 dB | 1.000000 | 0.001721 | 0.0 | **TRUE (100%)** | 132.0 ms |

---

## 5. Key Findings & Operating Limits
1. **Content Independence:** The integer-parity modulation operates with mathematical independence from the carrier's underlying spatial texture. In all cases, zero bit errors occurred during the lossless PNG round trip.
2. **Quadratic Scaling Rule:** Usable capacity scales strictly as $\approx 0.027 \times H \times W$ bytes for 1 HH channel. A $2048 \times 2048$ carrier would theoretically support $\approx 114\text{ KB}$ of protected payload.
3. **Hardware & Environment Constraint:** High-resolution testing is constrained by repository dataset availability; acquiring native 2K/4K uncompressed datasets is recommended for future testing if single-carrier high payloads are needed.
