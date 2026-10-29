# Experiment E06: Supported Embedding Densities Evaluation

## Document Control
- **Experiment ID:** E06
- **Phase:** Phase 2 — Signal Processing + Carrier Infrastructure
- **Date:** 2026-10-09
- **Status:** COMPLETED & VERIFIED
- **Related Specifications:** PaladinMedia Architecture Specification §10/§12, IRS §6/§9
- **Evidence Path:** `docs/experiments/e06_results.json`

---

## 1. Objective & Hypothesis
* **Objective:** Evaluate the mathematical feasibility, distortion impact, and extraction fidelity of increasing embedding density from the baseline $1\text{ bit/coefficient}$ to multi-bit modulation ($2\text{ bits/coefficient}$ and $3\text{ bits/coefficient}$) within the integer wavelet domain.
* **Hypothesis:** Higher embedding densities multiply raw capacity by $2\times$ and $3\times$ within the same subband area, but increase spatial pixel perturbation ($\max |\Delta|$ increases from $1$ to $4$) and reduce carrier PSNR proportionally to $\Delta^2$. Zero bit errors can only be maintained if boundary saturation ($0$ and $255$ uint8 limits) is rigorously handled.

---

## 2. Modulation Mechanics & Modulo Arithmetic
1. **1 bit/coefficient ($M=2$, Baseline):**
   - Condition: $(2 \times \text{HH}) \bmod 2 = \text{bit}$.
   - Optimal perturbation: $\Delta \in \{-1, 0, +1\}$ on spatial pixel $a$.
2. **2 bits/coefficient ($M=4$):**
   - Condition: $(2 \times \text{HH}) \bmod 4 = \text{symbol} \in \{0, 1, 2, 3\}$.
   - Optimal perturbation: $\Delta \in \{-1, 0, +1, +2\}$ on spatial pixel $a$.
3. **3 bits/coefficient ($M=8$):**
   - Condition: $(2 \times \text{HH}) \bmod 8 = \text{symbol} \in \{0 \dots 7\}$.
   - Optimal perturbation: $\Delta \in \{-3, -2, -1, 0, +1, +2, +3, +4\}$ on spatial pixel $a$.
   - **Boundary Risk:** In extreme dark ($a < 3$) or light ($a > 252$) regions, a perturbation of $\pm 4$ risks uint8 saturation clipping, which flips extracted bits unless reversible fallback wrapping is enforced.

---

## 3. Empirical Results Across Densities

Evaluated on `MVP 0 Dataset/0001x2.png` ($512 \times 512$ RGB, Channel 0) with secret fixture `tests/fixtures/secret_small.png`:

| Density | Modulus ($M$) | Raw Capacity | Usable Plaintext | PSNR (dB) | SSIM | Max Pixel Error | MAE | BER | Exact Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 bit/coeff** | 2 | 8,192 B | 7,090 B | **75.7202 dB** | 1.000000 | **1** | 0.00168 | **0.0** | **TRUE (100%)** |
| **2 bits/coeff** | 4 | 16,384 B | 14,226 B | **74.2409 dB** | 0.999999 | **2** | 0.00241 | **0.0** | **TRUE (100%)** |
| **3 bits/coeff** | 8 | 24,576 B | 21,362 B | **70.3748 dB** | 0.999999 | **4** | 0.00512 | **0.0** | **TRUE (100%)** |

---

## 4. Perceptual and Robustness Trade-offs
1. **Distortion Scaling:** Going from 1 to 3 bits/coeff drops PSNR by $5.35\text{ dB}$ (from $75.72\text{ dB}$ down to $70.37\text{ dB}$). While still perceptually transparent to the naked human eye ($> 70\text{ dB}$), spatial pixel differences reach $\pm 4$.
2. **Steganalysis Exposure:** Higher perturbation amplitudes create distinguishable histograms in difference matrices ($a - b - c + d$), significantly increasing vulnerability to higher-order steganalyzers (e.g., SPAM, SRM).
3. **Recommendation for Phase 3:**
   - For classical Phase 2 baseline: **1 bit/coefficient remains the safest canonical configuration** due to minimal perturbation ($\max |\Delta| = 1$) and zero saturation risk.
   - For Phase 3 INN: Higher embedding densities should be handled by the neural flow architecture (which naturally redistributes continuous latent variables across the entire frequency spectrum) rather than large discrete spatial jumps.
