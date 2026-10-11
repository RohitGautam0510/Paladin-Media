# Experiment E08: DWT Decomposition Level Evaluation

## Document Control
- **Experiment ID:** E08
- **Phase:** Phase 2 — Signal Processing + Carrier Infrastructure
- **Date:** 2026-10-09
- **Status:** COMPLETED & VERIFIED
- **Related Specifications:** PaladinMedia Architecture Specification §11/§12, IRS §9
- **Evidence Path:** `docs/experiments/e08_results.json`

---

## 1. Objective & Hypothesis
* **Objective:** Evaluate the coefficient geometry, spatial receptive field, invertibility error, and capacity scaling across 1-level, 2-level, and 3-level 2D Haar DWT decompositions.
* **Hypothesis:** Higher decomposition levels recursively divide spatial resolution by $2\times$ at each octave ($H/2^L, W/2^L$), decreasing available embedding positions by $75\%$ per level ($4\times$ drop). Furthermore, spatial receptive fields expand from $2 \times 2$ (Level 1) to $4 \times 4$ (Level 2) and $8 \times 8$ (Level 3), spreading spatial perturbations over wider pixel blocks upon IDWT synthesis.

---

## 2. Decomposition Geometry & Mathematical Analysis
For an input $512 \times 512 \times 3$ carrier:
1. **Level 1 ($\text{HH}_1$):**
   - Subband dimension: $256 \times 256$
   - Spatial Receptive Field: $2 \times 2$ spatial pixels
   - Eligible coefficient positions: $65,536$
2. **Level 2 ($\text{HH}_2$ on $\text{LL}_1$):**
   - Subband dimension: $128 \times 128$
   - Spatial Receptive Field: $4 \times 4$ spatial pixels
   - Eligible coefficient positions: $16,384$ ($25\%$ of Level 1)
3. **Level 3 ($\text{HH}_3$ on $\text{LL}_2$):**
   - Subband dimension: $64 \times 64$
   - Spatial Receptive Field: $8 \times 8$ spatial pixels
   - Eligible coefficient positions: $4,096$ ($6.25\%$ of Level 1)
4. **Multiscale Pyramid ($\text{HH}_1 + \text{HH}_2 + \text{HH}_3$):**
   - Total eligible positions: $65,536 + 16,384 + 4,096 = 86,016$ ($+31.25\%$ over Level 1 alone).

---

## 3. Empirical Geometry & Invertibility Measurements

| Level | Subband Grid | Spatial Receptive Field | Raw Capacity | Max Protected | Usable Plaintext | Invertibility Max Error | Baseline Payload Fits |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1** | $256 \times 256$ | $2 \times 2$ pixels | 8,192 B | 8,176 B | 7,090 B | **0.00000000** | **YES** ($316\text{ B} \le 8,176\text{ B}$) |
| **Level 2** | $128 \times 128$ | $4 \times 4$ pixels | 2,048 B | 2,032 B | 1,515 B | **0.00000000** | **YES** ($316\text{ B} \le 2,032\text{ B}$) |
| **Level 3** | $64 \times 64$ | $8 \times 8$ pixels | 512 B | 496 B | 177 B | **0.00000000** | **NO** ($316\text{ B} > 177\text{ B}$ usable) |
| **Multiscale** | $256^2 + 128^2 + 64^2$ | Multi-scale | 10,752 B | 10,736 B | 9,320 B | **0.00000000** | **YES** ($316\text{ B} \le 10,736\text{ B}$) |

---

## 4. Signal Processing & Architectural Trade-offs
1. **Capacity Depletion at High Octaves:** Restricting embedding to Level 2 or Level 3 alone drastically depresses usable capacity. Level 3 cannot even fit the minimal 316-byte RS-protected payload without exceeding the 177-byte usable secret threshold.
2. **Spatial Delocalization:** Because a Level 2 wavelet coefficient represents the average contrast over a $4 \times 4$ pixel quad, any perturbation at Level 2 diffuses into 16 spatial pixels during 2-stage IDWT, creating softer, larger-scale luminance ripples compared to the compact single-pixel adjustment at Level 1.
3. **Architectural Justification for Level 1 Baseline:**
   - Level 1 provides maximal raw capacity (8,192 B per channel).
   - Level 1 confines spatial adjustments to the tightest possible receptive field ($2 \times 2$).
   - Level 1 avoids multi-scale synchronization overhead.
   - **Conclusion:** 1-level 2D Haar DWT remains the optimal, mathematically justified foundation for PaladinMedia Phase 2 and early Phase 3.
