# Experiment E05: HH Channel Allocation Evaluation

## Document Control
- **Experiment ID:** E05
- **Phase:** Phase 2 — Signal Processing + Carrier Infrastructure
- **Date:** 2026-10-09
- **Status:** COMPLETED & VERIFIED
- **Related Specifications:** PaladinMedia Architecture Specification §10/§12, IRS §6/§9
- **Evidence Path:** `docs/experiments/e05_results.json`

---

## 1. Objective & Hypothesis
* **Objective:** Evaluate transform-domain capacity, per-channel distortion, and recovery accuracy when expanding embedding from a single HH channel (Red) to multiple HH color channels (Red + Green, and Red + Green + Blue).
* **Hypothesis:** Allocating additional color channels scales raw and usable capacity linearly ($1\times, 2\times, 3\times$) without degrading perceptual transparency (PSNR remains $> 70\text{ dB}$, $\text{SSIM} = 1.0$) and maintains zero bit errors across the 8-bit RGB PNG round trip.

---

## 2. Channel Allocation & Deterministic Ordering
To avoid circular bootstrapping, sender and receiver adhere to the strict canonical channel ordering:
1. **Ordering Rule:** Red (Channel 0), followed by Green (Channel 1), followed by Blue (Channel 2).
2. **Coordinate Linearization:**
   - Indices $[0 \dots 65,535]$: Channel 0 (Red)
   - Indices $[65,536 \dots 131,071]$: Channel 1 (Green)
   - Indices $[131,072 \dots 196,607]$: Channel 2 (Blue)
3. **Anchor Positions:** Carrier header (128 bits) is anchored in Channel 0 positions $[0 \dots 127]$.
4. **Payload Placement:** Payload bits are sampled uniformly without replacement from remaining positions across the active channels via PRNG seeded by $\text{SHA-256}(\text{domain} \parallel \text{session\_counter} \parallel \text{payload\_bits})$.

---

## 3. Empirical Capacity & Fidelity Measurements

Tested with cover `MVP 0 Dataset/0001x2.png` ($512 \times 512 \times 3$) and secret `tests/fixtures/secret_small.png`:

| Allocation | Active Channels | Raw Capacity | Max Protected | Usable Plaintext | PSNR (R) | PSNR (G) | PSNR (B) | Overall PSNR | SSIM | BER | Exact Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Channel** | Red (0) | 8,192 B | 8,176 B | 7,090 B | 71.29 dB | $\infty$ | $\infty$ | **76.06 dB** | 1.000000 | 0.0 | **TRUE (100%)** |
| **2 Channels** | Red (0) + Green (1) | 16,384 B | 16,368 B | 14,226 B | 74.06 dB | 74.15 dB | $\infty$ | **75.87 dB** | 1.000000 | 0.0 | **TRUE (100%)** |
| **3 Channels** | Red (0) + Green (1) + Blue (2) | 24,576 B | 24,560 B | 21,362 B | 75.45 dB | 76.19 dB | 75.76 dB | **75.79 dB** | 1.000000 | 0.0 | **TRUE (100%)** |

---

## 4. Performance & Resource Comparison

| Configuration | Embedding Time | Extraction Time | End-to-End Time | Memory Delta |
| :--- | :--- | :--- | :--- | :--- |
| **1 Channel (R)** | 18.2 ms | 7.8 ms | 134.8 ms | ~ 2.4 MB |
| **2 Channels (R+G)** | 19.5 ms | 8.6 ms | 138.2 ms | ~ 3.1 MB |
| **3 Channels (RGB)** | 21.0 ms | 9.4 ms | 142.5 ms | ~ 3.8 MB |

---

## 5. Architectural Conclusions & Phase Boundary
1. **Defensible Linear Scaling:** Multi-channel allocation provides a verified $3\times$ capacity gain (up to $21.36\text{ KB}$ usable secret plaintext) while maintaining extreme transparency ($> 75.7\text{ dB}$ overall PSNR) and zero bit errors.
2. **Perturbation Distribution:** Spreading bits across 3 channels distributes the small spatial adjustments across all color planes, avoiding concentrated single-plane artifacts.
3. **Phase 2 Baseline vs Phase 3:** In Phase 2, this provides the highest single-image capacity without modifying subband geometry or introducing lossy neural models. It is recommended as a standard carrier configuration candidate for Phase 3 INN integration.
