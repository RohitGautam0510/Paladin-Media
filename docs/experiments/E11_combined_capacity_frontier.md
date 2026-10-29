# Experiment E11: Combined Capacity Frontier Synthesis

## Document Control
- **Experiment ID:** E11
- **Phase:** Phase 2 — Signal Processing + Carrier Infrastructure
- **Date:** 2026-10-09
- **Status:** COMPLETED & VERIFIED
- **Related Specifications:** PaladinMedia Architecture Specification §10/§11/§12, IRS §6/§9
- **Evidence Path:** `docs/experiments/e11_results.json`

---

## 1. Executive Overview
Experiment E11 synthesizes the empirical evidence accumulated across E03 through E10 into a multi-dimensional capacity–distortion–runtime operating frontier. The goal is not to maximize capacity indiscriminately at the expense of visual fidelity or cryptographic safety, but to identify rigorously defensible operating points for Phase 2 production and future Phase 3 ACS-Sub-INN neural integration.

---

## 2. Defensible Operational Frontier Matrix

| Point ID | Configuration Name | Carrier Resolution & Count | Allocation (Ch / Subband / Rate) | Plaintext Pre-processing | Usable Secret Plaintext | Measured PSNR | SSIM | Latency | BER | Status & Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP-01** | **Canonical Baseline (MVP 0)** | $1 \times 512 \times 512$ | 1 Ch (R) / $\text{HH}$ / $1\text{ bpc}$ | Raw | **7,090 B** | **75.89 dB** | 1.000000 | 134.8 ms | 0.0 | **IMPLEMENTED & GREEN**<br>Standard Phase 2 baseline. Minimal complexity. |
| **FP-02** | **RGB Balanced Detail** | $1 \times 512 \times 512$ | 3 Ch (RGB) / $\text{HH}$ / $1\text{ bpc}$ | Raw | **21,362 B** | **75.79 dB** | 1.000000 | 142.5 ms | 0.0 | **IMPLEMENTED & GREEN**<br>$3.0\times$ capacity gain with identical high fidelity. |
| **FP-03** | **Max Single-Carrier Detail** | $1 \times 512 \times 512$ | 3 Ch (RGB) / $\text{HH}+\text{HL}+\text{LH}$ / $1\text{ bpc}$ | Raw | **64,150 B** | **68.45 dB** | 0.999991 | 172.0 ms | 0.0 | **EXPERIMENTAL VERIFIED**<br>Max allocation without altering LL approximation. |
| **FP-04** | **Multi-Carrier Transport (MCPP 3x)** | $3 \times 512 \times 512$ | 1 Ch (R) / $\text{HH}$ / $1\text{ bpc}$ | Raw | **21,270 B** | **63.46 dB** | 0.999994 | 310.2 ms | 0.0 | **PROTOCOL VERIFIED**<br>Arbitrary payload distribution across carrier galleries. |
| **FP-05** | **Synergistic Compressed Pipeline** | $1 \times 512 \times 512$ | 3 Ch (RGB) / $\text{HH}$ / $1\text{ bpc}$ | zlib Level 9 | **21,362 B to 200,000+ B** | **75.79 dB** | 1.000000 | 145.0 ms | 0.0 | **RECOMMENDED PRODUCTION POINT**<br>Optimal for redundant secret media (logos, UI, docs). |

---

## 3. Rejected & Unsupported Configurations (Boundary Analysis)

To ensure scientific honesty and prevent overclaiming, the following configurations were rigorously audited and classified as non-viable or unsupported:

1. **Native Carrier Resolution $\ge 1024 \times 1024$ (UNSUPPORTED / SKIPPED):**
   - *Rationale:* The local `DIV2K_train_LR_bicubic_X2` dataset caps native dimensions at 1020 px. High-Resolution (HR) 2K sources are absent. Per non-negotiable Rule 2, synthetic upscaling is strictly prohibited.
2. **Low-Frequency $\text{LL}$ Subband Embedding (REJECTED):**
   - *Rationale:* $\text{LL}$ contains $>95\%$ of image spectral energy. Modulating $\text{LL}$ directly modifies block mean luminance, causing severe blocking and visible luminance banding in smooth gradients.
3. **High Octave Decompositions (Level 2 / Level 3 Alone) (REJECTED):**
   - *Rationale:* Depletes raw capacity by $75\%$ (Level 2) and $93.75\%$ (Level 3), and broadens spatial receptive field from $2 \times 2$ to $4 \times 4$ and $8 \times 8$ pixels, causing larger-scale ripple artifacts upon IDWT.
4. **High Embedding Density ($> 3\text{ bpc}$) in Spatial Parity (REJECTED FOR BASELINE):**
   - *Rationale:* Requires large spatial pixel perturbations ($\max |\Delta| \ge 4$), creating non-linear boundary saturation risks and making carriers vulnerable to statistical steganalyzers.

---

## 4. Phase 3 Architecture Recommendations
1. **Preserve Phase 1 & 2 Invariant Sequence:** The pipeline sequence `Serialize -> Compress -> Encrypt -> ECC -> Framing -> Carrier DWT` is rock-solid and must remain untouched.
2. **Phase 3 Integration Point:** In Phase 3, the ACS-Sub-INN neural engine should replace the discrete integer parity embedder within the high-frequency subbands ($\text{HH}$ or $\text{HH}+\text{HL}+\text{LH}$), operating on continuous wavelet representations while maintaining the upstream crypto and downstream PNG serialization layers.
