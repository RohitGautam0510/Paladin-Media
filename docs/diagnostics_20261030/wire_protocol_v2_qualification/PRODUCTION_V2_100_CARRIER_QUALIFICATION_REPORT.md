# PaladinMedia Wire Protocol v2 Qualification Report

**Generated:** 2026-10-30 16:00:41 UTC  
**Evaluation Scope:** Canonical DIV2K test split (100 images)  
**Output Directory:** `docs\diagnostics_20261030\wire_protocol_v2_qualification`  

---

## 1. Executive Summary

| Metric | Wire Protocol v1 Baseline | Wire Protocol v2 Production | Delta / Status |
| :--- | :--- | :--- | :--- |
| **Exact Recovery Rate** | 95 / 100 (95.0%) | 98 / 100 (98.0%) | +3 |
| **Header Unpack Success** | 95 / 100 (95.0%) | 99 / 100 (99.0%) | +4.0% |
| **Payload RS Success** | 95 / 100 (95.0%) | 98 / 100 | +3 |
| **Mean PSNR** | 53.94 dB | 54.04 dB | Neutral |
| **Mean SSIM** | 0.99997 | 0.999961 | Neutral |
| **Mean Payload BER** | N/A (unmeasured) | 0.000649 | Measured |
| **Total Duration** | - | 33.97 s | - |

---

## 2. Failure Triage

Total Unresolved Failures: **2**

| Image ID | Filename | Failing Stage | Header Byte Errors | Root Cause Summary |
| :--- | :--- | :--- | :--- | :--- |
| `715` | `0715.png` | `HEADER_UNPACK` | 10 | `Uncorrectable Wire Protocol v2 carrier header: errors exceed RS(24, 16) correction capability (t=4 bytes)` |
| `774` | `0774.png` | `PAYLOAD_RS_DECODE` | 4 | `Reed-Solomon decoding failed in block 2/6: uncorrectable symbol errors exceed capacity` |

---

## 3. Engineering Recommendations for Controlled Retraining (Phase 6)

1. **Spatial Boundary Loss / Clipping Penalty:** Add an explicit penalty in the training objective for spatial pixel values exiting $[0, 255]$ before clamping, eliminating zero-floor and ceiling clipping on high-contrast edges.
2. **Dynamic Range Margin:** Encourage gain parameter $\gamma$ and context predictions to keep IDWT perturbations safely inside dynamic range $[4, 251]$.
3. **High-Frequency Texture Regularization:** Incorporate frequency-weighted loss to penalize high residual errors on complex texture carriers like `0774.png`.
