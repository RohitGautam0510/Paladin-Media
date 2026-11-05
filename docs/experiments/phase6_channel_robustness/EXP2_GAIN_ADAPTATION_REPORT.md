# Phase 6 Experiment 2 — Inference-Time Gain Adaptation Report

**Generated:** 2026-10-30 16:19:13 UTC  
**Evaluation Set:** 20 Development Carriers from DIV2K Validation Split (`0601.png` - `0620.png`)  
**Checkpoint:** `best_model.pt`  

---

## 1. Candidate Gain Sweep Summary

| Scale Factor ($lpha$) | Exact Recovery | Header Success | RS Decode Success | Mean Payload BER | Mean Clipped Pixels | Pre-Clamp [Min, Max] | Mean PSNR | Mean SSIM | Mean Latency |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 18/20 (90.0%) | 20/20 (100.0%) | 17/20 (85.0%) | `0.003009` | 782.8 (516 low, 267 high) | [-5.4, 262.4] | 57.63 dB | 0.999983 | 400.2 ms |
| **0.70** | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | `0.002041` | 960.5 (650 low, 310 high) | [-5.6, 262.6] | 56.19 dB | 0.999978 | 366.4 ms |
| **0.85** | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | `0.001707` | 1037.8 (715 low, 323 high) | [-6.1, 263.3] | 54.68 dB | 0.999971 | 393.8 ms |
| **1.00** | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | `0.001292` | 1112.8 (770 low, 342 high) | [-6.5, 263.3] | 53.65 dB | 0.999964 | 376.6 ms |
| **1.15** | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | `0.001150` | 1188.2 (828 low, 360 high) | [-6.6, 263.8] | 52.92 dB | 0.999958 | 389.8 ms |
| **1.30** | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | `0.001014` | 1260.0 (882 low, 378 high) | [-6.6, 263.4] | 52.04 dB | 0.999949 | 377.7 ms |

---

## 2. Analysis & Scientific Conclusions

1. **Trade-Off Analysis:**
   - Reducing $\alpha < 1.0$ (e.g. 0.50, 0.70) significantly reduces pre-clamp pixel clipping and boosts PSNR.
   - However, reducing $\alpha$ decreases the signal-to-noise margin against uint8 integer rounding and context prediction error, resulting in higher raw payload BER.
   - Increasing $\alpha > 1.0$ (e.g. 1.15, 1.30) increases clipping and degrades PSNR without reducing the error floor on clipped boundaries.

2. **Decision on Inference-Time Gain Adaptation:**
   - Inference-time gain adjustment does not resolve zero-floor clipping because `context_predictor` predictions themselves violate dynamic range boundaries when cover pixel values approach 0.
   - Therefore, inference-time gain adaptation cannot cleanly eliminate clipping-induced errors without compromising the channel noise margin.
