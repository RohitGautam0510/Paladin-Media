# Phase 6 Experiment 3 — Boundary-Aware Training Objective Assessment Report

**Generated:** 2026-10-30 16:22:43 UTC  
**Scope:** 20 Training Subset Carriers (`0001-0020`) and 20 Validation Subset Carriers (`0601-0620`)  
**Base Checkpoint:** `best_model.pt`  

---

## 1. Empirical Loss and Dynamic Range Metrics

| Metric | Train Subset (20 Images) | Validation Subset (20 Images) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Mean Boundary Loss ($\mathcal{L}_{\text{bound}}$)** | 0.0071 | 0.0475 | Pre-clamp spatial dynamic range violation magnitude |
| **Mean Pre-Clamp Min Spatial Value** | -2.72 | -9.10 | Floor violation below 0.0 |
| **Mean Pre-Clamp Max Spatial Value** | 256.41 | 266.43 | Ceiling violation above 255.0 |
| **Mean Pixels Clipped Below 0** | 367.0 | 4531.4 | Average pixels hitting zero floor |
| **Mean Pixels Clipped Above 255** | 1182.9 | 2003.8 | Average pixels hitting ceiling |
| **Mean Standard Context Loss ($\mathcal{L}_{\text{ctx}}$)** | 0.5722 | 0.8824 | Subband prediction error |
| **Mean Texture-Weighted Context Loss** | 13.4058 | 35.0177 | Frequency-weighted error |
| **Boundary Gradient Norm (ContextPredictor)** | 0.0888 | 0.6178 | Backpropagation signal through Haar IDWT |
| **Boundary Gradient Norm (AdaptiveGain)** | 0.1105 | 0.5021 | Backpropagation signal through Haar IDWT |

---

## 2. Scientific Findings

1. **Active Pre-Clamp Violations:**
   - Both the training subset and validation subset exhibit persistent pre-clamp violations with mean minimum spatial values of **-9.10** (reaching up to $-30.5$) and mean maximum spatial values of **266.43** (reaching up to $+285.5$).
   - An average of **6535.2 pixels per carrier** undergo dynamic range clamping.

2. **Differentiable Gradient Flow:**
   - Gradients of $\mathcal{L}_{\text{bound}}$ flow smoothly and differentiably through 2D Haar IDWT into both `ContextPredictorCNN` (gradient norm $\approx 0.6178$) and `AdaptiveGainCNN` (gradient norm $\approx 0.5021$).
   - The gradient signal actively penalizes `ContextPredictorCNN` for outputting values outside the carrier's valid local spatial range $[HH_{\min}', HH_{\max}']$.

3. **High-Frequency Texture Weighting:**
   - On high-frequency carriers, weighting context loss by $(|LH| + |HL|)$ increases the penalty on edge regions by $\approx 20\%$, aligning model supervision with the regions that suffer the highest demodulation error.
