# PaladinMedia — ACS-Sub-INN Integration & Qualification Report

**Date:** 2026-11-02  
**Status:** QUALIFIED & INTEGRATED  
**Artifacts Evaluated:**  
- `checkpoints/phase5a_onnx/acs_sub_inn_embedding_opset14.onnx` (SHA-256: `3cd22ba35e...`)  
- `checkpoints/phase5a_onnx/acs_sub_inn_extraction_opset14.onnx` (SHA-256: `ab69681463...`)  
- `checkpoints/phase5a_onnx/neural_spatial_embedding_opset14.onnx` (SHA-256: `7cda5a3bec...`)  
- `checkpoints/phase5a_onnx/neural_spatial_extraction_opset14.onnx` (SHA-256: `bf85d84044...`)  
**Results Source:** `docs/experiments/ACS_SUB_INN_QUALIFICATION_RESULTS.json`

---

## 1. Executive Summary

The transform-domain subband ONNX models (`acs_sub_inn_embedding_opset14.onnx` and `acs_sub_inn_extraction_opset14.onnx`), previously orphaned in diagnostic scripts, have been integrated into the PaladinMedia production hierarchy through `ACSSubINNModulationEngine`.

A qualification benchmark was conducted across held-out test carriers (`0701`–`0725`) to evaluate end-to-end parity between:
1. `neural_spatial_onnx`: The end-to-end spatial model with internal Haar DWT/IDWT.
2. `acs_sub_inn_onnx`: The explicit subband model operating with external Python 2D Haar DWT/IDWT.

### Key Results:
- **Exact Parity:** 100.0% outcome consistency between both backends across all 25 test carriers.
- **Recovery Rate:** 24/25 (96.0%) exact plaintext recovery under Wire Protocol v2 for both backends (with `0715.png` experiencing the previously documented RS(24,16) header clipping failure on both backends).
- **Stego Image Fidelity:**
  - `neural_spatial_onnx`: Average PSNR = **54.36 dB**, Average SSIM = **0.99997**.
  - `acs_sub_inn_onnx`: Average PSNR = **54.36 dB**, Average SSIM = **0.99997**.
- **Execution Performance (CPU):**
  - `neural_spatial_onnx`: Average Embedding = 178.4 ms, Average Extraction = 48.2 ms.
  - `acs_sub_inn_onnx`: Average Embedding = 181.6 ms, Average Extraction = 62.1 ms.

---

## 2. Architectural Integration

### 2.1 Engine Implementation
The `ACSSubINNModulationEngine` was added to `src/carrier/modulation.py`, extending `CarrierModulationEngine`:
- **Embedding Contract:**
  1. Computes 1-level 2D Haar DWT on cover RGB using `HaarDWT2D.forward(cover_array)`.
  2. Slices Red-HH (`hh_drop`, shape $1 \times 1 \times 256 \times 256$) and Green/Blue-HH context (`hh_keep`, shape $1 \times 2 \times 256 \times 256$).
  3. Modulates coefficients via `acs_sub_inn_embedding_opset14.onnx` using deterministic mask and bipolar secret bits.
  4. Assembles stego subbands and reconstructs spatial RGB via `HaarDWT2D.inverse(stego_subbands)`.
  5. Enforces uint8 pixel clamping in $[0.0, 255.0]$.
- **Extraction Contract:**
  1. Computes 1-level 2D Haar DWT on stego container.
  2. Passes received Red-HH and Green/Blue-HH subbands to `acs_sub_inn_extraction_opset14.onnx`.
  3. Returns `NeuralBitExtractor` wrapping the 2D binary decision grid.

### 2.2 Factory & Selection Integration
The unified factory function `create_modulation_engine(name)` supports:
- `"neural_spatial_onnx"` (Default production qualified spatial model)
- `"acs_sub_inn_onnx"` (Transform subband ONNX model with explicit DWT/IDWT)
- `"acs_sub_inn_pytorch"` (Reference PyTorch implementation)
- `"parity_classical"` (Classical Phase 2 parity modulation fallback)

`MediaStegoPipeline` and `NeuralModulationEngine` dynamically accept `backend="acs_sub_inn_onnx"` without breaking any existing test cases or callers.

---

## 3. Comparison Metrics

| Metric | `neural_spatial_onnx` | `acs_sub_inn_onnx` | Delta / Status |
| :--- | :--- | :--- | :--- |
| **Exact Recovery Rate** | 24 / 25 (96.0%) | 24 / 25 (96.0%) | **0.0% (Exact Match)** |
| **Mean PSNR (dB)** | 54.36 dB | 54.36 dB | $\pm 0.00 \text{ dB}$ |
| **Mean SSIM** | 0.99997 | 0.99997 | $\pm 0.00000$ |
| **Mean Embedding Latency** | 178.4 ms | 181.6 ms | +3.2 ms (negligible) |
| **Mean Extraction Latency** | 48.2 ms | 62.1 ms | +13.9 ms |
| **DWT/IDWT Execution Domain** | Inside ONNX graph | Python NumPy C-extension | Transparent subbands |
| **Memory Footprint** | ~85 MB RAM | ~90 MB RAM | Within 512 MB Free Tier |

---

## 4. Deployment Recommendation

1. **Production Default:** Retain `neural_spatial_onnx` as the default deployment engine in `MediaStegoPipeline.create_qualified_deployment()`. It has minimal Python overhead and is fully qualified.
2. **Subband Inspection Mode:** Expose `acs_sub_inn_onnx` in the frontend Model Selector. This enables laboratory inspection of explicit subband coefficients before and after IDWT reconstruction.
3. **Parity Fallback:** Maintain `parity_classical` for ultra-low resource devices or zero-ML environments.

---

**Report Approved By:** Antigravity Engineering Runtime  
**Status:** `ACS-SUB-INN INTEGRATION: PRODUCTION-QUALIFIED`
