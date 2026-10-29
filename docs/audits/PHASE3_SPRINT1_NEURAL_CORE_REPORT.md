# PaladinMedia Phase 3 — Sprint 1: Neural Core Scaffolding Report

**Document ID:** `PDM-AUDIT-P3-SPRINT1-NEURAL-CORE-20261018`
**Date:** 2026-10-18
**Repository:** `E:\Paladin_media`
**Status:** COMPLETED & AUDITED
**Governing Documents:** PaladinMedia Architecture Specification §10–§15, Phase 3 Acceptance Criteria (`PDM-P3-GATE-001`), Phase 3 Final Contract Verification (`PDM-AUDIT-P3-FINAL-VERIFICATION-20261018`), Phase 3 Environment Setup Report (`PDM-AUDIT-P3-ENV-SETUP-20261018`)
**Target Hardware:** NVIDIA GeForce GTX 1650 (TU117, Compute Capability 7.5, 4 GB VRAM), Driver 610.47, Python 3.10.11, PyTorch 2.5.1+cu124

---

## 1. Executive Summary

Phase 3 Sprint 1 has successfully implemented and validated the isolated neural modules for the **ACS-Sub-INN v3** architecture in `src/neural/`.

### Sprint Accomplishments
1. **Isolated Neural Scaffolding:** Implemented `ContextPredictorCNN`, `AdaptiveGainCNN`, `RoundSTE`, `SpatialQuantizeSTE`, and the orchestrating `ACSSubINN` module in [`src/neural/`](file:///E:/Paladin_media/src/neural/).
2. **Strict Invariant Adherence:**
   - **No Secret-Side DWT:** Payload enters strictly as pre-encrypted binary bits mapped to bipolar continuous symbols.
   - **No Latent-Z:** Full invertible flow without unrecoverable latent variable discarding.
   - **Preserved Phase 2 Codebase:** Zero modifications to existing Phase 2 carrier, crypto, serialization, ECC implementations, or `pyproject.toml`.
3. **Comprehensive Unit Testing:** Authored [`tests/unit/test_neural_components.py`](file:///E:/Paladin_media/tests/unit/test_neural_components.py) with 24 dedicated test cases covering tensor dimensions, strict gain positivity, STE gradient flow, exact coefficient-domain recovery, reproducibility, input rejection, and CUDA execution.
4. **Full Regression Integrity:** All 24 new unit tests and all 103 pre-existing tests pass (**127/127 tests pass in 4.84s**).
5. **Measured GPU Footprint:** Measured peak CUDA memory on the GTX 1650 was **1.67 MB** for synthetic tests and **137.68 MB** during full $512\times512$ carrier subband ($B=4, 256\times256$) evaluation, safely within the 4,096 MB VRAM ceiling.

---

## 2. Module Interfaces and Tensor Contracts

### 2.1 File Map and Responsibilities

| Module Path | Primary Class / Function | Responsibility |
|:---|:---|:---|
| [`src/neural/context_predictor.py`](file:///E:/Paladin_media/src/neural/context_predictor.py) | `ContextPredictorCNN` | Predicts unmodulated target subband context from preserved detail subbands. |
| [`src/neural/adaptive_gain.py`](file:///E:/Paladin_media/src/neural/adaptive_gain.py) | `AdaptiveGainCNN` | Predicts spatially varying strictly positive modulation gain $\gamma(x, y) \ge \gamma_{\text{min}}$. |
| [`src/neural/quantization.py`](file:///E:/Paladin_media/src/neural/quantization.py) | `RoundSTE`, `SpatialQuantizeSTE` | Differentiable Straight-Through Estimator rounding and spatial pixel quantization. |
| [`src/neural/acs_sub_inn.py`](file:///E:/Paladin_media/src/neural/acs_sub_inn.py) | `ACSSubINN` | Composes predictors, bipolar symbol mapping, forward embedding, and extraction. |
| [`src/neural/__init__.py`](file:///E:/Paladin_media/src/neural/__init__.py) | Package Exports | Exposes clean, typed API for Phase 3 neural components. |

---

### 2.2 Detailed Tensor Contracts

#### A. ContextPredictorCNN
- **Input Tensor:** $\text{HH}_{\text{keep}} \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{sub}} \times W_{\text{sub}}}$.
  - $C_{\text{in}} = 2$ (Channel 0: Green-HH, Channel 1: Blue-HH).
  - Arbitrary batch size $B \ge 1$ and even spatial subband dimensions ($H_{\text{sub}} = H/2, W_{\text{sub}} = W/2$).
- **Architecture:**
  - Input Conv: $\text{Conv2d}(2, 32, k=3, p=1) + \text{LeakyReLU}(0.2)$.
  - Residual Backbone: $3 \times \text{ResidualBlock}(32 \to 32, k=3, p=1)$ with identity skip connections.
  - Output Conv: $\text{Conv2d}(32, 1, k=3, p=1)$ with linear unbounded output.
- **Output Tensor:** $\text{context} \in \mathbb{R}^{B \times 1 \times H_{\text{sub}} \times W_{\text{sub}}}$.
  - Matches the spatial and batch dimensions of $\text{HH}_{\text{keep}}$.

#### B. AdaptiveGainCNN
- **Input Tensor:** $\text{HH}_{\text{keep}} \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{sub}} \times W_{\text{sub}}}$ ($C_{\text{in}} = 2$).
- **Architecture:**
  - Input Conv: $\text{Conv2d}(2, 32, k=3, p=1) + \text{LeakyReLU}(0.2)$.
  - Residual Backbone: $2 \times \text{ResidualBlock}(32 \to 32, k=3, p=1)$ with identity skip connections.
  - Output Conv: $\text{Conv2d}(32, 1, k=3, p=1)$.
  - Positivity Activation: $\gamma = \text{Softplus}(\text{raw\_gain}) + \gamma_{\text{min}}$.
- **Output Tensor:** $\gamma \in \mathbb{R}^{B \times 1 \times H_{\text{sub}} \times W_{\text{sub}}}$.
  - Enforces strictly positive gain $\gamma(x, y) \ge \gamma_{\text{min}} > 0.0$ everywhere (default $\gamma_{\text{min}} = 1.0$).

#### C. Straight-Through Estimator (STE) Quantization
- **`round_ste(x)`:**
  - Forward: $y = \text{round}(x)$.
  - Backward: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}$ (identity gradient pass-through).
- **`spatial_quantize_ste(x, min_val=0.0, max_val=255.0, clip_grad=False)`:**
  - Forward: $y = \text{clamp}(\text{round}(x), \text{min\_val}, \text{max\_val})$.
  - Backward: Identity pass-through (or optional gradient masking outside $[\text{min\_val}, \text{max\_val}]$ when `clip_grad=True`).

#### D. ACSSubINN Orchestration Module
- **Bipolar Symbol Mapping:**
  - Binary to Bipolar: $P = 2b - 1.0 \in \{-1.0, +1.0\}$ for $b \in \{0, 1\}$.
  - Bipolar to Binary: $\hat{b} = (\Delta_{\text{rec}} > 0.0) \in \{0, 1\}$ (dtype `torch.uint8`).
- **Forward Embedding:**
  $$\text{context} = \text{ContextPredictorCNN}(\text{HH}_{\text{keep}})$$
  $$\gamma = \text{AdaptiveGainCNN}(\text{HH}_{\text{keep}})$$
  $$\text{HH}_{\text{drop}}' = (1 - M) \odot \text{HH}_{\text{drop}} + M \odot (\text{context} + \gamma \odot P)$$
  - Where $M \in [0.0, 1.0]^{B \times 1 \times H_{\text{sub}} \times W_{\text{sub}}}$ is the binary embedding mask.
- **Inverse Extraction (Coefficient Domain):**
  $$\text{context}_{\text{rec}} = \text{ContextPredictorCNN}(\text{HH}_{\text{keep}})$$
  $$\Delta_{\text{rec}} = \text{HH}_{\text{drop}}' - \text{context}_{\text{rec}}$$
  $$\hat{b} = (\Delta_{\text{rec}} > 0.0)$$

---

## 3. Architecture Decisions & Assumptions

### 3.1 Decisions Frozen in Implementation
1. **Subband Partition Option A Enforced:**
   - $\text{HH}_{\text{drop}}$ is strictly Channel 0 (Red-HH, 1 channel).
   - $\text{HH}_{\text{keep}}$ is strictly Channels 1 and 2 (Green-HH and Blue-HH, 2 channels).
   - Guarantees 1:1 capacity equivalence with the classical Phase 2 baseline.
2. **Fully Convolutional Flexibility:**
   - Zero hardcoded spatial resolutions. The network dynamically processes any even spatial resolution ($512\times512 \implies 256\times256$ subbands; $1024\times1024 \implies 512\times512$ subbands; synthetic test tensors down to $8\times8$).
3. **Strict Gain Floor Setting ($\gamma_{\text{min}} = 1.0$):**
   - Configured as the default lower bound. Enforces non-zero perturbation so joint optimization cannot collapse to trivial zero-gain states.

---

### 3.2 Known Limitations & Domain Distinctions

> [!IMPORTANT]
> **Fundamental Distinction: Coefficient-Domain Reversibility vs. Image-Domain Recovery**

1. **Coefficient-Domain Exact Reversibility (Proven):**
   In pure floating-point coefficient arithmetic without quantization or noise:
   $$\Delta_{\text{rec}} = (\text{context} + \gamma \odot P) - \text{context} = \gamma \odot P$$
   Because $\gamma \ge \gamma_{\text{min}} > 0.0$, $\text{sign}(\Delta_{\text{rec}}) \equiv P$. Bit error rate is analytically and empirically **0.000000** (verified in `test_acs_sub_inn_coefficient_domain_exact_recovery`).
2. **Image-Domain Recovery (Unproven & Requires End-to-End Training):**
   Exact coefficient-domain recovery does **not** imply exact recovery after:
   - **Inverse DWT (IDWT):** Reconstructs floating-point spatial pixels.
   - **Integer Rounding:** $\text{round}(I_{\text{spatial}})$. Rounding errors propagate back through Haar DWT as perturbations up to $|\delta_{\text{HH}}| \le 1.0$.
   - **Dynamic Range Clipping:** $\text{clamp}(I_{\text{spatial}}, 0, 255)$. Boundary saturation introduces non-linear errors $|\delta| > 1.0$.
   - **Lossy JPEG Compression:** DCT quantization perturbs both $\text{HH}_{\text{drop}}$ and $\text{HH}_{\text{keep}}$, inducing context prediction errors $\delta_{\text{context}}$.
3. **STE Limitation:**
   Straight-Through Estimators enable gradient flow through step functions during training backpropagation. STE does **not** make rounding or clipping invertible during deployment.

---

## 4. Test Suite Execution & Verification Record

All tests were executed in the active Python 3.10.11 environment with PyTorch 2.5.1+cu124 on Windows 10.

### 4.1 New Neural Unit Tests ([`tests/unit/test_neural_components.py`](file:///E:/Paladin_media/tests/unit/test_neural_components.py))

```powershell
pytest tests/unit/test_neural_components.py -v
```

**Actual Execution Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-8.3.2, pluggy-1.6.0
rootdir: E:\Paladin_media
configfile: pyproject.toml
plugins: anyio-4.14.1
collected 24 items

tests/unit/test_neural_components.py::test_context_predictor_shapes_and_batch_sizes[spatial_dim0-1] PASSED [  4%]
tests/unit/test_neural_components.py::test_context_predictor_shapes_and_batch_sizes[spatial_dim0-2] PASSED [  8%]
tests/unit/test_neural_components.py::test_context_predictor_shapes_and_batch_sizes[spatial_dim0-4] PASSED [ 12%]
tests/unit/test_neural_components.py::test_context_predictor_shapes_and_batch_sizes[spatial_dim1-1] PASSED [ 16%]
tests/unit/test_neural_components.py::test_context_predictor_shapes_and_batch_sizes[spatial_dim1-2] PASSED [ 20%]
tests/unit/test_neural_components.py::test_context_predictor_shapes_and_batch_sizes[spatial_dim1-4] PASSED [ 25%]
tests/unit/test_neural_components.py::test_context_predictor_shapes_and_batch_sizes[spatial_dim2-1] PASSED [ 29%]
tests/unit/test_neural_components.py::test_context_predictor_shapes_and_batch_sizes[spatial_dim2-2] PASSED [ 33%]
tests/unit/test_neural_components.py::test_context_predictor_shapes_and_batch_sizes[spatial_dim2-4] PASSED [ 37%]
tests/unit/test_neural_components.py::test_context_predictor_cpu_forward_and_backward PASSED [ 41%]
tests/unit/test_adaptive_gain_strict_positivity[0.5] PASSED [ 45%]
tests/unit/test_adaptive_gain_strict_positivity[1.0] PASSED [ 50%]
tests/unit/test_adaptive_gain_strict_positivity[2.5] PASSED [ 54%]
tests/unit/test_adaptive_gain_cpu_backward_gradient_flow PASSED [ 58%]
tests/unit/test_round_ste_forward_and_backward PASSED [ 62%]
tests/unit/test_spatial_quantize_ste_forward_and_backward PASSED [ 66%]
tests/unit/test_ste_does_not_make_rounding_invertible PASSED [ 70%]
tests/unit/test_bipolar_symbol_encoding_and_decoding PASSED [ 75%]
tests/unit/test_bipolar_encode_rejects_non_binary_values PASSED [ 79%]
tests/unit/test_acs_sub_inn_coefficient_domain_exact_recovery PASSED [ 83%]
tests/unit/test_acs_sub_inn_gradient_flow_end_to_end PASSED [ 87%]
tests/unit/test_reproducibility_under_controlled_seed PASSED [ 91%]
tests/unit/test_input_validation_rejections PASSED [ 95%]
tests/unit/test_cuda_execution_smoke_test PASSED [100%]

============================= 24 passed in 2.91s ==============================
```

---

### 4.2 Complete Regression Suite Execution

```powershell
pytest tests/
```

**Actual Execution Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-8.3.2, pluggy-1.6.0
rootdir: E:\Paladin_media
configfile: pyproject.toml
plugins: anyio-4.14.1
collected 127 items

tests\integration\test_fail_closed.py ......                             [  4%]
tests\integration\test_media_pipeline.py ...                             [  7%]
tests\integration\test_pipeline.py .........                             [ 14%]
tests\unit\test_carrier_capacity.py ....                                 [ 17%]
tests\unit\test_carrier_dwt.py ...                                       [ 19%]
tests\unit\test_carrier_embedding.py ...                                 [ 22%]
tests\unit\test_carrier_header.py ..                                     [ 23%]
tests\unit\test_carrier_mask.py ....                                     [ 26%]
tests\unit\test_crypto.py .......................                        [ 44%]
tests\unit\test_ecc.py ........................                          [ 63%]
tests\unit\test_exceptions.py ..                                         [ 65%]
tests\unit\test_neural_components.py ........................            [ 84%]
tests\unit\test_serialization.py ....................                    [100%]

============================= 127 passed in 4.84s =============================
```

**Outcome:** 100% pass rate. 103 legacy Phase 2 tests and 24 new neural tests run together with zero regressions.

---

### 4.3 Measured GPU Memory Footprint (GTX 1650, 4 GB VRAM)

Measured using `torch.cuda.max_memory_allocated()` on `cuda:0`:

| Benchmark / Operation | Input Dimensions | Execution Mode | Measured Peak CUDA VRAM | Headroom (4,096 MB Limit) |
|:---|:---:|:---:|:---:|:---:|
| **Synthetic Unit Test Smoke** | $B=2$, $16 \times 16$ Subband | Forward + Extract | **$1.67\text{ MB}$** | $4,094.33\text{ MB}$ ($99.96\%$) |
| **$512 \times 512$ Carrier Evaluation** | $B=4$, $256 \times 256$ Subband | `torch.no_grad()` | **$137.68\text{ MB}$** | $3,958.32\text{ MB}$ ($96.64\%$) |

---

## 5. Scope Boundaries Maintained

In strict compliance with directive restrictions:
- Zero training runs on DIV2K images were launched.
- Zero datasets were downloaded or moved.
- No end-to-end media-pipeline integration was performed.
- No IDWT/PNG roundtrip robustness was asserted.
- No ONNX export or GUI integrations were attempted.
- No Phase 2 source files or `pyproject.toml` dependency declarations were altered.

---

## 6. Sprint 1 Sign-Off & Review Gate

Sprint 1 (Neural Core Scaffolding) is complete, tested, and verified.

**Next Sprint Readiness:** Awaiting review and approval before proceeding to Sprint 2 (DWT/IDWT Neural Bridge and Loss Function Formulation).
