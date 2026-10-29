# PaladinMedia Phase 3 — Sprint 3: Loss-Function and Training-Loop Scaffolding Report

**Document ID:** `PDM-AUDIT-P3-SPRINT3-SCAFFOLD-20261019`
**Date:** 2026-10-19
**Repository:** `E:\Paladin_media`
**Status:** **PASS** (Scaffolding and Verification Complete — ZERO Training Executed)
**Governing Documents:**
- PaladinMedia Architecture Specification §8, §10, §11, §14
- Phase 3 Acceptance Criteria ([`docs/specifications/Phase3_Acceptance_Criteria.md`](file:///e:/Paladin_media/docs/specifications/Phase3_Acceptance_Criteria.md), `PDM-P3-GATE-001` through `PDM-P3-GATE-005`)
- DIV2K Dataset Split Manifest ([`docs/project/div2k_hr_split.json`](file:///e:/Paladin_media/docs/project/div2k_hr_split.json))
- Synthetic Smoke Test Benchmark Results ([`docs/experiments/sprint3_training_scaffold_results.json`](file:///e:/Paladin_media/docs/experiments/sprint3_training_scaffold_results.json))
**Target Hardware:** NVIDIA GeForce GTX 1650 (4 GB VRAM, Driver 610.47, CUDA 12.4), Python 3.10.11, PyTorch 2.5.1+cu124

---

## 1. Executive Summary & Verification Mandate

This report documents the design, implementation, and verification of the training infrastructure for PaladinMedia's frozen ACS-Sub-INN v3 neural steganography engine.

### Strict Governance Constraints Met:
1. **NO MODEL TRAINING EXECUTED:** No optimizer parameter updates were applied (`optimizer.step()` was not invoked). No epochs were run, no DIV2K training loops were initiated, and zero learned improvements are claimed. Model weights before and after the synthetic forward/backward pass were verified to be **100% bit-for-bit identical**.
2. **Cryptographic Invariants Preserved:** The secret payload never enters the neural model as plaintext. All inputs to the neural embedding path remain opaque, authenticated, Reed–Solomon protected bitstreams.
3. **Frozen Architecture Protected:** The ACS-Sub-INN v3 design was strictly preserved:
   - Target subband: Red-HH detail subband (channel 0).
   - Context subbands: Green-HH and Blue-HH detail subbands (channels 1–2).
   - Modulation gain floor: `gamma_min = 1.0` remains the default baseline.
   - No secret-side DWT, no latent-Z discarding, no differentiable JPEG layers, and no extra subbands.
   - Phase 2 production code (`src/carrier/`, `src/security/`, `src/core/`) was completely untouched.
4. **Dataset Partition Boundaries Enforced:** The canonical split (600 train / 100 val / 100 test) was audited. The loader raises a hard terminal `ValueError` on any attempt to select the held-out test partition for training or tuning.
5. **Synthetic Smoke Test & Regression Passed:** A single-step synthetic smoke test executed forward and backward passes with finite gradients across all trainable parameters. The full regression suite confirms **175/175 tests passing in 7.46s**.

---

## 2. Pre-Edit Architectural Audit & Interface Findings

Before constructing any training scaffold code, all relevant components were audited directly from source files:

| Module / Component | File Path | Actual Verified Interface / Lines | Architectural Role & Observed Behavior |
|:---|:---|:---|:---|
| **Context Predictor** | [`src/neural/context_predictor.py`](file:///e:/Paladin_media/src/neural/context_predictor.py) | `ContextPredictorCNN` (Lines 52–150) | Accepts `(B, 2, H, W)` (`hh_keep`), outputs `(B, 1, H, W)` unconstrained linear prediction of Red-HH. 3 residual blocks with LeakyReLU ($\alpha = 0.2$). |
| **Adaptive Gain** | [`src/neural/adaptive_gain.py`](file:///e:/Paladin_media/src/neural/adaptive_gain.py) | `AdaptiveGainCNN` (Lines 27–135) | Accepts `(B, 2, H, W)` (`hh_keep`), outputs `(B, 1, H, W)` gain $\gamma \ge \gamma_{\text{min}} = 1.0$ via $\text{Softplus}(\text{raw}) + \gamma_{\text{min}}$. |
| **Quantization (STE)** | [`src/neural/quantization.py`](file:///e:/Paladin_media/src/neural/quantization.py) | `SpatialQuantizeSTE` (Lines 138–159) | Applies `clamp(round(x), 0, 255)` forward; passes identity gradient $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}$ backward. |
| **ACS-Sub-INN** | [`src/neural/acs_sub_inn.py`](file:///e:/Paladin_media/src/neural/acs_sub_inn.py) | `ACSSubINN` (Lines 60–250) | Embeds bipolar payload: $hh_{\text{drop}}' = (1-M)hh_{\text{drop}} + M(\text{context} + \gamma P)$. Extracts: $\delta_{\text{rec}} = hh_{\text{drop}}' - \text{context}_{\text{rec}}$. |
| **Spatial Bridge** | [`src/neural/spatial_bridge.py`](file:///e:/Paladin_media/src/neural/spatial_bridge.py) | `NeuralSpatialBridge` (Lines 271–438) | Differentiable forward Haar DWT / IDWT. When `differentiable=True`, stego image is processed by `SpatialQuantizeSTE` without detaching the autograd graph. |
| **Haar Wavelet Ops** | [`src/neural/spatial_bridge.py`](file:///e:/Paladin_media/src/neural/spatial_bridge.py) | `haar_dwt2d`, `haar_idwt2d` (Lines 81–194) | Tensor slicing arithmetic; mathematically invertible; 100% differentiable in PyTorch. |
| **Split Manifest** | [`docs/project/div2k_hr_split.json`](file:///e:/Paladin_media/docs/project/div2k_hr_split.json) | Entire JSON file (800 records) | Composite SHA-256: `d0f5f4ee...1ed`. Exact partitions: 600 train (`0001..0600`), 100 val (`0601..0700`), 100 test (`0701..0800`). |

---

## 3. Mathematical Loss Formulation and Gradient Path Routing

All candidate loss functions have been implemented in [`src/neural/losses.py`](file:///e:/Paladin_media/src/neural/losses.py).

### 3.1 Loss A: Bit-Recovery Surrogate Loss ($L_{\text{bit}}$)
- **Mathematical Definition:**
  Demodulation uses the sign of the received residual: $\hat{b} = 1 \text{ if } \delta_{\text{rec}} > 0 \text{ else } 0$, corresponding to bipolar $\hat{P} = \text{sign}(\delta_{\text{rec}})$. Because $\text{sign}(\cdot)$ has a zero derivative almost everywhere, a continuous surrogate is required.
  We implement the stable **Logistic / Softplus** margin loss:
  $$\ell_{\text{bit}}(P, \delta_{\text{rec}}) = \text{Softplus}\left( -P \cdot \frac{\delta_{\text{rec}}}{\tau} \right) = \ln\left(1 + \exp\left(-P \cdot \frac{\delta_{\text{rec}}}{\tau}\right)\right)$$
  where $\tau > 0$ is a temperature scaling hyperparameter (default $\tau = 1.0$).
  Over the active spatial mask $M \in \{0, 1\}^{B \times 1 \times H \times W}$:
  $$L_{\text{bit}} = \frac{1}{\sum M + \epsilon} \sum_{i,j} M_{i,j} \cdot \text{Softplus}\left( -P_{i,j} \cdot \frac{\delta_{\text{rec}, i, j}}{\tau} \right)$$
- **Input Tensors & Shapes:** $\delta_{\text{rec}} \in \mathbb{R}^{B \times 1 \times H \times W}$, $P \in \{-1.0, +1.0\}^{B \times 1 \times H \times W}$, $M \in [0, 1]^{B \times 1 \times H \times W}$.
- **Reduction:** Mean over active mask positions ($\sum M$).
- **Units & Range:** Dimensionless scalar in $(0, \infty)$; zero at infinite positive margin, $\ln(2) \approx 0.693$ at the decision boundary ($\delta_{\text{rec}} = 0$).
- **Gradient Path:** Flows back through $\delta_{\text{rec}} \to hh_{\text{drop\_rec}} \to \text{DWT} \to \text{STE} \to \text{IDWT} \to hh_{\text{drop\_stego}} \to \text{AdaptiveGainCNN}$ (via $\gamma \cdot P$).
- **Critical Distinction from Exact Plaintext Recovery:**
  Optimizing $L_{\text{bit}}$ continuous margins in the coefficient domain does **not** ensure verbatim byte recovery after:
  1. Integer uint8 spatial rounding and dynamic range clipping ($[0, 255]$),
  2. Lossless PNG serialization,
  3. Reed–Solomon block-level decoding ($t=16$ symbol error correction threshold per 255-byte block),
  4. AES-256-GCM GHASH authentication (where 1 uncorrected bit causes terminal rejection and 0 plaintext recovery).

### 3.2 Loss B: Context-Prediction Loss ($L_{\text{ctx}}$)
- **Mathematical Definition:**
  Supervises `ContextPredictorCNN` to predict the unmodulated Red-HH coefficients from preserved Green/Blue-HH subbands:
  $$L_{\text{ctx}} = \frac{1}{B \cdot H_{\text{sub}} \cdot W_{\text{sub}}} \sum_{b,y,x} \text{SmoothL1}_\beta\left( \text{context}_{b,y,x} - hh_{\text{drop}, b, y, x} \right)$$
  where $\beta = 1.0$:
  $$\text{SmoothL1}_\beta(z) = \begin{cases} 0.5 \frac{z^2}{\beta}, & \text{if } |z| < \beta \\ |z| - 0.5 \beta, & \text{otherwise} \end{cases}$$
- **Input Tensors & Shapes:** $\text{context} \in \mathbb{R}^{B \times 1 \times H \times W}$ (predicted), $hh_{\text{drop}} \in \mathbb{R}^{B \times 1 \times H \times W}$ (target).
- **Target Tensor Identity:** Ground-truth unmodulated Red-HH carrier coefficients. **Secret payload content is NEVER a prediction target.**
- **The Context Cancellation Mathematical Invariant:**
  In the lossless coefficient domain:
  $$hh_{\text{drop\_stego}} = \text{context} + \gamma \cdot P$$
  $$\delta_{\text{rec}} = hh_{\text{drop\_stego}} - \text{context}_{\text{rec}} = (\text{context} + \gamma \cdot P) - \text{context} = \gamma \cdot P$$
  Because the context added during embedding is subtracted during extraction, $\frac{\partial \delta_{\text{rec}}}{\partial \text{context}} \equiv 0$.
  **Therefore, $L_{\text{bit}}$ cannot train the ContextPredictorCNN.** The context predictor requires explicit direct supervision via $L_{\text{ctx}}$.

### 3.3 Loss C: Adaptive-Gain Regularization ($L_{\text{gain}}$)
- **Mathematical Definition:**
  Penalizes excess gain above the architectural minimum $\gamma_{\text{min}} = 1.0$:
  $$L_{\text{gain}} = \frac{1}{\sum M + \epsilon} \sum_{i,j} M_{i,j} \cdot \text{ReLU}(\gamma_{i,j} - \gamma_{\text{min}})$$
- **Input Tensors & Shapes:** $\gamma \in \mathbb{R}^{B \times 1 \times H \times W}$ (where $\gamma \ge \gamma_{\text{min}}$ by softplus construction).
- **Trade-off:**
  - High gain increases the demodulation margin against noise, but drives spatial pixels outside $[0, 255]$ (causing severe clipping errors) and degrades PSNR.
  - Low gain ($\gamma \to 1.0$) minimizes image distortion, but leaves narrow margin against spatial rounding noise ($|\epsilon| \le 1.0$).
  - $L_{\text{gain}}$ prevents the optimizer from trivially inflating $\gamma \to \infty$.

### 3.4 Loss D: Spatial Image-Fidelity Loss ($L_{\text{img}}$)
- **Mathematical Definition:**
  Mean Absolute Error between reconstructed spatial stego image and cover image:
  $$L_{\text{img}} = \frac{1}{3 \cdot H \cdot W} \sum_{c,y,x} |x_{\text{stego}, c, y, x} - x_{\text{cover}, c, y, x}|$$
- **Input Tensors & Shapes:** $x_{\text{stego}} \in \mathbb{R}^{B \times 3 \times H \times W}$ (post-STE float tensor in $[0, 255]$), $x_{\text{cover}} \in \mathbb{R}^{B \times 3 \times H \times W}$.
- **Surrogate Limitation:** STE passes an identity gradient ($\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}$). It does not capture non-linear pixel clamping at boundary saturation.

### 3.5 Diagnostic Metric E: Saturation/Clipping Metric (Non-Optimized)
- **Metric Definition (DIAGNOSTIC ONLY):**
  Monitors pre-clamp pixel values outside $[0, 255]$ and post-clamp pinned boundary pixels:
  $$\text{clip\_fraction} = \frac{\sum (x_{\text{raw}} < 0.0) + \sum (x_{\text{raw}} > 255.0)}{3 \cdot H \cdot W}$$
  $$\text{boundary\_fraction} = \frac{\sum (x_{\text{stego}} == 0) + \sum (x_{\text{stego}} == 255)}{3 \cdot H \cdot W}$$
  $$\text{saturation\_penalty} = \frac{1}{3HW} \left( \|\text{ReLU}(-x_{\text{raw}})\|_1 + \|\text{ReLU}(x_{\text{raw}} - 255.0)\|_1 \right)$$
- **Governance:** Evaluated by [`evaluate_clipping_diagnostic()`](file:///e:/Paladin_media/src/neural/losses.py#L296-L349). Strictly non-trainable; not included in the backpropagated loss in this sprint.

### 3.6 Combined Candidate Objective ($L_{\text{total}}$)
- **Equation:**
  $$L_{\text{total}} = \lambda_{\text{bit}} L_{\text{bit}} + \lambda_{\text{ctx}} L_{\text{ctx}} + \lambda_{\text{gain}} L_{\text{gain}} + \lambda_{\text{img}} L_{\text{img}}$$
- **Candidate Uncalibrated Weights:**
  - $\lambda_{\text{bit}} = 1.0$ (Primary demodulation margin objective)
  - $\lambda_{\text{ctx}} = 0.5$ (High-frequency subband correlation learning)
  - $\lambda_{\text{gain}} = 0.01$ (Gain runaway prevention)
  - $\lambda_{\text{img}} = 0.1$ (Spatial pixel fidelity constraint)
- **Status:** **PROPOSED_UNVALIDATED (NOT TRAINING-APPROVED)**. These weights represent engineering hypotheses requiring empirical hyperparameter sweeps in future sprints.

---

## 4. Graph Breaks and Differentiability Verification

During the audit, the autograd graph was traced through the entire spatial roundtrip:
1. `HaarDWT2DTorch`: Uses standard PyTorch strided slicing (`0::2`, `1::2`) and addition/subtraction. **100% differentiable; zero graph breaks.**
2. `ACSSubINN.embed_coefficients`: Uses standard PyTorch tensor addition and multiplication. **100% differentiable.**
3. `HaarDWT2DTorch.inverse`: Contiguous tensor assembly and slice assignments. **100% differentiable.**
4. `SpatialQuantizeSTE`: Forward pass clamps and rounds; backward pass substitutes identity gradient ($\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}$). **Differentiable surrogate; no graph breaks.**
5. `ACSSubINN.extract_coefficients`: Linear subtraction $\delta_{\text{rec}} = hh_{\text{drop\_stego}} - \text{context}$. **100% differentiable.**
6. **Intentional Non-Differentiable Boundaries:**
   - `ACSSubINN.bipolar_decode`: Uses `(delta_rec > 0).to(uint8)`. Non-differentiable; used strictly for evaluation BER and diagnostic output.
   - `save_and_reload_png`: Uses PIL PNG codec on uint8 tensors. Non-differentiable; used strictly during inference and validation roundtrips, bypassed by `differentiable=True` during training steps.

---

## 5. Dataset Split Audit and Test-Set Protection

The dataset split manifest was audited directly from [`docs/project/div2k_hr_split.json`](file:///e:/Paladin_media/docs/project/div2k_hr_split.json):
- **Source Directory:** `dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR`
- **Total Images:** 800 images (`0001.png` through `0800.png`)
- **Composite SHA-256:** `d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed` (Verified intact)
- **Partitions:**
  - `train`: 600 images (`0001.png` – `0600.png`, 75.0%)
  - `val`: 100 images (`0601.png` – `0700.png`, 12.5%)
  - `test`: 100 images (`0701.png` – `0800.png`, 12.5%) — **HELD-OUT EVALUATION ONLY**

### Programmatic Test-Set Protection:
[`DatasetSplitManager`](file:///e:/Paladin_media/src/neural/training_scaffold.py#L38-L109) enforces split security:
```python
def get_records_for_training(self, split: str) -> List[Dict[str, Any]]:
    if split == "test":
        raise ValueError(
            "DATA INTEGRITY VIOLATION: Accessing the held-out 'test' split for training, "
            "hyperparameter tuning, or validation is strictly forbidden by Acceptance Criteria §5."
        )
```
This invariant was tested in `test_test_set_protection_enforcement()` and confirmed to terminally reject test partition access.

---

## 6. Training Scaffold Design and Optimizer-Step Safety Invariant

The training scaffold was implemented in [`src/neural/training_scaffold.py`](file:///e:/Paladin_media/src/neural/training_scaffold.py).

### The Invariant: `optimizer_step_executed == False`
The training step function [`execute_training_step()`](file:///e:/Paladin_media/src/neural/training_scaffold.py#L206-L309):
1. Takes the model, optimizer, loss, carrier, payload bits, and mask.
2. Runs differentiable embedding and extraction.
3. Computes the composite loss.
4. Clears previous gradients with `optimizer.zero_grad()`.
5. Executes `loss.backward()` to compute parameter gradients.
6. Verifies that all gradients are finite.
7. Optionally clips gradients via `torch.nn.utils.clip_grad_norm_`.
8. **EXPLICITLY OMITS `optimizer.step()`**.
9. Returns `TrainingStepResult(..., optimizer_step_executed=False)`.

In `test_training_step_does_not_modify_weights()`, all parameter tensors in `bridge.acs_model` were fingerprinted before and after step execution. Every parameter tensor was **bit-for-bit identical**, proving that no learning occurred.

---

## 7. Synthetic Smoke-Test Evidence & Checkpoint Verification

A synthetic smoke test was executed via [`scripts/experiments/run_sprint3_training_smoke_test.py`](file:///e:/Paladin_media/scripts/experiments/run_sprint3_training_smoke_test.py).

### Smoke Test Execution Log:
```
================================================================================
PHASE 3 SPRINT 3: TRAINING SCAFFOLD SYNTHETIC SMOKE TEST (NO TRAINING)
================================================================================
[1] Reproducibility seeds configured: {'seed': 42, 'deterministic': True, 'cuda_configured': True, 'python_seeded': True, 'numpy_seeded': True, 'torch_seeded': True}
[2] Dataset split verified: Train=600, Val=100, Held-out Test=100
    Held-out test protection verified: successfully caught unauthorized access (DATA INTEGRITY VIOLATION)
[3] Configuration constructed on device: CUDA
    NeuralSpatialBridge initialized (gamma_min=1.0)
    CompositeLoss initialized: PROPOSED_UNVALIDATED (NOT TRAINING-APPROVED)
[4] Synthetic carrier generated: shape=(1, 3, 64, 64), range=[20.0, 234.0]
    Synthetic payload bits generated: shape=(1, 1, 32, 32), active_mask=500 positions

[5] Executing forward and backward pass...
[6] Parameter Invariant Verified:
    optimizer_step_executed: False (Expected: False)
    All parameters bit-for-bit identical before and after: True

[7] Loss Component Breakdown:
    Total Composite Loss:    27.352457
    Bit Recovery Loss:       0.202587  (weight=1.0)
    Context Prediction Loss: 53.388039 (weight=0.5)
    Adaptive Gain Reg:       1.269608  (weight=0.01)
    Image Fidelity Loss:     4.431560  (weight=0.1)

    Gradient Distribution Analysis:
    Total Clipped Grad Norm:     129.422577
    Context Predictor Mean Norm: 0.077005 (across 16 layers)
    Adaptive Gain Mean Norm:     0.045668 (across 12 layers)
    Max Param Gradient:          23.823399
    All Gradients Finite:        True

[8] Clipping Diagnostic Metrics (Diagnostic Only):
    Pre-clamp Raw Range:      [-23.99, 275.04]
    Total Clipped Pixels:     16 (0.130%)
    Boundary Pinned Pixels:   17 (0.138%)
    Saturation L1 Penalty:    0.011103

[9] Synthetic smoke-test checkpoint saved to: E:\Paladin_media\checkpoints\sprint3_smoke\synthetic_smoke_checkpoint.pt
    Metadata Type: SYNTHETIC_SMOKE_TEST_ARTIFACT
    Training Authorized: False
    Checkpoint reloaded and verified: True
```

### Checkpoint Governance:
The checkpoint saved at `checkpoints/sprint3_smoke/synthetic_smoke_checkpoint.pt` contains:
```json
{
  "checkpoint_type": "SYNTHETIC_SMOKE_TEST_ARTIFACT",
  "training_authorized": false,
  "is_synthetic_smoke_test": true,
  "optimizer_step_applied": false
}
```
It is explicitly isolated from any production model checkpoint.

---

## 8. Authoritative Acceptance Criteria and Governance Catalog

All Phase 3 metrics have been registered in [`GateRegistry`](file:///e:/Paladin_media/src/neural/training_scaffold.py#L463-L569) under strict governance classifications:

| Acceptance Criterion | Standard / Threshold | Governance Classification | Verification Status in Sprint 3 |
|:---|:---:|:---:|:---:|
| **Pre-ECC Bit Error Rate (PNG)** | $\text{BER} \le 1.0 \times 10^{-3}$ ($0.1\%$) | `EXISTING_PROJECT_GATE` | Monitored in diagnostic evaluation |
| **Post-ECC Plaintext Identity** | $\text{BER} = 0.000000$; `ExactMatch = True` | `EXISTING_PROJECT_GATE` | Verified in Sprint 2.1 / 2.1.2 |
| **AES-GCM Authentication** | 100% valid tags; 0 tag failures | `EXISTING_PROJECT_GATE` | Verified in Sprint 2.1 / 2.1.2 |
| **Reed–Solomon Headroom** | $\le 16$ symbol errors per block | `EXISTING_PROJECT_GATE` | Verified in Sprint 2.1 / 2.1.2 |
| **Stego PSNR (PNG)** | $\text{PSNR} \ge 60.0\text{ dB}$ | `EXISTING_PROJECT_GATE` | Monitored in evaluation step |
| **Stego SSIM (PNG)** | $\text{SSIM} \ge 0.9995$ | `EXISTING_PROJECT_GATE` | Monitored in evaluation step |
| **Capacity Hard Gate** | $\ge 6,000\text{ B}$ / fail-closed at $8,176\text{ B}$ | `EXISTING_PROJECT_GATE` | Verified in Sprint 2.1.2 |
| **Hardware Budget** | $\text{VRAM} \le 512\text{ MB}$, Latency $\le 250\text{ ms}$ | `EXISTING_PROJECT_GATE` | Peak 137.7 MB, ~25 ms verified |
| **Composite Loss Weighting** | $\lambda_{\text{bit}}=1.0, \lambda_{\text{ctx}}=0.5, \lambda_{\text{gain}}=0.01, \lambda_{\text{img}}=0.1$ | `PROPOSED_UNVALIDATED` | Implemented with uncalibrated defaults |
| **Pre-Clamp Out-of-Range Fraction** | Monitored diagnostic | `DIAGNOSTIC_ONLY` | Implemented in `evaluate_clipping_diagnostic` |
| **Post-Clamp Boundary Fraction** | Monitored diagnostic | `DIAGNOSTIC_ONLY` | Implemented in `evaluate_clipping_diagnostic` |
| **Lossy JPEG Channel ($Q=80$)** | Post-ECC $\text{BER} = 0.0$ at $Q=80$ | `PROPOSED_UNVALIDATED` | Scaffolding prepared; requires channel training |

---

## 9. Test Suite Execution & Non-Regression Audit

Targeted training scaffold unit tests and the complete repository regression test suite were executed:

### Command 1: Targeted Scaffold Tests
```bash
pytest tests/unit/test_neural_training_scaffold.py
```
- **Result:** **11 passed in 8.22s**, 0 failures, 0 warnings.
- **Coverage:** Individual loss formulas, context cancellation proof, gradient routing, dataset split integrity, test-set lockout, optimizer-step prohibition, smoke checkpointing, clipping diagnostics, and gate registry classifications.

### Command 2: Full Repository Regression Suite
```bash
pytest tests/
```
- **Result:** **175 passed in 7.46s**, 0 failures, 0 warnings.
- **Test Summary Across Modules:**
  - `tests/integration/test_ecc_spatial_roundtrip.py`: 6 passed
  - `tests/integration/test_fail_closed.py`: 6 passed
  - `tests/integration/test_media_pipeline.py`: 3 passed
  - `tests/integration/test_pipeline.py`: 9 passed
  - `tests/unit/test_carrier_capacity.py`: 4 passed
  - `tests/unit/test_carrier_dwt.py`: 3 passed
  - `tests/unit/test_carrier_embedding.py`: 3 passed
  - `tests/unit/test_carrier_header.py`: 2 passed
  - `tests/unit/test_carrier_mask.py`: 4 passed
  - `tests/unit/test_crypto.py`: 23 passed
  - `tests/unit/test_ecc.py`: 24 passed
  - `tests/unit/test_exceptions.py`: 2 passed
  - `tests/unit/test_neural_components.py`: 24 passed
  - `tests/unit/test_neural_extraction.py`: 15 passed
  - `tests/unit/test_neural_reproducibility.py`: 3 passed
  - `tests/unit/test_neural_spatial_bridge.py`: 13 passed
  - `tests/unit/test_neural_training_scaffold.py`: 11 passed
  - `tests/unit/test_serialization.py`: 20 passed
  - **Total:** **175 passed (100% passing)**.

---

## 10. Scope Control & Files Modified Audit

### Files Created:
1. `src/neural/losses.py` — Candidate loss functions, clipping diagnostic, and composite loss module.
2. `src/neural/training_scaffold.py` — Dataset split manager, training configuration, step functions, and gate registry.
3. `tests/unit/test_neural_training_scaffold.py` — 11 targeted unit tests verifying gradient paths, split protection, and step safety.
4. `scripts/experiments/run_sprint3_training_smoke_test.py` — Standalone synthetic smoke-test runner.
5. `docs/experiments/sprint3_training_scaffold_results.json` — Versioned machine-readable results from the synthetic smoke test.
6. `checkpoints/sprint3_smoke/synthetic_smoke_checkpoint.pt` — Clearly labelled synthetic smoke-test checkpoint artifact.
7. `docs/audits/PHASE3_SPRINT3_LOSS_AND_TRAINING_SCAFFOLD_REPORT.md` — This audit report.

### Files Modified:
1. `src/neural/__init__.py` — Cleanly exports the new loss functions and training scaffold classes.

### Prohibited Files Confirmed Untouched:
- `pyproject.toml` — Untouched.
- `docs/project/div2k_hr_split.json` — Untouched.
- All Phase 2 production code (`src/carrier/`, `src/security/`, `src/core/`) — Untouched.
- Historical benchmark JSON files (`sprint2_1_ecc_validation_results.json`, `sprint2_1_1_ecc_validation_results.json`, etc.) — Untouched.
- Prior audit reports — Untouched.

---

## 11. Separation of Facts, Implementation Claims, and Hypotheses

1. **Verified Empirical Facts:**
   - The autograd graph flows continuously from `CompositeLoss` through `SpatialExtractionResult`, `SpatialQuantizeSTE`, `HaarDWT2DTorch`, and `ACSSubINN` into both `ContextPredictorCNN` and `AdaptiveGainCNN`.
   - The extraction residual $\delta_{\text{rec}}$ gradient to `ContextPredictorCNN` cancels out under lossless roundtrip, proving mathematically and empirically that explicit context supervision $L_{\text{ctx}}$ is required.
   - `DatasetSplitManager` strictly enforces the 600/100/100 DIV2K partitioning and rejects unauthorized test split access.
   - The synthetic smoke test executed forward and backward passes with finite gradients on CUDA (GTX 1650) without calling `optimizer.step()`, leaving weights bit-for-bit identical.
   - The full test suite passed (175/175 tests).
2. **Implementation Claims:**
   - The loss equations are implemented without detached graph breaks in the training path.
   - Straight-Through Estimation (STE) is used as an autograd surrogate for non-differentiable rounding and clamping.
3. **Untested Hypotheses & Open Decisions (For Sprint 4+):**
   - *Loss Weight Balancing:* Whether $\lambda_{\text{bit}}=1.0, \lambda_{\text{ctx}}=0.5, \lambda_{\text{gain}}=0.01, \lambda_{\text{img}}=0.1$ converges stably on natural DIV2K carriers without causing gain collapse or PSNR degradation requires empirical calibration.
   - *Clipping Mitigation:* Whether training with $L_{\text{ctx}}$ and $L_{\text{img}}$ sufficiently suppresses high-contrast pixel clipping on carriers like `0005.png`, or whether a differentiable saturation penalty ($L_{\text{sat}}$) must be added to the loss objective.
   - *Lossy Channel Integration:* Differentiable JPEG approximation layers will be needed to train against lossy compression ($Q=80$).

---

## 12. Final Completion Verdict

Sprint 3 scaffolding, gradient routing verification, split protection, and synthetic smoke testing are **COMPLETE**.

```
TRAINING STATUS: NOT AUTHORIZED — awaiting human review of loss equations, gradient paths, split protection, and smoke-test evidence.
```
