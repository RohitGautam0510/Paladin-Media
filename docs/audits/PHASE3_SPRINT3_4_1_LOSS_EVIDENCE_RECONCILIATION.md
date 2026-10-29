# PaladinMedia — Phase 3, Sprint 3.4.1
# Loss Diagnostics, Gradient Evidence, and Readiness Report Reconciliation

**Project:** PaladinMedia
**Repository:** `E:\Paladin_media`
**Phase:** 3 — ACS-Sub-INN v3 Integration
**Sprint:** 3.4.1 — Loss Diagnostics, Gradient Evidence, and Readiness Report Reconciliation
**Date:** 2026-10-20
**Status:** COMPLETE (VERDICT: `PASS`)
**Governing Skill:** Gravitas Core (`.agents/skills/gravitas/SKILL.md`)
**Authorization Boundary:** Investigation, autograd diagnostics, metric reconciliation, and audit reporting only. **NO MODEL TRAINING WAS CONDUCTED.**

---

## 1. Executive Summary & Purpose

Sprint 3.4.1 resolves all empirical and narrative discrepancies identified during the audit review of Sprint 3.4. Specifically, this sprint reconciles:
1. **Raw vs. Weighted Loss Nomenclature & Summations:** Corrects the JSON field labeling and table presentation in which raw unweighted components were stored with `_weighted` suffixes, causing an apparent discrepancy between component values and the composite total.
2. **Gradient Norm Measurements & Clipping Truth:** Corrects the reporting of composite gradient norms by measuring the actual unclipped gradient norms from the autograd graph, establishing the Pythagorean relationship between disjoint parameter subnets ($\Vert g_{\text{model}} \Vert_2 \equiv \sqrt{\Vert g_{\text{ctx}} \Vert_2^2 + \Vert g_{\text{gain}} \Vert_2^2}$), and clarifying that earlier reported norms were unclipped rather than post-clipped.
3. **Context-Supervision Claim Re-Evaluation:** Refines the scientific claim regarding context supervision, proving that while `ContextPredictionLoss` is the sole source of *direct high-frequency target reconstruction supervision*, `ImageFidelityLoss` also contributes an *indirect spatial fidelity gradient* to `ContextPredictorCNN` via the differentiable inverse DWT and straight-through estimator (STE).

Historical records (`docs/experiments/sprint3_4_readiness_audit_results.json` and `docs/audits/PHASE3_SPRINT3_4_TRAINING_READINESS_REPORT.md`) are preserved untouched as prior evidence. Reconciled measurements are versioned at:
- **Reconciliation Data:** `docs/experiments/sprint3_4_1_loss_reconciliation_results.json`
- **Reconciliation Runner:** `scripts/experiments/run_sprint3_4_1_loss_reconciliation.py`
- **Targeted Regression Suite:** `tests/unit/test_loss_reconciliation.py`

---

## 2. Corrected Raw and Weighted Component Loss Table

In Sprint 3.4, `run_sprint3_4_readiness_audit.py` extracted fields from `LossBreakdown` using keys named `_weighted` (e.g. `ctx_loss_weighted: 3.332684`), even though `LossBreakdown` stores the raw, unweighted losses (`l_ctx`). When displayed in Section 5.1 of the Sprint 3.4 report, raw values and loss weights were listed without separate columns for raw vs. weighted contributions, making the composite total ($1.9271$) appear inconsistent with the sum of displayed numbers ($0.1864 + 3.3327 + 0.7179 + 0.6712 = 4.9082$).

Under deterministic diagnostic evaluation on DIV2K carrier `0001.png` ($512 \times 512$ center crop) with `seed=42`:

### Reconciled Loss Breakdown Table

| Loss Component | Mathematical Objective | Raw Component Loss ($\mathcal{L}_i$) | Loss Weight ($\lambda_i$) | Weighted Contribution ($\lambda_i \cdot \mathcal{L}_i$) | % of Total Loss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`BitRecoveryLoss`** | Demodulation margin (logistic) | 0.187489 | 1.00 | **0.187489** | 9.83% |
| **`ContextPredictionLoss`** | Smooth L1 subband prediction | 3.292353 | 0.50 | **1.646176** | 86.31% |
| **`AdaptiveGainRegularization`**| Excess gain penalty above $\gamma_{\min}$ | 0.714299 | 0.01 | **0.007143** | 0.37% |
| **`ImageFidelityLoss`** | Spatial cover-stego MAE (L1) | 0.664441 | 0.10 | **0.066444** | 3.48% |
| **`CompositeLoss` (Total)** | Multi-task objective sum | — | — | **1.907252** | **100.00%** |

*Note: In the historical Sprint 3.4 run, the carrier was evaluated after an intermediate seed call, yielding raw losses $(0.186413, 3.332684, 0.717876, 0.671244)$, whose weighted sum was $1.0(0.186413) + 0.5(3.332684) + 0.01(0.717876) + 0.1(0.671244) = 1.927058$, exactly matching the reported total loss $1.927058$. In Sprint 3.4.1, initializing directly under seed 42 on CUDA yields $(0.187489, 3.292353, 0.714299, 0.664441)$, whose weighted sum is $1.907252$, matching `comp_breakdown.total_loss` to machine precision.*

---

## 3. Composite-Loss Equation and Numerical Check

### 3.1 Composite Loss Equation
The governing loss formulation defined in `src/neural/losses.py` is:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{bit}} \cdot \mathcal{L}_{\text{bit}} + \lambda_{\text{ctx}} \cdot \mathcal{L}_{\text{ctx}} + \lambda_{\text{gain}} \cdot \mathcal{L}_{\text{gain}} + \lambda_{\text{img}} \cdot \mathcal{L}_{\text{img}}$$

Substituting the configured weights from source:
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{bit}} + 0.5 \cdot \mathcal{L}_{\text{ctx}} + 0.01 \cdot \mathcal{L}_{\text{gain}} + 0.1 \cdot \mathcal{L}_{\text{img}}$$

### 3.2 Numerical Consistency Verification
Evaluating the constituent terms to full IEEE 754 float32 precision:
- $1.0 \times 0.1874885857105255 = 0.1874885857105255$
- $0.5 \times 3.2923526763916016 = 1.6461763381958008$
- $0.01 \times 0.7142987251281738 = 0.0071429872512817$
- $0.1 \times 0.6644414663314819 = 0.0664441466331482$

$$\sum_{i} \lambda_i \mathcal{L}_i = 1.9072520577907563$$
$$\text{Tensor Total Loss (`total_loss.item()`)} = 1.9072520732879639$$

$$\left| \sum \lambda_i \mathcal{L}_i - \mathcal{L}_{\text{total}} \right| = 1.55 \times 10^{-8} < 10^{-6}$$

**Verdict:** The total loss is mathematically identical to the weighted sum of its components within floating-point representation limits.

---

## 4. Gradient Norms by Loss and Module

### 4.1 Submodule Orthogonality Theorem
Let the total model parameter vector $\theta$ be partitioned into two disjoint subnets:
$$\theta = \begin{bmatrix} \theta_C \\ \theta_G \end{bmatrix}$$
where $\theta_C \in \mathbb{R}^{d_C}$ represents the parameters of `ContextPredictorCNN`, and $\theta_G \in \mathbb{R}^{d_G}$ represents the parameters of `AdaptiveGainCNN` ($d_C + d_G = d$).

Because the parameter sets are mutually exclusive (no shared weights between subnets), their gradient vectors are orthogonal in parameter space:
$$\langle \nabla_{\theta_C} \mathcal{L}, \nabla_{\theta_G} \mathcal{L} \rangle = 0$$

Therefore, the whole-model Euclidean gradient norm is strictly Pythagorean:
$$\Vert \nabla_\theta \mathcal{L} \Vert_2 = \sqrt{\Vert \nabla_{\theta_C} \mathcal{L} \Vert_2^2 + \Vert \nabla_{\theta_G} \mathcal{L} \Vert_2^2}$$

### 4.2 Comprehensive Gradient Measurement Table

The table below reports gradient norms measured directly from the autograd graph for each isolated backward pass and the composite loss backward pass:

| Objective Pass | ContextPredictor $\Vert \nabla_{\theta_C} \Vert_2$ | AdaptiveGain $\Vert \nabla_{\theta_G} \Vert_2$ | Whole Model $\Vert \nabla_\theta \Vert_2$ | Pythagorean Identity Check | Finite? | Clipping Applied? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BitRecoveryLoss` (Isolated)** | **0.000000** | 9.268129 | 9.268129 | $\sqrt{0^2 + 9.27^2} = 9.27$ | Yes | No |
| **`ContextPredictionLoss` (Isolated)** | 44.812527 | **0.000000** | 44.812527 | $\sqrt{44.81^2 + 0^2} = 44.81$ | Yes | No |
| **`AdaptiveGainRegularization` (Isolated)** | **0.000000** | 55.618015 | 55.618015 | $\sqrt{0^2 + 55.62^2} = 55.62$ | Yes | No |
| **`ImageFidelityLoss` (Isolated)** | 6.942451 | 2.509330 | 7.382030 | $\sqrt{6.94^2 + 2.51^2} = 7.38$ | Yes | No |
| **`CompositeLoss` (Unclipped)** | **23.100409** | **8.461601** | **24.601376** | $\sqrt{23.10^2 + 8.46^2} = 24.60$ | **Yes** | **No** |
| **`CompositeLoss` (Clipped @ 1.0)** | **0.938988** | **0.343948** | **1.000000** | $\sqrt{0.94^2 + 0.34^2} = 1.00$ | **Yes** | **Yes** |

### 4.3 Resolution of the Sprint 3.4 Table Discrepancy
In Sprint 3.4 Table 5.1, the composite row listed ContextPredictor norm as `35.172481` and AdaptiveGain norm as `73.633945`. Analysis of the calculation reveals that those numbers were synthetically computed from the unweighted isolated runs ($\sqrt{34.76^2 + 5.35^2} = 35.17$ and $\sqrt{12.01^2 + 72.57^2 + 3.26^2} = 73.63$), ignoring the loss weights ($\lambda_{\text{ctx}}=0.5, \lambda_{\text{gain}}=0.01, \lambda_{\text{img}}=0.1$).

When the composite objective is actually backpropagated:
$$\nabla_\theta \mathcal{L}_{\text{total}} = 1.0 \nabla \mathcal{L}_{\text{bit}} + 0.5 \nabla \mathcal{L}_{\text{ctx}} + 0.01 \nabla \mathcal{L}_{\text{gain}} + 0.1 \nabla \mathcal{L}_{\text{img}}$$
The true unclipped composite gradient norms are **23.100409** for ContextPredictor and **8.461601** for AdaptiveGain, combining to a total whole-model norm of **24.601376**.

---

## 5. Verified Gradient Clipping Execution Path

### 5.1 Diagnostic Path Inspection
Inspection of the execution paths reveals two distinct contexts in the codebase:
1. **Production Scaffold Path (`execute_training_step` in `src/neural/training_scaffold.py`):**
   Calls `torch.nn.utils.clip_grad_norm_(bridge.acs_model.parameters(), clip_grad_norm)` when `clip_grad_norm` is not None.
2. **Diagnostic Evaluation Path (`run_sprint3_4_readiness_audit.py`):**
   Diagnostic E directly invoked `comp_breakdown.total_loss.backward()` and calculated the L2 norm across all parameters without calling `clip_grad_norm_`.

### 5.2 Clarification of Prior Report Narrative
The Sprint 3.4 report stated:
> `Total Composite Gradient L2 Norm: 21.003736 (post gradient clipping at norm 1.0: normalized to 1.0)`

This phrasing was a descriptive error. The reported norm ($21.003736$ in Sprint 3.4, $24.601376$ in Sprint 3.4.1) was the **raw unclipped gradient norm**. No clipping had been executed on that gradient set.

### 5.3 Formal Verification of Clipping Behavior
When `clip_grad_norm_(max_norm=1.0)` is applied to the composite gradients:
- Scaling factor: $\alpha = \frac{\text{max\_norm}}{\Vert g \Vert_2} = \frac{1.0}{24.601376} = 0.040648$
- Clipped ContextPredictor norm: $23.100409 \times 0.040648 = 0.938988$
- Clipped AdaptiveGain norm: $8.461601 \times 0.040648 = 0.343948$
- Clipped Whole-Model norm: $24.601376 \times 0.040648 = \mathbf{1.000000}$

All clipped gradients remain finite, non-zero, and properly proportioned.

---

## 6. Revised Context-Cancellation and Context-Supervision Conclusions

### 6.1 Context Cancellation in Bit-Recovery Loss
Under idealized straight-through estimation (STE) without pixel-domain dynamic range saturation:
$$\Delta_{\text{rec}} = HH_{\text{drop}}' - C_{\theta_C}(HH_{\text{keep}})$$
$$HH_{\text{drop}}' = C_{\theta_C}(HH_{\text{keep}}) + \gamma \cdot P$$
$$\Delta_{\text{rec}} = \left[ C_{\theta_C}(HH_{\text{keep}}) + \gamma \cdot P \right] - C_{\theta_C}(HH_{\text{keep}}) = \gamma \cdot P$$

Because $C_{\theta_C}(HH_{\text{keep}})$ cancels out identically in $\Delta_{\text{rec}}$:
$$\frac{\partial \Delta_{\text{rec}}}{\partial C_{\theta_C}} = 0 \implies \frac{\partial \mathcal{L}_{\text{bit}}}{\partial \theta_C} \equiv \mathbf{0}$$
The measured gradient norm is **0.000000**, confirming this algebraic cancellation.

### 6.2 Context Gradient Flow from Image-Fidelity Loss
In the Sprint 3.4 report, Section 1 and Section 12 claimed that `ContextPredictionLoss` was the "sole provider of context gradient". This was imprecise.

`ImageFidelityLoss` compares the spatial stego carrier $\tilde{S}$ against cover $S$:
$$\mathcal{L}_{\text{img}} = \Vert \tilde{S} - S \Vert_1$$
Because $\tilde{S} = \text{IDWT}(LL, LH, HL, HH')$ where $HH' = C_{\theta_C}(HH_{\text{keep}}) + \gamma \cdot P$, the spatial loss backpropagates through the inverse DWT and straight-through rounding:
$$\frac{\partial \mathcal{L}_{\text{img}}}{\partial \theta_C} = \frac{\partial \mathcal{L}_{\text{img}}}{\partial \tilde{S}} \cdot \frac{\partial \tilde{S}}{\partial HH'} \cdot \frac{\partial C_{\theta_C}}{\partial \theta_C} \neq \mathbf{0}$$
The measured gradient norm is **6.942451** (unweighted) and **0.694245** (weighted).

### 6.3 Scientific Distinction: Direct Reconstruction vs. Indirect Regularization
We must clearly distinguish the roles of the two loss functions:
1. **`ContextPredictionLoss` (Direct Target Reconstruction Supervision):**
   Supervises $C_{\theta_C}(HH_{\text{keep}})$ to match the unmodulated target subband $HH_{\text{drop}}$. This is the **sole source of direct high-frequency texture reconstruction learning**.
2. **`ImageFidelityLoss` (Indirect Distortion Regularization):**
   Penalizes spatial differences between cover and stego images. It acts as an indirect regularizer that prevents $C_{\theta_C}$ from predicting excessively large coefficients that would distort the image, but it does **not** teach the model to reconstruct the true high-frequency drop subband.

**Revised Conclusion:** `ContextPredictionLoss` is the sole provider of *direct target-reconstruction supervision* for `ContextPredictorCNN`. `ImageFidelityLoss` provides a non-zero *indirect spatial fidelity gradient*, while `BitRecoveryLoss` provides zero gradient under lossless STE.

---

## 7. Verification Evidence & Test Outcomes

### 7.1 Targeted Regression Tests
A new targeted regression suite was created in `tests/unit/test_loss_reconciliation.py`:
- `test_composite_loss_weighted_sum_exactness`: PASSED
- `test_gradient_norm_orthogonality`: PASSED
- `test_gradient_clipping_scaling_behavior`: PASSED
- `test_context_supervision_gradient_paths`: PASSED

Targeted test execution:
```bash
pytest tests/unit/test_loss_gradient_isolation.py tests/unit/test_loss_reconciliation.py tests/unit/test_neural_training_scaffold.py tests/unit/test_dataset_boundary.py
```
**Outcome:** **52 passed in 6.81s** (0 failed).

### 7.2 Full Test Suite Execution
A single complete test suite execution was performed under the `MAX_FULL_SUITE_RUNS = 1` budget:
```bash
pytest tests/
```
**Outcome:** **284 passed in 35.13s** (0 failed, 0 warnings).

---

## 8. Integrity Invariants Audit

| Invariant Requirement | Status | Evidence |
| :--- | :---: | :--- |
| **Model Parameters Immutability** | **PASS** | Parameter digest before and after diagnostic execution matches bit-for-bit: `fe7462957b0194d109ec32a78834d7bf44b9d1aa86510a40625f2e485fa40369` |
| **Zero Optimizer Steps** | **PASS** | `optimizer.step()` was never invoked; `optimizer_step_executed = False` |
| **Dataset Manifest Integrity** | **PASS** | `docs/project/div2k_hr_split.json` SHA-256 (`db605887...`) and composite SHA-256 (`d0f5f4ee...`) unchanged |
| **800-Image Inventory** | **PASS** | All 800 images intact; no files modified or deleted |
| **Loss Weight Immutability** | **PASS** | Approved candidate weights ($\lambda = 1.0, 0.5, 0.01, 0.1$) preserved without alteration |
| **Architecture Invariance** | **PASS** | ACS-Sub-INN v3 frozen; Red-HH target and Green/Blue-HH context preserved |
| **Historical Preservation** | **PASS** | Historical Sprint 3.4 reports and JSON preserved untouched |

---

## 9. Outstanding Objective Risks & Decisions for Phase 4

1. **Loss Weight Calibration:** The candidate loss weights ($\lambda_{\text{bit}}=1.0, \lambda_{\text{ctx}}=0.5, \lambda_{\text{gain}}=0.01, \lambda_{\text{img}}=0.1$) balance gradient magnitudes well in diagnostics (total unclipped norm $\approx 24.6$). However, they remain unvalidated by multi-epoch training dynamics. A pilot calibration phase in Phase 4 is required to tune these weights empirically against validation BER and PSNR targets.
2. **Gradient Clipping Threshold:** Diagnostic evidence proves that with whole-model unclipped norm $\approx 24.6$, a clipping threshold of $1.0$ scales all parameter gradients down by a factor of $\approx 24.6$. Phase 4 must monitor whether a clip threshold of $1.0$ overly slows down early optimization or if a threshold of $5.0$–$10.0$ allows faster initial convergence without instability.

---

## 10. Final Readiness Verdict

### Final Verdict: `PASS`

All discrepancies identified in Sprint 3.4 have been fully investigated, corrected, and verified with fresh, reproducible empirical evidence. The loss summations, gradient norms, clipping mechanisms, and context-supervision dynamics are mathematically sound and consistent across code, data, and documentation.

> **TRAINING STATUS: NOT AUTHORIZED**
> Sprint 3.4.1 was strictly a reconciliation and audit sprint. Real model training remains gated on Phase 4 training loop implementation and explicit user authorization.
