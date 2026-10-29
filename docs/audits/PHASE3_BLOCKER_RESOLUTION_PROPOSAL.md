# PaladinMedia — Phase 3 Blocker Resolution Proposal

**Document ID:** `PDM-DESIGN-P3-BLOCKER-PROPOSAL-20261018`
**Date:** 2026-10-18
**Repository:** `E:\Paladin_media`
**Status:** PROPOSED ENGINEERING DESIGN REVIEW (NON-EXECUTED)
**Governing Specifications:** PaladinMedia Architecture Specification §10–§15, System & Research Document Draft §21, Phase 3 Acceptance Criteria (`PDM-P3-GATE-001`), Phase 3 Technical Readiness Checkpoint (`PDM-AUDIT-P3-TECH-READINESS-20261018`)
**Target Hardware:** NVIDIA GeForce GTX 1650 (TU117, Compute Capability 7.5, 4 GB VRAM), Driver 610.47, Python 3.10.11

---

## 1. Executive Summary & Recommendation Overview

This proposal provides evidence-grounded engineering recommendations for the four technical blockers identified in [`docs/audits/PHASE3_TECHNICAL_READINESS_CHECKPOINT.md`](file:///E:/Paladin_media/docs/audits/PHASE3_TECHNICAL_READINESS_CHECKPOINT.md).

It preserves all non-negotiable architectural invariants:
- **No secret-side DWT:** Secret media is serialized and encrypted into an opaque bitstream.
- **No discarded latent-Z:** Full bidirectional invertibility without unrecoverable latent variables.
- **Fail-closed cryptographic sequence:** `Serialize -> Encrypt (AES-256-GCM) -> Reed-Solomon RS(255, 223)`.
- **Classical Phase 2 baseline preservation:** Modular retention of `BaselineEmbedder` for controlled benchmarking.

### Summary of Recommendations

| Blocker ID | Core Technical Blocker | Recommended Engineering Solution | Next Smallest Validation Step | Requires User Approval? |
|:---|:---|:---|:---|:---:|
| **Blocker 1** | Carrier Header Lossy Synchronization Failure | **Dual-Mode Framing:** Retain 16B `CarrierHeader` for lossless PNG; adopt **Implicit Fixed-Tier Framing** anchored by AES-GCM for lossy JPEG channels. | Offline script validating implicit framing on Phase 2 pipeline | **YES** |
| **Blocker 2** | Subband Channel Partition Specification | **Freeze Option A:** $\text{HH}_{\text{keep}} = [\text{Green}, \text{Blue}]$, $\text{HH}_{\text{drop}} = [\text{Red}]$. Maintains 1:1 baseline capacity and mask shape. | Verify tensor shape consistency with `src/carrier/dwt.py` | **YES** |
| **Blocker 3** | Neural Module Topology & Hyperparameters | **Compact ResNet Topology:** 3-block `ContextPredictorCNN`, 2-block `AdaptiveGainCNN` ($\gamma_{\text{min}}=1.0$). Peak memory $< 1.2\text{ GB}$ VRAM. | Synthetic dry-run test of PyTorch model parameters | **YES** |
| **Blocker 4** | PyTorch & Hardware Compatibility | **Official Wheel Bundle:** Install `torch==2.5.1+cu124` and `torchvision` using bundled CUDA 12.4 runtime on driver 610.47. | User executes single `pip install` command | **YES** |

---

## 2. Task 1 — Carrier Header Robustness Under Lossy Channels

### 2.1 Empirical Problem Recap
In Experiment E14 ([`docs/experiments/E14_lossy_channel_degradation.md`](file:///E:/Paladin_media/docs/experiments/E14_lossy_channel_degradation.md)), 100% of JPEG runs (140/140) and Gaussian noise runs (60/60) failed at the carrier header stage (`CarrierHeaderError`). The 16-byte carrier header (128 bits) is embedded in the first 128 coefficients of Red-HH via raw integer parity. Because high-frequency DCT quantization perturbs spatial parity with bit error probability $p \approx 0.50$, the probability of all 128 uncorrected bits surviving is $(1 - p)^{128} \approx 2.9 \times 10^{-39}$. Downstream Reed–Solomon decoding and AES-GCM decryption are never reached.

---

### 2.2 Detailed Evaluation of Candidate Options

```
Carrier Header Architectures under Evaluation:
├── Option 1: Repetition / BCH Error-Correcting Code
│   ├── 1A: Simple Majority-Voting Repetition (15x or 31x)
│   └── 1B: Concatenated Golay(24, 12) + Repetition
├── Option 2: Lower-Frequency Carrier Channel
│   ├── 2A: Red-LH / Red-HL Detail Subband Embedding
│   └── 2B: Spatial Luminance Micro-Gain QIM
└── Option 3: Implicit / Deterministic Fixed-Tier Framing
    └── 3A: Out-of-Band Session Context + Capacity-Tier Alignment
```

#### Comparison Matrix

| Evaluation Dimension | Option 1: Repetition / ECC Code | Option 2: Lower-Frequency Channel | Option 3: Implicit Fixed-Tier Framing |
|:---|:---|:---|:---|
| **Synchronization Mechanism** | Fixed anchor positions (indices $0 \dots 128R-1$) in Red-HH. | Fixed anchor positions in LH/HL subband or spatial quad corners. | Container dimensions $(H, W)$ + authenticated session context. Zero in-band header bits. |
| **Header Overhead** | $15\times \implies 1,920\text{ bits}$ ($240\text{ B}$)<br>$31\times \implies 3,968\text{ bits}$ ($496\text{ B}$). | $128\text{ bits}$ ($16\text{ B}$) in LH/HL subband. | **0 bits (0 B overhead).** |
| **Capacity Impact on HH** | Reduces usable payload by $224\text{ B}$ (at $15\times$) or $480\text{ B}$ (at $31\times$) on $512\times512$. | **Zero impact on HH.** HH retains full $8,192\text{ B}$ raw capacity. | **Increases usable capacity by 16 B.** |
| **Fidelity Impact** | Localized to HH detail band; $\Delta \text{PSNR} < 0.1\text{ dB}$. | Modulates LH/HL subbands. E07 proved LH/HL maintains $\text{PSNR} > 75\text{ dB}$. | **Zero additional distortion.** |
| **Failure Behavior under JPEG** | **Fails mathematically if channel BER $p \approx 0.50$.** At $p=0.48, R=15$, majority survival is $(0.56)^{128} \approx 10^{-33}$. Only works if neural INN already achieves $p < 0.05$. | Parity in LH/HL still flips if DCT quantization shifts pixels by $\pm 1$, unless continuous modulation is used. | **Immune to header bit flips.** Cryptographic integrity is verified fail-closed by AES-256-GCM GHASH tag. |
| **Compatibility with Wire Framing** | 100% compatible with existing `CarrierHeader` dataclass. | 100% compatible with `CarrierHeader`; requires extractor to inspect LH/HL. | Requires fixed payload tiers ($1.5\text{ KB}, 7\text{ KB}, 20\text{ KB}, \text{Max}$) or container padding. |
| **Evaluation Independence** | Depends on channel BER; cannot survive raw parity JPEG without neural engine. | Can be evaluated independently in Phase 2 using LH/HL script. | **Completely independent of neural engine.** Evaluatable immediately. |

---

### 2.3 Task 1 Recommendation & Next Smallest Validation Step

- **Primary Recommendation:** **Dual-Mode Framing Strategy**
  1. **Mode A (Lossless PNG Workflow):** Retain the existing 16-byte in-band `CarrierHeader` embedded at indices 0..127 to guarantee $100\%$ backward compatibility with Phase 2 regression tests and lossless benchmarks.
  2. **Mode B (Robust JPEG $Q=80$ Workflow):** Use **Implicit Fixed-Tier Framing (Option 3)**. The neural network embeds the protected payload into the positions determined by container size and fixed protocol tiers ($1.5\text{ KB}$, $7.0\text{ KB}$, $20.0\text{ KB}$, or full carrier capacity). The receiver derives the candidate bitstream directly from container dimensions, executes Reed–Solomon decoding, and authenticates the ciphertext via AES-256-GCM. If tampering or channel corruption occurs, AES-GCM fails closed, leaking zero plaintext.
- **Smallest Defensible Next Validation Step (Pre-Implementation):**
  Author an offline validation script (`scripts/experiments/verify_implicit_framing.py`) that tests whether `MediaStegoPipeline` can embed and extract fixed-tier payloads without reading an in-band header, verifying fail-closed AES-GCM rejection under simulated channel errors.

---

## 3. Task 2 — Freeze the Subband Partition Contract

### 3.1 Mathematical Context in ACS-Sub-INN v3
Per Architecture Specification §11 and §15, the 1-level 2D Haar transform produces four subbands with shape $(3, H/2, W/2)$ in float64. The neural architecture partitions the high-frequency detail representation:
$$\text{HH} = [\text{HH}_{\text{keep}}, \text{HH}_{\text{drop}}]$$
$$\text{context} = \text{ContextPredictorCNN}(\text{HH}_{\text{keep}})$$
$$\gamma = \text{Softplus}(\text{AdaptiveGainCNN}(\text{HH}_{\text{keep}})) + \gamma_{\text{min}}$$
$$\text{HH}_{\text{drop}}' = (1 - M) \odot \text{HH}_{\text{drop}} + M \odot [\text{context} + \gamma \odot P]$$

**Non-Negotiable Invertibility Invariant:** The receiver extracts payload bits by re-evaluating:
$$\text{context}_{\text{rec}} = \text{ContextPredictorCNN}(\text{HH}_{\text{keep}})$$
$$\hat{P} = \text{sign}(\text{HH}_{\text{drop}}' - \text{context}_{\text{rec}})$$
Therefore, **$\text{HH}_{\text{keep}}$ must remain strictly unmodified during embedding** ($HH_{\text{keep}}' = HH_{\text{keep}}$). Any perturbation to $\text{HH}_{\text{keep}}$ alters the predicted context, breaking receiver inversion.

---

### 3.2 Evaluation of Subband Partition Options

```
Partition Contract Candidates:
├── Option A: Color-Channel Partition (Red-HH Drop, Green+Blue-HH Keep)
│   └── 1 Payload Channel (Red), 2 Context Channels (Green, Blue)
├── Option B: Spatial Sub-Grid Partition (Even/Odd Checkerboard)
│   └── Downsampled half-grids across all 3 color channels
└── Option C: Cross-Subband Detail Partition (LH+HL Keep, HH Drop)
    └── 6 Detail Context Channels (LH, HL), 3 Payload Channels (HH)
```

#### Comparison Matrix

| Evaluation Dimension | Option A: Color-Channel Partition (Red Drop, Green/Blue Keep) | Option B: Spatial Sub-Grid Partition (Even Keep, Odd Drop) | Option C: Cross-Subband Detail Partition (LH+HL Keep, HH Drop) |
|:---|:---|:---|:---|
| **Input Shape to Predictor/Gain** | $(B, 2, H/2, W/2)$ | $(B, 3, H/4, W/4)$ (downsampled) | $(B, 6, H/2, W/2)$ ($\text{LH} + \text{HL}$) |
| **Output Shape (Modulation Target)** | $(B, 1, H/2, W/2)$ (Red HH) | $(B, 3, H/4, W/4)$ | $(B, 3, H/2, W/2)$ (All 3 HH channels) |
| **Payload Capacity ($512\times512$)** | $8,192\text{ raw bytes}$ ($0.218\text{ bpp}_{\text{spatial}}$) | $12,288\text{ raw bytes}$ ($0.327\text{ bpp}_{\text{spatial}}$) | $24,576\text{ raw bytes}$ ($0.654\text{ bpp}_{\text{spatial}}$) |
| **Mask Dimensions & Placement** | 2D boolean mask $(H/2, W/2)$. Exactly matches `src/carrier/mask.py`. | Complex non-contiguous checkerboard indexing. | 3D boolean mask $(3, H/2, W/2)$. |
| **Receiver Context Reproducibility** | **Exact in lossless PNG** ($HH_{\text{keep}}$ untouched). Robust under JPEG when trained with noise. | High spatial phase distortion; aliasing across sub-grids. | **Exact in lossless PNG** ($\text{LH}, \text{HL}$ untouched). Stable under JPEG. |
| **Fair Comparison to Phase 2** | **100% fair 1:1 comparison.** Exactly matches Phase 2 Red-HH allocation and capacity frontier. | Unfair: uses partial spatial grids across 3 channels. | Unfair: represents 3-channel capacity extension (E05/E07). |
| **Visual Distortion Concentration** | Isolated to Red channel where human eye is less sensitive than Green. | Distributed across all 3 color channels. | Distributed across all 3 color channels in HH. |

---

### 3.3 Task 2 Recommendation

- **Primary Recommendation:** **Freeze Option A (Red-HH Drop, Green+Blue-HH Keep) as the Primary Phase 3 Invariant.**
  - **Reason 1 (Exact Baseline Parity):** Phase 2 classical baseline embeds strictly in Red-HH. Benchmarking ACS-Sub-INN against Phase 2 under Option A isolates the embedding engine's performance without confounding capacity scaling.
  - **Reason 2 (Mask Reusability):** Uses the exact existing 2D deterministic mask shape $(H/2, W/2)$ from [`src/carrier/mask.py`](file:///E:/Paladin_media/src/carrier/mask.py).
  - **Reason 3 (Context Ratio):** Provides two context channels (Green and Blue) to predict one payload channel (Red), ensuring an overdetermined prediction space.
  - **Extension Pathway:** Option C (Cross-Subband) is formally cataloged as the **Phase 3.1 High-Capacity Extension** ($0.654\text{ bpp}_{\text{spatial}}$) after Option A achieves Gate 1–3 acceptance.

---

## 4. Task 3 — Neural Topology Specification

### 4.1 Separation of Invariants vs. Tunable Hyperparameters

To prevent premature over-constraining while ensuring strict architectural alignment with Architecture §15, the neural design is separated into:

```
Neural Engine Specifications:
├── ARCHITECTURAL INVARIANTS (FROZEN - Implementation Contract)
│   ├── Dimension-preserving fully convolutional topology (stride=1, padding='same')
│   ├── Input tensor: (B, 2, H/2, W/2) -> Output tensor: (B, 1, H/2, W/2)
│   ├── Linear activation on ContextPredictorCNN output (unbounded baseline)
│   ├── Strictly positive gain: gamma = Softplus(x) + gamma_min
│   ├── Straight-Through Estimator (STE) rounding layer for 8-bit backprop
│   └── Reversible modulation: HH_drop' = (1-M)*HH_drop + M*(context + gamma*P)
└── TUNABLE HYPERPARAMETERS (EXPERIMENTAL - Explored in Phase 4)
    ├── Number of residual blocks (K in {3, 4, 6})
    ├── Hidden channel width (C_hidden in {32, 64})
    ├── Gain floor parameter (gamma_min in [0.5, 2.0])
    ├── Convolution kernel sizes (3x3 vs 5x5)
    └── Loss weights (lambda_stego, lambda_payload, lambda_hf)
```

---

### 4.2 Proposed Minimal Initial Topology (GTX 1650 4GB Optimized)

```
ContextPredictorCNN:
(B, 2, H/2, W/2) [Green, Blue HH]
       │
       ▼
Conv2D(in=2, out=32, k=3, p=1) + LeakyReLU(0.2)
       │
       ▼
3 × ResidualBlock(Conv32 -> LeakyReLU -> Conv32 + Residual)
       │
       ▼
Conv2D(in=32, out=1, k=3, p=1) [Linear Output]
       │
       ▼
(B, 1, H/2, W/2) [Predicted Red HH Context]

AdaptiveGainCNN:
(B, 2, H/2, W/2) [Green, Blue HH]
       │
       ▼
Conv2D(in=2, out=32, k=3, p=1) + LeakyReLU(0.2)
       │
       ▼
2 × ResidualBlock(Conv32 -> LeakyReLU -> Conv32 + Residual)
       │
       ▼
Conv2D(in=32, out=1, k=3, p=1) + Softplus() + gamma_min (1.0)
       │
       ▼
(B, 1, H/2, W/2) [Spatially Varying Gain gamma(x, y) >= 1.0]
```

- **Parameter Budget:**
  - `ContextPredictorCNN`: $\approx 55,425$ parameters ($\approx 222\text{ KB}$ FP32).
  - `AdaptiveGainCNN`: $\approx 36,929$ parameters ($\approx 148\text{ KB}$ FP32).
  - **Combined Model Size:** $\approx 92,354$ parameters ($\approx 370\text{ KB}$ FP32).
- **VRAM Footprint on GTX 1650 (4 GB):**
  - Model weights: $< 1\text{ MB}$.
  - Forward/backward activations ($B=4, 512\times512$ carriers): $\approx 380\text{ MB}$.
  - Optimizer states (AdamW): $\approx 3\text{ MB}$.
  - Total training VRAM working set: $\approx 1.1\text{ GB} \ll 4.0\text{ GB}$ hardware ceiling.
- **Role of the Gain Floor ($\gamma_{\text{min}} = 1.0$):**
  Without $\gamma_{\text{min}}$, joint optimization collapses $\gamma \to 0$ to achieve zero distortion ($\text{PSNR} \to \infty$), destroying payload recovery. Setting $\gamma_{\text{min}} = 1.0$ enforces a minimum perturbation matching the $\pm 1$ step size of the Phase 2 classical baseline.

---

## 5. Task 4 — PyTorch and Hardware Compatibility

### 5.1 Environmental Inspection
- **Hardware:** NVIDIA GeForce GTX 1650 (Turing TU117, Compute Capability 7.5, 4,096 MiB VRAM).
- **Host Driver:** NVIDIA Display Driver 610.47 (Supports CUDA up to version 13.3).
- **Host Platform:** Windows 10 (Build 10.0.26200, 64-bit AMD64).
- **Python Runtime:** Python 3.10.11 (tags/v3.10.11:7d4cc5a).
- **Existing Dependencies:** `cryptography>=41.0.0`, `reedsolo>=1.7.0`, `numpy>=1.24.0`, `pillow>=10.0.0`. `torch` is not currently installed.

---

### 5.2 Key Technical Distinction: CUDA Driver vs. PyTorch Bundled Runtime

The locally reported `CUDA UMD Version: 13.3` reflects the maximum CUDA driver version supported by the installed 610.47 display driver. **It does not require a local CUDA 13 toolkit installation.**

PyTorch binary wheels distributed via `download.pytorch.org` bundle their own self-contained CUDA runtime libraries (`cudart64_*.dll`, `cublas64_*.dll`, `cudnn64_*.dll`). Turing architecture (Compute Capability 7.5, `sm_75`) is natively supported in official PyTorch CUDA 12.4 binary builds.

### 5.3 Recommended Installation Command

When authorized, the user should install the official PyTorch 2.5.1 + CUDA 12.4 wheel bundle:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 5.4 VRAM Management Strategy for 4 GB GTX 1650

1. **Training Mode ($512 \times 512$ Carriers):**
   - Batch size $B=4$.
   - Mixed-precision training via `torch.cuda.amp.autocast(dtype=torch.float16)`.
   - Gradient accumulation (4 steps) to simulate effective batch size $B=16$.
   - Total peak memory: $\approx 1.1\text{ GB}$ (leaving $\approx 2.9\text{ GB}$ headroom).
2. **Evaluation & Inference Mode ($1024 \times 1024$ Carriers):**
   - Batch size $B=1$.
   - Inference wrapped in `torch.no_grad()`.
   - Total peak memory: $\approx 420\text{ MB}$.

---

## 6. Dependencies Between Architectural Decisions

The four blocker decisions exhibit strict sequential coupling:

```
[Blocker 4: PyTorch & Hardware Setup]
               │
               ▼
[Blocker 2: Subband Channel Partition (Option A)]
               │
               ├── Determines tensor shapes (B, 2, H/2, W/2) -> (B, 1, H/2, W/2)
               ▼
[Blocker 3: Neural Model Topology Specification]
               │
               ├── Defines forward embedding & residual extraction
               ▼
[Blocker 1: Carrier Header Synchronization Strategy]
               │
               └── Resolves whether extraction requires header parsing or implicit framing
```

1. **Blocker 2 governs Blocker 3:** The channel partition directly dictates the input and output channel counts of `ContextPredictorCNN` and `AdaptiveGainCNN`.
2. **Blocker 1 governs the extraction pipeline:** If implicit framing is selected for lossy channels, the receiver bypasses `Extractor.extract()` header parsing and passes the neural-extracted bitstream directly to `ReedSolomonEncoder.decode()`.
3. **Blocker 4 is an environment prerequisite:** Neural code cannot be tested without PyTorch.

---

## 7. Minimum Validation Work Required Before Implementation

Prior to authoring full PyTorch neural models, the following minimal, zero-risk validation checks should be executed:

1. **Check 1: Implicit Framing Feasibility Script:** An isolated script (`scripts/experiments/test_implicit_framing_dryrun.py`) testing the classical baseline with implicit tier-based length parsing to confirm 100% exact byte recovery without in-band header reading.
2. **Check 2: PyTorch CUDA Sanity Test:** Following package installation, a one-line sanity check verifying CUDA availability, Turing `sm_75` capability, and tensor allocation on the GTX 1650:
   ```powershell
   python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0), '| VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
   ```
3. **Check 3: Synthetic Forward/Inverse DWT Tensor Shape Verification:** Verify that PyTorch tensor conversion `(B, 3, H, W) <-> (B, 3, H/2, W/2)` preserves exact parity with `HaarDWT2D.forward()`.

---

## 8. Explicit User Approvals Required

The following engineering decisions require explicit user confirmation before proceeding to Phase 3 implementation:

1. **Approval 1 (Carrier Header):** Confirm adoption of **Dual-Mode Framing** (Option 3 Implicit Fixed-Tier Framing for lossy JPEG channels, preserving the 16B header for lossless PNG).
2. **Approval 2 (Subband Partition):** Confirm freezing **Option A** ($\text{HH}_{\text{keep}} = [\text{Green}, \text{Blue}]$, $\text{HH}_{\text{drop}} = [\text{Red}]$) as the Phase 3 primary contract, keeping Option C (Cross-Subband) as a Phase 3.1 high-capacity extension.
3. **Approval 3 (Neural Topology):** Confirm the compact ResNet parameter budget ($\approx 92\text{K}$ weights, $\gamma_{\text{min}}=1.0$, FP32 memory $< 1.2\text{ GB}$) tailored for the 4 GB GTX 1650.
4. **Approval 4 (PyTorch Installation):** Confirm permission to execute `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` when implementation begins.

---

## 9. Separation of Verified Facts, Engineering Judgments, and Untested Hypotheses

| Category | Statement | Basis / Evidence |
|:---|:---|:---|
| **VERIFIED FACT** | Classical baseline raw parity carrier header fails 100% under JPEG $Q \in [70, 100]$ and noise $\sigma \ge 0.5$. | E14 JSON run records (`docs/experiments/e14_results.json`). |
| **VERIFIED FACT** | Target GPU is NVIDIA GeForce GTX 1650 with 4,096 MiB VRAM running driver 610.47 on Windows 10. | Output of `nvidia-smi` executed on host system. |
| **VERIFIED FACT** | Phase 2 implementation strictly enforces fail-closed rejection and exact byte equality in lossless mode. | 103 passing tests in `pytest tests/`. |
| **ENGINEERING JUDGMENT** | Option A (Red-HH Drop) provides the fairest comparison against Phase 2 by keeping carrier capacity identical. | Based on scientific control principles; isolating embedding algorithm from capacity expansion. |
| **ENGINEERING JUDGMENT** | Repetition coding in discrete integer parity space cannot survive JPEG compression. | Mathematical probability: $(0.56)^{128} \approx 6.5 \times 10^{-33}$. |
| **ENGINEERING JUDGMENT** | A 92K-parameter model with $B=4$ batch size will fit within 1.2 GB VRAM on the GTX 1650. | Analytical activation memory calculation ($4 \times 32 \times 256 \times 256 \times 4\text{ B} \approx 33.5\text{ MB}$ per layer). |
| **UNTESTED HYPOTHESIS** | ACS-Sub-INN v3 will achieve pre-ECC $\text{BER} \le 3.0\%$ through JPEG compression at $Q=80$. | Primary research hypothesis of Phase 3; requires training and empirical testing against E14 baseline. |
| **UNTESTED HYPOTHESIS** | The Adaptive Gain CNN will maintain stego PSNR $> 42\text{ dB}$ under active JPEG simulation. | Acceptance Criteria Gate 2 target; unproven until model training is executed. |
