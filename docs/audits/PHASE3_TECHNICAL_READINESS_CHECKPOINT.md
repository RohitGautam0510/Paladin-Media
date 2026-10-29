# PaladinMedia — Phase 3 Technical Readiness Checkpoint

**Document ID:** `PDM-AUDIT-P3-TECH-READINESS-20261018`
**Date:** 2026-10-18
**Repository:** `E:\Paladin_media`
**Auditor:** Senior Research Validation & Systems Architecture Team
**Governing Specifications:** PaladinMedia Architecture Specification §10–§15, System & Research Document Draft §21, Sprint-0 IRS, Phase 3 Acceptance Criteria (`PDM-P3-GATE-001`)
**Scope:** Engineering & Scientific Traceability Review prior to Phase 3 Implementation

---

## 1. Executive Verdict & Readiness Overview

### Executive Verdict: **READY WITH BLOCKERS**

The PaladinMedia Phase 2 signal-processing, cryptographic foundation, and capacity-accounting implementations are mathematically rigorous, fully tested, and verified on local datasets. However, before authoring Phase 3 neural steganography (ACS-Sub-INN v3) implementation code, four discrete architectural and interface prerequisites must be resolved.

```
+-------------------------------------------------------------------------------------------------+
|                                    PHASE 3 TECHNICAL READINESS                                  |
+-------------------------------------------------+-----------------------------------------------+
| Subsystem / Dimension                           | Status                                        |
+-------------------------------------------------+-----------------------------------------------+
| 1. Cryptographic Wire Framing & Fail-Closed     | VERIFIED & READY                              |
| 2. Signal Processing (Haar DWT / IDWT)          | VERIFIED & READY                              |
| 3. Capacity Hard Gate & Exact Accounting        | VERIFIED & RECONCILED                         |
| 4. Dataset Provenance & Split Manifest          | VERIFIED & RECONCILED                         |
| 5. Classical Channel Degradation Baseline (E14) | VERIFIED & CLOSED (Narrow Scope)              |
| 6. Classical Baseline Regression Suite (pytest) | VERIFIED (103/103 tests passing in 3.22s)     |
| 7. Frozen ACS-Sub-INN v3 Architecture Alignment | ALIGNED (No secret DWT, no discarded Z)       |
+-------------------------------------------------+-----------------------------------------------+
| 8. Carrier Header Lossy Protection Strategy     | BLOCKER (Fails under JPEG in classical setup) |
| 9. Subband Channel Partition Specification      | BLOCKER (HH_keep vs HH_drop channels unbound) |
| 10. Neural Module Topology & Hyperparameters    | BLOCKER (Layer depths/kernel sizes unbound)   |
| 11. PyTorch Dependency & Environment Freeze     | BLOCKER (Missing from pyproject.toml)         |
+-------------------------------------------------+-----------------------------------------------+
```

---

## 2. Capacity Accounting Reconciliation Against Implementation

The capacity calculations across code, unit tests, and experiment reports have been audited and traced to their underlying mathematical definitions.

### 2.1 Full Mathematical Breakdown & Overhead Hierarchy

The raw carrier bit capacity for a 1-level 2D Haar transform on a single subband channel ($2\text{HH}$ parity) is:
$$C_{\text{raw\_bits}}(N) = \left(\frac{N}{2}\right)^2 \implies C_{\text{raw\_bytes}}(N) = \frac{N^2}{32} \text{ bytes}$$

From raw carrier bytes, the transmission hierarchy enforces strict sequential deductions:
1. **Fixed Carrier Header ($16\text{ B}$):** Reserves the first 128 coefficients for metadata (`PLDH`, version, flags, reserved, session counter, payload length).
   $$C_{\text{max\_protected}}(N) = C_{\text{raw\_bytes}}(N) - 16 = \frac{N^2}{32} - 16$$
2. **Reed–Solomon Length Prefix ($4\text{ B}$):** Systematic framing prepends a 4-byte big-endian payload length:
   $$C_{\text{avail\_blocks}}(N) = C_{\text{max\_protected}}(N) - 4$$
3. **Reed–Solomon Error Correction ($\text{RS}(255, 223)$):** Appends 32 parity bytes per block ($k=223, 2t=32$).
   - Full blocks: $B_{\text{full}} = \lfloor C_{\text{avail\_blocks}} / 255 \rfloor$.
   - Remainder: $R = C_{\text{avail\_blocks}} \pmod{255}$.
   - Shortened blocks: A partial block of $d$ data bytes requires $d + 32$ total bytes ($1 \le d \le 222$). If $R \ge 33$, it accommodates $d = R - 32$ data bytes; if $R < 33$, $d = 0$.
   $$C_{\text{max\_encrypted}}(N) = B_{\text{full}} \times 223 + \max(0, R - 32)$$
4. **AES-256-GCM Encryption Overhead ($28\text{ B}$):** Comprises a 12-byte initialization vector (IV/nonce) and a 16-byte GHASH authentication tag:
   $$C_{\text{max\_serialized}}(N) = C_{\text{max\_encrypted}}(N) - 28$$
5. **SecretSerializer Framing Overhead ($18\text{ B} + M$):** Base wire framing requires 18 bytes (10-byte magic prefix `PLD_SECRET`, 4-byte payload length, 4-byte CRC-32 checksum) plus $M$ bytes of JSON-serialized metadata:
   $$C_{\text{usable\_secret}}(N, M) = C_{\text{max\_serialized}}(N) - 18 - M = C_{\text{max\_encrypted}}(N) - 46 - M$$

---

### 2.2 Numerical Reconciliation Table

| Dimension ($N \times N$) | Subband Dim | Raw Capacity | Max Protected | Full RS Blocks | Remainder $R$ | Shortened RS Data | Max Encrypted | Usable Plaintext ($M=0$) | Usable Plaintext ($M=2$, Ref `{}`) | Usable Plaintext ($M=48$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$512 \times 512$** | $256 \times 256$ | 8,192 B | 8,176 B | 32 | 12 B | 0 B ($12 < 33$) | 7,136 B | **7,090 B** | **7,088 B** | 7,042 B |
| **$1024 \times 1024$ (Full Block Floor)** | $512 \times 512$ | 32,768 B | 32,752 B | 128 | 108 B | 0 B (Ignored) | 28,544 B | **28,498 B** | **28,496 B** | 28,450 B |
| **$1024 \times 1024$ (Actual Shortened RS)** | $512 \times 512$ | 32,768 B | 32,752 B | 128 | 108 B | 76 B ($108 - 32$) | 28,620 B | **28,574 B** | **28,572 B** | 28,526 B |

---

### 2.3 Resolution of the 28,498 B vs. 28,574 B Discrepancy

- **The Source of the Difference:** The apparent contradiction between $28,498\text{ B}$ (reported in `PHASE2_SYNTHESIS_REPORT.md` §4.1) and $28,574\text{ B}$ (reported in `test_carrier_capacity.py` and `E04_hr_audit_remediation.md`) is **not a code defect or empirical inconsistency**. It is a difference in assumptions regarding Reed–Solomon block remainder handling:
  1. **Conservative Closed-Form Floor ($28,498\text{ B}$):** Assumes only full 255-byte codewords are permitted ($128 \times 223 - 46 = 28,498\text{ B}$), discarding the 108-byte remainder.
  2. **Shortened RS Encoder Capability ($28,574\text{ B}$):** Reflects the actual behavior of [`src/security/ecc/encoder.py`](file:///E:/Paladin_media/src/security/ecc/encoder.py#L186-L191) and [`src/carrier/capacity.py`](file:///E:/Paladin_media/src/carrier/capacity.py#L104-L117). The underlying `reedsolo.RSCodec` naturally supports shortened codewords for arbitrary payload lengths. The 108-byte remainder accommodates 76 encrypted data bytes and 32 parity bytes ($28,498 + 76 = 28,574\text{ B}$).
- **Implementation Enforcement:** Both modes are explicitly parameterized in [`src/carrier/capacity.py`](file:///E:/Paladin_media/src/carrier/capacity.py#L83-L139) via `include_shortened_blocks: bool = True`.
- **Unit Test Verification:** Both values are verified in [`tests/unit/test_carrier_capacity.py`](file:///E:/Paladin_media/tests/unit/test_carrier_capacity.py#L27-L49):
  - Line 37: `assert usable_1024_zero == 28498` (with `include_shortened_blocks=False`).
  - Line 48: `assert usable_1024_shortened_zero == 28574` (with `include_shortened_blocks=True`).
  - Lines 160–173: Exact boundary test encoding $28,574\text{ B}$ produces precisely $32,752\text{ B}$ protected payload (passes), while $28,575\text{ B}$ produces $32,753\text{ B}$ (raises `PayloadCapacityExceededError`).
- **Conclusion:** $28,574\text{ B}$ is the definitive usable plaintext capacity ceiling of the actual runtime implementation for $1024 \times 1024$ carriers ($M=0$). $28,498\text{ B}$ is the theoretical full-block floor.

---

## 3. Dataset Provenance and Spatial-Scaling Evidence Reconciliation

The repository contains three distinct image datasets, each serving a separate role in Phase 2 validation:

```
E:\Paladin_media\dataset\
├── MVP 0 Dataset\
│   └── 60 PNG images (0001x2.png to 0060x2.png), exactly 512x512 RGB
│       └── Role: Fixed reference carriers for baseline pipeline tests (E03, E05–E11)
├── DIV2K\
│   ├── DIV2K_train_LR_bicubic_X2\DIV2K_train_LR_bicubic\X2\
│   │   └── 800 PNG images, 558 <= W <= 1020, 324 <= H <= 1020
│   │       └── Role: Initial texture diversity evaluation (E04 LR)
│   └── DIV2K_train_HR\DIV2K_train_HR\
│       └── 800 PNG images, native high resolution (up to 2040x2040)
│           └── Role: High-resolution carrier validation (E04-HR, E12, E13, E14, Phase 3 training)
```

---

### 3.1 Dataset Inventory & Capability Matrix

| Dataset Identifier | Physical Filesystem Path | Image Count | Min / Max Dimensions | Native $512\times512$ Crops | Native $1024\times1024$ Crops | Native $2048\times2048$ Crops | Provenance & Associated Experiments |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **MVP 0 Fixed** | `dataset/MVP 0 Dataset/` | 60 | Exactly $512 \times 512$ | 60 (100%) | 0 (0%) | 0 (0%) | Synthetic/curated baseline carriers. Supports E03, E05, E06, E07, E08, E09, E10, E11. |
| **DIV2K LR (X2)** | `dataset/DIV2K/DIV2K_train_LR_bicubic_X2/DIV2K_train_LR_bicubic/X2/` | 800 | $\min(W,H) \in [324, 1020]$<br>$\max(W,H) \in [558, 1020]$ | 785 (98.1%) | **0 (0.0%)** | **0 (0.0%)** | Bicubic downscaled DIV2K. Maximum native dimension across all 800 images is $1020\text{ px}$. Used in original E04. |
| **DIV2K HR** | `dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/` | 800 | $1020 \le \min(W,H) \le 2040$<br>$1356 \le \max(W,H) \le 2040$ | 800 (100%) | **785 (98.1%)** | **0 (0.0%)** | Native 2K high-resolution DIV2K dataset. 7 images are $2040 \times 2040$. Used in E04-HR, E12, E13, E14, and Phase 3 split. |

---

### 3.2 Resolution of Apparent Scaling Ambiguities

1. **Why was $1024 \times 1024$ skipped in original E04?**
   In the original preflight audit (2026-10-09, [`docs/experiments/PREFLIGHT_INSPECTION_REPORT.md`](file:///E:/Paladin_media/docs/experiments/PREFLIGHT_INSPECTION_REPORT.md)), only `DIV2K_train_LR_bicubic_X2` was present. Because the maximum dimension was 1020 px, anti-upscaling rules strictly prohibited synthetic resizing to 1024 px. Target resolutions $\ge 1024$ were properly marked as **SKIPPED / UNAVAILABLE**.
2. **Why is $1024 \times 1024$ native in E04-HR?**
   Subsequent integration of `DIV2K_train_HR` provided 800 uncompressed native images. Inspection confirmed that 785 images have both width and height $\ge 1024\text{ px}$, allowing native unscaled center crops.
3. **Status of $2048 \times 2048$:**
   Zero images in `DIV2K_train_HR` reach 2048 pixels (maximum dimension is 2040 px). Therefore, $2048 \times 2048$ remains legitimately unavailable for native crops and is marked **SKIPPED** in all empirical matrices.
4. **Partition Manifest for Phase 3:**
   The canonical split manifest [`docs/project/div2k_hr_split.json`](file:///E:/Paladin_media/docs/project/div2k_hr_split.json) (Composite SHA-256: `d0f5f4ee...`) strictly partitions the 800 HR images into:
   - Training: Images `0001.png`–`0600.png` (600 images).
   - Validation: Images `0601.png`–`0700.png` (100 images).
   - Held-Out Test: Images `0701.png`–`0800.png` (100 images).

---

## 4. E14 Phase 2 Exit Evidence Verification

Experiment E14 represents the exit gate for Phase 2 classical baseline characterization. Consistency was verified across [`docs/experiments/e14_results.json`](file:///E:/Paladin_media/docs/experiments/e14_results.json), [`scripts/experiments/run_e14.py`](file:///E:/Paladin_media/scripts/experiments/run_e14.py), [`docs/experiments/E14_lossy_channel_degradation.md`](file:///E:/Paladin_media/docs/experiments/E14_lossy_channel_degradation.md), and Phase 3 specifications.

### 4.1 Empirical Findings Summary (220 Executed Runs)

```
E14 Channel Distribution (220 Total Runs across 20 Carrier-Payload Configs):
├── Lossless PNG Control: 20 runs
│   └── 20/20 Verbatim Recovery (100.0%), BER = 0.000000, Header Valid = 100%
├── Lossy JPEG Recompression: 140 runs (20 runs per Q in {100, 95, 90, 85, 80, 75, 70})
│   └── 0/140 Recovery (0.0%), Raw Parity BER = 47.59% - 49.94%, Header Valid = 0.0%
└── Additive Gaussian Noise: 60 runs (20 runs per sigma in {0.5, 1.0, 2.0}, seed=42)
    └── 0/60 Recovery (0.0%), Raw Parity BER = 49.10% - 50.20%, Header Valid = 0.0%
```

### 4.2 Critical Boundary Distinctions Confirmed

1. **Header Failure vs. Downstream Subsystems:** In 100% of lossy and noisy runs (200/200), extraction halted at the uncorrected 16-byte carrier header (`CarrierHeaderError`). Downstream Reed–Solomon decoding and AES-256-GCM authenticated decryption were **never exercised on extracted, corrupted payload ciphertext**. E14 demonstrates complete pipeline failure, not isolated payload ECC failure.
2. **Empirical Findings vs. Theoretical Modeling:** The empirical result is catastrophic parity inversion ($p \approx 0.50$). The byte symbol error probability $P_s = 1 - (1 - p)^8 \approx 99.61\%$ is documented strictly as an independent-error theoretical model demonstrating why Reed–Solomon ($t/n \approx 6.27\%$) would fail even if the header were preserved.
3. **Fidelity vs. Steganographic Detectability:** PSNR and SSIM measure mathematical fidelity. High PSNR ($\approx 52.8\text{ dB}$ for $\sigma=0.5$) does not establish steganographic undetectability. Statistical steganalysis (SRM, SPAM) was not evaluated in E14.
4. **Untested Transformations Explicitly Isolated:** Spatial downscaling/upscaling, cropping, rotation, optical capture, social-media pipelines, and neural robustness remain documented as **NOT TESTED IN E14**.

---

## 5. Phase 2 Implementation Readiness Matrix

The existing codebase was evaluated to verify interface stability, test coverage, and readiness for Phase 3 integration:

| Component / Subsystem | Source Implementation | Associated Test Files | Operational Status | Integration Readiness Assessment |
|:---|:---|:---|:---:|:---|
| **Carrier DWT / IDWT** | [`src/carrier/dwt.py`](file:///E:/Paladin_media/src/carrier/dwt.py) | `tests/unit/test_carrier_dwt.py` | **STABLE & VERIFIED** | **READY.** Perfectly symmetric forward/inverse transforms. Natively handles arbitrary even dimensions ($512, 1024$). Channel-first layout matches PyTorch. |
| **Capacity Hard Gate** | [`src/carrier/capacity.py`](file:///E:/Paladin_media/src/carrier/capacity.py) | `tests/unit/test_carrier_capacity.py` | **STABLE & VERIFIED** | **READY.** Exact byte accounting; strictly enforces fail-closed `PayloadCapacityExceededError`. |
| **Carrier Header Codec** | [`src/carrier/header.py`](file:///E:/Paladin_media/src/carrier/header.py) | `tests/unit/test_carrier_header.py` | **STABLE (Lossless Only)** | **REQUIRES HARDENING.** 16-byte framing is uncorrected in parity embedding. Must add robust coding (repetition/BCH) for Phase 3 JPEG mode. |
| **Deterministic Mask** | [`src/carrier/mask.py`](file:///E:/Paladin_media/src/carrier/mask.py) | `tests/unit/test_carrier_mask.py` | **STABLE & VERIFIED** | **READY.** SHA-256 PRNG yields non-circular, collision-free placement. Mask logic applies directly to neural coefficient selection. |
| **Multi-Carrier Framing (MCPP)** | `scripts/experiments/run_e09.py` | *None in `tests/`* | **EXPERIMENTAL ONLY** | **NOT INTEGRATED IN `src/`.** MCPP logic exists only in an experiment runner script. Must be factored into `src/carrier/mcpp.py` if needed for Phase 3. |
| **Cryptographic Pipeline** | `src/security/crypto/`, `src/security/ecc/`, `src/core/serialization/` | `tests/unit/test_crypto.py`, `test_ecc.py`, `test_serialization.py` | **STABLE & VERIFIED** | **READY.** AES-256-GCM and Reed–Solomon contracts are frozen and fail-closed. Zero plaintext leakage verified under all failure modes. |
| **Lossless End-to-End Pipeline** | [`src/core/media_pipeline.py`](file:///E:/Paladin_media/src/core/media_pipeline.py) | `tests/integration/test_media_pipeline.py`, `test_fail_closed.py` | **STABLE & VERIFIED** | **READY.** Full roundtrip verified ($100\%$ exact byte match, PSNR $> 75\text{ dB}$, SSIM $= 1.0$). 103/103 tests passing. |

---

## 6. ACS-Sub-INN v3 Integration Prerequisites & Unresolved Decisions

Comparison between the frozen ACS-Sub-INN v3 specification (Architecture §15, System & Research Draft §21) and the current codebase identified the following concrete prerequisites:

### 6.1 Architectural Contracts Frozen in Specification
1. **Unidirectional Cryptographic Sequence (INV-01):** The pipeline `Serialize -> Encrypt (AES-256-GCM) -> Reed-Solomon RS(255, 223)` remains inviolable. The neural network operates strictly on the protected bitstream $P \in \{-1, +1\}$ and never encounters plaintext.
2. **No Secret-Side DWT:** Secret images are serialized into opaque binary streams, never transformed via DWT.
3. **No Discarded Latent-Z:** Full bidirectional invertibility without unrecoverable latent variables.
4. **Transform-Domain Embedding:** Embedding operates strictly on Haar wavelet coefficients.

### 6.2 Precise Mathematical Formulation for ACS-Sub-INN
Per Architecture §15, the forward embedding operates on high-frequency detail subbands:
$$\text{HH} = [\text{HH}_{\text{keep}}, \text{HH}_{\text{drop}}]$$
$$\text{context} = \text{ContextPredictorCNN}(\text{HH}_{\text{keep}})$$
$$\gamma = \text{Softplus}(\text{AdaptiveGainCNN}(\text{HH}_{\text{keep}})) + \gamma_{\text{min}}$$
$$\text{HH}_{\text{drop}}' = (1 - M) \odot \text{HH}_{\text{drop}} + M \odot [\text{context} + \gamma \odot P]$$

On the receiver side, recovery computes:
$$\text{context}_{\text{rec}} = \text{ContextPredictorCNN}(\text{HH}_{\text{keep}})$$
$$\Delta_{\text{rec}} = \text{HH}_{\text{drop}}' - \text{context}_{\text{rec}}$$
$$\hat{P} = \text{sign}(\Delta_{\text{rec}}) \implies \hat{b}_i = (\hat{P}_i > 0)$$

---

### 6.3 Unresolved Decisions Required Before Implementation

The following technical decisions are not frozen in existing documents and must be formally decided before implementing neural code:

1. **Subband Channel Partition Specification:**
   - *Issue:* Architecture §15 specifies $\text{HH} = [\text{HH}_{\text{keep}}, \text{HH}_{\text{drop}}]$, but does not bind which channels belong to $\text{keep}$ vs. $\text{drop}$.
   - *Options:*
     - Option A (Baseline Matched): $\text{HH}_{\text{keep}} = [\text{Green}, \text{Blue}]$ ($C_{\text{keep}}=2$), $\text{HH}_{\text{drop}} = [\text{Red}]$ ($C_{\text{drop}}=1$). Matches Phase 2 Red-HH allocation.
     - Option B (3-Channel Equal Split): Context derived from spatial half-grids across all 3 channels.
     - Option C (Cross-Subband Context): Context derived from $\text{LH} + \text{HL}$, while all 3 channels of $\text{HH}$ are modulated.
   - *Impact:* Dictates tensor input/output channels for `ContextPredictorCNN` and `AdaptiveGainCNN`.
2. **Carrier Header Protection Under Lossy Channels:**
   - *Issue:* E14 proved that uncorrected integer-parity header framing fails $100\%$ under JPEG. If Phase 3 embeds the header in parity, the pipeline will fail before the neural network can be evaluated.
   - *Options:*
     - Option A: High-redundancy repetition code ($15\times$) or Golay $(24, 12)$ code in the high-frequency subband.
     - Option B: Low-frequency embedding (modulating LL/LH with micro-gain $\epsilon$).
     - Option C: Implicit headerless framing (fixed session counter, payload length derived from container dimensions).
   - *Impact:* Dictates carrier synchronization under JPEG $Q=80$.
3. **Neural Network Topologies & Parameter Budgets:**
   - *Issue:* System & Research Draft §21 intentionally left model depth and filter counts unfrozen.
   - *Decisions Needed:*
     - Number of convolutional residual blocks (e.g., 4 or 6 blocks).
     - Channel width (e.g., 64 feature channels).
     - Kernel sizes ($3 \times 3$ vs. $5 \times 5$).
     - Gain floor parameter $\gamma_{\text{min}}$ (e.g., $1.0$ or $2.0$).
4. **Quantization & Training Loss Formulations:**
   - Straight-Through Estimator (STE) implementation for 8-bit rounding: $\text{round}(x) + x - \text{detach}(x)$.
   - Differentiable JPEG simulation layer for training (e.g., Shin & Song or Fourier approximation).
   - Loss weighting parameters: $\mathcal{L} = \lambda_{\text{stego}} \text{MSE} + \lambda_{\text{payload}} \text{BCE} + \lambda_{\text{hf}} \mathcal{L}_{\text{hf}}$.

---

## 7. Ordered List of Blockers Prior to Implementation

Before beginning Phase 3 implementation, the engineering team must resolve the following four items in order:

1. **[BLOCKER 1 - P0] Specify Carrier Header Robustness Strategy:** Formulate and document an error-tolerant framing mechanism for the 16-byte carrier header that survives JPEG compression at $Q=80$.
2. **[BLOCKER 2 - P0] Bind the Subband Channel Split:** Officially freeze whether $\text{HH}_{\text{keep}}$ is Green+Blue (Option A) or Cross-Subband (Option C) to fix model input/output tensor shapes.
3. **[BLOCKER 3 - P1] Author the PyTorch Model Topology Specification:** Document the exact layer-by-layer topology, filter widths, kernel sizes, and activation functions for `ContextPredictorCNN` and `AdaptiveGainCNN`.
4. **[BLOCKER 4 - P1] Update Development Environment Dependencies:** Add `torch >= 2.0.0` and `torchvision` to `pyproject.toml` and verify GPU/CUDA availability on the target machine.

---

## 8. Separation of Verified Facts, Inferences, and Unresolved Items

| Category | Item Description | Supporting Evidence / Source File |
|:---|:---|:---|
| **VERIFIED FACT** | Classical baseline achieves exact byte recovery and $\text{BER}=0.000000$ in lossless PNG. | `tests/integration/test_media_pipeline.py`, `e03_results.json`, `e13_results.json` |
| **VERIFIED FACT** | $512\times512$ usable secret capacity ceiling is strictly $7,090\text{ B}$ ($M=0$) and $7,088\text{ B}$ ($M=2$). | `src/carrier/capacity.py`, `tests/unit/test_carrier_capacity.py` |
| **VERIFIED FACT** | $1024\times1024$ usable capacity ceiling is $28,574\text{ B}$ with shortened RS blocks, $28,498\text{ B}$ full-block floor. | `src/carrier/capacity.py`, `tests/unit/test_carrier_capacity.py` |
| **VERIFIED FACT** | 785 DIV2K HR images natively support $1024\times1024$ crops; 0 images natively support $2048\times2048$. | `e04_hr_audit_results.json`, `docs/project/div2k_hr_split.json` |
| **VERIFIED FACT** | Classical baseline fails 100% under tested JPEG ($Q \le 100$) and noise ($\sigma \ge 0.5$) at the header stage. | `docs/experiments/e14_results.json`, `docs/experiments/E14_lossy_channel_degradation.md` |
| **VERIFIED FACT** | Existing regression test suite passes 103 of 103 tests in 3.22 seconds. | `pytest tests/` execution log |
| **INFERENCE** | A continuous-domain neural INN is necessary to overcome JPEG DCT quantization parity collapse. | Grounded in E14 mathematical failure analysis (§3.2); remains a hypothesis to be proven in Phase 3. |
| **INFERENCE** | Downstream Reed–Solomon decoding would fail under lossy channels even if the header survived. | Supported by theoretical symbol error rate model $P_s = 1 - (1-p)^8 \approx 99.61\% \gg 6.27\%$. |
| **UNRESOLVED** | Exact channel split ($\text{HH}_{\text{keep}}$ vs. $\text{HH}_{\text{drop}}$) for Context Predictor CNN input. | Architecture §15 leaves channel assignment open; requires formal binding (Blocker 2). |
| **UNRESOLVED** | Mechanism to synchronize/extract the carrier header under JPEG $Q=80$. | Unaddressed in Phase 2; uncorrected header is fatal to JPEG survival (Blocker 1). |
| **UNRESOLVED** | Statistical detectability baseline (SRM/SPAM) for classical parity embedding. | Marked as open classical gap (Gap B / Future E15) in Phase 3 Entry-Gate Audit. |
| **UNRESOLVED** | Geometric transformation (resizing, cropping, rotation) robustness. | Marked as open classical gap (Gap A) in Phase 3 Entry-Gate Audit. |
