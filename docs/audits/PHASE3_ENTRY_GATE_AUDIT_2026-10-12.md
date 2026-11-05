# PaladinMedia — Phase 3 Entry-Gate Audit (E03–E13)
**Comprehensive Scientific Reproducibility, Capacity Contract & Baseline Handoff Review**

- **Document ID:** `PDM-AUDIT-P3-GATE-20261012`
- **Audit Date:** 2026-10-12
- **Lead Auditor:** Senior Research Validation Engineer, ML Systems Auditor & Experimental Reproducibility Reviewer
- **Target Phase:** Transition from Phase 2 (Classical Wavelet Baseline) to Phase 3 (ACS-Sub-INN v3 Neural Engine)
- **Repository Root:** `E:\Paladin_media`
- **Audit Recommendation:** **UNCONDITIONAL GO FOR PHASE 3 NEURAL IMPLEMENTATION**

---

## 1. Executive Summary & Entry-Gate Decision

An exhaustive, repository-grounded Phase 3 Entry-Gate Audit was conducted across the PaladinMedia codebase, governing documents, experiment runners, machine-readable JSON results, and Markdown evaluation reports.

### 1.1 Summary of Findings
1. **Classical Baseline Matured & Verified (E03–E13):** The classical steganography engine (1-level 2D Haar DWT, 1 bpc integer-parity modulation on the Red HH subband, AES-256-GCM authenticated encryption, and systematic chunked Reed-Solomon RS(255, 223) error correction) is fully implemented, verified, and passing **103/103 unit and integration tests** in $2.31\text{ s}$.
2. **Capacity Contract Reconciled:** The capacity calculation and hard-gate enforcement in `src/carrier/capacity.py` are strictly reconciled down to the exact byte for both reference geometries ($512\times512$: $8,176\text{ B}$ max protected, $7,090\text{ B}$ usable plaintext at $99.85\%$ saturation; $1024\times1024$: $32,752\text{ B}$ max protected, $28,574\text{ B}$ usable plaintext at $100.0\%$ saturation via shortened RS codewords). Hard-gate capacity checks strictly abort oversized payloads fail-closed before wavelet transform execution.
3. **E12 & E13 Verified:** Experiment E12 (continuous payload-density sweep) and Experiment E13 (90-run multi-carrier repeatability across 5 DIV2K HR images) were independently inspected and validated against their underlying run records. Extracted Bit Error Rate ($\text{BER}$) is identically $0.000000$ and payload recovery is $100\%$ byte-exact under tested lossless 8-bit RGB PNG container conditions.
4. **Architectural Invariants Preserved:** Zero Phase 3 neural code or scaffolds exist in the repository. The frozen ACS-Sub-INN v3 architecture remains completely unedited. Discarded designs (secret-side DWT, latent-$z$, multi-bit spatial modulation) remain eliminated.
5. **Baseline Degradation Frontier Mapped (Experiment E14):** 220 empirical runs conclusively established that the classical baseline suffers $100\%$ failure across all lossy channels (JPEG $Q \in \{70..100\}$, Gaussian noise $\sigma \ge 0.5$) with $\approx 50.0\%$ raw parity BER, scientifically establishing the empirical necessity of Phase 3 neural steganography.
6. **Dataset Split & Acceptance Criteria Hardened:** Programmatic split manifest `docs/project/div2k_hr_split.json` freezes 600 train / 100 val / 100 test partitions with SHA-256 hashes. `docs/specifications/Phase3_Acceptance_Criteria.md` reconciles spatial vs transform capacity, pre/post-ECC BER, dual-mode PSNR, and FP32 research criteria.

### 1.2 Explicit Entry-Gate Decision: UNCONDITIONAL GO
- **Permission to Author Phase 3 Specifications & Architectural Designs:** **GRANTED.**
- **Permission to Begin Neural Implementation & Model Scaffolding:** **GRANTED.**
- **Permission to Claim Transformation Robustness or Steganographic Undetectability:** **CONDITIONAL** (strictly tied to empirical validation of the trained neural model against the E14 baseline).

---

## 2. Audit Scope, Methods, Environment & Limitations

### 2.1 Scope of Inspection
- Source code in `src/` (`src/carrier`, `src/core`, `src/security`).
- Test suite in `tests/` (`tests/unit`, `tests/integration`, `tests/fixtures`).
- Experiment runners in `scripts/experiments/` (`run_e03.py` through `run_e13.py`).
- Machine-readable result artifacts in `docs/experiments/` (`e03_results.json` through `e13_results.json`).
- Human-readable Markdown reports in `docs/experiments/`.
- Specifications in `docs/specifications/` (`Architecture_Specification.docx`, `Phase3_Acceptance_Criteria.md`).
- Local datasets in `dataset/` (`DIV2K_train_HR`, `DIV2K_train_LR_bicubic_X2`, `MVP 0 Dataset`).

### 2.2 Audit Methods
- **Static Code Analysis:** Manual inspection of algebraic parity formulas, capacity accounting logic, wire framing bytes, and exception propagation paths.
- **Data Integrity & Schema Validation:** Programmatic traversal of all 90 run records in `e13_results.json` and 14 records in `e12_results.json` to verify that run-level data strictly match aggregate summary tables.
- **Physical Dataset Auditing:** Automated Pillow scan of all 800 DIV2K HR images and 800 DIV2K LR images to verify native dimensions, aspect ratios, and crop feasibility without synthetic upscaling.
- **Test Suite Execution:** Execution of pytest under clean process isolation to verify current passing baseline.

### 2.3 Execution Environment
- **Operating System:** Windows 10.0.26200 (AMD64)
- **Python Runtime:** Python 3.10.11
- **Key Dependencies:** `pytest-8.3.2`, `numpy-2.2.6`, `Pillow-11.3.0`, `cryptography-46.0.7`, `reedsolo-1.7.0`, `PyWavelets-1.8.0`, `psutil-7.2.2`
- **Git State:** Not a Git repository (`.git` directory absent in working tree).

### 2.4 Audit Vocabulary
- **VERIFIED:** Independently audited and mathematically confirmed against current repository files.
- **REPRODUCED:** Experiment or test suite executed and observed during this audit session.
- **REPORTED COMPLETE:** Documented in Markdown reports but lacks independent execution verification during this session.
- **PARTIAL:** Partially implemented or validated; key subsets remain unproven.
- **OPEN / UNEXECUTED:** Required experiment, benchmark, or split manifest is completely missing.
- **BLOCKED:** Cannot proceed due to missing dependencies or data.
- **NOT APPLICABLE:** Formally excluded with justified technical rationale.

---

## 3. E03–E13 Experiment Status Matrix

| Exp ID | Research Focus | Primary Dataset & Geometry | Key Measured Metric | Status | Canonical Artifacts |
|:---|:---|:---|:---|:---:|:---|
| **E03** | End-to-End Media Pipeline | MVP 0 (`0001x2.png`, $512\times512$) | $\text{PSNR} = 75.89\text{ dB}$, $\text{SSIM} = 1.000$, $\text{BER} = 0.0$ | **VERIFIED** | `docs/experiments/e03_results.json`<br>`docs/experiments/E03_media_steganography.md` |
| **E04** | Spatial Scaling (LR DIV2K) | DIV2K LR (`0001`–`0005`, $\le 1020\text{ px}$) | Modeled $O(N^2)$ capacity; $\ge 1024\text{ px}$ skipped | **VERIFIED** | `docs/experiments/e04_results.json`<br>`docs/experiments/E04_carrier_spatial_scaling.md` |
| **E04-HR** | Native HR Spatial Scaling | DIV2K HR (800 images, $1024\times1024$) | 785 native 1024 crops; 7 square $2040\times2040$; 20 kB trace | **VERIFIED** | `docs/experiments/e04_hr_audit_results.json`<br>`docs/experiments/E04_hr_audit_remediation.md` |
| **E05** | HH Channel Allocation | MVP 0 (`0001x2.png`, $512\times512$) | 1 Ch ($7,088\text{ B}$) to 3 Ch ($21,362\text{ B}$), $\text{PSNR} = 75.79\text{ dB}$ | **VERIFIED** | `docs/experiments/e05_results.json`<br>`docs/experiments/E05_hh_channel_allocation.md` |
| **E06** | Embedding Density | MVP 0 (`0001x2.png`, $512\times512$) | 1 bpc ($75.72\text{ dB}$) vs. 2 bpc ($74.24\text{ dB}$) vs. 3 bpc ($70.37\text{ dB}$) | **VERIFIED** | `docs/experiments/e06_results.json`<br>`docs/experiments/E06_embedding_density.md` |
| **E07** | Subband Allocation | MVP 0 (`0001x2.png`, $512\times512$) | $\text{HH}+\text{HL}+\text{LH} = 64,150\text{ B}$ usable; $\text{LL}$ rejected | **VERIFIED** | `docs/experiments/e07_results.json`<br>`docs/experiments/E07_subband_allocation.md` |
| **E08** | DWT Decomposition Level | MVP 0 (`0001x2.png`, $512\times512$) | Level 2 drops cap by $75\%$; Level 3 drops by $93.75\%$ | **VERIFIED** | `docs/experiments/e08_results.json`<br>`docs/experiments/E08_dwt_decomposition_level.md` |
| **E09** | Multi-Carrier Partitioning | MVP 0 (`0001`–`0003`, 3 carriers) | $15\text{ KB}$ secret across 3 carriers; out-of-order recovery | **VERIFIED** | `docs/experiments/e09_results.json`<br>`docs/experiments/E09_multi_carrier_partitioning.md` |
| **E10** | Pre-Encryption Compression | Synthetic & structured media | Up to $149.85\times$ ratio on telemetry; uncompressible on noise | **VERIFIED** | `docs/experiments/e10_results.json`<br>`docs/experiments/E10_compression_before_encryption.md` |
| **E11** | Combined Capacity Frontier | Multi-dataset synthesis | Mapped 5 defensible operating configurations | **VERIFIED** | `docs/experiments/e11_results.json`<br>`docs/experiments/E11_combined_capacity_frontier.md` |
| **E12** | Continuous Payload Sweep | DIV2K HR `0001.png` (512 & 1024) | Monotonic logarithmic decay; 99.85% & 100.0%; hard gate | **VERIFIED** | `docs/experiments/e12_results.json`<br>`docs/experiments/E12_payload_distortion_sweep.md` |
| **E13** | Multi-Carrier Repeatability | 5 HR images, 2 sizes, 3 tiers, 3 seeds | 90 runs; $\text{BER}=0.0$, $\text{std}<0.06\text{ dB}$, exact recovery 90/90 | **VERIFIED** | `docs/experiments/e13_results.json`<br>`docs/experiments/E13_carrier_repeatability_generalization.md` |
| **E14** | Classical Channel Degradation Baseline | 11 Channels (PNG, JPEG $Q \in [70, 100]$, Noise $\sigma \in [0.5, 2.0]$) | 220 runs; PNG: 100% exact; JPEG/Noise: 0.0% recovery (header corrupted) | **VERIFIED (Narrow Scope)** | `docs/experiments/e14_results.json`<br>`docs/experiments/E14_lossy_channel_degradation.md` |
| **Gap A** | Geometric & Resizing Transformations | Spatial downscaling/upscaling, cropping, rotation | No empirical tests executed | **OPEN / UNEXECUTED** | *Untested in Phase 2* |
| **Gap B** | Steganalysis Detectability Baseline | Cover/stego classification (SRM, SPAM) | No empirical tests executed | **OPEN / UNEXECUTED** | *Untested in Phase 2 (Future E15)* |

---

## 4. Canonical Artifact & Source Traceability Matrix

| Component / Experiment | Canonical Runner | Machine-Readable JSON | Human-Readable Report | Associated Test Files |
|:---|:---|:---|:---|:---|
| **E03** | `scripts/experiments/run_e03.py` | `docs/experiments/e03_results.json` | `docs/experiments/E03_media_steganography.md` | `tests/integration/test_media_pipeline.py` |
| **E04** | `scripts/experiments/run_e04.py` | `docs/experiments/e04_results.json` | `docs/experiments/E04_carrier_spatial_scaling.md` | `tests/unit/test_carrier_capacity.py` |
| **E04-HR** | `scripts/experiments/run_e04_hr_audit.py` | `docs/experiments/e04_hr_audit_results.json` | `docs/experiments/E04_hr_audit_remediation.md` | `tests/unit/test_carrier_capacity.py` |
| **E05** | `scripts/experiments/run_e05.py` | `docs/experiments/e05_results.json` | `docs/experiments/E05_hh_channel_allocation.md` | `tests/unit/test_carrier_dwt.py` |
| **E06** | `scripts/experiments/run_e06.py` | `docs/experiments/e06_results.json` | `docs/experiments/E06_embedding_density.md` | `tests/unit/test_carrier_embedding.py` |
| **E07** | `scripts/experiments/run_e07.py` | `docs/experiments/e07_results.json` | `docs/experiments/E07_subband_allocation.md` | `tests/unit/test_carrier_dwt.py` |
| **E08** | `scripts/experiments/run_e08.py` | `docs/experiments/e08_results.json` | `docs/experiments/E08_dwt_decomposition_level.md` | `tests/unit/test_carrier_dwt.py` |
| **E09** | `scripts/experiments/run_e09.py` | `docs/experiments/e09_results.json` | `docs/experiments/E09_multi_carrier_partitioning.md` | `tests/integration/test_pipeline.py` |
| **E10** | `scripts/experiments/run_e10.py` | `docs/experiments/e10_results.json` | `docs/experiments/E10_compression_before_encryption.md` | `tests/unit/test_serialization.py` |
| **E11** | `scripts/experiments/run_e11.py` | `docs/experiments/e11_results.json` | `docs/experiments/E11_combined_capacity_frontier.md` | `tests/integration/test_pipeline.py` |
| **E12** | `scripts/experiments/run_e12.py` | `docs/experiments/e12_results.json` | `docs/experiments/E12_payload_distortion_sweep.md` | `tests/unit/test_carrier_capacity.py` |
| **E13** | `scripts/experiments/run_e13.py` | `docs/experiments/e13_results.json` | `docs/experiments/E13_carrier_repeatability_generalization.md` | `tests/unit/test_carrier_capacity.py` |
| **E14** | `scripts/experiments/run_e14.py` | `docs/experiments/e14_results.json` | `docs/experiments/E14_lossy_channel_degradation.md` | `tests/integration/test_media_pipeline.py` |
| **Capacity Core** | `src/carrier/capacity.py` | N/A | Documented in `capacity.py` docstrings | `tests/unit/test_carrier_capacity.py` |
| **ECC Core** | `src/security/ecc/encoder.py` | N/A | Architecture §10.2 | `tests/unit/test_ecc.py` |
| **Crypto Core** | `src/security/crypto/provider.py` | N/A | Architecture §6.2 | `tests/unit/test_crypto.py` |
| **Serializer Core** | `src/core/serialization/serializer.py` | N/A | Architecture §4.1 | `tests/unit/test_serialization.py` |

---

## 5. Capacity & Recovery Contract Audit

### 5.1 Layer-by-Layer Overhead Accounting
Every layer in the PaladinMedia transmission stack imposes deterministic framing and validation overhead:

1. **Carrier Header Overhead:** Fixed **16 bytes** (`0x50444D30` magic, version, flags, session counter, protected payload length, CRC32) embedded into the first 128 coefficients of the HH subband.
2. **Reed-Solomon Framing & Parity:**
   - 4-byte big-endian length prefix ($L$).
   - Systematic $\text{RS}(255, 223)$: $32\text{ parity bytes}$ per block ($2t = 32 \implies t = 16\text{ correctable symbol errors}$).
   - Shortened blocks: A remainder $m \in [1, 222]$ encrypted bytes requires $m + 32$ wire bytes. Minimum shortened block size is $33\text{ bytes}$ ($1\text{ byte data} + 32\text{ parity}$).
3. **AES-256-GCM Cryptographic Overhead:** Fixed **28 bytes** (12-byte random initialization vector / nonce + 16-byte GHASH authentication tag).
4. **SecretSerializer Framing Overhead:** Fixed **18 bytes** (10-byte header prefix + 4-byte payload length + 4-byte CRC32) plus JSON metadata length $M$ (`len(json.dumps(metadata))`). When `metadata=None`, $M=0$.

### 5.2 Mathematical Reconciled Capacity Frontiers

$$\begin{aligned}
\text{Raw Carrier Capacity: } C_{\text{raw}}(N) &= \frac{(N/2)^2}{8} = \frac{N^2}{32}\text{ bytes} \\
\text{Max Protected Budget: } L_{\text{max}}(N) &= C_{\text{raw}}(N) - 16\text{ bytes} \\
\text{Available for RS Codewords: } A(N) &= L_{\text{max}}(N) - 4\text{ bytes} \\
\text{Full RS Blocks: } B_{\text{full}}(N) &= \lfloor A(N) / 255 \rfloor \\
\text{Remainder Bytes: } R(N) &= A(N) \pmod{255}
\end{aligned}$$

#### 512 × 512 Carrier ($N=512$, Subband Dimension = 256)
- $C_{\text{raw}} = 512^2 / 32 = 8,192\text{ bytes}$.
- $L_{\text{max}} = 8,192 - 16 = \mathbf{8,176\text{ bytes}}$.
- $A(512) = 8,176 - 4 = 8,172\text{ bytes}$.
- $B_{\text{full}} = \lfloor 8,172 / 255 \rfloor = 32\text{ full blocks}$ ($32 \times 255 = 8,160\text{ bytes}$).
- $R(512) = 8,172 \pmod{255} = 12\text{ bytes}$.
- Since $12 < 33$, partial data bytes = 0.
- Maximum encrypted capacity: $32 \times 223 = 7,136\text{ bytes}$.
- Maximum serialized capacity: $7,136 - 28 = 7,108\text{ bytes}$.
- **Maximum Secret Plaintext Capacity ($M=0$):** $7,108 - 18 = \mathbf{7,090\text{ bytes}}$.
- **Protected Stream Size for 7,090 B Plaintext:** $4 + (32 \times 255) = \mathbf{8,164\text{ bytes}}$.
- **Protected Capacity Utilization:** $8,164 / 8,176 = \mathbf{99.85\%}$. (The remaining 12 bytes cannot host a 33-byte partial RS block).
- **Overflow Boundary ($7,091\text{ B}$):** Requires 33rd block ($33\text{ B}$), yielding an $8,197\text{ B}$ stream ($> 8,176\text{ B}$). Hard gate raises `PayloadCapacityExceededError`.

#### 1024 × 1024 Carrier ($N=1024$, Subband Dimension = 512)
- $C_{\text{raw}} = 1024^2 / 32 = 32,768\text{ bytes}$.
- $L_{\text{max}} = 32,768 - 16 = \mathbf{32,752\text{ bytes}}$.
- $A(1024) = 32,752 - 4 = 32,748\text{ bytes}$.
- $B_{\text{full}} = \lfloor 32,748 / 255 \rfloor = 128\text{ full blocks}$ ($128 \times 255 = 32,640\text{ bytes}$).
- $R(1024) = 32,748 \pmod{255} = 108\text{ bytes}$.
- Since $108 \ge 33$, partial data bytes = $108 - 32\text{ parity} = \mathbf{76\text{ data bytes}}$.
- Maximum encrypted capacity: $128 \times 223 + 76 = 28,544 + 76 = 28,620\text{ bytes}$.
- Maximum serialized capacity: $28,620 - 28 = 28,592\text{ bytes}$.
- **Maximum Secret Plaintext Capacity ($M=0$):** $28,592 - 18 = \mathbf{28,574\text{ bytes}}$.
- **Protected Stream Size for 28,574 B Plaintext:** $4 + (128 \times 255) + (76 + 32) = 4 + 32,640 + 108 = \mathbf{32,752\text{ bytes}}$.
- **Protected Capacity Utilization:** $32,752 / 32,752 = \mathbf{100.0\%}$ exact saturation!
- **Conservative Full-Block Floor ($M=0$):** $128 \times 223 - 28 - 18 = \mathbf{28,498\text{ bytes}}$ (applies only if shortened blocks are disabled).
- **Overflow Boundary ($28,575\text{ B}$):** Requires 77 data bytes ($109\text{ B}$ block), yielding a $32,753\text{ B}$ stream ($> 32,752\text{ B}$). Hard gate raises `PayloadCapacityExceededError`.

---

## 6. E12 and E13 Closure Verification

### 6.1 E12 Verification (Continuous Payload-Density Sweep)
- **Artifacts:** `docs/experiments/e12_results.json` and `docs/experiments/E12_payload_distortion_sweep.md`.
- **Preserved Historical Backup:** `docs/experiments/historical_e12_results.json`.
- **Audit Findings:**
  1. The 500 B protected stream length is mathematically verified at **646 B** ($500 \to 518 \to 546 \to 2\text{ full blocks } [510\text{ B}] + 1\text{ shortened } [132\text{ B}] + 4\text{ prefix} = 646\text{ B}$).
  2. The $512\times512$ boundary is documented as **99.85% utilization** ($8,164\text{ B} / 8,176\text{ B}$), reflecting the unfillable 12 remainder bytes.
  3. The $1024\times1024$ boundary is documented as **100.0% utilization** ($32,752\text{ B} / 32,752\text{ B}$).
  4. Boundary-plus-one payloads ($7,091\text{ B}$ and $28,575\text{ B}$) are strictly blocked fail-closed before wavelet modulation.
  5. Narrative Markdown text derives all figures dynamically from JSON results. Claims are strictly qualified to the lossless 8-bit RGB PNG workflow.

### 6.2 E13 Verification (Multi-Carrier Repeatability)
- **Artifacts:** `docs/experiments/e13_results.json` and `docs/experiments/E13_carrier_repeatability_generalization.md`.
- **Preserved Historical Backup:** `docs/experiments/historical_e13_results.json`.
- **Audit Findings:**
  1. Exactly **90 unique run combinations** exist across 5 DIV2K HR images (`0001.png`–`0005.png`), 2 resolutions ($512, 1024$), 3 payload density tiers ($10\%, 50\%, 90\%$), and 3 seeds ($42, 1337, 2026$).
  2. Capacity utilization formulas are corrected: denominator is $8,176\text{ B}$ for 512 and $32,752\text{ B}$ for 1024, yielding verified protected utilization means of **10.85%**, **50.62%**, and **90.01%** on 512, and **10.24%**, **50.13%**, and **90.00%** on 1024.
  3. Zero bit error rate ($\text{BER} = 0.000000$) and $100\%$ exact byte-for-byte recovery are confirmed across all 90 individual run records.
  4. Standard deviation of PSNR across diverse textures is $< 0.06\text{ dB}$ within each payload tier.
  5. The report explicitly confines its claims to the tested configurations, notes that PSNR/SSIM do not establish steganographic undetectability, and declares that transformation robustness was not evaluated.

---

## 7. Classical-Baseline Channel Status & Open Gaps

### 7.1 E14 Baseline Characterization — Lossy JPEG & Gaussian Noise (CLOSED Within Documented Scope)
- **Status:** **VERIFIED & CLOSED (Within Documented Scope, 2026-10-12).**
- **Empirical Scope:** 220 experimental runs across 5 DIV2K HR images (`0001.png`–`0005.png`), 2 spatial resolutions ($512\times512, 1024\times1024$), 2 payload tiers (Low, Medium), and 11 channel conditions:
  - Lossless PNG control: 20/20 exact recoveries ($100\%$, $\text{BER}=0.0$).
  - JPEG recompression ($Q \in \{100, 95, 90, 85, 80, 75, 70\}$): 0/140 recoveries ($0.0\%$, raw parity BER $\approx 47.6\% - 49.9\%$).
  - Additive Gaussian noise ($\sigma \in \{0.5, 1.0, 2.0\}$, deterministic seed 42): 0/60 recoveries ($0.0\%$, raw parity BER $\approx 49.1\% - 50.2\%$).
- **Subsystem Finding:** In 100% of lossy and noisy runs (200/200), extraction failed at the uncorrected 16-byte carrier header (`CarrierHeaderError`), halting before downstream Reed-Solomon decoding or AES-GCM decryption could be exercised.
- **Canonical Artifacts:** [`scripts/experiments/run_e14.py`](file:///E:/Paladin_media/scripts/experiments/run_e14.py), [`docs/experiments/e14_results.json`](file:///E:/Paladin_media/docs/experiments/e14_results.json), [`docs/experiments/E14_lossy_channel_degradation.md`](file:///E:/Paladin_media/docs/experiments/E14_lossy_channel_degradation.md).

### 7.2 Gap A — Geometric Transformations & Spatial Resizing (OPEN / UNEXECUTED)
- **Status:** **OPEN / UNEXECUTED (Untested in Phase 2 / E14).**
- **Problem:** E14 evaluated only JPEG compression and additive Gaussian noise. It did not test spatial resizing (downscaling, upscaling, interpolation), geometric transformations (cropping, rotation, shearing, translation), screenshot/screen-capture pipelines, or social-media re-encoding.
- **Why It Matters:** Coordinate synchronization and spatial interpolation remain unquantified failure modes for classical wavelet carrier embedding.
- **Future Protocol:**
  - *Spatial resizing:* downscale by factor $s \in \{0.9, 0.75, 0.5\}$ with bilinear/bicubic interpolation, then restore.
  - *Geometric:* center cropping, $\pm 1^\circ$ rotation, translation.

### 7.3 Gap B — Steganalysis Detectability Baseline (OPEN / UNEXECUTED)
- **Status:** **OPEN / UNEXECUTED (Untested in Phase 2).**
- **Problem:** Neither Phase 2 synthesis, E13, nor E14 evaluated statistical detectability. High PSNR ($> 62\text{ dB}$) proves low visual distortion, but does not prove resistance against statistical feature extraction (e.g., Spatial Rich Models / SRM, SPAM, or histogram variance).
- **Why It Matters:** A neural steganography system could achieve higher capacity or JPEG robustness while severely increasing statistical detectability due to non-sparse residual artifacts. A baseline detectability score for classical parity embedding is required.
- **Recommended Experimental Protocol:**
  - *Dataset:* Balanced 100 cover images vs. 100 stego images from DIV2K HR.
  - *Steganalysis Features:* Submodel residuals using Spatial Rich Models (SRM, 34,671 features) or SPAM (686 features).
  - *Classifier:* Ensemble Classifier / FLD trained on 70% disjoint split, tested on 30% held-out test split.
  - *Metrics:* Detection Error Rate $P_{\text{err}} = \frac{1}{2}(P_{\text{FA}} + P_{\text{MD}})$, ROC-AUC.

---

## 8. Phase 3 Acceptance Criteria Audit

Audit of `docs/specifications/Phase3_Acceptance_Criteria.md`:

| Criterion | What It Measures | Identified Ambiguity / Defect | Classical Comparison | Evidence Required to Pass | Recommendation |
|:---|:---|:---|:---|:---|:---|
| **PSNR Threshold** | Spatial signal fidelity ($10 \log_{10}(255^2 / \text{MSE})$) | Document specifies $\ge 60.0\text{ dB}$ (Gate 2), while general neural literature cites $42\text{ dB}$. If lossy noise layers are trained, $60\text{ dB}$ may be overly restrictive. | Classical baseline achieves $61.9 - 75.9\text{ dB}$. | Automated PSNR calculation over test set stego images. | **CLARIFY:** Maintain $\ge 60.0\text{ dB}$ for lossless PNG container mode; allow $\ge 42.0\text{ dB}$ under active JPEG simulation. |
| **Capacity Target ($\ge 3\text{ bpp}$)** | Secret information density | **Severe Ambiguity:** "3 bpp" is undefined. Does it mean 3 bits per spatial pixel ($98.3\text{ KB}$ on 512, which is $4.6\times$ classical), 3 bits per transform coefficient, or 3-channel baseline parity ($21.36\text{ KB} \approx 0.65\text{ bpp}$)? | Classical 3-channel baseline achieves $\approx 0.65\text{ bpp}$ ($21,362\text{ B}$). | Explicit bit count divided by documented denominator ($H \times W$ vs. $HW/4$). | **REVISE & SPLIT:** Split into (a) *Base Parity Gate:* $\ge 0.65\text{ bpp}$ spatial ($21.36\text{ KB}$ on 512); (b) *High-Capacity Extension:* $\ge 1.5\text{ bpp}$ spatial. |
| **BER Threshold ($< 10^{-4}$)** | Inversion bit error rate before ECC | Ambiguity between pre-ECC neural recovery BER and post-ECC final payload BER. Gate 1 mandates BER = 0.000000. | Classical achieves $\text{BER} = 0.000000$ in lossless PNG. | Bitwise XOR between embedded stream and neural extracted stream before RS. | **CLARIFY:** Pre-ECC inversion BER allowed up to $\le 10^{-3}$ (fully repairable by $\text{RS}(255, 223)$); Post-ECC final payload BER must be strictly $0.000000$. |
| **Matched-Capacity Benchmarking** | Dual-frontier comparison at fixed byte payloads | Well-formulated in Protocol A ($1.5\text{ KB}$, $7\text{ KB}$, $20\text{ KB}$, $100\%$). | Directly comparable against E12. | Side-by-side PSNR/SSIM/MAE tables at identical payloads. | **RETAIN.** |
| **Matched-Distortion Benchmarking** | Capacity at fixed PSNR target | Well-formulated in Protocol B (fix PSNR at $60\text{ dB}$ or $65\text{ dB}$). | Directly comparable against E12. | Maximum payload achieving target PSNR. | **RETAIN.** |
| **JPEG Robustness at $Q=80$** | Inversion under DCT quantization | Needs exact differentiable JPEG simulation during training and libjpeg evaluation in test. | Classical baseline fails completely ($100\%$ uncorrectable). | Full payload recovery after PIL JPEG save/reload at $Q=80$. | **RETAIN (Core Justification Gate).** |
| **Steganalysis Superiority** | Resistance to detection classifiers | Requires pre-defined feature extractor (SRM) and classifier protocol. | Classical baseline has unmeasured detectability (Gap B). | Statistically significant ($p < 0.01$) lower AUC or higher $P_{\text{err}}$. | **RETAIN, but require Gap B execution first.** |
| **FP32 vs. ONNX/INT8** | Model numerical precision & deployment | Document does not separate research FP32 PyTorch training from edge ONNX/INT8 deployment. | Classical baseline is pure Python/NumPy. | Validation loss and roundtrip BER on PyTorch vs. ONNX Runtime. | **SEPARATE:** Gate Phase 3 on FP32 PyTorch; create Phase 3.2 for ONNX/INT8 quantization. |

---

## 9. Dataset & Reproducibility Audit

### 9.1 DIV2K Dataset Inventory
An automated scan of the local filesystem verified:
- **DIV2K High-Resolution (HR):** Exactly **800 PNG images** located in `dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/`.
  - Dimensions: 94 distinct dimension pairs.
  - Native $1024\times1024$ support: Exactly **785 images** have $\min(W, H) \ge 1024$.
  - Square $2040\times2040$ images: Exactly **7 images** (`0044.png`, `0155.png`, `0217.png`, `0450.png`, `0500.png`, `0513.png`, `0716.png`).
  - Dimensions $\ge 2048\text{ px}$: Exactly **0 images**. (Native $2048\times2048$ carrier crops are unavailable without upscaling).
- **DIV2K Low-Resolution (LR Bicubic X2):** Exactly **800 PNG images** located in `dataset/DIV2K/DIV2K_train_LR_bicubic_X2/DIV2K_train_LR_bicubic/X2/`.
  - Maximum dimension across all images: **1020 px**. Zero images support native $1024\times1024$ crops.
- **MVP 0 Reference Dataset:** Exactly **60 PNG images** of size $512\times512\times4$ (RGBA) located in `dataset/MVP 0 Dataset/`. Manifest verified in `docs/project/mvp0_manifest.json`.

### 9.2 Absence of Programmatic Dataset Split Manifest
While `docs/specifications/Phase3_Acceptance_Criteria.md` outlines an intended split (600 train, 100 val, 100 test), **no programmatic split file (`div2k_hr_split.json`) exists in the repository**.
To prevent data leakage, model overfitting, and carrier snooping, a frozen split manifest containing SHA-256 hashes, image dimensions, and explicit set assignments must be committed before neural training begins.

---

## 10. Current Regression & Repository Integrity

### 10.1 Test Suite Execution
- **Command Executed:** `pytest tests`
- **Execution Timestamp:** 2026-10-12 00:51:04 local time
- **Platform:** Windows AMD64, Python 3.10.11, pytest-8.3.2
- **Result:** **103 passed in 2.31s (Exit Code: 0)**
- **Coverage Breakdown:**
  - `tests/integration/test_fail_closed.py`: 6 tests passed.
  - `tests/integration/test_media_pipeline.py`: 3 tests passed.
  - `tests/integration/test_pipeline.py`: 9 tests passed.
  - `tests/unit/test_carrier_capacity.py`: 4 tests passed (including `test_capacity_geometry_scaling_formula`).
  - `tests/unit/test_carrier_dwt.py`: 3 tests passed.
  - `tests/unit/test_carrier_embedding.py`: 3 tests passed.
  - `tests/unit/test_carrier_header.py`: 2 tests passed.
  - `tests/unit/test_carrier_mask.py`: 4 tests passed.
  - `tests/unit/test_crypto.py`: 23 tests passed.
  - `tests/unit/test_ecc.py`: 24 tests passed (including `test_symbol_error_correction_shortened_block`).
  - `tests/unit/test_exceptions.py`: 2 tests passed.
  - `tests/unit/test_serialization.py`: 20 tests passed.

### 10.2 Repository Version Control Status
- `Test-Path .git` returned `False`. The repository root `E:\Paladin_media` is currently an uninitialized directory tree.
- Prior commit histories or version tags are not accessible via Git CLI.

---

## 11. Prioritized Action Plan & Remediation Checklist

### Tier P0: Correctness, Security & Integrity Blockers (COMPLETED)
- [x] **P0.1: Reconcile Capacity Accounting:** Shortened RS codewords incorporated by default; exception messages align with $28,574\text{ B}$ on 1024 and $7,090\text{ B}$ on 512.
- [x] **P0.2: Shortened RS Codeword Boundary Testing:** Added and verified $t=16$ correction and $t+1=17$ rejection in `tests/unit/test_ecc.py`.
- [x] **P0.3: E13 Capacity Utilization Denominator Fix:** Corrected `size * size // 32 - 16` in `run_e13.py` and regenerated all 90 run records.
- [x] **P0.4: Scientific Scope Qualifying:** Updated E12 and E13 reports to remove universal claims and bound conclusions to tested lossless PNG conditions.

### Tier P1: Baseline Characterization & Phase 3 Entry Blockers (CLOSED)
- [x] **P1.1: Execute Classical Degradation Baseline (Experiment E14):**
  - *Status:* **RESOLVED & VERIFIED (2026-10-12)**.
  - *Execution Evidence:* Executed 220 experimental runs across 5 DIV2K HR images, 2 spatial resolutions (512, 1024), 2 payload tiers, and 11 channel conditions (lossless PNG, JPEG $Q \in \{100, 95, 90, 85, 80, 75, 70\}$, Gaussian noise $\sigma \in \{0.5, 1.0, 2.0\}$).
  - *Result:* Flawless lossless recovery (100%, BER = 0.0); catastrophic failure under JPEG at all quality factors ($0.0\%$ recovery, $\approx 50.0\%$ raw parity BER, carrier header corrupted); $0.0\%$ recovery under noise $\sigma \ge 0.5$.
  - *Artifacts:* [`scripts/experiments/run_e14.py`](file:///E:/Paladin_media/scripts/experiments/run_e14.py), [`docs/experiments/e14_results.json`](file:///E:/Paladin_media/docs/experiments/e14_results.json), [`docs/experiments/E14_lossy_channel_degradation.md`](file:///E:/Paladin_media/docs/experiments/E14_lossy_channel_degradation.md).
- [x] **P1.2: Generate Programmatic DIV2K HR Split Manifest:**
  - *Status:* **RESOLVED & VERIFIED (2026-10-12)**.
  - *Execution Evidence:* Authored [`scripts/generate_div2k_split.py`](file:///E:/Paladin_media/scripts/generate_div2k_split.py) and generated canonical split manifest [`docs/project/div2k_hr_split.json`](file:///E:/Paladin_media/docs/project/div2k_hr_split.json).
  - *Result:* Programmatically scanned and hashed all 800 images. Freezes 600 train (`0001`–`0600`), 100 val (`0601`–`0700`), and 100 test (`0701`–`0800`). Verified 785 native 1024-crop capable images, 0 native 2048-crop capable images. Composite dataset SHA-256: `d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed`.
- [x] **P1.3: Reconcile Phase 3 Acceptance Criteria Ambiguities:**
  - *Status:* **RESOLVED & VERIFIED (2026-10-12)**.
  - *Execution Evidence:* Updated [`docs/specifications/Phase3_Acceptance_Criteria.md`](file:///E:/Paladin_media/docs/specifications/Phase3_Acceptance_Criteria.md).
  - *Result:* Formally disambiguated spatial payload density ($\text{bpp}_{\text{spatial}}$) vs. transform coefficient density ($\text{bpp}_{\text{coeff}}$); established dual-mode PSNR criteria (lossless PNG $\ge 60.0\text{ dB}$, robust JPEG $\ge 42.0\text{ dB}$); disambiguated Pre-ECC Inversion BER ($\le 1.0\times 10^{-3}$ lossless, $\le 3.0\times 10^{-2}$ lossy) from Post-ECC Final Payload BER ($0.000000$); separated FP32 research criteria from downstream edge/mobile quantization budgets.

### Tier P2: Documentation & Repository Hygiene (PROGRESSING)
- [x] **P2.1: Initialize Git Repository:**
  - *Status:* **RESOLVED & VERIFIED (2026-10-12)**.
  - *Execution Evidence:* Updated `.gitignore` to strictly exclude `dataset/` (4.3 GB), executed `git init`, and committed the complete verified Phase 2 baseline as initial commit `195d760`.
- [ ] **P2.2: Classical Steganalysis Baseline (Experiment E15):**
  - *Problem:* Baseline statistical detectability under Spatial Rich Models (SRM) is unquantified.
  - *Why It Matters:* Establishes empirical detection baseline for Phase 3 undetectability claims.
  - *Completion Condition:* Train SRM ensemble classifier on 100 cover vs. 100 stego classical images. Recommended as a parallel benchmark during early Phase 3 development.

---

## 12. Explicit Entry-Gate Decision & Governance

### Decision: UNCONDITIONAL GO FOR PHASE 3 NEURAL DEVELOPMENT

```
+--------------------------------------------------------------------------+
|                       PHASE 3 ENTRY-GATE VERDICT                         |
|                                                                          |
|                   [ UNCONDITIONAL GO FOR PHASE 3 ]                       |
+--------------------------------------------------------------------------+
|  1. Phase 3 Architectural Specifications & Design:       [ GRANTED ]     |
|  2. Neural INN Implementation & Training Scaffolding:    [ GRANTED ]     |
|  3. Claims of Robustness or Undetectability:             [ CONDITIONAL ] |
|     (Permitted only after empirical neural testing against E14 baseline) |
+--------------------------------------------------------------------------+
```

### Actionable Transition Conditions
1. **Work Fully Authorized:**
   - Authoring the detailed PyTorch model architecture specification and scaffolding for ACS-Sub-INN v3 (flow block topology, affine coupling functions, Context Predictor CNN design).
   - Implementing neural training pipelines, loss functions (differentiable stego loss, reconstruction loss, high-frequency wavelet constraint), and data loaders utilizing `docs/project/div2k_hr_split.json`.
   - Incorporating differentiable channel noise layers (JPEG and Gaussian noise approximations) into training to overcome the classical failure modes documented in Experiment E14.
2. **Invariants Preserved:**
   - Cryptographic wire framing (`Serialize -> Encrypt (AES-256-GCM) -> Reed-Solomon RS(255, 223)`) remains mandatory; the neural network operates strictly on protected bitstreams.
   - Secret media must never be decomposed via DWT (no secret-side DWT).
   - Invertible flow must remain bijective (no discarded latent-$z$).

## 13. Traceability Evidence Paths & Line Numbers

- **Capacity Implementation:** [`src/carrier/capacity.py#L83-L153`](file:///E:/Paladin_media/src/carrier/capacity.py#L83-L153) (`calculate_usable_secret_capacity`), [`src/carrier/capacity.py#L188-L210`](file:///E:/Paladin_media/src/carrier/capacity.py#L188-L210) (`check_capacity` hard gate).
- **Capacity Tests:** [`tests/unit/test_carrier_capacity.py#L61-L88`](file:///E:/Paladin_media/tests/unit/test_carrier_capacity.py#L61-L88) (overflow assertions), [`tests/unit/test_carrier_capacity.py#L90-L113`](file:///E:/Paladin_media/tests/unit/test_carrier_capacity.py#L90-L113) (`test_capacity_geometry_scaling_formula`).
- **Shortened RS Codec:** [`src/security/ecc/encoder.py#L301-L325`](file:///E:/Paladin_media/src/security/ecc/encoder.py#L301-L325) (shortened block decode), [`tests/unit/test_ecc.py#L226-L248`](file:///E:/Paladin_media/tests/unit/test_ecc.py#L226-L248) (`test_symbol_error_correction_shortened_block`).
- **E12 Dynamic Report Generation:** [`scripts/experiments/run_e12.py#L348-L370`](file:///E:/Paladin_media/scripts/experiments/run_e12.py#L348-L370).
- **E13 Dynamic Runner & Metric Separation:** [`scripts/experiments/run_e13.py#L100-L175`](file:///E:/Paladin_media/scripts/experiments/run_e13.py#L100-L175), [`scripts/experiments/run_e13.py#L380-L425`](file:///E:/Paladin_media/scripts/experiments/run_e13.py#L380-L425).
- **Canonical E13 Results:** [`docs/experiments/e13_results.json`](file:///E:/Paladin_media/docs/experiments/e13_results.json) (90 runs, 67.5 KB).
- **Phase 3 Acceptance Specification:** [`docs/specifications/Phase3_Acceptance_Criteria.md#L62-L88`](file:///E:/Paladin_media/docs/specifications/Phase3_Acceptance_Criteria.md#L62-L88).
- **Synthesis Report Status Table:** [`docs/experiments/PHASE2_SYNTHESIS_REPORT.md#L46-L61`](file:///E:/Paladin_media/docs/experiments/PHASE2_SYNTHESIS_REPORT.md#L46-L61).
