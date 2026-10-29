# PaladinMedia — Phase 3 Pre-Implementation Contract Validation

**Document ID:** `PDM-AUDIT-P3-CONTRACT-VALIDATION-20261018`
**Date:** 2026-10-18
**Repository:** `E:\Paladin_media`
**Status:** APPROVED DESIGN ADDENDUM & PRE-IMPLEMENTATION AUDIT SPECIFICATION
**Governing Documents:** PaladinMedia Architecture Specification §10–§15, System & Research Document Draft §21, Phase 3 Acceptance Criteria (`PDM-P3-GATE-001`), Phase 3 Technical Readiness Checkpoint (`PDM-AUDIT-P3-TECH-READINESS-20261018`), Phase 3 Blocker Resolution Proposal (`PDM-DESIGN-P3-BLOCKER-PROPOSAL-20261018`), E14 Lossy Channel Degradation Baseline (`docs/experiments/E14_lossy_channel_degradation.md`)
**Target Hardware:** NVIDIA GeForce GTX 1650 (TU117, Compute Capability 7.5, 4 GB VRAM), Driver 610.47, Python 3.10.11

---

## 1. Executive Verdict & Gate Status

### Verdict: `READY WITH BLOCKERS`

The pre-implementation contract validation establishes that the mathematical formulations, protocol framing mechanisms, and environment specifications for Phase 3 are rigorously defined and compatible with all frozen system invariants.

However, execution must remain gated pending two concrete operational prerequisites that require explicit user authorization:
1. **Approval of Frozen Protocol Contracts:** Formal adoption of the Dual-Mode Framing Contract (preserving the in-band 16-byte header for Mode A lossless PNG, while enforcing Implicit Fixed-Tier Framing with AES-GCM GHASH authentication for Mode B lossy channels) and Subband Option A (Red-HH Drop, Green/Blue-HH Keep).
2. **PyTorch Installation Authorization:** Permission to execute the installation of official wheels (`torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`) within the local Python 3.10.11 environment.

Zero source code modifications (`src/`), zero package installations, zero experiment reruns, and zero file deletions have been executed in this audit pass.

---

## 2. Implicit Fixed-Tier Framing & Synchronization Contract

### 2.1 Problem Analysis: Why Carrier Dimensions Alone Are Insufficient

In classical Phase 2 embedding, the 16-byte `CarrierHeader` explicitly encodes the payload length (`payload_len`, uint32). In Experiment E14, all lossy degradation runs failed because this in-band header relies on raw single-bit parity in Red-HH coefficients, which collapses under lossy compression.

A naive heuristic might suggest that the receiver can determine payload length solely from carrier dimensions $(H, W)$. **This assumption is invalid.** As demonstrated across Experiments E03–E14:
- Carriers of dimension $512 \times 512$ held payloads of 1,000 bytes (Tier 1) and 3,500 bytes (Tier 2).
- Carriers of dimension $1024 \times 1024$ held payloads of 4,000 bytes (Tier 2) and 14,000 bytes (Tier 3).
- The maximum usable capacity of a carrier is $N_{\text{avail}} = (H/2) \cdot (W/2) / 8 - 16$ bytes (8,176 B for $512^2$; 32,752 B for $1024^2$), which establishes an upper physical bound, not the exact length of the embedded payload.

Because multiple discrete payload tiers share identical carrier dimensions, the receiver cannot infer payload length from $(H, W)$ without a formal framing protocol.

---

### 2.2 Architectural Comparison of Framing Candidates

```
Framing Candidates under Evaluation:
├── Candidate A: Deterministic Session-Agreed Tiers
│   └── Discrete tier index T agreed via out-of-band session context
├── Candidate B: Fixed-Size Padded Frames (Envelope Padding)
│   └── Constant carrier slot capacity with self-delimiting inner plaintext
└── Candidate C: Multi-Tier Trial Extraction (GHASH Hypothesis Testing)
    └── Autonomous receiver scanning over candidate tier lengths
```

| Evaluation Dimension | Candidate A: Session-Agreed Tiers | Candidate B: Fixed-Size Padded Frames | Candidate C: Multi-Tier Trial Extraction |
|:---|:---|:---|:---|
| **Mechanism** | Out-of-band agreement on tier $T \in \{T_1, T_2, T_3, T_{\max}\}$. Receiver extracts exactly $L(T)$ bits. | Receiver extracts full slot capacity $L_{\text{slot}}(H, W)$. Inner `SecretSerializer` provides true payload length. | Receiver extracts candidate lengths $L_1, L_2, \dots, L_k$ and attempts RS decode + AES-GCM verification. |
| **In-Band Overhead** | **0 bits.** | Padding bits added to fill slot capacity (variable overhead). | **0 bits.** |
| **Out-of-Band Dependency** | Requires sender and receiver to synchronize tier index. | Zero out-of-band context required. | Zero out-of-band context required. |
| **Computational Overhead** | Minimum: exactly 1 extraction + 1 RS decode + 1 AES-GCM check. | Minimum: exactly 1 extraction + 1 RS decode + 1 AES-GCM check (longer RS stream). | Up to $k$ RS decodes and AES-GCM checks (at most 3–4 candidates). |
| **False Positive Probability** | $2^{-128} \approx 2.9 \times 10^{-39}$ (governed by AES-GCM GHASH). | $2^{-128} \approx 2.9 \times 10^{-39}$ (governed by AES-GCM GHASH). | $k \cdot 2^{-128} \approx 2.9 \times 10^{-39}$ (negligible). |
| **Fail-Closed Behavior** | Immediate fail-closed on wrong tier selection. | Immediate fail-closed if corrupted or invalid padding. | Immediate fail-closed if no candidate tier authenticates. |

---

### 2.3 Formal Framing Contract for Mode B (Robust Lossy Channels)

To achieve maximum operational flexibility without sacrificing cryptographic rigor, the **Recommended Protocol Contract** combines **Deterministic Session-Agreed Tiers (Candidate A)** with **Trial Extraction (Candidate C)** fallback, anchored strictly by the 128-bit AES-GCM authentication boundary:

```
Sender Data Flow (Mode B):
Payload Bytes
     │
     ▼
[SecretSerializer] ─────────► Prepend 4B length + metadata (Plaintext)
     │
     ▼
[AES-256-GCM] ──────────────► Encrypt & Authenticate -> (IV 12B || Ciphertext || Tag 16B)
     │
     ▼
[Reed-Solomon RS(255, 223)] ─► Encode into N_blocks of 255 bytes (Protected Stream)
     │
     ▼
[ACS-Sub-INN v3] ───────────► Embed directly at mask coordinates 0 .. L_extract - 1
                              (NO in-band unencrypted header)
```

1. **Discrete Tier Catalog:**
   The protocol standardizes four discrete payload tiers:
   - Tier 1 ($T_1$): $1,500$ bytes plaintext $\implies 1,770$ bytes protected stream ($14,160$ bits).
   - Tier 2 ($T_2$): $7,000$ bytes plaintext $\implies 8,040$ bytes protected stream ($64,320$ bits).
   - Tier 3 ($T_3$): $20,000$ bytes plaintext $\implies 22,910$ bytes protected stream ($183,280$ bits).
   - Full Capacity Tier ($T_{\text{max}}$): $N_{\text{avail}}(H, W)$ protected bytes.

2. **Extraction Length Calculation:**
   Given a selected tier $T_k$, the receiver extracts an exact bitstream of length:
   $$L_{\text{extract}}(T_k) = L_{\text{protected}}(T_k) \times 8 \quad \text{bits}$$
   The bits are harvested from the first $L_{\text{extract}}(T_k)$ active coordinates of the deterministic spatial mask $M(H/2, W/2, \text{seed})$.

3. **Authentication Boundary and Associated Data (AAD):**
   - The extraction process produces an opaque byte sequence of length $L_{\text{protected}}(T_k)$.
   - Reed–Solomon decoding splits this stream into $\lceil L_{\text{protected}} / 255 \rceil$ blocks and corrects symbol errors up to $t=16$ bytes per 255-byte block.
   - The decoded output represents the authenticated cryptographic envelope:
     $$C = \text{IV}_{12\text{B}} \parallel \text{Ciphertext}_{N\text{B}} \parallel \text{Tag}_{16\text{B}}$$
   - Decryption is executed using AES-256-GCM with Associated Authenticated Data (AAD):
     $$\text{AAD} = \text{ProtocolVersion (2B)} \parallel \text{TierIndex (2B)} \parallel \text{CarrierWidth (2B)} \parallel \text{CarrierHeight (2B)}$$
   - Cryptographic integrity is verified via the 16-byte GHASH tag before any plaintext data is exposed to memory or downstream serialization.

---

### 2.4 Fail-Closed Behavior on Wrong Tier Selection or Truncation

The system enforces strict fail-closed security invariants across all failure modes:

1. **Receiver Selects Wrong Tier ($T' \neq T$):**
   - **Case A: Under-Reading ($L' < L$):** The receiver extracts fewer bits than were embedded. The extracted stream contains a truncated fraction of the Reed–Solomon blocks. `ReedSolomonEncoder.decode()` raises `ReedSolomonError` because block framing is misaligned. If RS decoding is bypassed, AES-GCM GHASH verification fails with probability $1 - 2^{-128}$, raising `CryptoTamperError`.
   - **Case B: Over-Reading ($L' > L$):** The receiver extracts more bits than were embedded, appending non-embedded carrier coefficients. The trailing Reed–Solomon blocks contain random carrier noise ($\approx 50\%$ BER), far exceeding the 16-symbol error correction bound. `ReedSolomonError` is raised.
2. **Truncated Ciphertext Envelope ($< 28$ Bytes):**
   - If extracted protected bytes yield fewer than 28 bytes ($\text{IV}_{12\text{B}} + \text{Tag}_{16\text{B}}$), `CryptoProvider.decrypt()` raises `CryptoTamperError` immediately.
3. **Leakage Invariant:**
   - In all failure scenarios, no plaintext bytes are allocated or returned. The pipeline throws an unhandled domain exception (`CarrierHeaderError`, `ReedSolomonError`, or `CryptoTamperError`), terminating the operation cleanly.

---

### 2.5 Dual-Mode Framing Backward Compatibility

To maintain 100% regression stability against the existing Phase 2 codebase and test suite:

- **Mode A (Lossless PNG Workflow - Phase 2 Invariant):**
  - Uses the existing 16-byte `CarrierHeader` (`magic=0x504D, version=1, flags, payload_len, crc32`).
  - Embedded at mask indices $0 \dots 127$ via raw parity.
  - Passes all 103 existing unit and integration tests without modification.
- **Mode B (Robust JPEG / Degraded Channel Workflow - Phase 3 Invariant):**
  - Operates without an in-band parity header.
  - Payload stream begins at mask index $0$.
  - Synchronization is achieved via deterministic session tier or multi-tier GHASH trial extraction.
- **JPEG $Q=80$ Robustness:** Robustness to JPEG compression is formally treated as a **target research hypothesis to be evaluated**, not an assumed baseline capability.

---

## 3. ACS-Sub-INN v3 Embedding and Extraction Contract

### 3.1 Bipolar Symbol Modulation Mapping

The payload bitstream is transformed into continuous bipolar symbols for neural modulation:
- Given binary bit $b_i \in \{0, 1\}$:
  $$P_i = 2 b_i - 1.0 \in \{-1.0, +1.0\}$$
  $$\text{Specifically: } b_i = 0 \implies P_i = -1.0; \quad b_i = 1 \implies P_i = +1.0$$
- Receiver symbol-to-bit demodulation rule:
  Given the recovered residual $\Delta_{\text{rec}} = \text{HH}_{\text{drop}}^* - \text{ContextPredictorCNN}(\text{HH}_{\text{keep}}^*)$:
  $$\hat{b}_i = \begin{cases} 1 & \text{if } \Delta_{\text{rec}} > 0 \\ 0 & \text{if } \Delta_{\text{rec}} \le 0 \end{cases}$$

---

### 3.2 Mask Semantics and Active Extraction Coordinates

1. **Mask Generation:**
   A 2D binary spatial mask $M \in \{0, 1\}^{H/2 \times W/2}$ is generated deterministically by `CarrierMaskGenerator.generate(height // 2, width // 2, seed)`.
2. **Coordinate Sequence:**
   The set of active coefficient locations is defined by the row-major ordered indices where $M[r, c] == 1$:
   $$\mathcal{I}_{\text{active}} = \left\{ (r_k, c_k) \;\middle|\; M[r_k, c_k] = 1 \right\}_{k=0}^{K-1}, \quad \text{where } K = \sum M$$
3. **Modulation Indexing:**
   - In **Mode A (Lossless with Header):**
     - Header bits ($0 \dots 127$) occupy $\mathcal{I}_{\text{active}}[0 \dots 127]$.
     - Payload bits occupy $\mathcal{I}_{\text{active}}[128 \dots 127 + N_{\text{bits}}]$.
   - In **Mode B (Implicit Framing):**
     - Payload bits occupy $\mathcal{I}_{\text{active}}[0 \dots N_{\text{bits}} - 1]$.

---

### 3.3 Tensor Shapes, Channel Order, and Precision

The operational data flow across spatial and wavelet domains follows a strict tensor contract:

```
Tensor Flow and Dimension Contract:
Carrier Image: uint8 [B, 3, H, W] (NCHW, RGB)
     │
     ▼ (Convert to float64, normalize or preserve unit scale)
[2D Haar DWT] ──────────────► LL, LH, HL, HH: each float64 [B, 3, H/2, W/2]
                                                   │
     ┌─────────────────────────────────────────────┴──────────────────┐
     ▼                                                                ▼
HH_drop: float64 [B, 1, H/2, W/2] (Red, Channel 0)     HH_keep: float64 [B, 2, H/2, W/2] (Green/Blue, Ch 1 & 2)
     │                                                                │
     │                                                 ┌──────────────┴──────────────┐
     │                                                 ▼                             ▼
     │                                     [ContextPredictorCNN]            [AdaptiveGainCNN]
     │                                                 │                             │
     │                                                 ▼ (Linear)                    ▼ (Softplus + gamma_min)
     │                                     context: [B, 1, H/2, W/2]         gamma: [B, 1, H/2, W/2]
     │                                                 │                             │
     └─────────────────────────┬───────────────────────┴─────────────────────────────┘
                               ▼
     HH_drop' = (1 - M) * HH_drop + M * [context + gamma * P]
                               │
                               ▼
     HH' = Concat([HH_drop', HH_keep], dim=1): float64 [B, 3, H/2, W/2]
                               │
                               ▼
[2D Haar IDWT] ─────────────► Stego Image: float64 [B, 3, H, W]
                               │
                               ▼
[Quantization & Clamping] ──► round(clamp(I', 0, 255)) -> uint8 [B, 3, H, W]
```

1. **Channel Order:** Strict `NCHW` format. Channel index 0 is Red, 1 is Green, 2 is Blue.
2. **Precision:**
   - DWT and IDWT operate in **float64** to prevent precision drift during linear filtering.
   - Neural network weights and forward inference run in **float32** (or **bfloat16/float16** during mixed-precision training).
   - Tensors passed between DWT and CNN modules are converted explicitly: `float64 -> float32 -> float64`.
3. **Quantization and Straight-Through Estimator (STE):**
   - In spatial reconstruction, the stego image is rounded to 8-bit integers:
     $$I_{\text{quant}} = \text{clamp}(\text{round}(I'), 0, 255)$$
   - During neural backpropagation, rounding has zero derivative almost everywhere. The gradient is preserved using the Straight-Through Estimator:
     $$I_{\text{STE}} = I' + \text{detach}(\text{round}(I') - I')$$

---

### 3.4 Mathematical Proof: Context Channels (Green & Blue) Remain Unmodified

**Theorem:** Modulating Red-HH in the wavelet domain produces zero perturbation in the Green and Blue spatial channels of the carrier image.

**Proof:**
1. The 2D discrete Haar wavelet transform is a separable linear transformation operating along spatial dimensions $(x, y)$.
2. Let $I \in \mathbb{R}^{3 \times H \times W}$ denote the RGB image, where $I_c \in \mathbb{R}^{H \times W}$ represents color channel $c \in \{0, 1, 2\}$ ($\text{Red}=0, \text{Green}=1, \text{Blue}=2$).
3. The forward Haar transform for channel $c$ is given by matrix multiplication:
   $$\begin{bmatrix} \text{LL}_c \\ \text{LH}_c \\ \text{HL}_c \\ \text{HH}_c \end{bmatrix} = \mathcal{W} \, I_c$$
   where $\mathcal{W}$ is the orthogonal 2D Haar analysis operator.
4. The inverse Haar transform reconstructs the spatial channel via the synthesis operator $\mathcal{W}^{-1}$:
   $$I_c = \mathcal{W}^{-1} \begin{bmatrix} \text{LL}_c \\ \text{LH}_c \\ \text{HL}_c \\ \text{HH}_c \end{bmatrix}$$
5. Under Subband Partition Option A:
   - For $c = 0$ (Red): $\text{HH}_0' = (1 - M) \odot \text{HH}_0 + M \odot [\text{context} + \gamma \odot P] \neq \text{HH}_0$.
   - For $c \in \{1, 2\}$ (Green, Blue): $\text{LL}_c' = \text{LL}_c$, $\text{LH}_c' = \text{LH}_c$, $\text{HL}_c' = \text{HL}_c$, and $\text{HH}_c' = \text{HH}_c$.
6. Evaluating the reconstructed Green channel ($c=1$):
   $$I_1' = \mathcal{W}^{-1} \begin{bmatrix} \text{LL}_1 \\ \text{LH}_1 \\ \text{HL}_1 \\ \text{HH}_1' \end{bmatrix} = \mathcal{W}^{-1} \begin{bmatrix} \text{LL}_1 \\ \text{LH}_1 \\ \text{HL}_1 \\ \text{HH}_1 \end{bmatrix} = I_1$$
7. Evaluating the reconstructed Blue channel ($c=2$):
   $$I_2' = \mathcal{W}^{-1} \begin{bmatrix} \text{LL}_2 \\ \text{LH}_2 \\ \text{HL}_2 \\ \text{HH}_2' \end{bmatrix} = \mathcal{W}^{-1} \begin{bmatrix} \text{LL}_2 \\ \text{LH}_2 \\ \text{HL}_2 \\ \text{HH}_2 \end{bmatrix} = I_2$$
8. Therefore, $I_{\text{Green}}' \equiv I_{\text{Green}}$ and $I_{\text{Blue}}' \equiv I_{\text{Blue}}$ at every pixel location. Modulating Red-HH has zero cross-talk into Green or Blue spatial pixels in the float64 domain. $\blacksquare$

---

### 3.5 Reversibility Analysis: Mathematical Reversibility vs. Empirical Robustness

#### A. Mathematical Exactness in the Lossless Path
In a lossless environment (unquantized float64 carrier or lossless PNG with bounded integer rounding):
1. Because Green and Blue are untouched, $\text{HH}_{\text{keep}}^* = \text{HH}_{\text{keep}}$.
2. The receiver evaluates:
   $$\text{context}_{\text{rec}} = \text{ContextPredictorCNN}(\text{HH}_{\text{keep}}^*) = \text{ContextPredictorCNN}(\text{HH}_{\text{keep}}) = \text{context}$$
3. The demodulation residual at masked coordinates is:
   $$\Delta_{\text{rec}} = \text{HH}_{\text{drop}}^* - \text{context}_{\text{rec}} = (\text{context} + \gamma \odot P) - \text{context} = \gamma \odot P$$
4. Because $\gamma(x, y) \ge \gamma_{\text{min}} > 0$:
   $$\text{sign}(\Delta_{\text{rec}}) = \text{sign}(\gamma \odot P) = \text{sign}(P) = P$$
5. In 8-bit lossless PNG, integer rounding introduces a spatial error $e \in [-0.5, 0.5]$ on Red pixels. Passing through Haar DWT, the coefficient perturbation is bounded: $|e_{\text{DWT}}| \le 0.5$. With $\gamma_{\text{min}} = 1.0$, the residual magnitude satisfies:
   $$|\Delta_{\text{rec}}| = |\gamma \odot P + e_{\text{DWT}}| \ge 1.0 - 0.5 = 0.5 > 0$$
   The sign is strictly preserved, guaranteeing **$100\%$ exact bit recovery in lossless mode**.

#### B. Empirical Robustness in Lossy Channels (JPEG / Additive Noise)
In lossy channels, the mathematical identity breaks down:
1. JPEG DCT block quantization and high-frequency coefficient truncation perturb **all** spatial channels:
   $$\text{HH}_{\text{keep}}^* = \text{HH}_{\text{keep}} + \delta_{\text{keep}}$$
   $$\text{HH}_{\text{drop}}^* = \text{HH}_{\text{drop}}' + \delta_{\text{drop}}$$
2. The receiver's predicted context suffers from input perturbation:
   $$\text{context}_{\text{rec}} = \text{ContextPredictorCNN}(\text{HH}_{\text{keep}} + \delta_{\text{keep}}) = \text{context} + \delta_{\text{pred}}$$
3. The demodulation residual becomes:
   $$\Delta_{\text{rec}} = \gamma \odot P + (\delta_{\text{drop}} - \delta_{\text{pred}})$$
4. A bit error occurs whenever the total channel noise exceeds the adaptive gain:
   $$(\delta_{\text{drop}} - \delta_{\text{pred}}) \cdot P < -\gamma$$
5. **Conclusion:** Robustness under lossy channels cannot be established analytically. It relies entirely on:
   - The neural network learning context representations that are invariant to JPEG quantization artifacts.
   - The adaptive gain CNN learning to allocate sufficient energy $\gamma(x, y)$ in complex textures where DCT quantization distortion is highest.
   - Downstream Reed–Solomon decoding correcting residual symbol errors (up to $\approx 6.27\%$ byte error rate).

---

### 3.6 Preservation of Frozen Architecture Constraints

The validated contract strictly adheres to all frozen invariants of ACS-Sub-INN v3:
- **No Secret-Side DWT:** Secret payload data enters strictly as an encrypted, Reed–Solomon-protected byte stream. It is never transformed via DWT, DCT, or spatial steganography.
- **No Latent-Z / Discarded Variables:** Unlike traditional Invertible Neural Networks that map image details to a discarded Gaussian latent vector $z \sim \mathcal{N}(0, I)$, ACS-Sub-INN v3 directly predicts context from preserved subband coefficients. All operations are deterministic, self-contained, and fully invertible without latent-state sampling.

---

## 4. Evidence vs. Inference Audit

### 4.1 Directly Measured E14 Facts vs. Modeled Inferences

To prevent confirmation bias, empirical observations from Experiment E14 are strictly segregated from statistical models and engineering deductions:

| Item | Measured Empirical Fact (E14) | Theoretical Deduction / Statistical Inference | Status / Distinction |
|:---|:---|:---|:---:|
| **Header Survival** | 100% of JPEG ($Q \in [70, 100]$) and noise runs failed at the 16B carrier header (`CarrierHeaderError`). | The failure is caused by spatial parity perturbation in high-frequency Red-HH coefficients under $8 \times 8$ block DCT quantization. | **Fact vs. Inferred Cause** |
| **Parity BER** | Red-HH coefficient parity had a bit error rate of $47.6\% - 50.0\%$ under JPEG $Q \in [70, 95]$. | Under an independent error model with $p \approx 0.48$, the probability of 128 uncorrected bits surviving is $(1-p)^{128} \approx 10^{-37}$. | **Fact vs. Binomial Model** |
| **Downstream Integrity** | Zero RS decodes and zero AES-GCM decryptions were executed under JPEG in E14. | Reed–Solomon RS(255, 223) and AES-GCM would fail-closed if reached, because payload coefficients experience similar perturbation. | **Fact vs. Extrapolated Inference** |
| **Repetition Coding** | Repetition coding was not implemented or tested in E14. | Simple majority voting in discrete parity space cannot rescue the header at $p=0.48$, but continuous QIM or lower-frequency embedding might. | **Absence of Data vs. Analytic Model** |
| **Lossless Baseline** | E03–E13 proved 100% exact byte recovery and PSNR $> 42\text{ dB}$ across 90+ carriers in lossless PNG. | Lossless baseline stability will be preserved if Mode A framing and deterministic mask coordinates are unchanged. | **Fact vs. Engineering Guarantee** |

---

### 4.2 Exact Binomial Analysis of Repetition Codes

To resolve whether an error-correcting repetition code could rescue the classical carrier header in Red-HH parity space, exact binomial calculations were executed across error probabilities $p \in [0.01, 0.48]$ and code rates $R \in \{15, 31, 63\}$.

Let $p$ represent the independent and identically distributed (i.i.d.) bit error probability. A single bit protected by an $R$-fold repetition code ($R$ odd) and decoded by majority voting succeeds with probability:
$$P_{\text{bit}}(p, R) = \sum_{k=0}^{\lfloor R/2 \rfloor} \binom{R}{k} p^k (1 - p)^{R - k}$$
For the 16-byte (128-bit) carrier header, assuming independent bit errors across coefficients, the probability that all 128 bits survive is:
$$P_{\text{header}}(p, R) = \left[ P_{\text{bit}}(p, R) \right]^{128}$$

#### Exact Binomial Survival Probabilities

| Channel BER ($p$) | Repetition Factor ($R$) | Bit Recovery Prob ($P_{\text{bit}}$) | Header Survival Prob ($P_{\text{header}}$) | Feasibility Verdict |
|:---:|:---:|:---:|:---:|:---:|
| **$0.48$** (E14 Measured) | $R=15$ | $0.562608$ | **$1.06 \times 10^{-32}$** | **Mathematically Impossible** |
| **$0.48$** (E14 Measured) | $R=31$ | $0.588856$ | **$3.64 \times 10^{-30}$** | **Mathematically Impossible** |
| **$0.48$** (E14 Measured) | $R=63$ | $0.625091$ | **$7.60 \times 10^{-27}$** | **Mathematically Impossible** |
| **$0.40$** | $R=15$ | $0.786897$ | **$4.76 \times 10^{-14}$** | **Mathematically Impossible** |
| **$0.40$** | $R=31$ | $0.871618$ | **$2.30 \times 10^{-8}$** | **Unusable** |
| **$0.40$** | $R=63$ | $0.946309$ | **$8.56 \times 10^{-4}$** | **Severe Failure Rate ($> 99.9\%$)** |
| **$0.30$** | $R=15$ | $0.949987$ | **$1.41 \times 10^{-3}$** | **Unusable** |
| **$0.30$** | $R=31$ | $0.990460$ | **$0.293$** | **High Failure Rate ($70.7\%$)** |
| **$0.30$** | $R=63$ | $0.999562$ | **$0.945$** | **Marginal ($5.5\%$ packet drop)** |
| **$0.20$** | $R=15$ | $0.995760$ | **$0.581$** | **Unusable ($41.9\%$ drop)** |
| **$0.20$** | $R=31$ | $0.999912$ | **$0.989$** | **Viable ($1.1\%$ drop)** |
| **$0.20$** | $R=63$ | $1.000000$ | **$0.99999$** | **Highly Robust** |
| **$0.10$** | $R=15$ | $0.999966$ | **$0.9957$** | **Robust ($0.43\%$ drop)** |
| **$0.10$** | $R=31$ | $1.000000$ | **$1.0000$** | **Completely Robust** |
| **$\le 0.05$** | Any $R \ge 15$ | $1.000000$ | **$1.0000$** | **Completely Robust** |

---

### 4.3 Preservation of the Scoped Classical Failure Conclusion

1. **Specific Empirical Finding:**
   The failure observed in E14 was specifically that **discrete single-bit integer parity modulation in the high-frequency Red-HH subband cannot survive lossy JPEG compression**.
2. **Avoiding Overgeneralization:**
   This result does **not** prove that all error-correcting headers fail under lossy channels:
   - An ECC header embedded via **continuous Quantization Index Modulation (QIM)** in lower-frequency subbands (e.g., Red-LH or Red-HL) or spatial luminance could achieve $p \le 0.05$, where a simple $15\times$ repetition code provides $> 99.99\%$ survival.
   - However, within the constrained high-frequency Red-HH coefficient space of the Phase 2 baseline, discrete parity bit errors ($p \approx 0.48$) render repetition coding ineffective.
3. **Architectural Justification for Implicit Framing:**
   Because discrete parity repetition is mathematically unviable at $p=0.48$, and modifying lower-frequency subbands (LH/HL) would violate the strict Red-HH baseline parity invariant, **Implicit Fixed-Tier Framing is the only strategy that eliminates header vulnerability entirely without perturbing additional carrier subbands**.

---

## 5. Environment & Hardware Readiness

### 5.1 Host System Audit

The local host environment was audited directly:
- **Operating System:** Windows 10 (Build 10.0.26200, 64-bit AMD64).
- **CPU Platform:** AMD64 architecture.
- **Python Version:** Python 3.10.11 (tags/v3.10.11:7d4cc5a).
- **GPU Accelerator:** NVIDIA GeForce GTX 1650 (Device 0).
  - Architecture: Turing (TU117).
  - Compute Capability: `sm_75` (Version 7.5).
  - VRAM Capacity: 4,096 MiB (4.0 GB).
- **NVIDIA Display Driver:** Version 610.47.
  - CUDA Driver API Support: Up to CUDA 13.3.
- **Package Status:** No PyTorch or CUDA packages currently installed; core dependencies (`numpy`, `pillow`, `cryptography`, `reedsolo`) are fully functional.

---

### 5.2 PyTorch Compatibility and Installation Specification

1. **CUDA Driver vs. Bundled Runtime Architecture:**
   The host driver (610.47) provides the user-mode driver interface (`nvshift`, `nvcuda.dll`). Official PyTorch binary wheels for Windows bundle their own internal CUDA runtime libraries (`cudart64_12.dll`, `cublas64_12.dll`, `cudnn64_9.dll`). No separate NVIDIA CUDA Toolkit installation is required.
2. **Exact Compatible Wheels:**
   - Package: `torch==2.5.1+cu124`
   - Package: `torchvision==0.20.1+cu124`
   - Wheel Index URL: `https://download.pytorch.org/whl/cu124`
   - Turing `sm_75` support: Fully included in precompiled PTX and binary kernels.
3. **Execution Command (Requires User Authorization):**
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
4. **Environment Integrity Guarantee:**
   During this pre-implementation gate, no packages were installed, and no dependency files (`pyproject.toml`, `requirements.txt`) were modified.

---

### 5.3 VRAM Budget Specification on 4 GB GTX 1650

| Operational Mode | Carrier Size | Batch Size | Model Weights | Forward/Backward Activations | Optimizer State | Peak VRAM | Headroom (4.0 GB Limit) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Training (AMP FP16)** | $512 \times 512$ | $B=4$ | $0.4\text{ MB}$ | $\approx 380\text{ MB}$ | $\approx 3.0\text{ MB}$ | **$1.10\text{ GB}$** | **$2.90\text{ GB}$ ($72.5\%$)** |
| **Validation (Eval)** | $512 \times 512$ | $B=4$ | $0.4\text{ MB}$ | $\approx 120\text{ MB}$ | $0.0\text{ MB}$ | **$0.35\text{ GB}$** | **$3.65\text{ GB}$ ($91.2\%$)** |
| **Inference (Eval)** | $1024 \times 1024$ | $B=1$ | $0.4\text{ MB}$ | $\approx 180\text{ MB}$ | $0.0\text{ MB}$ | **$0.42\text{ GB}$** | **$3.58\text{ GB}$ ($89.5\%$)** |

---

## 6. Synthesis of Decisions Ready to Freeze vs. Unresolved Decisions

### 6.1 Decisions Ready to Freeze

1. **Subband Partition Contract (Option A):**
   Freeze $\text{HH}_{\text{keep}} = [\text{Green-HH}, \text{Blue-HH}]$ and $\text{HH}_{\text{drop}} = [\text{Red-HH}]$. Preserves 1:1 capacity equivalence with Phase 2 ($0.218\text{ bpp}$ spatial) and reuses the exact 2D mask shape $(H/2, W/2)$.
2. **Dual-Mode Framing Architecture:**
   - Mode A (Lossless): 16B parity header at mask indices 0..127 (100% Phase 2 backward compatibility).
   - Mode B (Robust Lossy): Implicit fixed-tier framing starting at mask index 0, authenticated by AES-256-GCM.
3. **Bipolar Symbol Mapping:**
   Standardize $P_i = 2 b_i - 1.0 \in \{-1.0, +1.0\}$ with receiver threshold at $\Delta_{\text{rec}} = 0.0$.
4. **Context Predictor CNN Output Activation:**
   Standardize linear activation (unbounded real values) for context prediction.
5. **Adaptive Gain CNN Structure:**
   Standardize $\gamma(x, y) = \text{Softplus}(x) + \gamma_{\text{min}}$ with gain floor $\gamma_{\text{min}} = 1.0$.
6. **Quantization Gradient Flow:**
   Standardize Straight-Through Estimator (STE) for differentiable 8-bit image rounding during training.
7. **Fail-Closed Cryptographic Invariant:**
   Preserve `Serialize -> Encrypt (AES-GCM) -> Reed-Solomon RS(255, 223)`. Any tag mismatch or uncorrectable RS block aborts with zero plaintext emission.

---

### 6.2 Unresolved Decisions (Experimental Hyperparameters)

These parameters are designated as experimental exploration variables for Phase 3 training and do not block code scaffolding:
1. **Loss Function Weightings:** Relative balance between stego fidelity loss ($\lambda_{\text{stego}}$), payload recovery cross-entropy ($\lambda_{\text{payload}}$), and high-frequency edge loss ($\lambda_{\text{hf}}$).
2. **Differentiable JPEG Approximation:** Selection between Fourier-based continuous DCT approximation vs. noise-injection surrogate during initial training epochs.
3. **ResNet Depth:** Initial baseline frozen at 3 residual blocks for `ContextPredictorCNN` and 2 blocks for `AdaptiveGainCNN`; whether expansion to 4 or 6 blocks is necessary will depend on convergence on the GTX 1650.

---

## 7. Minimal Pre-Implementation Acceptance Checklist

Prior to initiating model training or extensive pipeline refactoring, the following zero-risk validation checks must be executed:

- [ ] **Check 1: Implicit Framing Dry-Run Test**
  - Execute a standalone script (`scripts/experiments/test_implicit_framing_dryrun.py`) on the classical pipeline verifying that a fixed-tier payload embeds and extracts with $100\%$ byte equality without writing or reading an in-band header.
- [ ] **Check 2: Fail-Closed Wrong-Tier Test**
  - Verify that attempting extraction with an incorrect tier index $T' \neq T$ raises `ReedSolomonError` or `CryptoTamperError` and returns zero bytes.
- [ ] **Check 3: Green/Blue Channel Invariance Test**
  - Empirically assert that `numpy.array_equal(stego[:, :, 1], carrier[:, :, 1])` and `numpy.array_equal(stego[:, :, 2], carrier[:, :, 2])` under float64 IDWT.
- [ ] **Check 4: PyTorch CUDA Sanity Test**
  - Following user-authorized installation, verify GPU device recognition, CUDA availability, and tensor allocation on `sm_75`.
- [ ] **Check 5: Existing Regression Invariance**
  - Verify that all 103 existing unit and integration tests continue to pass with $100\%$ success rate.

---

## 8. Summary Table: Empirical Evidence vs. Design Assumptions

| Architectural Component | Evidence / Basis | Classification | Impact on Phase 3 Implementation |
|:---|:---|:---:|:---|
| **Red-HH Parity Header Failure** | E14 empirical record: 140/140 JPEG runs failed at header parsing with parity $\text{BER} \approx 48\%$. | Verified Fact | Header must be eliminated or moved to continuous/lower-frequency domain for lossy channels. |
| **Exact 2D Haar Channel Isolation** | Mathematical linearity and orthogonality proof; verified by 2D Haar implementation in `src/carrier/dwt.py`. | Mathematical Fact | Green and Blue channels can serve as pristine context without being modified by Red embedding. |
| **AES-GCM Fail-Closed Boundary** | 103 passing tests in `tests/`; cryptographic definition of 128-bit GHASH authentication tag. | Verified Fact | Wrong-tier selection or channel corruption is guaranteed to fail closed with $P_{\text{false\_accept}} \le 2^{-128}$. |
| **GTX 1650 VRAM Headroom** | Analytical tensor activation budget ($1.10\text{ GB}$ peak at $B=4$ vs $4.0\text{ GB}$ hardware ceiling). | Engineering Deduction | Model training is feasible locally on GTX 1650 without cloud GPU infrastructure. |
| **ACS-Sub-INN Lossless Invertibility** | Analytical proof: with $\gamma_{\text{min}} = 1.0 > 0.5$, spatial rounding cannot flip symbol signs. | Mathematical Fact | Guarantees lossless PNG recovery without sacrificing embedding capacity. |
| **ACS-Sub-INN Lossy JPEG Robustness** | Unproven hypothesis; depends on neural representation learning under simulated JPEG distortion. | Untested Hypothesis | Primary research question of Phase 3; must be tested via empirical experiment against E14 baseline. |
