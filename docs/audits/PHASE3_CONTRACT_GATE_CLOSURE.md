# PaladinMedia — Phase 3 Contract Corrections and Gate Closure

**Document ID:** `PDM-AUDIT-P3-GATE-CLOSURE-20261018`
**Date:** 2026-10-18
**Repository:** `E:\Paladin_media`
**Status:** CANONICAL AUDIT CORRECTION & PRE-IMPLEMENTATION GATE CLOSURE
**Governing Documents:** PaladinMedia Architecture Specification §10–§15, Phase 3 Acceptance Criteria (`PDM-P3-GATE-001`), Phase 3 Technical Readiness Checkpoint (`PDM-AUDIT-P3-TECH-READINESS-20261018`), Phase 3 Blocker Resolution Proposal (`PDM-DESIGN-P3-BLOCKER-PROPOSAL-20261018`), Phase 3 Pre-Implementation Contract Validation (`PDM-AUDIT-P3-CONTRACT-VALIDATION-20261018`)
**Target Hardware:** NVIDIA GeForce GTX 1650 (TU117, Compute Capability 7.5, 4 GB VRAM), Driver 610.47, Python 3.10.11

---

## 1. Executive Summary

This report reviews and corrects [`docs/audits/PHASE3_PREIMPLEMENTATION_CONTRACT_VALIDATION.md`](file:///E:/Paladin_media/docs/audits/PHASE3_PREIMPLEMENTATION_CONTRACT_VALIDATION.md), resolving framing precedence, rectifying reversibility assertions, establishing exact derived tier lengths, validating tensor and mask contracts against the codebase, and tightening evidence language.

In accordance with strict directive restrictions:
- Zero source code files in `src/` have been modified.
- Zero packages have been installed and no environment configuration files altered.
- Zero experiments have been rerun.
- Zero repository files have been deleted or moved.
- Phase 3 neural implementation has not been scaffolded or initiated.

---

## 2. Check 1 — Framing Precedence, Authenticated Metadata, and Recalculated Lengths

### 2.1 Normative Receiver Protocol Precedence for Mode B (Lossy Channels)

To resolve ambiguity between candidate mechanisms, the Mode B receiver protocol establishes a strict normative hierarchy:

1. **Primary Normative Protocol: Deterministic Session-Agreed Tiers**
   - The sender and receiver agree on an operating tier index $T \in \{T_1, T_2, T_3, T_{\max}\}$ out-of-band as part of protocol session context (e.g., handshake metadata, API query parameters, or fixed channel configuration).
   - The receiver directly extracts the exact bitstream length $L_{\text{extract}}(T) = L_{\text{protected}}(T) \times 8$ from the deterministic mask positions.
   - Exactly one extraction, one Reed–Solomon decoding pass, and one AES-GCM verification pass are executed.

2. **Secondary Fallback Protocol: Multi-Tier Trial Extraction**
   - **Invocation Condition:** Invoked **if and only if** the stego carrier is received without accompanying out-of-band session metadata (autonomous container ingestion mode).
   - **Execution Sequence:** The receiver iterates through candidate tiers in strictly ascending order:
     $$T_1 \to T_2 \to T_3 \to T_{\max}$$
     For each candidate tier $T_k$, the receiver extracts $L_{\text{extract}}(T_k)$ bits, attempts Reed–Solomon decoding, and evaluates AES-256-GCM authentication.
   - **Ambiguity Resolution:**
     Each tier is bound to Associated Authenticated Data (AAD):
     $$\text{AAD}(T_k, H, W) = \text{struct.pack(">2sBBHH", b"PM", 1, tier\_id, W, H)}$$
     Because the AES-256-GCM GHASH tag is 128 bits, the probability that an incorrect tier candidate produces a valid tag is bounded by:
     $$P_{\text{false\_accept}} \le 2^{-128} \approx 2.9 \times 10^{-39}$$
     The first tier candidate that passes GHASH verification is accepted as the unique valid payload. If all candidates fail, execution terminates with `AuthenticationError`.

---

### 2.2 Correction: Cryptographic Boundary vs. Reed–Solomon Decoding

The claim in the previous audit that Reed–Solomon decoding *must* fail for every wrong tier selection is **technically incorrect and hereby retracted**:
- **Why RS Decoding May Not Fail:**
  In `ReedSolomonEncoder.decode()` ([`src/security/ecc/encoder.py`](file:///E:/Paladin_media/src/security/ecc/encoder.py#L227-L268)), the decoder unpacks a 4-byte length header from the start of the buffer and verifies whether `len(raw_enc) == expected_len(total_length)`. If candidate tier extraction yields random carrier bits whose first 4 bytes happen to encode a length value matching the buffer length, or if an over-reading receiver encounters trailing coefficients that coincidentally satisfy the RS parity syndromes, RS decoding can execute without raising `ECCDecodeError`.
- **The True Fail-Closed Security Boundary:**
  The definitive fail-closed boundary of PaladinMedia is **AES-256-GCM Authenticated Encryption with Associated Data (NIST SP 800-38D)** implemented in `AESGCMProvider.decrypt()` ([`src/security/crypto/provider.py`](file:///E:/Paladin_media/src/security/crypto/provider.py#L234-L292)). Even if Reed–Solomon decoding emits arbitrary candidate bytes, the 16-byte GHASH tag verification fails with probability $1 - 2^{-128}$, immediately raising `AuthenticationError` and returning zero plaintext.

---

### 2.3 Authenticated Tier Metadata & AES-GCM AAD Compatibility

The proposed tier metadata binding is fully compatible with the existing cryptographic implementation:
- `AESGCMProvider.encrypt` and `decrypt` take an optional `aad: Optional[bytes]` parameter and pass it directly to `cryptography.hazmat.primitives.ciphers.aead.AESGCM`.
- In `AESGCMProvider`, the wire packaging format is strictly:
  $$\text{packed} = \text{nonce}_{12\text{B}} \parallel \text{ciphertext} \parallel \text{tag}_{16\text{B}}$$
  The AAD is **not** appended to the ciphertext or embedded in the carrier; it is an out-of-band byte string authenticated via GHASH.
- In `SecurityPipeline.protect(payload, key, aad=aad)` and `unprotect(protected_bytes, key, aad=aad)` ([`src/core/pipeline.py`](file:///E:/Paladin_media/src/core/pipeline.py#L49-L116)), the `aad` argument is forwarded directly to `CryptoProvider`.
- Thus, binding `aad = struct.pack(">2sBBHH", b"PM", 1, tier_id, width, height)` requires zero alterations to the wire format or cryptographic classes.

---

### 2.4 Mathematical Derivation & Recalculation of Tier Protected Lengths

The protected-stream sizes presented in the previous audit omitted the 4-byte Reed–Solomon length framing prefix and miscalculated RS block overhead. Below is the rigorous derivation based on the actual codebase implementation:

#### Governing Equations:
1. **Serialized Plaintext ($L_{\text{ser}}$):**
   Per `SecretSerializer.serialize()` ([`src/core/serialization/serializer.py`](file:///E:/Paladin_media/src/core/serialization/serializer.py#L166-L186)), with $M=0$ metadata bytes:
   $$L_{\text{ser}} = \text{Prefix}_{10\text{B}} + \text{Meta}_{0\text{B}} + \text{PayloadLen}_{4\text{B}} + L_{\text{plain}} + \text{CRC32}_{4\text{B}} = L_{\text{plain}} + 18\text{ bytes}$$
2. **Encrypted Envelope ($L_{\text{enc}}$):**
   Per `AESGCMProvider.encrypt()` ([`src/security/crypto/provider.py`](file:///E:/Paladin_media/src/security/crypto/provider.py#L222)), AES-GCM preserves plaintext length:
   $$L_{\text{enc}} = \text{Nonce}_{12\text{B}} + L_{\text{ser}} + \text{Tag}_{16\text{B}} = L_{\text{ser}} + 28 = L_{\text{plain}} + 46\text{ bytes}$$
3. **Reed–Solomon Protected Stream ($L_{\text{prot}}$):**
   Per `ReedSolomonEncoder.encode()` ([`src/security/ecc/encoder.py`](file:///E:/Paladin_media/src/security/ecc/encoder.py#L178-L191)), using RS(255, 223) with $k=223$ data symbols and $2t=32$ parity symbols per block, prefixed with a 4-byte length:
   $$N_{\text{full}} = \lfloor L_{\text{enc}} / 223 \rfloor, \quad R = L_{\text{enc}} \pmod{223}$$
   $$N_{\text{blocks}} = N_{\text{full}} + (1 \text{ if } R > 0 \text{ else } 0)$$
   $$L_{\text{prot}} = 4 + L_{\text{enc}} + (N_{\text{blocks}} \times 32)\text{ bytes}$$
   $$L_{\text{extract}} = L_{\text{prot}} \times 8\text{ bits}$$

#### Derived Tier Size Table

| Tier Identifier | Target Plaintext ($L_{\text{plain}}$) | Serialized ($L_{\text{ser}}$) | Encrypted ($L_{\text{enc}}$) | RS Blocks ($N_{\text{blocks}}$) | RS Parity Bytes | Protected Stream ($L_{\text{prot}}$) | Extraction Bits ($L_{\text{extract}}$) | Audit Validation Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tier 1 ($T_1$, Low)** | $1,500\text{ B}$ | $1,518\text{ B}$ | $1,546\text{ B}$ | $7$ ($6$ full, $R=208$) | $224\text{ B}$ | **$1,774\text{ B}$** | $14,192\text{ bits}$ | Corrected (was $1,770\text{ B}$) |
| **Tier 2 ($T_2$, Medium)** | $7,000\text{ B}$ | $7,018\text{ B}$ | $7,046\text{ B}$ | $32$ ($31$ full, $R=133$) | $1,024\text{ B}$ | **$8,074\text{ B}$** | $64,592\text{ bits}$ | Corrected (was $8,040\text{ B}$) |
| **Tier 3 ($T_3$, High 1024)** | $20,000\text{ B}$ | $20,018\text{ B}$ | $20,046\text{ B}$ | $90$ ($89$ full, $R=199$) | $2,880\text{ B}$ | **$22,930\text{ B}$** | $183,440\text{ bits}$ | Corrected (was $22,910\text{ B}$) |
| **$T_{\max512}$ (512 Boundary)** | $7,090\text{ B}$ | $7,108\text{ B}$ | $7,136\text{ B}$ | $32$ ($32$ full, $R=0$) | $1,024\text{ B}$ | **$8,164\text{ B}$** | $65,312\text{ bits}$ | Verified (E12 baseline) |
| **$T_{\max1024}$ (1024 Boundary)**| $28,574\text{ B}$ | $28,592\text{ B}$ | $28,620\text{ B}$ | $129$ ($128$ full, $R=76$) | $4,128\text{ B}$ | **$32,752\text{ B}$** | $262,016\text{ bits}$ | Verified (E12 baseline) |

---

## 3. Check 2 — Reversibility Analysis and Receiver Derivation

### 3.1 Retraction of the Flawed Analytical Reversibility Claim

The previous document stated that setting $\gamma_{\text{min}} = 1.0$ analytically proves $100\%$ exact bit recovery in lossless PNG mode based on the inequality $|\Delta_{\text{rec}}| \ge 1.0 - 0.5 = 0.5 > 0$. **This claim is invalid and is hereby retracted**:
1. **Haar DWT Error Amplification:**
   The 2D Haar DWT detail coefficient is:
   $$\text{HH} = \frac{a - b - c + d}{2.0}$$
   If spatial integer rounding perturbations across the $2 \times 2$ block are $(\Delta a, \Delta b, \Delta c, \Delta d) = (+0.5, -0.5, -0.5, +0.5)$, the resulting perturbation in $\text{HH}$ is:
   $$\delta_{\text{HH}} = \frac{0.5 - (-0.5) - (-0.5) + 0.5}{2.0} = \frac{2.0}{2.0} = 1.0$$
   Thus, spatial rounding can perturb the Red-HH coefficient by up to $\pm 1.0$, which can exactly cancel a gain floor of $\gamma = 1.0$, producing an ambiguous residual $\Delta_{\text{rec}} = 0.0$.
2. **Spatial Dynamic Range Clipping:**
   When modulated pixel values exceed 255 or fall below 0, spatial clamping ($\text{clamp}(I', 0, 255)$) introduces non-linear truncation errors $|\Delta I| > 0.5$, which propagate through the DWT as large non-linear perturbations.
3. **Conclusion:** Exact recovery under 8-bit spatial rounding and clamping **cannot be proven by scalar gain bounding alone**. It requires empirical validation.

---

### 3.2 Derivation of the Ideal Receiver

In continuous unquantized float64 arithmetic, the forward modulation equation at active mask coordinates $(r, c)$ where $M[r, c] = 1$ is:
$$\text{HH}_{\text{drop}}'[r, c] = f_{\theta}(\text{HH}_{\text{keep}})[r, c] + g_{\phi}(\text{HH}_{\text{keep}})[r, c] \cdot P[r, c]$$
where $P[r, c] \in \{-1.0, +1.0\}$, $f_{\theta}$ is `ContextPredictorCNN`, and $g_{\phi}(\text{HH}_{\text{keep}}) = \text{Softplus}(\dots) + \gamma_{\text{min}} > 0$.

The receiver receives carrier subbands $\text{HH}^* = [\text{HH}_{\text{drop}}^* \parallel \text{HH}_{\text{keep}}^*]$. The ideal receiver evaluates:
1. **Context Reconstruction:**
   $$\text{context}_{\text{rec}} = f_{\theta}(\text{HH}_{\text{keep}}^*)$$
2. **Demodulation Residual:**
   $$\Delta_{\text{rec}}[r, c] = \text{HH}_{\text{drop}}^*[r, c] - \text{context}_{\text{rec}}[r, c]$$
3. **Symbol & Bit Demapping:**
   Under continuous unquantized conditions where $\text{HH}^* \equiv \text{HH}'$:
   $$\Delta_{\text{rec}}[r, c] = \left(f_{\theta}(\text{HH}_{\text{keep}}) + g_{\phi}(\text{HH}_{\text{keep}}) \cdot P\right) - f_{\theta}(\text{HH}_{\text{keep}}) = g_{\phi}(\text{HH}_{\text{keep}})[r, c] \cdot P[r, c]$$
   Because $g_{\phi}(\text{HH}_{\text{keep}}) > 0$:
   $$\hat{P}[r, c] = \text{sign}(\Delta_{\text{rec}}[r, c]) = P[r, c]$$
   $$\hat{b}[r, c] = \mathbb{I}(\Delta_{\text{rec}}[r, c] > 0)$$

---

### 3.3 Rigorous Separation of Recovery Stages

| Pipeline Recovery Stage | Mathematical Formulation | Status | Justification / Verification Requirement |
|:---|:---|:---:|:---|
| **Stage 1: Continuous Coefficient Domain** | $\text{HH}_{\text{rec}} = \text{HH}'$, $\text{HH}_{\text{keep}}^* \equiv \text{HH}_{\text{keep}}$ | **Mathematically Proven** | Direct cancellation in float64 algebra: $\Delta_{\text{rec}} = \gamma \odot P$. Since $\gamma > 0$, sign recovery is exact. |
| **Stage 2: Post-IDWT, Rounding, & PNG Serialization** | $\text{HH}_{\text{rec}} = \text{DWT}(\text{round}(\text{clamp}(\text{IDWT}(\text{HH}'))))$ | **Requires Empirical Test** | Spatial rounding introduces perturbations $|\delta_{\text{HH}}| \le 1.0$. Clipping can introduce larger errors. Exact recovery depends on network avoidance of saturation. |
| **Stage 3: Lossy JPEG Compression ($Q \ge 80$)** | $\text{HH}_{\text{rec}} = \text{DWT}(\text{JPEG}_Q(\text{IDWT}(\text{HH}')))$ | **Untested Hypothesis** | DCT quantization perturbs both $\text{HH}_{\text{drop}}$ and $\text{HH}_{\text{keep}}$, inducing context error $\delta_{\text{pred}}$. Requires training and test. |
| **Stage 4: Additive Channel Noise ($\sigma \ge 0.5$)** | $\text{HH}_{\text{rec}} = \text{DWT}(\text{IDWT}(\text{HH}') + \mathcal{N}(0, \sigma^2))$ | **Untested Hypothesis** | Linear and non-linear noise perturbations require empirical evaluation against the E14 baseline. |

---

## 4. Check 3 — Tensor and Mask Contract Validation

### 4.1 Ground-Truth DWT Implementation Properties

Audit of `HaarDWT2D` ([`src/carrier/dwt.py`](file:///E:/Paladin_media/src/carrier/dwt.py#L55-L160)) confirms:
- **Input Image Scale:** Spatial uint8 array $[0, 255]$ converted to `np.float64`.
- **Channel Order:** Spatial input is $(H, W, 3)$ RGB. Output subband tensors are transposed to `channel-first` $(3, H/2, W/2)$ in `float64`:
  - Channel 0: **Red**
  - Channel 1: **Green**
  - Channel 2: **Blue**
- **Subband Output Shapes:**
  - $512 \times 512 \implies \text{LL}, \text{LH}, \text{HL}, \text{HH}$ each of shape $(3, 256, 256)$ in `float64`.
  - $1024 \times 1024 \implies \text{LL}, \text{LH}, \text{HL}, \text{HH}$ each of shape $(3, 512, 512)$ in `float64`.
- **Subband Dynamic Ranges:**
  - $\text{LL} = (a + b + c + d) / 2.0 \in [0.0, 510.0]$
  - $\text{LH}, \text{HL}, \text{HH} \in [-255.0, 255.0]$
- **IDWT Formulation:** Output is reconstructed to $(H, W, 3)$ in `float64` before spatial rounding.

---

### 4.2 Available Mask Positions and Capacity Allocation

Per `DeterministicMaskGenerator` ([`src/carrier/mask.py`](file:///E:/Paladin_media/src/carrier/mask.py#L42-L100)) and `CapacityCalculator` ([`src/carrier/capacity.py`](file:///E:/Paladin_media/src/carrier/capacity.py#L66-L100)):
- **Total Red-HH Positions:**
  - $512 \times 512$: $256 \times 256 = 65,536\text{ positions}$ ($8,192\text{ bytes}$).
  - $1024 \times 1024$: $512 \times 512 = 262,144\text{ positions}$ ($32,768\text{ bytes}$).
- **Mode A (Lossless PNG with In-Band Header):**
  - 128 positions reserved for the 16-byte `CarrierHeader` at indices $0 \dots 127$.
  - Available payload capacity: $65,408\text{ bits}$ ($8,176\text{ bytes}$) on $512^2$; $262,016\text{ bits}$ ($32,752\text{ bytes}$) on $1024^2$.
- **Mode B (Lossy Channels with Implicit Framing):**
  - 0 positions reserved for headers.
  - Available payload capacity: $65,536\text{ bits}$ ($8,192\text{ bytes}$) on $512^2$; $262,144\text{ bits}$ ($32,768\text{ bytes}$) on $1024^2$.

#### Tier Capacity Fit Audit

1. **Tier 1 ($1,774\text{ B}$ protected):**
   - Fits on $512^2$ in Mode A ($1,774 \le 8,176\text{ B}$, utilization $21.7\%$) and Mode B ($1,774 \le 8,192\text{ B}$, utilization $21.6\%$).
2. **Tier 2 ($8,074\text{ B}$ protected):**
   - Fits on $512^2$ in Mode A ($8,074 \le 8,176\text{ B}$, utilization $98.75\%$, headroom $102\text{ B}$) and Mode B ($8,074 \le 8,192\text{ B}$, utilization $98.56\%$, headroom $118\text{ B}$).
3. **Tier 3 ($22,930\text{ B}$ protected):**
   - Exceeds $512^2$ (as designed; target carrier is $1024^2$).
   - Fits on $1024^2$ in Mode A ($22,930 \le 32,752\text{ B}$, utilization $70.01\%$, headroom $9,822\text{ B}$) and Mode B ($22,930 \le 32,768\text{ B}$, utilization $69.98\%$).
4. **Boundary Tiers ($T_{\max512}=8,164\text{ B}$, $T_{\max1024}=32,752\text{ B}$):**
   - $T_{\max512}$ fits $512^2$ Mode A with $12\text{ B}$ remainder.
   - $T_{\max1024}$ fits $1024^2$ Mode A with exactly $100.0\%$ utilization.

---

### 4.3 Preservation of Frozen Architecture Constraints

The frozen ACS-Sub-INN v3 architecture is strictly preserved without modification:
- **No Secret-Side DWT:** Secret payloads enter exclusively as binary byte sequences.
- **No Latent-$Z$:** Inversion is deterministic context subtraction without stochastic latent variables.
- **Subband Option A:** $\text{HH}_{\text{keep}} = [\text{Green}, \text{Blue}]$ and $\text{HH}_{\text{drop}} = [\text{Red}]$.

---

## 5. Check 4 — Evidence Language and Scoped Classical Baseline Conclusions

### 5.1 Verification of Repetition-Code Binomial Calculations

Under the explicitly stated assumption that coefficient bit errors follow an independent and identically distributed (i.i.d.) Bernoulli distribution with error probability $p$:
- A single bit protected by an $R$-fold repetition code ($R$ odd) and decoded by majority voting succeeds with probability:
  $$P_{\text{bit}}(p, R) = \sum_{k=0}^{\lfloor R/2 \rfloor} \binom{R}{k} p^k (1 - p)^{R - k}$$
- For a 128-bit header, the probability that all 128 bits survive is:
  $$P_{\text{header}}(p, R) = \left[ P_{\text{bit}}(p, R) \right]^{128}$$

The calculations reproduce identically:
- At $p = 0.48$:
  - $R=15$: $P_{\text{bit}} \approx 0.562608 \implies P_{\text{header}} \approx 1.06 \times 10^{-32}$
  - $R=31$: $P_{\text{bit}} \approx 0.588856 \implies P_{\text{header}} \approx 3.64 \times 10^{-30}$
  - $R=63$: $P_{\text{bit}} \approx 0.625091 \implies P_{\text{header}} \approx 7.60 \times 10^{-27}$
- At $p \le 0.05$:
  - Any $R \ge 15$: $P_{\text{header}} \ge 0.9999$

---

### 5.2 Explicit Labeling of Model Assumptions and Scoped E14 Findings

1. **Model Limitations:**
   The binomial analysis represents an idealized statistical model. Real JPEG compression produces spatially correlated error clusters due to $8 \times 8$ block DCT quantization and run-length encoding.
2. **Non-Extrapolation to General ECC Headers:**
   The mathematical impossibility of majority-voting repetition at $p \approx 0.48$ must **not** be extrapolated into a claim that error-correcting headers in general cannot survive lossy channels:
   - Error-correcting codes (BCH, Golay, repetition) function reliably when channel BER is below the code rate threshold ($p < 0.10$).
   - A header embedded via continuous modulation in lower-frequency DCT bands or spatial luminance could easily achieve $p < 0.05$ under JPEG $Q=80$.
3. **Preservation of the Scoped E14 Conclusion:**
   Experiment E14 established specifically that **discrete single-bit integer parity modulation in the high-frequency Red-HH subband fails completely under JPEG compression and additive noise**, with parity BER $\approx 47.6\% - 50.0\%$. The conclusion is strictly confined to this specific classical modulation mechanism under the tested channel degradation parameters.

---

## 6. Gate Closure Table

The architectural and operational contracts are categorized into three distinct gate tiers:

| Contract / Decision Area | Current Status | Gate Verdict | Scope of Commitment |
|:---|:---:|:---:|:---|
| **Subband Partition Contract (Option A)** | Fully Specified | **READY TO FREEZE** | $\text{HH}_{\text{keep}} = [\text{Green}, \text{Blue}]$, $\text{HH}_{\text{drop}} = [\text{Red}]$. 1:1 capacity equivalence ($0.218\text{ bpp}$). |
| **Normative Mode B Receiver Hierarchy** | Fully Specified | **READY TO FREEZE** | Primary: Session-Agreed Tier; Fallback: Strictly ordered Trial Extraction ($T_1 \to T_2 \to T_3 \to T_{\max}$). |
| **AES-GCM GHASH Fail-Closed Boundary** | Fully Specified | **READY TO FREEZE** | 128-bit GHASH tag verification ($P_{\text{false\_accept}} \le 2^{-128}$). AAD binds version, tier ID, and dimensions. |
| **Derived Tier Sizes & Bit Budgets** | Fully Specified | **READY TO FREEZE** | $T_1 = 1,774\text{ B}$, $T_2 = 8,074\text{ B}$, $T_3 = 22,930\text{ B}$, $T_{\max512} = 8,164\text{ B}$, $T_{\max1024} = 32,752\text{ B}$. |
| **Continuous Domain Reversibility** | Mathematically Proven | **READY TO FREEZE** | Proven for continuous float64 arithmetic: $\Delta_{\text{rec}} = \gamma \odot P$ with $\gamma > 0$. |
| **Mode A Backward Compatibility** | Fully Specified | **READY TO FREEZE** | 16-byte in-band parity header at indices 0..127 preserved for lossless PNG workflow. |
| **Hardware & Wheel Compatibility** | Fully Specified | **READY TO FREEZE** | Windows 10, Python 3.10.11, Driver 610.47, GTX 1650 (`sm_75`), `torch==2.5.1+cu124`. |
| **Spatial Rounding / Clamping Reversibility**| Empirically Dependent | **REQUIRES VALIDATION** | Exact bit recovery after IDWT, integer rounding, clamping, and PNG save/reload must be empirically tested. |
| **ACS-Sub-INN JPEG Robustness ($Q=80$)** | Research Hypothesis | **REQUIRES VALIDATION** | Pre-ECC $\text{BER} \le 3.0\%$ under JPEG $Q=80$ requires model training and benchmarking against E14. |
| **Loss Function Weightings ($\lambda$)** | Experimental Hyperparam | **REQUIRES VALIDATION** | Stego fidelity vs. payload recovery loss balance to be tuned during Phase 3. |
| **Prerequisite 1: Framing Contract Approval** | Pending User Authorization | **OPERATIONAL BLOCKER** | Requires explicit user sign-off to freeze Dual-Mode Framing and Option A subband partition. |
| **Prerequisite 2: PyTorch Installation** | Pending User Authorization | **OPERATIONAL BLOCKER** | Requires explicit user sign-off to execute `pip install torch torchvision --index-url ...`. |

---

## 7. Required Pre-Implementation Validation Commands & Tests (Non-Executed)

The following commands and test scripts define the exact sequence required to validate the remaining empirical claims once authorization is granted. **In accordance with directive instructions, these commands have not been executed**:

### Validation Test 1: Implicit Framing and Cryptographic Boundary Verification
```powershell
# Verify that fixed-tier protected streams embed and extract without an in-band header,
# and confirm that wrong-tier extraction fails closed via AES-GCM GHASH verification:
python scripts/experiments/test_implicit_framing_dryrun.py
pytest tests/integration/test_wrong_tier_fail_closed.py -v
```

### Validation Test 2: Spatial Rounding and Saturation Margin Test
```powershell
# Empirically verify whether neural modulation with gamma_min=1.0 induces integer rounding
# bit flips or boundary clipping under lossless PNG save/reload:
python scripts/experiments/verify_spatial_rounding_margin.py
```

### Validation Test 3: PyTorch Wheel Installation & CUDA Sanity
```powershell
# Execute official wheel installation upon user authorization:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA 12.4 runtime, Turing sm_75 capability, and memory allocation on GTX 1650:
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0), '| Capability:', torch.cuda.get_device_capability(0), '| VRAM:', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), 'GB')"
```

---

## 8. Final Audit Sign-Off

All contract ambiguities, reversibility over-statements, and tier length discrepancies have been rigorously resolved and reconciled against the PaladinMedia repository.

**Auditor:** Antigravity Research Validation & Scientific Audit
**Status:** AUDIT COMPLETE — STOPPING EXECUTION AS DIRECTED
