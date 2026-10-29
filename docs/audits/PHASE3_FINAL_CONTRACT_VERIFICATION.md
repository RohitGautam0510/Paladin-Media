# PaladinMedia — Final Phase 3 Contract Verification

**Document ID:** `PDM-AUDIT-P3-FINAL-VERIFICATION-20261018`
**Date:** 2026-10-18
**Repository:** `E:\Paladin_media`
**Status:** CANONICAL AUDIT VERIFICATION & CONTRACT FREEZE RECOMMENDATION
**Audit Target:** [`docs/audits/PHASE3_CONTRACT_GATE_CLOSURE.md`](file:///E:/Paladin_media/docs/audits/PHASE3_CONTRACT_GATE_CLOSURE.md)
**Governing Documents:** PaladinMedia Architecture Specification §10–§15, System & Research Document Draft §21, Phase 3 Acceptance Criteria (`PDM-P3-GATE-001`), Phase 3 Technical Readiness Checkpoint (`PDM-AUDIT-P3-TECH-READINESS-20261018`), Phase 3 Blocker Resolution Proposal (`PDM-DESIGN-P3-BLOCKER-PROPOSAL-20261018`), Phase 3 Pre-Implementation Contract Validation (`PDM-AUDIT-P3-CONTRACT-VALIDATION-20261018`)
**Target Hardware:** NVIDIA GeForce GTX 1650 (TU117, Compute Capability 7.5, 4 GB VRAM), Driver 610.47, Python 3.10.11

---

## 1. Executive Summary & Verification Verdict

### Contract Freeze Decision: `CONDITIONAL GO`
*(This verdict applies strictly to freezing the architecture and contract specifications; it is entirely separate from authorization to implement Phase 3 or install packages).*

This audit resolves the remaining evidence, protocol, and capacity questions identified across prior audit iterations. Every mathematical derivation, tensor shape, subband partition, and framing rule has been verified against the physical source code in `E:\Paladin_media`.

### Strict Audit Restrictions Maintained
- Zero source code files in `src/` have been modified.
- Zero packages have been installed or upgraded; `pyproject.toml` remains untouched.
- Phase 3 neural implementation has not been scaffolded or initiated.
- Zero repository files have been deleted, moved, renamed, or consolidated.
- Full regression test execution confirms **103/103 tests pass in 2.34s**.
- Uncreated test scripts are explicitly cataloged as non-existent rather than reported as executed.

---

## 2. Check 1 — AES-GCM Associated Authenticated Data (AAD) Contract

### 2.1 Trace of Existing Source-Code AAD Dataflow

Inspection of the cryptographic pipeline reveals how AAD is currently handled across the codebase:
1. **Crypto Layer:**
   - In [`src/security/crypto/provider.py`](file:///E:/Paladin_media/src/security/crypto/provider.py#L159-L223), `AESGCMProvider.encrypt(plaintext, key, aad=None)` passes `bytes(aad) if aad is not None else None` directly to the underlying `cryptography.hazmat.primitives.ciphers.aead.AESGCM.encrypt()` implementation.
   - The emitted wire packaging format ([`src/security/crypto/provider.py`](file:///E:/Paladin_media/src/security/crypto/provider.py#L48-L50)) is strictly:
     $$\text{packed} = \text{nonce}_{12\text{B}} \parallel \text{ciphertext} \parallel \text{tag}_{16\text{B}}$$
     The AAD is **never** appended, prefixed, or stored in the packed byte stream.
   - In [`src/security/crypto/provider.py`](file:///E:/Paladin_media/src/security/crypto/provider.py#L224-L292), `AESGCMProvider.decrypt(protected_payload, key, aad=None)` unpacks the 12-byte nonce and 16-byte tag, then calls `AESGCM.decrypt(nonce, ciphertext_and_tag, aad)`. If the provided AAD differs by even a single bit from the encryption AAD, an `AuthenticationError` is raised fail-closed.
2. **Pipeline Integration:**
   - In [`src/core/pipeline.py`](file:///E:/Paladin_media/src/core/pipeline.py#L49-L79), `SecurityPipeline.protect(payload, key, aad=None)` receives `aad` as a caller-supplied parameter and forwards it to `crypto_provider.encrypt()`.
   - In [`src/core/pipeline.py`](file:///E:/Paladin_media/src/core/pipeline.py#L81-L116), `SecurityPipeline.unprotect(protected_bytes, key, aad=None)` forwards `aad` to `crypto_provider.decrypt()`.
   - In [`src/core/media_pipeline.py`](file:///E:/Paladin_media/src/core/media_pipeline.py#L114-L144), `MediaStegoPipeline.protect_and_embed()` and `extract_and_unprotect()` pass `aad` through to `SecurityPipeline`.

---

### 2.2 Existing vs. Proposed AAD Functionality

| Dimension | Existing Implementation (`src/`) | Proposed Phase 3 Functionality | Distinction Status |
|:---|:---|:---|:---:|
| **AAD Generation** | Caller must construct and pass `aad: Optional[bytes]`. Defaults to `None`. | Automated derivation: `struct.pack(">2sBBHH", b"PM", 1, tier_id, width, height)`. | **PROPOSED** |
| **Tier Binding** | No tier information is encoded or enforced in `aad`. | Binds discrete tier index ($T_1, T_2, T_3, T_{\max}$) into the GHASH calculation. | **PROPOSED** |
| **Carrier Geometry Binding**| Carrier dimensions $(W, H)$ are not verified via AAD. | Prevents cross-resolution transplanting of ciphertext streams. | **PROPOSED** |
| **Wire Storage** | AAD is purely out-of-band; zero bits stored in carrier. | Preserved: AAD remains purely out-of-band; zero bits in carrier. | **ALIGNED** |
| **Fail-Closed Rejection** | Tested and verified in [`tests/integration/test_fail_closed.py#L145-L162`](file:///E:/Paladin_media/tests/integration/test_fail_closed.py#L145-L162). | Preserved: Any AAD mismatch aborts with `AuthenticationError`. | **VERIFIED FACT** |

---

### 2.3 Rigorous Recalculation of Aggregate False-Acceptance Bounds

The prior audit asserted a nominal false-acceptance probability of $2^{-128}$ based solely on the tag length. To evaluate autonomous trial extraction across multiple candidate tiers, the aggregate security bound must be formally derived under explicit cryptographic assumptions:

#### Explicit Modeling Assumptions:
1. **Secret Key Uniformity:** The 256-bit symmetric key $K$ is generated via a cryptographically secure pseudorandom number generator (CSPRNG) and is completely unknown to any adversary.
2. **Cipher Security (PRP):** AES-256 behaves as a Pseudorandom Permutation (NIST SP 800-38D). Under key $K$, the output ciphertext and GHASH authentication tag are computationally indistinguishable from uniform random bitstrings.
3. **Trial Independence:** Because each candidate tier $T_k$ specifies a distinct AAD ($\text{tier\_id}_k$) and/or a distinct extracted bitstream slice, the candidate GHASH evaluations correspond to independent oracle queries.
4. **Bounded Evaluation Budget:** The receiver evaluates a maximum of $M$ discrete candidate tiers ($M \le 4$, corresponding to $\{T_1, T_2, T_3, T_{\max}\}$).

#### Union Bound Derivation:
Let $E_i$ denote the event that incorrect candidate tier $i \in \{1, \dots, M-1\}$ produces an accidental valid 128-bit GHASH tag. For each individual trial:
$$P(E_i) \le 2^{-128} \approx 2.9387 \times 10^{-39}$$
By Boole's inequality (the union bound), the aggregate probability that **at least one** incorrect candidate falsely authenticates is:
$$P_{\text{agg}}(\text{false accept}) = P\left(\bigcup_{i=1}^{M-1} E_i\right) \le \sum_{i=1}^{M-1} P(E_i) \le (M - 1) \cdot 2^{-128}$$
For the standardized 4-tier catalog ($M = 4$, yielding at most 3 incorrect candidates):
$$P_{\text{agg}}(\text{false accept}) \le 3 \cdot 2^{-128} \approx 8.8162 \times 10^{-39}$$

**Conclusion:** Even when evaluating all candidate tiers autonomously, the aggregate probability of an accidental cryptographic false positive remains vanishingly small ($\approx 8.82 \times 10^{-39}$). Cryptographic integrity is unconditionally preserved.

---

## 3. Check 2 — Protected-Length Verification & Capacity Headroom

### 3.1 Source Implementation Reconciliation

The exact lengths for serialized, encrypted, and error-corrected streams were audited against the physical classes:
1. **`SecretSerializer.serialize()` ([`src/core/serialization/serializer.py#L111-L186`](file:///E:/Paladin_media/src/core/serialization/serializer.py#L111-L186)):**
   - Fixed Prefix Header: Magic (`PLDS`, 4B) + Version (`0x01`, 1B) + MediaType (1B) + Flags (`0x0000`, 2B) + MetaLen (2B) = **10 bytes**.
   - Metadata JSON: Deterministic UTF-8 bytes of length $M$. (For pure raw bytes with no metadata, $M = 0$).
   - Payload Length: 4-byte big-endian uint32 = **4 bytes**.
   - Payload Data: Raw bytes of length $L_{\text{plain}}$.
   - CRC-32 Checksum: IEEE 802.3 over Magic through Data = **4 bytes**.
   - **Total Serialized Plaintext:** $L_{\text{ser}} = L_{\text{plain}} + M + 18\text{ bytes}$.
2. **`AESGCMProvider.encrypt()` ([`src/security/crypto/provider.py#L139-L223`](file:///E:/Paladin_media/src/security/crypto/provider.py#L139-L223)):**
   - Nonce: 12 bytes.
   - Ciphertext: Exactly preserves plaintext length ($L_{\text{ser}}$).
   - GHASH Tag: 16 bytes.
   - **Total Encrypted Envelope:** $L_{\text{enc}} = L_{\text{ser}} + 28 = L_{\text{plain}} + M + 46\text{ bytes}$.
3. **`ReedSolomonEncoder.encode()` ([`src/security/ecc/encoder.py#L99-L191`](file:///E:/Paladin_media/src/security/ecc/encoder.py#L99-L191)):**
   - Framing Prefix: 4-byte big-endian uint32 encoding $L_{\text{enc}}$ = **4 bytes**.
   - Systematic Codewords: Chunks of $k=223$ data bytes + 32 parity bytes ($n=255$).
   - Shortened Final Block: Remainder $R = L_{\text{enc}} \pmod{223}$ receives 32 parity bytes.
   - Total Blocks: $N_{\text{blocks}} = \lfloor L_{\text{enc}} / 223 \rfloor + (1 \text{ if } R > 0 \text{ else } 0)$.
   - Total Parity: $N_{\text{blocks}} \times 32\text{ bytes}$.
   - **Total Protected Stream:** $L_{\text{prot}} = 4 + L_{\text{enc}} + (N_{\text{blocks}} \times 32)\text{ bytes}$.
   - **Total Active Mask Extraction Bits:** $L_{\text{extract}} = L_{\text{prot}} \times 8\text{ bits}$.

Both the 4-byte length prefix and all Reed–Solomon parity bytes are strictly part of `protected_payload` and are embedded directly into the carrier.

---

### 3.2 Carrier Capacity Headroom Audit ($M = 0$ Metadata Baseline)

Physical capacity of 1 Red-HH subband:
- **$512 \times 512$ Carrier ($256 \times 256$ subband):**
  - Raw Capacity: $256 \times 256 = 65,536\text{ bits}$ ($8,192\text{ bytes}$).
  - **Mode A (Lossless PNG):** 128 bits (16 bytes) reserved for in-band `CarrierHeader`. Maximum usable protected stream: **8,176 bytes** ([`src/carrier/capacity.py#L73`](file:///E:/Paladin_media/src/carrier/capacity.py#L73)).
  - **Mode B (Lossy Channels):** 0 bits reserved for headers. Maximum usable protected stream: **8,192 bytes**.
- **$1024 \times 1024$ Carrier ($512 \times 512$ subband):**
  - Raw Capacity: $512 \times 512 = 262,144\text{ bits}$ ($32,768\text{ bytes}$).
  - **Mode A (Lossless PNG):** 128 bits (16 bytes) reserved for `CarrierHeader`. Maximum usable protected stream: **32,752 bytes** ([`src/carrier/capacity.py#L94`](file:///E:/Paladin_media/src/carrier/capacity.py#L94)).
  - **Mode B (Lossy Channels):** 0 bits reserved for headers. Maximum usable protected stream: **32,768 bytes**.

#### Exact Derived Headroom Table

| Tier Identifier | Target Plaintext ($L_{\text{plain}}$) | Protected Stream ($L_{\text{prot}}$) | $512^2$ Mode A Headroom ($8,176\text{ B}$) | $512^2$ Mode B Headroom ($8,192\text{ B}$) | $1024^2$ Mode A Headroom ($32,752\text{ B}$) | $1024^2$ Mode B Headroom ($32,768\text{ B}$) | Verification Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tier 1 ($T_1$)** | $1,500\text{ B}$ | **$1,774\text{ B}$** ($14,192\text{ b}$) | $+6,402\text{ B}$ ($78.30\%$) | $+6,418\text{ B}$ ($78.34\%$) | $+30,978\text{ B}$ ($94.58\%$) | $+30,994\text{ B}$ ($94.59\%$) | Fully Derived & Verified |
| **Tier 2 ($T_2$)** | $7,000\text{ B}$ | **$8,074\text{ B}$** ($64,592\text{ b}$) | **$+102\text{ B}$** ($1.25\%$) | **$+118\text{ B}$** ($1.44\%$) | $+24,678\text{ B}$ ($75.35\%$) | $+24,694\text{ B}$ ($75.36\%$) | Fully Derived & Verified |
| **Tier 3 ($T_3$)** | $20,000\text{ B}$ | **$22,930\text{ B}$** ($183,440\text{ b}$) | *Exceeds Capacity* | *Exceeds Capacity* | **$+9,822\text{ B}$** ($29.99\%$) | **$+9,838\text{ B}$** ($30.02\%$) | Fully Derived & Verified |
| **$T_{\max512}$** | $7,090\text{ B}$ | **$8,164\text{ B}$** ($65,312\text{ b}$) | **$+12\text{ B}$** ($0.15\%$) | **$+28\text{ B}$** ($0.34\%$) | $+24,588\text{ B}$ ($75.07\%$) | $+24,604\text{ B}$ ($75.08\%$) | Fully Derived & Verified |
| **$T_{\max1024}$** | $28,574\text{ B}$ | **$32,752\text{ B}$** ($262,016\text{ b}$) | *Exceeds Capacity* | *Exceeds Capacity* | **$+0\text{ B}$** ($0.00\%$) | **$+16\text{ B}$** ($0.05\%$) | Fully Derived & Verified |

---

### 3.3 Non-Zero Metadata Cost Sensitivity

Metadata is **not zero-cost**. In `SecretSerializer.serialize()`, non-empty metadata is serialized to compact JSON of length $M$ bytes:
- Every byte of metadata increases $L_{\text{ser}}$ by 1 byte and $L_{\text{enc}}$ by 1 byte.
- Furthermore, if adding $M$ bytes pushes $L_{\text{enc}}$ across an RS chunk boundary ($L_{\text{enc}} \pmod{223} == 0$), it incurs an immediate **32-byte parity penalty** ($N_{\text{blocks}} \to N_{\text{blocks}} + 1$).
- **Vulnerability Example on Tier 2 ($512 \times 512$ Carrier):**
  - With $M=0$, $L_{\text{prot}} = 8,074\text{ B}$, leaving a narrow headroom of $102\text{ bytes}$ in Mode A.
  - If a caller attaches metadata with $M \ge 90\text{ bytes}$, $L_{\text{enc}}$ increases from $7,046$ to $\ge 7,136$, requiring a 33rd RS block (+32B parity). The resulting protected stream reaches $\ge 8,198\text{ bytes}$, exceeding the $8,176\text{ B}$ carrier ceiling and aborting with `PayloadCapacityExceededError`.
- **Contract Rule:** Tiers $T_1, T_2, T_3$ are strictly specified with $M = 0$ (or a fixed metadata budget $\le 32\text{ B}$ explicitly deducted from the payload data budget).

---

### 3.4 Mode A vs. Mode B Mask Alignment

1. **Mode A (Existing Code):**
   - In [`src/carrier/mask.py#L97`](file:///E:/Paladin_media/src/carrier/mask.py#L97), `candidate_range = range(cls.HEADER_BITS, total_positions)`.
   - Indices $0 \dots 127$ are reserved exclusively for the 16-byte carrier header.
   - Fully matches existing `BaselineEmbedder` and `Extractor`.
2. **Mode B (Proposed Architecture):**
   - Zero bits reserved for headers.
   - `candidate_range = range(0, total_positions)`.
   - Requires introducing a parameter `reserved_header_bits: int = 0` to `DeterministicMaskGenerator.generate_mask()` during Phase 3 implementation.

---

## 4. Check 3 — Receiver Protocol Specification & Ambiguity Resolution

### 4.1 Normative Receiver Protocol Workflow

```
Receiver Execution Flow (Mode B Container Ingestion):
Container Image (H x W)
         │
         ▼
[Session Metadata Present?]
   ├── YES ──► Protocol 1: Deterministic Session-Agreed Tier
   │                 │
   │                 ├── 1. Read agreed tier T in {T1, T2, T3, T_max}
   │                 ├── 2. Extract L_extract(T) bits from mask indices 0 .. L-1
   │                 ├── 3. Reed-Solomon decode (up to t=16 symbol corrections/block)
   │                 ├── 4. AES-256-GCM decrypt with AAD(T, W, H)
   │                 └── 5. SecretSerializer deserialize (Magic, CRC-32)
   │
   └── NO ───► Protocol 2: Autonomous Trial Extraction Fallback
                     │
                     ├── For tier in [T1, T2, T3, T_max]:
                     │     ├── Extract L_extract(tier) bits
                     │     ├── Attempt RS decode (catch ECCDecodeError)
                     │     ├── Attempt AES-GCM decrypt with AAD(tier, W, H) (catch AuthenticationError)
                     │     └── If authentic: Deserializer parse -> RETURN PAYLOAD
                     │
                     └── All candidates failed: Raise AuthenticationError (Fail-Closed)
```

1. **Protocol 1 (Session-Agreed Framing - Primary):**
   - Sender and receiver agree out-of-band on tier index $T$.
   - Exactly one extraction and decryption attempt is executed.
   - If corrupted, `AuthenticationError` terminates execution with zero plaintext emission.
2. **Protocol 2 (Trial Extraction - Fallback):**
   - Invoked only when container arrives without session tier context.
   - Candidate order is strictly ascending by size: $T_1 \to T_2 \to T_3 \to T_{\max}$.
   - For each candidate tier:
     - Extraction length $L_{\text{extract}}$ is computed from the derived tier table.
     - Candidate AAD is constructed: `struct.pack(">2sBBHH", b"PM", 1, tier_id, W, H)`.
     - Reed–Solomon decoding is attempted. If `ECCDecodeError` is raised, candidate fails; loop advances.
     - If RS succeeds, AES-256-GCM verification is evaluated. If `AuthenticationError` is raised, candidate fails; loop advances.
     - If both succeed, `SecretSerializer.deserialize()` verifies CRC-32 and emits the payload.
   - If all candidates fail, execution terminates with terminal `AuthenticationError`.

---

### 4.2 Reiteration: RS Decoding Failure Is Not Guaranteed on Wrong Tiers

As established in Check 1, Reed–Solomon decoding can occasionally succeed when given random or misaligned data if:
- The first 4 bytes of random noise happen to decode to a length integer matching the buffer length.
- The trailing carrier noise happens to satisfy the systematic parity syndromes.

Therefore, the protocol **never assumes RS decoding will fail**. The security boundary is strictly anchored by **AES-256-GCM GHASH verification**, which guarantees rejection of invalid candidates with probability $1 - 2^{-128}$.

---

### 4.3 Backward Compatibility Reconciliation

- **Mode A (Phase 2 Baseline):** Unchanged. The classical pipeline (`MediaStegoPipeline`, `BaselineEmbedder`, `Extractor`, `CarrierHeader`) continues to operate at 100% fidelity on lossless PNG images with in-band headers.
- **Mode B (Phase 3 Neural Engine):** Fully encapsulated in new Phase 3 classes. No existing Phase 2 methods are deprecated or overwritten.

---

## 5. Check 4 — Validation Readiness & Empirical Testing Status

### 5.1 Audit of Proposed Validation Scripts

The proposed scripts listed in prior reports were audited against the physical filesystem:

| Proposed Script / Test Path | Physical Existence | Execution Status | Audit Finding |
|:---|:---:|:---:|:---|
| `scripts/experiments/test_implicit_framing_dryrun.py` | **DOES NOT EXIST** (`False`) | Not Run | Correctly reported as uncreated. Must not be claimed as passing. |
| `tests/integration/test_wrong_tier_fail_closed.py` | **DOES NOT EXIST** (`False`) | Not Run | Correctly reported as uncreated. Must not be claimed as passing. |
| `scripts/experiments/verify_spatial_rounding_margin.py` | **DOES NOT EXIST** (`False`) | Not Run | Correctly reported as uncreated. Must not be claimed as passing. |

In strict adherence to directives, **these scripts were not created in this pass**. They are cataloged below as the exact specification for implementation validation tests.

---

### 5.2 Required Implementation Acceptance Tests (To Be Authored in Phase 3)

When Phase 3 implementation is authorized, the following minimal unit and integration tests must be authored to validate the contract:
1. `test_mode_b_wrong_tier_rejection()`:
   - Embed carrier using Tier $T_2$.
   - Attempt extraction using candidate Tier $T_1$ (under-reading) and Tier $T_3$ (over-reading).
   - Assert terminal `AuthenticationError` is raised and zero plaintext is returned.
2. `test_mode_b_aad_tamper_rejection()`:
   - Embed carrier using Tier $T_1$ with canonical AAD.
   - Attempt extraction with mutated AAD (e.g., mismatched dimensions or altered tier ID).
   - Assert terminal `AuthenticationError` is raised.
3. `test_mode_b_malformed_rs_length_rejection()`:
   - Extract raw bits, mutate the 4-byte RS length prefix to an impossible value, and decode.
   - Assert `ECCDecodeError` or `AuthenticationError` terminates execution fail-closed.
4. `test_mode_b_spatial_rounding_and_clamping_reversibility()`:
   - Execute forward neural modulation with $\gamma_{\text{min}} = 1.0$.
   - Execute IDWT, spatial integer rounding (`round(clamp(I', 0, 255))`), and lossless PNG serialization/reload.
   - Recompute Haar DWT and measure pre-ECC BER. Assert $\text{BER}_{\text{pre-ECC}} \le 1.0 \times 10^{-3}$.

---

### 5.3 Classification of Evidence: Verified Facts vs. Hypotheses

| Statement | Classification | Empirical Basis / Verification Method |
|:---|:---:|:---|
| **E14 Classical Header Parity Failure** | **VERIFIED FACT** | 140/140 JPEG runs and 60/60 Gaussian noise runs failed at header parsing with parity $\text{BER} \approx 48\%$ in `docs/experiments/e14_results.json`. |
| **AES-GCM Fail-Closed Security** | **VERIFIED FACT** | 103/103 tests passing in `tests/`, specifically `tests/integration/test_fail_closed.py`. |
| **Continuous Domain Reversibility** | **MATHEMATICAL FACT** | Exact algebraic cancellation in unquantized float64 arithmetic: $\Delta_{\text{rec}} = \gamma \odot P$ with $\gamma > 0$. |
| **GTX 1650 Hardware Compatibility** | **ENVIRONMENT FACT** | Turing TU117, Compute Capability 7.5, 4 GB VRAM verified via `nvidia-smi`. No packages installed in this pass. |
| **PNG Rounding / Saturation Margins** | **UNTESTED HYPOTHESIS** | Exact bit recovery after spatial rounding and dynamic range clamping is unproven; requires empirical test script. |
| **ACS-Sub-INN Lossy JPEG Robustness** | **UNTESTED HYPOTHESIS** | Pre-ECC $\text{BER} \le 3.0\%$ at JPEG $Q=80$ is the central research hypothesis of Phase 3; requires training and empirical testing. |

---

## 6. Exact Decisions Requiring User Approval

The following specific decisions require explicit user authorization:

1. **Authorization to Freeze the Phase 3 Architecture and Contract Specification:**
   - Freeze Subband Option A ($\text{HH}_{\text{keep}} = [\text{Green}, \text{Blue}]$, $\text{HH}_{\text{drop}} = [\text{Red}]$).
   - Freeze the Dual-Mode Framing Contract (Mode A retaining the 16B header for lossless PNG; Mode B using session-agreed tiers with trial extraction fallback).
   - Freeze the Derived Tier Size Catalog ($T_1=1,774\text{ B}$, $T_2=8,074\text{ B}$, $T_3=22,930\text{ B}$, $T_{\max512}=8,164\text{ B}$, $T_{\max1024}=32,752\text{ B}$).
   - Freeze the Compact ResNet Topology ($\approx 92\text{K}$ parameters, $\gamma_{\text{min}}=1.0$, peak VRAM $< 1.2\text{ GB}$).

2. **Authorization to Execute Environment Setup (Pending Freeze):**
   - Execute `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` to equip the Python 3.10.11 environment with official PyTorch wheels for the GTX 1650.
