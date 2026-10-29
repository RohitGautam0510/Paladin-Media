# PaladinMedia — Phase 3 Sprint 3.3.2 Audit Report
# Production Integration & ECC Framing Remediation Assessment

**Sprint:** 3.3.2
**Sprint Type:** Read-only Architecture & Remediation Assessment
**Repository:** `E:\Paladin_media`
**Authorization Boundary:** Investigation, source inspection, test inspection, and documentation only. **No implementation.**
**Status:** Complete
**Date:** 2026-10-20
**Runtime:** Gravitas Core Engineering Reliability Runtime

---

## 1. Executive Verdict & Key Architectural Conclusions

### 1.1 Executive Verdict: ASSESSMENT COMPLETE
Sprint 3.3.2 investigated the two architectural blockers identified during Sprint 3.3.1:
1. **Production Carrier-Path Incompatibility:** The Phase 2 carrier components ([`BaselineEmbedder`](file:///e:/Paladin_media/src/carrier/embedding.py#L57), [`Extractor`](file:///e:/Paladin_media/src/carrier/extraction.py#L26), and [`MediaStegoPipeline`](file:///e:/Paladin_media/src/core/media_pipeline.py#L101)) are tightly coupled to classical spatial integer-parity modulation $\text{bit} = (a - b - c + d) \bmod 2$. They cannot process continuous Haar wavelet coefficient perturbations from [`ACSSubINN`](file:///e:/Paladin_media/src/neural/model.py#L20) v3 / [`NeuralSpatialBridge`](file:///e:/Paladin_media/src/neural/spatial_bridge.py#L271).
2. **ECC Length-Prefix Vulnerability:** [`ReedSolomonEncoder.encode()`](file:///e:/Paladin_media/src/security/ecc/encoder.py#L151) prepends a 4-byte big-endian length prefix outside the Reed-Solomon codewords. This prefix is completely unprotected by error correction. A single bit flip in these 32 bits causes an immediate length mismatch error ([`ECCDecodeError`](file:///e:/Paladin_media/src/security/ecc/encoder.py#L265)) during decoding, completely bypassing the $t=16$ codeword error correction capability.

### 1.2 Key Architectural Conclusions
1. **Carrier Path Resolution:** **Option A (Shared Modulation Contract via Dependency Injection)** is the recommended path. By defining a narrow `CarrierModulationEngine` abstraction (with `ParityModulationEngine` as the backward-compatible default and `NeuralModulationEngine` as the Phase 3 neural implementation), [`MediaStegoPipeline`](file:///e:/Paladin_media/src/core/media_pipeline.py#L101) retains sole ownership of wire framing ([`CarrierHeader`](file:///e:/Paladin_media/src/carrier/header.py#L37)), spatial permutation ([`DeterministicMaskGenerator`](file:///e:/Paladin_media/src/carrier/mask.py#L42)), capacity gating ([`CapacityCalculator`](file:///e:/Paladin_media/src/carrier/capacity.py#L66)), and PNG clamping, while avoiding logic duplication.
2. **ECC Framing Resolution:** **Alternative 2 (Length Derivation from Valid Encoded Frame / Carrier Framing)** provides a robust, zero-breaking-change remediation. In systematic chunked RS(255, 223, 32), the payload data length $L$ is **100% mathematically determined** by the total encoded stream length $L_{\text{enc}}$. When `raw_enc[:4]` suffers bit errors during carrier transmission, the decoder can derive the true $L$ from $L_{\text{enc}}$ (which is already securely communicated in [`CarrierHeader.payload_length`](file:///e:/Paladin_media/src/carrier/header.py#L41)), eliminating the single-point-of-failure without breaking existing serialized Phase 1 payloads.
3. **Sprint 3.3.1 Evidence Quality Finding:** The Sprint 3.3.1 experiment runner claimed deterministic seed controls (`seed=42`), but used `os.urandom(plen)` ([line 157](file:///e:/Paladin_media/scripts/experiments/run_sprint3_3_1_production_carrier_path.py#L157)) to generate payload bytes. The failure in Trial 16 was genuine and correctly diagnosed, but payload bytes varied across script runs.

---

## 2. Verified Current Architecture and Call Paths

### 2.1 Reconstructed Call Chains

#### Path 1: Classical Parity Production Route (`MediaStegoPipeline`)
```
Sender Path:
SecretPayload -> SecretSerializer.serialize() -> AESGCMProvider.encrypt()
  -> ReedSolomonEncoder.encode() -> protected_bytes (len <= 8176 B)
  -> CapacityCalculator.check_capacity() (Hard Gate)
  -> CarrierHeader.pack() (16 B: b'PLDH' || session_counter || payload_length)
  -> DeterministicMaskGenerator.generate_mask() (128 anchor + N PRNG coordinates)
  -> BaselineEmbedder.embed():
       * Computes 2D Haar DWT
       * In-place pixel adjustment: stego[2r, 2c, 0] += 1 (or -1) if (a - b - c + d) % 2 != bit
       * IDWT back to 8-bit spatial RGB
  -> PIL Image.save(format="PNG") -> stego_png_bytes

Receiver Path:
stego_png_bytes -> PIL Image.open() -> 8-bit RGB array
  -> Extractor.extract():
       * Reads 128 anchor pixels: bit = (a - b - c + d) % 2
       * CarrierHeader.unpack() (validates b'PLDH', extracts session_counter & payload_length)
       * DeterministicMaskGenerator.generate_payload_positions()
       * Reads payload pixels: bit = (a - b - c + d) % 2 -> protected_bytes
  -> ReedSolomonEncoder.decode() (unpacks 4B prefix, decodes blocks, repairs errors)
  -> AESGCMProvider.decrypt() (verifies 16B GHASH tag)
  -> SecretSerializer.deserialize() -> SecretPayload
```

#### Path 2: Neural Wavelet Route (`NeuralSpatialBridge` / `ACSSubINN` v3)
```
Sender Path:
2D Spatial Carrier Tensor (1, 3, H, W)
  -> NeuralSpatialBridge.embed_spatial():
       * Differentiable Haar DWT 2D: LL, LH, HL, HH
       * Subband Partition: Red-HH (1, 1, H/2, W/2), Green/Blue-HH (1, 2, H/2, W/2)
       * ACSSubINN.embed_coefficients():
           gamma(x) >= 1.0 (clamped)
           HH_drop' = HH_drop * gamma(x) + delta(payload_bipolar) * mask
       * Reassemble HH' = [HH_drop', HH_keep]
       * Differentiable Haar IDWT 2D
       * Quantization: clamp to [0, 255], round to uint8
  -> save_and_reload_png() -> reloaded_stego (1, 3, H, W) uint8

Receiver Path:
reloaded_stego
  -> NeuralSpatialBridge.extract_spatial():
       * Differentiable Haar DWT 2D
       * Subband Partition: Red-HH received, Green/Blue-HH received
       * ACSSubINN.extract_coefficients():
           context_rec = ContextPrediction(Green/Blue-HH)
           delta_rec = (Red-HH - context_rec) / gamma_rec
           extracted_binary = (delta_rec > 0).float()
```

#### Path 3: The Divergence Point and Failure Mechanism
In the classical path, [`BaselineEmbedder.embed`](file:///e:/Paladin_media/src/carrier/embedding.py#L146-L168) explicitly adjusts the spatial integer parity of $2 \times 2$ pixel blocks on channel 0:
$$\text{parity} = (a - b - c + d) \bmod 2$$
In the neural path, [`NeuralSpatialBridge`](file:///e:/Paladin_media/src/neural/spatial_bridge.py#L335-L347) modulates the real-valued wavelet coefficient:
$$HH = \frac{a - b - c + d}{2.0}$$
via continuous affine scaling and translation: $HH' = \gamma \cdot HH + \text{delta}$. After inverse DWT and integer rounding/clamping, the continuous modulation does **not** guarantee that $(a - b - c + d) \bmod 2$ matches the embedded bit.

When a neural stego image is passed to [`Extractor.extract()`](file:///e:/Paladin_media/src/carrier/extraction.py#L41):
1. [`Extractor.extract`](file:///e:/Paladin_media/src/carrier/extraction.py#L75-L84) evaluates $(a - b - c + d) \bmod 2$ over the 128 anchor positions $[(0,0)..(0,127)]$.
2. Because integer parity was never modulated, the extracted bits evaluate to arbitrary values (typically all 0s in smooth mid-gray carriers).
3. The resulting 16 bytes evaluate to `b'\x00\x00\x00\x00...'` instead of `b'PLDH...'`.
4. [`CarrierHeader.unpack()`](file:///e:/Paladin_media/src/carrier/header.py#L87-L90) raises:
   ```
   CarrierHeaderError: Invalid carrier header magic: expected b'PLDH', got b'\x00\x00\x00\x00'
   ```
5. [`MediaStegoPipeline.extract_and_unprotect()`](file:///e:/Paladin_media/src/core/media_pipeline.py#L205) statically calls `Extractor.extract()` and fails with the identical exception.

---

## 3. Evidence-Backed Root-Cause Analysis

### 3.1 Defect 1: Hardcoded Classical Parity Coupling in Carrier Subsystem
- **Source Lines:**
  - [`src/carrier/embedding.py`, lines 146–168](file:///e:/Paladin_media/src/carrier/embedding.py#L146-L168)
  - [`src/carrier/extraction.py`, lines 75–84, 102–109](file:///e:/Paladin_media/src/carrier/extraction.py#L75-L84)
  - [`src/core/media_pipeline.py`, lines 151–155, 205](file:///e:/Paladin_media/src/core/media_pipeline.py#L151-L155)
- **Root Cause:**
  `MediaStegoPipeline` lacks an abstraction layer for carrier modulation. It directly imports and invokes `BaselineEmbedder` and `Extractor`, which are tightly coupled to spatial integer-parity arithmetic. No mechanism exists to inject alternative modulation engines, preventing `NeuralSpatialBridge` from participating in the production pipeline orchestrator.

### 3.2 Defect 2: Unprotected 4-Byte Length Prefix in Reed-Solomon Codec
- **Source Lines:**
  - [`src/security/ecc/encoder.py`, line 178](file:///e:/Paladin_media/src/security/ecc/encoder.py#L178): `header = struct.pack(">I", length)`
  - [`src/security/ecc/encoder.py`, line 191](file:///e:/Paladin_media/src/security/ecc/encoder.py#L191): `return header + b"".join(blocks)`
  - [`src/security/ecc/encoder.py`, lines 233, 260–268](file:///e:/Paladin_media/src/security/ecc/encoder.py#L233)
- **Root Cause:**
  `ReedSolomonEncoder.encode()` prepends the 4-byte `length` as raw, unencoded bytes before the first RS block. During transmission through a physical or lossy channel (or high-capacity neural embedding near clipping limits), if even **1 bit** in those 32 bits flips:
  1. `struct.unpack(">I", raw_enc[:4])` yields a wildly inaccurate `total_length`.
  2. The expected length check `if len(raw_enc) != expected_len:` fails immediately.
  3. The codec raises `ECCDecodeError: Encoded data length mismatch` before attempting any RS decoding.
  4. The failure occurs even when every individual RS codeword has error count $e \le 3$, far below the correction capacity $t = 16$.

---

## 4. Evaluation of Modulation Integration Alternatives

| Evaluation Dimension | Option A: Shared Modulation Contract | Option B: Separate Neural Pipeline | Option C: Engine Strategy Injection |
| :--- | :--- | :--- | :--- |
| **Architectural Concept** | Abstract `CarrierModulationEngine` base class in `src/carrier/`. Parity and Neural implementations injected into `MediaStegoPipeline`. | Leave `MediaStegoPipeline` untouched. Create new `NeuralMediaStegoPipeline` in `src/neural/`. | Add optional callable/protocol arguments (`carrier_embedder`, `carrier_extractor`) to `MediaStegoPipeline`. |
| **Files Modified in Future Sprint** | `src/carrier/base.py` (new), `src/carrier/embedding.py`, `src/carrier/extraction.py`, `src/core/media_pipeline.py` | `src/neural/pipeline.py` (new), `tests/integration/test_neural_media_pipeline.py` (new) | `src/core/media_pipeline.py`, `src/neural/carrier_adapter.py` (new) |
| **Public APIs Changed** | `MediaStegoPipeline(..., modulation_engine=...)` | None in Phase 2. New class `NeuralMediaStegoPipeline`. | `MediaStegoPipeline(..., embedder=..., extractor=...)` |
| **Phase 2 Compatibility** | 100% backward-compatible (default engine is `ParityModulationEngine`). | 100% backward-compatible (Phase 2 files completely untouched). | 100% backward-compatible (default arguments point to `BaselineEmbedder`/`Extractor`). |
| **Logic Duplication** | **Zero.** Header framing, mask generation, capacity checks, PNG handling shared. | **High.** Framing, PNG conversion, metrics, and unprotect workflows duplicated. | **Minimal.** Framing stays in pipeline; embed/extract delegated. |
| **Header & Mask Ownership** | Owned strictly by `MediaStegoPipeline`. Modulation engine receives bits and coordinates. | Owned by `NeuralMediaStegoPipeline`. Direct calls to `CarrierHeader` and `MaskGenerator`. | Owned by `MediaStegoPipeline` or adapter. |
| **Capacity & Fail-Closed** | Unified in `MediaStegoPipeline` via `CapacityCalculator`. | Handled in `NeuralMediaStegoPipeline`. | Unified in `MediaStegoPipeline`. |
| **Implementation Risk** | Low-Medium (data conversions: `np.ndarray` $\leftrightarrow$ `torch.Tensor`). | Low (zero blast radius on existing files). | Low (minimal edits to existing pipeline). |

### Detailed Assessment of Option A (Recommended)
- **Contract Definition:**
  ```python
  class CarrierModulationEngine(ABC):
      @abstractmethod
      def embed(self, carrier: np.ndarray, header_bits: np.ndarray, payload_bits: np.ndarray, mask: EmbeddingMask) -> np.ndarray: ...
      @abstractmethod
      def extract(self, carrier: np.ndarray, mask: EmbeddingMask) -> Tuple[np.ndarray, np.ndarray]: ...
  ```
- **Advantages:**
  1. Preserves single-source-of-truth for pipeline security invariants (INV-01 through INV-04).
  2. Ensures carrier header framing, deterministic mask coordinates, and capacity gates cannot diverge between classical and neural modes.
  3. Eliminates code duplication across serialization, PNG encoding, and fidelity metrics.
- **Disadvantages & Mitigations:**
  - Requires clean handling of tensor vs. numpy arrays. Mitigation: `NeuralModulationEngine` accepts `np.ndarray`, converts to `torch.Tensor` internally, executes `NeuralSpatialBridge`, and returns `np.ndarray`.

---

## 5. ECC Framing Alternatives and Compatibility Analysis

### 5.1 The Mathematical Redundancy in Systematic Chunked RS
In PaladinMedia's `ReedSolomonEncoder`, data is chunked into blocks of $k = 223$ bytes with $2t = 32$ parity symbols. Each full block is 255 bytes. A shortened final block with $R$ data bytes ($1 \le R < 223$) has length $R + 32$ bytes ($33 \le R + 32 \le 254$).

Let $L_{\text{enc}} = \text{len}(\text{raw\_enc})$ and $M = L_{\text{enc}} - 4$:
- If $M == 0 \implies L = 0$.
- If $M \pmod{255} == 0 \implies N_{\text{full}} = M // 255, R = 0 \implies L = N_{\text{full}} \times 223$.
- If $M \pmod{255} \ge 33 \implies N_{\text{full}} = M // 255, R = (M \pmod{255}) - 32 \implies L = N_{\text{full}} \times 223 + R$.
- If $1 \le M \pmod{255} \le 32 \implies$ Invalid framing length (a valid RS codeword cannot have $\le 32$ symbols).

**Proof:** Because $33 \le R + 32 < 255$, $M \pmod{255}$ is strictly bijective to $R$. Therefore, **the payload data length $L$ is uniquely and deterministically derived from the total encoded stream length $L_{\text{enc}}$ alone.**

### 5.2 Remediation Alternatives

#### Alternative 1: Protect Length Inside First RS Codeword
- **Mechanism:** Embed `struct.pack(">I", length)` as the first 4 bytes of data inside Block 0.
- **Pros:** Full $t=16$ error correction protects the length.
- **Cons:** **Breaks backward compatibility.** Existing serialized Phase 1 payloads cannot be decoded by the new decoder without format versioning.
- **Verdict:** Viable only if a wire-format breaking change is explicitly authorized.

#### Alternative 2: Derive Length from Valid Encoded Frame / Carrier Header (Recommended)
- **Mechanism:** In `ReedSolomonEncoder.decode_with_result()`:
  1. If `raw_enc[:4]` matches the mathematical length $L_{\text{derived}}$ computed from $L_{\text{enc}}$, proceed as normal.
  2. If `raw_enc[:4]` differs from $L_{\text{derived}}$, but $L_{\text{enc}}$ is valid and matches outer `CarrierHeader.payload_length`, log a warning, use $L_{\text{derived}}$, and proceed to decode codewords.
- **Pros:**
  1. **100% backward-compatible.** Existing valid payloads decode identically.
  2. Zero wire-format changes. No extra bytes, no capacity reduction.
  3. Completely eliminates the length-prefix vulnerability. Bit errors in `raw_enc[:4]` no longer abort decoding.
- **Cons:** Requires `ReedSolomonEncoder` decoder to use the derived length fallback.

#### Alternative 3: Triple-Modular Redundancy (TMR) Length Prefix
- **Mechanism:** Write `length` 3 times (12 bytes) and use majority voting.
- **Pros:** Simple bit-level error tolerance.
- **Cons:** Breaks wire format; consumes 8 additional bytes of carrier capacity.

---

## 6. Audit of Sprint 3.3.1 Evidence Quality

1. **Reconciliation of 15/16 Plaintext Recoveries:**
   - 15 trials achieved verbatim plaintext recovery (`assert recovered == original`) with valid AES-GCM tags.
   - The 1 failed trial (`0005.png` at 7,088 B, 99.83% capacity) was caused entirely by Defect 2 (1 bit error in `protected[:4]`). All payload codewords had $\le 3$ errors ($t=16$).
2. **Capacity and BER Denominator:**
   - 512×512 carrier subband: $256 \times 256 = 65,536$ positions ($8,192$ bytes).
   - Carrier header: 16 bytes $= 128$ bits (fixed anchor coordinates $[(0,0)..(0,127)]$).
   - Protected stream limit: $65,536 - 128 = 65,408$ bits ($8,176$ bytes).
   - `raw_ber` denominator: In `run_sprint3_3_1_production_carrier_path.py` [lines 222–223](file:///e:/Paladin_media/scripts/experiments/run_sprint3_3_1_production_carrier_path.py#L222-L223), `len(payload_bits)` was used ($65,296$ bits). It measures strictly the payload region, excluding the 128 header bits. Header bits had zero errors in all trials.
3. **Purpose of Direct Production Interface Tests:**
   - The tests calling `Extractor.extract` and `MediaStegoPipeline.extract_and_unprotect` in [`test_production_carrier_path.py` lines 582–590](file:///e:/Paladin_media/tests/integration/test_production_carrier_path.py#L582-L590) are negative defect-proof tests (`pytest.raises(CarrierHeaderError)`). They prove the architectural defect; they do not claim compatibility.
4. **Header Corruption Test Scope:**
   - In `test_production_carrier_path.py` [lines 463–477](file:///e:/Paladin_media/tests/integration/test_production_carrier_path.py#L463-L477), container pixels $[0:10, 0:10]$ were corrupted, extracted via `spatial_bridge.extract_spatial()`, and confirmed to fail closed via `CarrierHeaderError`.
5. **Use of `os.urandom()` Discrepancy:**
   - **Identified Gap:** Line 157 of `run_sprint3_3_1_production_carrier_path.py` uses `os.urandom(plen)` instead of a seeded PRNG. While model weights and spatial masks were deterministic, the payload bytes differed between script runs.
   - **Remediation Requirement:** Future benchmarks must derive test payloads deterministically from a fixed seed (e.g. `hashlib.sha256(b"seed" || index)`).
6. **Artifact Preservation:**
   - Historical experiment outputs (`sprint3_3_integration_results.json`, `sprint2_1_ecc_validation_results.json`) remain untouched. Sprint 3.3.1 outputs were written to dedicated paths.

---

## 7. Recommended Remediation Design & Unresolved Decisions

### 7.1 Recommended Architecture (Proposed, Not Approved)
1. **Carrier Path:** Implement **Option A** (`CarrierModulationEngine` contract in `src/carrier/`).
   - Create `src/carrier/base.py` defining `CarrierModulationEngine`.
   - Implement `ParityModulationEngine` wrapping classical Phase 2 logic (default).
   - Implement `NeuralModulationEngine` wrapping `NeuralSpatialBridge`.
   - Update `MediaStegoPipeline.__init__()` to accept `modulation_engine: CarrierModulationEngine = ParityModulationEngine()`.
2. **ECC Framing:** Implement **Alternative 2** (Mathematical Length Derivation Fallback).
   - Update `ReedSolomonEncoder.decode_with_result()` to validate `raw_enc[:4]` against $L_{\text{derived}}(L_{\text{enc}})$. If `raw_enc[:4]` is corrupted but $L_{\text{enc}}$ matches valid framing, use $L_{\text{derived}}$, correct the codeword errors, and decode cleanly.

### 7.2 Explicit Decisions Requiring User Authorization
Before any implementation sprint can begin, the following architectural decisions require explicit approval:
1. **Decision 1: Authorization of Option A vs. Option B.** Does the user approve refactoring `MediaStegoPipeline` via dependency injection (Option A), or is an isolated `NeuralMediaStegoPipeline` (Option B) required to keep Phase 2 code entirely frozen?
2. **Decision 2: Authorization of ECC Length Fallback (Alternative 2).** Does the user approve adding mathematical length derivation fallback to `ReedSolomonEncoder.decode_with_result()`, or should the wire format be modified (Alternative 1) despite breaking Phase 1 backward compatibility?
3. **Decision 3: Scope of Sprint 3.3.3.** Confirm that Sprint 3.3.3 will be authorized strictly for implementing the approved remediation and running the integration gate, with model training remaining unauthorized.

---

## 8. Future Implementation Acceptance Checklist

For the future implementation sprint:
- [ ] **Production Path Roundtrip:** `MediaStegoPipeline.run_full_roundtrip()` executes end-to-end using `NeuralModulationEngine` with 100% exact plaintext recovery across all 16 matrix trials.
- [ ] **Lossless PNG Persistence:** Stego container is saved to disk as an 8-bit RGB PNG and reloaded before extraction in all tests.
- [ ] **Phase 2 Non-Regression:** All existing classical unit and integration tests (`test_carrier_embedding.py`, `test_carrier_extraction.py`, `test_media_pipeline.py`) pass with zero modifications.
- [ ] **ECC Framing Hardening:** Injecting 1 to 4 byte errors into `protected[:4]` does not prevent decoding when codeword errors are within $t=16$.
- [ ] **ECC Error Boundaries:** Codeword errors $\le 16$ symbols are corrected; errors $> 16$ symbols raise `ECCDecodeError` and leak 0 plaintext.
- [ ] **Deterministic Benchmarking:** Replace `os.urandom()` in test runners with deterministic seeded byte generation.
- [ ] **Model Invariance:** Verify parameter SHA-256 digest `fe7462957b0194d109ec32a78834d7bf44b9d1aa86510a40625f2e485fa40369` is unchanged after all tests.
- [ ] **Zero Optimizer Steps:** Confirm 0 training epochs, 0 optimizer steps.

---

## 9. Files Inspected & Line References

| File Inspected | Key Lines | Purpose of Inspection |
| :--- | :--- | :--- |
| [`src/carrier/header.py`](file:///e:/Paladin_media/src/carrier/header.py) | [Lines 37–102](file:///e:/Paladin_media/src/carrier/header.py#L37-L102) | 16-byte fixed header framing, `b'PLDH'` magic, and unpack validation. |
| [`src/carrier/mask.py`](file:///e:/Paladin_media/src/carrier/mask.py) | [Lines 42–109](file:///e:/Paladin_media/src/carrier/mask.py#L42-L109) | Deterministic anchor (0..127) and PRNG payload coordinate generation. |
| [`src/carrier/capacity.py`](file:///e:/Paladin_media/src/carrier/capacity.py) | [Lines 66–140](file:///e:/Paladin_media/src/carrier/capacity.py#L66-L140) | 8,176 B protected capacity limit and shortened block accounting. |
| [`src/carrier/embedding.py`](file:///e:/Paladin_media/src/carrier/embedding.py) | [Lines 57–184](file:///e:/Paladin_media/src/carrier/embedding.py#L57-L184) | In-place spatial integer-parity modulation $((a-b-c+d) \bmod 2)$. |
| [`src/carrier/extraction.py`](file:///e:/Paladin_media/src/carrier/extraction.py) | [Lines 26–113](file:///e:/Paladin_media/src/carrier/extraction.py#L26-L113) | Spatial integer-parity extraction and `CarrierHeader.unpack` failure call site. |
| [`src/core/media_pipeline.py`](file:///e:/Paladin_media/src/core/media_pipeline.py) | [Lines 101–260](file:///e:/Paladin_media/src/core/media_pipeline.py#L101-L260) | Direct static invocation of `BaselineEmbedder` and `Extractor`. |
| [`src/neural/spatial_bridge.py`](file:///e:/Paladin_media/src/neural/spatial_bridge.py) | [Lines 271–450](file:///e:/Paladin_media/src/neural/spatial_bridge.py#L271-L450) | Wavelet-domain embedding/extraction and PNG save/reload implementation. |
| [`src/security/ecc/encoder.py`](file:///e:/Paladin_media/src/security/ecc/encoder.py) | [Lines 151–320](file:///e:/Paladin_media/src/security/ecc/encoder.py#L151-L320) | 4-byte unencoded length prefix and length-mismatch exception logic. |
| [`src/security/crypto/provider.py`](file:///e:/Paladin_media/src/security/crypto/provider.py) | [Lines 35–80](file:///e:/Paladin_media/src/security/crypto/provider.py#L35-L80) | AES-256-GCM wire packaging (12B nonce + ciphertext + 16B tag). |
| [`src/core/serialization/serializer.py`](file:///e:/Paladin_media/src/core/serialization/serializer.py) | [Lines 35–100](file:///e:/Paladin_media/src/core/serialization/serializer.py#L35-L100) | `SecretPayload` definition and canonical `b'PLDS'` wire format. |
| [`tests/integration/test_production_carrier_path.py`](file:///e:/Paladin_media/tests/integration/test_production_carrier_path.py) | [Lines 1–609](file:///e:/Paladin_media/tests/integration/test_production_carrier_path.py#L1-L609) | Full test suite verifying framing integration, boundaries, and incompatibility. |
| [`scripts/experiments/run_sprint3_3_1_production_carrier_path.py`](file:///e:/Paladin_media/scripts/experiments/run_sprint3_3_1_production_carrier_path.py) | [Lines 140–280](file:///e:/Paladin_media/scripts/experiments/run_sprint3_3_1_production_carrier_path.py#L140-L280) | Experiment runner implementation, `os.urandom` usage, and BER denominator. |

---

## 10. Scope Compliance Declaration

In strict compliance with the Sprint 3.3.2 prompt and repository governance:
- **No Implementation:** Zero production or test code files were modified.
- **No Training:** Zero optimizer steps applied, zero training epochs run, zero loss weights altered.
- **Frozen Architecture:** ACS-Sub-INN v3 design (`gamma_min=1.0`, transform-domain Red-HH target, Green/Blue-HH context) remains completely intact.
- **No Package Changes:** Zero packages installed, updated, or removed.
- **No Dataset Changes:** DIV2K dataset images and `docs/project/div2k_hr_split.json` were not altered.
- **Artifact Preservation:** All historical reports and experiment outputs were strictly preserved.
- **Only New File Created:** `docs/audits/PHASE3_SPRINT3_3_2_REMEDIATION_ASSESSMENT.md`.

```
TRAINING STATUS: NOT AUTHORIZED — Sprint 3.3.2 implements assessment and audit only.
```
