# Experiment E09: Multi-Carrier Partitioning Protocol (MCPP)

## Document Control
- **Experiment ID:** E09
- **Phase:** Phase 2 — Signal Processing + Carrier Infrastructure
- **Date:** 2026-10-09
- **Status:** COMPLETED & VERIFIED
- **Related Specifications:** PaladinMedia Architecture Specification §8/§10/§14, IRS §6
- **Evidence Path:** `docs/experiments/e09_results.json`

---

## 1. Objective & Hypothesis
* **Objective:** Design and experimentally evaluate a Multi-Carrier Partitioning Protocol (MCPP) to transmit large protected payloads that exceed the single-carrier capacity boundary ($8,176\text{ B}$) across a set of $K$ independent carrier images.
* **Hypothesis:** By embedding canonical partition indexing and session binding into the existing 16-byte `CarrierHeader` wire framing, arbitrary-sized protected payloads can be sliced and distributed across $K$ carriers, reassembled in any presentation order by the receiver, and recovered with zero bit errors. Missing partitions or tampered partitions will strictly fail closed before leaking plaintext.

---

## 2. Multi-Carrier Protocol Wire Specification
The canonical 16-byte `CarrierHeader` framing natively supports multi-carrier partitioning without breaking single-carrier compatibility:
1. **Partition Flag:** Bit 0 of `flags` set to `0x01` (`FLAG_PARTITIONED`). Single-carrier images have `flags = 0x00`.
2. **Partition Index & Total Count:** Encoded within the 2-byte big-endian `reserved` field:
   $$\text{reserved} = (\text{partition\_index} \ll 8) \mid K$$
   where $\text{partition\_index} \in [0 \dots K-1]$ and $K \in [2 \dots 255]$ is the total partition count.
3. **Session Binding:** All $K$ carriers carry the identical 32-bit `session_counter`.
4. **Local Payload Length:** The 4-byte `payload_length` specifies the exact byte size of the slice embedded in that specific carrier.

---

## 3. Protocol Invariants & Reassembly Rules
1. **Carrier Set Grouping:** Receiver groups scanned images by `session_counter`.
2. **Completeness Gate:** Receiver validates that all partition indices $i \in [0 \dots K-1]$ are present.
   - If any partition is omitted: raises `IncompleteCarrierSetError` and halts.
3. **Reordering Resilience:** Slices are sorted strictly by `partition_index` before concatenation, allowing arbitrary input order (e.g. $[C_2, C_0, C_1]$).
4. **End-to-End Cryptographic Authentication:**
   - The concatenated stream is decoded by Reed-Solomon.
   - The decoded stream is authenticated and decrypted by AES-256-GCM.
   - If any slice has been tampered with or substituted, the AES-GCM GHASH tag fails authentication and rejects the entire payload fail-closed.

---

## 4. Empirical Validation (15,000 Byte Secret across 3 Carriers)

Evaluated with a $15,000$-byte secret image payload requiring a $17,280$-byte Reed-Solomon protected stream ($> 2.1\times$ the single-carrier ceiling of $8,176\text{ B}$), distributed across 3 carriers (`0001x2.png`, `0002x2.png`, `0003x2.png`):

| Test Scenario | Input Presentation | Observed Receiver Behavior | Cryptographic Integrity | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Scenario A: Sequential** | In-order $[C_0, C_1, C_2]$ | Full reassembly, RS decode, AES decrypt | GHASH Authenticated | **PASS (100% Byte Match)** |
| **Scenario B: Out-of-Order** | Shuffled $[C_2, C_0, C_1]$ | Automatic partition re-sorting | GHASH Authenticated | **PASS (100% Byte Match)** |
| **Scenario C: Incomplete Set** | Missing partition $[C_0, C_2]$ | Halts at partition validator | Zero plaintext leaked | **PASS (Fail-Closed)** |
| **Scenario D: Tampered Carrier** | Corrupted partition $C_1$ | Halts at RS/AES authentication | Zero plaintext leaked | **PASS (Fail-Closed)** |

---

## 5. Image Fidelity Across Partition Set

With $5,760$ bytes embedded per carrier (high density, $70.4\%$ of carrier capacity):

| Carrier Image | Embedded Slice | PSNR (dB) | SSIM | Exact Recovery |
| :--- | :--- | :--- | :--- | :--- |
| **Carrier 0 (`0001x2.png`)** | 5,760 bytes | 63.4705 dB | 0.999994 | TRUE |
| **Carrier 1 (`0002x2.png`)** | 5,760 bytes | 63.4406 dB | 0.999993 | TRUE |
| **Carrier 2 (`0003x2.png`)** | 5,760 bytes | 63.4624 dB | 0.999995 | TRUE |

---

## 6. Architectural Status
* **Status:** Fully validated protocol extension.
* **Separation of Concerns:** MCPP operates as an outer transport framing layer without modifying Phase 1 cryptography or underlying Phase 2 single-carrier DWT transform functions.
* **Recommendation:** MCPP should be integrated into the standard carrier transport layer whenever payloads exceed single-carrier capacity limits.
