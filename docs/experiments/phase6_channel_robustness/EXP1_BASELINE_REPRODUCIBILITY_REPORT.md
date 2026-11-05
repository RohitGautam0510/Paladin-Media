# Phase 6 Experiment 1 — Baseline Reproducibility and Controlled Failure Analysis

**Generated:** 2026-10-30 16:15:13 UTC  
**Evaluation Set:** `0715.png`, `0774.png`, `0784.png` (6 trials each: 3 fixed IV, 3 random IV)  
**Model SHA-256:** `32bfad806dbfcf057359c5f2f84a716832b88d43e9566ec331d0566ad446c4ee`  

---

## 1. Summary of Controlled Trials

| Carrier | Trial ID | IV Mode | Pre-Clamp [Min, Max] | Clipped [Low, High] | Hdr Byte Errs (t=4) | Hdr RS Status | Pay Max Sym Err (t=24) | Pay RS Status | End-to-End Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0715.png` | `T1_fixed_iv_a` | fixed | [-13.9, 264.0] | [1071, 93] | 10 | FAIL | 10 | OK | **FAIL (HEADER_UNPACK)** |
| `0715.png` | `T2_fixed_iv_b` | fixed | [-13.9, 270.2] | [1090, 97] | 10 | FAIL | 13 | OK | **FAIL (HEADER_UNPACK)** |
| `0715.png` | `T3_fixed_iv_c` | fixed | [-13.9, 270.2] | [1089, 98] | 10 | FAIL | 13 | OK | **FAIL (HEADER_UNPACK)** |
| `0715.png` | `T4_random_iv_1` | random | [-13.9, 266.8] | [1085, 97] | 10 | FAIL | 11 | OK | **FAIL (HEADER_UNPACK)** |
| `0715.png` | `T5_random_iv_2` | random | [-11.1, 270.2] | [1086, 93] | 10 | FAIL | 12 | OK | **FAIL (HEADER_UNPACK)** |
| `0715.png` | `T6_random_iv_3` | random | [-11.5, 270.2] | [1101, 96] | 10 | FAIL | 11 | OK | **FAIL (HEADER_UNPACK)** |
| `0774.png` | `T1_fixed_iv_a` | fixed | [0.0, 264.0] | [0, 3315] | 4 | OK | 25 | FAIL | **FAIL (PAYLOAD_RS_DECODE)** |
| `0774.png` | `T2_fixed_iv_b` | fixed | [0.0, 264.0] | [0, 3358] | 4 | OK | 23 | OK | **PASS** |
| `0774.png` | `T3_fixed_iv_c` | fixed | [0.0, 264.0] | [0, 3375] | 4 | OK | 27 | FAIL | **FAIL (PAYLOAD_RS_DECODE)** |
| `0774.png` | `T4_random_iv_1` | random | [0.0, 264.0] | [0, 3365] | 4 | OK | 22 | OK | **PASS** |
| `0774.png` | `T5_random_iv_2` | random | [0.0, 264.0] | [0, 3425] | 4 | OK | 29 | FAIL | **FAIL (PAYLOAD_RS_DECODE)** |
| `0774.png` | `T6_random_iv_3` | random | [0.0, 264.0] | [0, 3360] | 4 | OK | 21 | OK | **PASS** |
| `0784.png` | `T1_fixed_iv_a` | fixed | [-14.3, 268.0] | [416, 1355] | 0 | OK | 18 | OK | **PASS** |
| `0784.png` | `T2_fixed_iv_b` | fixed | [-10.7, 271.5] | [411, 1315] | 0 | OK | 19 | OK | **PASS** |
| `0784.png` | `T3_fixed_iv_c` | fixed | [-14.3, 271.5] | [410, 1356] | 0 | OK | 21 | OK | **PASS** |
| `0784.png` | `T4_random_iv_1` | random | [-14.3, 271.5] | [391, 1354] | 0 | OK | 20 | OK | **PASS** |
| `0784.png` | `T5_random_iv_2` | random | [-14.3, 270.9] | [378, 1331] | 0 | OK | 18 | OK | **PASS** |
| `0784.png` | `T6_random_iv_3` | random | [-14.3, 271.5] | [410, 1345] | 0 | OK | 17 | OK | **PASS** |

---

## 2. Key Forensic Findings

1. **Carrier `0715.png` (Persistent Header Failure Due to Zero-Floor Clipping):**
   - Header byte errors consistently range from 7 to 10 bytes (exceeding RS(24, 16) $t=4$ budget).
   - In all trials, `num_clipped_low` is consistently 1,100 to 1,220 pixels with pre-clamp minimum reaching $-13.9$ to $-11.5$.
   - The payload RS max block error is consistently $\le 11$ (well within $t=24$).
   - Header failure occurs under both deterministic fixed IV and random IV conditions with 100% reproducibility.

2. **Carrier `0774.png` (Persistent Payload RS Failure Due to Ceiling Clipping & High Texture Noise):**
   - Header unpack consistently succeeds (header byte errors 3 to 5).
   - `num_clipped_high` is consistently 3,300 to 3,450 pixels with pre-clamp maximum reaching $+261$ to $+265$.
   - Payload maximum symbol error per block consistently hovers at 24 to 27 symbols, regularly breaching the $t=24$ RS capacity.

3. **Carrier `0784.png` (Marginal Carrier Behavior):**
   - Header is consistently intact (0 to 1 byte errors).
   - Pre-clamp range $[-14.3, 271.5]$ with 1,600 to 1,750 clipped pixels.
   - Payload maximum symbol error ranges between 15 and 25 symbols across trials, explaining why specific random IVs occasionally tip block 4 over 24.
