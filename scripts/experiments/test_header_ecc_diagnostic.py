"""Task 4: Header-ECC Design and Offline Diagnostic.

Compares:
1. Current unprotected header (16 bytes = 128 bits)
2. 3x Bit Repetition with Majority Voting (48 bytes = 384 bits)
3. 5x Bit Repetition with Majority Voting (80 bytes = 640 bits)
4. RS(24, 16) Codec over GF(2^8) (24 bytes = 192 bits, t=4 bytes correctable)

Runs:
- Part 1: Capacity & Overhead Theoretical Comparison
- Part 2: Synthetic Bit-Flip Monte Carlo Simulation (1,000 trials per error count)
- Part 3: Real Carrier Modulation Experiment on 0701.png, 0715.png, 0774.png, 0777.png, 0785.png, 0795.png

Output: docs/diagnostics_20261029/task4_header_ecc_diagnostic.json
"""

import hashlib
import io
import json
import os
import random
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import reedsolo
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.carrier.capacity import CapacityCalculator
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator
from src.core.pipeline import SecurityPipeline
from src.core.serialization import MediaType, SecretPayload
from src.neural.dataset import DIV2KStegoDataset
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder

ONNX_DIR = REPO_ROOT / "checkpoints" / "phase5a_onnx"
DIAG_DIR = REPO_ROOT / "docs" / "diagnostics_20261029"
OUTPUT_JSON = DIAG_DIR / "task4_header_ecc_diagnostic.json"

# =============================================================================
# Helper Codecs for Header Schemes
# =============================================================================

def encode_repetition(bits: np.ndarray, repeat: int = 3) -> np.ndarray:
    return np.repeat(bits, repeat)

def decode_repetition(bits: np.ndarray, repeat: int = 3) -> np.ndarray:
    reshaped = bits.reshape(-1, repeat)
    threshold = (repeat // 2) + 1
    return (np.sum(reshaped, axis=1) >= threshold).astype(np.uint8)

def encode_rs24_16(hdr_bytes: bytes) -> bytes:
    rsc = reedsolo.RSCodec(nsym=8)
    return bytes(rsc.encode(hdr_bytes))

def decode_rs24_16(encoded_bytes: bytes) -> Tuple[Optional[bytes], bool, int]:
    rsc = reedsolo.RSCodec(nsym=8)
    try:
        res = rsc.decode(encoded_bytes)
        corrected = len(res[2]) if len(res) > 2 and res[2] is not None else 0
        return bytes(res[0]), True, corrected
    except Exception:
        return None, False, 0


def main():
    print("Beginning Task 4 Header-ECC Diagnostic...")

    # =========================================================================
    # Part 1: Theoretical Capacity & Overhead Accounting
    # =========================================================================
    subband_dim = 256
    total_positions = subband_dim * subband_dim  # 65,536
    rs_payload_parity = 48  # Primary qualified deployment profile

    candidates = {
        "candidate_1_unprotected": {
            "name": "Current Unprotected Header",
            "header_raw_bytes": 16,
            "header_encoded_bits": 128,
            "header_encoded_bytes": 16,
            "redundancy_overhead_bits": 0,
            "redundancy_overhead_pct": 0.0,
            "theoretical_bit_error_tolerance": 0,
            "theoretical_symbol_error_tolerance": 0,
        },
        "candidate_2_repetition_3x": {
            "name": "3x Bit Repetition Majority Code",
            "header_raw_bytes": 16,
            "header_encoded_bits": 384,
            "header_encoded_bytes": 48,
            "redundancy_overhead_bits": 256,
            "redundancy_overhead_pct": 200.0,
            "theoretical_bit_error_tolerance": 1,  # 1 bit per triplet (up to 128 total)
            "theoretical_symbol_error_tolerance": 0,
        },
        "candidate_3_rs24_16": {
            "name": "Shortened Reed-Solomon RS(24, 16) Codec over GF(2^8)",
            "header_raw_bytes": 16,
            "header_encoded_bits": 192,
            "header_encoded_bytes": 24,
            "redundancy_overhead_bits": 64,
            "redundancy_overhead_pct": 50.0,
            "theoretical_bit_error_tolerance": 32,  # up to 32 bits if concentrated in 4 symbols
            "theoretical_symbol_error_tolerance": 4,  # t=4 bytes anywhere
        },
        "candidate_4_repetition_5x": {
            "name": "5x Bit Repetition Majority Code",
            "header_raw_bytes": 16,
            "header_encoded_bits": 640,
            "header_encoded_bytes": 80,
            "redundancy_overhead_bits": 512,
            "redundancy_overhead_pct": 400.0,
            "theoretical_bit_error_tolerance": 2,  # 2 bits per 5-bit block
            "theoretical_symbol_error_tolerance": 0,
        },
    }

    for k, cand in candidates.items():
        hdr_bits = cand["header_encoded_bits"]
        rem_payload_bits = total_positions - hdr_bits
        rem_payload_bytes = rem_payload_bits // 8

        # Calculate exact usable secret capacity under RS 2t=48
        # Using CapacityCalculator logic
        k_val = 255 - rs_payload_parity  # 207
        num_full = rem_payload_bytes // 255
        rem = rem_payload_bytes % 255
        usable_prot = num_full * k_val + (max(0, rem - rs_payload_parity) if rem > 0 else 0) - 4
        # Metadata overhead: MediaType(1) + MetaLen(2) + IV(12) + Tag(16) = 31 bytes
        usable_secret = max(0, usable_prot - 31)

        cand["subband_footprint_positions"] = hdr_bits
        cand["subband_footprint_pct"] = round(hdr_bits / total_positions * 100.0, 4)
        cand["available_payload_bits"] = rem_payload_bits
        cand["available_protected_payload_bytes"] = rem_payload_bytes
        cand["usable_secret_payload_bytes_rs48"] = usable_secret
        cand["capacity_reduction_vs_current_bytes"] = 6578 - usable_secret

    print("\nPart 1: Candidate Comparison Matrix:")
    for k, cand in candidates.items():
        print(f"  {cand['name']}: Footprint={cand['subband_footprint_positions']} pos ({cand['subband_footprint_pct']}%), Usable Secret Cap={cand['usable_secret_payload_bytes_rs48']} B (Delta={cand['capacity_reduction_vs_current_bytes']} B)")

    # =========================================================================
    # Part 2: Synthetic Bit-Flip Monte Carlo Simulation
    # =========================================================================
    print("\nPart 2: Running Synthetic Bit-Flip Monte Carlo Simulation (1,000 trials per error count)...")
    raw_hdr = CarrierHeader(session_counter=774, payload_length=1338, flags=1).pack()
    raw_bits = BaselineEmbedder.bytes_to_bits(raw_hdr)

    error_levels = [1, 2, 3, 4, 5, 6, 8, 10]
    monte_carlo_results = {}

    rng = random.Random(42)
    n_trials = 1000

    for err_count in error_levels:
        success_unprotected = 0
        success_rep3 = 0
        success_rs24 = 0
        success_rep5 = 0

        # Encode candidates
        enc_rep3 = encode_repetition(raw_bits, repeat=3)
        enc_rs24 = encode_rs24_16(raw_hdr)
        enc_rep5 = encode_repetition(raw_bits, repeat=5)

        for _ in range(n_trials):
            # 1. Unprotected (flips err_count bits out of 128)
            corrupted_unprot = np.array(raw_bits, copy=True)
            flip_pos_unprot = rng.sample(range(128), err_count)
            for p in flip_pos_unprot:
                corrupted_unprot[p] ^= 1
            if np.array_equal(corrupted_unprot, raw_bits):
                success_unprotected += 1

            # 2. 3x Repetition (flips err_count bits out of 384)
            corrupted_rep3 = np.array(enc_rep3, copy=True)
            flip_pos_rep3 = rng.sample(range(384), err_count)
            for p in flip_pos_rep3:
                corrupted_rep3[p] ^= 1
            dec_rep3 = decode_repetition(corrupted_rep3, repeat=3)
            if np.array_equal(dec_rep3, raw_bits):
                success_rep3 += 1

            # 3. RS(24, 16) (flips err_count bits out of 192)
            corrupted_rs24_bits = BaselineEmbedder.bytes_to_bits(enc_rs24)
            flip_pos_rs24 = rng.sample(range(192), err_count)
            for p in flip_pos_rs24:
                corrupted_rs24_bits[p] ^= 1
            corrupted_rs24_bytes = Extractor.bits_to_bytes(corrupted_rs24_bits)
            dec_rs24_bytes, ok_rs24, _ = decode_rs24_16(corrupted_rs24_bytes)
            if ok_rs24 and dec_rs24_bytes == raw_hdr:
                success_rs24 += 1

            # 4. 5x Repetition (flips err_count bits out of 640)
            corrupted_rep5 = np.array(enc_rep5, copy=True)
            flip_pos_rep5 = rng.sample(range(640), err_count)
            for p in flip_pos_rep5:
                corrupted_rep5[p] ^= 1
            dec_rep5 = decode_repetition(corrupted_rep5, repeat=5)
            if np.array_equal(dec_rep5, raw_bits):
                success_rep5 += 1

        monte_carlo_results[str(err_count)] = {
            "bit_errors_injected": err_count,
            "trials": n_trials,
            "success_rate_unprotected": round(success_unprotected / n_trials * 100.0, 2),
            "success_rate_repetition_3x": round(success_rep3 / n_trials * 100.0, 2),
            "success_rate_rs24_16": round(success_rs24 / n_trials * 100.0, 2),
            "success_rate_repetition_5x": round(success_rep5 / n_trials * 100.0, 2),
        }
        print(f"  Errors={err_count:02d} | Unprot: {monte_carlo_results[str(err_count)]['success_rate_unprotected']}% | "
              f"Rep3x: {monte_carlo_results[str(err_count)]['success_rate_repetition_3x']}% | "
              f"RS24: {monte_carlo_results[str(err_count)]['success_rate_rs24_16']}% | "
              f"Rep5x: {monte_carlo_results[str(err_count)]['success_rate_repetition_5x']}%")

    # =========================================================================
    # Part 3: Real Carrier Modulation Experiment on Failed Carriers
    # =========================================================================
    print("\nPart 3: Executing Real Carrier Modulation Experiment on Target Carriers...")
    test_dataset = DIV2KStegoDataset(split="test", crop_size=512, random_crop=False)
    target_filenames = ["0701.png", "0715.png", "0774.png", "0777.png", "0785.png", "0795.png"]

    # Load ONNX spatial sessions
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_embed = ort.InferenceSession(
        str(ONNX_DIR / "neural_spatial_embedding_opset14.onnx"), sess_opts, providers=["CPUExecutionProvider"]
    )
    sess_extract = ort.InferenceSession(
        str(ONNX_DIR / "neural_spatial_extraction_opset14.onnx"), sess_opts, providers=["CPUExecutionProvider"]
    )

    # Security pipeline setup (RS 2t=48)
    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder(parity_symbols=48)
    sec_pipe = SecurityPipeline(crypto_provider=crypto, ecc_encoder=ecc)

    payload_len = 1000
    secret_bytes = bytes(((i * 37 + 13) % 256) for i in range(payload_len))
    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.TEXT,
        metadata={"filename": "task4_secret.txt"},
    )
    aes_key = hashlib.sha256(b"PaladinMedia-Task4-HeaderECC-2026").digest()
    aad = b"Task4-HeaderECC"
    protected_payload = sec_pipe.protect(secret_payload, aes_key, aad=aad)

    real_carrier_results = {}

    for fn in target_filenames:
        idx = [i for i, r in enumerate(test_dataset.records) if r["filename"] == fn][0]
        rec = test_dataset[idx]
        img_id = rec["image_id"]
        carrier_t = rec["carrier"].unsqueeze(0)
        carrier_np = carrier_t.numpy()

        raw_hdr = CarrierHeader(
            session_counter=img_id,
            payload_length=len(protected_payload),
            flags=CarrierHeader.ECC_PROFILE_RS_2T_48,
        ).pack()
        raw_hdr_bits = BaselineEmbedder.bytes_to_bits(raw_hdr)
        pay_bits = BaselineEmbedder.bytes_to_bits(protected_payload)

        # -------------------------------------------------------------
        # Scheme A: Current Unprotected (128 bits anchor: 0..127)
        # -------------------------------------------------------------
        mask_a = np.zeros((1, 1, 256, 256), dtype=np.float32)
        payload_a = np.zeros((1, 1, 256, 256), dtype=np.float32)
        pos_hdr_a = tuple(divmod(i, 256) for i in range(128))
        pos_pay_a = DeterministicMaskGenerator.generate_payload_positions(img_id, len(protected_payload), 256)

        for b, (r, c) in zip(raw_hdr_bits, pos_hdr_a):
            mask_a[0, 0, r, c] = 1.0
            payload_a[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0
        for b, (r, c) in zip(pay_bits, pos_pay_a):
            mask_a[0, 0, r, c] = 1.0
            payload_a[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0

        stego_a, _, _ = sess_embed.run(None, {"carrier": carrier_np, "mask": mask_a, "payload_bipolar": payload_a})
        # PNG serialization and reload
        buf_a = io.BytesIO()
        Image.fromarray(np.round(stego_a[0].transpose(1, 2, 0)).astype(np.uint8)).save(buf_a, format="PNG")
        reloaded_a = np.array(Image.open(io.BytesIO(buf_a.getvalue())), dtype=np.float32).transpose(2, 0, 1)[None, ...]

        bin_grid_a, _ = sess_extract.run(None, {"stego_carrier": reloaded_a})
        ext_hdr_bits_a = np.array([int(bin_grid_a[0, 0, r, c]) for r, c in pos_hdr_a], dtype=np.uint8)
        hdr_bit_errs_a = int(np.sum(ext_hdr_bits_a != raw_hdr_bits))
        hdr_recovered_a = (hdr_bit_errs_a == 0)

        # Unprotect attempt under scheme A
        e2e_match_a = False
        pay_rs_success_a = False
        if hdr_recovered_a:
            ext_pay_bits_a = np.array([int(bin_grid_a[0, 0, r, c]) for r, c in pos_pay_a], dtype=np.uint8)
            ext_prot_bytes_a = Extractor.bits_to_bytes(ext_pay_bits_a)
            try:
                dec_a = ecc.decode_with_result(ext_prot_bytes_a)
                pay_rs_success_a = True
                rec_a = sec_pipe.unprotect(ext_prot_bytes_a, aes_key, aad=aad)
                e2e_match_a = (rec_a.data == secret_bytes)
            except Exception:
                pass

        # -------------------------------------------------------------
        # Scheme B: 3x Bit Repetition (384 bits anchor: 0..383)
        # -------------------------------------------------------------
        mask_b = np.zeros((1, 1, 256, 256), dtype=np.float32)
        payload_b = np.zeros((1, 1, 256, 256), dtype=np.float32)
        pos_hdr_b = tuple(divmod(i, 256) for i in range(384))
        # Payload positions start after 384 positions: sample from [384..65535]
        # Re-derive payload positions safely avoiding first 384 positions
        hasher_b = hashlib.sha256(b"PALADIN_MASK_SEED_V1" + struct.pack(">II", img_id, len(pay_bits)))
        seed_b = int.from_bytes(hasher_b.digest()[:8], "big")
        rng_b = random.Random(seed_b)
        pos_pay_b = tuple(divmod(i, 256) for i in rng_b.sample(range(384, 65536), len(pay_bits)))

        enc_rep3_bits = encode_repetition(raw_hdr_bits, repeat=3)

        for b, (r, c) in zip(enc_rep3_bits, pos_hdr_b):
            mask_b[0, 0, r, c] = 1.0
            payload_b[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0
        for b, (r, c) in zip(pay_bits, pos_pay_b):
            mask_b[0, 0, r, c] = 1.0
            payload_b[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0

        stego_b, _, _ = sess_embed.run(None, {"carrier": carrier_np, "mask": mask_b, "payload_bipolar": payload_b})
        buf_b = io.BytesIO()
        Image.fromarray(np.round(stego_b[0].transpose(1, 2, 0)).astype(np.uint8)).save(buf_b, format="PNG")
        reloaded_b = np.array(Image.open(io.BytesIO(buf_b.getvalue())), dtype=np.float32).transpose(2, 0, 1)[None, ...]

        bin_grid_b, _ = sess_extract.run(None, {"stego_carrier": reloaded_b})
        ext_rep3_bits = np.array([int(bin_grid_b[0, 0, r, c]) for r, c in pos_hdr_b], dtype=np.uint8)
        raw_channel_hdr_errs_b = int(np.sum(ext_rep3_bits != enc_rep3_bits))

        dec_hdr_bits_b = decode_repetition(ext_rep3_bits, repeat=3)
        hdr_post_decode_errs_b = int(np.sum(dec_hdr_bits_b != raw_hdr_bits))
        hdr_recovered_b = (hdr_post_decode_errs_b == 0)

        e2e_match_b = False
        pay_rs_success_b = False
        max_cw_err_b = None
        if hdr_recovered_b:
            ext_pay_bits_b = np.array([int(bin_grid_b[0, 0, r, c]) for r, c in pos_pay_b], dtype=np.uint8)
            ext_prot_bytes_b = Extractor.bits_to_bytes(ext_pay_bits_b)

            k_48 = ecc.k
            block_len_48 = k_48 + ecc.ecc_symbols
            offset = 4
            num_blocks_48 = (len(protected_payload) - 4) // block_len_48 + (1 if (len(protected_payload) - 4) % block_len_48 > 0 else 0)
            cw_errs = []
            for blk in range(num_blocks_48):
                cur_len = block_len_48 if (offset + block_len_48 <= len(protected_payload)) else (len(protected_payload) - offset)
                e_blk = ext_prot_bytes_b[offset : offset + cur_len]
                a_blk = protected_payload[offset : offset + cur_len]
                cw_errs.append(sum(1 for x, y in zip(e_blk, a_blk) if x != y))
                offset += cur_len
            max_cw_err_b = max(cw_errs) if cw_errs else 0

            try:
                dec_b = ecc.decode_with_result(ext_prot_bytes_b)
                pay_rs_success_b = True
                rec_b = sec_pipe.unprotect(ext_prot_bytes_b, aes_key, aad=aad)
                e2e_match_b = (rec_b.data == secret_bytes)
            except Exception:
                pass

        # -------------------------------------------------------------
        # Scheme C: RS(24, 16) Codec (192 bits anchor: 0..191)
        # -------------------------------------------------------------
        mask_c = np.zeros((1, 1, 256, 256), dtype=np.float32)
        payload_c = np.zeros((1, 1, 256, 256), dtype=np.float32)
        pos_hdr_c = tuple(divmod(i, 256) for i in range(192))
        hasher_c = hashlib.sha256(b"PALADIN_MASK_SEED_V1" + struct.pack(">II", img_id, len(pay_bits)))
        seed_c = int.from_bytes(hasher_c.digest()[:8], "big")
        rng_c = random.Random(seed_c)
        pos_pay_c = tuple(divmod(i, 256) for i in rng_c.sample(range(192, 65536), len(pay_bits)))

        enc_rs24_bytes = encode_rs24_16(raw_hdr)
        enc_rs24_bits = BaselineEmbedder.bytes_to_bits(enc_rs24_bytes)

        for b, (r, c) in zip(enc_rs24_bits, pos_hdr_c):
            mask_c[0, 0, r, c] = 1.0
            payload_c[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0
        for b, (r, c) in zip(pay_bits, pos_pay_c):
            mask_c[0, 0, r, c] = 1.0
            payload_c[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0

        stego_c, _, _ = sess_embed.run(None, {"carrier": carrier_np, "mask": mask_c, "payload_bipolar": payload_c})
        buf_c = io.BytesIO()
        Image.fromarray(np.round(stego_c[0].transpose(1, 2, 0)).astype(np.uint8)).save(buf_c, format="PNG")
        reloaded_c = np.array(Image.open(io.BytesIO(buf_c.getvalue())), dtype=np.float32).transpose(2, 0, 1)[None, ...]

        bin_grid_c, _ = sess_extract.run(None, {"stego_carrier": reloaded_c})
        ext_rs24_bits = np.array([int(bin_grid_c[0, 0, r, c]) for r, c in pos_hdr_c], dtype=np.uint8)
        raw_channel_hdr_errs_c = int(np.sum(ext_rs24_bits != enc_rs24_bits))

        ext_rs24_bytes = Extractor.bits_to_bytes(ext_rs24_bits)
        dec_hdr_bytes_c, ok_rs24_c, corrected_c = decode_rs24_16(ext_rs24_bytes)
        hdr_recovered_c = ok_rs24_c and (dec_hdr_bytes_c == raw_hdr)

        e2e_match_c = False
        pay_rs_success_c = False
        max_cw_err_c = None
        if hdr_recovered_c:
            ext_pay_bits_c = np.array([int(bin_grid_c[0, 0, r, c]) for r, c in pos_pay_c], dtype=np.uint8)
            ext_prot_bytes_c = Extractor.bits_to_bytes(ext_pay_bits_c)

            k_48 = ecc.k
            block_len_48 = k_48 + ecc.ecc_symbols
            offset = 4
            num_blocks_48 = (len(protected_payload) - 4) // block_len_48 + (1 if (len(protected_payload) - 4) % block_len_48 > 0 else 0)
            cw_errs = []
            for blk in range(num_blocks_48):
                cur_len = block_len_48 if (offset + block_len_48 <= len(protected_payload)) else (len(protected_payload) - offset)
                e_blk = ext_prot_bytes_c[offset : offset + cur_len]
                a_blk = protected_payload[offset : offset + cur_len]
                cw_errs.append(sum(1 for x, y in zip(e_blk, a_blk) if x != y))
                offset += cur_len
            max_cw_err_c = max(cw_errs) if cw_errs else 0

            try:
                dec_c = ecc.decode_with_result(ext_prot_bytes_c)
                pay_rs_success_c = True
                rec_c = sec_pipe.unprotect(ext_prot_bytes_c, aes_key, aad=aad)
                e2e_match_c = (rec_c.data == secret_bytes)
            except Exception:
                pass

        real_carrier_results[fn] = {
            "image_id": img_id,
            "filename": fn,
            "scheme_a_current_unprotected": {
                "header_bit_errors": hdr_bit_errs_a,
                "header_recovered": hdr_recovered_a,
                "payload_rs_success": pay_rs_success_a,
                "exact_recovery": e2e_match_a,
            },
            "scheme_b_repetition_3x": {
                "raw_channel_bit_errors_384": raw_channel_hdr_errs_b,
                "header_errors_after_majority_decode": hdr_post_decode_errs_b,
                "header_recovered": hdr_recovered_b,
                "payload_max_codeword_error": max_cw_err_b,
                "payload_rs_success": pay_rs_success_b,
                "exact_recovery": e2e_match_b,
            },
            "scheme_c_rs24_16": {
                "raw_channel_bit_errors_192": raw_channel_hdr_errs_c,
                "rs_corrected_symbols": corrected_c,
                "header_recovered": hdr_recovered_c,
                "payload_max_codeword_error": max_cw_err_c,
                "payload_rs_success": pay_rs_success_c,
                "exact_recovery": e2e_match_c,
            },
        }

        print(f"[{fn}] Scheme A: HdrRecovered={hdr_recovered_a}, E2E={e2e_match_a} | "
              f"Scheme B (3x Rep): HdrRecovered={hdr_recovered_b}, MaxErr={max_cw_err_b}/24, E2E={e2e_match_b} | "
              f"Scheme C (RS24): HdrRecovered={hdr_recovered_c}, MaxErr={max_cw_err_c}/24, E2E={e2e_match_c}")

    output_payload = {
        "theoretical_candidate_comparison": candidates,
        "synthetic_monte_carlo_simulation": monte_carlo_results,
        "real_carrier_empirical_results": real_carrier_results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nTask 4 Header-ECC Diagnostic completed! Results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
