"""PaladinMedia — Full 100-Carrier Header ECC Qualification Runner.

Evaluates:
- Scheme A: Current Unprotected 16-byte Header (128 bits anchor, 0..127)
- Scheme B: 3x Bit Repetition with Majority Voting (384 bits anchor, 0..383)
- Scheme C: Shortened RS(24, 16) Codec over GF(2^8) (192 bits anchor, 0..191)

Across all 100 canonical DIV2K test split images (0701.png - 0800.png).

Non-negotiable Safeguards:
- Strictly read-only on model weights: NO training, retraining, or fine-tuning.
- Read-only on production source code under src/.
- Preserves docs/experiments/held_out_test_results_20261029.json and model hashes.
- Emits results to docs/diagnostics_20261029/full_100_carrier_qualification/.
"""

import csv
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
import reedsolo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator
from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.core.pipeline import SecurityPipeline
from src.core.serialization import MediaType, SecretPayload
from src.neural.dataset import DIV2KStegoDataset
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("100CarrierQualification")

ONNX_DIR = REPO_ROOT / "checkpoints" / "phase5a_onnx"
OUTPUT_DIR = REPO_ROOT / "docs" / "diagnostics_20261029" / "full_100_carrier_qualification"
OUTPUT_JSON = OUTPUT_DIR / "header_ecc_100_carrier_results.json"
OUTPUT_CSV = OUTPUT_DIR / "header_ecc_100_carrier_comparison.csv"

PROTECTED_PATHS = [
    REPO_ROOT / "checkpoints" / "phase4c_extended_30ep" / "best_model.pt",
    ONNX_DIR / "neural_spatial_embedding_opset14.onnx",
    ONNX_DIR / "neural_spatial_extraction_opset14.onnx",
    REPO_ROOT / "docs" / "experiments" / "held_out_test_results_20261029.json",
]


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Codecs for Scheme B & Scheme C
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


def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def main():
    logger.info("=" * 80)
    logger.info("PaladinMedia — Full 100-Carrier Header ECC Qualification")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify Pre-Execution Hashes
    pre_hashes = {}
    for p in PROTECTED_PATHS:
        if not p.is_file():
            raise FileNotFoundError(f"Required protected file missing: {p}")
        h = compute_file_sha256(p)
        pre_hashes[str(p.relative_to(REPO_ROOT))] = h
        logger.info(f"Pre-hash {p.name}: {h}")

    # 2. Setup Sessions and Pipeline
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

    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder(parity_symbols=48)
    sec_pipe = SecurityPipeline(crypto_provider=crypto, ecc_encoder=ecc)

    # Constant 1,000-byte test secret payload (identical across all carriers and schemes)
    payload_len = 1000
    secret_bytes = bytes(((i * 37 + 13) % 256) for i in range(payload_len))
    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.TEXT,
        metadata={"filename": "qualification_1000b.txt", "benchmark": "test_0701_0800"},
    )
    aes_key = hashlib.sha256(b"PaladinMedia-100Carrier-HeaderECC-2026").digest()
    aad = b"PaladinMedia-100Carrier-HeaderECC-Qualification"

    protected_payload = sec_pipe.protect(secret_payload, aes_key, aad=aad)
    protected_len = len(protected_payload)
    pay_bits = BaselineEmbedder.bytes_to_bits(protected_payload)
    num_pay_bits = len(pay_bits)

    logger.info(f"Secret bytes: {payload_len} B | Protected payload: {protected_len} B | Payload bits: {num_pay_bits} bits")

    test_dataset = DIV2KStegoDataset(split="test", crop_size=512, random_crop=False)
    num_images = len(test_dataset)
    logger.info(f"Loaded {num_images} canonical DIV2K test carriers (0701.png - 0800.png).")
    if num_images != 100:
        raise ValueError(f"Expected 100 test carriers, got {num_images}")

    records: List[Dict[str, Any]] = []

    t_start_total = time.time()

    for idx in range(num_images):
        rec = test_dataset[idx]
        img_id = int(rec["image_id"])
        filename = rec["filename"]
        carrier_t = rec["carrier"].unsqueeze(0)  # (1, 3, 512, 512)
        carrier_np = carrier_t.numpy()
        cover_uint8 = np.clip(carrier_np[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)

        raw_hdr = CarrierHeader(
            session_counter=img_id,
            payload_length=protected_len,
            flags=CarrierHeader.ECC_PROFILE_RS_2T_48,
        ).pack()
        raw_hdr_bits = BaselineEmbedder.bytes_to_bits(raw_hdr)

        # =====================================================================
        # Scheme A: Current Unprotected (128 bits anchor: 0..127)
        # =====================================================================
        t0_a = time.perf_counter()
        mask_a = np.zeros((1, 1, 256, 256), dtype=np.float32)
        payload_a = np.zeros((1, 1, 256, 256), dtype=np.float32)
        pos_hdr_a = tuple(divmod(i, 256) for i in range(128))
        pos_pay_a = DeterministicMaskGenerator.generate_payload_positions(img_id, protected_len, 256)

        for b, (r, c) in zip(raw_hdr_bits, pos_hdr_a):
            mask_a[0, 0, r, c] = 1.0
            payload_a[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0
        for b, (r, c) in zip(pay_bits, pos_pay_a):
            mask_a[0, 0, r, c] = 1.0
            payload_a[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0

        stego_a_raw, _, _ = sess_embed.run(None, {"carrier": carrier_np, "mask": mask_a, "payload_bipolar": payload_a})
        embed_ms_a = (time.perf_counter() - t0_a) * 1000.0

        stego_a_uint8 = np.clip(np.round(stego_a_raw[0].transpose(1, 2, 0)), 0, 255).astype(np.uint8)
        psnr_a = calculate_psnr(cover_uint8, stego_a_uint8)
        ssim_a = calculate_ssim(cover_uint8, stego_a_uint8)

        # Lossless PNG roundtrip
        buf_a = io.BytesIO()
        Image.fromarray(stego_a_uint8).save(buf_a, format="PNG")
        reloaded_a = np.array(Image.open(io.BytesIO(buf_a.getvalue())), dtype=np.float32).transpose(2, 0, 1)[None, ...]

        t1_a = time.perf_counter()
        bin_grid_a, _ = sess_extract.run(None, {"stego_carrier": reloaded_a})
        extract_ms_a = (time.perf_counter() - t1_a) * 1000.0

        ext_hdr_bits_a = np.array([int(bin_grid_a[0, 0, r, c]) for r, c in pos_hdr_a], dtype=np.uint8)
        hdr_bit_errs_a = int(np.sum(ext_hdr_bits_a != raw_hdr_bits))
        hdr_ber_a = float(hdr_bit_errs_a) / 128.0
        hdr_valid_a = (hdr_bit_errs_a == 0)
        hdr_err_msg_a = None
        if not hdr_valid_a:
            try:
                CarrierHeader.unpack(Extractor.bits_to_bytes(ext_hdr_bits_a))
            except Exception as e:
                hdr_err_msg_a = f"{type(e).__name__}: {e}"

        # Ground truth payload extraction
        ext_pay_bits_a = np.array([int(bin_grid_a[0, 0, r, c]) for r, c in pos_pay_a], dtype=np.uint8)
        pay_bit_errs_a = int(np.sum(ext_pay_bits_a != pay_bits))
        pay_ber_a = float(pay_bit_errs_a) / num_pay_bits
        ext_prot_bytes_a = Extractor.bits_to_bytes(ext_pay_bits_a)

        # Codeword error analysis
        k_48 = ecc.k
        block_len_48 = k_48 + ecc.ecc_symbols
        offset = 4
        num_blocks_48 = (protected_len - 4) // block_len_48 + (1 if (protected_len - 4) % block_len_48 > 0 else 0)
        cw_errs_a = []
        for blk in range(num_blocks_48):
            cur_len = block_len_48 if (offset + block_len_48 <= protected_len) else (protected_len - offset)
            e_blk = ext_prot_bytes_a[offset : offset + cur_len]
            a_blk = protected_payload[offset : offset + cur_len]
            cw_errs_a.append(sum(1 for x, y in zip(e_blk, a_blk) if x != y))
            offset += cur_len
        max_cw_err_a = max(cw_errs_a) if cw_errs_a else 0

        pay_rs_success_a = False
        try:
            ecc.decode_with_result(ext_prot_bytes_a)
            pay_rs_success_a = True
        except Exception:
            pay_rs_success_a = False

        exact_match_a = False
        aes_auth_a = False
        receiver_err_a = None
        if hdr_valid_a:
            try:
                rec_a = sec_pipe.unprotect(ext_prot_bytes_a, aes_key, aad=aad)
                aes_auth_a = True
                exact_match_a = (rec_a.data == secret_bytes)
            except Exception as e:
                receiver_err_a = f"{type(e).__name__}: {e}"
        else:
            receiver_err_a = f"HeaderCorrupted: {hdr_bit_errs_a} bit errors"

        # =====================================================================
        # Scheme B: 3x Bit Repetition (384 bits anchor: 0..383)
        # =====================================================================
        t0_b = time.perf_counter()
        mask_b = np.zeros((1, 1, 256, 256), dtype=np.float32)
        payload_b = np.zeros((1, 1, 256, 256), dtype=np.float32)
        pos_hdr_b = tuple(divmod(i, 256) for i in range(384))

        hasher_b = hashlib.sha256(b"PALADIN_MASK_SEED_V1" + struct.pack(">II", img_id, num_pay_bits))
        seed_b = int.from_bytes(hasher_b.digest()[:8], "big")
        rng_b = random.Random(seed_b)
        pos_pay_b = tuple(divmod(i, 256) for i in rng_b.sample(range(384, 65536), num_pay_bits))

        enc_rep3_bits = encode_repetition(raw_hdr_bits, repeat=3)

        for b, (r, c) in zip(enc_rep3_bits, pos_hdr_b):
            mask_b[0, 0, r, c] = 1.0
            payload_b[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0
        for b, (r, c) in zip(pay_bits, pos_pay_b):
            mask_b[0, 0, r, c] = 1.0
            payload_b[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0

        stego_b_raw, _, _ = sess_embed.run(None, {"carrier": carrier_np, "mask": mask_b, "payload_bipolar": payload_b})
        embed_ms_b = (time.perf_counter() - t0_b) * 1000.0

        stego_b_uint8 = np.clip(np.round(stego_b_raw[0].transpose(1, 2, 0)), 0, 255).astype(np.uint8)
        psnr_b = calculate_psnr(cover_uint8, stego_b_uint8)
        ssim_b = calculate_ssim(cover_uint8, stego_b_uint8)

        buf_b = io.BytesIO()
        Image.fromarray(stego_b_uint8).save(buf_b, format="PNG")
        reloaded_b = np.array(Image.open(io.BytesIO(buf_b.getvalue())), dtype=np.float32).transpose(2, 0, 1)[None, ...]

        t1_b = time.perf_counter()
        bin_grid_b, _ = sess_extract.run(None, {"stego_carrier": reloaded_b})
        extract_ms_b = (time.perf_counter() - t1_b) * 1000.0

        ext_rep3_bits = np.array([int(bin_grid_b[0, 0, r, c]) for r, c in pos_hdr_b], dtype=np.uint8)
        raw_channel_hdr_errs_b = int(np.sum(ext_rep3_bits != enc_rep3_bits))
        raw_channel_hdr_ber_b = float(raw_channel_hdr_errs_b) / 384.0

        dec_hdr_bits_b = decode_repetition(ext_rep3_bits, repeat=3)
        post_hdr_errs_b = int(np.sum(dec_hdr_bits_b != raw_hdr_bits))
        post_hdr_ber_b = float(post_hdr_errs_b) / 128.0
        hdr_valid_b = (post_hdr_errs_b == 0)
        hdr_err_msg_b = None
        if not hdr_valid_b:
            try:
                CarrierHeader.unpack(Extractor.bits_to_bytes(dec_hdr_bits_b))
            except Exception as e:
                hdr_err_msg_b = f"{type(e).__name__}: {e}"

        ext_pay_bits_b = np.array([int(bin_grid_b[0, 0, r, c]) for r, c in pos_pay_b], dtype=np.uint8)
        pay_bit_errs_b = int(np.sum(ext_pay_bits_b != pay_bits))
        pay_ber_b = float(pay_bit_errs_b) / num_pay_bits
        ext_prot_bytes_b = Extractor.bits_to_bytes(ext_pay_bits_b)

        offset = 4
        cw_errs_b = []
        for blk in range(num_blocks_48):
            cur_len = block_len_48 if (offset + block_len_48 <= protected_len) else (protected_len - offset)
            e_blk = ext_prot_bytes_b[offset : offset + cur_len]
            a_blk = protected_payload[offset : offset + cur_len]
            cw_errs_b.append(sum(1 for x, y in zip(e_blk, a_blk) if x != y))
            offset += cur_len
        max_cw_err_b = max(cw_errs_b) if cw_errs_b else 0

        pay_rs_success_b = False
        try:
            ecc.decode_with_result(ext_prot_bytes_b)
            pay_rs_success_b = True
        except Exception:
            pay_rs_success_b = False

        exact_match_b = False
        aes_auth_b = False
        receiver_err_b = None
        if hdr_valid_b:
            try:
                rec_b = sec_pipe.unprotect(ext_prot_bytes_b, aes_key, aad=aad)
                aes_auth_b = True
                exact_match_b = (rec_b.data == secret_bytes)
            except Exception as e:
                receiver_err_b = f"{type(e).__name__}: {e}"
        else:
            receiver_err_b = f"HeaderCorruptedPostMajority: {post_hdr_errs_b} bit errors"

        # =====================================================================
        # Scheme C: Shortened RS(24, 16) Codec (192 bits anchor: 0..191)
        # =====================================================================
        t0_c = time.perf_counter()
        mask_c = np.zeros((1, 1, 256, 256), dtype=np.float32)
        payload_c = np.zeros((1, 1, 256, 256), dtype=np.float32)
        pos_hdr_c = tuple(divmod(i, 256) for i in range(192))

        hasher_c = hashlib.sha256(b"PALADIN_MASK_SEED_V1" + struct.pack(">II", img_id, num_pay_bits))
        seed_c = int.from_bytes(hasher_c.digest()[:8], "big")
        rng_c = random.Random(seed_c)
        pos_pay_c = tuple(divmod(i, 256) for i in rng_c.sample(range(192, 65536), num_pay_bits))

        enc_rs24_bytes = encode_rs24_16(raw_hdr)
        enc_rs24_bits = BaselineEmbedder.bytes_to_bits(enc_rs24_bytes)

        for b, (r, c) in zip(enc_rs24_bits, pos_hdr_c):
            mask_c[0, 0, r, c] = 1.0
            payload_c[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0
        for b, (r, c) in zip(pay_bits, pos_pay_c):
            mask_c[0, 0, r, c] = 1.0
            payload_c[0, 0, r, c] = 1.0 if int(b) == 1 else -1.0

        stego_c_raw, _, _ = sess_embed.run(None, {"carrier": carrier_np, "mask": mask_c, "payload_bipolar": payload_c})
        embed_ms_c = (time.perf_counter() - t0_c) * 1000.0

        stego_c_uint8 = np.clip(np.round(stego_c_raw[0].transpose(1, 2, 0)), 0, 255).astype(np.uint8)
        psnr_c = calculate_psnr(cover_uint8, stego_c_uint8)
        ssim_c = calculate_ssim(cover_uint8, stego_c_uint8)

        buf_c = io.BytesIO()
        Image.fromarray(stego_c_uint8).save(buf_c, format="PNG")
        reloaded_c = np.array(Image.open(io.BytesIO(buf_c.getvalue())), dtype=np.float32).transpose(2, 0, 1)[None, ...]

        t1_c = time.perf_counter()
        bin_grid_c, _ = sess_extract.run(None, {"stego_carrier": reloaded_c})
        extract_ms_c = (time.perf_counter() - t1_c) * 1000.0

        ext_rs24_bits = np.array([int(bin_grid_c[0, 0, r, c]) for r, c in pos_hdr_c], dtype=np.uint8)
        raw_channel_hdr_errs_c = int(np.sum(ext_rs24_bits != enc_rs24_bits))
        raw_channel_hdr_ber_c = float(raw_channel_hdr_errs_c) / 192.0

        ext_rs24_bytes = Extractor.bits_to_bytes(ext_rs24_bits)
        dec_hdr_bytes_c, ok_rs24_c, corrected_c = decode_rs24_16(ext_rs24_bytes)
        hdr_valid_c = ok_rs24_c and (dec_hdr_bytes_c == raw_hdr)
        hdr_err_msg_c = None
        if not hdr_valid_c:
            if not ok_rs24_c:
                hdr_err_msg_c = "RSCodecError: Uncorrectable errors (>4 bytes)"
            else:
                hdr_err_msg_c = "RSCodecMismatch: Decoded header does not match ground truth"

        ext_pay_bits_c = np.array([int(bin_grid_c[0, 0, r, c]) for r, c in pos_pay_c], dtype=np.uint8)
        pay_bit_errs_c = int(np.sum(ext_pay_bits_c != pay_bits))
        pay_ber_c = float(pay_bit_errs_c) / num_pay_bits
        ext_prot_bytes_c = Extractor.bits_to_bytes(ext_pay_bits_c)

        offset = 4
        cw_errs_c = []
        for blk in range(num_blocks_48):
            cur_len = block_len_48 if (offset + block_len_48 <= protected_len) else (protected_len - offset)
            e_blk = ext_prot_bytes_c[offset : offset + cur_len]
            a_blk = protected_payload[offset : offset + cur_len]
            cw_errs_c.append(sum(1 for x, y in zip(e_blk, a_blk) if x != y))
            offset += cur_len
        max_cw_err_c = max(cw_errs_c) if cw_errs_c else 0

        pay_rs_success_c = False
        try:
            ecc.decode_with_result(ext_prot_bytes_c)
            pay_rs_success_c = True
        except Exception:
            pay_rs_success_c = False

        exact_match_c = False
        aes_auth_c = False
        receiver_err_c = None
        if hdr_valid_c:
            try:
                rec_c = sec_pipe.unprotect(ext_prot_bytes_c, aes_key, aad=aad)
                aes_auth_c = True
                exact_match_c = (rec_c.data == secret_bytes)
            except Exception as e:
                receiver_err_c = f"{type(e).__name__}: {e}"
        else:
            receiver_err_c = f"HeaderCorruptedPostRS24: {hdr_err_msg_c}"

        def classify_delta(a_pass: bool, new_pass: bool) -> str:
            if not a_pass and new_pass:
                return "FIXED"
            elif a_pass and not new_pass:
                return "REGRESSED"
            elif a_pass and new_pass:
                return "IDENTICAL_PASS"
            else:
                return "IDENTICAL_FAIL"

        delta_b = classify_delta(exact_match_a, exact_match_b)
        delta_c = classify_delta(exact_match_a, exact_match_c)

        carrier_record = {
            "image_id": img_id,
            "filename": filename,
            "scheme_a": {
                "header_raw_errors": hdr_bit_errs_a,
                "header_ber": hdr_ber_a,
                "header_valid": hdr_valid_a,
                "header_error_msg": hdr_err_msg_a,
                "payload_raw_errors": pay_bit_errs_a,
                "payload_ber": pay_ber_a,
                "max_codeword_errors": max_cw_err_a,
                "payload_rs_success": pay_rs_success_a,
                "exact_match": exact_match_a,
                "aes_authenticated": aes_auth_a,
                "receiver_error": receiver_err_a,
                "psnr_db": psnr_a,
                "ssim": ssim_a,
                "embed_latency_ms": embed_ms_a,
                "extract_latency_ms": extract_ms_a,
            },
            "scheme_b": {
                "header_channel_errors_384": raw_channel_hdr_errs_b,
                "header_channel_ber": raw_channel_hdr_ber_b,
                "header_post_errors_128": post_hdr_errs_b,
                "header_post_ber": post_hdr_ber_b,
                "header_valid": hdr_valid_b,
                "header_error_msg": hdr_err_msg_b,
                "payload_raw_errors": pay_bit_errs_b,
                "payload_ber": pay_ber_b,
                "max_codeword_errors": max_cw_err_b,
                "payload_rs_success": pay_rs_success_b,
                "exact_match": exact_match_b,
                "aes_authenticated": aes_auth_b,
                "receiver_error": receiver_err_b,
                "psnr_db": psnr_b,
                "ssim": ssim_b,
                "embed_latency_ms": embed_ms_b,
                "extract_latency_ms": extract_ms_b,
            },
            "scheme_c": {
                "header_channel_errors_192": raw_channel_hdr_errs_c,
                "header_channel_ber": raw_channel_hdr_ber_c,
                "header_rs_corrected_bytes": corrected_c,
                "header_valid": hdr_valid_c,
                "header_error_msg": hdr_err_msg_c,
                "payload_raw_errors": pay_bit_errs_c,
                "payload_ber": pay_ber_c,
                "max_codeword_errors": max_cw_err_c,
                "payload_rs_success": pay_rs_success_c,
                "exact_match": exact_match_c,
                "aes_authenticated": aes_auth_c,
                "receiver_error": receiver_err_c,
                "psnr_db": psnr_c,
                "ssim": ssim_c,
                "embed_latency_ms": embed_ms_c,
                "extract_latency_ms": extract_ms_c,
            },
            "delta_b_vs_a": delta_b,
            "delta_c_vs_a": delta_c,
        }
        records.append(carrier_record)

        if (idx + 1) % 10 == 0 or idx == 0 or not exact_match_a or not exact_match_b or not exact_match_c:
            logger.info(
                f"[{idx+1:03d}/{num_images}] {filename} | "
                f"A: Match={exact_match_a} (HdrErr={hdr_bit_errs_a}, MaxCW={max_cw_err_a}) | "
                f"B(3x): Match={exact_match_b} ({delta_b}, PostHdr={post_hdr_errs_b}, MaxCW={max_cw_err_b}) | "
                f"C(RS24): Match={exact_match_c} ({delta_c}, Cor={corrected_c}, MaxCW={max_cw_err_c})"
            )

    t_elapsed = time.time() - t_start_total
    logger.info(f"100-Carrier qualification completed in {t_elapsed:.2f} seconds.")

    # 3. Post-execution Protected File Hashes Verification
    post_hashes = {}
    for p in PROTECTED_PATHS:
        h = compute_file_sha256(p)
        post_hashes[str(p.relative_to(REPO_ROOT))] = h
        pre_h = pre_hashes[str(p.relative_to(REPO_ROOT))]
        if h != pre_h:
            raise RuntimeError(f"CRITICAL SAFETY VIOLATION: Protected file {p.name} mutated! Pre: {pre_h}, Post: {h}")
        logger.info(f"Post-hash {p.name} [VERIFIED UNCHANGED]: {h}")

    # 4. Aggregations
    exact_a = sum(1 for r in records if r["scheme_a"]["exact_match"])
    exact_b = sum(1 for r in records if r["scheme_b"]["exact_match"])
    exact_c = sum(1 for r in records if r["scheme_c"]["exact_match"])

    hdr_valid_a_count = sum(1 for r in records if r["scheme_a"]["header_valid"])
    hdr_valid_b_count = sum(1 for r in records if r["scheme_b"]["header_valid"])
    hdr_valid_c_count = sum(1 for r in records if r["scheme_c"]["header_valid"])

    rs_pay_a_count = sum(1 for r in records if r["scheme_a"]["payload_rs_success"])
    rs_pay_b_count = sum(1 for r in records if r["scheme_b"]["payload_rs_success"])
    rs_pay_c_count = sum(1 for r in records if r["scheme_c"]["payload_rs_success"])

    fails_a = [r["filename"] for r in records if not r["scheme_a"]["exact_match"]]
    fails_b = [r["filename"] for r in records if not r["scheme_b"]["exact_match"]]
    fails_c = [r["filename"] for r in records if not r["scheme_c"]["exact_match"]]

    fixed_by_b = [r["filename"] for r in records if r["delta_b_vs_a"] == "FIXED"]
    fixed_by_c = [r["filename"] for r in records if r["delta_c_vs_a"] == "FIXED"]
    regressed_by_b = [r["filename"] for r in records if r["delta_b_vs_a"] == "REGRESSED"]
    regressed_by_c = [r["filename"] for r in records if r["delta_c_vs_a"] == "REGRESSED"]

    summary = {
        "metadata": {
            "evaluation_title": "PaladinMedia Full 100-Carrier Header ECC Qualification",
            "date": "2026-10-29",
            "num_carriers": num_images,
            "secret_payload_bytes": payload_len,
            "protected_payload_bytes": protected_len,
            "total_subband_positions": 65536,
            "elapsed_seconds": round(t_elapsed, 2),
            "protected_hashes_verified": True,
            "pre_post_sha256": pre_hashes,
        },
        "overall_comparison": {
            "scheme_a_unprotected_128b": {
                "name": "Scheme A: Unprotected 16-byte Header (128 bits anchor)",
                "exact_recovery_count": exact_a,
                "exact_recovery_rate_pct": round(exact_a / num_images * 100.0, 2),
                "header_valid_count": hdr_valid_a_count,
                "header_valid_rate_pct": round(hdr_valid_a_count / num_images * 100.0, 2),
                "payload_rs_decode_count": rs_pay_a_count,
                "payload_rs_decode_rate_pct": round(rs_pay_a_count / num_images * 100.0, 2),
                "failures": fails_a,
                "psnr_stats": compute_stats([r["scheme_a"]["psnr_db"] for r in records]),
                "ssim_stats": compute_stats([r["scheme_a"]["ssim"] for r in records]),
                "header_ber_stats": compute_stats([r["scheme_a"]["header_ber"] for r in records]),
                "payload_ber_stats": compute_stats([r["scheme_a"]["payload_ber"] for r in records]),
                "embed_latency_ms": compute_stats([r["scheme_a"]["embed_latency_ms"] for r in records]),
                "extract_latency_ms": compute_stats([r["scheme_a"]["extract_latency_ms"] for r in records]),
            },
            "scheme_b_repetition_3x": {
                "name": "Scheme B: 3x Bit Repetition Majority Code (384 bits anchor)",
                "exact_recovery_count": exact_b,
                "exact_recovery_rate_pct": round(exact_b / num_images * 100.0, 2),
                "header_valid_count": hdr_valid_b_count,
                "header_valid_rate_pct": round(hdr_valid_b_count / num_images * 100.0, 2),
                "payload_rs_decode_count": rs_pay_b_count,
                "payload_rs_decode_rate_pct": round(rs_pay_b_count / num_images * 100.0, 2),
                "failures": fails_b,
                "fixed_carriers_vs_a": fixed_by_b,
                "regressed_carriers_vs_a": regressed_by_b,
                "psnr_stats": compute_stats([r["scheme_b"]["psnr_db"] for r in records]),
                "ssim_stats": compute_stats([r["scheme_b"]["ssim"] for r in records]),
                "header_channel_ber_stats": compute_stats([r["scheme_b"]["header_channel_ber"] for r in records]),
                "header_post_ber_stats": compute_stats([r["scheme_b"]["header_post_ber"] for r in records]),
                "payload_ber_stats": compute_stats([r["scheme_b"]["payload_ber"] for r in records]),
                "embed_latency_ms": compute_stats([r["scheme_b"]["embed_latency_ms"] for r in records]),
                "extract_latency_ms": compute_stats([r["scheme_b"]["extract_latency_ms"] for r in records]),
            },
            "scheme_c_rs24_16": {
                "name": "Scheme C: Shortened Reed-Solomon RS(24, 16) Codec (192 bits anchor)",
                "exact_recovery_count": exact_c,
                "exact_recovery_rate_pct": round(exact_c / num_images * 100.0, 2),
                "header_valid_count": hdr_valid_c_count,
                "header_valid_rate_pct": round(hdr_valid_c_count / num_images * 100.0, 2),
                "payload_rs_decode_count": rs_pay_c_count,
                "payload_rs_decode_rate_pct": round(rs_pay_c_count / num_images * 100.0, 2),
                "failures": fails_c,
                "fixed_carriers_vs_a": fixed_by_c,
                "regressed_carriers_vs_a": regressed_by_c,
                "psnr_stats": compute_stats([r["scheme_c"]["psnr_db"] for r in records]),
                "ssim_stats": compute_stats([r["scheme_c"]["ssim"] for r in records]),
                "header_channel_ber_stats": compute_stats([r["scheme_c"]["header_channel_ber"] for r in records]),
                "payload_ber_stats": compute_stats([r["scheme_c"]["payload_ber"] for r in records]),
                "embed_latency_ms": compute_stats([r["scheme_c"]["embed_latency_ms"] for r in records]),
                "extract_latency_ms": compute_stats([r["scheme_c"]["extract_latency_ms"] for r in records]),
            },
        },
        "per_carrier_results": records,
    }

    # Write Output JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved full JSON results to: {OUTPUT_JSON}")

    # Write Flattened Comparison CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "filename",
            "a_match", "a_hdr_errs", "a_max_cw_err", "a_psnr", "a_ssim",
            "b_match", "b_delta", "b_post_hdr_errs", "b_max_cw_err", "b_psnr", "b_ssim",
            "c_match", "c_delta", "c_rs_corrected", "c_max_cw_err", "c_psnr", "c_ssim",
        ])
        for r in records:
            writer.writerow([
                r["image_id"],
                r["filename"],
                r["scheme_a"]["exact_match"],
                r["scheme_a"]["header_raw_errors"],
                r["scheme_a"]["max_codeword_errors"],
                round(r["scheme_a"]["psnr_db"], 2),
                round(r["scheme_a"]["ssim"], 6),
                r["scheme_b"]["exact_match"],
                r["delta_b_vs_a"],
                r["scheme_b"]["header_post_errors_128"],
                r["scheme_b"]["max_codeword_errors"],
                round(r["scheme_b"]["psnr_db"], 2),
                round(r["scheme_b"]["ssim"], 6),
                r["scheme_c"]["exact_match"],
                r["delta_c_vs_a"],
                r["scheme_c"]["header_rs_corrected_bytes"],
                r["scheme_c"]["max_codeword_errors"],
                round(r["scheme_c"]["psnr_db"], 2),
                round(r["scheme_c"]["ssim"], 6),
            ])
    logger.info(f"Saved flattened CSV comparison to: {OUTPUT_CSV}")

    # Log Final Summary
    logger.info("=" * 80)
    logger.info("FINAL QUALIFICATION VERDICT")
    logger.info("=" * 80)
    logger.info(f"Scheme A (Unprotected):  {exact_a}/{num_images} ({exact_a/num_images*100:.1f}%) | Failures: {fails_a}")
    logger.info(f"Scheme B (Repetition 3x): {exact_b}/{num_images} ({exact_b/num_images*100:.1f}%) | Fixed: {fixed_by_b} | Regressed: {regressed_by_b} | Failures: {fails_b}")
    logger.info(f"Scheme C (RS 24, 16):    {exact_c}/{num_images} ({exact_c/num_images*100:.1f}%) | Fixed: {fixed_by_c} | Regressed: {regressed_by_c} | Failures: {fails_c}")


if __name__ == "__main__":
    main()
