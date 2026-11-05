"""Held-Out ACS-Sub-INN Evaluation and Verification on Canonical DIV2K Test Split.

Sprint: Held-Out ACS-Sub-INN Evaluation and Readiness Verification
Date: 2026-10-29
Repository: E:\\Paladin_media
Baseline: checkpoints/phase4c_extended_30ep/best_model.pt
ONNX models: checkpoints/phase5a_onnx/
Canonical Split: docs/project/div2k_hr_split.json (test split: 0701.png - 0800.png, 100 images)

Safety & Invariant Constraints:
- STRICTLY ZERO MODEL TRAINING. Zero optimizer.step(), zero weight mutations.
- Checkpoint SHA-256 verified before and after execution.
- Canonical manifest integrity strictly audited.
- Dedicated unique artifact destination: docs/experiments/held_out_test_results_20261029.json.
- No existing result files or manifests overwritten.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("held_out_evaluation")

from src.carrier.capacity import CapacityCalculator
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import DeterministicMaskGenerator, MaskGenerationError
from src.carrier.modulation import NeuralModulationEngine
from src.core.exceptions import AuthenticationError, ECCDecodeError, PaladinSecurityError
from src.core.media_pipeline import MediaStegoPipeline, calculate_psnr, calculate_ssim
from src.core.serialization import MediaType, SecretPayload
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import (
    DEFAULT_DATASET_REL_PATH,
    DEFAULT_MANIFEST_REL_PATH,
    EXPECTED_COMPOSITE_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    DIV2KStegoDataset,
    build_stego_dataloader,
    validate_div2k_manifest,
)
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_engine import load_training_checkpoint
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "phase4c_extended_30ep" / "best_model.pt"
ONNX_DIR = REPO_ROOT / "checkpoints" / "phase5a_onnx"
OUTPUT_PATH = REPO_ROOT / "docs" / "experiments" / "held_out_test_results_20261029.json"
EXPECTED_PARAM_DIGEST = "ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92"
EXPECTED_CHECKPOINT_FILE_SHA256 = "32bfad806dbfcf057359c5f2f84a716832b88d43e9566ec331d0566ad446c4ee"

ONNX_MODEL_HASHES = {
    "acs_sub_inn_embedding_opset14.onnx": "3cd22ba35e55d4addfb89e59cdc695f1086383bef5d507f39a3c91dd4917c173",
    "acs_sub_inn_extraction_opset14.onnx": "ab696814635b39dec34b432b78e80bb949c5fcd68517a3cec5890102152ff8ba",
    "neural_spatial_embedding_opset14.onnx": "7cda5a3becb8283c6b84572e7adb6d64ea29ecbf9f1e667cf34dcd0324e2d784",
    "neural_spatial_extraction_opset14.onnx": "bf85d8404469d90c08aa50282db582e3c82b2c6ea2159293423ae6e846b82be2",
}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_param_digest(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def run_evaluation() -> Dict[str, Any]:
    logger.info("Starting Held-Out ACS-Sub-INN Evaluation on Canonical Test Split (0701..0800)...")

    # 1. Preflight Invariants & Verification
    logger.info("Executing Preflight Integrity Audits...")
    pre_ckpt_sha = compute_file_sha256(CHECKPOINT_PATH)
    logger.info(f"Pre-evaluation Checkpoint SHA-256: {pre_ckpt_sha}")
    if pre_ckpt_sha != EXPECTED_CHECKPOINT_FILE_SHA256:
        raise ValueError(f"Checkpoint file SHA mismatch: expected {EXPECTED_CHECKPOINT_FILE_SHA256}, got {pre_ckpt_sha}")

    manifest_audit = validate_div2k_manifest(
        manifest_path=REPO_ROOT / DEFAULT_MANIFEST_REL_PATH,
        dataset_root=REPO_ROOT / DEFAULT_DATASET_REL_PATH,
        enforce_file_existence=True,
    )
    logger.info(f"Manifest verified: {manifest_audit['partition_counts']}")

    onnx_status = {}
    for fn, exp_hash in ONNX_MODEL_HASHES.items():
        p = ONNX_DIR / fn
        if not p.is_file():
            raise FileNotFoundError(f"Missing ONNX model: {p}")
        h = compute_file_sha256(p)
        if h != exp_hash:
            raise ValueError(f"ONNX model hash mismatch for {fn}: expected {exp_hash}, got {h}")
        onnx_status[fn] = {"sha256": h, "size_bytes": p.stat().st_size, "verified": True}

    py_model = ACSSubINN(gamma_min=1.0)
    py_bridge = NeuralSpatialBridge(acs_model=py_model)
    ckpt_meta = load_training_checkpoint(CHECKPOINT_PATH, bridge=py_bridge)
    param_digest = compute_param_digest(py_bridge.acs_model)
    logger.info(f"Parameter digest: {param_digest}")
    if param_digest != EXPECTED_PARAM_DIGEST:
        raise ValueError(f"Parameter digest mismatch: expected {EXPECTED_PARAM_DIGEST}, got {param_digest}")

    py_bridge.eval()
    py_bridge.to("cpu")

    test_dataset = DIV2KStegoDataset(
        split="test",
        dataset_root=REPO_ROOT / DEFAULT_DATASET_REL_PATH,
        manifest_path=REPO_ROOT / DEFAULT_MANIFEST_REL_PATH,
        crop_size=512,
        random_crop=False,
    )
    num_test_images = len(test_dataset)
    logger.info(f"Loaded {num_test_images} canonical test images (IDs 0701-0800).")
    if num_test_images != 100:
        raise ValueError(f"Expected 100 test images, got {num_test_images}")

    # Constant Test Evaluation Secret Payload (1,000 bytes)
    payload_len = 1000
    secret_bytes = bytes(((i * 37 + 13) % 256) for i in range(payload_len))
    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.TEXT,
        metadata={"filename": "held_out_test_payload.txt", "benchmark": "test_0701_0800"},
    )
    aes_key = hashlib.sha256(b"PaladinMedia-HeldOut-Evaluation-SecretKey-2026").digest()
    aad = b"PaladinMedia-HeldOut-Test-Session"

    sender_onnx_q = MediaStegoPipeline.create_qualified_deployment(backend="onnx")
    receiver_onnx_q = MediaStegoPipeline(backend="onnx")

    sender_onnx_base = MediaStegoPipeline(backend="onnx")
    receiver_onnx_base = MediaStegoPipeline(backend="onnx")

    sender_py_ref = MediaStegoPipeline(
        backend="pytorch",
        spatial_bridge=py_bridge,
        ecc_encoder=ReedSolomonEncoder(parity_symbols=48),
    )
    receiver_py_ref = MediaStegoPipeline(backend="pytorch", spatial_bridge=py_bridge)

    rs_encoder_48 = ReedSolomonEncoder(parity_symbols=48)
    rs_encoder_32 = ReedSolomonEncoder(parity_symbols=32)

    per_image_results: List[Dict[str, Any]] = []

    onnx_q_exact_matches = 0
    onnx_q_aes_successes = 0
    onnx_q_header_successes = 0
    onnx_q_payload_rs_successes = 0

    onnx_base_exact_matches = 0
    onnx_base_payload_rs_successes = 0

    py_ref_exact_matches = 0
    py_ref_payload_rs_successes = 0

    onnx_q_psnr_list: List[float] = []
    onnx_q_ssim_list: List[float] = []
    onnx_q_header_ber_list: List[float] = []
    onnx_q_payload_ber_list: List[float] = []
    onnx_q_overall_ber_list: List[float] = []
    onnx_q_max_codeword_errors_list: List[int] = []
    onnx_q_embed_ms_list: List[float] = []
    onnx_q_extract_ms_list: List[float] = []

    onnx_base_max_codeword_errors_list: List[int] = []
    onnx_base_payload_ber_list: List[float] = []

    py_ref_psnr_list: List[float] = []
    py_ref_ssim_list: List[float] = []
    py_ref_payload_ber_list: List[float] = []
    py_ref_max_codeword_errors_list: List[int] = []

    logger.info("Beginning 100-image evaluation loop...")
    t_loop_start = time.time()

    for idx in range(num_test_images):
        rec = test_dataset[idx]
        img_id = rec["image_id"]
        filename = rec["filename"]
        carrier_t = rec["carrier"]
        cover_np = carrier_t.permute(1, 2, 0).numpy().astype(np.uint8)
        session_counter = int(img_id)

        # -----------------------------------------------------------------
        # EVAL 1: Primary Qualified Production (ONNX Runtime, RS 2t=48, t=24)
        # -----------------------------------------------------------------
        t0 = time.perf_counter()
        stego_res_onnx_q = sender_onnx_q.protect_and_embed(
            secret_payload=secret_payload,
            cover_image=cover_np,
            key=aes_key,
            aad=aad,
            session_counter=session_counter,
        )
        t_embed_onnx_q = (time.perf_counter() - t0) * 1000.0

        reloaded_arr_q = np.array(Image.open(io.BytesIO(stego_res_onnx_q.stego_png_bytes)).convert("RGB"), dtype=np.uint8)

        engine_onnx_q = sender_onnx_q.modulation_engine
        actual_hdr_bits_q = engine_onnx_q.last_header_bits
        actual_payload_bits_q = engine_onnx_q.last_payload_bits
        actual_protected_bytes_q = Extractor.bits_to_bytes(actual_payload_bits_q)

        bit_extractor_q = engine_onnx_q.begin_extraction(reloaded_arr_q, subband_dim=256)
        hdr_positions_q = DeterministicMaskGenerator.generate_header_positions(256)
        extracted_hdr_bits_q = bit_extractor_q.extract_bits(hdr_positions_q)
        hdr_bit_errs_q = int(np.sum(extracted_hdr_bits_q != actual_hdr_bits_q))
        hdr_ber_q = float(hdr_bit_errs_q) / len(actual_hdr_bits_q)

        header_valid_q = (hdr_bit_errs_q == 0)
        unpacked_header_q = None
        header_error_str_q = None
        try:
            unpacked_header_q = CarrierHeader.unpack(Extractor.bits_to_bytes(extracted_hdr_bits_q))
        except Exception as e:
            header_error_str_q = f"{type(e).__name__}: {e}"

        payload_positions_gt_q = DeterministicMaskGenerator.generate_payload_positions(
            session_counter=session_counter,
            payload_length_bytes=stego_res_onnx_q.header.payload_length,
            subband_dim=256,
        )
        extracted_payload_bits_gt_q = bit_extractor_q.extract_bits(payload_positions_gt_q)
        extracted_protected_bytes_gt_q = Extractor.bits_to_bytes(extracted_payload_bits_gt_q)

        raw_payload_errs_q = int(np.sum(extracted_payload_bits_gt_q != actual_payload_bits_q))
        payload_ber_q = float(raw_payload_errs_q) / len(actual_payload_bits_q)

        total_bits_q = len(actual_hdr_bits_q) + len(actual_payload_bits_q)
        total_errors_q = hdr_bit_errs_q + raw_payload_errs_q
        overall_ber_q = float(total_errors_q) / total_bits_q

        k_48 = rs_encoder_48.k
        block_len_48 = k_48 + rs_encoder_48.ecc_symbols
        offset = 4
        num_blocks_48 = (len(actual_protected_bytes_q) - 4) // block_len_48 + (1 if (len(actual_protected_bytes_q) - 4) % block_len_48 > 0 else 0)
        per_codeword_errors_q = []
        for blk_idx in range(num_blocks_48):
            cur_len = block_len_48 if (offset + block_len_48 <= len(actual_protected_bytes_q)) else (len(actual_protected_bytes_q) - offset)
            ext_block = extracted_protected_bytes_gt_q[offset : offset + cur_len]
            act_block = actual_protected_bytes_q[offset : offset + cur_len]
            diff_syms = sum(1 for a, b in zip(ext_block, act_block) if a != b)
            per_codeword_errors_q.append(diff_syms)
            offset += cur_len

        max_codeword_err_q = max(per_codeword_errors_q) if per_codeword_errors_q else 0
        correctable_margin_q = rs_encoder_48.t - max_codeword_err_q

        payload_rs_success_q = False
        try:
            rs_res_q = rs_encoder_48.decode_with_result(extracted_protected_bytes_gt_q)
            payload_rs_success_q = True
            rs_total_corrected_q = rs_res_q.corrected_symbols
        except Exception:
            payload_rs_success_q = False
            rs_total_corrected_q = sum(per_codeword_errors_q)

        if payload_rs_success_q:
            onnx_q_payload_rs_successes += 1
        if header_valid_q:
            onnx_q_header_successes += 1

        t1 = time.perf_counter()
        exact_match_onnx_q = False
        aes_auth_onnx_q = False
        receiver_error_q = None
        try:
            recovery_onnx_q = receiver_onnx_q.extract_and_unprotect(
                stego_carrier=stego_res_onnx_q.stego_png_bytes,
                key=aes_key,
                aad=aad,
                reference_payload=secret_payload,
            )
            exact_match_onnx_q = bool(recovery_onnx_q.exact_match)
            aes_auth_onnx_q = True
        except Exception as e:
            receiver_error_q = f"{type(e).__name__}: {e}"
        t_extract_onnx_q = (time.perf_counter() - t1) * 1000.0

        if exact_match_onnx_q:
            onnx_q_exact_matches += 1
        if aes_auth_onnx_q:
            onnx_q_aes_successes += 1

        onnx_q_psnr_list.append(stego_res_onnx_q.psnr_db)
        onnx_q_ssim_list.append(stego_res_onnx_q.ssim)
        onnx_q_header_ber_list.append(hdr_ber_q)
        onnx_q_payload_ber_list.append(payload_ber_q)
        onnx_q_overall_ber_list.append(overall_ber_q)
        onnx_q_max_codeword_errors_list.append(max_codeword_err_q)
        onnx_q_embed_ms_list.append(t_embed_onnx_q)
        onnx_q_extract_ms_list.append(t_extract_onnx_q)

        # -----------------------------------------------------------------
        # EVAL 2: Baseline Deployment Profile (ONNX Runtime, RS 2t=32, t=16)
        # -----------------------------------------------------------------
        stego_res_onnx_base = sender_onnx_base.protect_and_embed(
            secret_payload=secret_payload,
            cover_image=cover_np,
            key=aes_key,
            aad=aad,
            session_counter=session_counter,
        )
        reloaded_arr_base = np.array(Image.open(io.BytesIO(stego_res_onnx_base.stego_png_bytes)).convert("RGB"), dtype=np.uint8)

        engine_onnx_base = sender_onnx_base.modulation_engine
        actual_payload_bits_base = engine_onnx_base.last_payload_bits
        actual_protected_bytes_base = Extractor.bits_to_bytes(actual_payload_bits_base)

        bit_extractor_base = engine_onnx_base.begin_extraction(reloaded_arr_base, subband_dim=256)
        payload_positions_gt_base = DeterministicMaskGenerator.generate_payload_positions(
            session_counter=session_counter,
            payload_length_bytes=stego_res_onnx_base.header.payload_length,
            subband_dim=256,
        )
        extracted_payload_bits_base = bit_extractor_base.extract_bits(payload_positions_gt_base)
        extracted_protected_bytes_base = Extractor.bits_to_bytes(extracted_payload_bits_base)

        raw_payload_errs_base = int(np.sum(extracted_payload_bits_base != actual_payload_bits_base))
        payload_ber_base = float(raw_payload_errs_base) / len(actual_payload_bits_base)

        k_32 = rs_encoder_32.k
        block_len_32 = k_32 + rs_encoder_32.ecc_symbols
        offset = 4
        num_blocks_32 = (len(actual_protected_bytes_base) - 4) // block_len_32 + (1 if (len(actual_protected_bytes_base) - 4) % block_len_32 > 0 else 0)
        per_codeword_errors_base = []
        for blk_idx in range(num_blocks_32):
            cur_len = block_len_32 if (offset + block_len_32 <= len(actual_protected_bytes_base)) else (len(actual_protected_bytes_base) - offset)
            ext_block = extracted_protected_bytes_base[offset : offset + cur_len]
            act_block = actual_protected_bytes_base[offset : offset + cur_len]
            diff_syms = sum(1 for a, b in zip(ext_block, act_block) if a != b)
            per_codeword_errors_base.append(diff_syms)
            offset += cur_len

        max_codeword_err_base = max(per_codeword_errors_base) if per_codeword_errors_base else 0

        payload_rs_success_base = False
        try:
            rs_res_base = rs_encoder_32.decode_with_result(extracted_protected_bytes_base)
            payload_rs_success_base = True
        except Exception:
            payload_rs_success_base = False

        if payload_rs_success_base:
            onnx_base_payload_rs_successes += 1

        exact_match_onnx_base = False
        try:
            recovery_onnx_base = receiver_onnx_base.extract_and_unprotect(
                stego_carrier=stego_res_onnx_base.stego_png_bytes,
                key=aes_key,
                aad=aad,
                reference_payload=secret_payload,
            )
            exact_match_onnx_base = bool(recovery_onnx_base.exact_match)
        except Exception:
            exact_match_onnx_base = False

        if exact_match_onnx_base:
            onnx_base_exact_matches += 1

        onnx_base_payload_ber_list.append(payload_ber_base)
        onnx_base_max_codeword_errors_list.append(max_codeword_err_base)

        # -----------------------------------------------------------------
        # EVAL 3: PyTorch Reference Model (PyTorch, RS 2t=48, t=24)
        # -----------------------------------------------------------------
        stego_res_py = sender_py_ref.protect_and_embed(
            secret_payload=secret_payload,
            cover_image=cover_np,
            key=aes_key,
            aad=aad,
            session_counter=session_counter,
        )
        reloaded_arr_py = np.array(Image.open(io.BytesIO(stego_res_py.stego_png_bytes)).convert("RGB"), dtype=np.uint8)

        engine_py = sender_py_ref.modulation_engine
        actual_payload_bits_py = engine_py.last_payload_bits
        actual_protected_bytes_py = Extractor.bits_to_bytes(actual_payload_bits_py)

        bit_extractor_py = engine_py.begin_extraction(reloaded_arr_py, subband_dim=256)
        extracted_payload_bits_py = bit_extractor_py.extract_bits(payload_positions_gt_q)
        extracted_protected_bytes_py = Extractor.bits_to_bytes(extracted_payload_bits_py)

        raw_payload_errs_py = int(np.sum(extracted_payload_bits_py != actual_payload_bits_py))
        payload_ber_py = float(raw_payload_errs_py) / len(actual_payload_bits_py)

        offset = 4
        per_codeword_errors_py = []
        for blk_idx in range(num_blocks_48):
            cur_len = block_len_48 if (offset + block_len_48 <= len(actual_protected_bytes_py)) else (len(actual_protected_bytes_py) - offset)
            ext_block = extracted_protected_bytes_py[offset : offset + cur_len]
            act_block = actual_protected_bytes_py[offset : offset + cur_len]
            diff_syms = sum(1 for a, b in zip(ext_block, act_block) if a != b)
            per_codeword_errors_py.append(diff_syms)
            offset += cur_len

        max_codeword_err_py = max(per_codeword_errors_py) if per_codeword_errors_py else 0

        payload_rs_success_py = False
        try:
            rs_res_py = rs_encoder_48.decode_with_result(extracted_protected_bytes_py)
            payload_rs_success_py = True
        except Exception:
            payload_rs_success_py = False

        if payload_rs_success_py:
            py_ref_payload_rs_successes += 1

        exact_match_py = False
        try:
            recovery_py = receiver_py_ref.extract_and_unprotect(
                stego_carrier=stego_res_py.stego_png_bytes,
                key=aes_key,
                aad=aad,
                reference_payload=secret_payload,
            )
            exact_match_py = bool(recovery_py.exact_match)
        except Exception:
            exact_match_py = False

        if exact_match_py:
            py_ref_exact_matches += 1

        py_ref_psnr_list.append(stego_res_py.psnr_db)
        py_ref_ssim_list.append(stego_res_py.ssim)
        py_ref_payload_ber_list.append(payload_ber_py)
        py_ref_max_codeword_errors_list.append(max_codeword_err_py)

        # Per-Image Record
        img_record = {
            "image_id": img_id,
            "filename": filename,
            "original_dimensions": rec["original_dimensions"],
            "crop_box": list(rec["crop_box"]),
            "onnx_qualified_rs48": {
                "psnr_db": float(stego_res_onnx_q.psnr_db),
                "ssim": float(stego_res_onnx_q.ssim),
                "stego_png_size_bytes": len(stego_res_onnx_q.stego_png_bytes),
                "header_bit_errors": hdr_bit_errs_q,
                "header_ber": float(hdr_ber_q),
                "header_intact": header_valid_q,
                "header_error_msg": header_error_str_q,
                "payload_bit_errors": raw_payload_errs_q,
                "payload_ber": float(payload_ber_q),
                "overall_ber": float(overall_ber_q),
                "rs_per_codeword_errors": per_codeword_errors_q,
                "rs_max_codeword_error": max_codeword_err_q,
                "rs_correctable_margin": correctable_margin_q,
                "rs_max_allowed": rs_encoder_48.t,
                "payload_rs_decode_success": payload_rs_success_q,
                "receiver_exact_byte_match": exact_match_onnx_q,
                "receiver_aes_authenticated": aes_auth_onnx_q,
                "receiver_error": receiver_error_q,
                "embed_latency_ms": round(t_embed_onnx_q, 2),
                "extract_latency_ms": round(t_extract_onnx_q, 2),
            },
            "onnx_baseline_rs32": {
                "payload_ber": float(payload_ber_base),
                "rs_per_codeword_errors": per_codeword_errors_base,
                "rs_max_codeword_error": max_codeword_err_base,
                "rs_max_allowed": rs_encoder_32.t,
                "rs_exceeded_t16": max_codeword_err_base > rs_encoder_32.t,
                "payload_rs_decode_success": payload_rs_success_base,
                "receiver_exact_byte_match": exact_match_onnx_base,
            },
            "pytorch_reference_rs48": {
                "psnr_db": float(stego_res_py.psnr_db),
                "ssim": float(stego_res_py.ssim),
                "payload_ber": float(payload_ber_py),
                "rs_max_codeword_error": max_codeword_err_py,
                "payload_rs_decode_success": payload_rs_success_py,
                "receiver_exact_byte_match": exact_match_py,
            },
        }
        per_image_results.append(img_record)

        if (idx + 1) % 20 == 0 or idx == 0 or (idx + 1) == num_test_images:
            logger.info(
                f"Progress: [{idx+1:03d}/{num_test_images}] {filename} -> "
                f"ONNX(48): PSNR={stego_res_onnx_q.psnr_db:.2f}dB, HdrErr={hdr_bit_errs_q}, "
                f"PayloadBER={payload_ber_q:.5f}, MaxErr={max_codeword_err_q}/24, "
                f"PayloadRS={payload_rs_success_q}, ReceiverMatch={exact_match_onnx_q} | "
                f"Base(32): MaxErr={max_codeword_err_base}/16, RS={payload_rs_success_base}"
            )

    t_loop_total = time.time() - t_loop_start
    logger.info(f"Evaluation loop completed in {t_loop_total:.2f} seconds.")

    # Post-Evaluation Checkpoint Integrity Verification
    post_ckpt_sha = compute_file_sha256(CHECKPOINT_PATH)
    logger.info(f"Post-evaluation Checkpoint SHA-256: {post_ckpt_sha}")
    if post_ckpt_sha != pre_ckpt_sha:
        raise ValueError(
            f"CRITICAL INTEGRITY FAILURE: Checkpoint SHA-256 mutated during evaluation! "
            f"Pre: {pre_ckpt_sha}, Post: {post_ckpt_sha}"
        )
    logger.info("VERIFIED: Checkpoint was NOT modified (bit-for-bit identical pre/post evaluation).")

    post_param_digest = compute_param_digest(py_bridge.acs_model)
    if post_param_digest != EXPECTED_PARAM_DIGEST:
        raise ValueError(
            f"CRITICAL INTEGRITY FAILURE: Parameter digest mutated! "
            f"Expected {EXPECTED_PARAM_DIGEST}, got {post_param_digest}"
        )
    logger.info("VERIFIED: Model parameters remained completely frozen.")

    # Statistical Aggregation
    def compute_stats(arr: List[float]) -> Dict[str, float]:
        np_arr = np.array(arr)
        return {
            "mean": float(np.mean(np_arr)),
            "std": float(np.std(np_arr)),
            "min": float(np.min(np_arr)),
            "max": float(np.max(np_arr)),
            "median": float(np.median(np_arr)),
            "p05": float(np.percentile(np_arr, 5)),
            "p95": float(np.percentile(np_arr, 95)),
        }

    onnx_q_psnr_stats = compute_stats(onnx_q_psnr_list)
    onnx_q_ssim_stats = compute_stats(onnx_q_ssim_list)
    onnx_q_header_ber_stats = compute_stats(onnx_q_header_ber_list)
    onnx_q_payload_ber_stats = compute_stats(onnx_q_payload_ber_list)
    onnx_q_overall_ber_stats = compute_stats(onnx_q_overall_ber_list)
    onnx_q_err_stats = compute_stats([float(x) for x in onnx_q_max_codeword_errors_list])
    onnx_q_embed_time_stats = compute_stats(onnx_q_embed_ms_list)
    onnx_q_extract_time_stats = compute_stats(onnx_q_extract_ms_list)

    onnx_base_ber_stats = compute_stats(onnx_base_payload_ber_list)
    onnx_base_err_stats = compute_stats([float(x) for x in onnx_base_max_codeword_errors_list])

    py_ref_psnr_stats = compute_stats(py_ref_psnr_list)
    py_ref_ssim_stats = compute_stats(py_ref_ssim_list)
    py_ref_ber_stats = compute_stats(py_ref_payload_ber_list)
    py_ref_err_stats = compute_stats([float(x) for x in py_ref_max_codeword_errors_list])

    # Error Distribution Histograms
    err_hist_q = {str(err): int(sum(1 for e in onnx_q_max_codeword_errors_list if e == err)) for err in range(max(onnx_q_max_codeword_errors_list) + 1)}
    err_hist_base = {str(err): int(sum(1 for e in onnx_base_max_codeword_errors_list if e == err)) for err in range(max(onnx_base_max_codeword_errors_list) + 1)}

    # Identify Anomalies & Failure Modes
    header_corrupted_images = [
        {"filename": r["filename"], "header_errors": r["onnx_qualified_rs48"]["header_bit_errors"], "error_msg": r["onnx_qualified_rs48"]["receiver_error"]}
        for r in per_image_results if not r["onnx_qualified_rs48"]["header_intact"]
    ]

    payload_rs_failed_q_images = [
        {"filename": r["filename"], "max_codeword_error": r["onnx_qualified_rs48"]["rs_max_codeword_error"]}
        for r in per_image_results if not r["onnx_qualified_rs48"]["payload_rs_decode_success"]
    ]

    base_exceeded_t16_images = [
        {"filename": r["filename"], "max_codeword_error": r["onnx_baseline_rs32"]["rs_max_codeword_error"]}
        for r in per_image_results if r["onnx_baseline_rs32"]["rs_exceeded_t16"]
    ]

    # Worst Cases (ONNX Qualified RS 2t=48)
    worst_psnr = sorted(per_image_results, key=lambda r: r["onnx_qualified_rs48"]["psnr_db"])[:5]
    worst_payload_ber = sorted(per_image_results, key=lambda r: r["onnx_qualified_rs48"]["payload_ber"], reverse=True)[:5]
    worst_codeword_err = sorted(per_image_results, key=lambda r: r["onnx_qualified_rs48"]["rs_max_codeword_error"], reverse=True)[:5]

    # Validation reference from Phase 4C
    p4c_val_reference = {
        "dataset_split": "val (0601.png - 0700.png)",
        "sample_size": 100,
        "mean_psnr_db": 52.32,
        "mean_ssim": 0.99997,
        "pre_ecc_ber": 0.005118,
    }

    final_output = {
        "metadata": {
            "title": "PaladinMedia Held-Out ACS-Sub-INN Evaluation and Readiness Verification",
            "date": "2026-10-29",
            "eval_duration_sec": round(t_loop_total, 2),
            "device": "CPU",
            "pytorch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
        },
        "preflight_integrity": {
            "checkpoint_path": str(CHECKPOINT_PATH),
            "checkpoint_sha256_pre": pre_ckpt_sha,
            "checkpoint_sha256_post": post_ckpt_sha,
            "checkpoint_unmutated": (pre_ckpt_sha == post_ckpt_sha),
            "parameter_digest": param_digest,
            "parameter_digest_verified": (param_digest == EXPECTED_PARAM_DIGEST),
            "manifest_file_sha256": manifest_audit["file_sha256"],
            "manifest_composite_sha256": manifest_audit["composite_sha256"],
            "canonical_manifest_verified": manifest_audit["is_canonical"],
            "dataset_root": str(REPO_ROOT / DEFAULT_DATASET_REL_PATH),
            "onnx_models": onnx_status,
            "official_test_dir_div2k_test_hr_exists": False,
        },
        "test_dataset_partition": {
            "split": "test",
            "range": "0701.png - 0800.png",
            "image_count": num_test_images,
            "crop_size": 512,
            "payload_size_bytes": payload_len,
        },
        "primary_qualified_production_rs48": {
            "pipeline": "MediaStegoPipeline.create_qualified_deployment(backend='onnx')",
            "backend": "onnx",
            "ecc_profile": "rs_2t_48 (t=24, parity_symbols=48)",
            "exact_recovery_rate_pct": float(onnx_q_exact_matches / num_test_images * 100.0),
            "exact_recovery_count": onnx_q_exact_matches,
            "aes_gcm_auth_rate_pct": float(onnx_q_aes_successes / num_test_images * 100.0),
            "payload_rs_success_rate_pct": float(onnx_q_payload_rs_successes / num_test_images * 100.0),
            "header_intact_rate_pct": float(onnx_q_header_successes / num_test_images * 100.0),
            "total_images": num_test_images,
            "psnr_stats": onnx_q_psnr_stats,
            "ssim_stats": onnx_q_ssim_stats,
            "header_ber_stats": onnx_q_header_ber_stats,
            "payload_ber_stats": onnx_q_payload_ber_stats,
            "overall_ber_stats": onnx_q_overall_ber_stats,
            "max_codeword_error_stats": onnx_q_err_stats,
            "embed_latency_ms": onnx_q_embed_time_stats,
            "extract_latency_ms": onnx_q_extract_time_stats,
            "codeword_error_histogram": err_hist_q,
            "header_corrupted_cases": header_corrupted_images,
            "payload_rs_failed_cases": payload_rs_failed_q_images,
            "worst_psnr_cases": [
                {"filename": r["filename"], "psnr_db": r["onnx_qualified_rs48"]["psnr_db"]} for r in worst_psnr
            ],
            "worst_payload_ber_cases": [
                {"filename": r["filename"], "payload_ber": r["onnx_qualified_rs48"]["payload_ber"]} for r in worst_payload_ber
            ],
            "worst_codeword_error_cases": [
                {
                    "filename": r["filename"],
                    "max_codeword_error": r["onnx_qualified_rs48"]["rs_max_codeword_error"],
                    "per_codeword_errors": r["onnx_qualified_rs48"]["rs_per_codeword_errors"],
                    "margin": r["onnx_qualified_rs48"]["rs_correctable_margin"],
                }
                for r in worst_codeword_err
            ],
        },
        "baseline_deployment_rs32": {
            "pipeline": "MediaStegoPipeline(backend='onnx')",
            "backend": "onnx",
            "ecc_profile": "rs_2t_32 (t=16, parity_symbols=32)",
            "exact_recovery_rate_pct": float(onnx_base_exact_matches / num_test_images * 100.0),
            "exact_recovery_count": onnx_base_exact_matches,
            "payload_rs_success_rate_pct": float(onnx_base_payload_rs_successes / num_test_images * 100.0),
            "payload_ber_stats": onnx_base_ber_stats,
            "max_codeword_error_stats": onnx_base_err_stats,
            "codeword_error_histogram": err_hist_base,
            "exceeded_t16_images_count": len(base_exceeded_t16_images),
            "exceeded_t16_images": base_exceeded_t16_images,
        },
        "pytorch_reference_rs48": {
            "pipeline": "MediaStegoPipeline(backend='pytorch', ecc_encoder=RS(48))",
            "backend": "pytorch",
            "ecc_profile": "rs_2t_48 (t=24)",
            "exact_recovery_rate_pct": float(py_ref_exact_matches / num_test_images * 100.0),
            "payload_rs_success_rate_pct": float(py_ref_payload_rs_successes / num_test_images * 100.0),
            "psnr_stats": py_ref_psnr_stats,
            "ssim_stats": py_ref_ssim_stats,
            "payload_ber_stats": py_ref_ber_stats,
            "max_codeword_error_stats": py_ref_err_stats,
        },
        "held_out_vs_validation_comparison": {
            "validation_reference": p4c_val_reference,
            "held_out_test_onnx_q": {
                "dataset_split": "test (0701.png - 0800.png)",
                "sample_size": 100,
                "mean_psnr_db": onnx_q_psnr_stats["mean"],
                "mean_ssim": onnx_q_ssim_stats["mean"],
                "payload_ber": onnx_q_payload_ber_stats["mean"],
                "delta_psnr_db": onnx_q_psnr_stats["mean"] - p4c_val_reference["mean_psnr_db"],
                "delta_ssim": onnx_q_ssim_stats["mean"] - p4c_val_reference["mean_ssim"],
                "delta_ber": onnx_q_payload_ber_stats["mean"] - p4c_val_reference["pre_ecc_ber"],
            },
        },
        "per_image_results": per_image_results,
    }

    OUTPUT_PATH.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    logger.info(f"Successfully saved evaluation results to: {OUTPUT_PATH}")

    return final_output


if __name__ == "__main__":
    run_evaluation()
