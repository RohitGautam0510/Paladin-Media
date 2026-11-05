"""Diagnostic evaluation of ACS-Sub-INN ONNX pair vs PyTorch Reference and Spatial ONNX.

Compares:
1. PyTorch Reference: ACSSubINNEmbeddingGraph / ACSSubINNExtractionGraph
2. ACS-Sub-INN ONNX: acs_sub_inn_embedding_opset14.onnx / acs_sub_inn_extraction_opset14.onnx (with external Haar DWT/IDWT)
3. Spatial ONNX: neural_spatial_embedding_opset14.onnx / neural_spatial_extraction_opset14.onnx (internal Haar DWT/IDWT)

Carriers:
- Diagnostic subset: 0701.png (success), and the 5 failures: 0715.png, 0774.png, 0777.png, 0785.png, 0795.png

Output: docs/diagnostics_20261029/task3_inn_onnx_diagnostic.json
"""

import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.carrier.capacity import CapacityCalculator
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator
from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.core.pipeline import SecurityPipeline
from src.core.serialization import MediaType, SecretPayload
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import DEFAULT_DATASET_REL_PATH, DEFAULT_MANIFEST_REL_PATH, DIV2KStegoDataset
from src.neural.onnx_export import (
    ACSSubINNEmbeddingGraph,
    ACSSubINNExtractionGraph,
    NeuralSpatialEmbeddingGraph,
    NeuralSpatialExtractionGraph,
)
from src.neural.spatial_bridge import NeuralSpatialBridge, haar_dwt2d, haar_idwt2d
from src.neural.training_engine import load_training_checkpoint
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "phase4c_extended_30ep" / "best_model.pt"
ONNX_DIR = REPO_ROOT / "checkpoints" / "phase5a_onnx"
DIAG_DIR = REPO_ROOT / "docs" / "diagnostics_20261029"
OUTPUT_JSON = DIAG_DIR / "task3_inn_onnx_diagnostic.json"

def main():
    print("Loading models for Task 3 ACS-Sub-INN ONNX diagnostic...")

    # PyTorch setup
    py_model = ACSSubINN(gamma_min=1.0)
    py_bridge = NeuralSpatialBridge(acs_model=py_model)
    load_training_checkpoint(CHECKPOINT_PATH, bridge=py_bridge)
    py_bridge.eval()
    py_bridge.to("cpu")

    py_embed_coeff = ACSSubINNEmbeddingGraph(py_bridge.acs_model)
    py_extract_coeff = ACSSubINNExtractionGraph(py_bridge.acs_model)

    py_embed_spatial = NeuralSpatialEmbeddingGraph(py_bridge)
    py_extract_spatial = NeuralSpatialExtractionGraph(py_bridge)

    # ONNX setup
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_embed_coeff = ort.InferenceSession(
        str(ONNX_DIR / "acs_sub_inn_embedding_opset14.onnx"), sess_opts, providers=["CPUExecutionProvider"]
    )
    sess_extract_coeff = ort.InferenceSession(
        str(ONNX_DIR / "acs_sub_inn_extraction_opset14.onnx"), sess_opts, providers=["CPUExecutionProvider"]
    )
    sess_embed_spatial = ort.InferenceSession(
        str(ONNX_DIR / "neural_spatial_embedding_opset14.onnx"), sess_opts, providers=["CPUExecutionProvider"]
    )
    sess_extract_spatial = ort.InferenceSession(
        str(ONNX_DIR / "neural_spatial_extraction_opset14.onnx"), sess_opts, providers=["CPUExecutionProvider"]
    )

    # Security pipeline (RS 2t=48)
    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder(parity_symbols=48)
    sec_pipe = SecurityPipeline(crypto_provider=crypto, ecc_encoder=ecc)

    # Constant payload
    payload_len = 1000
    secret_bytes = bytes(((i * 37 + 13) % 256) for i in range(payload_len))
    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.TEXT,
        metadata={"filename": "task3_secret.txt"},
    )
    aes_key = hashlib.sha256(b"PaladinMedia-Task3-DiagnosticKey-2026").digest()
    aad = b"Task3-Diagnostic"

    protected_payload = sec_pipe.protect(secret_payload, aes_key, aad=aad)

    # Dataset
    test_dataset = DIV2KStegoDataset(split="test", crop_size=512, random_crop=False)
    
    # Target images: 0701 (clean baseline) + 0715, 0774, 0777, 0785, 0795 (failures)
    target_filenames = ["0701.png", "0715.png", "0774.png", "0777.png", "0785.png", "0795.png"]
    target_records = [r for r in test_dataset.records if r["filename"] in target_filenames]

    results = {}

    for rec in target_records:
        fn = rec["filename"]
        img_id = rec["image_id"]
        session_counter = int(img_id)
        
        # Load carrier tensor (3, 512, 512)
        idx = [i for i, r in enumerate(test_dataset.records) if r["filename"] == fn][0]
        data_item = test_dataset[idx]
        carrier_t = data_item["carrier"].unsqueeze(0)  # (1, 3, 512, 512)
        carrier_np = carrier_t.numpy()
        cover_arr = (carrier_t[0].permute(1, 2, 0).numpy()).astype(np.uint8)

        # Build CarrierHeader and bitstream
        hdr = CarrierHeader(
            session_counter=session_counter,
            payload_length=len(protected_payload),
            flags=CarrierHeader.ECC_PROFILE_RS_2T_48,
        )
        hdr_bytes = hdr.pack()
        hdr_bits = BaselineEmbedder.bytes_to_bits(hdr_bytes)
        pay_bits = BaselineEmbedder.bytes_to_bits(protected_payload)

        # Generate embedding masks
        hdr_pos = DeterministicMaskGenerator.generate_header_positions(256)
        pay_pos = DeterministicMaskGenerator.generate_payload_positions(session_counter, len(protected_payload), 256)

        sub_h, sub_w = 256, 256
        mask_arr = np.zeros((1, 1, sub_h, sub_w), dtype=np.float32)
        payload_arr = np.zeros((1, 1, sub_h, sub_w), dtype=np.float32)

        for bit, (r, c) in zip(hdr_bits, hdr_pos):
            mask_arr[0, 0, r, c] = 1.0
            payload_arr[0, 0, r, c] = 1.0 if int(bit) == 1 else -1.0

        for bit, (r, c) in zip(pay_bits, pay_pos):
            mask_arr[0, 0, r, c] = 1.0
            payload_arr[0, 0, r, c] = 1.0 if int(bit) == 1 else -1.0

        mask_t = torch.from_numpy(mask_arr)
        payload_t = torch.from_numpy(payload_arr)

        # -------------------------------------------------------------
        # Path 1: PyTorch Reference (ACSSubINNEmbeddingGraph + external DWT/IDWT)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        with torch.no_grad():
            ll_py, lh_py, hl_py, hh_py = haar_dwt2d(carrier_t)
            hh_drop_py = hh_py[:, 0:1, :, :]
            hh_keep_py = hh_py[:, 1:3, :, :]
            hh_stego_drop_py, ctx_py, gamma_py = py_embed_coeff(hh_drop_py, hh_keep_py, mask_t, payload_t)
            hh_stego_py = torch.cat([hh_stego_drop_py, hh_keep_py], dim=1)
            stego_raw_py = haar_idwt2d(ll_py, lh_py, hl_py, hh_stego_py)
            stego_clamped_py = torch.clamp(stego_raw_py, 0.0, 255.0)
        t_py_embed = (time.perf_counter() - t0) * 1000.0

        stego_arr_py = np.round(stego_clamped_py[0].permute(1, 2, 0).numpy()).astype(np.uint8)

        # PyTorch extraction
        t1 = time.perf_counter()
        with torch.no_grad():
            stego_t_py = torch.from_numpy(stego_arr_py.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            _, _, _, hh_rec_py = haar_dwt2d(stego_t_py)
            hh_drop_rec_py = hh_rec_py[:, 0:1, :, :]
            hh_keep_rec_py = hh_rec_py[:, 1:3, :, :]
            delta_rec_py, ctx_rec_py, bin_rec_py = py_extract_coeff(hh_drop_rec_py, hh_keep_rec_py)
        t_py_extract = (time.perf_counter() - t1) * 1000.0

        bin_grid_py = bin_rec_py[0, 0].numpy().astype(np.uint8)

        # -------------------------------------------------------------
        # Path 2: ACS-Sub-INN ONNX Pair (acs_sub_inn_*_opset14.onnx + external DWT/IDWT)
        # -------------------------------------------------------------
        t2 = time.perf_counter()
        # External DWT via torch or numpy
        with torch.no_grad():
            ll_onnx, lh_onnx, hl_onnx, hh_onnx = haar_dwt2d(carrier_t)
            hh_drop_np = hh_onnx[:, 0:1, :, :].numpy()
            hh_keep_np = hh_onnx[:, 1:3, :, :].numpy()

        ort_hh_stego_drop, ort_ctx, ort_gamma = sess_embed_coeff.run(
            None,
            {
                "hh_drop": hh_drop_np,
                "hh_keep": hh_keep_np,
                "mask": mask_arr,
                "payload_bipolar": payload_arr,
            },
        )
        # External IDWT
        with torch.no_grad():
            hh_stego_onnx = torch.cat([torch.from_numpy(ort_hh_stego_drop), hh_keep_onnx := torch.from_numpy(hh_keep_np)], dim=1)
            stego_raw_onnx = haar_idwt2d(ll_onnx, lh_onnx, hl_onnx, hh_stego_onnx)
            stego_clamped_onnx = torch.clamp(stego_raw_onnx, 0.0, 255.0)
        t_inn_onnx_embed = (time.perf_counter() - t2) * 1000.0

        stego_arr_inn_onnx = np.round(stego_clamped_onnx[0].permute(1, 2, 0).numpy()).astype(np.uint8)

        # ACS-Sub-INN ONNX extraction
        t3 = time.perf_counter()
        with torch.no_grad():
            stego_t_inn = torch.from_numpy(stego_arr_inn_onnx.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            _, _, _, hh_rec_inn = haar_dwt2d(stego_t_inn)
            hh_drop_rec_inn_np = hh_rec_inn[:, 0:1, :, :].numpy()
            hh_keep_rec_inn_np = hh_rec_inn[:, 1:3, :, :].numpy()

        ort_delta_rec, ort_ctx_rec, ort_bin_rec = sess_extract_coeff.run(
            None,
            {
                "hh_drop_stego": hh_drop_rec_inn_np,
                "hh_keep": hh_keep_rec_inn_np,
            },
        )
        t_inn_onnx_extract = (time.perf_counter() - t3) * 1000.0

        bin_grid_inn_onnx = ort_bin_rec[0, 0].astype(np.uint8)

        # -------------------------------------------------------------
        # Path 3: Spatial ONNX Pair (neural_spatial_*_opset14.onnx)
        # -------------------------------------------------------------
        t4 = time.perf_counter()
        ort_sp_stego, ort_sp_ctx, ort_sp_gamma = sess_embed_spatial.run(
            None,
            {
                "carrier": carrier_np,
                "mask": mask_arr,
                "payload_bipolar": payload_arr,
            },
        )
        t_sp_onnx_embed = (time.perf_counter() - t4) * 1000.0
        stego_arr_sp_onnx = np.round(ort_sp_stego[0].transpose(1, 2, 0)).astype(np.uint8)

        t5 = time.perf_counter()
        ort_sp_bin, ort_sp_delta = sess_extract_spatial.run(
            None,
            {"stego_carrier": ort_sp_stego},
        )
        t_sp_onnx_extract = (time.perf_counter() - t5) * 1000.0
        bin_grid_sp_onnx = ort_sp_bin[0, 0].astype(np.uint8)

        # -------------------------------------------------------------
        # Parity & Discrepancy Calculations
        # -------------------------------------------------------------
        # 1. Continuous Graph Discrepancy between PyTorch and ACS-Sub-INN ONNX (on hh_stego_drop)
        diff_drop = np.abs(hh_stego_drop_py.numpy() - ort_hh_stego_drop)
        max_abs_err_drop = float(np.max(diff_drop))
        denom_drop = np.abs(hh_stego_drop_py.numpy()) + 1e-7
        max_rel_err_drop = float(np.max(diff_drop / denom_drop))

        # 2. Continuous Graph Discrepancy between PyTorch and Spatial ONNX (on stego_clamped)
        diff_stego = np.abs(stego_clamped_py.numpy() - ort_sp_stego)
        max_abs_err_stego = float(np.max(diff_stego))
        max_rel_err_stego = float(np.max(diff_stego / (np.abs(stego_clamped_py.numpy()) + 1e-7)))

        # 3. Discrepancy between ACS-Sub-INN ONNX stego and Spatial ONNX stego
        diff_inn_vs_sp = np.abs(stego_arr_inn_onnx.astype(np.float32) - stego_arr_sp_onnx.astype(np.float32))
        max_abs_diff_inn_vs_sp = float(np.max(diff_inn_vs_sp))

        # 4. Binary Bit Agreement on Header and Payload Coordinates
        def extract_bits_from_grid(grid, positions):
            return np.array([int(grid[r, c]) for r, c in positions], dtype=np.uint8)

        # Header bits
        hdr_bits_py = extract_bits_from_grid(bin_grid_py, hdr_pos)
        hdr_bits_inn = extract_bits_from_grid(bin_grid_inn_onnx, hdr_pos)
        hdr_bits_sp = extract_bits_from_grid(bin_grid_sp_onnx, hdr_pos)

        # Payload bits
        pay_bits_py = extract_bits_from_grid(bin_grid_py, pay_pos)
        pay_bits_inn = extract_bits_from_grid(bin_grid_inn_onnx, pay_pos)
        pay_bits_sp = extract_bits_from_grid(bin_grid_sp_onnx, pay_pos)

        # Agreement
        hdr_agree_inn_vs_py = bool(np.array_equal(hdr_bits_inn, hdr_bits_py))
        pay_agree_inn_vs_py = bool(np.array_equal(pay_bits_inn, pay_bits_py))
        pay_agree_sp_vs_inn = bool(np.array_equal(pay_bits_sp, pay_bits_inn))

        # Bit errors against ground truth
        hdr_err_py = int(np.sum(hdr_bits_py != hdr_bits))
        hdr_err_inn = int(np.sum(hdr_bits_inn != hdr_bits))
        hdr_err_sp = int(np.sum(hdr_bits_sp != hdr_bits))

        pay_err_py = int(np.sum(pay_bits_py != pay_bits))
        pay_err_inn = int(np.sum(pay_bits_inn != pay_bits))
        pay_err_sp = int(np.sum(pay_bits_sp != pay_bits))

        ber_pay_py = float(pay_err_py) / len(pay_bits)
        ber_pay_inn = float(pay_err_inn) / len(pay_bits)
        ber_pay_sp = float(pay_err_sp) / len(pay_bits)

        # Reed-Solomon per-codeword symbol errors (ACS-Sub-INN ONNX)
        prot_bytes_inn = Extractor.bits_to_bytes(pay_bits_inn)
        k_48 = ecc.k
        block_len_48 = k_48 + ecc.ecc_symbols
        offset = 4
        num_blocks_48 = (len(protected_payload) - 4) // block_len_48 + (1 if (len(protected_payload) - 4) % block_len_48 > 0 else 0)
        per_cw_errors_inn = []
        for blk in range(num_blocks_48):
            cur_len = block_len_48 if (offset + block_len_48 <= len(protected_payload)) else (len(protected_payload) - offset)
            e_blk = prot_bytes_inn[offset : offset + cur_len]
            a_blk = protected_payload[offset : offset + cur_len]
            diff = sum(1 for a, b in zip(e_blk, a_blk) if a != b)
            per_cw_errors_inn.append(diff)
            offset += cur_len
        max_cw_err_inn = max(per_cw_errors_inn) if per_cw_errors_inn else 0

        # Attempt RS decode on INN ONNX bits
        rs_success_inn = False
        try:
            res_dec = ecc.decode_with_result(prot_bytes_inn)
            rs_success_inn = True
        except Exception:
            rs_success_inn = False

        # Attempt end-to-end unprotect if header intact
        e2e_match_inn = False
        if hdr_err_inn == 0:
            try:
                rec_pay = sec_pipe.unprotect(prot_bytes_inn, aes_key, aad=aad)
                e2e_match_inn = (rec_pay.data == secret_bytes)
            except Exception:
                e2e_match_inn = False

        # PSNR and SSIM
        psnr_py = calculate_psnr(cover_arr, stego_arr_py)
        ssim_py = calculate_ssim(cover_arr, stego_arr_py)

        psnr_inn = calculate_psnr(cover_arr, stego_arr_inn_onnx)
        ssim_inn = calculate_ssim(cover_arr, stego_arr_inn_onnx)

        psnr_sp = calculate_psnr(cover_arr, stego_arr_sp_onnx)
        ssim_sp = calculate_ssim(cover_arr, stego_arr_sp_onnx)

        results[fn] = {
            "image_id": img_id,
            "filename": fn,
            "continuous_parity_inn_vs_py": {
                "max_abs_err_drop": max_abs_err_drop,
                "max_rel_err_drop": max_rel_err_drop,
                "allclose_1e4_1e5": bool(np.allclose(hh_stego_drop_py.numpy(), ort_hh_stego_drop, rtol=1e-4, atol=1e-5)),
                "allclose_1e4_2e5": bool(np.allclose(hh_stego_drop_py.numpy(), ort_hh_stego_drop, rtol=1e-4, atol=2e-5)),
            },
            "continuous_parity_sp_vs_py": {
                "max_abs_err_stego": max_abs_err_stego,
                "max_rel_err_stego": max_rel_err_stego,
                "allclose_1e4_1e5": bool(np.allclose(stego_clamped_py.numpy(), ort_sp_stego, rtol=1e-4, atol=1e-5)),
                "allclose_1e4_2e5": bool(np.allclose(stego_clamped_py.numpy(), ort_sp_stego, rtol=1e-4, atol=2e-5)),
            },
            "inn_vs_sp_pixel_diff": {
                "max_abs_pixel_diff": max_abs_diff_inn_vs_sp,
                "mean_abs_pixel_diff": float(np.mean(diff_inn_vs_sp)),
            },
            "binary_payload_agreement": {
                "inn_onnx_vs_pytorch_header_identical": hdr_agree_inn_vs_py,
                "inn_onnx_vs_pytorch_payload_identical": pay_agree_inn_vs_py,
                "inn_onnx_vs_spatial_onnx_payload_identical": pay_agree_sp_vs_inn,
            },
            "header_bit_errors": {
                "pytorch": hdr_err_py,
                "acs_sub_inn_onnx": hdr_err_inn,
                "spatial_onnx": hdr_err_sp,
            },
            "payload_bit_errors": {
                "pytorch": pay_err_py,
                "acs_sub_inn_onnx": pay_err_inn,
                "spatial_onnx": pay_err_sp,
            },
            "payload_ber": {
                "pytorch": ber_pay_py,
                "acs_sub_inn_onnx": ber_pay_inn,
                "spatial_onnx": ber_pay_sp,
            },
            "acs_sub_inn_onnx_rs_codeword_errors": {
                "max_codeword_error": max_cw_err_inn,
                "per_codeword_errors": per_cw_errors_inn,
                "exceeded_t24": max_cw_err_inn > 24,
                "rs_decode_success": rs_success_inn,
            },
            "recovered_payload_equality": {
                "acs_sub_inn_onnx_exact_recovery": e2e_match_inn,
            },
            "stego_fidelity": {
                "psnr_pytorch": psnr_py,
                "psnr_acs_sub_inn_onnx": psnr_inn,
                "psnr_spatial_onnx": psnr_sp,
                "ssim_pytorch": ssim_py,
                "ssim_acs_sub_inn_onnx": ssim_inn,
                "ssim_spatial_onnx": ssim_sp,
            },
            "latency_ms": {
                "embed": {
                    "pytorch": round(t_py_embed, 2),
                    "acs_sub_inn_onnx_with_ext_dwt": round(t_inn_onnx_embed, 2),
                    "spatial_onnx_internal_dwt": round(t_sp_onnx_embed, 2),
                },
                "extract": {
                    "pytorch": round(t_py_extract, 2),
                    "acs_sub_inn_onnx_with_ext_dwt": round(t_inn_onnx_extract, 2),
                    "spatial_onnx_internal_dwt": round(t_sp_onnx_extract, 2),
                },
            },
        }

        print(f"[{fn}] Parity Drop MaxAbsErr: {max_abs_err_drop:.2e} | "
              f"HdrErrs: Py={hdr_err_py}, INN={hdr_err_inn}, Sp={hdr_err_sp} | "
              f"PayErrs: Py={pay_err_py}, INN={pay_err_inn}, Sp={pay_err_sp} | "
              f"MaxCWErr={max_cw_err_inn}/24 | E2E={e2e_match_inn}")

    summary = {
        "status": "PASS",
        "tested_carriers": target_filenames,
        "supported_inference_path_exists_in_production": False,
        "production_integration_status": (
            "No production integration for acs_sub_inn_*_opset14.onnx exists in MediaStegoPipeline or NeuralModulationEngine. "
            "NeuralModulationEngine hardcodes 'neural_spatial_*_opset14.onnx'. "
            "Evaluation was conducted via offline diagnostic pipeline executing external Haar DWT/IDWT."
        ),
        "results_by_image": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTask 3 Diagnostic completed! Results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
