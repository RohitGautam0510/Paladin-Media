"""Experiment Script: Phase 3 Sprint 2.1 Spatial Bridge and ECC Gate Validation.

Validates the Phase 3 neural spatial bridge against byte-level recovery requirements:
1. Controlled Reed-Solomon symbol error correction limits (below, at, and above t=16).
2. Wire framing prefix corruption handling.
3. Full real security pipeline roundtrip:
   SecretPayload -> SecretSerializer -> AES-256-GCM -> ReedSolomon -> SpatialBridge ->
   IDWT -> uint8 -> PNG save/reload -> Extract -> RS decode -> AES decrypt -> Deserialize.
4. Clipping causality proof: spatial block analysis of every extracted bit error.
5. Capacity, mask occupancy, and fidelity metric validation.

Outputs:
- docs/experiments/sprint2_1_ecc_validation_results.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core.exceptions import ECCDecodeError
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


def load_carrier_crop(image_path: Path, crop_size: int = 512) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Loads a 512x512 deterministic center crop from a DIV2K carrier image."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    crop = img.crop((left, top, left + crop_size, top + crop_size))

    arr = np.array(crop, dtype=np.uint8)
    metadata = {
        "filename": image_path.name,
        "original_size": (w, h),
        "crop_size": (crop_size, crop_size),
        "mean_brightness": float(arr.mean()),
        "std_brightness": float(arr.std()),
        "min_pixel": int(arr.min()),
        "max_pixel": int(arr.max()),
    }

    tensor = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    return tensor, metadata


def run_controlled_ecc_tests(ecc: ReedSolomonEncoder) -> List[Dict[str, Any]]:
    """Tests RS error correction limits (t=16) with controlled symbol corruption."""
    print("\n" + "=" * 80)
    print("TASK 2A: CONTROLLED REED-SOLOMON ERROR CORRECTION LIMITS")
    print("=" * 80)

    results = []
    data = os.urandom(200)  # Single RS block (k=223)
    encoded = bytearray(ecc.encode(data))  # 4B header + 232B block

    test_error_counts = [0, 1, 4, 8, 12, 15, 16, 17, 18, 20]

    for err_count in test_error_counts:
        corrupted = bytearray(encoded)
        # Corrupt err_count distinct bytes in the block body (offset >= 4)
        for i in range(err_count):
            corrupted[4 + i] ^= 0xFF

        status = "UNKNOWN"
        corrected_symbols = 0
        exact_match = False
        error_msg = ""

        try:
            res = ecc.decode_with_result(bytes(corrupted))
            status = "SUCCESS"
            corrected_symbols = res.corrected_symbols
            exact_match = (res.data == data)
        except ECCDecodeError as exc:
            status = "FAILED_FAIL_CLOSED"
            error_msg = str(exc)

        expected_status = "SUCCESS" if err_count <= ecc.t else "FAILED_FAIL_CLOSED"
        passed = (status == expected_status) and (exact_match if status == "SUCCESS" else True)

        entry = {
            "error_symbols_injected": err_count,
            "max_correctable_t": ecc.t,
            "within_limit": err_count <= ecc.t,
            "status": status,
            "corrected_symbols": corrected_symbols,
            "exact_match": exact_match,
            "error_msg": error_msg,
            "test_passed": passed,
        }
        results.append(entry)

        res_str = "OK (Corrected)" if status == "SUCCESS" else "REJECTED (Fail-Closed)"
        print(
            f"Errors: {err_count:2d} / t={ecc.t} | Status: {res_str:<22} | "
            f"Corrected: {corrected_symbols:2d} | ExactMatch: {exact_match!s:<5} | Passed: {passed}"
        )

    # Test Wire Framing Prefix Corruption
    corrupted_hdr = bytearray(encoded)
    corrupted_hdr[0] ^= 0x01
    hdr_status = "UNKNOWN"
    try:
        ecc.decode(bytes(corrupted_hdr))
        hdr_status = "UNNOTICED_CORRUPTION"
    except ECCDecodeError as exc:
        hdr_status = "REJECTED_FAIL_CLOSED"

    hdr_passed = (hdr_status == "REJECTED_FAIL_CLOSED")
    print(f"\nWire Framing Prefix Tamper Test: {hdr_status} | Passed: {hdr_passed}")
    results.append({
        "test_name": "framing_prefix_tamper",
        "status": hdr_status,
        "test_passed": hdr_passed,
    })

    return results


def run_real_payload_roundtrip_tests(
    carriers: List[Tuple[torch.Tensor, Dict[str, Any]]],
    bridge: NeuralSpatialBridge,
    crypto: AESGCMProvider,
    ecc: ReedSolomonEncoder,
) -> List[Dict[str, Any]]:
    """Exercises the complete end-to-end security and spatial bridge pipeline."""
    print("\n" + "=" * 80)
    print("TASK 3: REAL PAYLOAD ROUNDTRIP VALIDATION ACROSS MULTIPLE SIZES")
    print("=" * 80)

    key = bytes.fromhex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
    aad = b"PaladinMedia-Sprint2.1-VerificationContext"

    # Diverse payload sizes: 500 B, 1500 B, 3500 B, 6000 B
    payload_sizes = [500, 1500, 3500, 6000]
    results = []

    header_fmt = (
        f"{'Carrier':<10} | {'Payload Size':<12} | {'Occupancy':<10} | "
        f"{'Pre-ECC BER':<12} | {'Bit Errs':<8} | {'Worst Blk Err':<13} | "
        f"{'RS Decode':<10} | {'AES Tag':<8} | {'ExactMatch':<10} | {'PSNR (dB)':<9}"
    )
    print(header_fmt)
    print("-" * len(header_fmt))

    for carrier_tensor, meta in carriers:
        img_name = meta["filename"]

        for p_size in payload_sizes:
            # Deterministic pseudo-random payload content
            np.random.seed(p_size)
            raw_data = bytes(np.random.randint(0, 256, size=p_size, dtype=np.uint8))

            secret_payload = SecretPayload(
                data=raw_data,
                media_type=MediaType.IMAGE,
                metadata={"test": "sprint_2_1", "size": p_size, "carrier": img_name},
            )

            # Step 1: Serialization
            serialized = SecretSerializer.serialize(secret_payload)

            # Step 2: AES-256-GCM Encryption
            ciphertext_envelope = crypto.encrypt(serialized, key, aad=aad)

            # Step 3: Reed-Solomon Protection
            protected_bytes = ecc.encode(ciphertext_envelope)

            # Step 4: Bit Conversion & Subband Mapping
            bits_1d = np.unpackbits(np.frombuffer(protected_bytes, dtype=np.uint8))
            num_bits = len(bits_1d)
            occupancy = num_bits / 65536.0

            sub_dim = 256
            mask = torch.zeros(1, 1, sub_dim, sub_dim)
            payload_tensor = torch.zeros(1, 1, sub_dim, sub_dim)
            mask.view(-1)[:num_bits] = 1.0
            payload_tensor.view(-1)[:num_bits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

            # Step 5: Spatial Embedding -> IDWT -> Quantize -> PNG Roundtrip -> Extract
            emb_res, ext_res, metrics = bridge.run_full_roundtrip(
                carrier=carrier_tensor,
                payload_bits=(payload_tensor > 0).to(torch.uint8),
                mask=mask,
                reload_png=True,
                image_id=img_name,
            )

            # Step 6: Extract Bits & Measure Errors
            extracted_bits = ext_res.extracted_binary.view(-1)[:num_bits].cpu().numpy().astype(np.uint8)
            bit_diff = (bits_1d != extracted_bits)
            bit_err_count = int(bit_diff.sum())
            pre_ecc_ber = bit_err_count / num_bits

            # Step 7: Repack Bytes & Per-Codeword Error Distribution
            extracted_bytes = np.packbits(extracted_bits).tobytes()

            # Measure symbol errors per RS codeword
            # Framed wire structure: 4-byte length prefix + chunked blocks
            block_k = ecc.k  # 223
            block_len = block_k + ecc.ecc_symbols  # 255
            num_blocks = (len(protected_bytes) - 4) // block_len
            remainder_bytes = (len(protected_bytes) - 4) % block_len

            codeword_errors = []
            offset = 4
            for b_idx in range(num_blocks):
                orig_block = protected_bytes[offset : offset + block_len]
                rec_block = extracted_bytes[offset : offset + block_len]
                offset += block_len
                sym_errs = sum(1 for o, r in zip(orig_block, rec_block) if o != r)
                codeword_errors.append(sym_errs)

            if remainder_bytes > 0:
                orig_block = protected_bytes[offset : offset + remainder_bytes]
                rec_block = extracted_bytes[offset : offset + remainder_bytes]
                sym_errs = sum(1 for o, r in zip(orig_block, rec_block) if o != r)
                codeword_errors.append(sym_errs)

            worst_codeword_err = max(codeword_errors) if codeword_errors else 0

            # Step 8: RS Decoding
            rs_status = "UNKNOWN"
            rs_res = None
            try:
                rs_res = ecc.decode_with_result(extracted_bytes)
                rs_status = "SUCCESS"
            except ECCDecodeError:
                rs_status = "FAILED"

            # Step 9: AES-GCM Decryption & Tag Authentication
            aes_status = "SKIPPED"
            decrypted = b""
            if rs_status == "SUCCESS" and rs_res is not None:
                try:
                    decrypted = crypto.decrypt(rs_res.data, key, aad=aad)
                    aes_status = "VALID"
                except Exception:
                    aes_status = "FAILED"

            # Step 10: Deserialization & Exact Plaintext Equality
            exact_match = False
            if aes_status == "VALID":
                try:
                    recovered = SecretSerializer.deserialize(decrypted)
                    exact_match = (recovered.data == secret_payload.data) and (
                        recovered.metadata == secret_payload.metadata
                    )
                except Exception:
                    exact_match = False

            entry = {
                "carrier": img_name,
                "payload_plaintext_bytes": p_size,
                "protected_bytes": len(protected_bytes),
                "payload_bits": num_bits,
                "mask_occupancy": occupancy,
                "pre_ecc_bit_errors": bit_err_count,
                "pre_ecc_ber": pre_ecc_ber,
                "codeword_errors": codeword_errors,
                "worst_codeword_errors": worst_codeword_err,
                "rs_decode_status": rs_status,
                "rs_corrected_symbols": rs_res.corrected_symbols if rs_res else 0,
                "rs_blocks_count": rs_res.blocks_count if rs_res else len(codeword_errors),
                "aes_gcm_status": aes_status,
                "exact_plaintext_match": exact_match,
                "psnr_db": metrics.psnr_db,
                "ssim": metrics.ssim,
                "mae": metrics.mae,
                "num_clipped_total": metrics.clip_stats["num_clipped_total"],
            }
            results.append(entry)

            print(
                f"{img_name:<10} | {p_size:<12} | {occupancy*100:<9.1f}% | "
                f"{pre_ecc_ber:<12.6f} | {bit_err_count:<8} | {worst_codeword_err:<4} / 16 limit | "
                f"{rs_status:<10} | {aes_status:<8} | {str(exact_match):<10} | {metrics.psnr_db:<9.2f}"
            )

    return results


def run_clipping_causality_analysis(
    carriers: List[Tuple[torch.Tensor, Dict[str, Any]]],
    bridge: NeuralSpatialBridge,
) -> List[Dict[str, Any]]:
    """Analyzes spatial 2x2 blocks for every extracted bit error to prove clipping causality."""
    print("\n" + "=" * 80)
    print("TASK 4: CLIPPING CAUSALITY INVESTIGATION")
    print("=" * 80)

    results = []

    for carrier_tensor, meta in carriers:
        img_name = meta["filename"]

        # Run with full 100% capacity random payload
        torch.manual_seed(42)
        mask = torch.ones(1, 1, 256, 256)
        payload_bits = torch.randint(0, 2, (1, 1, 256, 256), dtype=torch.uint8)

        emb_res, ext_res, metrics = bridge.run_full_roundtrip(
            carrier=carrier_tensor,
            payload_bits=payload_bits,
            mask=mask,
            reload_png=True,
            image_id=img_name,
        )

        # 1. Identify all bit error coordinates (r, c)
        bit_diff = (ext_res.extracted_binary != payload_bits)
        err_indices = torch.nonzero(bit_diff.squeeze())  # (N, 2) coords (r, c)
        total_errs = int(err_indices.shape[0])

        # 2. Identify pre-clamp float pixels that were clipped
        stego_raw = emb_res.stego_raw_float[0]  # (3, 512, 512)
        red_raw = stego_raw[0]  # (512, 512) Red channel
        clipped_mask = (red_raw < 0.0) | (red_raw > 255.0)
        num_clipped_pixels = int(clipped_mask.sum().item())

        # 3. For each bit error at subband coordinate (r, c), check the corresponding 2x2 spatial block:
        # Spatial rows: [2r, 2r+1], spatial cols: [2c, 2c+1]
        errs_in_clipped_block = 0
        errs_in_unclipped_block = 0
        error_details = []

        for idx in range(total_errs):
            r = int(err_indices[idx, 0].item())
            c = int(err_indices[idx, 1].item())

            block_clipped = bool(clipped_mask[2 * r : 2 * r + 2, 2 * c : 2 * c + 2].any().item())
            if block_clipped:
                errs_in_clipped_block += 1
            else:
                errs_in_unclipped_block += 1

            if len(error_details) < 10:  # Sample up to 10 for inspection
                sub_pixels_raw = red_raw[2 * r : 2 * r + 2, 2 * c : 2 * c + 2].detach().cpu().numpy().tolist()
                error_details.append({
                    "subband_coord": (r, c),
                    "block_clipped": block_clipped,
                    "spatial_raw_2x2": sub_pixels_raw,
                })

        causality_ratio = (
            (errs_in_clipped_block / total_errs) if total_errs > 0 else 1.0
        )

        # 4. Compare with unmodulated control
        ctrl = bridge.evaluate_no_embedding_control(carrier_tensor, reload_png=True, image_id=img_name)

        entry = {
            "carrier": img_name,
            "min_raw_float": metrics.clip_stats["min_val_raw"],
            "max_raw_float": metrics.clip_stats["max_val_raw"],
            "clipped_pixels_count": num_clipped_pixels,
            "clipped_fraction": metrics.clip_stats["clip_fraction"],
            "total_bit_errors": total_errs,
            "errors_in_clipped_blocks": errs_in_clipped_block,
            "errors_in_unclipped_blocks": errs_in_unclipped_block,
            "clipping_causality_ratio": causality_ratio,
            "control_max_pixel_diff": ctrl["max_pixel_diff"],
            "sample_error_details": error_details,
        }
        results.append(entry)

        print(
            f"Carrier: {img_name:<10} | Raw Range: [{metrics.clip_stats['min_val_raw']:<6.1f}, {metrics.clip_stats['max_val_raw']:<6.1f}] | "
            f"Clipped Pixels: {num_clipped_pixels:<4} | Bit Errs: {total_errs:<4} | "
            f"Errs in Clipped: {errs_in_clipped_block:<4} | Errs in Unclipped: {errs_in_unclipped_block:<2} | "
            f"Causality: {causality_ratio*100:.1f}%"
        )

    return results


def main():
    dataset_dir = REPO_ROOT / "dataset" / "DIV2K" / "DIV2K_train_HR" / "DIV2K_train_HR"
    target_filenames = ["0001.png", "0002.png", "0004.png", "0005.png"]
    image_paths = [dataset_dir / f for f in target_filenames if (dataset_dir / f).exists()]

    if not image_paths:
        print("Error: Target DIV2K images not found.")
        sys.exit(1)

    carriers = [load_carrier_crop(p) for p in image_paths]

    bridge = NeuralSpatialBridge()
    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder()

    # Task 2: Controlled RS error correction limits
    ecc_results = run_controlled_ecc_tests(ecc)

    # Task 3: Full real security pipeline roundtrip
    roundtrip_results = run_real_payload_roundtrip_tests(carriers, bridge, crypto, ecc)

    # Task 4: Clipping causality investigation
    causality_results = run_clipping_causality_analysis(carriers, bridge)

    # Compile comprehensive machine-readable results
    output_data = {
        "metadata": {
            "experiment": "Phase 3 Sprint 2.1 Spatial Bridge and ECC Gate Validation",
            "date": "2026-10-18",
            "ecc_configuration": {
                "scheme": "Reed-Solomon GF(2^8)",
                "n": 255,
                "k": 223,
                "parity_symbols": 32,
                "max_correctable_symbols_t": 16,
            },
            "crypto_configuration": {
                "cipher": "AES-256-GCM",
                "nonce_bytes": 12,
                "tag_bytes": 16,
            },
        },
        "task2_controlled_ecc": ecc_results,
        "task3_real_payload_roundtrip": roundtrip_results,
        "task4_clipping_causality": causality_results,
    }

    output_dir = REPO_ROOT / "docs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "sprint2_1_ecc_validation_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[OK] Validation results saved to {out_file}")


if __name__ == "__main__":
    main()
