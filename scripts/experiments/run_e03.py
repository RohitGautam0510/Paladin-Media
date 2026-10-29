"""Experiment E03: MVP0 Baseline End-to-End Validation.

Executes the real PaladinMedia Phase 2 pipeline on MVP 0 dataset:
Cover: MVP 0 Dataset/0001x2.png (512x512 RGB)
Secret: tests/fixtures/secret_small.png (24x24 RGB PNG)
"""

import io
import json
import os
import sys
import time
import numpy as np
from PIL import Image
import psutil

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.carrier.capacity import CapacityCalculator, PayloadCapacityExceededError
from src.carrier.dwt import HaarDWT2D
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.core.media_pipeline import MediaStegoPipeline, calculate_psnr, calculate_ssim
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


def run_e03():
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024 * 1024)

    cover_path = os.path.join("MVP 0 Dataset", "0001x2.png")
    secret_path = os.path.join("tests", "fixtures", "secret_small.png")
    key = b"01234567890123456789012345678901"
    aad = b"PALADIN_MEDIA_MVP0_E03"
    session_counter = 42

    print(f"Loading cover: {cover_path}")
    with Image.open(cover_path) as img:
        cover_img = img.convert("RGB")
    cover_arr = np.array(cover_img, dtype=np.uint8)

    print(f"Loading secret: {secret_path}")
    with open(secret_path, "rb") as f:
        secret_bytes = f.read()

    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.IMAGE,
        metadata={"filename": "secret_small.png", "experiment": "E03"},
    )

    t_start = time.perf_counter()

    # Step 1: Serialization
    t0 = time.perf_counter()
    serialized = SecretSerializer.serialize(secret_payload)
    t_serialize = time.perf_counter() - t0

    # Step 2: AES-256-GCM Encryption
    t0 = time.perf_counter()
    crypto = AESGCMProvider()
    encrypted = crypto.encrypt(serialized, key, aad=aad)
    t_encrypt = time.perf_counter() - t0

    # Step 3: Reed-Solomon ECC
    t0 = time.perf_counter()
    ecc = ReedSolomonEncoder()
    protected = ecc.encode(encrypted)
    t_ecc = time.perf_counter() - t0

    # Step 4: Capacity check
    cap_report = CapacityCalculator.check_capacity(len(protected))

    # Step 5: Pure DWT Invertibility Check (before embedding)
    subbands_orig = HaarDWT2D.forward(cover_arr)
    recon_arr = HaarDWT2D.inverse(subbands_orig)
    dwt_recon_diff = np.abs(cover_arr.astype(np.float64) - recon_arr)
    dwt_max_err = float(np.max(dwt_recon_diff))
    dwt_mse = float(np.mean(dwt_recon_diff**2))

    # Step 6: Transform-domain Embedding
    t0 = time.perf_counter()
    embed_result = BaselineEmbedder.embed(
        cover_image=cover_arr,
        protected_payload=protected,
        session_counter=session_counter,
    )
    t_embed = time.perf_counter() - t0

    # Step 7: PNG Round Trip (Save & Reload)
    t0 = time.perf_counter()
    png_buf = io.BytesIO()
    embed_result.stego_image.save(png_buf, format="PNG")
    stego_png_bytes = png_buf.getvalue()
    reloaded_img = Image.open(io.BytesIO(stego_png_bytes)).convert("RGB")
    reloaded_arr = np.array(reloaded_img, dtype=np.uint8)
    t_png_io = time.perf_counter() - t0

    # Measure Carrier Fidelity
    psnr_val = calculate_psnr(cover_arr, reloaded_arr)
    ssim_val = calculate_ssim(cover_arr, reloaded_arr)
    diff_arr = np.abs(cover_arr.astype(np.int32) - reloaded_arr.astype(np.int32))
    mae_val = float(np.mean(diff_arr))
    max_pixel_err = int(np.max(diff_arr))

    # Step 8: Extraction from Reloaded Stego Image
    t0 = time.perf_counter()
    extracted_header, extracted_protected = Extractor.extract(reloaded_img)
    t_extract = time.perf_counter() - t0

    # Raw Channel BER
    raw_embedded_bits = BaselineEmbedder.bytes_to_bits(protected)
    raw_extracted_bits = BaselineEmbedder.bytes_to_bits(extracted_protected)
    bit_errors = int(np.sum(raw_embedded_bits != raw_extracted_bits))
    ber = float(bit_errors / len(raw_embedded_bits)) if len(raw_embedded_bits) > 0 else 0.0

    # Step 9: Reed-Solomon Decode
    t0 = time.perf_counter()
    decrypted_enc = ecc.decode(extracted_protected)
    t_rs_decode = time.perf_counter() - t0

    # Step 10: AES-GCM Decrypt & Authenticate
    t0 = time.perf_counter()
    decrypted_serialized = crypto.decrypt(decrypted_enc, key, aad=aad)
    t_decrypt = time.perf_counter() - t0

    # Step 11: Deserialization
    t0 = time.perf_counter()
    recovered_payload = SecretSerializer.deserialize(decrypted_serialized)
    t_deserialize = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start
    mem_after = proc.memory_info().rss / (1024 * 1024)

    # Verification
    byte_match = (recovered_payload.data == secret_bytes)
    media_match = (recovered_payload.media_type == secret_payload.media_type)
    meta_match = (recovered_payload.metadata == secret_payload.metadata)
    exact_recovery = byte_match and media_match and meta_match

    # Step 12: Negative Test (Hard Gate Enforcement)
    print("Executing Hard Gate negative test (oversized payload)...")
    hard_gate_passed = False
    try:
        oversized_payload = b"X" * 10000
        CapacityCalculator.check_capacity(len(oversized_payload))
    except PayloadCapacityExceededError:
        hard_gate_passed = True

    results = {
        "experiment_id": "E03",
        "carrier_path": cover_path,
        "carrier_dimensions": list(cover_arr.shape),
        "secret_path": secret_path,
        "secret_dimensions": [24, 24, 3],
        "secret_plaintext_bytes": len(secret_bytes),
        "serialized_bytes": len(serialized),
        "encrypted_bytes": len(encrypted),
        "protected_payload_bytes": len(protected),
        "carrier_header_bytes": CarrierHeader.HEADER_SIZE,
        "total_embedded_bits": (len(protected) + CarrierHeader.HEADER_SIZE) * 8,
        "raw_capacity_bytes": cap_report.raw_capacity_bytes,
        "max_protected_bytes": cap_report.max_protected_bytes,
        "usable_secret_capacity_bytes": cap_report.usable_capacity_bytes,
        "remaining_capacity_bytes": cap_report.remaining_capacity_bytes,
        "capacity_status": cap_report.status,
        "dwt_max_error": dwt_max_err,
        "dwt_mse": dwt_mse,
        "stego_psnr_db": round(psnr_val, 4),
        "stego_ssim": round(ssim_val, 6),
        "stego_mae": round(mae_val, 6),
        "stego_max_pixel_error": max_pixel_err,
        "bit_errors": bit_errors,
        "ber": ber,
        "header_session_counter": extracted_header.session_counter,
        "header_payload_length": extracted_header.payload_length,
        "rs_outcome": "SUCCESS",
        "aes_outcome": "AUTHENTICATED",
        "exact_byte_equality": exact_recovery,
        "hard_gate_negative_test": hard_gate_passed,
        "timing_ms": {
            "serialize": round(t_serialize * 1000, 3),
            "encrypt": round(t_encrypt * 1000, 3),
            "ecc_encode": round(t_ecc * 1000, 3),
            "embed": round(t_embed * 1000, 3),
            "png_io": round(t_png_io * 1000, 3),
            "extract": round(t_extract * 1000, 3),
            "rs_decode": round(t_rs_decode * 1000, 3),
            "decrypt": round(t_decrypt * 1000, 3),
            "deserialize": round(t_deserialize * 1000, 3),
            "total_roundtrip": round(t_total * 1000, 3),
        },
        "memory_mb": {
            "initial": round(mem_before, 2),
            "final": round(mem_after, 2),
            "delta": round(mem_after - mem_before, 2),
        },
    }

    out_file = os.path.join("docs", "experiments", "e03_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== E03 EXPERIMENT RESULTS ===")
    print(f"Status: SUCCESS")
    print(f"Exact Byte Recovery: {exact_recovery}")
    print(f"PSNR: {psnr_val:.4f} dB | SSIM: {ssim_val:.6f} | MAE: {mae_val:.6f} | MaxPixelErr: {max_pixel_err}")
    print(f"DWT/IDWT Invertibility Max Error: {dwt_max_err:.8f}")
    print(f"Bit Error Rate (BER): {ber:.6f} ({bit_errors} errors)")
    print(f"Protected Payload: {len(protected)} B / {cap_report.max_protected_bytes} B max")
    print(f"Hard Gate Negative Test: {'PASSED' if hard_gate_passed else 'FAILED'}")
    print(f"Total Roundtrip Runtime: {t_total*1000:.2f} ms")
    print(f"Saved results to: {out_file}")

    return results


if __name__ == "__main__":
    run_e03()
