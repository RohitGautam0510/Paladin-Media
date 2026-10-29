"""Experiment E04: Carrier Spatial Scaling & Content Analysis via DIV2K.

Evaluates carrier capacity and fidelity across target dimensions and DIV2K content.
Target resolutions:
- 512x512: Natively supported via deterministic center crops from DIV2K LR_bicubic_X2.
- 1024x1024, 2048x2048, 3072x3072, 4096x4096, 5120x5120: Documented as SKIPPED / UNAVAILABLE
  due to lack of native HR images (max native dimension is 1020 px; upscaling prohibited).
"""

import io
import json
import os
import sys
import time
import numpy as np
from PIL import Image
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.carrier.capacity import CapacityCalculator
from src.carrier.dwt import HaarDWT2D
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


def extract_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def run_e04():
    div2k_dir = os.path.join("DIV2K_train_LR_bicubic_X2", "DIV2K_train_LR_bicubic", "X2")
    secret_path = os.path.join("tests", "fixtures", "secret_small.png")
    key = b"01234567890123456789012345678901"
    aad = b"PALADIN_DIV2K_E04"

    with open(secret_path, "rb") as f:
        secret_bytes = f.read()

    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.IMAGE,
        metadata={"filename": "secret_small.png", "experiment": "E04"},
    )
    serialized = SecretSerializer.serialize(secret_payload)
    crypto = AESGCMProvider()
    encrypted = crypto.encrypt(serialized, key, aad=aad)
    ecc = ReedSolomonEncoder()
    protected = ecc.encode(encrypted)

    # 1. Theoretical Scaling Table
    target_resolutions = [512, 1024, 2048, 3072, 4096, 5120]
    scaling_analysis = []

    for dim in target_resolutions:
        subband_dim = dim // 2
        raw_bits = subband_dim * subband_dim
        raw_bytes = raw_bits // 8
        max_protected = raw_bytes - 16
        # Formula for max usable secret:
        # RS blocks = floor((max_protected - 4) / 255)
        # encrypted = RS_blocks * 223
        # secret = encrypted - 28 - 18
        rs_blocks = (max_protected - 4) // 255
        max_encrypted = rs_blocks * 223
        usable_secret = max(0, max_encrypted - 28 - 18)

        # Memory estimation: 3 channels * float64 subbands + spatial arrays
        # ~ (dim * dim * 3 * 8 * 5) bytes = ~ 120 bytes per pixel
        est_mem_mb = round((dim * dim * 3 * 8 * 4) / (1024 * 1024), 2)

        supported_natively = (dim <= 512)
        status = "EXECUTED (DIV2K Crop)" if supported_natively else "SKIPPED (HR Unavailable)"
        skip_reason = None if supported_natively else (
            f"Native maximum resolution in DIV2K_train_LR_bicubic_X2 is 1020 px. "
            f"Square crop of {dim}x{dim} cannot be formed without upscaling. "
            f"Per Rule 2/3, synthetic upscaling is strictly prohibited."
        )

        scaling_analysis.append({
            "dimension": f"{dim}x{dim}",
            "subband_dimension": f"{subband_dim}x{subband_dim}",
            "raw_positions": raw_bits,
            "raw_capacity_bytes": raw_bytes,
            "max_protected_bytes": max_protected,
            "usable_secret_capacity_bytes": usable_secret,
            "estimated_dwt_memory_mb": est_mem_mb,
            "status": status,
            "skip_reason": skip_reason,
        })

    # 2. Empirical Execution on Supported DIV2K 512x512 Images
    test_div2k_images = ["0001x2.png", "0002x2.png", "0003x2.png", "0004x2.png", "0005x2.png"]
    empirical_results = []

    for img_name in test_div2k_images:
        img_path = os.path.join(div2k_dir, img_name)
        if not os.path.exists(img_path):
            continue

        with Image.open(img_path) as full_img:
            orig_size = full_img.size
            crop_512 = extract_center_crop(full_img.convert("RGB"), 512)

        crop_arr = np.array(crop_512, dtype=np.uint8)

        t0 = time.perf_counter()
        embed_result = BaselineEmbedder.embed(
            cover_image=crop_arr,
            protected_payload=protected,
            session_counter=100 + int(img_name[:4]),
        )
        t_embed = time.perf_counter() - t0

        # PNG save / reload
        t0 = time.perf_counter()
        png_buf = io.BytesIO()
        embed_result.stego_image.save(png_buf, format="PNG")
        reloaded_img = Image.open(io.BytesIO(png_buf.getvalue())).convert("RGB")
        reloaded_arr = np.array(reloaded_img, dtype=np.uint8)
        t_io = time.perf_counter() - t0

        # Extraction
        t0 = time.perf_counter()
        header, extracted_protected = Extractor.extract(reloaded_img)
        decrypted_enc = ecc.decode(extracted_protected)
        decrypted_serialized = crypto.decrypt(decrypted_enc, key, aad=aad)
        recovered = SecretSerializer.deserialize(decrypted_serialized)
        t_extract = time.perf_counter() - t0

        psnr_val = calculate_psnr(crop_arr, reloaded_arr)
        ssim_val = calculate_ssim(crop_arr, reloaded_arr)
        diff_arr = np.abs(crop_arr.astype(np.int32) - reloaded_arr.astype(np.int32))
        mae_val = float(np.mean(diff_arr))

        exact_match = (recovered.data == secret_bytes)

        empirical_results.append({
            "image_id": img_name,
            "original_dimensions": list(orig_size),
            "crop_dimensions": [512, 512, 3],
            "crop_coordinates": {
                "left": (orig_size[0] - 512) // 2,
                "top": (orig_size[1] - 512) // 2,
                "width": 512,
                "height": 512,
            },
            "psnr_db": round(psnr_val, 4),
            "ssim": round(ssim_val, 6),
            "mae": round(mae_val, 6),
            "exact_match": exact_match,
            "embed_time_ms": round(t_embed * 1000, 2),
            "io_time_ms": round(t_io * 1000, 2),
            "extract_time_ms": round(t_extract * 1000, 2),
        })

    report_data = {
        "experiment_id": "E04",
        "description": "Carrier Spatial Scaling & DIV2K Image Content Analysis",
        "scaling_analysis": scaling_analysis,
        "empirical_div2k_runs": empirical_results,
    }

    out_file = os.path.join("docs", "experiments", "e04_results.json")
    with open(out_file, "w") as f:
        json.dump(report_data, f, indent=2)

    print("\n=== E04 SCALING ANALYSIS & DIV2K RUNS ===")
    print("Scaling Table:")
    for row in scaling_analysis:
        print(f"  {row['dimension']}: Usable Capacity = {row['usable_secret_capacity_bytes']} B | Status: {row['status']}")
    print("\nEmpirical DIV2K Runs (512x512 center crops):")
    for r in empirical_results:
        print(f"  {r['image_id']}: PSNR={r['psnr_db']} dB, SSIM={r['ssim']}, ExactMatch={r['exact_match']}")
    print(f"Saved E04 results to {out_file}")

    return report_data


if __name__ == "__main__":
    run_e04()
