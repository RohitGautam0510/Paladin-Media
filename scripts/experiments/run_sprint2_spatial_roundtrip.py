"""Experiment Script: Phase 3 Sprint 2 DWT/IDWT Spatial Bridge Roundtrip Benchmark.

Measures payload recovery, coefficient errors, residual decision margins, clipping,
and stego image fidelity after a complete lossless PNG container roundtrip across
representative DIV2K HR carrier images.

Outputs:
- docs/experiments/sprint2_spatial_roundtrip_results.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.spatial_bridge import NeuralSpatialBridge


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


def generate_mask_and_payload(
    sub_dim: int,
    payload_type: str,
    occupancy: float,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a spatial embedding mask and corresponding binary payload bits."""
    torch.manual_seed(seed)
    total_positions = sub_dim * sub_dim

    if occupancy >= 1.0:
        mask = torch.ones(1, 1, sub_dim, sub_dim, dtype=torch.float32)
    else:
        num_active = int(round(total_positions * occupancy))
        indices = torch.randperm(total_positions)[:num_active]
        mask_flat = torch.zeros(total_positions, dtype=torch.float32)
        mask_flat[indices] = 1.0
        mask = mask_flat.view(1, 1, sub_dim, sub_dim)

    if payload_type == "all_zeros":
        payload = torch.zeros(1, 1, sub_dim, sub_dim, dtype=torch.uint8)
    elif payload_type == "all_ones":
        payload = torch.ones(1, 1, sub_dim, sub_dim, dtype=torch.uint8)
    else:  # random
        payload = torch.randint(0, 2, (1, 1, sub_dim, sub_dim), dtype=torch.uint8)

    return mask, payload


def run_benchmark():
    dataset_dir = REPO_ROOT / "dataset" / "DIV2K" / "DIV2K_train_HR" / "DIV2K_train_HR"
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        sys.exit(1)

    # 5 Representative carrier images with diverse texture and dynamic range
    target_filenames = ["0001.png", "0002.png", "0003.png", "0004.png", "0005.png"]
    image_paths = [dataset_dir / fname for fname in target_filenames if (dataset_dir / fname).exists()]

    if not image_paths:
        print("Error: No target DIV2K images found.")
        sys.exit(1)

    print(f"Loaded {len(image_paths)} representative carrier images:")
    carriers_data = []
    for p in image_paths:
        tensor, meta = load_carrier_crop(p)
        carriers_data.append((tensor, meta))
        print(
            f"  - {meta['filename']}: Brightness mean={meta['mean_brightness']:.1f}, "
            f"std={meta['std_brightness']:.1f}, range=[{meta['min_pixel']}, {meta['max_pixel']}]"
        )

    all_results: Dict[str, Any] = {
        "metadata": {
            "experiment": "Sprint 2: DWT/IDWT Neural Spatial Bridge Lossless PNG Roundtrip",
            "date": "2026-10-18",
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "evaluated_images": [m["filename"] for _, m in carriers_data],
        },
        "controls": [],
        "experiments": [],
    }

    # =========================================================================
    # Part 1: No-Embedding Control (DWT -> IDWT -> PNG Reload)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: NO-EMBEDDING CONTROL (DWT -> IDWT -> Lossless PNG Reload)")
    print("=" * 80)

    control_bridge = NeuralSpatialBridge()
    for tensor, meta in carriers_data:
        ctrl = control_bridge.evaluate_no_embedding_control(
            tensor, reload_png=True, image_id=meta["filename"]
        )
        ctrl["carrier_metadata"] = meta
        all_results["controls"].append(ctrl)
        psnr_str = "inf" if ctrl["psnr_db"] == float("inf") else f"{ctrl['psnr_db']:.2f} dB"
        print(
            f"Control [{meta['filename']}]: PSNR={psnr_str}, SSIM={ctrl['ssim']:.6f}, "
            f"MAE={ctrl['mae']:.6f}, MaxPixDiff={ctrl['max_pixel_diff']}, "
            f"SubbandMaxErr={ctrl['max_subband_err']:.6f}, ExactMatch={ctrl['exact_bit_identity']}"
        )

    # =========================================================================
    # Part 2: Spatial Roundtrip Across Gain Floors and Carriers
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: SPATIAL EMBEDDING & ROUNDTRIP BENCHMARK")
    print("=" * 80)

    # Evaluated gain floors: 0.8 (sub-margin), 1.0 (baseline), 1.5, 2.0, 3.0
    gamma_floors = [0.8, 1.0, 1.5, 2.0, 3.0]
    payload_configs = [
        {"name": "100% (65,536 bits / 8,192 B)", "occupancy": 1.0, "type": "random"},
        {"name": "Phase 2 Matched (56,720 bits / 7,090 B)", "occupancy": 56720 / 65536, "type": "random"},
        {"name": "Low Load (12,000 bits / 1,500 B)", "occupancy": 12000 / 65536, "type": "random"},
        {"name": "All-Zeros (65,536 bits)", "occupancy": 1.0, "type": "all_zeros"},
        {"name": "All-Ones (65,536 bits)", "occupancy": 1.0, "type": "all_ones"},
    ]

    header_fmt = (
        f"{'Image':<10} | {'Payload Config':<36} | {'g_floor':<7} | "
        f"{'BER (pre-ECC)':<14} | {'Exact':<5} | {'PSNR (dB)':<9} | {'SSIM':<8} | "
        f"{'Clipped':<7} | {'MaxRedErr':<9} | {'GB-Drift':<8}"
    )
    print(header_fmt)
    print("-" * len(header_fmt))

    for tensor, meta in carriers_data:
        for p_cfg in payload_configs:
            mask, payload_bits = generate_mask_and_payload(
                sub_dim=256,
                payload_type=p_cfg["type"],
                occupancy=p_cfg["occupancy"],
                seed=42,
            )

            for g_floor in gamma_floors:
                # Initialize model with deterministic seed and specified gamma_min
                torch.manual_seed(42)
                acs_model = ACSSubINN(gamma_min=g_floor)
                bridge = NeuralSpatialBridge(acs_model=acs_model)

                emb_res, ext_res, metrics = bridge.run_full_roundtrip(
                    carrier=tensor,
                    payload_bits=payload_bits,
                    mask=mask,
                    reload_png=True,
                    image_id=meta["filename"],
                )

                # Unmasked Position Verification
                unmasked = (mask == 0.0)
                if unmasked.any():
                    # Compare unmasked red subband before vs after roundtrip
                    orig_red_hh = emb_res.subbands_original[3][:, 0:1, :, :]
                    rec_red_hh = ext_res.hh_drop_rec
                    unmasked_diff = (rec_red_hh[unmasked] - orig_red_hh[unmasked]).abs()
                    unmasked_max_err = float(unmasked_diff.max().item())
                else:
                    unmasked_max_err = 0.0

                result_entry = {
                    "image": meta["filename"],
                    "payload_config": p_cfg["name"],
                    "payload_type": p_cfg["type"],
                    "payload_bits": metrics.payload_bits_count,
                    "mask_occupancy": metrics.mask_occupancy,
                    "gamma_min": g_floor,
                    "gamma_stats": metrics.gamma_stats,
                    "clip_stats": metrics.clip_stats,
                    "green_blue_max_abs_err": metrics.green_blue_max_abs_err,
                    "red_hh_err_stats": metrics.red_hh_err_stats,
                    "context_drift_stats": metrics.context_drift_stats,
                    "combined_residual_err_stats": metrics.combined_residual_err_stats,
                    "ber_payload": metrics.ber_payload,
                    "exact_recovery": metrics.exact_recovery,
                    "unmasked_max_err": unmasked_max_err,
                    "psnr_db": metrics.psnr_db,
                    "ssim": metrics.ssim,
                    "mae": metrics.mae,
                    "max_pixel_diff": metrics.max_pixel_diff,
                    "reloaded_png": metrics.reloaded_png,
                }
                all_results["experiments"].append(result_entry)

                exact_str = "YES" if metrics.exact_recovery else "NO"
                print(
                    f"{meta['filename']:<10} | {p_cfg['name']:<36} | {g_floor:<7.1f} | "
                    f"{metrics.ber_payload:<14.6f} | {exact_str:<5} | {metrics.psnr_db:<9.2f} | "
                    f"{metrics.ssim:<8.5f} | {metrics.clip_stats['num_clipped_total']:<7} | "
                    f"{metrics.red_hh_err_stats['max']:<9.3f} | {metrics.green_blue_max_abs_err:<8.1e}"
                )

    # Save output JSON
    output_dir = REPO_ROOT / "docs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sprint2_spatial_roundtrip_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] Complete experiment results written to {output_path}")


if __name__ == "__main__":
    run_benchmark()
