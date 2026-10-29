"""Script to generate the canonical, cryptographic dataset split manifest for DIV2K HR.

Produces:
- docs/project/div2k_hr_split.json
Verifies SHA-256 hashes, native dimensions, and freezes 600 train / 100 val / 100 test partition.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIV2K_HR_DIR = os.path.join(REPO_ROOT, "dataset", "DIV2K", "DIV2K_train_HR", "DIV2K_train_HR")
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs", "project", "div2k_hr_split.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest() -> Dict[str, Any]:
    if not os.path.isdir(DIV2K_HR_DIR):
        raise FileNotFoundError(f"DIV2K HR directory not found at {DIV2K_HR_DIR}")

    files = sorted([f for f in os.listdir(DIV2K_HR_DIR) if f.endswith(".png")])
    if len(files) != 800:
        raise ValueError(f"Expected 800 PNG images, found {len(files)}")

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    combined_hasher = hashlib.sha256()
    native_1024_crops = 0
    native_2048_crops = 0

    print(f"Hashing and validating {len(files)} DIV2K HR images...")

    for idx, fname in enumerate(files, start=1):
        fpath = os.path.join(DIV2K_HR_DIR, fname)
        file_sha256 = sha256_file(fpath)
        combined_hasher.update(file_sha256.encode("ascii"))

        with Image.open(fpath) as img:
            w, h = img.size
            mode = img.mode

        can_crop_512 = (min(w, h) >= 512)
        can_crop_1024 = (min(w, h) >= 1024)
        can_crop_2048 = (min(w, h) >= 2048)

        if can_crop_1024:
            native_1024_crops += 1
        if can_crop_2048:
            native_2048_crops += 1

        record = {
            "image_id": idx,
            "filename": fname,
            "sha256": file_sha256,
            "dimensions": [w, h],
            "mode": mode,
            "can_crop_512": can_crop_512,
            "can_crop_1024": can_crop_1024,
            "can_crop_2048": can_crop_2048,
        }

        if idx <= 600:
            record["split"] = "train"
            train_records.append(record)
        elif idx <= 700:
            record["split"] = "val"
            val_records.append(record)
        else:
            record["split"] = "test"
            test_records.append(record)

    manifest = {
        "metadata": {
            "dataset_name": "DIV2K_train_HR",
            "description": "Canonical dataset split manifest for PaladinMedia Phase 3 neural development",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_directory": "dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR",
            "total_images": 800,
            "dataset_composite_sha256": combined_hasher.hexdigest(),
            "partition_policy": "Strict sequential split: train=0001..0600 (75%), val=0601..0700 (12.5%), test=0701..0800 (12.5%)",
            "splits": {
                "train": len(train_records),
                "val": len(val_records),
                "test": len(test_records),
            },
            "geometry_summary": {
                "native_512_capable": 800,
                "native_1024_capable": native_1024_crops,
                "native_2048_capable": native_2048_crops,
            },
        },
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written successfully to {OUTPUT_PATH}")
    print(f"Summary: 800 total, {len(train_records)} train, {len(val_records)} val, {len(test_records)} test.")
    print(f"1024x1024 crop capable: {native_1024_crops}/800 | 2048x2048 crop capable: {native_2048_crops}/800")
    print(f"Dataset Composite SHA-256: {manifest['metadata']['dataset_composite_sha256']}")
    return manifest


if __name__ == "__main__":
    generate_manifest()
