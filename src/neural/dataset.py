"""DIV2K Dataset Loader and Training-Boundary Enforcement Module.

Implements the explicit, verifiable dataset boundary for Phase 3 neural steganography:
- validate_div2k_manifest: Audits manifest schema, composite hashes, split counts, disjointness,
  and filesystem presence against DIV2K High-Resolution images.
- DIV2KStegoDataset: PyTorch Dataset loading images strictly from versioned manifest partitions
  with deterministic cropping, lazy I/O, and path-traversal prevention.
- build_stego_dataloader: Factory constructing DataLoader instances with strict purpose-based
  split enforcement (preventing held-out test data from entering training or model selection).

Threat Model & Boundary Guarantees:
- Supported project workflows strictly enforce that only the 'train' partition can enter
  optimizer-facing training, and only 'val' can enter validation/model selection.
- Held-out 'test' images (IDs 0701-0800) cannot enter training or model selection workflows
  through the supported interface.
- Low-Level Bypass Disclaimer: In Python, raw internal functions (e.g. execute_training_step)
  can accept arbitrary caller-supplied tensors. Supported workflows use execute_verified_training_batch
  and build_stego_dataloader to enforce the split boundary fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Canonical default paths relative to repository root
DEFAULT_MANIFEST_REL_PATH = Path("docs") / "project" / "div2k_hr_split.json"
DEFAULT_DATASET_REL_PATH = Path("dataset") / "DIV2K" / "DIV2K_train_HR" / "DIV2K_train_HR"

# Expected manifest digests and partition counts
EXPECTED_COMPOSITE_SHA256 = "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed"
EXPECTED_MANIFEST_FILE_SHA256 = "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff"
EXPECTED_COUNTS: Dict[str, int] = {"train": 600, "val": 100, "test": 100}

# Filename pattern: exactly 4 digits followed by .png (e.g. '0001.png')
FILENAME_REGEX = re.compile(r"^\d{4}\.png$")


def normalize_split_name(split: str) -> str:
    """Normalizes split name aliases ('validation' -> 'val')."""
    if not isinstance(split, str):
        raise TypeError(f"Split must be string, got {type(split).__name__}")
    s = split.strip().lower()
    if s in ("val", "validation"):
        return "val"
    if s in ("train", "test"):
        return s
    raise ValueError(f"Invalid split name '{split}'. Supported splits: ('train', 'val', 'validation', 'test').")


def compute_manifest_composite_sha256(data: Dict[str, Any]) -> str:
    """Recomputes the canonical composite SHA-256 digest from partition records.

    The composite hash is defined as the SHA-256 digest of the sequential concatenation
    of the ASCII-encoded per-image SHA-256 digests, ordered strictly by positive image_id (1..N).
    """
    records: List[Dict[str, Any]] = []
    for split_key in ("train", "val", "test"):
        part = data.get(split_key, [])
        if isinstance(part, list):
            records.extend(part)

    def _sort_key(r: Any) -> int:
        if isinstance(r, dict):
            img_id = r.get("image_id", 0)
            if isinstance(img_id, int) and not isinstance(img_id, bool):
                return img_id
        return 0

    sorted_records = sorted(records, key=_sort_key)
    hasher = hashlib.sha256()
    for rec in sorted_records:
        if isinstance(rec, dict):
            rec_sha = rec.get("sha256")
            if isinstance(rec_sha, str):
                hasher.update(rec_sha.encode("ascii"))
    return hasher.hexdigest()


def validate_div2k_manifest(
    manifest_path: Union[str, Path],
    dataset_root: Optional[Union[str, Path]] = None,
    enforce_file_existence: bool = True,
    expected_composite_sha256: Optional[str] = EXPECTED_COMPOSITE_SHA256,
    expected_counts: Optional[Dict[str, int]] = None,
    expected_file_sha256: Optional[str] = None,
    is_custom_fixture: bool = False,
) -> Dict[str, Any]:
    """Audits manifest schema, file digest, composite hash, split counts, disjointness, and file presence.

    Integrity Enforcement Rules:
    - If manifest_path resolves to the canonical project manifest:
        * Exact file SHA-256 MUST match EXPECTED_MANIFEST_FILE_SHA256 unconditionally.
        * Composite hash MUST match EXPECTED_COMPOSITE_SHA256.
        * Split counts MUST match EXPECTED_COUNTS (600/100/100).
    - If manifest_path is non-canonical:
        * Fails closed unless is_custom_fixture=True (preventing accidental altered manifest injection).
        * If expected_file_sha256 is provided, validates exact digest.
    - Consistency Checks:
        * Each image_id must be a strictly positive integer (rejects booleans).
        * Numeric filename stem must match image_id exactly ('0001.png' <-> 1).
        * Record's declared 'split' field must agree with containing partition.
        * All three partition keys ('train', 'val', 'test') must exist.
        * No duplicate IDs or filenames across or within partitions.

    Args:
        manifest_path: Path to the JSON split manifest.
        dataset_root: Optional root directory of DIV2K images.
        enforce_file_existence: If True, checks that every listed image exists on disk.
        expected_composite_sha256: Expected composite SHA-256 (defaults to EXPECTED_COMPOSITE_SHA256).
        expected_counts: Optional explicit expected counts per split (defaults to manifest metadata 'splits').
        expected_file_sha256: Optional expected file SHA-256 for custom fixtures.
        is_custom_fixture: Must be True if validating a non-canonical test fixture manifest.

    Returns:
        Dictionary containing verified manifest metadata and partition statistics.

    Raises:
        FileNotFoundError: If manifest or referenced images are missing.
        ValueError: If schema, hash, counts, disjointness, consistency, or path checks fail.
    """
    manifest_p = Path(manifest_path).resolve()
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_p}")

    raw_bytes = manifest_p.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Malformed manifest JSON at {manifest_p}: {e}") from e

    # 1. Structural Schema Check
    required_keys = {"metadata", "train", "val", "test"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Manifest missing required root keys: {missing_keys}")

    meta = data["metadata"]
    if not isinstance(meta, dict):
        raise ValueError(f"Manifest 'metadata' must be a dictionary, got {type(meta).__name__}")

    # Determine canonical status
    repo_root = Path(__file__).resolve().parent.parent.parent
    canonical_manifest_path = (repo_root / DEFAULT_MANIFEST_REL_PATH).resolve()
    is_canonical = (manifest_p == canonical_manifest_path)

    # File SHA-256 integrity enforcement
    if is_canonical:
        # Canonical manifest MUST match the frozen, trusted file SHA-256
        if file_sha256 != EXPECTED_MANIFEST_FILE_SHA256:
            raise ValueError(
                f"SECURITY/INTEGRITY VIOLATION: Canonical manifest file SHA-256 mismatch! "
                f"Expected {EXPECTED_MANIFEST_FILE_SHA256}, got {file_sha256}"
            )
        expected_composite = EXPECTED_COMPOSITE_SHA256
        target_counts = EXPECTED_COUNTS
    else:
        # Non-canonical manifest (must be an explicitly declared test fixture)
        if not is_custom_fixture:
            raise ValueError(
                f"SECURITY VIOLATION: Non-canonical manifest '{manifest_p}' rejected. "
                f"Only the canonical project manifest is permitted for production workflows. "
                f"For test fixtures, set is_custom_fixture=True explicitly."
            )
        if expected_file_sha256 is not None and file_sha256 != expected_file_sha256:
            raise ValueError(
                f"Manifest file SHA-256 mismatch! Expected {expected_file_sha256}, got {file_sha256}"
            )
        expected_composite = expected_composite_sha256
        target_counts = (
            expected_counts
            if expected_counts is not None
            else meta.get("splits", EXPECTED_COUNTS)
        )

    # Composite hash validation: verify both metadata claim AND independently recomputed record digest
    comp_hash = meta.get("dataset_composite_sha256")
    recomputed_composite = compute_manifest_composite_sha256(data)

    if is_canonical:
        if comp_hash != EXPECTED_COMPOSITE_SHA256:
            raise ValueError(
                f"SECURITY/INTEGRITY VIOLATION: Manifest metadata composite SHA-256 mismatch! "
                f"Expected {EXPECTED_COMPOSITE_SHA256}, got {comp_hash}"
            )
        if recomputed_composite != EXPECTED_COMPOSITE_SHA256:
            raise ValueError(
                f"SECURITY/INTEGRITY VIOLATION: Recomputed dataset composite SHA-256 mismatch! "
                f"Expected {EXPECTED_COMPOSITE_SHA256}, got {recomputed_composite}"
            )
    else:
        if expected_composite is not None and comp_hash != expected_composite:
            raise ValueError(
                f"Dataset composite SHA-256 mismatch! Expected {expected_composite}, got {comp_hash}"
            )
        # For custom fixtures where records declare 'sha256' entries:
        has_record_shas = any(isinstance(r, dict) and "sha256" in r for r in data.get("train", []))
        if expected_composite is not None and has_record_shas:
            if recomputed_composite != expected_composite:
                raise ValueError(
                    f"Recomputed dataset composite SHA-256 mismatch! Expected {expected_composite}, got {recomputed_composite}"
                )

    # 2. Check each partition and record consistency
    seen_ids: Set[int] = set()
    seen_filenames: Set[str] = set()
    partition_counts: Dict[str, int] = {}
    partition_ranges: Dict[str, Tuple[int, int]] = {}

    root_p = Path(dataset_root).resolve() if dataset_root is not None else None
    if root_p is not None and enforce_file_existence and not root_p.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root_p}")

    for split_key in ("train", "val", "test"):
        records = data[split_key]
        if not isinstance(records, list):
            raise ValueError(f"Partition '{split_key}' must be a list, got {type(records).__name__}")

        if target_counts is not None and split_key in target_counts:
            exp_count = target_counts[split_key]
            if len(records) != exp_count:
                raise ValueError(
                    f"Partition count mismatch for '{split_key}': expected {exp_count}, got {len(records)}"
                )

        partition_counts[split_key] = len(records)
        split_ids: List[int] = []

        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Record in '{split_key}' is not a dict: {rec}")

            img_id = rec.get("image_id")
            filename = rec.get("filename")

            # Consistency Rule 1: Strict positive integer check (reject booleans masquerading as ints)
            if not isinstance(img_id, int) or isinstance(img_id, bool) or img_id <= 0:
                raise ValueError(
                    f"Invalid 'image_id' in '{split_key}': {img_id!r}. Must be a strictly positive integer (not boolean)."
                )

            # Consistency Rule 2: Filename string and regex check
            if not isinstance(filename, str):
                raise ValueError(f"Invalid or missing 'filename' in '{split_key}': {rec}")
            if not FILENAME_REGEX.match(filename):
                raise ValueError(
                    f"Unexpected filename format '{filename}' in '{split_key}'. Expected pattern '0000.png'."
                )
            if ".." in filename or "/" in filename or "\\" in filename:
                raise ValueError(f"Path traversal detected in filename: '{filename}'")

            # Consistency Rule 3: Duplicate checks within and across partitions
            if img_id in seen_ids:
                raise ValueError(f"Duplicate image_id {img_id} detected across partitions in manifest.")
            if filename in seen_filenames:
                raise ValueError(f"Duplicate filename '{filename}' detected across partitions in manifest.")

            # Consistency Rule 4: Numeric filename stem matches image_id exactly
            expected_fn = f"{img_id:04d}.png"
            if filename != expected_fn:
                raise ValueError(
                    f"Filename stem mismatch in '{split_key}': image_id {img_id} corresponds to '{expected_fn}', "
                    f"but manifest record specifies '{filename}'."
                )

            # Consistency Rule 5: Record's declared split agrees with containing partition
            rec_split = rec.get("split")
            if rec_split is not None:
                if not isinstance(rec_split, str) or rec_split.strip().lower() != split_key:
                    raise ValueError(
                        f"Record split mismatch for image '{filename}': declared split='{rec_split}' "
                        f"conflicts with containing partition '{split_key}'."
                    )

            seen_ids.add(img_id)
            seen_filenames.add(filename)
            split_ids.append(img_id)

            # Consistency Rule 6: Filesystem existence and path containment
            if root_p is not None:
                img_path = (root_p / filename).resolve()
                try:
                    is_contained = img_path.is_relative_to(root_p)
                except AttributeError:
                    # Python < 3.9 fallback
                    is_contained = str(img_path).startswith(str(root_p))

                if not is_contained:
                    raise ValueError(f"Path escapes dataset root: {img_path} not inside {root_p}")

                if enforce_file_existence and not img_path.is_file():
                    raise FileNotFoundError(f"Image file listed in manifest not found on disk: {img_path}")

        partition_ranges[split_key] = (min(split_ids), max(split_ids)) if split_ids else (0, 0)

    return {
        "manifest_path": str(manifest_p),
        "manifest_file_sha256": file_sha256,
        "file_sha256": file_sha256,
        "dataset_composite_sha256": comp_hash,
        "composite_sha256": comp_hash,
        "is_canonical": is_canonical,
        "total_images": len(seen_ids),
        "partition_counts": partition_counts,
        "partition_ranges": partition_ranges,
        "enforce_file_existence": enforce_file_existence,
    }


class DIV2KStegoDataset(Dataset):
    """PyTorch Dataset for DIV2K High-Resolution images bound to canonical partitions.

    Invariants:
    - Loads image membership exclusively from docs/project/div2k_hr_split.json.
    - Partition is strictly one of 'train', 'val', or 'test'.
    - Performs deterministic center crop (default 512x512) or seeded pseudo-random crop.
    - Yields float32 tensors in range [0.0, 255.0] matching NeuralSpatialBridge requirements.
    - Rejects missing files, malformed manifests, duplicates, and path escapes fail-closed.
    """

    def __init__(
        self,
        split: str,
        dataset_root: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        crop_size: int = 512,
        random_crop: bool = False,
        seed: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        enforce_file_existence: bool = True,
        is_custom_fixture: bool = False,
        expected_file_sha256: Optional[str] = None,
    ) -> None:
        """Initializes the DIV2K stego dataset for a specific split.

        Args:
            split: Partition name ('train', 'val', 'validation', 'test').
            dataset_root: Optional path to DIV2K image directory.
            manifest_path: Optional path to JSON split manifest.
            crop_size: Spatial dimension for square cropping (must be positive even int).
            random_crop: If True and split is 'train', uses random crop; otherwise center crop.
            seed: Seed for random cropping if random_crop is True.
            transform: Optional tensor transform applied to (3, H, W) carrier.
            enforce_file_existence: If True, validates file existence on disk at init.
            is_custom_fixture: Must be True if loading a non-canonical test fixture manifest.
            expected_file_sha256: Optional expected file SHA-256 for custom fixtures.
        """
        super().__init__()
        self.split = normalize_split_name(split)
        self.crop_size = int(crop_size)
        if self.crop_size <= 0 or self.crop_size % 2 != 0:
            raise ValueError(f"crop_size must be a positive even integer, got {self.crop_size}")

        self.random_crop = bool(random_crop) and (self.split == "train")
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None
        self.transform = transform
        self.is_custom_fixture = bool(is_custom_fixture)

        # Resolve paths
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else (repo_root / DEFAULT_MANIFEST_REL_PATH).resolve()
        )
        self.dataset_root = (
            Path(dataset_root).resolve()
            if dataset_root is not None
            else (repo_root / DEFAULT_DATASET_REL_PATH).resolve()
        )

        # Audit manifest integrity & membership
        self.manifest_audit = validate_div2k_manifest(
            manifest_path=self.manifest_path,
            dataset_root=self.dataset_root,
            enforce_file_existence=enforce_file_existence,
            is_custom_fixture=is_custom_fixture,
            expected_file_sha256=expected_file_sha256,
        )

        # Load partition records
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        self._records: List[Dict[str, Any]] = list(manifest_data[self.split])

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Returns a copy of partition records."""
        return list(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Loads and crops a single image record.

        Args:
            index: Record index (0 <= index < len(self)).

        Returns:
            Dict containing:
                - 'carrier': torch.Tensor of shape (3, crop_size, crop_size) in [0.0, 255.0]
                - 'image_id': int
                - 'filename': str
                - 'split': str ('train', 'val', or 'test')
                - 'original_dimensions': [width, height]
                - 'crop_box': (left, top, right, bottom)
        """
        if index < 0 or index >= len(self._records):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._records)}")

        rec = self._records[index]
        filename = rec["filename"]
        img_path = self.dataset_root / filename

        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")
            w, h = img.size

            if w < self.crop_size or h < self.crop_size:
                raise ValueError(
                    f"Image {filename} dimensions ({w}, {h}) smaller than requested crop_size {self.crop_size}"
                )

            # Compute crop coordinates
            if self.random_crop:
                rng = self.rng if self.rng is not None else np.random
                left = int(rng.randint(0, w - self.crop_size + 1))
                top = int(rng.randint(0, h - self.crop_size + 1))
            else:
                # Deterministic center crop
                left = (w - self.crop_size) // 2
                top = (h - self.crop_size) // 2

            right = left + self.crop_size
            bottom = top + self.crop_size
            crop_box = (left, top, right, bottom)

            cropped = img.crop(crop_box)
            arr = np.array(cropped, dtype=np.uint8)

        # Convert to torch.Tensor: (3, H, W) in float32 [0.0, 255.0]
        carrier_tensor = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)

        if self.transform is not None:
            carrier_tensor = self.transform(carrier_tensor)

        return {
            "carrier": carrier_tensor,
            "image_id": rec["image_id"],
            "filename": filename,
            "split": self.split,
            "original_dimensions": rec.get("dimensions", [w, h]),
            "crop_box": crop_box,
        }


def build_stego_dataloader(
    split: str,
    purpose: str = "train",
    dataset_root: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    batch_size: int = 1,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    crop_size: int = 512,
    random_crop: bool = False,
    seed: Optional[int] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    is_custom_fixture: bool = False,
    expected_file_sha256: Optional[str] = None,
) -> DataLoader:
    """Builds a PyTorch DataLoader enforcing split-to-purpose boundary invariants.

    Boundary Enforcement Rules:
    - purpose='train':
        * split MUST be 'train'.
        * Attempting to load 'test' or 'val' raises a hard ValueError.
        * Default shuffle: True.
    - purpose='validation' / 'model_selection':
        * split MUST be 'val' or 'validation'.
        * Attempting to load 'test' or 'train' raises a hard ValueError.
        * Default shuffle: False.
    - purpose='audit' / 'held_out_test' / 'benchmark':
        * Explicitly authorized for post-training held-out benchmark evaluation only.
        * Default shuffle: False.

    Args:
        split: Partition name ('train', 'val', 'validation', 'test').
        purpose: Stated workflow purpose ('train', 'validation', 'model_selection', 'audit').
        dataset_root: Optional DIV2K image directory path.
        manifest_path: Optional JSON manifest path.
        batch_size: Mini-batch size (default 1 for 512x512 images).
        shuffle: Optional override for loader shuffling.
        num_workers: DataLoader worker count (default 0 for single-thread safety).
        crop_size: Spatial crop dimension (default 512).
        random_crop: Random cropping flag (train split only).
        seed: Random seed for cropping / shuffling.
        pin_memory: DataLoader pin_memory flag.
        drop_last: DataLoader drop_last flag.
        is_custom_fixture: Must be True if loading a non-canonical test fixture manifest.
        expected_file_sha256: Optional expected file SHA-256 for custom fixtures.

    Returns:
        Configured PyTorch DataLoader instance.

    Raises:
        ValueError: If split and purpose violate security boundary rules.
    """
    norm_split = normalize_split_name(split)
    norm_purpose = purpose.strip().lower()

    # Boundary Rule 1: Training purpose requires strictly 'train' split
    if norm_purpose == "train":
        if norm_split != "train":
            raise ValueError(
                f"SECURITY/INTEGRITY VIOLATION: Cannot construct training DataLoader with split='{split}'. "
                f"Only split='train' is permitted for training purpose."
            )
        default_shuffle = True if shuffle is None else bool(shuffle)

    # Boundary Rule 2: Validation / Model Selection requires strictly 'val' split
    elif norm_purpose in ("validation", "model_selection"):
        if norm_split != "val":
            raise ValueError(
                f"SECURITY/INTEGRITY VIOLATION: Cannot construct validation DataLoader with split='{split}'. "
                f"Only split='val' is permitted for validation/model-selection purpose."
            )
        default_shuffle = False if shuffle is None else bool(shuffle)

    # Boundary Rule 3: Held-out test / Audit purpose requires explicit audit declaration
    elif norm_purpose in ("audit", "held_out_test", "benchmark"):
        default_shuffle = False if shuffle is None else bool(shuffle)

    else:
        raise ValueError(
            f"Unknown loader purpose '{purpose}'. Supported purposes: ('train', 'validation', 'model_selection', 'audit')."
        )

    # Construct dataset
    dataset = DIV2KStegoDataset(
        split=norm_split,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        crop_size=crop_size,
        random_crop=random_crop,
        seed=seed,
        is_custom_fixture=is_custom_fixture,
        expected_file_sha256=expected_file_sha256,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=default_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Attach governance metadata to loader object
    loader.dataset_split = norm_split  # type: ignore[attr-defined]
    loader.loader_purpose = norm_purpose  # type: ignore[attr-defined]
    loader.is_training_loader = (norm_purpose == "train")  # type: ignore[attr-defined]

    return loader
