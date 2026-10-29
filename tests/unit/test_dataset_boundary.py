"""Unit tests for Phase 3 Sprint 3.2 Dataset Loader and Training-Boundary Enforcement.

Verifies:
1. DIV2KStegoDataset correctness:
   - Explicit partition loading (train=600, val=100, test=100).
   - Deterministic membership, ordering, and lazy loading.
   - Proper tensor output shapes (3, 512, 512) and range [0.0, 255.0].
   - Fail-closed validation: rejects missing files, duplicates, overlapping splits,
     malformed manifests, path traversal attempts, unexpected filenames, and missing dataset root.
2. Training-boundary enforcement:
   - build_stego_dataloader restricts purpose='train' strictly to split='train'.
   - build_stego_dataloader restricts purpose='validation' strictly to split='val'.
   - Attempting to construct training dataloaders with 'test' or 'val' raises hard ValueError.
   - execute_partition_verified_step enforces split='train' and rejects 'test' and 'val' batches.
   - evaluate_partition_verified_step enforces split='val' and rejects 'test' and 'train' batches.
   - Boundary checks fail before any model computation or backward pass.
   - Model weights remain completely unchanged (NO optimizer.step()).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from PIL import Image

from src.neural.dataset import (
    DEFAULT_DATASET_REL_PATH,
    DEFAULT_MANIFEST_REL_PATH,
    DIV2KStegoDataset,
    EXPECTED_COMPOSITE_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    build_stego_dataloader,
    normalize_split_name,
    validate_div2k_manifest,
)
from src.neural.losses import CompositeLoss
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_scaffold import (
    DatasetSplitManager,
    TrainingConfig,
    build_model_and_optimizer,
    evaluate_partition_verified_step,
    execute_partition_verified_step,
    execute_training_step,
    set_reproducibility_seeds,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_MANIFEST = REPO_ROOT / DEFAULT_MANIFEST_REL_PATH
CANONICAL_DATASET_ROOT = REPO_ROOT / DEFAULT_DATASET_REL_PATH


def _require_canonical_dataset():
    if not CANONICAL_DATASET_ROOT.is_dir():
        pytest.skip(
            f"DIV2K dataset root missing at '{CANONICAL_DATASET_ROOT}'. "
            f"Download DIV2K_train_HR per README to run dataset integration tests."
        )


# =============================================================================
# 1. Dataset Correctness Tests
# =============================================================================

def test_dataset_valid_partitions_load_expected_members():
    """Verify that train (600), val (100), and test (100) partitions load correctly."""
    _require_canonical_dataset()
    train_ds = DIV2KStegoDataset(split="train")
    val_ds = DIV2KStegoDataset(split="val")
    val_alias_ds = DIV2KStegoDataset(split="validation")
    test_ds = DIV2KStegoDataset(split="test")

    assert len(train_ds) == 600
    assert len(val_ds) == 100
    assert len(val_alias_ds) == 100
    assert len(test_ds) == 100

    # ID ranges
    train_ids = [r["image_id"] for r in train_ds.records]
    val_ids = [r["image_id"] for r in val_ds.records]
    test_ids = [r["image_id"] for r in test_ds.records]

    assert min(train_ids) == 1 and max(train_ids) == 600
    assert min(val_ids) == 601 and max(val_ids) == 700
    assert min(test_ids) == 701 and max(test_ids) == 800

    # Filename format
    assert train_ds.records[0]["filename"] == "0001.png"
    assert train_ds.records[-1]["filename"] == "0600.png"
    assert val_ds.records[0]["filename"] == "0601.png"
    assert val_ds.records[-1]["filename"] == "0700.png"
    assert test_ds.records[0]["filename"] == "0701.png"
    assert test_ds.records[-1]["filename"] == "0800.png"


def test_dataset_repeated_loading_determinism():
    """Verify that repeated loading yields bit-for-bit identical membership and order."""
    _require_canonical_dataset()
    ds1 = DIV2KStegoDataset(split="train")
    ds2 = DIV2KStegoDataset(split="train")

    assert len(ds1) == len(ds2)
    assert [r["image_id"] for r in ds1.records] == [r["image_id"] for r in ds2.records]
    assert [r["filename"] for r in ds1.records] == [r["filename"] for r in ds2.records]

    item1 = ds1[0]
    item2 = ds2[0]
    assert item1["image_id"] == item2["image_id"]
    assert item1["crop_box"] == item2["crop_box"]
    assert torch.equal(item1["carrier"], item2["carrier"])


def test_dataset_getitem_output_shapes_and_types():
    """Verify that __getitem__ yields tensors conforming to NeuralSpatialBridge requirements."""
    _require_canonical_dataset()
    ds = DIV2KStegoDataset(split="train", crop_size=512)
    item = ds[0]

    carrier = item["carrier"]
    assert isinstance(carrier, torch.Tensor)
    assert carrier.shape == (3, 512, 512)
    assert carrier.dtype == torch.float32
    assert carrier.min().item() >= 0.0
    assert carrier.max().item() <= 255.0

    assert item["split"] == "train"
    assert item["image_id"] == 1
    assert item["filename"] == "0001.png"
    assert len(item["crop_box"]) == 4


def test_dataset_unknown_partition_rejected():
    """Verify that unknown partition names are rejected."""
    with pytest.raises(ValueError, match="Invalid split name"):
        DIV2KStegoDataset(split="eval")

    with pytest.raises(ValueError, match="Invalid split name"):
        DIV2KStegoDataset(split="train_and_val")


def test_dataset_missing_configured_root_rejected():
    """Verify that a non-existent dataset root fails closed."""
    fake_root = REPO_ROOT / "non_existent_dataset_directory_12345"
    with pytest.raises(FileNotFoundError, match="Dataset root directory does not exist"):
        DIV2KStegoDataset(split="train", dataset_root=fake_root)


# =============================================================================
# 2. Manifest Schema & Integrity Fixture Tests (Fail-Closed Validations)
# =============================================================================

@pytest.fixture
def dummy_dataset_fixture():
    """Creates a temporary dataset directory with minimal dummy images for negative tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        img_dir = tmp_p / "images"
        img_dir.mkdir()

        # Create dummy 512x512 PNG images
        for i in range(1, 11):
            fn = f"{i:04d}.png"
            img = Image.new("RGB", (512, 512), color=(i * 10, i * 20, i * 25))
            img.save(img_dir / fn)

        manifest_base: Dict[str, Any] = {
            "metadata": {
                "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
                "splits": {"train": 6, "val": 2, "test": 2},
            },
            "train": [
                {"image_id": i, "filename": f"{i:04d}.png", "dimensions": [512, 512], "split": "train"}
                for i in range(1, 7)
            ],
            "val": [
                {"image_id": i, "filename": f"{i:04d}.png", "dimensions": [512, 512], "split": "val"}
                for i in range(7, 9)
            ],
            "test": [
                {"image_id": i, "filename": f"{i:04d}.png", "dimensions": [512, 512], "split": "test"}
                for i in range(9, 11)
            ],
        }

        yield tmp_p, img_dir, manifest_base


def test_dataset_duplicate_id_rejected(dummy_dataset_fixture):
    """Verify that duplicate image IDs across or within partitions are rejected."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    # Duplicate ID 1 in val partition
    manifest_data["val"][0]["image_id"] = 1

    bad_manifest = tmp_p / "dup_id_manifest.json"
    bad_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate image_id 1 detected"):
        validate_div2k_manifest(bad_manifest, img_dir, is_custom_fixture=True)


def test_dataset_duplicate_filename_rejected(dummy_dataset_fixture):
    """Verify that duplicate filenames are rejected."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    # Duplicate filename '0001.png' in val partition
    manifest_data["val"][0]["filename"] = "0001.png"

    bad_manifest = tmp_p / "dup_file_manifest.json"
    bad_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate filename '0001.png' detected"):
        validate_div2k_manifest(bad_manifest, img_dir, is_custom_fixture=True)


def test_dataset_path_traversal_rejected(dummy_dataset_fixture):
    """Verify that path traversal attempts (e.g. '../secret.png') are caught and rejected."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    manifest_data["train"][0]["filename"] = "../secret.png"

    bad_manifest = tmp_p / "traversal_manifest.json"
    bad_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Path traversal detected|Unexpected filename format"):
        validate_div2k_manifest(bad_manifest, img_dir, is_custom_fixture=True)


def test_dataset_unexpected_filename_format_rejected(dummy_dataset_fixture):
    """Verify that non-conforming filenames (e.g. 'img_01.jpg') are rejected."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    manifest_data["train"][0]["filename"] = "img_0001.png"

    bad_manifest = tmp_p / "bad_fn_manifest.json"
    bad_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected filename format"):
        validate_div2k_manifest(bad_manifest, img_dir, is_custom_fixture=True)


def test_dataset_malformed_json_rejected(dummy_dataset_fixture):
    """Verify that malformed JSON files fail closed."""
    tmp_p, img_dir, _ = dummy_dataset_fixture
    bad_manifest = tmp_p / "corrupt.json"
    bad_manifest.write_text("{ incomplete json content ...", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed manifest JSON"):
        validate_div2k_manifest(bad_manifest, img_dir, is_custom_fixture=True)


def test_dataset_missing_image_file_rejected(dummy_dataset_fixture):
    """Verify that a manifest referencing an image absent from disk fails closed."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    # Delete 0001.png
    (img_dir / "0001.png").unlink()

    test_manifest = tmp_p / "missing_file_manifest.json"
    test_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Image file listed in manifest not found on disk"):
        validate_div2k_manifest(test_manifest, img_dir, enforce_file_existence=True, is_custom_fixture=True)


# =============================================================================
# 2.1 Manifest Integrity Hardening Tests (Sprint 3.2.1)
# =============================================================================

def test_canonical_manifest_integrity_success():
    """Verify that the unmodified canonical manifest passes file and composite SHA-256 checks."""
    _require_canonical_dataset()
    audit = validate_div2k_manifest(CANONICAL_MANIFEST, CANONICAL_DATASET_ROOT, enforce_file_existence=True)
    assert audit["file_sha256"] == EXPECTED_MANIFEST_FILE_SHA256
    assert audit["composite_sha256"] == EXPECTED_COMPOSITE_SHA256
    assert audit["total_images"] == 800
    assert audit["partition_counts"] == {"train": 600, "val": 100, "test": 100}
    assert audit["is_canonical"] is True


def test_non_canonical_manifest_rejected_without_fixture_flag(dummy_dataset_fixture):
    """Verify that any non-canonical manifest fails closed unless is_custom_fixture=True."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    test_manifest = tmp_p / "unflagged_fixture.json"
    test_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="SECURITY VIOLATION: Non-canonical manifest .* rejected"):
        validate_div2k_manifest(test_manifest, img_dir, is_custom_fixture=False)


def test_modified_canonical_manifest_rejected_even_if_composite_hash_retained(tmp_path):
    """Verify that a modified copy of the manifest with altered splits is rejected fail-closed.

    Even if it retains the canonical dataset_composite_sha256, it fails because:
    1. It is non-canonical and fails if is_custom_fixture=False.
    2. With is_custom_fixture=True and target counts (600/100/100), partition counts mismatch.
    """
    with open(CANONICAL_MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tamper: move last train image (0600.png) into val
    moved_rec = data["train"].pop()
    data["val"].append(moved_rec)
    # Note: data["metadata"]["dataset_composite_sha256"] remains the original canonical hash

    tampered_manifest = tmp_path / "tampered_manifest.json"
    tampered_manifest.write_text(json.dumps(data), encoding="utf-8")

    # Fail closed without fixture flag
    with pytest.raises(ValueError, match="SECURITY VIOLATION: Non-canonical manifest .* rejected"):
        validate_div2k_manifest(tampered_manifest, CANONICAL_DATASET_ROOT, is_custom_fixture=False)

    # Even with fixture flag, partition count check or record split mismatch fails
    with pytest.raises(ValueError, match="Partition count mismatch for 'train': expected 600, got 599"):
        validate_div2k_manifest(
            tampered_manifest,
            CANONICAL_DATASET_ROOT,
            is_custom_fixture=True,
            expected_counts={"train": 600, "val": 100, "test": 100},
            enforce_file_existence=False,
        )


def test_manifest_custom_fixture_file_sha256_mismatch_rejected(dummy_dataset_fixture):
    """Verify that when expected_file_sha256 is supplied for a custom fixture, mismatches are rejected."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    test_manifest = tmp_p / "custom_fixture.json"
    test_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    wrong_sha = "0" * 64
    with pytest.raises(ValueError, match="Manifest file SHA-256 mismatch!"):
        validate_div2k_manifest(
            test_manifest,
            img_dir,
            is_custom_fixture=True,
            expected_file_sha256=wrong_sha,
            enforce_file_existence=False,
        )


def test_manifest_boolean_image_id_rejected(dummy_dataset_fixture):
    """Verify that a boolean image_id (e.g. True/False) is rejected (preventing isinstance(True, int) trap)."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    # Replace first train image_id with True
    manifest_data["train"][0]["image_id"] = True

    bad_manifest = tmp_p / "bool_id_manifest.json"
    bad_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid 'image_id' in 'train': True\. Must be a strictly positive integer \(not boolean\)\."):
        validate_div2k_manifest(bad_manifest, img_dir, is_custom_fixture=True, enforce_file_existence=False)


def test_manifest_zero_or_negative_image_id_rejected(dummy_dataset_fixture):
    """Verify that zero or negative image_id is rejected."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    manifest_data["train"][0]["image_id"] = 0

    bad_manifest = tmp_p / "zero_id_manifest.json"
    bad_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Must be a strictly positive integer"):
        validate_div2k_manifest(bad_manifest, img_dir, is_custom_fixture=True, enforce_file_existence=False)


def test_manifest_image_id_filename_stem_mismatch_rejected(dummy_dataset_fixture):
    """Verify that mismatch between image_id and numeric filename stem fails closed."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    # image_id is 1, but filename set to 0002.png
    manifest_data["train"][0]["image_id"] = 1
    manifest_data["train"][0]["filename"] = "0002.png"

    bad_manifest = tmp_p / "stem_mismatch_manifest.json"
    bad_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Filename stem mismatch in 'train': image_id 1 corresponds to '0001.png', but manifest record specifies '0002.png'"):
        validate_div2k_manifest(bad_manifest, img_dir, is_custom_fixture=True, enforce_file_existence=False)


def test_manifest_record_split_mismatch_rejected(dummy_dataset_fixture):
    """Verify that a record whose declared 'split' conflicts with containing partition fails closed."""
    tmp_p, img_dir, manifest_data = dummy_dataset_fixture
    # In 'train' partition, record specifies split='val'
    manifest_data["train"][0]["split"] = "val"

    bad_manifest = tmp_p / "split_conflict_manifest.json"
    bad_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Record split mismatch for image '0001.png': declared split='val' conflicts with containing partition 'train'"):
        validate_div2k_manifest(bad_manifest, img_dir, is_custom_fixture=True, enforce_file_existence=False)


def test_dataset_split_manager_canonical_integrity():
    """Verify that DatasetSplitManager validates the canonical manifest SHA-256 and rejects non-canonical manifests."""
    mgr = DatasetSplitManager()
    assert mgr.file_sha256 == DatasetSplitManager.EXPECTED_MANIFEST_FILE_SHA256
    assert mgr.file_sha256 == EXPECTED_MANIFEST_FILE_SHA256
    assert len(mgr.get_train_records()) == 600
    assert len(mgr.get_val_records()) == 100
    assert len(mgr.get_held_out_test_manifest_for_audit_only()) == 100

    # Non-canonical path without fixture flag raises security violation
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tf.write(b"{}")
        tf_path = Path(tf.name)

    try:
        with pytest.raises(ValueError, match="SECURITY VIOLATION: Non-canonical manifest .* rejected"):
            DatasetSplitManager(manifest_path=tf_path, is_custom_fixture=False)
    finally:
        if tf_path.exists():
            tf_path.unlink()


def test_manifest_recomputed_composite_sha256_detects_record_tamper():
    """Verify that tampering with an individual record's SHA-256 is detected by recomputing the composite hash,
    even when partition counts (600/100/100) and metadata composite hash are valid."""
    canonical_data = json.loads(CANONICAL_MANIFEST.read_bytes().decode("utf-8"))

    # Tamper with the first train record's sha256
    tampered_data = json.loads(json.dumps(canonical_data))
    tampered_data["train"][0]["sha256"] = "0" * 64

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tf.write(json.dumps(tampered_data).encode("utf-8"))
        tf_path = Path(tf.name)

    try:
        # 1. validate_div2k_manifest detects recomputed composite mismatch
        with pytest.raises(ValueError, match="Recomputed dataset composite SHA-256 mismatch"):
            validate_div2k_manifest(
                tf_path,
                is_custom_fixture=True,
                expected_composite_sha256=EXPECTED_COMPOSITE_SHA256,
                enforce_file_existence=False,
            )

        # 2. DatasetSplitManager detects recomputed composite mismatch
        with pytest.raises(ValueError, match="Recomputed dataset composite SHA-256 mismatch"):
            DatasetSplitManager(manifest_path=tf_path, is_custom_fixture=True)
    finally:
        if tf_path.exists():
            tf_path.unlink()


# =============================================================================
# 3. Training Boundary Enforcement Tests
# =============================================================================

def test_dataloader_purpose_train_requires_split_train():
    """Verify that build_stego_dataloader rejects split != 'train' when purpose='train'."""
    # Attempting to load 'test' for training MUST raise ValueError
    with pytest.raises(ValueError, match="SECURITY/INTEGRITY VIOLATION.*Only split='train' is permitted"):
        build_stego_dataloader(split="test", purpose="train")

    # Attempting to load 'val' for training MUST raise ValueError
    with pytest.raises(ValueError, match="SECURITY/INTEGRITY VIOLATION.*Only split='train' is permitted"):
        build_stego_dataloader(split="val", purpose="train")


def test_dataloader_purpose_validation_requires_split_val():
    """Verify that build_stego_dataloader rejects split != 'val' when purpose='validation'."""
    # Attempting to load 'test' for validation MUST raise ValueError
    with pytest.raises(ValueError, match="SECURITY/INTEGRITY VIOLATION.*Only split='val' is permitted"):
        build_stego_dataloader(split="test", purpose="validation")

    # Attempting to load 'train' for validation MUST raise ValueError
    with pytest.raises(ValueError, match="SECURITY/INTEGRITY VIOLATION.*Only split='val' is permitted"):
        build_stego_dataloader(split="train", purpose="validation")


def test_dataloader_train_iteration_membership():
    """Verify that batches yielded by train loader contain strictly train partition data."""
    _require_canonical_dataset()
    train_loader = build_stego_dataloader(split="train", purpose="train", batch_size=2, shuffle=False)
    assert train_loader.dataset_split == "train"
    assert train_loader.is_training_loader is True

    # Check first batch
    batch = next(iter(train_loader))
    assert batch["carrier"].shape == (2, 3, 512, 512)
    assert batch["split"] == ["train", "train"]
    assert batch["image_id"].tolist() == [1, 2]
    assert batch["filename"] == ["0001.png", "0002.png"]


def test_dataloader_validation_iteration_membership():
    """Verify that batches yielded by validation loader contain strictly val partition data."""
    _require_canonical_dataset()
    val_loader = build_stego_dataloader(split="val", purpose="validation", batch_size=2, shuffle=False)
    assert val_loader.dataset_split == "val"
    assert val_loader.is_training_loader is False

    # Check first batch
    batch = next(iter(val_loader))
    assert batch["carrier"].shape == (2, 3, 512, 512)
    assert batch["split"] == ["val", "val"]
    assert batch["image_id"].tolist() == [601, 602]
    assert batch["filename"] == ["0601.png", "0602.png"]


def test_execute_partition_verified_step_accepts_train_batch():
    """Verify that execute_partition_verified_step executes on a verified train batch."""
    _require_canonical_dataset()
    set_reproducibility_seeds(42, deterministic=True)
    bridge, optimizer, _ = build_model_and_optimizer(TrainingConfig(learning_rate=1e-4, device="cpu"))
    comp_loss = CompositeLoss()

    initial_params = [p.clone().detach() for p in bridge.acs_model.parameters()]

    train_loader = build_stego_dataloader(split="train", purpose="train", batch_size=1, shuffle=False)
    batch = next(iter(train_loader))

    result = execute_partition_verified_step(
        bridge=bridge,
        optimizer=optimizer,
        composite_loss=comp_loss,
        batch=batch,
        run_backward=True,
    )

    assert result.is_grad_finite is True
    assert result.total_grad_norm > 0.0
    assert result.optimizer_step_executed is False

    # Invariant: weights remain identical
    current_params = [p.clone().detach() for p in bridge.acs_model.parameters()]
    for init_p, curr_p in zip(initial_params, current_params):
        assert torch.equal(init_p, curr_p), "Model weights must remain bit-for-bit unchanged!"


def test_execute_partition_verified_step_rejects_test_batch():
    """Verify that execute_partition_verified_step hard-rejects any batch from the test partition."""
    bridge, optimizer, _ = build_model_and_optimizer(TrainingConfig(learning_rate=1e-4, device="cpu"))
    comp_loss = CompositeLoss()

    # Synthetic batch tagged with split='test'
    test_batch = {
        "carrier": torch.randn(1, 3, 512, 512),
        "split": ["test"],
        "image_id": torch.tensor([750]),
        "filename": ["0750.png"],
    }

    with pytest.raises(ValueError, match="SECURITY/INTEGRITY VIOLATION: Held-out 'test' partition detected"):
        execute_partition_verified_step(
            bridge=bridge,
            optimizer=optimizer,
            composite_loss=comp_loss,
            batch=test_batch,
            run_backward=True,
        )


def test_execute_partition_verified_step_rejects_val_batch():
    """Verify that execute_partition_verified_step hard-rejects batches from the val partition."""
    bridge, optimizer, _ = build_model_and_optimizer(TrainingConfig(learning_rate=1e-4, device="cpu"))
    comp_loss = CompositeLoss()

    val_batch = {
        "carrier": torch.randn(1, 3, 512, 512),
        "split": ["val"],
        "image_id": torch.tensor([650]),
        "filename": ["0650.png"],
    }

    with pytest.raises(ValueError, match="BOUNDARY VIOLATION: execute_partition_verified_step requires split='train'"):
        execute_partition_verified_step(
            bridge=bridge,
            optimizer=optimizer,
            composite_loss=comp_loss,
            batch=val_batch,
            run_backward=True,
        )


def test_execute_partition_verified_step_rejects_test_image_id_even_if_split_tampered():
    """Verify that even if split string is falsified to 'train', an image ID in 701-800 is rejected."""
    bridge, optimizer, _ = build_model_and_optimizer(TrainingConfig(learning_rate=1e-4, device="cpu"))
    comp_loss = CompositeLoss()

    tampered_batch = {
        "carrier": torch.randn(1, 3, 512, 512),
        "split": ["train"],  # Falsified split tag
        "image_id": torch.tensor([725]),  # True test set ID!
        "filename": ["0725.png"],
    }

    with pytest.raises(ValueError, match="Image ID 725 belongs to held-out test split"):
        execute_partition_verified_step(
            bridge=bridge,
            optimizer=optimizer,
            composite_loss=comp_loss,
            batch=tampered_batch,
            run_backward=True,
        )


def test_evaluate_partition_verified_step_boundary_enforcement():
    """Verify that evaluate_partition_verified_step enforces split='val' and rejects 'test'."""
    bridge = NeuralSpatialBridge()
    comp_loss = CompositeLoss()

    val_batch = {
        "carrier": torch.randn(1, 3, 512, 512),
        "split": ["val"],
        "image_id": torch.tensor([601]),
        "filename": ["0601.png"],
    }
    # Succeeds on val batch
    res = evaluate_partition_verified_step(bridge, comp_loss, val_batch)
    assert res.optimizer_step_executed is False

    # Rejects test batch
    test_batch = {
        "carrier": torch.randn(1, 3, 512, 512),
        "split": ["test"],
        "image_id": torch.tensor([701]),
        "filename": ["0701.png"],
    }
    with pytest.raises(ValueError, match="Held-out 'test' partition detected in validation/model-selection"):
        evaluate_partition_verified_step(bridge, comp_loss, test_batch)


def test_execute_training_step_split_tag_rejection():
    """Verify that low-level execute_training_step fails closed if passed split_tag='test'."""
    bridge, optimizer, _ = build_model_and_optimizer(TrainingConfig(learning_rate=1e-4, device="cpu"))
    comp_loss = CompositeLoss()

    carrier = torch.randn(1, 3, 64, 64)
    payload = torch.ones(1, 1, 32, 32)
    mask = torch.ones(1, 1, 32, 32)

    with pytest.raises(ValueError, match="execute_training_step explicitly rejects split_tag='test'"):
        execute_training_step(
            bridge=bridge,
            optimizer=optimizer,
            composite_loss=comp_loss,
            carrier=carrier,
            payload_bits=payload,
            mask=mask,
            split_tag="test",
        )
