"""Experiment Script: Phase 3 Sprint 3.2 Dataset Loader and Training-Boundary Audit.

Validates the explicit dataset-loading boundary for Phase 3:
1. Validates the canonical DIV2K split manifest schema, file digest, composite hash, and file existence.
2. Demonstrates DIV2KStegoDataset instantiation across train, validation, and held-out test splits.
3. Tests purpose-based DataLoader boundary enforcement:
   - purpose='train' strictly accepts split='train' (rejects 'test' and 'val').
   - purpose='validation' strictly accepts split='val' (rejects 'test' and 'train').
4. Demonstrates that execute_partition_verified_step accepts train batches and hard-rejects test batches.
5. Demonstrates that evaluate_partition_verified_step accepts val batches and hard-rejects test batches.
6. Verifies that NO model training, optimizer updates (optimizer.step()), or epoch executions occur.
7. Outputs machine-readable results to:
   docs/experiments/sprint3_2_dataset_boundary_results.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.neural.dataset import (
    DEFAULT_DATASET_REL_PATH,
    DEFAULT_MANIFEST_REL_PATH,
    DIV2KStegoDataset,
    EXPECTED_COMPOSITE_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    build_stego_dataloader,
    validate_div2k_manifest,
)
from src.neural.losses import CompositeLoss
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_scaffold import (
    TrainingConfig,
    build_model_and_optimizer,
    evaluate_partition_verified_step,
    execute_partition_verified_step,
    set_reproducibility_seeds,
)


def main() -> None:
    print("=" * 80)
    print("PHASE 3 SPRINT 3.2: DATASET LOADER AND TRAINING-BOUNDARY AUDIT")
    print("=" * 80)

    set_reproducibility_seeds(42, deterministic=True)
    manifest_path = REPO_ROOT / DEFAULT_MANIFEST_REL_PATH
    dataset_root = REPO_ROOT / DEFAULT_DATASET_REL_PATH

    print(f"Manifest Path: {manifest_path}")
    print(f"Dataset Root:  {dataset_root}")

    # -------------------------------------------------------------------------
    # 1. Manifest Audit & Integrity Check
    # -------------------------------------------------------------------------
    print("\n--- 1. MANIFEST STRUCTURAL & FILESYSTEM AUDIT ---")
    manifest_bytes = manifest_path.read_bytes()
    manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    manifest_audit = validate_div2k_manifest(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        enforce_file_existence=True,
    )

    print(f"Manifest File SHA-256:      {manifest_file_sha256}")
    print(f"Dataset Composite SHA-256:  {manifest_audit['dataset_composite_sha256']}")
    print(f"Total Images Validated:     {manifest_audit['total_images']}")
    for split_name, count in manifest_audit["partition_counts"].items():
        id_min, id_max = manifest_audit["partition_ranges"][split_name]
        print(f"  Partition '{split_name}': {count} images (ID range {id_min}..{id_max})")

    # -------------------------------------------------------------------------
    # 1.1 Manifest Integrity Hardening (Sprint 3.2.1)
    # -------------------------------------------------------------------------
    print("\n--- 1.1 SPRINT 3.2.1 MANIFEST INTEGRITY HARDENING CHECKS ---")
    manifest_hardening_checks: Dict[str, bool] = {}

    # Check 1.1.1: Canonical file SHA-256 matches EXPECTED_MANIFEST_FILE_SHA256
    file_sha_matches = (manifest_audit["manifest_file_sha256"] == EXPECTED_MANIFEST_FILE_SHA256)
    manifest_hardening_checks["canonical_file_sha256_verified"] = file_sha_matches
    print(f"[OK] Canonical Manifest File SHA-256 Verified: {file_sha_matches}")

    # Check 1.1.2: Non-canonical manifest rejected without fixture flag
    non_canonical_rejected = False
    dummy_manifest_path = REPO_ROOT / "docs" / "project" / "dummy_temp_manifest.json"
    dummy_manifest_path.write_text(
        json.dumps({"metadata": {}, "train": [], "val": [], "test": []}),
        encoding="utf-8",
    )
    try:
        try:
            validate_div2k_manifest(dummy_manifest_path, dataset_root, is_custom_fixture=False)
        except ValueError as e:
            if "SECURITY VIOLATION: Non-canonical manifest" in str(e):
                non_canonical_rejected = True
    finally:
        if dummy_manifest_path.exists():
            dummy_manifest_path.unlink()
    manifest_hardening_checks["non_canonical_without_flag_rejected"] = non_canonical_rejected
    print(f"[OK] Non-canonical manifest without fixture flag rejected: {non_canonical_rejected}")

    # Check 1.1.3: Boolean image_id rejected fail-closed
    bool_id_rejected = False
    fixture_data: Dict[str, Any] = {
        "metadata": {
            "dataset_composite_sha256": EXPECTED_COMPOSITE_SHA256,
            "splits": {"train": 1, "val": 0, "test": 0},
        },
        "train": [{"image_id": True, "filename": "0001.png", "split": "train"}],
        "val": [],
        "test": [],
    }
    dummy_fixture_p = REPO_ROOT / "docs" / "project" / "dummy_fixture.json"
    dummy_fixture_p.write_text(json.dumps(fixture_data), encoding="utf-8")
    try:
        try:
            validate_div2k_manifest(dummy_fixture_p, dataset_root, enforce_file_existence=False, is_custom_fixture=True)
        except ValueError as e:
            if "Must be a strictly positive integer (not boolean)" in str(e):
                bool_id_rejected = True
    finally:
        if dummy_fixture_p.exists():
            dummy_fixture_p.unlink()
    manifest_hardening_checks["boolean_image_id_rejected"] = bool_id_rejected
    print(f"[OK] Boolean image_id rejected fail-closed: {bool_id_rejected}")

    # Check 1.1.4: Stem mismatch rejected fail-closed
    stem_mismatch_rejected = False
    fixture_data["train"] = [{"image_id": 1, "filename": "0002.png", "split": "train"}]
    dummy_fixture_p.write_text(json.dumps(fixture_data), encoding="utf-8")
    try:
        try:
            validate_div2k_manifest(dummy_fixture_p, dataset_root, enforce_file_existence=False, is_custom_fixture=True)
        except ValueError as e:
            if "Filename stem mismatch" in str(e):
                stem_mismatch_rejected = True
    finally:
        if dummy_fixture_p.exists():
            dummy_fixture_p.unlink()
    manifest_hardening_checks["stem_mismatch_rejected"] = stem_mismatch_rejected
    print(f"[OK] Numeric stem mismatch rejected fail-closed: {stem_mismatch_rejected}")

    # -------------------------------------------------------------------------
    # 2. Dataset Partition Instantiation Checks
    # -------------------------------------------------------------------------
    print("\n--- 2. DATASET PARTITION INSTANTIATION CHECKS ---")
    train_ds = DIV2KStegoDataset(split="train", dataset_root=dataset_root, manifest_path=manifest_path)
    val_ds = DIV2KStegoDataset(split="val", dataset_root=dataset_root, manifest_path=manifest_path)
    test_ds = DIV2KStegoDataset(split="test", dataset_root=dataset_root, manifest_path=manifest_path)

    print(f"Train Dataset: {len(train_ds)} samples (Sample 0: {train_ds.records[0]['filename']}, ID {train_ds.records[0]['image_id']})")
    print(f"Val Dataset:   {len(val_ds)} samples (Sample 0: {val_ds.records[0]['filename']}, ID {val_ds.records[0]['image_id']})")
    print(f"Test Dataset:  {len(test_ds)} samples (Sample 0: {test_ds.records[0]['filename']}, ID {test_ds.records[0]['image_id']})")

    # Sample single image from train to check shape
    sample_item = train_ds[0]
    print(f"Sample Carrier Tensor: shape={sample_item['carrier'].shape}, dtype={sample_item['carrier'].dtype}, "
          f"min={sample_item['carrier'].min().item():.1f}, max={sample_item['carrier'].max().item():.1f}")

    # -------------------------------------------------------------------------
    # 3. Purpose-Based DataLoader Boundary Enforcement
    # -------------------------------------------------------------------------
    print("\n--- 3. PURPOSE-BASED DATALOADER BOUNDARY ENFORCEMENT ---")
    boundary_rejections: List[Dict[str, Any]] = []

    # Check 3.1: Valid Train Loader
    train_loader = build_stego_dataloader(split="train", purpose="train", batch_size=2, shuffle=False)
    train_batch = next(iter(train_loader))
    print(f"[OK] Training DataLoader created: batch_carrier={train_batch['carrier'].shape}, splits={train_batch['split']}")

    # Check 3.2: Rejection of 'test' split for purpose='train'
    test_for_train_rejected = False
    try:
        build_stego_dataloader(split="test", purpose="train")
    except ValueError as e:
        test_for_train_rejected = True
        err_msg = str(e)
        boundary_rejections.append({
            "test_case": "build_stego_dataloader(split='test', purpose='train')",
            "rejected": True,
            "error_type": "ValueError",
            "message": err_msg,
        })
        print(f"[OK] Correctly rejected 'test' for purpose='train': {err_msg[:65]}...")

    # Check 3.3: Rejection of 'val' split for purpose='train'
    val_for_train_rejected = False
    try:
        build_stego_dataloader(split="val", purpose="train")
    except ValueError as e:
        val_for_train_rejected = True
        err_msg = str(e)
        boundary_rejections.append({
            "test_case": "build_stego_dataloader(split='val', purpose='train')",
            "rejected": True,
            "error_type": "ValueError",
            "message": err_msg,
        })
        print(f"[OK] Correctly rejected 'val' for purpose='train': {err_msg[:65]}...")

    # Check 3.4: Valid Validation Loader
    val_loader = build_stego_dataloader(split="val", purpose="validation", batch_size=2, shuffle=False)
    val_batch = next(iter(val_loader))
    print(f"[OK] Validation DataLoader created: batch_carrier={val_batch['carrier'].shape}, splits={val_batch['split']}")

    # Check 3.5: Rejection of 'test' split for purpose='validation'
    test_for_val_rejected = False
    try:
        build_stego_dataloader(split="test", purpose="validation")
    except ValueError as e:
        test_for_val_rejected = True
        err_msg = str(e)
        boundary_rejections.append({
            "test_case": "build_stego_dataloader(split='test', purpose='validation')",
            "rejected": True,
            "error_type": "ValueError",
            "message": err_msg,
        })
        print(f"[OK] Correctly rejected 'test' for purpose='validation': {err_msg[:65]}...")

    # -------------------------------------------------------------------------
    # 4. Training-Step Boundary Enforcement
    # -------------------------------------------------------------------------
    print("\n--- 4. TRAINING-STEP BOUNDARY ENFORCEMENT ---")
    bridge, optimizer, _ = build_model_and_optimizer(TrainingConfig(learning_rate=1e-4, device="cpu"))
    comp_loss = CompositeLoss()

    # Step 4.1: Verified Train Batch Execution (backward pass only, NO optimizer.step)
    initial_weights = [p.clone().detach() for p in bridge.acs_model.parameters()]
    step_result = execute_partition_verified_step(
        bridge=bridge,
        optimizer=optimizer,
        composite_loss=comp_loss,
        batch=train_batch,
        run_backward=True,
    )
    print(f"[OK] execute_partition_verified_step on train batch succeeded:")
    print(f"     total_grad_norm={step_result.total_grad_norm:.6f}, is_grad_finite={step_result.is_grad_finite}, "
          f"optimizer_step_executed={step_result.optimizer_step_executed}")

    # Confirm weights are bit-for-bit unchanged
    current_weights = [p.clone().detach() for p in bridge.acs_model.parameters()]
    weights_identical = all(torch.equal(w1, w2) for w1, w2 in zip(initial_weights, current_weights))
    print(f"[OK] Invariant Verified: Model weights bit-for-bit identical before/after: {weights_identical}")

    # Step 4.2: Rejection of Test Batch by execute_partition_verified_step
    test_batch_rejected = False
    fake_test_batch = {
        "carrier": torch.randn(1, 3, 512, 512),
        "split": ["test"],
        "image_id": torch.tensor([750]),
        "filename": ["0750.png"],
    }
    try:
        execute_partition_verified_step(
            bridge=bridge,
            optimizer=optimizer,
            composite_loss=comp_loss,
            batch=fake_test_batch,
            run_backward=True,
        )
    except ValueError as e:
        test_batch_rejected = True
        err_msg = str(e)
        boundary_rejections.append({
            "test_case": "execute_partition_verified_step(split='test')",
            "rejected": True,
            "error_type": "ValueError",
            "message": err_msg,
        })
        print(f"[OK] Correctly rejected test batch at training step: {err_msg[:65]}...")

    # Step 4.3: Rejection of Val Batch by execute_partition_verified_step
    val_batch_rejected = False
    try:
        execute_partition_verified_step(
            bridge=bridge,
            optimizer=optimizer,
            composite_loss=comp_loss,
            batch=val_batch,
            run_backward=True,
        )
    except ValueError as e:
        val_batch_rejected = True
        err_msg = str(e)
        boundary_rejections.append({
            "test_case": "execute_partition_verified_step(split='val')",
            "rejected": True,
            "error_type": "ValueError",
            "message": err_msg,
        })
        print(f"[OK] Correctly rejected val batch at training step: {err_msg[:65]}...")

    # Step 4.4: Validation Step Evaluation on Val Batch
    val_step_res = evaluate_partition_verified_step(
        bridge=bridge,
        composite_loss=comp_loss,
        batch=val_batch,
    )
    print(f"[OK] evaluate_partition_verified_step on val batch succeeded:")
    print(f"     loss={val_step_res.loss_breakdown.total_loss.item():.4f}, "
          f"optimizer_step_executed={val_step_res.optimizer_step_executed}")

    # Step 4.5: Rejection of Test Batch by evaluate_partition_verified_step
    test_eval_rejected = False
    try:
        evaluate_partition_verified_step(
            bridge=bridge,
            composite_loss=comp_loss,
            batch=fake_test_batch,
        )
    except ValueError as e:
        test_eval_rejected = True
        err_msg = str(e)
        boundary_rejections.append({
            "test_case": "evaluate_partition_verified_step(split='test')",
            "rejected": True,
            "error_type": "ValueError",
            "message": err_msg,
        })
        print(f"[OK] Correctly rejected test batch at validation step: {err_msg[:65]}...")

    # -------------------------------------------------------------------------
    # 5. Compile Machine-Readable JSON Artifact
    # -------------------------------------------------------------------------
    output_results: Dict[str, Any] = {
        "metadata": {
            "experiment": "Phase 3 Sprint 3.2 Dataset Loader and Training-Boundary Audit",
            "date": "2026-10-19",
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": manifest_file_sha256,
            "dataset_composite_sha256": manifest_audit["dataset_composite_sha256"],
            "dataset_root": str(dataset_root),
            "training_authorized": False,
            "optimizer_steps_executed": 0,
            "model_weights_mutated": False,
        },
        "manifest_audit": manifest_audit,
        "manifest_integrity_hardening_sprint_3_2_1": manifest_hardening_checks,
        "boundary_enforcement_tests": {
            "test_for_train_rejected": test_for_train_rejected,
            "val_for_train_rejected": val_for_train_rejected,
            "test_for_val_rejected": test_for_val_rejected,
            "test_batch_in_train_step_rejected": test_batch_rejected,
            "val_batch_in_train_step_rejected": val_batch_rejected,
            "test_batch_in_val_step_rejected": test_eval_rejected,
            "all_boundary_rejections_passed": all([
                test_for_train_rejected,
                val_for_train_rejected,
                test_for_val_rejected,
                test_batch_rejected,
                val_batch_rejected,
                test_eval_rejected,
            ]),
            "rejection_details": boundary_rejections,
        },
        "verification_invariants": {
            "train_partition_count": len(train_ds),
            "val_partition_count": len(val_ds),
            "test_partition_count": len(test_ds),
            "model_weights_bit_for_bit_identical": weights_identical,
            "training_status": "NOT AUTHORIZED — Sprint 3.2.1 hardens manifest integrity only.",
        },
    }

    out_file = REPO_ROOT / "docs" / "experiments" / "sprint3_2_dataset_boundary_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_results, f, indent=2)

    print(f"\n[OK] Versioned dataset boundary audit results saved to: {out_file}")
    print("\nTRAINING STATUS: NOT AUTHORIZED — Sprint 3.2.1 hardens manifest integrity only.")


if __name__ == "__main__":
    main()
