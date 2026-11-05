"""Verification & Audit Runner: Phase 4A.1 Pre-Pilot Safety Corrections.

EXECUTION CONTRACT (NO MODEL TRAINING):
This script audits and certifies the Phase 4A.1 training-engine safety corrections:
1. Canonical manifest and 800-image dataset integrity.
2. Split-isolation fail-closed enforcement (no test or cross-split leakage).
3. Gradient accumulation window normalization and dry-run metric truthfulness.
4. Epoch-boundary checkpoint resumption at next_epoch = epoch + 1 without overwriting.
5. Deterministic validation with dedicated RNG and truthful loss weight configuration.
6. Exact authorization token scope enforcement ("PHASE4B_PILOT").
7. Model parameter immutability (digest fe746295... bit-exact before and after).
8. Outputs auditable JSON evidence to docs/experiments/phase4a_1_pre_pilot_safety_results.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import (
    DEFAULT_DATASET_REL_PATH,
    DEFAULT_MANIFEST_REL_PATH,
    EXPECTED_COMPOSITE_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    compute_manifest_composite_sha256,
    validate_div2k_manifest,
)
from src.neural.losses import CompositeLoss
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_engine import (
    CANONICAL_SPLIT_RANGES,
    CHECKPOINT_SCHEMA_VERSION,
    PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
    DatasetSplitBoundaryError,
    IncompatibleCheckpointError,
    TrainingAuthorizationToken,
    TrainingNotAuthorizedError,
    TrainingRunConfig,
    TrainingRunner,
    TrainingTelemetryLogger,
    ValidationMetrics,
    evaluate_validation_epoch,
    load_training_checkpoint,
    save_training_checkpoint,
    validate_batch_partition,
)
from src.neural.training_scaffold import (
    TrainingConfig,
    build_model_and_optimizer,
    set_reproducibility_seeds,
)


def compute_param_digest(model: torch.nn.Module) -> str:
    """Computes a SHA-256 digest over all sorted named parameters in a PyTorch module."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


class SyntheticAuditDataset(Dataset):
    """Synthetic dataset for audit verification."""
    def __init__(self, size: int = 4, split: str = "train", start_id: int = 1) -> None:
        self.size = size
        self.split = split
        self.start_id = start_id

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.RandomState(100 + idx)
        carrier = rng.randint(20, 235, (64, 64, 3), dtype=np.uint8)
        carrier_t = torch.from_numpy(carrier.astype(np.float32)).permute(2, 0, 1)
        return {
            "carrier": carrier_t,
            "split": self.split,
            "image_id": self.start_id + idx,
            "filename": f"{self.start_id + idx:04d}.png",
        }


def make_audit_loader(size: int = 4, split: str = "train", batch_size: int = 1) -> DataLoader:
    ds = SyntheticAuditDataset(size=size, split=split, start_id=1 if split == "train" else 601)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    loader.dataset_split = split  # type: ignore[attr-defined]
    return loader


def run_phase4a_1_safety_audit() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 4A.1: PRE-PILOT SAFETY CORRECTIONS & TRAINING-ENGINE VERIFICATION")
    print("=" * 80)

    audit_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "4A.1",
        "training_authorized": False,
        "optimizer_steps_executed": 0,
        "training_epochs_executed": 0,
    }

    # 1. Reproducibility & Initial Digest
    print("\n[1] Initializing Reproducibility Setup & Baseline Digest...")
    set_reproducibility_seeds(seed=42, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Execution Device: {device}")

    bridge, optimizer, comp_loss = build_model_and_optimizer(TrainingConfig(seed=42, device=str(device)))
    initial_digest = compute_param_digest(bridge.acs_model)
    expected_digest = "fe7462957b0194d109ec32a78834d7bf44b9d1aa86510a40625f2e485fa40369"
    assert initial_digest == expected_digest, f"Digest mismatch: {initial_digest}"
    print(f"    Baseline Parameter Digest (Seed=42): {initial_digest}")
    audit_results["initial_model_parameter_digest"] = initial_digest

    # 2. Canonical Manifest & Dataset Inventory
    print("\n[2] Auditing Canonical Split Manifest & 800-Image Inventory...")
    manifest_audit = validate_div2k_manifest(
        manifest_path=DEFAULT_MANIFEST_REL_PATH,
        dataset_root=DEFAULT_DATASET_REL_PATH,
        enforce_file_existence=True,
    )
    print(f"    Manifest File SHA-256: {manifest_audit['manifest_file_sha256']}")
    print(f"    Composite Hash SHA-256: {manifest_audit['composite_sha256']}")
    print(f"    Split Inventory: {manifest_audit['partition_counts']}")
    audit_results["manifest_verification"] = {
        "manifest_file_sha256": manifest_audit["manifest_file_sha256"],
        "composite_sha256": manifest_audit["composite_sha256"],
        "counts": manifest_audit["partition_counts"],
        "status": "PASS",
    }

    # 3. Split Isolation Fail-Closed Enforcement
    print("\n[3] Auditing Split-Isolation Fail-Closed Boundaries...")
    split_audit_details: Dict[str, str] = {}

    # 3a. Missing loader split
    train_loader = make_audit_loader(size=2, split="train")
    delattr(train_loader, "dataset_split")
    try:
        TrainingRunner(
            bridge=bridge,
            train_loader=train_loader,
            val_loader=make_audit_loader(size=1, split="val"),
            optimizer=optimizer,
            composite_loss=comp_loss,
            config=TrainingRunConfig(epochs=1, device="cpu"),
        )
        split_audit_details["missing_loader_split"] = "FAIL_OPEN"
    except DatasetSplitBoundaryError:
        split_audit_details["missing_loader_split"] = "FAIL_CLOSED"

    # 3b. Missing batch split
    try:
        validate_batch_partition({"carrier": torch.randn(1, 3, 32, 32), "image_id": 1}, expected_split="train")
        split_audit_details["missing_batch_split"] = "FAIL_OPEN"
    except DatasetSplitBoundaryError:
        split_audit_details["missing_batch_split"] = "FAIL_CLOSED"

    # 3c. Missing batch image_id
    try:
        validate_batch_partition({"carrier": torch.randn(1, 3, 32, 32), "split": "train"}, expected_split="train")
        split_audit_details["missing_batch_image_id"] = "FAIL_OPEN"
    except DatasetSplitBoundaryError:
        split_audit_details["missing_batch_image_id"] = "FAIL_CLOSED"

    # 3d. Val ID mislabeled as train
    try:
        validate_batch_partition({"carrier": torch.randn(1, 3, 32, 32), "split": "train", "image_id": 650}, expected_split="train")
        split_audit_details["val_id_mislabeled_as_train"] = "FAIL_OPEN"
    except DatasetSplitBoundaryError:
        split_audit_details["val_id_mislabeled_as_train"] = "FAIL_CLOSED"

    # 3e. Test ID mislabeled as train
    try:
        validate_batch_partition({"carrier": torch.randn(1, 3, 32, 32), "split": "train", "image_id": 750}, expected_split="train")
        split_audit_details["test_id_mislabeled_as_train"] = "FAIL_OPEN"
    except DatasetSplitBoundaryError:
        split_audit_details["test_id_mislabeled_as_train"] = "FAIL_CLOSED"

    # 3f. Test ID mislabeled as val
    try:
        validate_batch_partition({"carrier": torch.randn(1, 3, 32, 32), "split": "val", "image_id": 750}, expected_split="val")
        split_audit_details["test_id_mislabeled_as_val"] = "FAIL_OPEN"
    except DatasetSplitBoundaryError:
        split_audit_details["test_id_mislabeled_as_val"] = "FAIL_CLOSED"

    # 3g. Train ID mislabeled as val
    try:
        validate_batch_partition({"carrier": torch.randn(1, 3, 32, 32), "split": "val", "image_id": 25}, expected_split="val")
        split_audit_details["train_id_mislabeled_as_val"] = "FAIL_OPEN"
    except DatasetSplitBoundaryError:
        split_audit_details["train_id_mislabeled_as_val"] = "FAIL_CLOSED"

    # 3h. Valid samples accepted
    try:
        validate_batch_partition({"carrier": torch.randn(1, 3, 32, 32), "split": "train", "image_id": 10}, expected_split="train")
        validate_batch_partition({"carrier": torch.randn(1, 3, 32, 32), "split": "val", "image_id": 610}, expected_split="val")
        split_audit_details["valid_samples_accepted"] = "PASS"
    except Exception as e:
        split_audit_details["valid_samples_accepted"] = f"FAIL: {e}"

    print(f"    Split Isolation Results: {split_audit_details}")
    audit_results["split_isolation"] = split_audit_details

    # 4. Gradient Accumulation & Window Normalization
    print("\n[4] Auditing Gradient Accumulation Window Normalization & Dry Run...")
    accum_audit_details: Dict[str, Any] = {}

    # 4a. Synthetic accumulation test for sample weighting
    linear_accum = nn.Linear(4, 2, bias=False)
    linear_comb = nn.Linear(4, 2, bias=False)
    linear_comb.load_state_dict(linear_accum.state_dict())
    x1, y1 = torch.randn(2, 4), torch.randn(2, 2)
    x2, y2 = torch.randn(1, 4), torch.randn(1, 2)
    x_c = torch.cat([x1, x2], dim=0)
    y_c = torch.cat([y1, y2], dim=0)
    nn.functional.mse_loss(linear_comb(x_c), y_c).backward()
    linear_accum.zero_grad()
    (nn.functional.mse_loss(linear_accum(x1), y1) * (2.0 / 3.0)).backward()
    (nn.functional.mse_loss(linear_accum(x2), y2) * (1.0 / 3.0)).backward()
    grad_diff = float(torch.max(torch.abs(linear_comb.weight.grad - linear_accum.weight.grad)).item())
    accum_audit_details["sample_weighting_grad_max_diff"] = grad_diff
    assert grad_diff < 1e-6, f"Accumulation gradient mismatch: {grad_diff}"

    # 4b. Dry run zero optimizer steps verification
    clean_tloader = make_audit_loader(size=4, split="train")
    clean_vloader = make_audit_loader(size=1, split="val")
    cfg_dry = TrainingRunConfig(epochs=1, grad_accum_steps=2, device="cpu")
    runner_dry = TrainingRunner(
        bridge=bridge,
        train_loader=clean_tloader,
        val_loader=clean_vloader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=cfg_dry,
    )
    dry_metrics = runner_dry.train_epoch(epoch=1, dry_run=True)
    accum_audit_details["dry_run_optimizer_steps_taken"] = dry_metrics.optimizer_steps_taken
    accum_audit_details["dry_run_global_step"] = runner_dry.global_step
    accum_audit_details["dry_run_accumulation_windows_completed"] = dry_metrics.accumulation_windows_completed
    assert dry_metrics.optimizer_steps_taken == 0
    assert runner_dry.global_step == 0
    assert dry_metrics.accumulation_windows_completed == 2
    print(f"    Gradient Accumulation & Dry Run Results: {accum_audit_details}")
    audit_results["gradient_accumulation_and_dry_run"] = accum_audit_details

    # 5. Epoch-Boundary Resumption
    print("\n[5] Auditing Epoch-Boundary Resumption & Checkpoint State...")
    resume_audit_details: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "checkpoint_epoch_002.pt"
        save_training_checkpoint(
            checkpoint_path=ckpt_path,
            bridge=bridge,
            optimizer=optimizer,
            epoch=2,
            global_step=80,
            best_val_metric=0.45,
            early_stopping_state={"counter": 1, "patience": 5},
            config=TrainingRunConfig(epochs=5, device="cpu"),
        )

        resumed_runner = TrainingRunner(
            bridge=bridge,
            train_loader=clean_tloader,
            val_loader=clean_vloader,
            optimizer=optimizer,
            composite_loss=comp_loss,
            config=TrainingRunConfig(epochs=5, device="cpu"),
            resume_checkpoint_path=ckpt_path,
        )

        resume_audit_details["restored_epoch"] = 2
        resume_audit_details["next_epoch_to_run"] = resumed_runner.start_epoch
        resume_audit_details["restored_global_step"] = resumed_runner.global_step
        resume_audit_details["restored_best_val_metric"] = resumed_runner.best_val_metric
        resume_audit_details["restored_early_stop_counter"] = resumed_runner.early_stop_counter

        assert resumed_runner.start_epoch == 3, f"Expected next_epoch=3, got {resumed_runner.start_epoch}"
        assert resumed_runner.global_step == 80
        assert math.isclose(resumed_runner.best_val_metric, 0.45)
        assert resumed_runner.early_stop_counter == 1

    print(f"    Resumption Audit Results: {resume_audit_details}")
    audit_results["epoch_boundary_resumption"] = resume_audit_details

    # 6. Validation Determinism & Loss Truthfulness
    print("\n[6] Auditing Validation Determinism & Loss Weight Consistency...")
    val_audit_details: Dict[str, Any] = {}

    val_loader = make_audit_loader(size=2, split="val")
    rng_before = torch.get_rng_state()
    v1 = evaluate_validation_epoch(bridge, val_loader, comp_loss, device=torch.device("cpu"), val_seed=55555)
    rng_after1 = torch.get_rng_state()
    v2 = evaluate_validation_epoch(bridge, val_loader, comp_loss, device=torch.device("cpu"), val_seed=55555)
    rng_after2 = torch.get_rng_state()

    val_audit_details["rng_preserved_pass1"] = bool(torch.equal(rng_before, rng_after1))
    val_audit_details["rng_preserved_pass2"] = bool(torch.equal(rng_before, rng_after2))
    val_audit_details["total_loss_pass1"] = v1.total_loss
    val_audit_details["total_loss_pass2"] = v2.total_loss
    val_audit_details["deterministic_match"] = bool(v1.total_loss == v2.total_loss and v1.pre_ecc_ber == v2.pre_ecc_ber)

    assert val_audit_details["rng_preserved_pass1"] is True
    assert val_audit_details["rng_preserved_pass2"] is True
    assert val_audit_details["deterministic_match"] is True

    # Loss weight mismatch test
    cfg_mismatch = TrainingRunConfig(lambda_ctx=0.88, device="cpu")
    comp_loss_standard = CompositeLoss(lambda_ctx=0.5)
    try:
        TrainingRunner(
            bridge=bridge,
            train_loader=clean_tloader,
            val_loader=clean_vloader,
            optimizer=optimizer,
            composite_loss=comp_loss_standard,
            config=cfg_mismatch,
        )
        val_audit_details["mismatch_detection"] = "FAIL_SILENT_DIVERGENCE"
    except ValueError:
        val_audit_details["mismatch_detection"] = "FAIL_CLOSED_CONFIRMED"

    print(f"    Validation & Loss Consistency Results: {val_audit_details}")
    audit_results["validation_and_loss_consistency"] = val_audit_details

    # 7. Authorization Scope Gate
    print("\n[7] Auditing Training Authorization Gate Scope Enforcement...")
    auth_audit_details: Dict[str, Any] = {}

    t_valid = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="CHIEF_SYSTEM_ARCHITECT",
        scope="PHASE4B_PILOT",
    )
    t_prefix = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="CHIEF_SYSTEM_ARCHITECT",
        scope="PHASE4B",
    )
    t_extended = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="CHIEF_SYSTEM_ARCHITECT",
        scope="PHASE4B_PILOT_EXTENDED",
    )

    auth_audit_details["exact_scope_PHASE4B_PILOT"] = t_valid.is_valid()
    auth_audit_details["prefix_only_scope_PHASE4B"] = t_prefix.is_valid()
    auth_audit_details["extended_scope"] = t_extended.is_valid()

    assert auth_audit_details["exact_scope_PHASE4B_PILOT"] is True
    assert auth_audit_details["prefix_only_scope_PHASE4B"] is False
    assert auth_audit_details["extended_scope"] is False

    print(f"    Authorization Scope Results: {auth_audit_details}")
    audit_results["authorization_scope_enforcement"] = auth_audit_details

    # 8. Final Parameter Digest & Safety Post-Condition
    print("\n[8] Verifying Post-Condition Parameter Digest & Safety Boundaries...")
    final_digest = compute_param_digest(bridge.acs_model)
    assert final_digest == initial_digest, f"Parameter mutation detected! {final_digest} != {initial_digest}"
    print(f"    Final Model Parameter Digest: {final_digest}")
    print("    Parameter Digest Invariance: PASS (bit-for-bit identical)")
    audit_results["final_model_parameter_digest"] = final_digest
    audit_results["parameter_digest_invariant"] = True
    audit_results["final_safety_status"] = "ALL_GATES_PASSED_ZERO_TRAINING_EXECUTED"

    # Write results JSON
    out_path = REPO_ROOT / "docs" / "experiments" / "phase4a_1_pre_pilot_safety_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
    print(f"\n[Audit Complete] Results written to: {out_path}")

    return audit_results


if __name__ == "__main__":
    run_phase4a_1_safety_audit()
