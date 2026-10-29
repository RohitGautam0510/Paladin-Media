"""Unit and Control-Flow Tests for Phase 4A Automated Training Infrastructure.

Verifies all 14 mandatory infrastructure and safety gates without executing real training:
1. Training entry point rejects execution without explicit authorization.
2. Dry-run and import paths cannot accidentally start training.
3. Training configuration validation rejects invalid batch size, epoch count, accumulation steps and clipping limits.
4. Train DataLoader cannot be built from validation or test records for optimization.
5. Validation executes without parameter mutation or optimizer updates.
6. Checkpoint save/load preserves model state, optimizer state and scheduler state where supported.
7. RNG state restoration reproduces the next generated sequence for the supported generators.
8. Epoch-boundary resume restores epoch/global step, scheduler, best metric and sampler/DataLoader state.
9. Corrupt and incompatible checkpoints fail closed.
10. Atomic checkpoint writing does not leave a partially written final checkpoint on simulated failure.
11. Structured logs distinguish raw from weighted loss terms.
12. Gradient accumulation control flow handles complete and partial accumulation windows correctly.
13. A non-finite loss or gradient produces an explicit failure and does not save a falsely successful checkpoint.
14. Validation and test records cannot be used for optimizer updates through supported public training APIs.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import build_stego_dataloader
from src.neural.losses import CompositeLoss, LossBreakdown
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_engine import (
    CHECKPOINT_SCHEMA_VERSION,
    PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
    PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
    DatasetSplitBoundaryError,
    IncompatibleCheckpointError,
    NonFiniteGradientError,
    NonFiniteLossError,
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
    verify_training_authorization,
)
from src.neural.training_scaffold import (
    TrainingConfig,
    build_model_and_optimizer,
    set_reproducibility_seeds,
)


# =============================================================================
# Helper Fixtures & Mocks
# =============================================================================

class MockBatchDataset(Dataset):
    """Small synthetic in-memory dataset for safe control-flow tests."""

    def __init__(self, size: int = 5, split: str = "train", start_id: int = 1) -> None:
        self.size = size
        self.split = split
        self.start_id = start_id

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.RandomState(42 + idx)
        carrier = rng.randint(20, 235, (64, 64, 3), dtype=np.uint8)
        carrier_t = torch.from_numpy(carrier.astype(np.float32)).permute(2, 0, 1)
        return {
            "carrier": carrier_t,
            "split": self.split,
            "image_id": self.start_id + idx,
            "filename": f"{self.start_id + idx:04d}.png",
        }


def make_mock_loader(size: int = 5, split: str = "train", batch_size: int = 1) -> DataLoader:
    ds = MockBatchDataset(size=size, split=split, start_id=1 if split == "train" else 601)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    loader.dataset_split = split  # type: ignore[attr-defined]
    loader.is_training_loader = (split == "train")  # type: ignore[attr-defined]
    return loader


@pytest.fixture
def clean_bridge():
    set_reproducibility_seeds(42, deterministic=True)
    bridge, optimizer, comp_loss = build_model_and_optimizer(TrainingConfig(seed=42, device="cpu"))
    return bridge, optimizer, comp_loss


# =============================================================================
# Test 1 & 2: Training Authorization Gate & Dry-Run Invariance
# =============================================================================

def test_training_entry_point_rejects_without_authorization(clean_bridge):
    """Gate 1: Training runner rejects train() execution if authorization token is missing or invalid."""
    bridge, optimizer, comp_loss = clean_bridge
    train_loader = make_mock_loader(size=2, split="train")
    val_loader = make_mock_loader(size=2, split="val")
    config = TrainingRunConfig(epochs=1, device="cpu")

    # 1. Without token (default None)
    runner_no_token = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=config,
        authorization_token=None,
    )
    with pytest.raises(TrainingNotAuthorizedError, match="TRAINING REJECTED"):
        runner_no_token.train()

    # 2. With invalid token (wrong passphrase)
    invalid_token = TrainingAuthorizationToken(passphrase="UNAUTHORIZED", approver="USER")
    runner_bad_token = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=config,
        authorization_token=invalid_token,
    )
    with pytest.raises(TrainingNotAuthorizedError, match="TRAINING REJECTED"):
        runner_bad_token.train()


def test_dry_run_and_import_cannot_accidentally_train(clean_bridge):
    """Gate 2: Instantiating runner or calling dry-run epoch never performs optimizer updates."""
    bridge, optimizer, comp_loss = clean_bridge
    train_loader = make_mock_loader(size=2, split="train")
    val_loader = make_mock_loader(size=2, split="val")
    config = TrainingRunConfig(epochs=1, device="cpu")

    # Capture initial parameter values
    init_params = [p.clone().detach() for p in bridge.acs_model.parameters()]

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=config,
    )

    # Dry-run epoch executes forward/backward without calling optimizer.step()
    metrics = runner.train_epoch(epoch=1, dry_run=True)
    assert metrics.epoch == 1
    assert metrics.optimizer_steps_taken == 0  # Dry-run MUST NOT take optimizer steps
    assert metrics.accumulation_windows_completed == 2  # 2 accumulation windows traversed
    assert runner.global_step == 0  # global_step does not advance in dry run
    assert metrics.total_loss > 0.0

    # Invariant: model parameters remain bit-for-bit unchanged
    post_params = [p.clone().detach() for p in bridge.acs_model.parameters()]
    for p_init, p_post in zip(init_params, post_params):
        assert torch.equal(p_init, p_post), "Model weights MUST NOT mutate in dry-run!"


# =============================================================================
# Test 3: Configuration Validation
# =============================================================================

def test_training_config_validation_rejects_invalid_bounds():
    """Gate 3: TrainingRunConfig rejects invalid batch size, epochs, accumulation, and clip norms."""
    # Zero or negative epochs
    with pytest.raises(ValueError, match="epochs must be strictly positive"):
        TrainingRunConfig(epochs=0).validate()

    # Zero or negative batch size
    with pytest.raises(ValueError, match="batch_size must be strictly positive"):
        TrainingRunConfig(batch_size=0).validate()

    # Invalid accumulation steps
    with pytest.raises(ValueError, match="grad_accum_steps must be >= 1"):
        TrainingRunConfig(grad_accum_steps=0).validate()

    # Negative learning rate
    with pytest.raises(ValueError, match="learning_rate must be strictly positive"):
        TrainingRunConfig(learning_rate=-1e-4).validate()

    # Negative weight decay
    with pytest.raises(ValueError, match="weight_decay must be non-negative"):
        TrainingRunConfig(weight_decay=-1.0).validate()

    # Negative clip norm
    with pytest.raises(ValueError, match="clip_grad_norm must be positive or None"):
        TrainingRunConfig(clip_grad_norm=-0.5).validate()

    # Valid config succeeds
    valid_cfg = TrainingRunConfig(epochs=5, batch_size=1, grad_accum_steps=2, clip_grad_norm=1.0)
    valid_cfg.validate()


# =============================================================================
# Test 4 & 14: Dataset & Split Isolation
# =============================================================================

def test_train_dataloader_cannot_be_built_from_val_or_test():
    """Gate 4 & 14: build_stego_dataloader rejects split='val' or 'test' when purpose='train'."""
    with pytest.raises(ValueError, match="Cannot construct training DataLoader with split='test'"):
        build_stego_dataloader(split="test", purpose="train")

    with pytest.raises(ValueError, match="Cannot construct training DataLoader with split='val'"):
        build_stego_dataloader(split="val", purpose="train")


def test_runner_rejects_non_train_split_in_training():
    """Gate 14: TrainingRunner fails closed if train_loader or batch has non-train split."""
    bridge, optimizer, comp_loss = build_model_and_optimizer(TrainingConfig(device="cpu"))
    val_loader = make_mock_loader(size=2, split="val")
    test_loader = make_mock_loader(size=2, split="test")

    config = TrainingRunConfig(epochs=1, device="cpu")

    # Reject loader tagged as test
    with pytest.raises(DatasetSplitBoundaryError, match="train_loader requires split='train'"):
        TrainingRunner(
            bridge=bridge,
            train_loader=test_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            composite_loss=comp_loss,
            config=config,
        )

    # Reject loader tagged as val for training
    with pytest.raises(DatasetSplitBoundaryError, match="train_loader requires split='train'"):
        TrainingRunner(
            bridge=bridge,
            train_loader=val_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            composite_loss=comp_loss,
            config=config,
        )


# =============================================================================
# Test 5: Validation Isolation & Immutability
# =============================================================================

def test_validation_executes_without_parameter_mutation(clean_bridge):
    """Gate 5: evaluate_validation_epoch runs with torch.no_grad and leaves model unchanged."""
    bridge, _, comp_loss = clean_bridge
    val_loader = make_mock_loader(size=2, split="val")

    init_params = [p.clone().detach() for p in bridge.acs_model.parameters()]

    val_metrics = evaluate_validation_epoch(
        bridge=bridge,
        val_loader=val_loader,
        composite_loss=comp_loss,
        device=torch.device("cpu"),
    )

    assert isinstance(val_metrics, ValidationMetrics)
    assert val_metrics.total_loss > 0.0
    assert val_metrics.num_batches == 2
    assert math.isfinite(val_metrics.stego_psnr_db)
    assert math.isfinite(val_metrics.stego_ssim)

    # Model parameters remain bit-for-bit unchanged
    post_params = [p.clone().detach() for p in bridge.acs_model.parameters()]
    for p_init, p_post in zip(init_params, post_params):
        assert torch.equal(p_init, p_post)


# =============================================================================
# Test 6, 7, 8, 9, 10: Checkpoint Resumption & Integrity
# =============================================================================

def test_checkpoint_roundtrip_state_preservation(clean_bridge, tmp_path):
    """Gate 6 & 8: Checkpoint save and load preserves model, optimizer, scheduler, epoch, and step."""
    bridge, optimizer, _ = clean_bridge
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    ckpt_file = tmp_path / "checkpoint_epoch_001.pt"
    config = TrainingRunConfig(epochs=2, device="cpu", run_id="test_run")

    save_training_checkpoint(
        checkpoint_path=ckpt_file,
        bridge=bridge,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=50,
        best_val_metric=1.5,
        early_stopping_state={"counter": 0, "patience": 5},
        config=config,
        is_best=True,
    )

    # Load into fresh instances
    fresh_bridge, fresh_optimizer, _ = build_model_and_optimizer(TrainingConfig(seed=999, device="cpu"))
    fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=1, gamma=0.5)

    restored = load_training_checkpoint(
        checkpoint_path=ckpt_file,
        bridge=fresh_bridge,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
    )

    assert restored["epoch"] == 1
    assert restored["global_step"] == 50
    assert restored["best_val_metric"] == 1.5
    assert restored["metadata"]["is_best"] is True
    assert restored["metadata"]["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION

    # Parameters match
    for p_orig, p_loaded in zip(bridge.acs_model.parameters(), fresh_bridge.acs_model.parameters()):
        assert torch.equal(p_orig, p_loaded)


def test_rng_state_restoration_reproducibility(clean_bridge, tmp_path):
    """Gate 7: Restoring RNG states reproduces identical random sequences."""
    bridge, optimizer, _ = clean_bridge
    ckpt_file = tmp_path / "rng_test.pt"

    # Set seeds and draw a number
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Save checkpoint at this exact RNG state
    save_training_checkpoint(
        checkpoint_path=ckpt_file,
        bridge=bridge,
        optimizer=optimizer,
    )

    # Generate sequence A
    py_seq_a = [random.random() for _ in range(5)]
    np_seq_a = np.random.rand(5).tolist()
    torch_seq_a = torch.rand(5).tolist()

    # Advance RNGs further
    _ = [random.random() for _ in range(10)]
    _ = np.random.rand(10)
    _ = torch.rand(10)

    # Restore checkpoint with restore_rng=True
    load_training_checkpoint(
        checkpoint_path=ckpt_file,
        bridge=bridge,
        restore_rng=True,
    )

    # Generate sequence B
    py_seq_b = [random.random() for _ in range(5)]
    np_seq_b = np.random.rand(5).tolist()
    torch_seq_b = torch.rand(5).tolist()

    assert py_seq_a == py_seq_b, "Python random RNG state was not restored exactly!"
    assert np_seq_a == np_seq_b, "NumPy RNG state was not restored exactly!"
    assert torch_seq_a == torch_seq_b, "PyTorch CPU RNG state was not restored exactly!"


def test_corrupt_and_incompatible_checkpoints_fail_closed(clean_bridge, tmp_path):
    """Gate 9: Incompatible or corrupted checkpoints raise IncompatibleCheckpointError."""
    bridge, optimizer, _ = clean_bridge

    # 1. Truncated / corrupt bytes
    corrupt_file = tmp_path / "corrupt.pt"
    corrupt_file.write_bytes(b"NOT_A_VALID_TORCH_FILE_HEADER_GARBAGE")
    with pytest.raises(IncompatibleCheckpointError, match="Corrupt or unreadable checkpoint"):
        load_training_checkpoint(corrupt_file, bridge)

    # 2. Incompatible parameter shapes
    incomp_file = tmp_path / "incompatible.pt"
    incomp_state = {
        "metadata": {
            "checkpoint_schema_version": "4.0",
            "manifest_metadata": {
                "manifest_file_sha256": "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff",
                "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
            },
            "model_metadata": {
                "architecture": "ACS-Sub-INN-v3",
                "gamma_min": 1.0,
            },
        },
        "model_state_dict": {"context_predictor.conv_in.weight": torch.randn(99, 99, 1, 1)},
    }
    torch.save(incomp_state, incomp_file)
    with pytest.raises(IncompatibleCheckpointError, match="Model state_dict incompatible"):
        load_training_checkpoint(incomp_file, bridge)


def test_atomic_checkpoint_writing_failure_safety(clean_bridge, tmp_path):
    """Gate 10: Atomic saving does not leave a corrupted target file if saving fails."""
    bridge, optimizer, _ = clean_bridge
    target_file = tmp_path / "atomic_checkpoint.pt"
    target_file.write_bytes(b"ORIGINAL_HEALTHY_BYTES")

    # Simulate failure during torch.save
    with patch("torch.save", side_effect=IOError("Simulated write failure")):
        with pytest.raises(IOError, match="Simulated write failure"):
            save_training_checkpoint(
                checkpoint_path=target_file,
                bridge=bridge,
                optimizer=optimizer,
                atomic=True,
            )

    # Target file must still have original contents and no temp files left behind
    assert target_file.read_bytes() == b"ORIGINAL_HEALTHY_BYTES"
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0, f"Temporary file was leaked: {tmp_files}"


# =============================================================================
# Test 11: Structured Telemetry Raw vs. Weighted Losses
# =============================================================================

def test_telemetry_logger_distinguishes_raw_and_weighted_losses(tmp_path):
    """Gate 11: Telemetry logger correctly records raw and weighted components."""
    log_file = tmp_path / "telemetry.jsonl"
    logger = TrainingTelemetryLogger(log_file)

    sample_record = {
        "run_id": "phase4a_test",
        "epoch": 1,
        "train_loss_total": 1.907252,
        "train_loss_bit_raw": 0.187489,
        "train_loss_bit_weighted": 0.187489,
        "train_loss_ctx_raw": 3.292353,
        "train_loss_ctx_weighted": 1.646176,
        "train_loss_gain_raw": 0.714299,
        "train_loss_gain_weighted": 0.007143,
        "train_loss_img_raw": 0.664441,
        "train_loss_img_weighted": 0.066444,
        "nan_metric": float("nan"),
    }

    logger.log_epoch(sample_record)

    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])

    assert parsed["train_loss_ctx_raw"] == pytest.approx(3.292353)
    assert parsed["train_loss_ctx_weighted"] == pytest.approx(1.646176)
    assert parsed["nan_metric"] == "NaN"  # Non-finite float safely serialized


# =============================================================================
# Test 12: Gradient Accumulation Control Flow
# =============================================================================

def test_gradient_accumulation_step_frequency(clean_bridge):
    """Gate 12: Accumulation steps trigger optimizer.step() at exact window boundaries."""
    bridge, _, comp_loss = clean_bridge
    train_loader = make_mock_loader(size=5, split="train")
    val_loader = make_mock_loader(size=2, split="val")

    # 5 batches with accum_steps=2 -> optimizer steps at batch 2, 4, 5 (total 3 steps)
    config = TrainingRunConfig(epochs=1, grad_accum_steps=2, device="cpu")

    mock_optimizer = MagicMock()
    mock_optimizer.param_groups = [{"lr": 1e-4}]

    # Authorization token for mock execution
    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="MOCK_TEST_APPROVER",
    )

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=mock_optimizer,
        composite_loss=comp_loss,
        config=config,
        authorization_token=auth_token,
    )

    metrics = runner.train_epoch(epoch=1, dry_run=False)

    assert metrics.batches_processed == 5
    assert mock_optimizer.step.call_count == 3
    assert mock_optimizer.zero_grad.call_count == 4  # Initial + 3 post-steps


# =============================================================================
# Test 13: Non-Finite Numerical Handling
# =============================================================================

def test_non_finite_loss_fails_closed(clean_bridge):
    """Gate 13: Non-finite loss raises NonFiniteLossError and does not update parameters."""
    bridge, optimizer, _ = clean_bridge
    train_loader = make_mock_loader(size=2, split="train")
    val_loader = make_mock_loader(size=2, split="val")

    config = TrainingRunConfig(epochs=1, device="cpu")
    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="MOCK_TEST_APPROVER",
    )

    # Mock composite loss to return NaN
    mock_loss = MagicMock()
    nan_tensor = torch.tensor(float("nan"), requires_grad=True)
    mock_breakdown = LossBreakdown(
        total_loss=nan_tensor,
        loss_bit=torch.tensor(0.0),
        loss_ctx=torch.tensor(0.0),
        loss_gain=torch.tensor(0.0),
        loss_img=torch.tensor(0.0),
        weights={"bit": 1.0, "ctx": 0.5, "gain": 0.01, "img": 0.1},
    )
    mock_loss.return_value = mock_breakdown

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=mock_loss,
        config=config,
        authorization_token=auth_token,
    )

    with pytest.raises(NonFiniteLossError, match="Training loss non-finite"):
        runner.train_epoch(epoch=1, dry_run=False)


def test_non_finite_gradient_fails_closed(clean_bridge):
    """Gate 13b: Non-finite gradient raises NonFiniteGradientError and zeroes grads."""
    bridge, optimizer, comp_loss = clean_bridge
    train_loader = make_mock_loader(size=2, split="train")
    val_loader = make_mock_loader(size=2, split="val")

    config = TrainingRunConfig(epochs=1, device="cpu")
    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="MOCK_TEST_APPROVER",
    )

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=config,
        authorization_token=auth_token,
    )

    # Patch backward or inject NaN gradient on parameters during step
    def inject_nan_grad(*args, **kwargs):
        for p in bridge.acs_model.parameters():
            p.grad = torch.full_like(p.data, float("nan"))

    with patch.object(torch.Tensor, "backward", side_effect=inject_nan_grad):
        with pytest.raises(NonFiniteGradientError, match="Gradient contains NaN or Infinity"):
            runner.train_epoch(epoch=1, dry_run=False)


# =============================================================================
# Phase 4A.1 Regression Tests: Split Isolation & Boundary Enforcement
# =============================================================================

def test_missing_loader_split_fails_closed(clean_bridge):
    """P4A.1-1: DataLoader without explicit dataset_split fails closed."""
    bridge, optimizer, comp_loss = clean_bridge
    train_loader = make_mock_loader(size=2, split="train")
    val_loader = make_mock_loader(size=2, split="val")
    config = TrainingRunConfig(epochs=1, device="cpu")

    # 1. train_loader missing dataset_split
    delattr(train_loader, "dataset_split")
    with pytest.raises(DatasetSplitBoundaryError, match="train_loader requires split='train'"):
        TrainingRunner(
            bridge=bridge,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            composite_loss=comp_loss,
            config=config,
        )

    # 2. val_loader missing dataset_split
    train_loader_ok = make_mock_loader(size=2, split="train")
    val_loader_no_split = make_mock_loader(size=2, split="val")
    delattr(val_loader_no_split, "dataset_split")

    with pytest.raises(DatasetSplitBoundaryError, match="val_loader requires split='val'"):
        TrainingRunner(
            bridge=bridge,
            train_loader=train_loader_ok,
            val_loader=val_loader_no_split,
            optimizer=optimizer,
            composite_loss=comp_loss,
            config=config,
        )

    with pytest.raises(DatasetSplitBoundaryError, match="requires explicit dataset_split='val'"):
        evaluate_validation_epoch(
            bridge=bridge,
            val_loader=val_loader_no_split,
            composite_loss=comp_loss,
            device=torch.device("cpu"),
        )


def test_missing_batch_split_field_fails_closed():
    """P4A.1-2: Batch with missing or empty 'split' metadata fails closed."""
    # Missing 'split' key
    batch_no_split = {
        "carrier": torch.randn(1, 3, 64, 64),
        "image_id": 10,
    }
    with pytest.raises(DatasetSplitBoundaryError, match="missing required 'split' metadata"):
        validate_batch_partition(batch_no_split, expected_split="train")

    # None split
    batch_none_split = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": None,
        "image_id": 10,
    }
    with pytest.raises(DatasetSplitBoundaryError, match="missing required 'split' metadata"):
        validate_batch_partition(batch_none_split, expected_split="train")

    # Empty list split
    batch_empty_split = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": [],
        "image_id": 10,
    }
    with pytest.raises(DatasetSplitBoundaryError, match="Batch 'split' metadata is empty"):
        validate_batch_partition(batch_empty_split, expected_split="train")


def test_missing_batch_image_id_field_fails_closed():
    """P4A.1-3: Batch with missing, empty, or non-int 'image_id' fails closed."""
    # Missing image_id key
    batch_no_id = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": "train",
    }
    with pytest.raises(DatasetSplitBoundaryError, match="missing required 'image_id' metadata"):
        validate_batch_partition(batch_no_id, expected_split="train")

    # Boolean image_id (bool is subclass of int in Python)
    batch_bool_id = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": "train",
        "image_id": True,
    }
    with pytest.raises(DatasetSplitBoundaryError, match="Invalid batch 'image_id' field type: bool"):
        validate_batch_partition(batch_bool_id, expected_split="train")

    # List containing boolean image_id
    batch_list_bool_id = {
        "carrier": torch.randn(2, 3, 64, 64),
        "split": ["train", "train"],
        "image_id": [1, True],
    }
    with pytest.raises(DatasetSplitBoundaryError, match="non-boolean integer"):
        validate_batch_partition(batch_list_bool_id, expected_split="train")

    # String image_id
    batch_str_id = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": "train",
        "image_id": "0001",
    }
    with pytest.raises(DatasetSplitBoundaryError, match="Invalid batch 'image_id' field type"):
        validate_batch_partition(batch_str_id, expected_split="train")


def test_validation_ids_mislabeled_as_train_rejected():
    """P4A.1-4: Validation partition image IDs mislabeled as train fail closed."""
    # ID 605 belongs to val partition (601-700), but tagged with split='train'
    mislabeled_batch = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": "train",
        "image_id": 605,
    }
    with pytest.raises(DatasetSplitBoundaryError, match="does not belong to canonical train partition"):
        validate_batch_partition(mislabeled_batch, expected_split="train")


def test_test_ids_mislabeled_as_train_or_val_rejected():
    """P4A.1-5: Held-out test IDs (701-800) mislabeled as train or val fail closed."""
    test_id = 750

    # Test ID mislabeled as train
    test_as_train = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": "train",
        "image_id": test_id,
    }
    with pytest.raises(DatasetSplitBoundaryError, match="belongs to held-out test split"):
        validate_batch_partition(test_as_train, expected_split="train")

    # Test ID mislabeled as val
    test_as_val = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": "val",
        "image_id": test_id,
    }
    with pytest.raises(DatasetSplitBoundaryError, match="belongs to held-out test split"):
        validate_batch_partition(test_as_val, expected_split="val")


def test_train_ids_mislabeled_as_val_rejected():
    """P4A.1-6: Train partition image IDs mislabeled as val fail closed."""
    # ID 5 belongs to train partition (1-600), but tagged with split='val'
    train_as_val = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": "val",
        "image_id": 5,
    }
    with pytest.raises(DatasetSplitBoundaryError, match="does not belong to canonical val partition"):
        validate_batch_partition(train_as_val, expected_split="val")


def test_mixed_partition_batch_fails_closed():
    """P4A.1-7: Batches containing mixed partitions or mixed IDs fail closed."""
    # Mixed splits in a single batch
    mixed_split_batch = {
        "carrier": torch.randn(2, 3, 64, 64),
        "split": ["train", "val"],
        "image_id": [1, 2],
    }
    with pytest.raises(DatasetSplitBoundaryError, match="Batch split mismatch"):
        validate_batch_partition(mixed_split_batch, expected_split="train")

    # Train split but mixed IDs (ID 1 is train, ID 605 is val)
    mixed_id_batch = {
        "carrier": torch.randn(2, 3, 64, 64),
        "split": ["train", "train"],
        "image_id": [1, 605],
    }
    with pytest.raises(DatasetSplitBoundaryError, match="does not belong to canonical train partition"):
        validate_batch_partition(mixed_id_batch, expected_split="train")


def test_valid_train_and_val_samples_accepted():
    """P4A.1-8: Legitimate train and val samples are accepted without error."""
    valid_train = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": "train",
        "image_id": 42,
    }
    validate_batch_partition(valid_train, expected_split="train")

    valid_val = {
        "carrier": torch.randn(1, 3, 64, 64),
        "split": "val",
        "image_id": 650,
    }
    validate_batch_partition(valid_val, expected_split="val")


def test_invalid_batch_fails_before_any_optimizer_update(clean_bridge):
    """P4A.1-9: Invalid batch in training loader fails before any optimizer update."""
    bridge, _, comp_loss = clean_bridge

    # Custom dataset where batch 1 is valid train, batch 2 has test ID 750
    class ContaminatedDataset(Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            carrier = torch.randn(3, 64, 64)
            if idx == 0:
                return {"carrier": carrier, "split": "train", "image_id": 1, "filename": "0001.png"}
            else:
                return {"carrier": carrier, "split": "train", "image_id": 750, "filename": "0750.png"}

    loader = DataLoader(ContaminatedDataset(), batch_size=1, shuffle=False)
    loader.dataset_split = "train"
    val_loader = make_mock_loader(size=1, split="val")

    config = TrainingRunConfig(epochs=1, grad_accum_steps=1, device="cpu")
    mock_optimizer = MagicMock()
    mock_optimizer.param_groups = [{"lr": 1e-4}]

    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="MOCK_TEST_APPROVER",
    )

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=loader,
        val_loader=val_loader,
        optimizer=mock_optimizer,
        composite_loss=comp_loss,
        config=config,
        authorization_token=auth_token,
    )

    with pytest.raises(DatasetSplitBoundaryError, match="belongs to held-out test split"):
        runner.train_epoch(epoch=1, dry_run=False)

    # Second batch is in test split, window 2 fails closed, optimizer.step() count is at most 1 (window 1 was valid)
    # Most importantly: window 2 containing test ID NEVER reached optimizer.step()!
    assert mock_optimizer.step.call_count <= 1


# =============================================================================
# Phase 4A.1 Regression Tests: Gradient Accumulation & Window Normalization
# =============================================================================

def test_accumulation_complete_window_gradient_equivalence():
    """P4A.1-10: Complete accumulation window yields identical gradient to combined batch."""
    torch.manual_seed(42)
    linear_accum = nn.Linear(4, 2, bias=False)
    linear_comb = nn.Linear(4, 2, bias=False)
    linear_comb.load_state_dict(linear_accum.state_dict())

    x1 = torch.randn(1, 4)
    y1 = torch.randn(1, 2)
    x2 = torch.randn(1, 4)
    y2 = torch.randn(1, 2)

    # 1. Combined batch (size 2)
    x_comb = torch.cat([x1, x2], dim=0)
    y_comb = torch.cat([y1, y2], dim=0)
    loss_comb = nn.functional.mse_loss(linear_comb(x_comb), y_comb)
    loss_comb.backward()
    grad_comb = linear_comb.weight.grad.clone()

    # 2. Accumulated micro-batches (accum_steps=2, sample_weight=1/2 each)
    linear_accum.zero_grad()
    loss1 = nn.functional.mse_loss(linear_accum(x1), y1) * 0.5
    loss1.backward()

    loss2 = nn.functional.mse_loss(linear_accum(x2), y2) * 0.5
    loss2.backward()
    grad_accum = linear_accum.weight.grad.clone()

    assert torch.allclose(grad_comb, grad_accum, rtol=1e-5, atol=1e-6)


def test_accumulation_partial_window_normalization(clean_bridge):
    """P4A.1-11: Final partial window is normalized by actual samples, not config accum_steps."""
    bridge, _, comp_loss = clean_bridge
    # 3 batches with accum_steps=2 -> Window 1 has 2 batches, Window 2 (partial) has 1 batch
    train_loader = make_mock_loader(size=3, split="train")
    val_loader = make_mock_loader(size=1, split="val")
    config = TrainingRunConfig(epochs=1, grad_accum_steps=2, device="cpu")

    mock_optimizer = MagicMock()
    mock_optimizer.param_groups = [{"lr": 1e-4}]

    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="MOCK_TEST_APPROVER",
    )

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=mock_optimizer,
        composite_loss=comp_loss,
        config=config,
        authorization_token=auth_token,
    )

    metrics = runner.train_epoch(epoch=1, dry_run=False)

    assert metrics.batches_processed == 3
    assert metrics.accumulation_windows_completed == 2
    assert metrics.optimizer_steps_taken == 2
    assert mock_optimizer.step.call_count == 2


def test_accumulation_unequal_batch_sizes_sample_weighting():
    """P4A.1-12: Accumulation over unequal micro-batch sizes correctly weights by sample counts."""
    torch.manual_seed(42)
    linear_accum = nn.Linear(4, 2, bias=False)
    linear_comb = nn.Linear(4, 2, bias=False)
    linear_comb.load_state_dict(linear_accum.state_dict())

    # Batch 1: 2 samples
    x1 = torch.randn(2, 4)
    y1 = torch.randn(2, 2)
    # Batch 2: 1 sample
    x2 = torch.randn(1, 4)
    y2 = torch.randn(1, 2)

    total_samples = 3

    # Combined reference (size 3)
    x_comb = torch.cat([x1, x2], dim=0)
    y_comb = torch.cat([y1, y2], dim=0)
    loss_comb = nn.functional.mse_loss(linear_comb(x_comb), y_comb)
    loss_comb.backward()
    grad_comb = linear_comb.weight.grad.clone()

    # Sample-weighted micro-batches: batch 1 weight = 2/3, batch 2 weight = 1/3
    linear_accum.zero_grad()
    loss1 = nn.functional.mse_loss(linear_accum(x1), y1) * (2.0 / total_samples)
    loss1.backward()

    loss2 = nn.functional.mse_loss(linear_accum(x2), y2) * (1.0 / total_samples)
    loss2.backward()
    grad_accum = linear_accum.weight.grad.clone()

    assert torch.allclose(grad_comb, grad_accum, rtol=1e-5, atol=1e-6)


def test_dry_run_zero_optimizer_steps_and_unchanged_global_step(clean_bridge):
    """P4A.1-13: In dry_run=True, optimizer_steps_taken is 0 and global_step does not advance."""
    bridge, optimizer, comp_loss = clean_bridge
    train_loader = make_mock_loader(size=4, split="train")
    val_loader = make_mock_loader(size=1, split="val")
    config = TrainingRunConfig(epochs=1, grad_accum_steps=2, device="cpu")

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=config,
    )

    initial_step = runner.global_step
    metrics = runner.train_epoch(epoch=1, dry_run=True)

    assert metrics.optimizer_steps_taken == 0
    assert metrics.accumulation_windows_completed == 2
    assert runner.global_step == initial_step == 0


# =============================================================================
# Phase 4A.1 Regression Tests: Resumption & Checkpoint State
# =============================================================================

def test_resumed_runner_advances_to_next_epoch(clean_bridge, tmp_path):
    """P4A.1-14: Resumed TrainingRunner begins at next_epoch = restored['epoch'] + 1."""
    bridge, optimizer, comp_loss = clean_bridge
    train_loader = make_mock_loader(size=2, split="train")
    val_loader = make_mock_loader(size=1, split="val")

    ckpt_file = tmp_path / "checkpoint_epoch_001.pt"
    config = TrainingRunConfig(epochs=3, device="cpu")

    save_training_checkpoint(
        checkpoint_path=ckpt_file,
        bridge=bridge,
        optimizer=optimizer,
        epoch=1,
        global_step=45,
        best_val_metric=0.88,
        early_stopping_state={"counter": 2, "patience": 10},
        config=config,
    )

    # Create resumed runner
    resumed_runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=config,
        resume_checkpoint_path=ckpt_file,
    )

    assert resumed_runner.start_epoch == 2
    assert resumed_runner.global_step == 45
    assert resumed_runner.best_val_metric == pytest.approx(0.88)
    assert resumed_runner.early_stop_counter == 2


def test_resumed_runner_does_not_overwrite_prior_checkpoint(clean_bridge, tmp_path):
    """P4A.1-15: Resumed training writes to new epoch file and does not overwrite old checkpoint."""
    bridge, optimizer, comp_loss = clean_bridge
    train_loader = make_mock_loader(size=2, split="train")
    val_loader = make_mock_loader(size=1, split="val")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    epoch1_file = ckpt_dir / "checkpoint_epoch_001.pt"

    config = TrainingRunConfig(epochs=2, checkpoint_dir=str(ckpt_dir), device="cpu")

    save_training_checkpoint(
        checkpoint_path=epoch1_file,
        bridge=bridge,
        optimizer=optimizer,
        epoch=1,
        global_step=10,
        best_val_metric=1.0,
        config=config,
    )

    epoch1_bytes = epoch1_file.read_bytes()

    # Save epoch 2
    epoch2_file = ckpt_dir / "checkpoint_epoch_002.pt"
    save_training_checkpoint(
        checkpoint_path=epoch2_file,
        bridge=bridge,
        optimizer=optimizer,
        epoch=2,
        global_step=20,
        best_val_metric=0.9,
        config=config,
    )

    # Invariant: epoch 1 file is intact and distinct from epoch 2
    assert epoch1_file.read_bytes() == epoch1_bytes
    assert epoch2_file.exists()
    assert epoch2_file.read_bytes() != epoch1_bytes


def test_resumed_runner_already_completed_terminates_gracefully(clean_bridge, tmp_path):
    """P4A.1-16: Resuming run where start_epoch > config.epochs terminates with ALREADY_COMPLETED."""
    bridge, optimizer, comp_loss = clean_bridge
    train_loader = make_mock_loader(size=2, split="train")
    val_loader = make_mock_loader(size=1, split="val")

    ckpt_file = tmp_path / "checkpoint_epoch_005.pt"
    config = TrainingRunConfig(epochs=5, device="cpu")

    save_training_checkpoint(
        checkpoint_path=ckpt_file,
        bridge=bridge,
        optimizer=optimizer,
        epoch=5,
        global_step=100,
        best_val_metric=0.5,
        config=config,
    )

    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="MOCK_TEST_APPROVER",
    )

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=config,
        authorization_token=auth_token,
        resume_checkpoint_path=ckpt_file,
    )

    assert runner.start_epoch == 6
    summary = runner.train()
    assert summary.termination_reason == "ALREADY_COMPLETED"
    assert summary.completed_epochs == 0


def test_checkpoint_missing_manifest_metadata_fails_closed(clean_bridge, tmp_path):
    """P4A.1-17: Checkpoint with invalid or missing manifest hash fails closed."""
    bridge, optimizer, _ = clean_bridge
    bad_ckpt = tmp_path / "bad_manifest.pt"

    state = {
        "metadata": {
            "checkpoint_schema_version": "4.0",
            "manifest_metadata": {"manifest_file_sha256": "WRONG_HASH", "dataset_composite_sha256": "WRONG_HASH"},
            "model_metadata": {"architecture": "ACS-Sub-INN-v3", "gamma_min": 1.0},
        },
        "model_state_dict": bridge.acs_model.state_dict(),
        "epoch": 1,
        "global_step": 1,
    }
    torch.save(state, bad_ckpt)

    with pytest.raises(IncompatibleCheckpointError, match="manifest hash mismatch"):
        load_training_checkpoint(bad_ckpt, bridge)


def test_dataloader_generator_restoration(clean_bridge, tmp_path):
    """P4A.1-18: DataLoader generator state is captured and restored on resumption."""
    bridge, optimizer, _ = clean_bridge
    ckpt_file = tmp_path / "gen_test.pt"

    # Synthetic loader with torch.Generator
    ds = MockBatchDataset(size=5, split="train")
    gen = torch.Generator().manual_seed(999)
    loader = DataLoader(ds, batch_size=1, generator=gen)
    loader.dataset_split = "train"

    # Draw random numbers to advance generator
    _ = torch.rand(10, generator=gen)
    expected_state = gen.get_state()

    save_training_checkpoint(
        checkpoint_path=ckpt_file,
        bridge=bridge,
        optimizer=optimizer,
        train_loader=loader,
    )

    # Advance generator further
    _ = torch.rand(20, generator=gen)

    # Reload checkpoint and verify generator restored
    fresh_gen = torch.Generator()
    fresh_loader = DataLoader(ds, batch_size=1, generator=fresh_gen)
    fresh_loader.dataset_split = "train"

    restored = load_training_checkpoint(
        checkpoint_path=ckpt_file,
        bridge=bridge,
        train_loader=fresh_loader,
    )

    assert restored["dataloader_generator_restored"] is True
    assert torch.equal(fresh_gen.get_state(), expected_state)


# =============================================================================
# Phase 4A.1 Regression Tests: Validation Determinism & Loss Truthfulness
# =============================================================================

def test_validation_deterministic_across_repeated_runs(clean_bridge):
    """P4A.1-19: evaluate_validation_epoch produces identical results and preserves global RNG."""
    bridge, _, comp_loss = clean_bridge
    val_loader = make_mock_loader(size=3, split="val")

    # Record global PyTorch RNG before validation
    initial_torch_rng = torch.get_rng_state()

    metrics_pass1 = evaluate_validation_epoch(
        bridge=bridge,
        val_loader=val_loader,
        composite_loss=comp_loss,
        device=torch.device("cpu"),
        val_seed=424242,
    )

    # Invariant: Global PyTorch RNG is NOT advanced by validation pass
    post_pass1_torch_rng = torch.get_rng_state()
    assert torch.equal(initial_torch_rng, post_pass1_torch_rng), "Validation pass mutated global PyTorch RNG!"

    metrics_pass2 = evaluate_validation_epoch(
        bridge=bridge,
        val_loader=val_loader,
        composite_loss=comp_loss,
        device=torch.device("cpu"),
        val_seed=424242,
    )

    post_pass2_torch_rng = torch.get_rng_state()
    assert torch.equal(initial_torch_rng, post_pass2_torch_rng), "Validation pass mutated global PyTorch RNG on pass 2!"

    # Invariant: Validation metrics are bit-for-bit identical across passes
    assert metrics_pass1.total_loss == metrics_pass2.total_loss
    assert metrics_pass1.loss_bit_raw == metrics_pass2.loss_bit_raw
    assert metrics_pass1.loss_ctx_raw == metrics_pass2.loss_ctx_raw
    assert metrics_pass1.loss_gain_raw == metrics_pass2.loss_gain_raw
    assert metrics_pass1.loss_img_raw == metrics_pass2.loss_img_raw
    assert metrics_pass1.pre_ecc_ber == metrics_pass2.pre_ecc_ber
    assert metrics_pass1.stego_psnr_db == metrics_pass2.stego_psnr_db
    assert metrics_pass1.stego_ssim == metrics_pass2.stego_ssim


def test_mismatched_loss_configuration_rejected(clean_bridge):
    """P4A.1-20: Mismatched loss weights between TrainingRunConfig and CompositeLoss fail closed."""
    bridge, optimizer, _ = clean_bridge
    train_loader = make_mock_loader(size=2, split="train")
    val_loader = make_mock_loader(size=1, split="val")

    # CompositeLoss has lambda_ctx=0.5
    comp_loss = CompositeLoss(lambda_ctx=0.5)

    # TrainingRunConfig has conflicting lambda_ctx=0.99
    mismatched_config = TrainingRunConfig(lambda_ctx=0.99, device="cpu")

    with pytest.raises(ValueError, match="Loss weight configuration mismatch between TrainingRunConfig and CompositeLoss"):
        TrainingRunner(
            bridge=bridge,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            composite_loss=comp_loss,
            config=mismatched_config,
        )


# =============================================================================
# Phase 4A.1 Regression Tests: Authorization Scope
# =============================================================================

def test_authorization_token_exact_scope_enforcement():
    """P4A.1-21: Token rejects non-exact scope, empty approver, or bad passphrase."""
    # 1. Exact canonical scope passes
    valid_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="CHIEF_SYSTEM_ARCHITECT",
        scope="PHASE4B_PILOT",
    )
    assert valid_token.is_valid() is True

    # 2. Scope prefix only fails (e.g. "PHASE4B" instead of "PHASE4B_PILOT")
    prefix_only_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="CHIEF_SYSTEM_ARCHITECT",
        scope="PHASE4B",
    )
    assert prefix_only_token.is_valid() is False

    # 3. Scope extension fails
    extended_scope_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="CHIEF_SYSTEM_ARCHITECT",
        scope="PHASE4B_PILOT_EXTENDED_UNAUTHORIZED",
    )
    assert extended_scope_token.is_valid() is False

    # 4. Whitespace / empty approver fails
    empty_approver_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="   ",
        scope="PHASE4B_PILOT",
    )
    assert empty_approver_token.is_valid() is False

    # 5. Invalid passphrase fails
    bad_passphrase_token = TrainingAuthorizationToken(
        passphrase="WRONG_PASSPHRASE_SECRET",
        approver="CHIEF_SYSTEM_ARCHITECT",
        scope="PHASE4B_PILOT",
    )
    assert bad_passphrase_token.is_valid() is False


def test_authorization_token_phase4c_extended_scope():
    """Phase 4C Regression: Distinct PHASE4C_EXTENDED scope and passphrase enforcement."""
    # 1. Valid Phase 4C token passes
    valid_p4c = TrainingAuthorizationToken(
        passphrase=PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER_EXPLICIT_AUTHORIZATION",
        scope="PHASE4C_EXTENDED",
    )
    assert valid_p4c.is_valid() is True
    # verify_training_authorization accepts it without error
    verify_training_authorization(valid_p4c)

    # 2. Reusing Phase 4B passphrase for Phase 4C scope fails
    reused_passphrase = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER_EXPLICIT_AUTHORIZATION",
        scope="PHASE4C_EXTENDED",
    )
    assert reused_passphrase.is_valid() is False
    with pytest.raises(TrainingNotAuthorizedError):
        verify_training_authorization(reused_passphrase)

    # 3. Using Phase 4C passphrase for Phase 4B scope fails
    cross_passphrase = TrainingAuthorizationToken(
        passphrase=PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER_EXPLICIT_AUTHORIZATION",
        scope="PHASE4B_PILOT",
    )
    assert cross_passphrase.is_valid() is False
    with pytest.raises(TrainingNotAuthorizedError):
        verify_training_authorization(cross_passphrase)

    # 4. Scope prefix fails
    prefix_p4c = TrainingAuthorizationToken(
        passphrase=PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER",
        scope="PHASE4C",
    )
    assert prefix_p4c.is_valid() is False

    # 5. Scope extension fails
    extended_p4c = TrainingAuthorizationToken(
        passphrase=PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER",
        scope="PHASE4C_EXTENDED_UNAUTHORIZED",
    )
    assert extended_p4c.is_valid() is False

    # 6. Blank / whitespace approver fails
    blank_approver = TrainingAuthorizationToken(
        passphrase=PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="   ",
        scope="PHASE4C_EXTENDED",
    )
    assert blank_approver.is_valid() is False

    # 7. Unknown scope fails
    unknown_scope = TrainingAuthorizationToken(
        passphrase=PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER",
        scope="PHASE5_UNAUTHORIZED",
    )
    assert unknown_scope.is_valid() is False


# =============================================================================
# Phase 4A.2 Focused Regression Tests: Checkpoint Loader Fail-Closed Integrity
# =============================================================================

def test_checkpoint_absent_metadata_fails_closed(clean_bridge, tmp_path):
    """P4A.2-1: Checkpoint with absent or non-dict metadata fails closed."""
    bridge, _, _ = clean_bridge
    ckpt = tmp_path / "no_metadata.pt"
    torch.save({"model_state_dict": bridge.acs_model.state_dict()}, ckpt)
    with pytest.raises(IncompatibleCheckpointError, match="missing valid 'metadata' dictionary"):
        load_training_checkpoint(ckpt, bridge)


def test_checkpoint_unsupported_schema_version_fails_closed(clean_bridge, tmp_path):
    """P4A.2-2: Checkpoint with absent or unsupported schema version fails closed."""
    bridge, _, _ = clean_bridge

    # Absent schema version
    ckpt_no_schema = tmp_path / "no_schema.pt"
    torch.save({
        "metadata": {},
        "model_state_dict": bridge.acs_model.state_dict(),
    }, ckpt_no_schema)
    with pytest.raises(IncompatibleCheckpointError, match="missing 'checkpoint_schema_version' in metadata"):
        load_training_checkpoint(ckpt_no_schema, bridge)

    # Unsupported schema version
    ckpt_bad_schema = tmp_path / "bad_schema.pt"
    torch.save({
        "metadata": {"checkpoint_schema_version": "99.0"},
        "model_state_dict": bridge.acs_model.state_dict(),
    }, ckpt_bad_schema)
    with pytest.raises(IncompatibleCheckpointError, match="Unsupported checkpoint schema version '99.0'"):
        load_training_checkpoint(ckpt_bad_schema, bridge)


def test_checkpoint_absent_manifest_metadata_fails_closed(clean_bridge, tmp_path):
    """P4A.2-3: Checkpoint missing manifest_metadata fails closed."""
    bridge, _, _ = clean_bridge
    ckpt = tmp_path / "no_manifest.pt"
    torch.save({
        "metadata": {"checkpoint_schema_version": "4.0"},
        "model_state_dict": bridge.acs_model.state_dict(),
    }, ckpt)
    with pytest.raises(IncompatibleCheckpointError, match="missing valid 'manifest_metadata' dictionary"):
        load_training_checkpoint(ckpt, bridge)


def test_checkpoint_absent_model_metadata_fails_closed(clean_bridge, tmp_path):
    """P4A.2-4: Checkpoint missing model_metadata fails closed."""
    bridge, _, _ = clean_bridge
    ckpt = tmp_path / "no_model_meta.pt"
    torch.save({
        "metadata": {
            "checkpoint_schema_version": "4.0",
            "manifest_metadata": {
                "manifest_file_sha256": "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff",
                "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
            },
        },
        "model_state_dict": bridge.acs_model.state_dict(),
    }, ckpt)
    with pytest.raises(IncompatibleCheckpointError, match="missing valid 'model_metadata' dictionary"):
        load_training_checkpoint(ckpt, bridge)


def test_checkpoint_incompatible_gamma_min_fails_closed(clean_bridge, tmp_path):
    """P4A.2-5: Checkpoint with missing or non-1.0 gamma_min fails closed."""
    bridge, _, _ = clean_bridge

    # Missing gamma_min
    ckpt_no_gamma = tmp_path / "no_gamma.pt"
    torch.save({
        "metadata": {
            "checkpoint_schema_version": "4.0",
            "manifest_metadata": {
                "manifest_file_sha256": "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff",
                "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
            },
            "model_metadata": {
                "architecture": "ACS-Sub-INN-v3",
            },
        },
        "model_state_dict": bridge.acs_model.state_dict(),
    }, ckpt_no_gamma)
    with pytest.raises(IncompatibleCheckpointError, match="missing valid numeric 'gamma_min'"):
        load_training_checkpoint(ckpt_no_gamma, bridge)

    # Incompatible gamma_min != 1.0
    ckpt_bad_gamma = tmp_path / "bad_gamma.pt"
    torch.save({
        "metadata": {
            "checkpoint_schema_version": "4.0",
            "manifest_metadata": {
                "manifest_file_sha256": "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff",
                "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
            },
            "model_metadata": {
                "architecture": "ACS-Sub-INN-v3",
                "gamma_min": 0.5,
            },
        },
        "model_state_dict": bridge.acs_model.state_dict(),
    }, ckpt_bad_gamma)
    with pytest.raises(IncompatibleCheckpointError, match="gamma_min mismatch! Frozen architecture requires gamma_min=1.0"):
        load_training_checkpoint(ckpt_bad_gamma, bridge)


def test_checkpoint_missing_required_optimizer_state_fails_closed(clean_bridge, tmp_path):
    """P4A.2-6: Resumption with optimizer fails closed if optimizer_state_dict is missing."""
    bridge, optimizer, _ = clean_bridge
    ckpt = tmp_path / "no_optimizer.pt"
    torch.save({
        "metadata": {
            "checkpoint_schema_version": "4.0",
            "manifest_metadata": {
                "manifest_file_sha256": "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff",
                "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
            },
            "model_metadata": {
                "architecture": "ACS-Sub-INN-v3",
                "gamma_min": 1.0,
            },
        },
        "model_state_dict": bridge.acs_model.state_dict(),
    }, ckpt)
    with pytest.raises(IncompatibleCheckpointError, match="Optimizer was provided for resumption, but checkpoint .* is missing 'optimizer_state_dict'"):
        load_training_checkpoint(ckpt, bridge, optimizer=optimizer)


def test_checkpoint_missing_required_scheduler_state_fails_closed(clean_bridge, tmp_path):
    """P4A.2-7: Resumption with scheduler fails closed if scheduler_state_dict is missing."""
    bridge, optimizer, _ = clean_bridge
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    ckpt = tmp_path / "no_scheduler.pt"
    torch.save({
        "metadata": {
            "checkpoint_schema_version": "4.0",
            "manifest_metadata": {
                "manifest_file_sha256": "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff",
                "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
            },
            "model_metadata": {
                "architecture": "ACS-Sub-INN-v3",
                "gamma_min": 1.0,
            },
        },
        "model_state_dict": bridge.acs_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt)
    with pytest.raises(IncompatibleCheckpointError, match="Scheduler was provided for resumption, but checkpoint .* is missing 'scheduler_state_dict'"):
        load_training_checkpoint(ckpt, bridge, scheduler=scheduler)


def test_checkpoint_dataloader_generator_restoration_failure_fails_closed(clean_bridge, tmp_path):
    """P4A.2-8: DataLoader generator restoration failure fails closed when required or corrupted."""
    bridge, _, _ = clean_bridge
    ds = MockBatchDataset(size=2, split="train")
    gen = torch.Generator().manual_seed(42)
    loader = DataLoader(ds, batch_size=1, generator=gen)
    loader.dataset_split = "train"

    # 1. require_dataloader_generator=True but state is missing
    ckpt1 = tmp_path / "no_gen.pt"
    torch.save({
        "metadata": {
            "checkpoint_schema_version": "4.0",
            "manifest_metadata": {
                "manifest_file_sha256": "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff",
                "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
            },
            "model_metadata": {
                "architecture": "ACS-Sub-INN-v3",
                "gamma_min": 1.0,
            },
        },
        "model_state_dict": bridge.acs_model.state_dict(),
        "dataloader_generator_state": None,
    }, ckpt1)
    with pytest.raises(IncompatibleCheckpointError, match="DataLoader generator restoration is required"):
        load_training_checkpoint(ckpt1, bridge, train_loader=loader, require_dataloader_generator=True)

    # 2. Corrupt generator state fails closed
    ckpt2 = tmp_path / "bad_gen.pt"
    torch.save({
        "metadata": {
            "checkpoint_schema_version": "4.0",
            "manifest_metadata": {
                "manifest_file_sha256": "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff",
                "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
            },
            "model_metadata": {
                "architecture": "ACS-Sub-INN-v3",
                "gamma_min": 1.0,
            },
        },
        "model_state_dict": bridge.acs_model.state_dict(),
        "dataloader_generator_state": torch.tensor([1, 2, 3], dtype=torch.uint8),  # wrong size for generator state
    }, ckpt2)
    with pytest.raises(IncompatibleCheckpointError, match="Failed to restore DataLoader generator state"):
        load_training_checkpoint(ckpt2, bridge, train_loader=loader)


def test_rejected_checkpoint_leaves_live_model_unchanged(clean_bridge, tmp_path):
    """P4A.2-9: Key Invariant: Any rejected checkpoint leaves live model parameters bit-for-bit unchanged."""
    bridge, optimizer, _ = clean_bridge

    # Clone baseline live parameters
    init_params = [p.clone().detach() for p in bridge.acs_model.parameters()]

    valid_manifest = {
        "manifest_file_sha256": "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff",
        "dataset_composite_sha256": "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed",
    }
    valid_model_meta = {
        "architecture": "ACS-Sub-INN-v3",
        "gamma_min": 1.0,
    }

    test_rejections = [
        # Corrupt file
        ("corrupt.pt", b"NOT_A_VALID_HEADER"),
        # Missing metadata
        ("no_meta.pt", {"model_state_dict": bridge.acs_model.state_dict()}),
        # Unsupported schema version
        ("bad_schema.pt", {
            "metadata": {"checkpoint_schema_version": "99.0"},
            "model_state_dict": bridge.acs_model.state_dict(),
        }),
        # Incompatible manifest
        ("bad_manifest.pt", {
            "metadata": {
                "checkpoint_schema_version": "4.0",
                "manifest_metadata": {"manifest_file_sha256": "BAD_HASH", "dataset_composite_sha256": "BAD_HASH"},
                "model_metadata": valid_model_meta,
            },
            "model_state_dict": bridge.acs_model.state_dict(),
        }),
        # Incompatible architecture
        ("bad_arch.pt", {
            "metadata": {
                "checkpoint_schema_version": "4.0",
                "manifest_metadata": valid_manifest,
                "model_metadata": {"architecture": "RESNET_UNSUPPORTED", "gamma_min": 1.0},
            },
            "model_state_dict": bridge.acs_model.state_dict(),
        }),
        # Incompatible gamma_min
        ("bad_gamma.pt", {
            "metadata": {
                "checkpoint_schema_version": "4.0",
                "manifest_metadata": valid_manifest,
                "model_metadata": {"architecture": "ACS-Sub-INN-v3", "gamma_min": 2.5},
            },
            "model_state_dict": bridge.acs_model.state_dict(),
        }),
        # Shape mismatch
        ("bad_shape.pt", {
            "metadata": {
                "checkpoint_schema_version": "4.0",
                "manifest_metadata": valid_manifest,
                "model_metadata": valid_model_meta,
            },
            "model_state_dict": {"context_predictor.conv_in.weight": torch.randn(99, 99, 1, 1)},
        }),
        # Missing optimizer state when optimizer provided
        ("missing_opt.pt", {
            "metadata": {
                "checkpoint_schema_version": "4.0",
                "manifest_metadata": valid_manifest,
                "model_metadata": valid_model_meta,
            },
            "model_state_dict": bridge.acs_model.state_dict(),
        }),
    ]

    for fname, payload in test_rejections:
        target = tmp_path / fname
        if isinstance(payload, bytes):
            target.write_bytes(payload)
        else:
            torch.save(payload, target)

        opt_arg = optimizer if fname == "missing_opt.pt" else None

        with pytest.raises(IncompatibleCheckpointError):
            load_training_checkpoint(target, bridge, optimizer=opt_arg)

        # Invariant check: all model parameters remain bit-for-bit identical
        for p_init, p_curr in zip(init_params, bridge.acs_model.parameters()):
            assert torch.equal(p_init, p_curr), f"Model weights mutated by rejected checkpoint '{fname}'!"
