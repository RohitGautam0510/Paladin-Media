"""Phase 4A Automated Training Infrastructure, Telemetry, and Authorization Gate.

This module implements the production-grade training engine for ACS-Sub-INN v3:
1. TrainingAuthorizationGate: Fail-closed authorization gate preventing any optimizer update
   or model training without explicit, documented passphrase opt-in.
2. TrainingRunConfig: Strongly-typed, validated hyperparameter and runtime configuration.
3. Checkpoint Management: Atomic epoch-boundary checkpoint serialization with complete
   resumption state (model, optimizer, scheduler, RNGs, epoch/step, manifest metadata).
4. TrainingTelemetryLogger: JSONL telemetry logger capturing raw and weighted loss components,
   gradient norms (pre/post-clip), and validation metrics per epoch.
5. evaluate_validation_epoch: Robust, inference-only validation evaluator fail-closed against test split.
6. TrainingRunner: Epoch-based training orchestrator with gradient accumulation, finite-loss checks,
   gradient clipping, LR scheduler stepping, and early stopping.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import (
    EXPECTED_COMPOSITE_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    normalize_split_name,
)
from src.neural.losses import (
    ClippingDiagnosticReport,
    CompositeLoss,
    LossBreakdown,
    evaluate_clipping_diagnostic,
)
from src.neural.spatial_bridge import NeuralSpatialBridge


# =============================================================================
# 1. Custom Exceptions
# =============================================================================

class TrainingInfrastructureError(Exception):
    """Base exception for Phase 4 training infrastructure."""
    pass


class TrainingNotAuthorizedError(TrainingInfrastructureError):
    """Raised when real training or optimizer updates are attempted without authorization."""
    pass


class NonFiniteLossError(TrainingInfrastructureError):
    """Raised when a computed loss evaluates to NaN or Infinity."""
    pass


class NonFiniteGradientError(TrainingInfrastructureError):
    """Raised when parameter gradients evaluate to NaN or Infinity."""
    pass


class IncompatibleCheckpointError(TrainingInfrastructureError):
    """Raised when a checkpoint fails schema, manifest, or parameter shape verification."""
    pass


class DatasetSplitBoundaryError(TrainingInfrastructureError, ValueError):
    """Raised when a non-train split or held-out test data enters an optimization step."""
    pass


CANONICAL_SPLIT_RANGES: Dict[str, Tuple[int, int]] = {
    "train": (1, 600),
    "val": (601, 700),
    "test": (701, 800),
}


def validate_batch_partition(batch: Dict[str, Any], expected_split: str) -> None:
    """Validates that a mini-batch strictly adheres to the canonical partition boundary.

    Enforces fail-closed rules:
    1. expected_split must be normalized ('train' or 'val').
    2. batch['split'] must be present, non-empty, and every entry must match expected_split.
    3. batch['image_id'] must be present, non-empty, and contain only valid integer IDs (no booleans).
    4. Each image_id strictly belongs to the expected_split canonical range (train: 1-600, val: 601-700).
    5. Any image_id in held-out test range (701-800) triggers immediate fail-closed DatasetSplitBoundaryError.
    6. Any cross-partition contamination, mislabeling, or mixed batch triggers DatasetSplitBoundaryError.

    Args:
        batch: Mini-batch dictionary containing 'split', 'image_id', and 'carrier'.
        expected_split: Expected partition ('train' or 'val').

    Raises:
        DatasetSplitBoundaryError: If any boundary, split, or image identity check fails.
    """
    if not isinstance(batch, dict):
        raise DatasetSplitBoundaryError(f"Batch must be a dictionary, got {type(batch).__name__}")

    norm_expected = normalize_split_name(expected_split)

    # 1. Validate 'split' field
    if "split" not in batch or batch["split"] is None:
        raise DatasetSplitBoundaryError("SECURITY/INTEGRITY VIOLATION: Batch missing required 'split' metadata.")

    raw_splits = batch["split"]
    if isinstance(raw_splits, str):
        split_list = [raw_splits]
    elif isinstance(raw_splits, (list, tuple)):
        split_list = list(raw_splits)
    else:
        raise DatasetSplitBoundaryError(f"Invalid batch 'split' field type: {type(raw_splits).__name__}")

    if len(split_list) == 0:
        raise DatasetSplitBoundaryError("SECURITY/INTEGRITY VIOLATION: Batch 'split' metadata is empty.")

    for s in split_list:
        if not isinstance(s, str) or not s.strip():
            raise DatasetSplitBoundaryError(f"Invalid split entry in batch: {s!r}")
        try:
            norm_s = normalize_split_name(s)
        except ValueError as err:
            raise DatasetSplitBoundaryError(f"Invalid split name in batch '{s}': {err}") from err

        if norm_s == "test":
            raise DatasetSplitBoundaryError("SECURITY VIOLATION: Held-out 'test' partition detected in batch!")
        if norm_s != norm_expected:
            raise DatasetSplitBoundaryError(
                f"BOUNDARY VIOLATION: Batch split mismatch! Expected '{norm_expected}', got '{norm_s}'."
            )

    # 2. Validate 'image_id' field
    if "image_id" not in batch or batch["image_id"] is None:
        raise DatasetSplitBoundaryError("SECURITY/INTEGRITY VIOLATION: Batch missing required 'image_id' metadata.")

    raw_ids = batch["image_id"]
    if isinstance(raw_ids, torch.Tensor):
        id_list = raw_ids.view(-1).tolist()
    elif isinstance(raw_ids, (list, tuple)):
        id_list = list(raw_ids)
    elif isinstance(raw_ids, (int, np.integer)) and not isinstance(raw_ids, bool):
        id_list = [int(raw_ids)]
    else:
        raise DatasetSplitBoundaryError(f"Invalid batch 'image_id' field type: {type(raw_ids).__name__}")

    if len(id_list) == 0:
        raise DatasetSplitBoundaryError("SECURITY/INTEGRITY VIOLATION: Batch 'image_id' metadata is empty.")

    min_id, max_id = CANONICAL_SPLIT_RANGES[norm_expected]
    test_min, test_max = CANONICAL_SPLIT_RANGES["test"]

    for img_id in id_list:
        if isinstance(img_id, bool) or not isinstance(img_id, (int, np.integer)):
            raise DatasetSplitBoundaryError(f"Image ID must be a non-boolean integer, got {img_id!r}")

        int_id = int(img_id)

        # Held-out test boundary guard
        if test_min <= int_id <= test_max:
            raise DatasetSplitBoundaryError(
                f"SECURITY VIOLATION: Image ID {int_id} belongs to held-out test split ({test_min}-{test_max})!"
            )

        # Canonical split partition guard
        if not (min_id <= int_id <= max_id):
            raise DatasetSplitBoundaryError(
                f"BOUNDARY VIOLATION: Image ID {int_id} does not belong to canonical {norm_expected} "
                f"partition ({min_id}-{max_id})!"
            )


# =============================================================================
# 2. Workstream A — Explicit Training Authorization Gate
# =============================================================================

PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE = "PHASE4B_PILOT_OPT_IN_AUTHORIZED_2026"
PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE = "PHASE4C_EXTENDED_OPT_IN_AUTHORIZED_2026"

VALID_TRAINING_SCOPES: Dict[str, str] = {
    "PHASE4B_PILOT": PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
    "PHASE4C_EXTENDED": PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
}


@dataclass(frozen=True)
class TrainingAuthorizationToken:
    """Explicit authorization token required for live training execution.

    In accordance with Phase 4 governance, real training is disabled by default.
    To authorize training, the user or governing script must construct a valid token:
        token = TrainingAuthorizationToken(
            passphrase=PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
            approver="USER",
            scope="PHASE4C_EXTENDED"
        )
    """

    passphrase: str
    approver: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    scope: str = "PHASE4B_PILOT"

    def is_valid(self) -> bool:
        """Returns True only if passphrase matches the canonical secret for the exact scope, and approver is non-empty."""
        if not self.approver or not self.approver.strip():
            return False
        expected_passphrase = VALID_TRAINING_SCOPES.get(self.scope)
        if expected_passphrase is None:
            return False
        return self.passphrase == expected_passphrase


def verify_training_authorization(token: Optional[TrainingAuthorizationToken]) -> None:
    """Validates authorization token, raising TrainingNotAuthorizedError if missing or invalid.

    Raises:
        TrainingNotAuthorizedError: If token is None or fails validation.
    """
    if token is None or not token.is_valid():
        raise TrainingNotAuthorizedError(
            "TRAINING REJECTED: Model training and optimizer updates are NOT authorized. "
            "Real training requires an explicit, valid TrainingAuthorizationToken with the approved "
            "canonical passphrase and an authorized scope (e.g. 'PHASE4B_PILOT' or 'PHASE4C_EXTENDED')."
        )


# =============================================================================
# 3. Workstream B — Configuration & Telemetry
# =============================================================================

@dataclass
class TrainingRunConfig:
    """Strongly-typed, validated hyperparameter and runtime configuration."""

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    deterministic: bool = True

    # Epoch and batch sizing
    epochs: int = 5
    batch_size: int = 1
    crop_size: int = 512
    grad_accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = False

    # Optimization parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    clip_grad_norm: Optional[float] = 1.0

    # Candidate loss weights (PROPOSED_UNVALIDATED)
    lambda_bit: float = 1.0
    lambda_ctx: float = 0.5
    lambda_gain: float = 0.01
    lambda_img: float = 0.1

    # Checkpoint and telemetry
    checkpoint_dir: str = "checkpoints/phase4"
    save_every_n_epochs: int = 1
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    telemetry_path: str = "docs/experiments/training_telemetry.jsonl"
    run_id: str = "run_phase4"

    def validate(self) -> None:
        """Validates configuration parameters, failing closed on invalid bounds."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be strictly positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be strictly positive, got {self.batch_size}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be strictly positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0.0:
            raise ValueError(f"clip_grad_norm must be positive or None, got {self.clip_grad_norm}")
        if self.crop_size <= 0 or self.crop_size % 2 != 0:
            raise ValueError(f"crop_size must be positive even integer, got {self.crop_size}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(val: Any) -> Any:
    """Sanitizes floats for standard JSON compliance, converting NaN/Inf to strings."""
    if isinstance(val, (float, np.floating)):
        if math.isnan(val):
            return "NaN"
        elif math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return float(val)
    return val


def sanitize_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively converts NaN and Infinity floats to safe string representations."""
    clean: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            clean[k] = sanitize_dict_for_json(v)
        elif isinstance(v, list):
            clean[k] = [_safe_float(x) for x in v]
        else:
            clean[k] = _safe_float(v)
    return clean


class TrainingTelemetryLogger:
    """Structured JSONL telemetry logger capturing detailed per-epoch records."""

    def __init__(self, log_path: Union[str, Path]) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_epoch(self, record: Dict[str, Any]) -> None:
        """Appends a sanitized JSONL record to the telemetry file."""
        clean_record = sanitize_dict_for_json(record)
        clean_record["log_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps(clean_record) + "\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


# =============================================================================
# 4. Workstream C — Validation Engine
# =============================================================================

@dataclass(frozen=True)
class ValidationMetrics:
    """Aggregated validation metrics over an evaluation epoch."""

    total_loss: float
    loss_bit_raw: float
    loss_bit_weighted: float
    loss_ctx_raw: float
    loss_ctx_weighted: float
    loss_gain_raw: float
    loss_gain_weighted: float
    loss_img_raw: float
    loss_img_weighted: float
    pre_ecc_ber: float
    stego_psnr_db: float
    stego_ssim: float
    clip_fraction: float
    num_batches: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_validation_epoch(
    bridge: NeuralSpatialBridge,
    val_loader: DataLoader,
    composite_loss: CompositeLoss,
    device: Optional[torch.device] = None,
    val_seed: int = 99999,
) -> ValidationMetrics:
    """Executes an inference-only validation evaluation across the validation partition.

    CRITICAL BOUNDARY ENFORCEMENT:
    - Never invokes optimizer updates (strictly torch.no_grad).
    - Enforces that val_loader has explicit dataset_split == 'val' (fails closed on missing/mislabeled split).
    - Rejects any batch containing images from held-out 'test' or non-val partition.
    - Generates validation payload bits with a dedicated CPU generator without mutating global PyTorch RNG.
    - Rejects non-finite loss metrics.

    Args:
        bridge: NeuralSpatialBridge instance.
        val_loader: DataLoader yielding validation batches.
        composite_loss: CompositeLoss module.
        device: Execution device.
        val_seed: Seed for deterministic validation payload generation.

    Returns:
        ValidationMetrics containing aggregated metrics.

    Raises:
        DatasetSplitBoundaryError: If non-val or test split data is detected.
        NonFiniteLossError: If loss evaluates to NaN or Infinity.
    """
    bridge.eval()

    # Guard 1: Verify DataLoader metadata (fail closed if missing or not 'val')
    loader_split = getattr(val_loader, "dataset_split", None)
    if loader_split is None or loader_split != "val":
        raise DatasetSplitBoundaryError(
            f"SECURITY/INTEGRITY VIOLATION: evaluate_validation_epoch requires explicit dataset_split='val', "
            f"got '{loader_split}'."
        )

    total_loss_accum = 0.0
    loss_bit_raw_accum = 0.0
    loss_bit_w_accum = 0.0
    loss_ctx_raw_accum = 0.0
    loss_ctx_w_accum = 0.0
    loss_gain_raw_accum = 0.0
    loss_gain_w_accum = 0.0
    loss_img_raw_accum = 0.0
    loss_img_w_accum = 0.0
    ber_accum = 0.0
    psnr_accum = 0.0
    ssim_accum = 0.0
    clip_frac_accum = 0.0
    batch_count = 0

    if device is None:
        device = next(bridge.parameters()).device

    saved_cpu_rng = torch.get_rng_state()
    saved_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        with torch.no_grad():
            for batch in val_loader:
                # Guard 2 & 3: Fail-closed batch partition & image ID validation
                validate_batch_partition(batch, expected_split="val")

                carrier = batch["carrier"].to(device)
                if carrier.ndim == 3:
                    carrier = carrier.unsqueeze(0)

                b, c, h, w = carrier.shape
                sub_h, sub_w = h // 2, w // 2

                # Deterministic validation payload generation using dedicated CPU generator
                # Global PyTorch RNG is NOT advanced!
                raw_id = batch["image_id"]
                if isinstance(raw_id, torch.Tensor):
                    first_img_id = int(raw_id.view(-1)[0].item())
                elif isinstance(raw_id, (list, tuple)):
                    first_img_id = int(raw_id[0])
                else:
                    first_img_id = int(raw_id)

                val_gen = torch.Generator(device="cpu").manual_seed((val_seed + first_img_id * 10007) & 0xFFFFFFFF)
                payload_bits = torch.randint(0, 2, (b, 1, sub_h, sub_w), generator=val_gen, dtype=torch.float32).to(device)
                payload_bipolar = ACSSubINN.bipolar_encode(payload_bits)
                mask = torch.ones(b, 1, sub_h, sub_w, device=device, dtype=torch.float32)

                emb_res = bridge.embed_spatial(
                    carrier=carrier,
                    payload_bipolar=payload_bipolar,
                    mask=mask,
                    differentiable=False,
                )
                ext_res = bridge.extract_spatial(
                    stego_carrier=emb_res.stego_image,
                    mask=mask,
                )

                hh_drop_orig = emb_res.subbands_original[3][:, 0:1, :, :]
                predicted_context = emb_res.acs_embedding_result.context
                gamma = emb_res.acs_embedding_result.gamma

                breakdown: LossBreakdown = composite_loss(
                    delta_rec=ext_res.delta_rec,
                    payload_bipolar=payload_bipolar,
                    predicted_context=predicted_context,
                    target_hh_drop=hh_drop_orig,
                    gamma=gamma,
                    stego_spatial=emb_res.stego_image,
                    cover_spatial=carrier,
                    mask=mask,
                )

                t_loss = float(breakdown.total_loss.item())
                if not math.isfinite(t_loss):
                    raise NonFiniteLossError(f"Validation loss non-finite: {t_loss}")

                total_loss_accum += t_loss
                loss_bit_raw_accum += float(breakdown.loss_bit.item())
                loss_bit_w_accum += float(composite_loss.lambda_bit * breakdown.loss_bit.item())
                loss_ctx_raw_accum += float(breakdown.loss_ctx.item())
                loss_ctx_w_accum += float(composite_loss.lambda_ctx * breakdown.loss_ctx.item())
                loss_gain_raw_accum += float(breakdown.loss_gain.item())
                loss_gain_w_accum += float(composite_loss.lambda_gain * breakdown.loss_gain.item())
                loss_img_raw_accum += float(breakdown.loss_img.item())
                loss_img_w_accum += float(composite_loss.lambda_img * breakdown.loss_img.item())

                # BER, PSNR, SSIM
                active = (mask == 1.0)
                num_act = int(active.sum().item())
                if num_act > 0:
                    rec_bits = ext_res.extracted_binary[active]
                    orig_bits = payload_bits[active]
                    ber = float((rec_bits != orig_bits).sum().item()) / float(num_act)
                else:
                    ber = 0.0
                ber_accum += ber

                c_np = carrier[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                s_np = emb_res.stego_image[0].clamp(0, 255).round().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                psnr_accum += calculate_psnr(c_np, s_np)
                ssim_accum += calculate_ssim(c_np, s_np)

                clip_diag = evaluate_clipping_diagnostic(
                    stego_raw_float=emb_res.stego_raw_float,
                    stego_quantized=emb_res.stego_image,
                )
                clip_frac_accum += clip_diag.clip_fraction

                batch_count += 1
    finally:
        # Guarantee that evaluation passes leave the global PyTorch RNG state unmodified
        torch.set_rng_state(saved_cpu_rng)
        if saved_cuda_rng is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(saved_cuda_rng)
            except Exception:
                pass

    if batch_count == 0:
        raise ValueError("Validation DataLoader yielded zero batches!")

    return ValidationMetrics(
        total_loss=total_loss_accum / batch_count,
        loss_bit_raw=loss_bit_raw_accum / batch_count,
        loss_bit_weighted=loss_bit_w_accum / batch_count,
        loss_ctx_raw=loss_ctx_raw_accum / batch_count,
        loss_ctx_weighted=loss_ctx_w_accum / batch_count,
        loss_gain_raw=loss_gain_raw_accum / batch_count,
        loss_gain_weighted=loss_gain_w_accum / batch_count,
        loss_img_raw=loss_img_raw_accum / batch_count,
        loss_img_weighted=loss_img_w_accum / batch_count,
        pre_ecc_ber=ber_accum / batch_count,
        stego_psnr_db=psnr_accum / batch_count,
        stego_ssim=ssim_accum / batch_count,
        clip_fraction=clip_frac_accum / batch_count,
        num_batches=batch_count,
    )


# =============================================================================
# 5. Workstream D — Checkpoint Resumption & Complete Epoch-Boundary State
# =============================================================================

CHECKPOINT_SCHEMA_VERSION = "4.0"


def save_training_checkpoint(
    checkpoint_path: Union[str, Path],
    bridge: NeuralSpatialBridge,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    best_val_metric: float = float("inf"),
    early_stopping_state: Optional[Dict[str, Any]] = None,
    config: Optional[TrainingRunConfig] = None,
    is_best: bool = False,
    authorization_token: Optional[TrainingAuthorizationToken] = None,
    atomic: bool = True,
    train_loader: Optional[DataLoader] = None,
) -> Path:
    """Atomically saves an epoch-boundary training checkpoint with complete resumption state.

    State persisted:
    - Model state_dict (ContextPredictorCNN + AdaptiveGainCNN).
    - Optimizer state_dict (AdamW momentum and variance buffers).
    - LR Scheduler state_dict (if provided).
    - Completed epoch, next_epoch, and global step counters.
    - Best validation metric and early-stopping state.
    - Multi-library RNG states: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    - DataLoader generator state if present on train_loader.
    - Configuration and candidate loss weights.
    - Dataset manifest verification hashes.
    - Governance metadata.
    """
    target_path = Path(checkpoint_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    is_authorized = (authorization_token is not None and authorization_token.is_valid())

    # Multi-library RNG capture
    rng_states = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    # DataLoader generator state if supported
    dl_gen_state = None
    if train_loader is not None and getattr(train_loader, "generator", None) is not None:
        try:
            dl_gen_state = train_loader.generator.get_state()
        except Exception:
            dl_gen_state = None

    metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": "PRODUCTION_EPOCH_CHECKPOINT" if is_authorized else "UNAUTHORIZED_TRAINING_CHECKPOINT",
        "training_authorized": is_authorized,
        "is_best": is_best,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": config.run_id if config else "unknown_run",
        "software_environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        },
        "config": config.to_dict() if config else {},
        "loss_weights": {
            "lambda_bit": config.lambda_bit if config else 1.0,
            "lambda_ctx": config.lambda_ctx if config else 0.5,
            "lambda_gain": config.lambda_gain if config else 0.01,
            "lambda_img": config.lambda_img if config else 0.1,
        },
        "manifest_metadata": {
            "manifest_file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
            "dataset_composite_sha256": EXPECTED_COMPOSITE_SHA256,
            "dataset_split": "train",
        },
        "model_metadata": {
            "architecture": "ACS-Sub-INN-v3",
            "gamma_min": bridge.acs_model.gamma_min,
        },
    }

    checkpoint = {
        "metadata": metadata,
        "model_state_dict": bridge.acs_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "global_step": global_step,
        "best_val_metric": best_val_metric,
        "early_stopping_state": early_stopping_state or {},
        "rng_states": rng_states,
        "dataloader_generator_state": dl_gen_state,
    }

    if atomic:
        tmp_path = target_path.with_suffix(target_path.suffix + f".tmp_{os.getpid()}")
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return target_path
    else:
        torch.save(checkpoint, target_path)
        return target_path


def load_training_checkpoint(
    checkpoint_path: Union[str, Path],
    bridge: NeuralSpatialBridge,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    restore_rng: bool = True,
    map_location: str = "cpu",
    train_loader: Optional[DataLoader] = None,
    require_dataloader_generator: bool = False,
) -> Dict[str, Any]:
    """Loads a Phase 4 epoch-boundary checkpoint, validating integrity before applying states.

    Validations enforced:
    - Target file presence and valid Torch pickle structure.
    - Checkpoint metadata must exist, be a structurally valid dict, and have exact schema version '4.0'.
    - manifest_metadata must exist and match canonical file and composite SHA-256 digests.
    - model_metadata must exist, match architecture 'ACS-Sub-INN-v3', and match frozen gamma_min=1.0.
    - Parameter keys, names, and tensor shapes match bridge.acs_model exactly.
    - If optimizer or scheduler is provided, checkpoint must contain corresponding valid state_dict.
    - If DataLoader generator restoration is required, fails closed if generator state is missing or invalid.
    - ALL structural and compatibility checks are validated BEFORE mutating live model, optimizer,
      scheduler, RNG, or DataLoader generator state.

    Returns:
        Dictionary containing restored metadata, counters, and early-stopping state.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        IncompatibleCheckpointError: If architecture, shapes, gamma_min, schema, or manifest hashes mismatch or are missing.
    """
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise IncompatibleCheckpointError(f"Corrupt or unreadable checkpoint at {path}: {e}") from e

    if not isinstance(checkpoint, dict):
        raise IncompatibleCheckpointError(f"Malformed checkpoint at {path}: root object is not a dict.")

    # 1. Require checkpoint metadata to exist and be structurally valid dict
    if "metadata" not in checkpoint or not isinstance(checkpoint["metadata"], dict):
        raise IncompatibleCheckpointError(f"Checkpoint at {path} missing valid 'metadata' dictionary.")

    metadata = checkpoint["metadata"]

    # 2. Require checkpoint_schema_version to equal exact supported CHECKPOINT_SCHEMA_VERSION ("4.0")
    schema_ver = metadata.get("checkpoint_schema_version")
    if schema_ver is None:
        raise IncompatibleCheckpointError(f"Checkpoint at {path} missing 'checkpoint_schema_version' in metadata.")
    if schema_ver != CHECKPOINT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError(
            f"Unsupported checkpoint schema version '{schema_ver}'. "
            f"Exact supported version is '{CHECKPOINT_SCHEMA_VERSION}'."
        )

    # 3. Require manifest_metadata to be present and structurally valid
    if "manifest_metadata" not in metadata or not isinstance(metadata["manifest_metadata"], dict):
        raise IncompatibleCheckpointError(f"Checkpoint at {path} missing valid 'manifest_metadata' dictionary.")

    m_meta = metadata["manifest_metadata"]
    f_hash = m_meta.get("manifest_file_sha256")
    c_hash = m_meta.get("dataset_composite_sha256")
    if not f_hash or f_hash != EXPECTED_MANIFEST_FILE_SHA256:
        raise IncompatibleCheckpointError(
            f"Checkpoint manifest hash mismatch! Expected {EXPECTED_MANIFEST_FILE_SHA256}, got '{f_hash}'"
        )
    if not c_hash or c_hash != EXPECTED_COMPOSITE_SHA256:
        raise IncompatibleCheckpointError(
            f"Checkpoint composite hash mismatch! Expected {EXPECTED_COMPOSITE_SHA256}, got '{c_hash}'"
        )

    # 4. Require model_metadata to be present and structurally valid (architecture and frozen gamma_min=1.0)
    if "model_metadata" not in metadata or not isinstance(metadata["model_metadata"], dict):
        raise IncompatibleCheckpointError(f"Checkpoint at {path} missing valid 'model_metadata' dictionary.")

    model_meta = metadata["model_metadata"]
    arch = model_meta.get("architecture")
    if arch != "ACS-Sub-INN-v3":
        raise IncompatibleCheckpointError(
            f"Checkpoint architecture mismatch! Expected 'ACS-Sub-INN-v3', got '{arch}'"
        )

    gamma_min = model_meta.get("gamma_min")
    if gamma_min is None or not isinstance(gamma_min, (int, float)) or isinstance(gamma_min, bool):
        raise IncompatibleCheckpointError(
            f"Checkpoint model_metadata missing valid numeric 'gamma_min', got {gamma_min!r}"
        )
    if not math.isclose(float(gamma_min), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise IncompatibleCheckpointError(
            f"Checkpoint model_metadata gamma_min mismatch! Frozen architecture requires gamma_min=1.0, got {gamma_min}"
        )

    # 5. Check model_state_dict presence, keys, and tensor shapes BEFORE mutating live model
    if "model_state_dict" not in checkpoint or not isinstance(checkpoint["model_state_dict"], dict):
        raise IncompatibleCheckpointError(f"Checkpoint at {path} missing valid 'model_state_dict'.")

    incoming_model_state = checkpoint["model_state_dict"]
    current_model_state = bridge.acs_model.state_dict()

    if set(current_model_state.keys()) != set(incoming_model_state.keys()):
        missing = set(current_model_state.keys()) - set(incoming_model_state.keys())
        extra = set(incoming_model_state.keys()) - set(current_model_state.keys())
        raise IncompatibleCheckpointError(
            f"Model state_dict incompatible with ACS-Sub-INN: key mismatch. Missing: {missing}, Extra: {extra}"
        )

    for k, cur_t in current_model_state.items():
        inc_t = incoming_model_state[k]
        if not isinstance(inc_t, torch.Tensor):
            raise IncompatibleCheckpointError(f"Model state_dict incompatible: '{k}' is not a Tensor.")
        if cur_t.shape != inc_t.shape:
            raise IncompatibleCheckpointError(
                f"Model state_dict incompatible with ACS-Sub-INN: shape mismatch for '{k}'. "
                f"Expected {cur_t.shape}, got {inc_t.shape}"
            )

    # 6. Check required optimizer state if optimizer is provided
    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint or checkpoint["optimizer_state_dict"] is None:
            raise IncompatibleCheckpointError(
                f"Optimizer was provided for resumption, but checkpoint at {path} is missing 'optimizer_state_dict'."
            )
        if not isinstance(checkpoint["optimizer_state_dict"], dict):
            raise IncompatibleCheckpointError(
                f"Checkpoint at {path} has invalid 'optimizer_state_dict' (expected dict)."
            )

    # 7. Check required scheduler state if scheduler is provided
    if scheduler is not None:
        if "scheduler_state_dict" not in checkpoint or checkpoint["scheduler_state_dict"] is None:
            raise IncompatibleCheckpointError(
                f"Scheduler was provided for resumption, but checkpoint at {path} is missing 'scheduler_state_dict'."
            )
        if not isinstance(checkpoint["scheduler_state_dict"], dict):
            raise IncompatibleCheckpointError(
                f"Checkpoint at {path} has invalid 'scheduler_state_dict' (expected dict)."
            )

    # 8. Check DataLoader generator state if restoration is required
    dl_gen_state = checkpoint.get("dataloader_generator_state")
    if train_loader is not None and getattr(train_loader, "generator", None) is not None:
        if require_dataloader_generator and dl_gen_state is None:
            raise IncompatibleCheckpointError(
                f"DataLoader generator restoration is required, but checkpoint at {path} lacks generator state."
            )

    # --- ALL VALIDATIONS PASSED: APPLY CHECKPOINT STATES ---

    # Apply model parameters
    try:
        bridge.acs_model.load_state_dict(incoming_model_state, strict=True)
    except Exception as e:
        raise IncompatibleCheckpointError(f"Model state_dict incompatible with ACS-Sub-INN: {e}") from e

    # Apply optimizer state if provided
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            raise IncompatibleCheckpointError(f"Optimizer state_dict incompatible: {e}") from e

    # Apply scheduler state if provided
    if scheduler is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            raise IncompatibleCheckpointError(f"Scheduler state_dict incompatible: {e}") from e

    # Restore multi-library RNG states
    if restore_rng and "rng_states" in checkpoint:
        rng_dict = checkpoint["rng_states"]
        if isinstance(rng_dict, dict):
            if "python_rng_state" in rng_dict and rng_dict["python_rng_state"] is not None:
                random.setstate(rng_dict["python_rng_state"])
            if "numpy_rng_state" in rng_dict and rng_dict["numpy_rng_state"] is not None:
                np.random.set_state(rng_dict["numpy_rng_state"])
            if "torch_cpu_rng_state" in rng_dict and rng_dict["torch_cpu_rng_state"] is not None:
                torch.set_rng_state(rng_dict["torch_cpu_rng_state"])
            if (
                torch.cuda.is_available()
                and "torch_cuda_rng_state" in rng_dict
                and rng_dict["torch_cuda_rng_state"] is not None
            ):
                try:
                    torch.cuda.set_rng_state_all(rng_dict["torch_cuda_rng_state"])
                except Exception:
                    pass

    # Restore DataLoader generator state if supported
    dl_restored = False
    if train_loader is not None and getattr(train_loader, "generator", None) is not None:
        if dl_gen_state is not None:
            try:
                train_loader.generator.set_state(dl_gen_state)
                dl_restored = True
            except Exception as e:
                raise IncompatibleCheckpointError(f"Failed to restore DataLoader generator state: {e}") from e

    return {
        "metadata": metadata,
        "epoch": checkpoint.get("epoch", 0),
        "next_epoch": checkpoint.get("next_epoch", checkpoint.get("epoch", 0) + 1),
        "global_step": checkpoint.get("global_step", 0),
        "best_val_metric": checkpoint.get("best_val_metric", float("inf")),
        "early_stopping_state": checkpoint.get("early_stopping_state", {}),
        "resumed_from": str(path),
        "dataloader_generator_restored": dl_restored,
    }


# =============================================================================
# 6. Workstream B — Epoch Training Runner
# =============================================================================

@dataclass
class EpochTrainMetrics:
    """Metrics aggregated over a single training epoch."""

    epoch: int
    global_step: int
    total_loss: float
    loss_bit_raw: float
    loss_bit_weighted: float
    loss_ctx_raw: float
    loss_ctx_weighted: float
    loss_gain_raw: float
    loss_gain_weighted: float
    loss_img_raw: float
    loss_img_weighted: float
    grad_norm_pre_clip: float
    grad_norm_post_clip: float
    learning_rate: float
    duration_sec: float
    batches_processed: int
    optimizer_steps_taken: int
    accumulation_windows_completed: int = 0


@dataclass
class TrainingRunSummary:
    """Summary produced upon completion or early stopping of a training run."""

    run_id: str
    completed_epochs: int
    global_steps: int
    best_val_metric: float
    best_checkpoint_path: Optional[str]
    termination_reason: str
    history: List[Dict[str, Any]]


class TrainingRunner:
    """Orchestrates multi-epoch ACS-Sub-INN v3 training with strict boundary protection.

    Features:
    - Fail-closed training authorization gate.
    - Mathematically sound gradient accumulation with sample-weighted loss scaling.
    - True accumulation window normalization for full and partial windows.
    - Resumption at epoch + 1 with restored counters, RNGs, best metrics, and DataLoader state.
    - Gradient norm clipping monitoring pre- and post-clip norms.
    - Finite loss and finite gradient validation.
    - Validation evaluation and structured telemetry logging per epoch.
    - Epoch-boundary atomic checkpointing and early stopping.
    """

    def __init__(
        self,
        bridge: NeuralSpatialBridge,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        composite_loss: CompositeLoss,
        config: TrainingRunConfig,
        scheduler: Optional[Any] = None,
        telemetry_logger: Optional[TrainingTelemetryLogger] = None,
        authorization_token: Optional[TrainingAuthorizationToken] = None,
        resume_checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        config.validate()

        self.bridge = bridge
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.composite_loss = composite_loss
        self.config = config
        self.scheduler = scheduler
        self.authorization_token = authorization_token
        self.resume_checkpoint_path = resume_checkpoint_path

        if telemetry_logger is not None:
            self.telemetry = telemetry_logger
        else:
            self.telemetry = TrainingTelemetryLogger(config.telemetry_path)

        self.device = torch.device(config.device)
        self.bridge.to(self.device)

        # Verify loader splits fail closed (must have explicit dataset_split attribute)
        train_split = getattr(train_loader, "dataset_split", None)
        val_split = getattr(val_loader, "dataset_split", None)
        if train_split is None or train_split != "train":
            raise DatasetSplitBoundaryError(f"TrainingRunner train_loader requires split='train', got '{train_split}'.")
        if val_split is None or val_split != "val":
            raise DatasetSplitBoundaryError(f"TrainingRunner val_loader requires split='val', got '{val_split}'.")

        # Validate that TrainingRunConfig loss weights match CompositeLoss loss weights
        loss_mismatches = []
        for term in ("bit", "ctx", "gain", "img"):
            cfg_val = getattr(config, f"lambda_{term}", None)
            loss_val = getattr(composite_loss, f"lambda_{term}", None)
            if isinstance(loss_val, (int, float)) and isinstance(cfg_val, (int, float)):
                if not math.isclose(float(cfg_val), float(loss_val), rel_tol=1e-7, abs_tol=1e-7):
                    loss_mismatches.append(f"lambda_{term}: config={cfg_val} != loss={loss_val}")
            elif loss_val is not None and not isinstance(loss_val, (int, float)):
                # If loss_val is not numeric and not a Mock object
                if not hasattr(loss_val, "_mock_return_value") and not hasattr(loss_val, "assert_called"):
                    loss_mismatches.append(f"lambda_{term}: invalid loss weight type {type(loss_val).__name__}")
        if loss_mismatches:
            raise ValueError(
                f"Loss weight configuration mismatch between TrainingRunConfig and CompositeLoss: "
                f"{', '.join(loss_mismatches)}. Both configurations must match to prevent silent divergence."
            )

        self.start_epoch = 1
        self.global_step = 0
        self.best_val_metric = float("inf")
        self.early_stop_counter = 0
        self.best_checkpoint_path: Optional[str] = None
        self.history: List[Dict[str, Any]] = []

        if self.resume_checkpoint_path is not None:
            restored = load_training_checkpoint(
                checkpoint_path=self.resume_checkpoint_path,
                bridge=self.bridge,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                restore_rng=True,
                train_loader=self.train_loader,
            )
            self.start_epoch = restored["next_epoch"]
            self.global_step = restored["global_step"]
            self.best_val_metric = restored["best_val_metric"]
            self.early_stop_counter = restored.get("early_stopping_state", {}).get("counter", 0)

    def train_epoch(self, epoch: int, dry_run: bool = False) -> EpochTrainMetrics:
        """Executes a single epoch over train_loader with sample-weighted gradient accumulation.

        Args:
            epoch: Epoch index (1-based).
            dry_run: If True, executes control flow without optimizer.step() or global_step increments.

        Raises:
            TrainingNotAuthorizedError: If not in dry_run and authorization is missing.
            NonFiniteLossError: If loss is NaN or Inf.
            NonFiniteGradientError: If gradient is NaN or Inf.
        """
        if not dry_run:
            verify_training_authorization(self.authorization_token)

        self.bridge.train()
        start_time = time.time()

        total_loss_accum = 0.0
        loss_bit_raw_accum = 0.0
        loss_bit_w_accum = 0.0
        loss_ctx_raw_accum = 0.0
        loss_ctx_w_accum = 0.0
        loss_gain_raw_accum = 0.0
        loss_gain_w_accum = 0.0
        loss_img_raw_accum = 0.0
        loss_img_w_accum = 0.0

        last_pre_clip_norm = 0.0
        last_post_clip_norm = 0.0
        optimizer_steps_taken = 0
        accumulation_windows_completed = 0
        batches_processed = 0

        self.optimizer.zero_grad()
        accum_steps = self.config.grad_accum_steps

        # Group batches into accumulation windows
        window_batches: List[Dict[str, Any]] = []

        def _process_accumulation_window(batches: List[Dict[str, Any]]) -> None:
            nonlocal batches_processed, optimizer_steps_taken, accumulation_windows_completed
            nonlocal total_loss_accum, loss_bit_raw_accum, loss_bit_w_accum
            nonlocal loss_ctx_raw_accum, loss_ctx_w_accum, loss_gain_raw_accum
            nonlocal loss_gain_w_accum, loss_img_raw_accum, loss_img_w_accum
            nonlocal last_pre_clip_norm, last_post_clip_norm

            # Validate each batch fails closed before forward/backward pass
            for b_item in batches:
                validate_batch_partition(b_item, expected_split="train")

            window_sample_counts = [
                b_item["carrier"].shape[0] if b_item["carrier"].ndim == 4 else 1
                for b_item in batches
            ]
            total_window_samples = sum(window_sample_counts)
            if total_window_samples == 0:
                return

            for b_item in batches:
                carrier = b_item["carrier"].to(self.device)
                if carrier.ndim == 3:
                    carrier = carrier.unsqueeze(0)

                b_sz, c_ch, h_dim, w_dim = carrier.shape
                sub_h, sub_w = h_dim // 2, w_dim // 2

                payload_bits = torch.randint(0, 2, (b_sz, 1, sub_h, sub_w), device=self.device, dtype=torch.float32)
                payload_bipolar = ACSSubINN.bipolar_encode(payload_bits)
                mask = torch.ones(b_sz, 1, sub_h, sub_w, device=self.device, dtype=torch.float32)

                emb_res = self.bridge.embed_spatial(
                    carrier=carrier,
                    payload_bipolar=payload_bipolar,
                    mask=mask,
                    differentiable=True,
                )
                ext_res = self.bridge.extract_spatial(
                    stego_carrier=emb_res.stego_image,
                    mask=mask,
                )

                hh_drop_orig = emb_res.subbands_original[3][:, 0:1, :, :]
                predicted_context = emb_res.acs_embedding_result.context
                gamma = emb_res.acs_embedding_result.gamma

                breakdown: LossBreakdown = self.composite_loss(
                    delta_rec=ext_res.delta_rec,
                    payload_bipolar=payload_bipolar,
                    predicted_context=predicted_context,
                    target_hh_drop=hh_drop_orig,
                    gamma=gamma,
                    stego_spatial=emb_res.stego_image,
                    cover_spatial=carrier,
                    mask=mask,
                )

                raw_total = breakdown.total_loss.item()
                if not math.isfinite(raw_total):
                    raise NonFiniteLossError(f"Training loss non-finite at batch {batches_processed}: {raw_total}")

                # Sample-weighted scaling for exact window objective representation
                sample_weight = float(b_sz) / float(total_window_samples)
                scaled_loss = breakdown.total_loss * sample_weight
                scaled_loss.backward()

                total_loss_accum += raw_total
                loss_bit_raw_accum += float(breakdown.loss_bit.item())
                loss_bit_w_accum += float(self.composite_loss.lambda_bit * breakdown.loss_bit.item())
                loss_ctx_raw_accum += float(breakdown.loss_ctx.item())
                loss_ctx_w_accum += float(self.composite_loss.lambda_ctx * breakdown.loss_ctx.item())
                loss_gain_raw_accum += float(breakdown.loss_gain.item())
                loss_gain_w_accum += float(self.composite_loss.lambda_gain * breakdown.loss_gain.item())
                loss_img_raw_accum += float(breakdown.loss_img.item())
                loss_img_w_accum += float(self.composite_loss.lambda_img * breakdown.loss_img.item())

                batches_processed += 1

            # Window completed: check finite grads, compute norms, clip, and step
            accumulation_windows_completed += 1

            # Finite gradient check
            for p in self.bridge.acs_model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    self.optimizer.zero_grad()
                    raise NonFiniteGradientError("Gradient contains NaN or Infinity values!")

            # Pre-clip gradient norm
            grads = [p.grad for p in self.bridge.acs_model.parameters() if p.grad is not None]
            unclipped_norm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)).item()) if grads else 0.0
            last_pre_clip_norm = unclipped_norm

            # Gradient clipping
            if self.config.clip_grad_norm is not None and self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.bridge.acs_model.parameters(),
                    max_norm=self.config.clip_grad_norm,
                )
                grads_post = [p.grad for p in self.bridge.acs_model.parameters() if p.grad is not None]
                last_post_clip_norm = float(torch.sqrt(sum((g ** 2).sum() for g in grads_post)).item()) if grads_post else 0.0
            else:
                last_post_clip_norm = unclipped_norm

            if not dry_run:
                self.optimizer.step()
                optimizer_steps_taken += 1
                self.global_step += 1

            self.optimizer.zero_grad()

        for batch in self.train_loader:
            window_batches.append(batch)
            if len(window_batches) == accum_steps:
                _process_accumulation_window(window_batches)
                window_batches = []

        if len(window_batches) > 0:
            _process_accumulation_window(window_batches)
            window_batches = []

        duration = time.time() - start_time
        current_lr = float(self.optimizer.param_groups[0]["lr"])

        return EpochTrainMetrics(
            epoch=epoch,
            global_step=self.global_step,
            total_loss=total_loss_accum / max(1, batches_processed),
            loss_bit_raw=loss_bit_raw_accum / max(1, batches_processed),
            loss_bit_weighted=loss_bit_w_accum / max(1, batches_processed),
            loss_ctx_raw=loss_ctx_raw_accum / max(1, batches_processed),
            loss_ctx_weighted=loss_ctx_w_accum / max(1, batches_processed),
            loss_gain_raw=loss_gain_raw_accum / max(1, batches_processed),
            loss_gain_weighted=loss_gain_w_accum / max(1, batches_processed),
            loss_img_raw=loss_img_raw_accum / max(1, batches_processed),
            loss_img_weighted=loss_img_w_accum / max(1, batches_processed),
            grad_norm_pre_clip=last_pre_clip_norm,
            grad_norm_post_clip=last_post_clip_norm,
            learning_rate=current_lr,
            duration_sec=duration,
            batches_processed=batches_processed,
            optimizer_steps_taken=optimizer_steps_taken,
            accumulation_windows_completed=accumulation_windows_completed,
        )

    def train(self) -> TrainingRunSummary:
        """Executes multi-epoch training loop.

        CRITICAL GATE: Fails closed immediately if authorization_token is missing or invalid.
        """
        verify_training_authorization(self.authorization_token)

        early_stop_counter = self.early_stop_counter
        termination_reason = "COMPLETED"

        if self.start_epoch > self.config.epochs:
            return TrainingRunSummary(
                run_id=self.config.run_id,
                completed_epochs=len(self.history),
                global_steps=self.global_step,
                best_val_metric=self.best_val_metric,
                best_checkpoint_path=self.best_checkpoint_path,
                termination_reason="ALREADY_COMPLETED",
                history=self.history,
            )

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_metrics = self.train_epoch(epoch, dry_run=False)
            val_metrics = evaluate_validation_epoch(
                bridge=self.bridge,
                val_loader=self.val_loader,
                composite_loss=self.composite_loss,
                device=self.device,
            )

            # Step learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.total_loss)
                else:
                    self.scheduler.step()

            # Check best validation performance (lower loss is better)
            is_best = (val_metrics.total_loss < (self.best_val_metric - self.config.early_stopping_min_delta))
            if is_best:
                self.best_val_metric = val_metrics.total_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Save epoch checkpoint
            checkpoint_path = None
            if epoch % self.config.save_every_n_epochs == 0 or is_best:
                ckpt_filename = f"checkpoint_epoch_{epoch:03d}.pt"
                ckpt_dest = Path(self.config.checkpoint_dir) / ckpt_filename
                save_training_checkpoint(
                    checkpoint_path=ckpt_dest,
                    bridge=self.bridge,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_val_metric=self.best_val_metric,
                    early_stopping_state={"counter": early_stop_counter, "patience": self.config.early_stopping_patience},
                    config=self.config,
                    is_best=is_best,
                    authorization_token=self.authorization_token,
                    train_loader=self.train_loader,
                )
                checkpoint_path = str(ckpt_dest)

                if is_best:
                    best_ckpt_dest = Path(self.config.checkpoint_dir) / "best_model.pt"
                    save_training_checkpoint(
                        checkpoint_path=best_ckpt_dest,
                        bridge=self.bridge,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        global_step=self.global_step,
                        best_val_metric=self.best_val_metric,
                        early_stopping_state={"counter": early_stop_counter, "patience": self.config.early_stopping_patience},
                        config=self.config,
                        is_best=True,
                        authorization_token=self.authorization_token,
                        train_loader=self.train_loader,
                    )
                    self.best_checkpoint_path = str(best_ckpt_dest)

            # Structured Telemetry Logging
            telemetry_record = {
                "run_id": self.config.run_id,
                "epoch": epoch,
                "global_step": self.global_step,
                "train_total_loss": train_metrics.total_loss,
                "train_loss_bit_raw": train_metrics.loss_bit_raw,
                "train_loss_bit_weighted": train_metrics.loss_bit_weighted,
                "train_loss_ctx_raw": train_metrics.loss_ctx_raw,
                "train_loss_ctx_weighted": train_metrics.loss_ctx_weighted,
                "train_loss_gain_raw": train_metrics.loss_gain_raw,
                "train_loss_gain_weighted": train_metrics.loss_gain_weighted,
                "train_loss_img_raw": train_metrics.loss_img_raw,
                "train_loss_img_weighted": train_metrics.loss_img_weighted,
                "val_total_loss": val_metrics.total_loss,
                "val_loss_bit_raw": val_metrics.loss_bit_raw,
                "val_loss_bit_weighted": val_metrics.loss_bit_weighted,
                "val_loss_ctx_raw": val_metrics.loss_ctx_raw,
                "val_loss_ctx_weighted": val_metrics.loss_ctx_weighted,
                "val_loss_gain_raw": val_metrics.loss_gain_raw,
                "val_loss_gain_weighted": val_metrics.loss_gain_weighted,
                "val_loss_img_raw": val_metrics.loss_img_raw,
                "val_loss_img_weighted": val_metrics.loss_img_weighted,
                "val_pre_ecc_ber": val_metrics.pre_ecc_ber,
                "val_stego_psnr_db": val_metrics.stego_psnr_db,
                "val_stego_ssim": val_metrics.stego_ssim,
                "learning_rate": train_metrics.learning_rate,
                "grad_norm_pre_clip": train_metrics.grad_norm_pre_clip,
                "grad_norm_post_clip": train_metrics.grad_norm_post_clip,
                "effective_batch_size": self.config.batch_size * self.config.grad_accum_steps,
                "grad_accum_steps": self.config.grad_accum_steps,
                "epoch_duration_sec": train_metrics.duration_sec,
                "checkpoint_path": checkpoint_path,
                "is_best": is_best,
                "training_authorized": True,
            }
            self.telemetry.log_epoch(telemetry_record)
            self.history.append(telemetry_record)

            # Early stopping check
            if early_stop_counter >= self.config.early_stopping_patience:
                termination_reason = "EARLY_STOPPING"
                break

        return TrainingRunSummary(
            run_id=self.config.run_id,
            completed_epochs=len(self.history),
            global_steps=self.global_step,
            best_val_metric=self.best_val_metric,
            best_checkpoint_path=self.best_checkpoint_path,
            termination_reason=termination_reason,
            history=self.history,
        )
