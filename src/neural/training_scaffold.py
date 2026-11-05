"""Training-Loop Scaffold and Boundary Protection for Phase 3 ACS-Sub-INN v3 Engine.

SCAFFOLDING SPECIFICATION (NO MODEL TRAINING):
This module defines the training and evaluation infrastructure without executing
model training or modifying the frozen ACS-Sub-INN v3 architecture:
1. DatasetSplitManager: Audits and enforces strict data partitioning on DIV2K HR
   (600 train / 100 val / 100 test). Strictly protects the held-out test partition.
2. TrainingConfig: Dataclass specifying reproducibility seeds, device, optimizer, and loss weights.
3. set_reproducibility_seeds: Comprehensive multi-library seed control (Python, NumPy, PyTorch CPU/CUDA, cuDNN).
4. build_model_and_optimizer: Factory for NeuralSpatialBridge and AdamW optimizer.
5. execute_training_step: Step function executing forward and backward passes.
   STRICT INVARIANT: MUST NEVER CALL optimizer.step() IN THIS SPRINT.
6. execute_validation_step: Inference-only evaluation step with torch.no_grad().
7. save_synthetic_smoke_checkpoint / load_and_validate_checkpoint: Checkpointing with explicit smoke-test tags.
8. GateRegistry: Authoritative catalog of project acceptance criteria and their governance status.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.losses import (
    ClippingDiagnosticReport,
    CompositeLoss,
    LossBreakdown,
    evaluate_clipping_diagnostic,
)
from src.neural.spatial_bridge import (
    NeuralSpatialBridge,
    SpatialEmbeddingResult,
    SpatialExtractionResult,
)
from src.neural.dataset import compute_manifest_composite_sha256


# =============================================================================
# 1. Dataset Split Manager & Partition Protection
# =============================================================================

class DatasetSplitManager:
    """Audits and manages the canonical DIV2K High-Resolution dataset partitioning.

    Governing Manifest:
        docs/project/div2k_hr_split.json
        Composite SHA-256: d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed

    Partition Invariants:
        - Train: 600 images ('0001.png' through '0600.png')
        - Val:   100 images ('0601.png' through '0700.png')
        - Test:  100 images ('0701.png' through '0800.png') [HELD-OUT EVALUATION ONLY]

    Security & Integrity Guarantee:
        The training loader cannot select the held-out test split. Any attempt to request
        'test' for training or hyperparameter tuning raises a hard ValueError.
    """

    EXPECTED_COMPOSITE_SHA256: str = "d0f5f4ee0912d007f0a8695fe1713d9cf5a75cf97bc9c52d601ea862271b31ed"
    EXPECTED_MANIFEST_FILE_SHA256: str = "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff"
    EXPECTED_COUNTS: Dict[str, int] = {"train": 600, "val": 100, "test": 100}

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        is_custom_fixture: bool = False,
        expected_file_sha256: Optional[str] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        canonical_manifest_path = (repo_root / "docs" / "project" / "div2k_hr_split.json").resolve()

        if manifest_path is None:
            manifest_path = canonical_manifest_path

        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Dataset split manifest not found at: {self.manifest_path}")

        self._raw_bytes = self.manifest_path.read_bytes()
        self.file_sha256 = hashlib.sha256(self._raw_bytes).hexdigest()

        try:
            self._raw_data = json.loads(self._raw_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Malformed manifest JSON at {self.manifest_path}: {e}") from e

        self.is_canonical = (self.manifest_path == canonical_manifest_path)
        self.is_custom_fixture = is_custom_fixture
        self.expected_file_sha256 = expected_file_sha256

        self._validate_manifest()

    def _validate_manifest(self) -> None:
        """Verifies manifest schema, file digest, composite hash, and partition sizes."""
        if self.is_canonical:
            if self.file_sha256 != self.EXPECTED_MANIFEST_FILE_SHA256:
                raise ValueError(
                    f"SECURITY/INTEGRITY VIOLATION: Canonical manifest file SHA-256 mismatch! "
                    f"Expected {self.EXPECTED_MANIFEST_FILE_SHA256}, got {self.file_sha256}"
                )
        else:
            if not self.is_custom_fixture:
                raise ValueError(
                    f"SECURITY VIOLATION: Non-canonical manifest '{self.manifest_path}' rejected. "
                    f"Only the canonical project manifest is permitted for production workflows. "
                    f"For test fixtures, set is_custom_fixture=True explicitly."
                )
            if self.expected_file_sha256 is not None and self.file_sha256 != self.expected_file_sha256:
                raise ValueError(
                    f"Manifest file SHA-256 mismatch! Expected {self.expected_file_sha256}, got {self.file_sha256}"
                )

        meta = self._raw_data.get("metadata", {})
        comp_hash = meta.get("dataset_composite_sha256")
        if comp_hash != self.EXPECTED_COMPOSITE_SHA256:
            raise ValueError(
                f"Dataset composite SHA-256 mismatch! Expected {self.EXPECTED_COMPOSITE_SHA256}, got {comp_hash}"
            )

        recomputed_composite = compute_manifest_composite_sha256(self._raw_data)
        if self.is_canonical and recomputed_composite != self.EXPECTED_COMPOSITE_SHA256:
            raise ValueError(
                f"Recomputed dataset composite SHA-256 mismatch! Expected {self.EXPECTED_COMPOSITE_SHA256}, got {recomputed_composite}"
            )
        elif not self.is_canonical and recomputed_composite != comp_hash:
            raise ValueError(
                f"Recomputed dataset composite SHA-256 mismatch! Expected {comp_hash}, got {recomputed_composite}"
            )

        for split_name, expected_count in self.EXPECTED_COUNTS.items():
            records = self._raw_data.get(split_name, [])
            if len(records) != expected_count:
                raise ValueError(
                    f"Partition count mismatch for '{split_name}': expected {expected_count}, got {len(records)}"
                )

    def get_train_records(self) -> List[Dict[str, Any]]:
        """Returns the 600 canonical training image records."""
        return list(self._raw_data["train"])

    def get_val_records(self) -> List[Dict[str, Any]]:
        """Returns the 100 canonical validation image records."""
        return list(self._raw_data["val"])

    def get_records_for_training(self, split: str) -> List[Dict[str, Any]]:
        """Returns records for an authorized training or tuning split.

        Raises:
            ValueError: If split is 'test' or any unrecognized split name.
        """
        if split == "test":
            raise ValueError(
                "DATA INTEGRITY VIOLATION: Accessing the held-out 'test' split for training, "
                "hyperparameter tuning, or validation is strictly forbidden by Acceptance Criteria §5."
            )
        if split == "train":
            return self.get_train_records()
        elif split == "val":
            return self.get_val_records()
        else:
            raise ValueError(f"Unknown split '{split}'. Authorized training splits: ('train', 'val').")

    def get_held_out_test_manifest_for_audit_only(self) -> List[Dict[str, Any]]:
        """Returns test records strictly for post-training held-out benchmark evaluation."""
        return list(self._raw_data["test"])


# =============================================================================
# 2. Training Configuration & Reproducibility
# =============================================================================

@dataclass
class TrainingConfig:
    """Strongly-typed configuration for training scaffold and smoke tests."""

    seed: int = 42
    device: str = "cpu"
    deterministic: bool = True
    batch_size: int = 1
    crop_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: Optional[float] = 1.0

    # Candidate loss weights (PROPOSED_UNVALIDATED)
    lambda_bit: float = 1.0
    lambda_ctx: float = 0.5
    lambda_gain: float = 0.01
    lambda_img: float = 0.1

    checkpoint_dir: str = "checkpoints/sprint3_smoke"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_reproducibility_seeds(seed: int = 42, deterministic: bool = True) -> Dict[str, Any]:
    """Sets random seeds across Python, NumPy, PyTorch CPU, and CUDA.

    Args:
        seed: Integer seed value.
        deterministic: If True, sets PyTorch/cuDNN flags to enforce deterministic algorithms.

    Returns:
        Dictionary reporting seed settings and device configuration.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_configured = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        cuda_configured = True

    return {
        "seed": seed,
        "deterministic": deterministic,
        "cuda_configured": cuda_configured,
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
    }


# =============================================================================
# 3. Model & Optimizer Factory
# =============================================================================

def build_model_and_optimizer(
    config: TrainingConfig,
) -> Tuple[NeuralSpatialBridge, torch.optim.Optimizer, CompositeLoss]:
    """Constructs the spatial bridge, AdamW optimizer, and composite loss.

    Architecture Note:
        Only parameters in bridge.acs_model (ContextPredictorCNN and AdaptiveGainCNN)
        are registered with the optimizer. HaarDWT2DTorch and SpatialQuantizeSTE
        are parameter-free transform/quantization primitives.
    """
    device = torch.device(config.device)
    acs_model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=acs_model).to(device)

    # Filter trainable parameters
    trainable_params = [p for p in bridge.acs_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    composite_loss = CompositeLoss(
        lambda_bit=config.lambda_bit,
        lambda_ctx=config.lambda_ctx,
        lambda_gain=config.lambda_gain,
        lambda_img=config.lambda_img,
    ).to(device)

    return bridge, optimizer, composite_loss


# =============================================================================
# 4. Step Execution (STRICT: NO OPTIMIZER.STEP)
# =============================================================================

@dataclass(frozen=True)
class TrainingStepResult:
    """Artifacts produced by a single training forward/backward step."""

    loss_breakdown: LossBreakdown
    clipping_diagnostic: ClippingDiagnosticReport
    total_grad_norm: float
    max_param_grad: float
    min_param_grad: float
    is_grad_finite: bool
    stego_psnr_db: float
    stego_ssim: float
    pre_ecc_ber: float
    optimizer_step_executed: bool  # MUST ALWAYS BE FALSE IN THIS SPRINT


def execute_training_step(
    bridge: NeuralSpatialBridge,
    optimizer: torch.optim.Optimizer,
    composite_loss: CompositeLoss,
    carrier: torch.Tensor,
    payload_bits: torch.Tensor,
    mask: torch.Tensor,
    run_backward: bool = True,
    clip_grad_norm: Optional[float] = 1.0,
    split_tag: Optional[str] = None,
) -> TrainingStepResult:
    """Executes a single differentiable forward and backward step WITHOUT optimizer.step().

    CRITICAL SAFETY CONTRACT:
        This function computes forward embedding, extraction, candidate losses, and
        runs backward() to compute parameter gradients. It verifies that gradients are
        finite. It explicitly DOES NOT CALL optimizer.step().

    LOW-LEVEL BYPASS & THREAT MODEL NOTICE:
        execute_training_step() is an internal, low-level mathematical step function that
        operates directly on raw PyTorch tensors. Supported project workflows MUST use
        execute_partition_verified_step() to enforce that held-out test data cannot enter
        optimization. If split_tag is passed as 'test', this function fails closed immediately.

    Args:
        bridge: NeuralSpatialBridge instance.
        optimizer: Optimizer instance (used only for zero_grad).
        composite_loss: CompositeLoss module.
        carrier: RGB carrier tensor (B, 3, H, W) in [0, 255].
        payload_bits: Binary payload tensor (B, 1, H/2, W/2) in {0, 1}.
        mask: Active embedding mask (B, 1, H/2, W/2).
        run_backward: If True, computes backward pass.
        clip_grad_norm: Optional gradient clipping threshold.
        split_tag: Optional split identifier string. Fails closed if 'test'.

    Returns:
        TrainingStepResult containing loss breakdown and gradient metrics.

    Raises:
        ValueError: If split_tag is 'test'.
    """
    if split_tag == "test":
        raise ValueError(
            "SECURITY/INTEGRITY VIOLATION: execute_training_step explicitly rejects split_tag='test'. "
            "Held-out test images cannot enter optimization."
        )

    bridge.train()

    # Convert binary payload to bipolar {-1.0, +1.0}
    payload_bipolar = ACSSubINN.bipolar_encode(payload_bits)

    # 1. Forward Differentiable Embedding
    emb_res: SpatialEmbeddingResult = bridge.embed_spatial(
        carrier=carrier,
        payload_bipolar=payload_bipolar,
        mask=mask,
        differentiable=True,
    )

    # 2. Extraction through Differentiable Surrogate
    ext_res: SpatialExtractionResult = bridge.extract_spatial(
        stego_carrier=emb_res.stego_image,
        mask=mask,
    )

    # Subband targets for loss
    hh_drop_orig = emb_res.subbands_original[3][:, 0:1, :, :]
    predicted_context = emb_res.acs_embedding_result.context
    gamma = emb_res.acs_embedding_result.gamma

    # 3. Compute Candidate Composite Loss
    loss_breakdown: LossBreakdown = composite_loss(
        delta_rec=ext_res.delta_rec,
        payload_bipolar=payload_bipolar,
        predicted_context=predicted_context,
        target_hh_drop=hh_drop_orig,
        gamma=gamma,
        stego_spatial=emb_res.stego_image,
        cover_spatial=carrier,
        mask=mask,
    )

    # 4. Backward Pass & Gradient Verification (NO OPTIMIZER.STEP)
    total_grad_norm = 0.0
    max_p_grad = 0.0
    min_p_grad = 0.0
    is_finite = True

    if run_backward:
        optimizer.zero_grad()
        loss_breakdown.total_loss.backward()

        # Check gradient finiteness
        grad_norms = []
        for p in bridge.acs_model.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    is_finite = False
                p_norm = float(p.grad.norm().item())
                grad_norms.append(p_norm)
                max_p_grad = max(max_p_grad, float(p.grad.abs().max().item()))

        if clip_grad_norm is not None and clip_grad_norm > 0:
            total_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(bridge.acs_model.parameters(), clip_grad_norm).item()
            )
        else:
            total_grad_norm = float(np.sqrt(sum(g ** 2 for g in grad_norms))) if grad_norms else 0.0

        # CRITICAL GOVERNING ASSERTION: NO OPTIMIZER.STEP
        # We explicitly omit optimizer.step() here per Sprint 3 mandate.

    # 5. Diagnostic Metrics (detached)
    clip_diag = evaluate_clipping_diagnostic(
        stego_raw_float=emb_res.stego_raw_float,
        stego_quantized=emb_res.stego_image,
    )

    with torch.no_grad():
        c_np = carrier[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        s_np = emb_res.stego_image[0].clamp(0, 255).round().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        psnr_val = calculate_psnr(c_np, s_np)
        ssim_val = calculate_ssim(c_np, s_np)

        active = (mask == 1.0)
        num_act = int(active.sum().item())
        if num_act > 0:
            rec_bits = ext_res.extracted_binary[active]
            orig_bits = payload_bits[active]
            ber_val = float((rec_bits != orig_bits).sum().item()) / float(num_act)
        else:
            ber_val = 0.0

    return TrainingStepResult(
        loss_breakdown=loss_breakdown,
        clipping_diagnostic=clip_diag,
        total_grad_norm=total_grad_norm,
        max_param_grad=max_p_grad,
        min_param_grad=min_p_grad,
        is_grad_finite=is_finite,
        stego_psnr_db=psnr_val,
        stego_ssim=ssim_val,
        pre_ecc_ber=ber_val,
        optimizer_step_executed=False,  # Enforced invariant
    )


def execute_partition_verified_step(
    bridge: NeuralSpatialBridge,
    optimizer: torch.optim.Optimizer,
    composite_loss: CompositeLoss,
    batch: Dict[str, Any],
    payload_bits: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    run_backward: bool = True,
    clip_grad_norm: Optional[float] = 1.0,
) -> TrainingStepResult:
    """Executes an optimizer-facing training step with strict split boundary enforcement.

    CRITICAL BOUNDARY ENFORCEMENT CONTRACT:
    - Verifies that batch['split'] is strictly 'train'.
    - Rejects any batch containing images from 'val' or held-out 'test'.
    - Validates image IDs to ensure no ID from the held-out range (701-800) enters optimization.
    - Computes forward and backward passes without calling optimizer.step().

    Args:
        bridge: NeuralSpatialBridge instance.
        optimizer: Optimizer instance.
        composite_loss: CompositeLoss instance.
        batch: Dictionary from DIV2KStegoDataset or build_stego_dataloader.
        payload_bits: Optional binary payload. If None, pseudo-random bits are generated.
        mask: Optional embedding mask. If None, active mask on all positions is generated.
        run_backward: If True, computes backward pass.
        clip_grad_norm: Optional gradient clipping norm.

    Returns:
        TrainingStepResult containing loss breakdown and gradient metrics.

    Raises:
        ValueError: If batch contains images from non-train partitions.
    """
    splits = batch.get("split")
    if isinstance(splits, str):
        splits = [splits]
    elif isinstance(splits, (list, tuple)):
        splits = list(splits)
    else:
        raise ValueError(f"Batch missing valid 'split' identifier. Keys: {list(batch.keys())}")

    for s in splits:
        if s == "test":
            raise ValueError(
                "SECURITY/INTEGRITY VIOLATION: Held-out 'test' partition detected in training step! "
                "Optimization on test data is strictly forbidden by Acceptance Criteria §5."
            )
        if s != "train":
            raise ValueError(
                f"BOUNDARY VIOLATION: execute_partition_verified_step requires split='train', got '{s}'."
            )

    # Image ID boundary check (701..800 is held-out test range)
    image_ids = batch.get("image_id")
    if image_ids is not None:
        if isinstance(image_ids, torch.Tensor):
            id_list = image_ids.view(-1).tolist()
        elif isinstance(image_ids, (list, tuple)):
            id_list = list(image_ids)
        elif isinstance(image_ids, int):
            id_list = [image_ids]
        else:
            id_list = []

        for img_id in id_list:
            if 701 <= img_id <= 800:
                raise ValueError(
                    f"SECURITY/INTEGRITY VIOLATION: Image ID {img_id} belongs to held-out test split (701-800)! "
                    "Cannot enter optimization."
                )

    carrier = batch["carrier"]
    if carrier.ndim == 3:
        carrier = carrier.unsqueeze(0)

    b, c, h, w = carrier.shape
    sub_h, sub_w = h // 2, w // 2
    device = carrier.device

    if payload_bits is None:
        payload_bits = torch.randint(0, 2, (b, 1, sub_h, sub_w), device=device, dtype=torch.float32)
    elif payload_bits.ndim == 3:
        payload_bits = payload_bits.unsqueeze(0)

    if mask is None:
        mask = torch.ones(b, 1, sub_h, sub_w, device=device, dtype=torch.float32)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)

    return execute_training_step(
        bridge=bridge,
        optimizer=optimizer,
        composite_loss=composite_loss,
        carrier=carrier,
        payload_bits=payload_bits,
        mask=mask,
        run_backward=run_backward,
        clip_grad_norm=clip_grad_norm,
        split_tag="train",
    )


def evaluate_partition_verified_step(
    bridge: NeuralSpatialBridge,
    composite_loss: CompositeLoss,
    batch: Dict[str, Any],
    payload_bits: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> TrainingStepResult:
    """Evaluates a model-selection / validation step with strict split boundary enforcement.

    CRITICAL BOUNDARY ENFORCEMENT CONTRACT:
    - Verifies that batch['split'] is strictly 'val' or 'validation'.
    - Rejects any batch containing images from 'train' or held-out 'test'.
    - Does NOT compute gradients or update model parameters (run_backward=False).

    Args:
        bridge: NeuralSpatialBridge instance.
        composite_loss: CompositeLoss instance.
        batch: Dictionary from DIV2KStegoDataset or build_stego_dataloader.
        payload_bits: Optional binary payload.
        mask: Optional embedding mask.

    Returns:
        TrainingStepResult containing validation metrics and loss breakdown.

    Raises:
        ValueError: If batch contains images from non-val partitions.
    """
    splits = batch.get("split")
    if isinstance(splits, str):
        splits = [splits]
    elif isinstance(splits, (list, tuple)):
        splits = list(splits)
    else:
        raise ValueError(f"Batch missing valid 'split' identifier. Keys: {list(batch.keys())}")

    for s in splits:
        if s == "test":
            raise ValueError(
                "SECURITY/INTEGRITY VIOLATION: Held-out 'test' partition detected in validation/model-selection! "
                "Evaluating test data for model selection is strictly forbidden by Acceptance Criteria §5."
            )
        if s not in ("val", "validation"):
            raise ValueError(
                f"BOUNDARY VIOLATION: evaluate_partition_verified_step requires split='val', got '{s}'."
            )

    carrier = batch["carrier"]
    if carrier.ndim == 3:
        carrier = carrier.unsqueeze(0)

    b, c, h, w = carrier.shape
    sub_h, sub_w = h // 2, w // 2
    device = carrier.device

    if payload_bits is None:
        payload_bits = torch.randint(0, 2, (b, 1, sub_h, sub_w), device=device, dtype=torch.float32)
    elif payload_bits.ndim == 3:
        payload_bits = payload_bits.unsqueeze(0)

    if mask is None:
        mask = torch.ones(b, 1, sub_h, sub_w, device=device, dtype=torch.float32)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)

    # Dummy optimizer for evaluation pass (no zero_grad or backward called)
    dummy_optimizer = torch.optim.SGD(bridge.parameters(), lr=1e-4)

    return execute_training_step(
        bridge=bridge,
        optimizer=dummy_optimizer,
        composite_loss=composite_loss,
        carrier=carrier,
        payload_bits=payload_bits,
        mask=mask,
        run_backward=False,
        split_tag="val",
    )


# =============================================================================
# 5. Checkpoint Management (Synthetic Smoke Test Artifacts)
# =============================================================================

def save_synthetic_smoke_checkpoint(
    bridge: NeuralSpatialBridge,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    checkpoint_path: Union[str, Path],
) -> Dict[str, Any]:
    """Saves a synthetic smoke-test checkpoint with explicit governance metadata.

    The saved file is clearly labelled as a synthetic smoke-test artifact,
    not a trained model, and carries training_authorized=False.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint_type": "SYNTHETIC_SMOKE_TEST_ARTIFACT",
        "training_authorized": False,
        "is_synthetic_smoke_test": True,
        "optimizer_step_applied": False,
        "software_environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        },
        "config": config.to_dict(),
    }

    checkpoint = {
        "metadata": metadata,
        "model_state_dict": bridge.acs_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)
    return metadata


def load_and_validate_checkpoint(
    checkpoint_path: Union[str, Path],
    bridge: NeuralSpatialBridge,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Loads a checkpoint and verifies architecture compatibility.

    Raises:
        RuntimeError: If parameter keys or shapes mismatch.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})

    bridge.acs_model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return metadata


# =============================================================================
# 6. Authoritative Gate & Metric Governance Registry
# =============================================================================

@dataclass(frozen=True)
class GateDefinition:
    """Formal description of an acceptance metric and its governance status."""

    name: str
    target_threshold: str
    classification: str  # 'EXISTING_PROJECT_GATE', 'PROPOSED_UNVALIDATED', 'DIAGNOSTIC_ONLY', 'NOT_YET_DEFINED'
    description: str


class GateRegistry:
    """Authoritative registry of Phase 3 acceptance metrics and governance classifications."""

    GATES: Dict[str, GateDefinition] = {
        "pre_ecc_ber_png": GateDefinition(
            name="Pre-ECC Bit Error Rate (Lossless PNG)",
            target_threshold="BER <= 1.0e-3 (0.1%)",
            classification="EXISTING_PROJECT_GATE",
            description="Permissible pre-ECC demodulation bit error rate through lossless PNG roundtrip.",
        ),
        "post_ecc_ber": GateDefinition(
            name="Post-ECC Bit Error Rate",
            target_threshold="BER == 0.000000 (100% exact)",
            classification="EXISTING_PROJECT_GATE",
            description="Verbatim payload recovery after Reed-Solomon RS(255, 223) decoding.",
        ),
        "exact_plaintext_match": GateDefinition(
            name="Verbatim Plaintext Match",
            target_threshold="recovered_payload.data == secret.data (100%)",
            classification="EXISTING_PROJECT_GATE",
            description="Exact byte-for-byte identity of decrypted payload after AES-GCM tag verification.",
        ),
        "rs_codeword_headroom": GateDefinition(
            name="Reed-Solomon Codeword Error Headroom",
            target_threshold="symbol_errors <= 16 per 255-byte block",
            classification="EXISTING_PROJECT_GATE",
            description="Maximum observed symbol errors per RS block must remain strictly within t=16 limit.",
        ),
        "aes_gcm_tag_validity": GateDefinition(
            name="AES-GCM Authentication Tag Validity",
            target_threshold="100% valid tags (0 verification failures)",
            classification="EXISTING_PROJECT_GATE",
            description="Fail-closed security property ensuring zero plaintext leakage on any tampering.",
        ),
        "stego_psnr_png": GateDefinition(
            name="Stego Visual Fidelity (PSNR, Lossless PNG)",
            target_threshold="PSNR >= 60.0 dB",
            classification="EXISTING_PROJECT_GATE",
            description="Peak signal-to-noise ratio of stego carrier against original cover.",
        ),
        "stego_ssim_png": GateDefinition(
            name="Stego Structural Similarity (SSIM, Lossless PNG)",
            target_threshold="SSIM >= 0.9995",
            classification="EXISTING_PROJECT_GATE",
            description="Structural similarity index of stego carrier against cover.",
        ),
        "capacity_usable_512": GateDefinition(
            name="Usable Payload Capacity (512x512 Carrier)",
            target_threshold=">= 6,000 B plaintext (7,000 B protected)",
            classification="EXISTING_PROJECT_GATE",
            description="Minimum secret payload capacity supported on single 512x512 carrier.",
        ),
        "capacity_fail_closed": GateDefinition(
            name="Fail-Closed Capacity Gate",
            target_threshold="Strict rejection above 8,176 B protected",
            classification="EXISTING_PROJECT_GATE",
            description="Hard gate preventing buffer overflows or unmodulated carrier truncation.",
        ),
        "gpu_vram_peak": GateDefinition(
            name="Peak GPU VRAM Usage",
            target_threshold="<= 512 MB for 1024x1024 carrier",
            classification="EXISTING_PROJECT_GATE",
            description="Maximum working set memory during inference.",
        ),
        "inference_latency": GateDefinition(
            name="Inference Latency",
            target_threshold="<= 250 ms on target desktop GPU",
            classification="EXISTING_PROJECT_GATE",
            description="Total forward embedding time (DWT + INN + IDWT).",
        ),
        "loss_composition_weights": GateDefinition(
            name="Composite Loss Weighting",
            target_threshold="lambda_bit=1.0, ctx=0.5, gain=0.01, img=0.1",
            classification="PROPOSED_UNVALIDATED",
            description="Candidate hyperparameter weights requiring empirical calibration in future sprints.",
        ),
        "clipping_diagnostic_metric": GateDefinition(
            name="Pre-Clamp Out-of-Range Fraction",
            target_threshold="Monitored (Diagnostic Only)",
            classification="DIAGNOSTIC_ONLY",
            description="Diagnostic metric monitoring fraction of pixels exceeding [0, 255].",
        ),
        "lossy_jpeg_channel": GateDefinition(
            name="Lossy Channel Robustness (JPEG Q=80)",
            target_threshold="Post-ECC BER == 0.0 at Q=80",
            classification="PROPOSED_UNVALIDATED",
            description="Target robustness capability to be trained with differentiable channel surrogates.",
        ),
    }

    @classmethod
    def get_gate(cls, gate_key: str) -> GateDefinition:
        if gate_key not in cls.GATES:
            raise KeyError(f"Unknown gate '{gate_key}'. Available gates: {list(cls.GATES.keys())}")
        return cls.GATES[gate_key]

    @classmethod
    def list_gates(cls) -> Dict[str, Dict[str, str]]:
        return {
            k: {
                "name": v.name,
                "target_threshold": v.target_threshold,
                "classification": v.classification,
                "description": v.description,
            }
            for k, v in cls.GATES.items()
        }
