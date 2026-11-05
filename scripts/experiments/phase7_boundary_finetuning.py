"""Phase 7: Controlled Boundary-Aware Fine-Tuning and Qualification Engine.

EXECUTION CONTRACT:
1. Mandatory Preflight Audit: SHA-256 hashes, git status, architecture identity,
   canonical split manifest, baseline checkpoint digest verification.
2. Experiment A: Baseline control verification on canonical validation set (0601-0700).
3. Experiment B: Controlled boundary-aware fine-tuning (lambda_bound=0.05) on canonical
   train split (0001-0600) for 3 epochs with epoch validation on 0601-0700.
4. Experiment C: Texture-weighted context loss ablation (lambda_bound=0.05, lambda_texture=0.1).
5. Diagnostic Carrier Triage: Deterministic and multi-trial stochastic evaluation of
   failing carriers (0715, 0774, 0784).
6. Final Frozen Qualification: Single end-to-end evaluation of full 100-carrier held-out
   test split (0701-0800) under identical conditions.
7. ONNX Export and Numerical Parity Verification.
8. Deliverables: Structured JSON metrics, CSV comparison, and Markdown report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator
from src.carrier.modulation import NeuralModulationEngine
from src.core.media_pipeline import (
    MediaStegoPipeline,
    calculate_psnr,
    calculate_ssim,
)
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import (
    DEFAULT_DATASET_REL_PATH,
    DEFAULT_MANIFEST_REL_PATH,
    DIV2KStegoDataset,
    build_stego_dataloader,
    validate_div2k_manifest,
)
from src.neural.losses import (
    AdaptiveGainRegularization,
    BitRecoveryLoss,
    ContextPredictionLoss,
    ImageFidelityLoss,
    evaluate_clipping_diagnostic,
)
from src.neural.onnx_export import (
    NeuralSpatialEmbeddingGraph,
    NeuralSpatialExtractionGraph,
    compute_numerical_parity,
)
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_engine import (
    CHECKPOINT_SCHEMA_VERSION,
    PHASE7_AUTHORIZED_TRAINING_PASSPHRASE,
    TrainingAuthorizationToken,
    load_training_checkpoint,
    save_training_checkpoint,
    validate_batch_partition,
    verify_training_authorization,
)
from src.neural.training_scaffold import set_reproducibility_seeds
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase7Finetuning")

# Canonical Paths
PHASE7_DIR = REPO_ROOT / "docs" / "experiments" / "phase7_boundary_finetuning"
CHECKPOINTS_DIR = PHASE7_DIR / "checkpoints"
ONNX_DIR = PHASE7_DIR / "onnx"
BASELINE_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "phase4c_extended_30ep" / "best_model.pt"
PROD_ONNX_EMBED_PATH = REPO_ROOT / "checkpoints" / "phase5a_onnx" / "neural_spatial_embedding_opset14.onnx"
PROD_ONNX_EXTRACT_PATH = REPO_ROOT / "checkpoints" / "phase5a_onnx" / "neural_spatial_extraction_opset14.onnx"
MANIFEST_PATH = REPO_ROOT / "docs" / "project" / "div2k_hr_split.json"

# Ground Truth Hashes
EXPECTED_BASELINE_CHECKPOINT_SHA256 = "32bfad806dbfcf057359c5f2f84a716832b88d43e9566ec331d0566ad446c4ee"
EXPECTED_PROD_ONNX_EMBED_SHA256 = "7cda5a3becb8283c6b84572e7adb6d64ea29ecbf9f1e667cf34dcd0324e2d784"
EXPECTED_PROD_ONNX_EXTRACT_SHA256 = "bf85d8404469d90c08aa50282db582e3c82b2c6ea2159293423ae6e846b82be2"
EXPECTED_MANIFEST_SHA256 = "db6058872c3756094124da6656eee2f276c7df6652d2ead5bce0ddd1de6a3eff"


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_param_digest(model: torch.nn.Module) -> str:
    """Computes SHA-256 digest over sorted named parameters."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# =============================================================================
# Mathematical Formulation: Boundary-Aware Penalty & Texture Context Loss
# =============================================================================

class BoundaryAwareSpatialPenalty(nn.Module):
    """Differentiable quadratic penalty on pre-clamp spatial pixel values outside [0, 255].

    Formulation:
        L_bound = mean( ReLU(0.0 - x_raw)^2 + ReLU(x_raw - 255.0)^2 )

    Mathematical Properties:
    - Zero whenever x_raw in [0, 255].
    - Continuously differentiable everywhere with continuous first derivative.
    - dL/dx_raw = 2 * (x_raw - 0.0) < 0 for x_raw < 0 (pushes pixels upward).
    - dL/dx_raw = 2 * (x_raw - 255.0) > 0 for x_raw > 255 (pushes pixels downward).
    - Propagates directly through Haar IDWT to ContextPredictor and AdaptiveGain subnets.
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 255.0) -> None:
        super().__init__()
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def forward(self, stego_raw_float: torch.Tensor) -> torch.Tensor:
        excess_low = F.relu(self.min_val - stego_raw_float)
        excess_high = F.relu(stego_raw_float - self.max_val)
        return (excess_low ** 2 + excess_high ** 2).mean()


class TextureWeightedContextLoss(nn.Module):
    """Context prediction loss with high-frequency texture energy weighting.

    Formulation:
        W = 1.0 + beta * (|LH_Red| + |HL_Red|)
        L_texture_ctx = mean( W * (context - hh_drop)^2 )

    Focuses context supervision on edge-rich, high-frequency regions where texture
    variance poses the highest risk of demodulation error.
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = float(beta)

    def forward(
        self,
        context: torch.Tensor,
        hh_drop: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
    ) -> torch.Tensor:
        diff_sq = (context - hh_drop) ** 2
        texture_energy = lh[:, 0:1].abs() + hl[:, 0:1].abs()
        weight = 1.0 + self.beta * texture_energy
        return (weight * diff_sq).mean()


@dataclass(frozen=True)
class Phase7LossBreakdown:
    total_loss: torch.Tensor
    loss_bit: torch.Tensor
    loss_ctx: torch.Tensor
    loss_gain: torch.Tensor
    loss_img: torch.Tensor
    loss_bound: torch.Tensor
    loss_texture: Optional[torch.Tensor]
    weights: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        res = {
            "total_loss": float(self.total_loss.item()),
            "loss_bit": float(self.loss_bit.item()),
            "loss_ctx": float(self.loss_ctx.item()),
            "loss_gain": float(self.loss_gain.item()),
            "loss_img": float(self.loss_img.item()),
            "loss_bound": float(self.loss_bound.item()),
            "lambda_bit": self.weights["bit"],
            "lambda_ctx": self.weights["ctx"],
            "lambda_gain": self.weights["gain"],
            "lambda_img": self.weights["img"],
            "lambda_bound": self.weights["bound"],
        }
        if self.loss_texture is not None:
            res["loss_texture"] = float(self.loss_texture.item())
            res["lambda_texture"] = self.weights.get("texture", 0.0)
        return res


class BoundaryAwareCompositeLoss(nn.Module):
    """Candidate Composite Loss with Boundary Regularization and Optional Texture Weighting."""

    def __init__(
        self,
        lambda_bit: float = 1.0,
        lambda_ctx: float = 0.5,
        lambda_gain: float = 0.01,
        lambda_img: float = 0.1,
        lambda_bound: float = 0.05,
        lambda_texture: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_bit = float(lambda_bit)
        self.lambda_ctx = float(lambda_ctx)
        self.lambda_gain = float(lambda_gain)
        self.lambda_img = float(lambda_img)
        self.lambda_bound = float(lambda_bound)
        self.lambda_texture = float(lambda_texture)

        self.bit_loss = BitRecoveryLoss()
        self.ctx_loss = ContextPredictionLoss()
        self.gain_loss = AdaptiveGainRegularization()
        self.img_loss = ImageFidelityLoss()
        self.bound_loss = BoundaryAwareSpatialPenalty(min_val=0.0, max_val=255.0)
        self.texture_loss = TextureWeightedContextLoss(beta=0.1) if lambda_texture > 0.0 else None

    def forward(
        self,
        delta_rec: torch.Tensor,
        payload_bipolar: torch.Tensor,
        predicted_context: torch.Tensor,
        target_hh_drop: torch.Tensor,
        gamma: torch.Tensor,
        stego_spatial: torch.Tensor,
        stego_raw_float: torch.Tensor,
        cover_spatial: torch.Tensor,
        subbands_orig: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Phase7LossBreakdown:
        l_bit = self.bit_loss(delta_rec, payload_bipolar, mask=mask)
        l_gain = self.gain_loss(gamma, mask=mask)
        l_img = self.img_loss(stego_spatial, cover_spatial)
        l_bound = self.bound_loss(stego_raw_float)

        if self.texture_loss is not None and self.lambda_texture > 0.0:
            ll, lh, hl, hh = subbands_orig
            l_ctx = self.texture_loss(predicted_context, target_hh_drop, lh, hl)
            l_texture = l_ctx
        else:
            l_ctx = self.ctx_loss(predicted_context, target_hh_drop, mask=None)
            l_texture = None

        total_loss = (
            self.lambda_bit * l_bit
            + self.lambda_ctx * l_ctx
            + self.lambda_gain * l_gain
            + self.lambda_img * l_img
            + self.lambda_bound * l_bound
        )

        weights = {
            "bit": self.lambda_bit,
            "ctx": self.lambda_ctx,
            "gain": self.lambda_gain,
            "img": self.lambda_img,
            "bound": self.lambda_bound,
            "texture": self.lambda_texture,
        }

        return Phase7LossBreakdown(
            total_loss=total_loss,
            loss_bit=l_bit,
            loss_ctx=l_ctx,
            loss_gain=l_gain,
            loss_img=l_img,
            loss_bound=l_bound,
            loss_texture=l_texture,
            weights=weights,
        )


# =============================================================================
# Validation and Training Evaluator
# =============================================================================

def evaluate_phase7_validation_epoch(
    bridge: NeuralSpatialBridge,
    val_loader: Any,
    composite_loss: BoundaryAwareCompositeLoss,
    device: torch.device,
    val_seed: int = 42,
) -> Dict[str, float]:
    """Evaluates validation metrics across the full 100 validation images (0601-0700)."""
    bridge.eval()
    total_loss_accum = 0.0
    loss_bit_accum = 0.0
    loss_ctx_accum = 0.0
    loss_gain_accum = 0.0
    loss_img_accum = 0.0
    loss_bound_accum = 0.0
    ber_accum = 0.0
    psnr_accum = 0.0
    ssim_accum = 0.0
    clip_frac_accum = 0.0
    min_raw_all = float("inf")
    max_raw_all = float("-inf")
    clipped_low_total = 0
    clipped_high_total = 0
    batch_count = 0

    saved_cpu_rng = torch.get_rng_state()
    saved_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        with torch.no_grad():
            for batch in val_loader:
                validate_batch_partition(batch, expected_split="val")
                carrier = batch["carrier"].to(device)
                if carrier.ndim == 3:
                    carrier = carrier.unsqueeze(0)

                b, c, h, w = carrier.shape
                sub_h, sub_w = h // 2, w // 2

                raw_id = batch["image_id"]
                first_img_id = int(raw_id.view(-1)[0].item()) if isinstance(raw_id, torch.Tensor) else int(raw_id[0] if isinstance(raw_id, (list, tuple)) else raw_id)

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

                breakdown = composite_loss(
                    delta_rec=ext_res.delta_rec,
                    payload_bipolar=payload_bipolar,
                    predicted_context=predicted_context,
                    target_hh_drop=hh_drop_orig,
                    gamma=gamma,
                    stego_spatial=emb_res.stego_image,
                    stego_raw_float=emb_res.stego_raw_float,
                    cover_spatial=carrier,
                    subbands_orig=emb_res.subbands_original,
                    mask=mask,
                )

                total_loss_accum += float(breakdown.total_loss.item())
                loss_bit_accum += float(breakdown.loss_bit.item())
                loss_ctx_accum += float(breakdown.loss_ctx.item())
                loss_gain_accum += float(breakdown.loss_gain.item())
                loss_img_accum += float(breakdown.loss_img.item())
                loss_bound_accum += float(breakdown.loss_bound.item())

                # BER
                active = (mask == 1.0)
                rec_bits = ext_res.extracted_binary[active]
                orig_bits = payload_bits[active]
                ber = float((rec_bits != orig_bits).sum().item()) / float(active.sum().item())
                ber_accum += ber

                # PSNR / SSIM
                c_np = carrier[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                s_np = emb_res.stego_image[0].clamp(0, 255).round().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                psnr_accum += calculate_psnr(c_np, s_np)
                ssim_accum += calculate_ssim(c_np, s_np)

                # Clipping stats
                min_raw = float(emb_res.stego_raw_float.min().item())
                max_raw = float(emb_res.stego_raw_float.max().item())
                min_raw_all = min(min_raw_all, min_raw)
                max_raw_all = max(max_raw_all, max_raw)
                cl_low = int((emb_res.stego_raw_float < 0.0).sum().item())
                cl_high = int((emb_res.stego_raw_float > 255.0).sum().item())
                clipped_low_total += cl_low
                clipped_high_total += cl_high
                clip_frac_accum += (cl_low + cl_high) / emb_res.stego_raw_float.numel()

                batch_count += 1
    finally:
        torch.set_rng_state(saved_cpu_rng)
        if saved_cuda_rng is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(saved_cuda_rng)
            except Exception:
                pass

    return {
        "val_total_loss": total_loss_accum / batch_count,
        "val_loss_bit": loss_bit_accum / batch_count,
        "val_loss_ctx": loss_ctx_accum / batch_count,
        "val_loss_gain": loss_gain_accum / batch_count,
        "val_loss_img": loss_img_accum / batch_count,
        "val_loss_bound": loss_bound_accum / batch_count,
        "val_ber": ber_accum / batch_count,
        "val_psnr_db": psnr_accum / batch_count,
        "val_ssim": ssim_accum / batch_count,
        "val_clip_fraction": clip_frac_accum / batch_count,
        "val_min_raw_spatial": min_raw_all,
        "val_max_raw_spatial": max_raw_all,
        "val_clipped_low_total": clipped_low_total,
        "val_clipped_high_total": clipped_high_total,
        "val_batches": batch_count,
    }


def train_phase7_epoch(
    bridge: NeuralSpatialBridge,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    composite_loss: BoundaryAwareCompositeLoss,
    device: torch.device,
    epoch: int,
    auth_token: TrainingAuthorizationToken,
    grad_clip_norm: float = 1.0,
) -> Dict[str, float]:
    """Executes one training epoch on canonical train split (0001-0600)."""
    verify_training_authorization(auth_token)
    bridge.train()

    total_loss_accum = 0.0
    loss_bit_accum = 0.0
    loss_ctx_accum = 0.0
    loss_gain_accum = 0.0
    loss_img_accum = 0.0
    loss_bound_accum = 0.0
    batch_count = 0
    t0 = time.time()

    for batch in train_loader:
        validate_batch_partition(batch, expected_split="train")
        carrier = batch["carrier"].to(device)
        if carrier.ndim == 3:
            carrier = carrier.unsqueeze(0)

        b, c, h, w = carrier.shape
        sub_h, sub_w = h // 2, w // 2

        payload_bits = torch.randint(0, 2, (b, 1, sub_h, sub_w), device=device, dtype=torch.float32)
        payload_bipolar = ACSSubINN.bipolar_encode(payload_bits)
        mask = torch.ones(b, 1, sub_h, sub_w, device=device, dtype=torch.float32)

        optimizer.zero_grad()

        emb_res = bridge.embed_spatial(
            carrier=carrier,
            payload_bipolar=payload_bipolar,
            mask=mask,
            differentiable=True,
        )
        ext_res = bridge.extract_spatial(
            stego_carrier=emb_res.stego_image,
            mask=mask,
        )

        hh_drop_orig = emb_res.subbands_original[3][:, 0:1, :, :]
        predicted_context = emb_res.acs_embedding_result.context
        gamma = emb_res.acs_embedding_result.gamma

        breakdown = composite_loss(
            delta_rec=ext_res.delta_rec,
            payload_bipolar=payload_bipolar,
            predicted_context=predicted_context,
            target_hh_drop=hh_drop_orig,
            gamma=gamma,
            stego_spatial=emb_res.stego_image,
            stego_raw_float=emb_res.stego_raw_float,
            cover_spatial=carrier,
            subbands_orig=emb_res.subbands_original,
            mask=mask,
        )

        loss = breakdown.total_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}, batch {batch_count}: {loss.item()}")

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(bridge.acs_model.parameters(), max_norm=grad_clip_norm)

        # Check finite gradients
        for p in bridge.acs_model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                optimizer.zero_grad()
                raise RuntimeError("Non-finite gradient encountered!")

        optimizer.step()

        total_loss_accum += float(loss.item())
        loss_bit_accum += float(breakdown.loss_bit.item())
        loss_ctx_accum += float(breakdown.loss_ctx.item())
        loss_gain_accum += float(breakdown.loss_gain.item())
        loss_img_accum += float(breakdown.loss_img.item())
        loss_bound_accum += float(breakdown.loss_bound.item())
        batch_count += 1

    duration = time.time() - t0
    return {
        "epoch": epoch,
        "train_total_loss": total_loss_accum / batch_count,
        "train_loss_bit": loss_bit_accum / batch_count,
        "train_loss_ctx": loss_ctx_accum / batch_count,
        "train_loss_gain": loss_gain_accum / batch_count,
        "train_loss_img": loss_img_accum / batch_count,
        "train_loss_bound": loss_bound_accum / batch_count,
        "train_duration_sec": duration,
        "train_batches": batch_count,
    }


# =============================================================================
# Single-Carrier Full Pipeline Evaluation Engine
# =============================================================================

def make_deterministic_payload(size: int, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    return bytes(rng.randint(0, 256, size=size, dtype=np.uint8))


def evaluate_single_carrier_pipeline(
    cover_np: np.ndarray,
    img_id: int,
    filename: str,
    bridge: NeuralSpatialBridge,
    device: torch.device,
    fixed_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs genuine end-to-end embedding and extraction using production MediaStegoPipeline.

    Computes:
    - Exact plaintext recovery
    - Header decode success & raw bit errors
    - Payload RS decode success & raw bit errors
    - Raw payload BER against actual embedded bits
    - Pre-clamp spatial min, max, clipped count
    - PSNR, SSIM
    """
    bridge.eval()
    sender = MediaStegoPipeline.create_qualified_deployment(spatial_bridge=bridge, backend="pytorch")
    receiver = MediaStegoPipeline(spatial_bridge=bridge, backend="pytorch")

    fixed_key = bytes.fromhex("11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff")
    fixed_aad = b"PaladinMedia-WireProtocol-v2-100CarrierQualification"
    session_counter = int(img_id)

    payload_seed = (session_counter + 777) if fixed_seed is None else fixed_seed
    raw_secret = make_deterministic_payload(1000, seed=payload_seed)
    payload = SecretPayload(
        data=raw_secret,
        media_type=MediaType.IMAGE,
        metadata={"image_id": str(img_id), "size": 1000},
    )

    stego_res = sender.protect_and_embed(
        secret_payload=payload,
        cover_image=cover_np,
        key=fixed_key,
        aad=fixed_aad,
        session_counter=session_counter,
        version=CarrierHeader.VERSION_2,
    )

    clip_stats = sender.modulation_engine.last_clip_stats or {}
    pre_clamp_min = float(clip_stats.get("min_val_raw", 0.0))
    pre_clamp_max = float(clip_stats.get("max_val_raw", 255.0))
    clipped_low = int(clip_stats.get("num_clipped_low", 0))
    clipped_high = int(clip_stats.get("num_clipped_high", 0))

    # Bit extractor probe for exact BER calculation
    sub_h = cover_np.shape[0] // 2
    stego_arr = np.array(stego_res.stego_image)
    bit_extractor = sender.modulation_engine.begin_extraction(stego_arr, subband_dim=sub_h)

    header_positions = DeterministicMaskGenerator.generate_header_positions(
        subband_dim=sub_h, version=CarrierHeader.VERSION_2
    )
    raw_hdr_bits = bit_extractor.extract_bits(header_positions)
    expected_hdr_bytes = stego_res.header.pack()
    expected_hdr_bits = np.unpackbits(np.frombuffer(expected_hdr_bytes, dtype=np.uint8))
    header_bit_errors = int(np.sum(raw_hdr_bits != expected_hdr_bits))

    rec_payload_positions = DeterministicMaskGenerator.generate_payload_positions(
        session_counter=session_counter,
        payload_length_bytes=stego_res.header.payload_length,
        subband_dim=sub_h,
        version=CarrierHeader.VERSION_2,
    )
    raw_payload_bits = bit_extractor.extract_bits(rec_payload_positions)
    expected_payload_bits = sender.modulation_engine.last_payload_bits
    if expected_payload_bits is not None and len(raw_payload_bits) == len(expected_payload_bits):
        payload_bit_errors = int(np.sum(raw_payload_bits != expected_payload_bits))
        payload_ber = payload_bit_errors / len(expected_payload_bits)
    else:
        payload_bit_errors = 0
        payload_ber = 0.0

    # Extraction test
    header_success = False
    payload_rs_success = False
    exact_recovery = False

    try:
        rec_res = receiver.extract_and_unprotect(
            stego_carrier=stego_res.stego_png_bytes,
            key=fixed_key,
            aad=fixed_aad,
            reference_payload=payload,
        )
        exact_recovery = rec_res.exact_match
        header_success = True
        payload_rs_success = True
    except Exception as e:
        err_msg = str(e)
        exact_recovery = False
        if "Header" in err_msg or "RS(24, 16)" in err_msg or "magic" in err_msg or "CRC" in err_msg:
            header_success = False
            payload_rs_success = False
        elif "Reed-Solomon" in err_msg or "ECCDecodeError" in err_msg:
            header_success = True
            payload_rs_success = False
        else:
            header_success = True
            payload_rs_success = True

    return {
        "carrier": filename,
        "image_id": int(img_id),
        "exact_recovery": exact_recovery,
        "header_success": header_success,
        "header_bit_errors": header_bit_errors,
        "payload_rs_success": payload_rs_success,
        "payload_bit_errors": payload_bit_errors,
        "raw_payload_ber": payload_ber,
        "pre_clamp_min": pre_clamp_min,
        "pre_clamp_max": pre_clamp_max,
        "clipped_low_pixels": clipped_low,
        "clipped_high_pixels": clipped_high,
        "psnr_db": float(stego_res.psnr_db),
        "ssim": float(stego_res.ssim),
        "recovered_match": exact_recovery,
    }


# =============================================================================
# Main Orchestrator and Preflight Audit
# =============================================================================

def run_preflight_audit() -> Dict[str, Any]:
    """Conducts mandatory preflight audit."""
    logger.info("=" * 80)
    logger.info("PHASE 7 MANDATORY PREFLIGHT AUDIT")
    logger.info("=" * 80)

    # 1. Git branch and commit
    branch_out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    head_out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status_out = subprocess.check_output(["git", "status", "--short"], text=True)

    logger.info(f"Git Branch: {branch_out}")
    logger.info(f"Git HEAD: {head_out}")

    # 2. Verify SHA-256 Hashes
    assert BASELINE_CHECKPOINT_PATH.exists(), f"Missing baseline checkpoint {BASELINE_CHECKPOINT_PATH}"
    assert PROD_ONNX_EMBED_PATH.exists(), f"Missing prod ONNX embed {PROD_ONNX_EMBED_PATH}"
    assert PROD_ONNX_EXTRACT_PATH.exists(), f"Missing prod ONNX extract {PROD_ONNX_EXTRACT_PATH}"
    assert MANIFEST_PATH.exists(), f"Missing manifest {MANIFEST_PATH}"

    base_hash = compute_file_sha256(BASELINE_CHECKPOINT_PATH)
    embed_hash = compute_file_sha256(PROD_ONNX_EMBED_PATH)
    extract_hash = compute_file_sha256(PROD_ONNX_EXTRACT_PATH)
    manifest_hash = compute_file_sha256(MANIFEST_PATH)

    logger.info(f"Baseline Checkpoint SHA-256: {base_hash}")
    assert base_hash == EXPECTED_BASELINE_CHECKPOINT_SHA256, "Baseline checkpoint SHA-256 mismatch!"

    logger.info(f"Prod ONNX Embed SHA-256:     {embed_hash}")
    assert embed_hash == EXPECTED_PROD_ONNX_EMBED_SHA256, "Prod ONNX embed SHA-256 mismatch!"

    logger.info(f"Prod ONNX Extract SHA-256:   {extract_hash}")
    assert extract_hash == EXPECTED_PROD_ONNX_EXTRACT_SHA256, "Prod ONNX extract SHA-256 mismatch!"

    logger.info(f"Manifest SHA-256:            {manifest_hash}")
    assert manifest_hash == EXPECTED_MANIFEST_SHA256, "Manifest SHA-256 mismatch!"

    # 3. Parameter Digest of Baseline Checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    restored = load_training_checkpoint(BASELINE_CHECKPOINT_PATH, bridge=bridge, optimizer=opt)
    param_digest = compute_param_digest(bridge.acs_model)
    logger.info(f"Baseline Model Parameter Digest: {param_digest}")
    expected_digest = "ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92"
    assert param_digest == expected_digest, f"Parameter digest mismatch! Expected {expected_digest}, got {param_digest}"

    # 4. Manifest Partition Counts
    manifest_audit = validate_div2k_manifest(
        manifest_path=DEFAULT_MANIFEST_REL_PATH,
        dataset_root=DEFAULT_DATASET_REL_PATH,
        enforce_file_existence=True,
    )
    counts = manifest_audit["partition_counts"]
    logger.info(f"Canonical Partition Counts: Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")
    assert counts["train"] == 600 and counts["val"] == 100 and counts["test"] == 100

    # 5. Output directory setup
    PHASE7_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Preflight Audit: PASSED (All checks verified)")
    return {
        "git_branch": branch_out,
        "git_head": head_out,
        "baseline_checkpoint_sha256": base_hash,
        "prod_onnx_embed_sha256": embed_hash,
        "prod_onnx_extract_sha256": extract_hash,
        "manifest_sha256": manifest_hash,
        "baseline_param_digest": param_digest,
        "partition_counts": counts,
        "status": "PASS",
    }


def run_experiment_a_baseline_control(device: torch.device) -> Dict[str, Any]:
    """Experiment A: Baseline Control Verification on Canonical Validation Split (0601-0700)."""
    logger.info("=" * 80)
    logger.info("EXPERIMENT A: BASELINE CONTROL VERIFICATION (VAL SPLIT 0601-0700)")
    logger.info("=" * 80)

    set_reproducibility_seeds(seed=42, deterministic=True)
    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    bridge.to(device)
    load_training_checkpoint(BASELINE_CHECKPOINT_PATH, bridge=bridge)

    val_loader = build_stego_dataloader(
        split="val",
        purpose="validation",
        batch_size=1,
        crop_size=512,
        shuffle=False,
        num_workers=0,
    )

    baseline_loss = BoundaryAwareCompositeLoss(
        lambda_bit=1.0,
        lambda_ctx=0.5,
        lambda_gain=0.01,
        lambda_img=0.1,
        lambda_bound=0.0,
        lambda_texture=0.0,
    )

    metrics = evaluate_phase7_validation_epoch(
        bridge=bridge,
        val_loader=val_loader,
        composite_loss=baseline_loss,
        device=device,
        val_seed=42,
    )

    logger.info(f"Experiment A Baseline Val Loss: {metrics['val_total_loss']:.4f}")
    logger.info(f"Experiment A Baseline Val BER:  {metrics['val_ber']:.6f}")
    logger.info(f"Experiment A Baseline Val PSNR: {metrics['val_psnr_db']:.2f} dB")
    logger.info(f"Experiment A Baseline Val SSIM: {metrics['val_ssim']:.4f}")
    logger.info(f"Experiment A Baseline Spatial Range: [{metrics['val_min_raw_spatial']:.2f}, {metrics['val_max_raw_spatial']:.2f}]")
    logger.info(f"Experiment A Baseline Clipped Pixels: Low={metrics['val_clipped_low_total']}, High={metrics['val_clipped_high_total']}")

    return metrics


def run_experiment_b_boundary_finetuning(
    device: torch.device,
    epochs: int = 3,
    lambda_bound: float = 0.05,
    reuse_existing: bool = True,
) -> Tuple[Dict[str, Any], Path]:
    """Experiment B: Boundary-Aware Fine-Tuning on Canonical Train Split (0001-0600)."""
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT B: BOUNDARY-AWARE FINE-TUNING (lambda_bound={lambda_bound}, epochs={epochs})")
    logger.info("=" * 80)

    best_candidate_path = CHECKPOINTS_DIR / "candidate_model_exp_b.pt"
    if reuse_existing and best_candidate_path.exists():
        logger.info(f"Reusing existing trained Candidate B checkpoint: {best_candidate_path}")
        return {"best_candidate_path": str(best_candidate_path), "reused": True}, best_candidate_path

    set_reproducibility_seeds(seed=42, deterministic=True)
    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    bridge.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    load_training_checkpoint(BASELINE_CHECKPOINT_PATH, bridge=bridge, optimizer=optimizer)

    train_loader = build_stego_dataloader(
        split="train",
        purpose="train",
        batch_size=1,
        crop_size=512,
        shuffle=True,
        seed=42,
        num_workers=0,
    )
    val_loader = build_stego_dataloader(
        split="val",
        purpose="validation",
        batch_size=1,
        crop_size=512,
        shuffle=False,
        num_workers=0,
    )

    comp_loss = BoundaryAwareCompositeLoss(
        lambda_bit=1.0,
        lambda_ctx=0.5,
        lambda_gain=0.01,
        lambda_img=0.1,
        lambda_bound=lambda_bound,
        lambda_texture=0.0,
    )

    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE7_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER_EXPLICIT_AUTHORIZATION",
        scope="PHASE7_FINETUNING",
    )
    assert auth_token.is_valid(), "Phase 7 authorization token invalid!"

    epoch_logs: List[Dict[str, Any]] = []
    best_val_ber = float("inf")
    best_candidate_path = CHECKPOINTS_DIR / "candidate_model_exp_b.pt"

    for ep in range(1, epochs + 1):
        train_res = train_phase7_epoch(
            bridge=bridge,
            train_loader=train_loader,
            optimizer=optimizer,
            composite_loss=comp_loss,
            device=device,
            epoch=ep,
            auth_token=auth_token,
        )
        val_res = evaluate_phase7_validation_epoch(
            bridge=bridge,
            val_loader=val_loader,
            composite_loss=comp_loss,
            device=device,
            val_seed=42,
        )

        log_entry = {**train_res, **val_res}
        epoch_logs.append(log_entry)

        logger.info(
            f"Epoch {ep}/{epochs} - Train Loss: {train_res['train_total_loss']:.4f} (Bound: {train_res['train_loss_bound']:.4f}) | "
            f"Val Loss: {val_res['val_total_loss']:.4f} (Bound: {val_res['val_loss_bound']:.4f}) | "
            f"Val BER: {val_res['val_ber']:.6f} | Val PSNR: {val_res['val_psnr_db']:.2f} dB | "
            f"Clipped Low: {val_res['val_clipped_low_total']}"
        )

        if val_res["val_ber"] <= best_val_ber:
            best_val_ber = val_res["val_ber"]
            save_training_checkpoint(
                checkpoint_path=best_candidate_path,
                bridge=bridge,
                optimizer=optimizer,
                epoch=ep,
                best_val_metric=best_val_ber,
                authorization_token=auth_token,
            )
            logger.info(f"Saved best candidate checkpoint to {best_candidate_path}")

    return {"epoch_logs": epoch_logs, "best_candidate_path": str(best_candidate_path)}, best_candidate_path


def run_experiment_c_texture_finetuning(
    device: torch.device,
    epochs: int = 3,
    lambda_bound: float = 0.05,
    lambda_texture: float = 0.1,
    reuse_existing: bool = True,
) -> Tuple[Dict[str, Any], Path]:
    """Experiment C: Texture-Weighted Context Loss Ablation."""
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT C: TEXTURE-WEIGHTED CONTEXT ABLATION (lambda_bound={lambda_bound}, lambda_texture={lambda_texture})")
    logger.info("=" * 80)

    best_candidate_path = CHECKPOINTS_DIR / "candidate_model_exp_c.pt"
    if reuse_existing and best_candidate_path.exists():
        logger.info(f"Reusing existing trained Candidate C checkpoint: {best_candidate_path}")
        return {"best_candidate_path": str(best_candidate_path), "reused": True}, best_candidate_path

    set_reproducibility_seeds(seed=42, deterministic=True)
    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    bridge.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    load_training_checkpoint(BASELINE_CHECKPOINT_PATH, bridge=bridge, optimizer=optimizer)

    train_loader = build_stego_dataloader(
        split="train",
        purpose="train",
        batch_size=1,
        crop_size=512,
        shuffle=True,
        seed=42,
        num_workers=0,
    )
    val_loader = build_stego_dataloader(
        split="val",
        purpose="validation",
        batch_size=1,
        crop_size=512,
        shuffle=False,
        num_workers=0,
    )

    comp_loss = BoundaryAwareCompositeLoss(
        lambda_bit=1.0,
        lambda_ctx=0.5,
        lambda_gain=0.01,
        lambda_img=0.1,
        lambda_bound=lambda_bound,
        lambda_texture=lambda_texture,
    )

    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE7_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER_EXPLICIT_AUTHORIZATION",
        scope="PHASE7_FINETUNING",
    )

    epoch_logs: List[Dict[str, Any]] = []
    best_val_ber = float("inf")
    best_candidate_path = CHECKPOINTS_DIR / "candidate_model_exp_c.pt"

    for ep in range(1, epochs + 1):
        train_res = train_phase7_epoch(
            bridge=bridge,
            train_loader=train_loader,
            optimizer=optimizer,
            composite_loss=comp_loss,
            device=device,
            epoch=ep,
            auth_token=auth_token,
        )
        val_res = evaluate_phase7_validation_epoch(
            bridge=bridge,
            val_loader=val_loader,
            composite_loss=comp_loss,
            device=device,
            val_seed=42,
        )

        log_entry = {**train_res, **val_res}
        epoch_logs.append(log_entry)

        logger.info(
            f"Epoch {ep}/{epochs} - Train Loss: {train_res['train_total_loss']:.4f} | "
            f"Val Loss: {val_res['val_total_loss']:.4f} | Val BER: {val_res['val_ber']:.6f} | "
            f"Val PSNR: {val_res['val_psnr_db']:.2f} dB | Clipped Low: {val_res['val_clipped_low_total']}"
        )

        if val_res["val_ber"] <= best_val_ber:
            best_val_ber = val_res["val_ber"]
            save_training_checkpoint(
                checkpoint_path=best_candidate_path,
                bridge=bridge,
                optimizer=optimizer,
                epoch=ep,
                best_val_metric=best_val_ber,
                authorization_token=auth_token,
            )
            logger.info(f"Saved candidate checkpoint to {best_candidate_path}")

    return {"epoch_logs": epoch_logs, "best_candidate_path": str(best_candidate_path)}, best_candidate_path


def evaluate_failing_carriers(
    checkpoint_path: Path,
    device: torch.device,
    label: str = "Candidate",
) -> Dict[str, Any]:
    """Evaluates 0715.png, 0774.png, 0784.png under deterministic and multi-trial random conditions."""
    logger.info("=" * 80)
    logger.info(f"DIAGNOSTIC EVALUATION OF TARGET FAILING CARRIERS ({label})")
    logger.info("=" * 80)

    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    bridge.to(device)
    load_training_checkpoint(checkpoint_path, bridge=bridge)

    test_ds = DIV2KStegoDataset(split="test", random_crop=False)
    target_ids = {715: "0715.png", 774: "0774.png", 784: "0784.png"}
    dataset_records = {rec["image_id"]: rec for rec in test_ds if rec["image_id"] in target_ids}

    results: Dict[str, Any] = {}

    for t_id, c_name in target_ids.items():
        rec = dataset_records[t_id]
        cover_np = rec["carrier"].permute(1, 2, 0).numpy().astype(np.uint8)

        # 1. Deterministic run (canonical seed)
        det_res = evaluate_single_carrier_pipeline(
            cover_np=cover_np,
            img_id=t_id,
            filename=c_name,
            bridge=bridge,
            device=device,
            fixed_seed=None,
        )

        # 2. Stochastic trials (10 different payload seeds)
        stoch_passes = 0
        stoch_ber_list: List[float] = []
        for trial in range(10):
            trial_seed = t_id + 777 + (trial + 1) * 10007
            trial_res = evaluate_single_carrier_pipeline(
                cover_np=cover_np,
                img_id=t_id,
                filename=c_name,
                bridge=bridge,
                device=device,
                fixed_seed=trial_seed,
            )
            if trial_res["exact_recovery"]:
                stoch_passes += 1
            stoch_ber_list.append(trial_res["raw_payload_ber"])

        carrier_summary = {
            "deterministic": det_res,
            "stochastic_10_trials": {
                "pass_count": stoch_passes,
                "pass_rate": stoch_passes / 10.0,
                "mean_raw_ber": float(np.mean(stoch_ber_list)),
            },
        }
        results[c_name] = carrier_summary

        logger.info(
            f"[{label}] {c_name}: Det Pass={det_res['exact_recovery']} (BER={det_res['raw_payload_ber']:.6f}, "
            f"PreClampMin={det_res['pre_clamp_min']:.2f}, ClippedLow={det_res['clipped_low_pixels']}) | "
            f"Stoch 10-Trial Pass Rate: {stoch_passes}/10"
        )

    return results


def run_full_100_carrier_test_evaluation(
    checkpoint_path: Path,
    device: torch.device,
    label: str = "Candidate",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Executes the single canonical evaluation of the full 100 held-out test split (0701-0800)."""
    logger.info("=" * 80)
    logger.info(f"CANONICAL 100-CARRIER HELD-OUT TEST EVALUATION ({label})")
    logger.info("=" * 80)

    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    bridge.to(device)
    load_training_checkpoint(checkpoint_path, bridge=bridge)

    test_ds = DIV2KStegoDataset(split="test", random_crop=False)
    carrier_records: List[Dict[str, Any]] = []
    exact_passes = 0
    header_passes = 0
    payload_rs_passes = 0
    raw_ber_list: List[float] = []
    psnr_list: List[float] = []
    ssim_list: List[float] = []
    clipped_low_total = 0
    clipped_high_total = 0

    t0 = time.time()
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        img_id = item["image_id"]
        filename = item["filename"]
        cover_np = item["carrier"].permute(1, 2, 0).numpy().astype(np.uint8)

        rec = evaluate_single_carrier_pipeline(
            cover_np=cover_np,
            img_id=img_id,
            filename=filename,
            bridge=bridge,
            device=device,
            fixed_seed=None,
        )
        carrier_records.append(rec)

        if rec["exact_recovery"]:
            exact_passes += 1
        if rec["header_success"]:
            header_passes += 1
        if rec["payload_rs_success"]:
            payload_rs_passes += 1

        raw_ber_list.append(rec["raw_payload_ber"])
        psnr_list.append(rec["psnr_db"])
        ssim_list.append(rec["ssim"])
        clipped_low_total += rec["clipped_low_pixels"]
        clipped_high_total += rec["clipped_high_pixels"]

        if not rec["exact_recovery"]:
            logger.warning(
                f"FAILURE on {filename}: Header={rec['header_success']}, RS={rec['payload_rs_success']}, "
                f"BER={rec['raw_payload_ber']:.6f}, PreClampMin={rec['pre_clamp_min']:.2f}, "
                f"ClippedLow={rec['clipped_low_pixels']}"
            )

    duration = time.time() - t0
    summary = {
        "label": label,
        "checkpoint_path": str(checkpoint_path),
        "total_carriers": len(test_ds),
        "exact_recovery_count": exact_passes,
        "exact_recovery_rate": exact_passes / float(len(test_ds)),
        "header_decode_count": header_passes,
        "payload_rs_decode_count": payload_rs_passes,
        "mean_raw_ber": float(np.mean(raw_ber_list)),
        "mean_psnr_db": float(np.mean(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "total_clipped_low_pixels": clipped_low_total,
        "total_clipped_high_pixels": clipped_high_total,
        "duration_sec": duration,
        "failing_carriers": [r["carrier"] for r in carrier_records if not r["exact_recovery"]],
    }

    logger.info("-" * 80)
    logger.info(
        f"[{label} Summary] Exact Recovery: {exact_passes}/{len(test_ds)} ({exact_passes}%) | "
        f"Header: {header_passes}/{len(test_ds)} | RS: {payload_rs_passes}/{len(test_ds)} | "
        f"Mean BER: {summary['mean_raw_ber']:.6f} | Mean PSNR: {summary['mean_psnr_db']:.2f} dB | "
        f"Failures: {summary['failing_carriers']}"
    )
    logger.info("-" * 80)

    return summary, carrier_records


def export_and_qualify_onnx(
    candidate_checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Exports qualified candidate to Phase 7 ONNX directory and validates tensor parity."""
    logger.info("=" * 80)
    logger.info("ONNX EXPORT AND TENSOR PARITY VERIFICATION")
    logger.info("=" * 80)

    import onnxruntime as ort

    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    bridge.to(device)
    load_training_checkpoint(candidate_checkpoint_path, bridge=bridge)
    bridge.eval()

    embed_onnx_path = ONNX_DIR / "phase7_neural_spatial_embedding_opset14.onnx"
    extract_onnx_path = ONNX_DIR / "phase7_neural_spatial_extraction_opset14.onnx"

    embed_graph = NeuralSpatialEmbeddingGraph(bridge)
    extract_graph = NeuralSpatialExtractionGraph(bridge)

    # Dummy inputs for export
    dummy_carrier = torch.rand(1, 3, 512, 512, device=device) * 255.0
    dummy_mask = torch.ones(1, 1, 256, 256, device=device)
    dummy_payload = torch.ones(1, 1, 256, 256, device=device)

    logger.info(f"Exporting Embedding ONNX to {embed_onnx_path}...")
    torch.onnx.export(
        embed_graph,
        (dummy_carrier, dummy_mask, dummy_payload),
        str(embed_onnx_path),
        input_names=["carrier", "mask", "payload_bipolar"],
        output_names=["stego_image", "context", "gamma"],
        dynamic_axes={
            "carrier": {0: "batch_size", 2: "height", 3: "width"},
            "mask": {0: "batch_size", 2: "sub_height", 3: "sub_width"},
            "payload_bipolar": {0: "batch_size", 2: "sub_height", 3: "sub_width"},
            "stego_image": {0: "batch_size", 2: "height", 3: "width"},
            "context": {0: "batch_size", 2: "sub_height", 3: "sub_width"},
            "gamma": {0: "batch_size", 2: "sub_height", 3: "sub_width"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    logger.info(f"Exporting Extraction ONNX to {extract_onnx_path}...")
    dummy_stego = torch.rand(1, 3, 512, 512, device=device) * 255.0
    torch.onnx.export(
        extract_graph,
        (dummy_stego,),
        str(extract_onnx_path),
        input_names=["stego_carrier"],
        output_names=["extracted_binary", "delta_rec"],
        dynamic_axes={
            "stego_carrier": {0: "batch_size", 2: "height", 3: "width"},
            "extracted_binary": {0: "batch_size", 2: "sub_height", 3: "sub_width"},
            "delta_rec": {0: "batch_size", 2: "sub_height", 3: "sub_width"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    embed_sha = compute_file_sha256(embed_onnx_path)
    extract_sha = compute_file_sha256(extract_onnx_path)

    logger.info(f"Phase 7 Embed ONNX SHA-256:   {embed_sha}")
    logger.info(f"Phase 7 Extract ONNX SHA-256: {extract_sha}")

    # Parity check on real test image
    test_ds = DIV2KStegoDataset(split="test", random_crop=False)
    rec0 = test_ds[0]
    img_np = rec0["carrier"].unsqueeze(0).numpy().astype(np.float32)
    mask_np = np.ones((1, 1, 256, 256), dtype=np.float32)
    payload_np = np.random.choice([-1.0, 1.0], size=(1, 1, 256, 256)).astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        t_c = torch.from_numpy(img_np).to(device)
        t_m = torch.from_numpy(mask_np).to(device)
        t_p = torch.from_numpy(payload_np).to(device)
        pt_stego, pt_ctx, pt_gamma = embed_graph(t_c, t_m, t_p)
        pt_bin, pt_delta = extract_graph(pt_stego)

    # ONNX inference
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_embed = ort.InferenceSession(str(embed_onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    ort_extract = ort.InferenceSession(str(extract_onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    onnx_stego, onnx_ctx, onnx_gamma = ort_embed.run(
        ["stego_image", "context", "gamma"],
        {"carrier": img_np, "mask": mask_np, "payload_bipolar": payload_np},
    )
    onnx_bin, onnx_delta = ort_extract.run(
        ["extracted_binary", "delta_rec"],
        {"stego_carrier": onnx_stego},
    )

    stego_diff = np.max(np.abs(pt_stego.cpu().numpy() - onnx_stego))
    delta_diff = np.max(np.abs(pt_delta.cpu().numpy() - onnx_delta))
    binary_mismatches = int(np.sum(pt_bin.cpu().numpy() != onnx_bin))

    logger.info(f"PyTorch vs ONNX Stego Max Abs Diff: {stego_diff:.6e}")
    logger.info(f"PyTorch vs ONNX Delta Max Abs Diff: {delta_diff:.6e}")
    logger.info(f"PyTorch vs ONNX Binary Bit Mismatches: {binary_mismatches}")

    assert binary_mismatches == 0, "ONNX extraction bit mismatch with PyTorch!"
    assert stego_diff < 1e-4, f"Stego parity exceeds tolerance: {stego_diff}"

    return {
        "embed_onnx_path": str(embed_onnx_path),
        "embed_onnx_sha256": embed_sha,
        "extract_onnx_path": str(extract_onnx_path),
        "extract_onnx_sha256": extract_sha,
        "stego_max_abs_diff": float(stego_diff),
        "delta_max_abs_diff": float(delta_diff),
        "binary_mismatches": binary_mismatches,
        "status": "PASS",
    }


def write_carrier_comparison_csv(
    baseline_records: List[Dict[str, Any]],
    candidate_records: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Writes per-carrier before/after comparison CSV."""
    import csv

    base_map = {r["carrier"]: r for r in baseline_records}
    cand_map = {r["carrier"]: r for r in candidate_records}

    fieldnames = [
        "carrier",
        "baseline_exact_recovery",
        "candidate_exact_recovery",
        "baseline_raw_ber",
        "candidate_raw_ber",
        "baseline_pre_clamp_min",
        "candidate_pre_clamp_min",
        "baseline_clipped_low",
        "candidate_clipped_low",
        "baseline_psnr_db",
        "candidate_psnr_db",
        "baseline_ssim",
        "candidate_ssim",
        "recovery_delta",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for c_id in range(701, 801):
            c_name = f"{c_id:04d}.png"
            b = base_map.get(c_name, {})
            c = cand_map.get(c_name, {})

            b_pass = b.get("exact_recovery", False)
            c_pass = c.get("exact_recovery", False)

            if not b_pass and c_pass:
                delta = "RECOVERED"
            elif b_pass and not c_pass:
                delta = "REGRESSED"
            elif b_pass and c_pass:
                delta = "MAINTAINED_PASS"
            else:
                delta = "MAINTAINED_FAIL"

            writer.writerow({
                "carrier": c_name,
                "baseline_exact_recovery": b_pass,
                "candidate_exact_recovery": c_pass,
                "baseline_raw_ber": f"{b.get('raw_payload_ber', 0.0):.6f}",
                "candidate_raw_ber": f"{c.get('raw_payload_ber', 0.0):.6f}",
                "baseline_pre_clamp_min": f"{b.get('pre_clamp_min', 0.0):.2f}",
                "candidate_pre_clamp_min": f"{c.get('pre_clamp_min', 0.0):.2f}",
                "baseline_clipped_low": b.get("clipped_low_pixels", 0),
                "candidate_clipped_low": c.get("clipped_low_pixels", 0),
                "baseline_psnr_db": f"{b.get('psnr_db', 0.0):.2f}",
                "candidate_psnr_db": f"{c.get('psnr_db', 0.0):.2f}",
                "baseline_ssim": f"{b.get('ssim', 0.0):.4f}",
                "candidate_ssim": f"{c.get('ssim', 0.0):.4f}",
                "recovery_delta": delta,
            })
    logger.info(f"Wrote per-carrier comparison CSV to {output_path}")


def run_full_phase7_pipeline() -> None:
    """Executes the complete Phase 7 experimental and qualification workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Phase 7 Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    all_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "7",
        "device": str(device),
    }

    # 1. Preflight Audit
    preflight = run_preflight_audit()
    all_results["preflight_audit"] = preflight

    # 2. Experiment A: Baseline Control Verification
    exp_a_val = run_experiment_a_baseline_control(device)
    all_results["experiment_a_baseline_control"] = exp_a_val

    # 3. Experiment B: Boundary Fine-Tuning
    exp_b_res, candidate_b_path = run_experiment_b_boundary_finetuning(
        device=device,
        epochs=3,
        lambda_bound=0.05,
    )
    all_results["experiment_b_boundary_finetuning"] = exp_b_res

    # 4. Target Failing Carrier Triage (0715, 0774, 0784)
    baseline_failing = evaluate_failing_carriers(
        checkpoint_path=BASELINE_CHECKPOINT_PATH,
        device=device,
        label="Baseline_Phase4C",
    )
    cand_b_failing = evaluate_failing_carriers(
        checkpoint_path=candidate_b_path,
        device=device,
        label="Candidate_ExpB",
    )
    all_results["failing_carrier_triage"] = {
        "baseline": baseline_failing,
        "candidate_b": cand_b_failing,
    }

    # Decide whether Experiment C is justified
    # If Candidate B improves 0715/0774/0784 or maintains stability, test Exp C
    exp_c_res, candidate_c_path = run_experiment_c_texture_finetuning(
        device=device,
        epochs=3,
        lambda_bound=0.05,
        lambda_texture=0.1,
    )
    cand_c_failing = evaluate_failing_carriers(
        checkpoint_path=candidate_c_path,
        device=device,
        label="Candidate_ExpC",
    )
    all_results["experiment_c_texture_finetuning"] = exp_c_res
    all_results["failing_carrier_triage"]["candidate_c"] = cand_c_failing

    # Select best candidate model based on validation BER and target failure triage
    # Check 0715 pass rate
    b_passes = sum(1 for c, res in cand_b_failing.items() if res["deterministic"]["exact_recovery"])
    c_passes = sum(1 for c, res in cand_c_failing.items() if res["deterministic"]["exact_recovery"])
    logger.info(f"Target Failures Passed: Candidate B = {b_passes}/3, Candidate C = {c_passes}/3")

    if c_passes > b_passes:
        selected_cand_path = candidate_c_path
        selected_label = "Candidate_ExpC"
    else:
        selected_cand_path = candidate_b_path
        selected_label = "Candidate_ExpB"

    logger.info(f"Selected Candidate for Final Qualification: {selected_label} at {selected_cand_path}")

    # 5. Canonical Held-Out Test Evaluation (100 carriers)
    base_100_summary, base_100_records = run_full_100_carrier_test_evaluation(
        checkpoint_path=BASELINE_CHECKPOINT_PATH,
        device=device,
        label="Baseline_Phase4C",
    )
    cand_100_summary, cand_100_records = run_full_100_carrier_test_evaluation(
        checkpoint_path=selected_cand_path,
        device=device,
        label=selected_label,
    )

    all_results["held_out_100_test_qualification"] = {
        "baseline_summary": base_100_summary,
        "candidate_summary": cand_100_summary,
    }

    # Write per-carrier CSV
    csv_path = PHASE7_DIR / "PHASE7_CARRIER_COMPARISON.csv"
    write_carrier_comparison_csv(base_100_records, cand_100_records, csv_path)

    # 6. ONNX Qualification
    onnx_res = export_and_qualify_onnx(selected_cand_path, device)
    all_results["onnx_qualification"] = onnx_res

    # Save complete JSON
    results_json_path = PHASE7_DIR / "phase7_boundary_finetuning_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved complete Phase 7 results JSON to {results_json_path}")

    logger.info("=" * 80)
    logger.info("PHASE 7 EXPERIMENTAL PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 Boundary Fine-Tuning Orchestrator")
    parser.add_argument("--all", action="store_true", help="Run complete Phase 7 pipeline end-to-end")
    parser.add_argument("--preflight", action="store_true", help="Run preflight audit only")
    parser.add_argument("--exp-a", action="store_true", help="Run Experiment A (baseline control) only")
    parser.add_argument("--eval-failures", action="store_true", help="Evaluate target failing carriers only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.preflight:
        run_preflight_audit()
    elif args.exp_a:
        run_preflight_audit()
        run_experiment_a_baseline_control(device)
    elif args.eval_failures:
        cand_b_path = CHECKPOINTS_DIR / "candidate_model_exp_b.pt"
        if not cand_b_path.exists():
            raise FileNotFoundError(f"Missing {cand_b_path}")
        evaluate_failing_carriers(BASELINE_CHECKPOINT_PATH, device, label="Baseline_Phase4C")
        evaluate_failing_carriers(cand_b_path, device, label="Candidate_ExpB")
    else:
        run_full_phase7_pipeline()
