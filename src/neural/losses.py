"""Candidate Loss Functions and Composition for Phase 3 ACS-Sub-INN v3 Engine.

Implements differentiable loss functions for neural wavelet steganography training:
1. BitRecoveryLoss: Differentiable surrogate loss on received coefficient residuals.
2. ContextPredictionLoss: Direct supervision for ContextPredictorCNN using carrier coefficients.
3. AdaptiveGainRegularization: Penalty against unnecessarily large gain fields.
4. ImageFidelityLoss: Pixel-domain reconstruction loss through the differentiable STE bridge.
5. ClippingDiagnostic: Standalone diagnostic metric evaluating dynamic range clamping.
6. CompositeLoss: Configurable loss composition (MARKED AS PROPOSED_UNVALIDATED / NOT TRAINING-APPROVED).

THEORETICAL AND GRADIENT PATH SPECIFICATION:
- Bit Recovery: delta_rec = hh_drop_stego_rec - context_rec.
  Under lossless roundtrip, context_rec cancels with the context embedded in hh_drop_stego.
  Therefore, delta_rec gradients flow predominantly into AdaptiveGainCNN.
- Context Prediction: ContextPredictorCNN must be explicitly supervised via
  L_ctx = ||context - hh_drop|| to learn natural high-frequency inter-channel correlations.
  Secret payload content is NEVER a prediction target.
- Image Fidelity: Backpropagates through Haar IDWT and SpatialQuantizeSTE (identity gradient).
  Provides gradient pushback against large perturbations.
- Differentiable Surrogate vs. Reality: Differentiable training surrogates (STE) do not model
  discrete PNG container encoding, RS block-level thresholding, or AES-GCM GHASH authentication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Bit-Recovery Surrogate Loss
# =============================================================================

class BitRecoveryLoss(nn.Module):
    """Differentiable surrogate loss penalizing symbol demodulation errors.

    Bipolar convention:
        b in {0, 1} -> P = 2b - 1.0 in {-1.0, +1.0}
        Extractor decision rule: b_hat = 1 if delta_rec > 0 else 0 (P_hat = sign(delta_rec))

    Ideal margin:
        P * delta_rec > 0.

    Surrogate Formulations:
        1. 'logistic' (default):
           loss = Softplus(-P * delta_rec / temperature)
           Equivalently, Binary Cross-Entropy with Logits where logit = delta_rec / temperature.
           Continuous and non-zero gradient everywhere with exponential tail decay.
        2. 'hinge':
           loss = ReLU(margin - P * delta_rec)
           Linear gradient when margin is violated, zero gradient when satisfied.
    """

    def __init__(
        self,
        formulation: str = "logistic",
        temperature: float = 1.0,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        if formulation not in ("logistic", "bce", "hinge"):
            raise ValueError(f"Unsupported formulation '{formulation}'. Choose 'logistic' or 'hinge'.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive, got {temperature}")
        if margin <= 0.0:
            raise ValueError(f"margin must be strictly positive, got {margin}")

        self.formulation = "logistic" if formulation == "bce" else formulation
        self.temperature = float(temperature)
        self.margin = float(margin)

    def forward(
        self,
        delta_rec: torch.Tensor,
        payload_bipolar: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes masked bit-recovery loss.

        Args:
            delta_rec: Continuous extraction residual tensor (B, 1, H, W).
            payload_bipolar: Transmitted bipolar symbols P in {-1.0, +1.0} of shape (B, 1, H, W).
            mask: Optional spatial mask tensor (B, 1, H, W) in [0, 1].

        Returns:
            Scalar tensor representing the mean loss over active mask positions.
        """
        if not isinstance(delta_rec, torch.Tensor) or not isinstance(payload_bipolar, torch.Tensor):
            raise TypeError("delta_rec and payload_bipolar must be torch.Tensor instances")
        if delta_rec.shape != payload_bipolar.shape:
            raise ValueError(f"Shape mismatch: delta_rec {delta_rec.shape} vs payload {payload_bipolar.shape}")

        if self.formulation == "logistic":
            # Softplus(-P * delta_rec / tau)
            elem_loss = F.softplus(-payload_bipolar * (delta_rec / self.temperature))
        elif self.formulation == "hinge":
            elem_loss = F.relu(self.margin - payload_bipolar * delta_rec)
        else:
            raise RuntimeError(f"Unknown formulation {self.formulation}")

        if mask is not None:
            mask_f = mask.to(delta_rec.dtype)
            active_elements = mask_f.sum()
            if active_elements > 0:
                loss = (elem_loss * mask_f).sum() / active_elements
            else:
                loss = elem_loss.sum() * 0.0
        else:
            loss = elem_loss.mean()

        return loss


# =============================================================================
# 2. Context-Prediction Loss
# =============================================================================

class ContextPredictionLoss(nn.Module):
    """Direct supervision loss for ContextPredictorCNN.

    Supervises context prediction from preserved Green/Blue-HH coefficients:
        target    = hh_drop (unmodulated Red-HH coefficients)
        predicted = ContextPredictorCNN(hh_keep)

    Formulations:
        1. 'smooth_l1' / 'huber' (default):
           Quadratic for errors < beta (stable near zero), linear for errors >= beta (outlier robust).
        2. 'l1':
           Standard mean absolute error.

    CRITICAL NOTE:
        The target is strictly unmodulated carrier coefficients (hh_drop).
        Secret payload content is NEVER a target or input for this loss.
    """

    def __init__(self, formulation: str = "smooth_l1", beta: float = 1.0) -> None:
        super().__init__()
        if formulation not in ("smooth_l1", "huber", "l1"):
            raise ValueError(f"Unsupported formulation '{formulation}'. Choose 'smooth_l1' or 'l1'.")
        if beta <= 0.0:
            raise ValueError(f"beta must be strictly positive, got {beta}")

        self.formulation = "smooth_l1" if formulation == "huber" else formulation
        self.beta = float(beta)

    def forward(
        self,
        predicted_context: torch.Tensor,
        target_hh_drop: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes context prediction loss.

        Args:
            predicted_context: Predicted Red-HH coefficients (B, 1, H, W).
            target_hh_drop: Ground-truth unmodulated Red-HH coefficients (B, 1, H, W).
            mask: Optional mask tensor (B, 1, H, W). If None, supervises across full subband.

        Returns:
            Scalar tensor loss.
        """
        if not isinstance(predicted_context, torch.Tensor) or not isinstance(target_hh_drop, torch.Tensor):
            raise TypeError("Inputs must be torch.Tensor instances")
        if predicted_context.shape != target_hh_drop.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted_context.shape} vs target {target_hh_drop.shape}"
            )

        if self.formulation == "smooth_l1":
            elem_loss = F.smooth_l1_loss(predicted_context, target_hh_drop, beta=self.beta, reduction="none")
        elif self.formulation == "l1":
            elem_loss = F.l1_loss(predicted_context, target_hh_drop, reduction="none")
        else:
            raise RuntimeError(f"Unknown formulation {self.formulation}")

        if mask is not None:
            mask_f = mask.to(predicted_context.dtype)
            active_elements = mask_f.sum()
            if active_elements > 0:
                return (elem_loss * mask_f).sum() / active_elements
            return elem_loss.sum() * 0.0

        return elem_loss.mean()


# =============================================================================
# 3. Adaptive-Gain Regularization
# =============================================================================

class AdaptiveGainRegularization(nn.Module):
    """Penalizes unnecessarily large modulation gains to preserve carrier fidelity.

    Since gamma >= gamma_min > 0 is enforced by design, the excess gain is:
        excess = ReLU(gamma - gamma_min)

    Formulations:
        1. 'excess_l1' (default): Mean excess gain above gamma_min.
        2. 'excess_l2': Mean squared excess gain above gamma_min.
        3. 'l1': Mean absolute gain.

    Trade-off:
        Excessive gain creates spatial pixel clipping and visual artifacts (low PSNR).
        Insufficient gain reduces the margin against rounding/quantization noise.
    """

    def __init__(
        self,
        formulation: str = "excess_l1",
        gamma_min: float = 1.0,
    ) -> None:
        super().__init__()
        if formulation not in ("excess_l1", "excess_l2", "l1"):
            raise ValueError(f"Unsupported formulation '{formulation}'. Choose 'excess_l1', 'excess_l2', or 'l1'.")
        if gamma_min <= 0.0:
            raise ValueError(f"gamma_min must be strictly positive, got {gamma_min}")

        self.formulation = formulation
        self.gamma_min = float(gamma_min)

    def forward(
        self,
        gamma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes gain regularization penalty.

        Args:
            gamma: Spatially varying gain tensor (B, 1, H, W).
            mask: Optional active embedding mask (B, 1, H, W).

        Returns:
            Scalar penalty tensor.
        """
        if not isinstance(gamma, torch.Tensor):
            raise TypeError("gamma must be a torch.Tensor")

        if self.formulation == "excess_l1":
            excess = F.relu(gamma - self.gamma_min)
            elem_loss = excess
        elif self.formulation == "excess_l2":
            excess = F.relu(gamma - self.gamma_min)
            elem_loss = excess ** 2
        elif self.formulation == "l1":
            elem_loss = gamma
        else:
            raise RuntimeError(f"Unknown formulation {self.formulation}")

        if mask is not None:
            mask_f = mask.to(gamma.dtype)
            active_elements = mask_f.sum()
            if active_elements > 0:
                return (elem_loss * mask_f).sum() / active_elements
            return elem_loss.sum() * 0.0

        return elem_loss.mean()


# =============================================================================
# 4. Spatial Image-Fidelity Loss
# =============================================================================

class ImageFidelityLoss(nn.Module):
    """Pixel-domain fidelity loss between cover and reconstructed stego images.

    Backpropagates through the differentiable 2D Haar IDWT and SpatialQuantizeSTE.

    Formulations:
        1. 'l1' (default): Mean Absolute Error in spatial domain.
        2. 'charbonnier': sqrt((x - y)^2 + eps^2). Smooth, differentiable approximation to L1.
        3. 'mse': Mean Squared Error (directly proportional to PSNR).

    Normalization:
        - normalize_255=False (default): Operates directly in [0, 255] pixel scale.
        - normalize_255=True: Rescales inputs to [0, 1] before computing loss.
    """

    def __init__(
        self,
        formulation: str = "l1",
        normalize_255: bool = False,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if formulation not in ("l1", "charbonnier", "mse"):
            raise ValueError(f"Unsupported formulation '{formulation}'. Choose 'l1', 'charbonnier', or 'mse'.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.formulation = formulation
        self.normalize_255 = bool(normalize_255)
        self.eps = float(eps)

    def forward(
        self,
        stego_spatial: torch.Tensor,
        cover_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """Computes spatial fidelity loss.

        Args:
            stego_spatial: Reconstructed stego image tensor (B, 3, H, W).
            cover_spatial: Original cover carrier image tensor (B, 3, H, W).

        Returns:
            Scalar fidelity loss.
        """
        if not isinstance(stego_spatial, torch.Tensor) or not isinstance(cover_spatial, torch.Tensor):
            raise TypeError("Inputs must be torch.Tensor instances")
        if stego_spatial.shape != cover_spatial.shape:
            raise ValueError(
                f"Shape mismatch: stego {stego_spatial.shape} vs cover {cover_spatial.shape}"
            )

        scale = 255.0 if self.normalize_255 else 1.0
        s = stego_spatial / scale
        c = cover_spatial / scale

        diff = s - c
        if self.formulation == "l1":
            return diff.abs().mean()
        elif self.formulation == "charbonnier":
            return torch.sqrt(diff ** 2 + (self.eps / scale) ** 2).mean()
        elif self.formulation == "mse":
            return (diff ** 2).mean()
        else:
            raise RuntimeError(f"Unknown formulation {self.formulation}")


# =============================================================================
# 5. Dynamic Range / Saturation Diagnostic
# =============================================================================

@dataclass(frozen=True)
class ClippingDiagnosticReport:
    """Non-trainable diagnostic metrics for pixel dynamic range clamping."""

    min_val_raw: float
    max_val_raw: float
    num_clipped_low: int
    num_clipped_high: int
    num_clipped_total: int
    clip_fraction: float
    boundary_pixel_count: int
    boundary_pixel_fraction: float
    saturation_penalty_l1: float


def evaluate_clipping_diagnostic(
    stego_raw_float: torch.Tensor,
    stego_quantized: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 255.0,
) -> ClippingDiagnosticReport:
    """Computes diagnostic statistics on pre-clamp and post-clamp stego images.

    DIAGNOSTIC ONLY:
        Does not contribute gradients to the optimization graph in Sprint 3.
        Used to detect saturation risks on high-contrast carriers.

    Args:
        stego_raw_float: Pre-clamp spatial reconstruction (B, 3, H, W).
        stego_quantized: Clamped and rounded spatial reconstruction (B, 3, H, W).
        min_val: Lower dynamic range boundary (default 0.0).
        max_val: Upper dynamic range boundary (default 255.0).

    Returns:
        ClippingDiagnosticReport dataclass.
    """
    with torch.no_grad():
        raw_detached = stego_raw_float.detach()
        quant_detached = stego_quantized.detach()

        min_raw = float(raw_detached.min().item())
        max_raw = float(raw_detached.max().item())

        clipped_low = int((raw_detached < min_val).sum().item())
        clipped_high = int((raw_detached > max_val).sum().item())
        total_clipped = clipped_low + clipped_high
        total_elements = raw_detached.numel()
        clip_frac = float(total_clipped / total_elements) if total_elements > 0 else 0.0

        # Post-clamp boundary pixels (pixels pinned exactly at 0 or 255)
        pinned_low = (quant_detached <= min_val).sum().item()
        pinned_high = (quant_detached >= max_val).sum().item()
        boundary_count = int(pinned_low + pinned_high)
        boundary_frac = float(boundary_count / total_elements) if total_elements > 0 else 0.0

        # Pre-clamp excess margin (saturation penalty magnitude)
        excess_low = F.relu(min_val - raw_detached)
        excess_high = F.relu(raw_detached - max_val)
        sat_penalty = float((excess_low + excess_high).mean().item())

    return ClippingDiagnosticReport(
        min_val_raw=min_raw,
        max_val_raw=max_raw,
        num_clipped_low=clipped_low,
        num_clipped_high=clipped_high,
        num_clipped_total=total_clipped,
        clip_fraction=clip_frac,
        boundary_pixel_count=boundary_count,
        boundary_pixel_fraction=boundary_frac,
        saturation_penalty_l1=sat_penalty,
    )


# =============================================================================
# 6. Composite Loss Module (PROPOSED_UNVALIDATED)
# =============================================================================

@dataclass(frozen=True)
class LossBreakdown:
    """Container for individual loss components and composite total."""

    total_loss: torch.Tensor
    loss_bit: torch.Tensor
    loss_ctx: torch.Tensor
    loss_gain: torch.Tensor
    loss_img: torch.Tensor
    weights: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_loss": float(self.total_loss.item()),
            "loss_bit": float(self.loss_bit.item()),
            "loss_ctx": float(self.loss_ctx.item()),
            "loss_gain": float(self.loss_gain.item()),
            "loss_img": float(self.loss_img.item()),
            "lambda_bit": self.weights["bit"],
            "lambda_ctx": self.weights["ctx"],
            "lambda_gain": self.weights["gain"],
            "lambda_img": self.weights["img"],
        }


class CompositeLoss(nn.Module):
    """Composes candidate loss functions with configurable weighting.

    STATUS: NOT TRAINING-APPROVED (Awaiting empirical hyperparameter calibration).

    Mathematical Formulation:
        L_total = lambda_bit * L_bit
                + lambda_ctx * L_ctx
                + lambda_gain * L_gain
                + lambda_img * L_img

    Default Uncalibrated Weights:
        lambda_bit  = 1.0  (primary objective: demodulation margin)
        lambda_ctx  = 0.5  (subband inter-channel prediction fidelity)
        lambda_gain = 0.01 (regularization penalty against runaway gain)
        lambda_img  = 0.1  (spatial image reconstruction fidelity)
    """

    STATUS: str = "PROPOSED_UNVALIDATED (NOT TRAINING-APPROVED)"

    def __init__(
        self,
        lambda_bit: float = 1.0,
        lambda_ctx: float = 0.5,
        lambda_gain: float = 0.01,
        lambda_img: float = 0.1,
        bit_loss_fn: Optional[BitRecoveryLoss] = None,
        ctx_loss_fn: Optional[ContextPredictionLoss] = None,
        gain_loss_fn: Optional[AdaptiveGainRegularization] = None,
        img_loss_fn: Optional[ImageFidelityLoss] = None,
    ) -> None:
        super().__init__()
        self.lambda_bit = float(lambda_bit)
        self.lambda_ctx = float(lambda_ctx)
        self.lambda_gain = float(lambda_gain)
        self.lambda_img = float(lambda_img)

        self.bit_loss = bit_loss_fn if bit_loss_fn is not None else BitRecoveryLoss()
        self.ctx_loss = ctx_loss_fn if ctx_loss_fn is not None else ContextPredictionLoss()
        self.gain_loss = gain_loss_fn if gain_loss_fn is not None else AdaptiveGainRegularization()
        self.img_loss = img_loss_fn if img_loss_fn is not None else ImageFidelityLoss()

    def forward(
        self,
        delta_rec: torch.Tensor,
        payload_bipolar: torch.Tensor,
        predicted_context: torch.Tensor,
        target_hh_drop: torch.Tensor,
        gamma: torch.Tensor,
        stego_spatial: torch.Tensor,
        cover_spatial: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> LossBreakdown:
        """Computes all individual loss terms and weighted sum.

        Args:
            delta_rec: Received extraction residual (B, 1, H, W).
            payload_bipolar: Embedded bipolar payload (B, 1, H, W).
            predicted_context: Context predicted from HH_keep (B, 1, H, W).
            target_hh_drop: Original unmodulated Red-HH coefficients (B, 1, H, W).
            gamma: Adaptive gain field (B, 1, H, W).
            stego_spatial: Reconstructed stego image (B, 3, 2H, 2W).
            cover_spatial: Original cover image (B, 3, 2H, 2W).
            mask: Optional active embedding mask (B, 1, H, W).

        Returns:
            LossBreakdown containing total loss and individual components.
        """
        l_bit = self.bit_loss(delta_rec, payload_bipolar, mask=mask)
        l_ctx = self.ctx_loss(predicted_context, target_hh_drop, mask=None)
        l_gain = self.gain_loss(gamma, mask=mask)
        l_img = self.img_loss(stego_spatial, cover_spatial)

        total_loss = (
            self.lambda_bit * l_bit
            + self.lambda_ctx * l_ctx
            + self.lambda_gain * l_gain
            + self.lambda_img * l_img
        )

        weights = {
            "bit": self.lambda_bit,
            "ctx": self.lambda_ctx,
            "gain": self.lambda_gain,
            "img": self.lambda_img,
        }

        return LossBreakdown(
            total_loss=total_loss,
            loss_bit=l_bit,
            loss_ctx=l_ctx,
            loss_gain=l_gain,
            loss_img=l_img,
            weights=weights,
        )
