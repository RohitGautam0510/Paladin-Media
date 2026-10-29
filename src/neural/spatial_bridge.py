"""Phase 3 Neural Wavelet Steganography Spatial Bridge (Sprint 2).

Connects spatial RGB carrier images to Haar wavelet coefficients, performs neural
embedding via ACSSubINN v3, reconstructs spatial stego images, and extracts payload
from reloaded lossless PNG containers.

DATAFLOW:
    RGB Carrier (B, 3, H, W)
        |
        v
    Haar DWT 2D (differentiable)
        |
        +---> Preserve LL, LH, HL
        +---> Partition HH:
                - Red-HH (channel 0, target): HH_drop (B, 1, H/2, W/2)
                - Green/Blue-HH (channels 1-2, context): HH_keep (B, 2, H/2, W/2)
        |
        v
    ACSSubINN Embed (HH_drop, HH_keep, mask, payload_bipolar)
        |
        v
    Reassemble HH' = [HH_drop', HH_keep]
        |
        v
    Haar IDWT 2D (differentiable)
        |
        v
    Spatial Quantization:
        - Differentiable path: SpatialQuantizeSTE (clamp [0, 255] and round with STE)
        - Inference path: clamp [0, 255], round to uint8, save/reload lossless PNG
        |
        v
    Haar DWT 2D
        |
        v
    Partition Received HH:
        - Red-HH received: HH_drop_rec
        - Green/Blue-HH received: HH_keep_rec
        |
        v
    ACSSubINN Extract (HH_drop_rec, HH_keep_rec)
        |
        v
    Extracted Bits & Residual Analysis
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.neural.acs_sub_inn import (
    ACSEmbeddingResult,
    ACSExtractionResult,
    ACSSubINN,
)
from src.neural.quantization import SpatialQuantizeSTE, spatial_quantize_ste

__all__ = [
    "HaarDWT2DTorch",
    "haar_dwt2d",
    "haar_idwt2d",
    "SpatialEmbeddingResult",
    "SpatialExtractionResult",
    "SpatialRoundtripMetrics",
    "NeuralSpatialBridge",
]


# =============================================================================
# 1. Differentiable 2D Haar DWT / IDWT Implementation
# =============================================================================

def haar_dwt2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes differentiable 1-level 2D Haar Discrete Wavelet Transform.

    Operates on (B, C, H, W) tensors. Normalization matches Phase 2 HaarDWT2D:
        LL = (a + b + c + d) / 2.0
        LH = (a - b + c - d) / 2.0
        HL = (a + b - c - d) / 2.0
        HH = (a - b - c + d) / 2.0
    where [a, b; c, d] are 2x2 spatial pixel blocks:
        a: x[:, :, 0::2, 0::2]  (top-left)
        b: x[:, :, 0::2, 1::2]  (top-right)
        c: x[:, :, 1::2, 0::2]  (bottom-left)
        d: x[:, :, 1::2, 1::2]  (bottom-right)

    Args:
        x: Input image tensor of shape (B, C, H, W) with even H, W.

    Returns:
        Tuple of (LL, LH, HL, HH) subband tensors, each of shape (B, C, H/2, W/2).

    Raises:
        TypeError: If x is not a torch.Tensor.
        ValueError: If x is not 4D, has odd spatial dimensions, or contains NaN/Inf.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.ndim}D shape {tuple(x.shape)}")
    if x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0:
        raise ValueError(f"Image spatial dimensions must be even, got {tuple(x.shape[2:])}")
    if not torch.isfinite(x).all():
        raise ValueError("Input tensor contains NaN or Inf values")

    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]

    ll = (a + b + c + d) / 2.0
    lh = (a - b + c - d) / 2.0
    hl = (a + b - c - d) / 2.0
    hh = (a - b - c + d) / 2.0

    return ll, lh, hl, hh


def haar_idwt2d(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    """Computes differentiable 1-level 2D Haar Inverse Discrete Wavelet Transform.

    Formulation:
        a = (LL + LH + HL + HH) / 2.0
        b = (LL - LH + HL - HH) / 2.0
        c = (LL + LH - HL - HH) / 2.0
        d = (LL - LH - HL + HH) / 2.0

    Args:
        ll: Approximation subband tensor of shape (B, C, H_sub, W_sub).
        lh: Horizontal detail subband tensor of shape (B, C, H_sub, W_sub).
        hl: Vertical detail subband tensor of shape (B, C, H_sub, W_sub).
        hh: Diagonal detail subband tensor of shape (B, C, H_sub, W_sub).

    Returns:
        Reconstructed spatial tensor of shape (B, C, 2*H_sub, 2*W_sub).

    Raises:
        TypeError: If any subband is not a torch.Tensor.
        ValueError: If subband shapes mismatch or contain NaN/Inf.
    """
    for name, s in [("ll", ll), ("lh", lh), ("hl", hl), ("hh", hh)]:
        if not isinstance(s, torch.Tensor):
            raise TypeError(f"Subband {name} must be torch.Tensor, got {type(s).__name__}")
        if s.ndim != 4:
            raise ValueError(f"Subband {name} must be 4D (B, C, H, W), got {s.ndim}D shape {tuple(s.shape)}")

    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError(
            f"Subband shape mismatch: ll={tuple(ll.shape)}, lh={tuple(lh.shape)}, "
            f"hl={tuple(hl.shape)}, hh={tuple(hh.shape)}"
        )
    if not (
        torch.isfinite(ll).all()
        and torch.isfinite(lh).all()
        and torch.isfinite(hl).all()
        and torch.isfinite(hh).all()
    ):
        raise ValueError("Subbands contain NaN or Inf values")

    a = (ll + lh + hl + hh) / 2.0
    b = (ll - lh + hl - hh) / 2.0
    c = (ll + lh - hl - hh) / 2.0
    d = (ll - lh - hl + hh) / 2.0

    b_size, c_dim, h_sub, w_sub = ll.shape
    h_out, w_out = h_sub * 2, w_sub * 2

    # Allocate contiguous output tensor
    reconstructed = torch.empty(
        (b_size, c_dim, h_out, w_out),
        dtype=ll.dtype,
        device=ll.device,
    )

    reconstructed[:, :, 0::2, 0::2] = a
    reconstructed[:, :, 0::2, 1::2] = b
    reconstructed[:, :, 1::2, 0::2] = c
    reconstructed[:, :, 1::2, 1::2] = d

    return reconstructed


class HaarDWT2DTorch(nn.Module):
    """Module wrapper for differentiable 2D Haar DWT and IDWT."""

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return haar_dwt2d(x)

    def inverse(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        return haar_idwt2d(ll, lh, hl, hh)


# =============================================================================
# 2. Result Dataclasses
# =============================================================================

@dataclass(frozen=True)
class SpatialEmbeddingResult:
    """Artifacts resulting from spatial neural embedding."""

    stego_image: torch.Tensor
    stego_raw_float: torch.Tensor
    acs_embedding_result: ACSEmbeddingResult
    subbands_original: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    hh_stego: torch.Tensor
    clip_stats: Dict[str, Any]


@dataclass(frozen=True)
class SpatialExtractionResult:
    """Artifacts resulting from spatial neural extraction."""

    extracted_binary: torch.Tensor
    extracted_bipolar: torch.Tensor
    delta_rec: torch.Tensor
    context_rec: torch.Tensor
    subbands_received: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    hh_drop_rec: torch.Tensor
    hh_keep_rec: torch.Tensor


@dataclass(frozen=True)
class SpatialRoundtripMetrics:
    """Comprehensive performance and error metrics for a spatial roundtrip."""

    image_id: str
    carrier_shape: Tuple[int, int, int]
    payload_bits_count: int
    mask_occupancy: float
    gamma_min: float
    gamma_stats: Dict[str, float]
    clip_stats: Dict[str, Any]
    green_blue_max_abs_err: float
    red_hh_err_stats: Dict[str, float]
    context_drift_stats: Dict[str, float]
    combined_residual_err_stats: Dict[str, float]
    ber_payload: float
    exact_recovery: bool
    psnr_db: float
    ssim: float
    mae: float
    max_pixel_diff: int
    reloaded_png: bool


# =============================================================================
# 3. Neural Spatial Bridge Engine
# =============================================================================

class NeuralSpatialBridge(nn.Module):
    """Orchestrates spatial carrier DWT, neural embedding, spatial synthesis, and extraction."""

    def __init__(
        self,
        acs_model: Optional[ACSSubINN] = None,
        quantizer: Optional[SpatialQuantizeSTE] = None,
    ) -> None:
        super().__init__()
        self.acs_model = acs_model if acs_model is not None else ACSSubINN()
        self.quantizer = (
            quantizer
            if quantizer is not None
            else SpatialQuantizeSTE(min_val=0.0, max_val=255.0, clip_grad=False)
        )
        self.dwt = HaarDWT2DTorch()

    def embed_spatial(
        self,
        carrier: torch.Tensor,
        payload_bipolar: torch.Tensor,
        mask: torch.Tensor,
        differentiable: bool = False,
        clip_pixels: bool = True,
    ) -> SpatialEmbeddingResult:
        """Embeds bipolar payload into spatial carrier image using ACS-Sub-INN v3.

        Dataflow:
            1. DWT forward on carrier: LL, LH, HL, HH
            2. Partition HH into target HH_drop (channel 0: Red) and context HH_keep (channels 1-2: Green, Blue)
            3. Call acs_model.embed_coefficients(hh_drop, hh_keep, mask, payload_bipolar)
            4. Reassemble HH' = [hh_drop_stego, hh_keep]
            5. IDWT inverse: reconstruct spatial stego image
            6. Range quantization:
               - If differentiable=True: SpatialQuantizeSTE keeps autograd graph active
               - If differentiable=False: clamp to [0, 255] and cast to uint8

        Args:
            carrier: RGB carrier tensor of shape (B, 3, H, W) in range [0, 255].
            payload_bipolar: Bipolar symbols {-1.0, +1.0} of shape (B, 1, H/2, W/2).
            mask: Spatial mask in [0, 1] of shape (B, 1, H/2, W/2).
            differentiable: If True, uses STE quantizer and leaves float tensor.
            clip_pixels: If True and differentiable is False, clamps to [0, 255].

        Returns:
            SpatialEmbeddingResult with stego image and intermediate tensors.
        """
        if not isinstance(carrier, torch.Tensor):
            raise TypeError(f"carrier must be torch.Tensor, got {type(carrier).__name__}")
        if carrier.ndim != 4:
            raise ValueError(f"carrier must be 4D (B, 3, H, W), got {carrier.ndim}D shape {tuple(carrier.shape)}")
        if carrier.shape[1] != 3:
            raise ValueError(f"carrier must have 3 channels (RGB), got {carrier.shape[1]}")
        if not torch.isfinite(carrier).all():
            raise ValueError("carrier contains NaN or Inf values")

        # Step 1: 2D Haar DWT
        ll, lh, hl, hh = self.dwt(carrier)

        # Step 2: Partition subbands under Subband Partition Option A
        hh_drop = hh[:, 0:1, :, :]   # Red-HH
        hh_keep = hh[:, 1:3, :, :]   # Green-HH, Blue-HH

        # Step 3: ACS-Sub-INN neural coefficient modulation
        acs_res = self.acs_model.embed_coefficients(
            hh_drop=hh_drop,
            hh_keep=hh_keep,
            mask=mask,
            payload_bipolar=payload_bipolar,
        )

        # Step 4: Reassemble stego HH subband
        hh_stego = torch.cat([acs_res.hh_drop_stego, hh_keep], dim=1)

        # Step 5: 2D Haar IDWT back to spatial domain
        stego_raw_float = self.dwt.inverse(ll, lh, hl, hh_stego)

        # Step 6: Compute clipping statistics before clamping
        with torch.no_grad():
            min_raw = float(stego_raw_float.min().item())
            max_raw = float(stego_raw_float.max().item())
            clipped_low = int((stego_raw_float < 0.0).sum().item())
            clipped_high = int((stego_raw_float > 255.0).sum().item())
            total_elements = stego_raw_float.numel()
            clip_stats = {
                "min_val_raw": min_raw,
                "max_val_raw": max_raw,
                "num_clipped_low": clipped_low,
                "num_clipped_high": clipped_high,
                "num_clipped_total": clipped_low + clipped_high,
                "clip_fraction": (clipped_low + clipped_high) / total_elements,
            }

        # Step 7: Quantization / Pixel range handling
        if differentiable:
            # Training path: uses STE to maintain gradient flow
            stego_image = self.quantizer(stego_raw_float)
        else:
            # Inference path: standard clamp and uint8 rounding
            if clip_pixels:
                stego_clamped = torch.clamp(stego_raw_float, 0.0, 255.0)
            else:
                stego_clamped = stego_raw_float
            stego_image = torch.round(stego_clamped).to(torch.uint8)

        return SpatialEmbeddingResult(
            stego_image=stego_image,
            stego_raw_float=stego_raw_float,
            acs_embedding_result=acs_res,
            subbands_original=(ll, lh, hl, hh),
            hh_stego=hh_stego,
            clip_stats=clip_stats,
        )

    def extract_spatial(
        self,
        stego_carrier: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> SpatialExtractionResult:
        """Extracts embedded payload from a spatial stego image.

        Dataflow:
            1. DWT forward on stego carrier
            2. Partition received HH into target Red-HH and context Green/Blue-HH
            3. Call acs_model.extract_coefficients(hh_drop_rec, hh_keep_rec, mask)

        Args:
            stego_carrier: Stego image tensor of shape (B, 3, H, W) (uint8 or float).
            mask: Optional mask tensor of shape (B, 1, H/2, W/2).

        Returns:
            SpatialExtractionResult containing extracted bits, symbols, and residuals.
        """
        if not isinstance(stego_carrier, torch.Tensor):
            raise TypeError(f"stego_carrier must be torch.Tensor, got {type(stego_carrier).__name__}")
        if stego_carrier.ndim != 4:
            raise ValueError(f"stego_carrier must be 4D (B, 3, H, W), got {stego_carrier.ndim}D")
        if stego_carrier.shape[1] != 3:
            raise ValueError(f"stego_carrier must have 3 channels (RGB), got {stego_carrier.shape[1]}")
        if not torch.isfinite(stego_carrier).all():
            raise ValueError("stego_carrier contains NaN or Inf values")

        carrier_f = stego_carrier.float()

        # Step 1: Forward DWT
        ll_rec, lh_rec, hl_rec, hh_rec = self.dwt(carrier_f)

        # Step 2: Subband partition
        hh_drop_rec = hh_rec[:, 0:1, :, :]
        hh_keep_rec = hh_rec[:, 1:3, :, :]

        # Step 3: ACS-Sub-INN extraction
        acs_extract = self.acs_model.extract_coefficients(
            hh_drop_stego=hh_drop_rec,
            hh_keep=hh_keep_rec,
            mask=mask,
        )

        return SpatialExtractionResult(
            extracted_binary=acs_extract.extracted_binary,
            extracted_bipolar=acs_extract.extracted_bipolar,
            delta_rec=acs_extract.delta_rec,
            context_rec=acs_extract.context_rec,
            subbands_received=(ll_rec, lh_rec, hl_rec, hh_rec),
            hh_drop_rec=hh_drop_rec,
            hh_keep_rec=hh_keep_rec,
        )

    @classmethod
    def save_and_reload_png(cls, stego_uint8: torch.Tensor) -> torch.Tensor:
        """Saves a (B, 3, H, W) uint8 tensor to PNG bytes via PIL and reloads it.

        Guarantees realistic, non-simulated lossless PNG serialization roundtrip.

        Args:
            stego_uint8: Tensor of shape (B, 3, H, W), dtype uint8.

        Returns:
            Reloaded tensor of shape (B, 3, H, W), dtype uint8.
        """
        if not isinstance(stego_uint8, torch.Tensor):
            raise TypeError("stego_uint8 must be a torch.Tensor")
        if stego_uint8.ndim != 4 or stego_uint8.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W) tensor, got {tuple(stego_uint8.shape)}")

        reloaded_list = []
        batch_size = stego_uint8.shape[0]

        for i in range(batch_size):
            img_np = stego_uint8[i].permute(1, 2, 0).cpu().numpy()
            pil_img = Image.fromarray(img_np, mode="RGB")

            # Save to in-memory PNG buffer
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)

            # Reload from in-memory PNG buffer
            reloaded_pil = Image.open(buf).convert("RGB")
            reloaded_arr = np.array(reloaded_pil, dtype=np.uint8)

            reloaded_tensor = torch.from_numpy(reloaded_arr).permute(2, 0, 1)
            reloaded_list.append(reloaded_tensor)

        reloaded = torch.stack(reloaded_list, dim=0).to(stego_uint8.device)
        return reloaded

    def run_full_roundtrip(
        self,
        carrier: torch.Tensor,
        payload_bits: torch.Tensor,
        mask: torch.Tensor,
        reload_png: bool = True,
        image_id: str = "carrier",
    ) -> Tuple[SpatialEmbeddingResult, SpatialExtractionResult, SpatialRoundtripMetrics]:
        """Executes complete spatial embedding, PNG roundtrip, extraction, and metric auditing.

        Args:
            carrier: Original RGB carrier of shape (B, 3, H, W) in [0, 255].
            payload_bits: Binary payload of shape (B, 1, H/2, W/2) in {0, 1}.
            mask: Embedding mask of shape (B, 1, H/2, W/2) in [0, 1].
            reload_png: If True, saves to PNG and reloads before extraction.
            image_id: Human-readable identifier for reporting.

        Returns:
            Tuple of (SpatialEmbeddingResult, SpatialExtractionResult, SpatialRoundtripMetrics).
        """
        payload_bipolar = ACSSubINN.bipolar_encode(payload_bits)

        # 1. Forward Embedding
        embed_res = self.embed_spatial(
            carrier=carrier,
            payload_bipolar=payload_bipolar,
            mask=mask,
            differentiable=False,
            clip_pixels=True,
        )

        # 2. Container Roundtrip
        stego_uint8 = embed_res.stego_image
        if reload_png:
            stego_reloaded = self.save_and_reload_png(stego_uint8)
        else:
            stego_reloaded = stego_uint8

        # 3. Extraction
        extract_res = self.extract_spatial(
            stego_carrier=stego_reloaded,
            mask=mask,
        )

        # 4. Metric Computation
        with torch.no_grad():
            carrier_np = carrier.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            stego_np = stego_reloaded.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

            psnr_val = calculate_psnr(carrier_np, stego_np)
            ssim_val = calculate_ssim(carrier_np, stego_np)
            mae_val = float(np.mean(np.abs(carrier_np.astype(np.float64) - stego_np.astype(np.float64))))
            max_pix_diff = int(np.max(np.abs(carrier_np.astype(np.int32) - stego_np.astype(np.int32))))

            # Subband Errors & Residual Analysis
            active_mask = (mask == 1.0)
            num_active = int(active_mask.sum().item())

            # Payload BER
            rec_active_bits = extract_res.extracted_binary[active_mask]
            orig_active_bits = payload_bits[active_mask]
            bit_errors = int((rec_active_bits != orig_active_bits).sum().item())
            ber_payload = bit_errors / num_active if num_active > 0 else 0.0
            exact_recovery = (bit_errors == 0)

            # Preserved Subband Drift (Green/Blue HH)
            hh_keep_orig = embed_res.subbands_original[3][:, 1:3, :, :]
            hh_keep_rec = extract_res.hh_keep_rec
            green_blue_max_err = float((hh_keep_rec - hh_keep_orig).abs().max().item())

            # Target Red-HH Error
            hh_drop_stego_orig = embed_res.acs_embedding_result.hh_drop_stego
            hh_drop_rec = extract_res.hh_drop_rec
            red_hh_err = (hh_drop_rec - hh_drop_stego_orig).abs()
            red_hh_err_active = red_hh_err[active_mask]
            red_hh_stats = {
                "min": float(red_hh_err_active.min().item()) if num_active > 0 else 0.0,
                "max": float(red_hh_err_active.max().item()) if num_active > 0 else 0.0,
                "mean": float(red_hh_err_active.mean().item()) if num_active > 0 else 0.0,
                "std": float(red_hh_err_active.std().item()) if num_active > 0 else 0.0,
                "p99": float(np.percentile(red_hh_err_active.detach().cpu().numpy(), 99)) if num_active > 0 else 0.0,
            }

            # Context Prediction Drift
            context_orig = embed_res.acs_embedding_result.context
            context_rec = extract_res.context_rec
            context_diff = (context_rec - context_orig).abs()[active_mask]
            context_drift_stats = {
                "min": float(context_diff.min().item()) if num_active > 0 else 0.0,
                "max": float(context_diff.max().item()) if num_active > 0 else 0.0,
                "mean": float(context_diff.mean().item()) if num_active > 0 else 0.0,
                "std": float(context_diff.std().item()) if num_active > 0 else 0.0,
            }

            # Gamma Statistics
            gamma_active = embed_res.acs_embedding_result.gamma[active_mask]
            gamma_stats = {
                "min": float(gamma_active.min().item()) if num_active > 0 else 0.0,
                "max": float(gamma_active.max().item()) if num_active > 0 else 0.0,
                "mean": float(gamma_active.mean().item()) if num_active > 0 else 0.0,
                "std": float(gamma_active.std().item()) if num_active > 0 else 0.0,
            }

            # Combined Residual Error Relative to intended gamma * P:
            intended_delta = embed_res.acs_embedding_result.gamma * payload_bipolar
            residual_err = (extract_res.delta_rec - intended_delta).abs()[active_mask]
            combined_err_stats = {
                "min": float(residual_err.min().item()) if num_active > 0 else 0.0,
                "max": float(residual_err.max().item()) if num_active > 0 else 0.0,
                "mean": float(residual_err.mean().item()) if num_active > 0 else 0.0,
                "std": float(residual_err.std().item()) if num_active > 0 else 0.0,
            }

        metrics = SpatialRoundtripMetrics(
            image_id=image_id,
            carrier_shape=tuple(carrier.shape[1:]),
            payload_bits_count=num_active,
            mask_occupancy=num_active / mask.numel(),
            gamma_min=self.acs_model.gamma_min,
            gamma_stats=gamma_stats,
            clip_stats=embed_res.clip_stats,
            green_blue_max_abs_err=green_blue_max_err,
            red_hh_err_stats=red_hh_stats,
            context_drift_stats=context_drift_stats,
            combined_residual_err_stats=combined_err_stats,
            ber_payload=ber_payload,
            exact_recovery=exact_recovery,
            psnr_db=psnr_val,
            ssim=ssim_val,
            mae=mae_val,
            max_pixel_diff=max_pix_diff,
            reloaded_png=reload_png,
        )

        return embed_res, extract_res, metrics

    def evaluate_no_embedding_control(
        self,
        carrier: torch.Tensor,
        reload_png: bool = True,
        image_id: str = "control",
    ) -> Dict[str, Any]:
        """Runs a no-embedding DWT -> IDWT -> PNG control to measure transform/container baseline error.

        Args:
            carrier: RGB carrier of shape (B, 3, H, W) in [0, 255].
            reload_png: Whether to serialize and reload as PNG.
            image_id: Image identifier.

        Returns:
            Dict containing control fidelity and numerical reconstruction metrics.
        """
        # Forward DWT
        ll, lh, hl, hh = self.dwt(carrier)

        # IDWT with zero modifications
        rec_float = self.dwt.inverse(ll, lh, hl, hh)

        # Float difference before rounding
        max_float_diff = float((carrier - rec_float).abs().max().item())

        # Quantize to uint8
        rec_uint8 = torch.clamp(torch.round(rec_float), 0.0, 255.0).to(torch.uint8)

        if reload_png:
            rec_reloaded = self.save_and_reload_png(rec_uint8)
        else:
            rec_reloaded = rec_uint8

        carrier_np = carrier.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rec_np = rec_reloaded.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        psnr_val = calculate_psnr(carrier_np, rec_np)
        ssim_val = calculate_ssim(carrier_np, rec_np)
        mae_val = float(np.mean(np.abs(carrier_np.astype(np.float64) - rec_np.astype(np.float64))))
        max_pix_diff = int(np.max(np.abs(carrier_np.astype(np.int32) - rec_np.astype(np.int32))))

        # Subband verification on reloaded control
        ll_rec, lh_rec, hl_rec, hh_rec = self.dwt(rec_reloaded.float())
        max_subband_err = max(
            float((ll - ll_rec).abs().max().item()),
            float((lh - lh_rec).abs().max().item()),
            float((hl - hl_rec).abs().max().item()),
            float((hh - hh_rec).abs().max().item()),
        )

        return {
            "image_id": image_id,
            "max_float_diff": max_float_diff,
            "max_pixel_diff": max_pix_diff,
            "psnr_db": psnr_val,
            "ssim": ssim_val,
            "mae": mae_val,
            "max_subband_err": max_subband_err,
            "exact_bit_identity": (max_pix_diff == 0),
        }
