"""ACS-Sub-INN v3 Coefficient-Domain Orchestration Module.

Implements the core coefficient-domain embedding and extraction equations for the
Adaptive Coefficient Steganography Subband Invertible Neural Network (ACS-Sub-INN v3):
- Forward embedding:
    context = ContextPredictorCNN(HH_keep)
    gamma   = Softplus(AdaptiveGainCNN(HH_keep)) + gamma_min
    HH_drop' = (1 - M) * HH_drop + M * (context + gamma * P)
- Inverse extraction:
    context_rec = ContextPredictorCNN(HH_keep)
    delta_rec   = HH_drop' - context_rec
    P_hat       = sign(delta_rec)
    b_hat       = (delta_rec > 0)
- Bipolar symbol mapping:
    P = 2b - 1 in {-1.0, +1.0} for b in {0, 1}

ARCHITECTURAL INVARIANTS ENFORCED:
1. No secret-side DWT: Secret payload enters as an encrypted binary bitstream.
2. No latent-Z: Invertibility is achieved via direct context cancellation.
3. Transform-domain only: Operates strictly on Haar DWT wavelet coefficients.
4. Channel isolation: HH_keep (Green/Blue) is unmodulated, ensuring context is bit-exact
   at the receiver in unquantized coefficient-domain evaluation.
5. Domain separation: Exact recovery in coefficient domain does NOT imply exact recovery
   after spatial IDWT, 8-bit integer rounding, dynamic range clipping, or JPEG compression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.neural.adaptive_gain import AdaptiveGainCNN
from src.neural.context_predictor import ContextPredictorCNN


@dataclass(frozen=True)
class ACSEmbeddingResult:
    """Artifacts resulting from coefficient-domain neural embedding."""

    hh_drop_stego: torch.Tensor
    context: torch.Tensor
    gamma: torch.Tensor
    mask: torch.Tensor
    payload_bipolar: torch.Tensor


@dataclass(frozen=True)
class ACSExtractionResult:
    """Artifacts resulting from coefficient-domain neural extraction."""

    extracted_binary: torch.Tensor
    extracted_bipolar: torch.Tensor
    delta_rec: torch.Tensor
    context_rec: torch.Tensor


class ACSSubINN(nn.Module):
    """Orchestrates ACS-Sub-INN v3 coefficient-domain forward embedding and extraction."""

    def __init__(
        self,
        context_predictor: Optional[ContextPredictorCNN] = None,
        adaptive_gain: Optional[AdaptiveGainCNN] = None,
        in_channels: int = 2,
        out_channels: int = 1,
        hidden_channels: int = 32,
        num_blocks_context: int = 3,
        num_blocks_gain: int = 2,
        gamma_min: float = 1.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.context_predictor = (
            context_predictor
            if context_predictor is not None
            else ContextPredictorCNN(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks_context,
                negative_slope=negative_slope,
            )
        )
        self.adaptive_gain = (
            adaptive_gain
            if adaptive_gain is not None
            else AdaptiveGainCNN(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks_gain,
                gamma_min=gamma_min,
                negative_slope=negative_slope,
            )
        )
        self.gamma_min = float(gamma_min)

    @staticmethod
    def bipolar_encode(binary_bits: torch.Tensor) -> torch.Tensor:
        """Maps binary bits b in {0, 1} to bipolar symbols P in {-1.0, +1.0}.

        Formulation:
            P = 2b - 1.0
            b = 0 -> P = -1.0
            b = 1 -> P = +1.0

        Args:
            binary_bits: Tensor containing binary values 0 or 1.

        Returns:
            Floating-point tensor of bipolar symbols in {-1.0, +1.0}.

        Raises:
            ValueError: If input tensor contains values other than 0 and 1.
        """
        if not isinstance(binary_bits, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(binary_bits).__name__}")

        # Check that all elements are binary (0 or 1)
        is_binary = ((binary_bits == 0) | (binary_bits == 1)).all()
        if not is_binary:
            raise ValueError("Input binary_bits tensor must contain only values in {0, 1}")

        return 2.0 * binary_bits.to(torch.float32) - 1.0

    @staticmethod
    def bipolar_decode(symbols: torch.Tensor) -> torch.Tensor:
        """Demodulates continuous residual symbols into binary bits in {0, 1}.

        Formulation:
            b_hat = 1 if symbol > 0 else 0

        Args:
            symbols: Continuous or bipolar symbol tensor.

        Returns:
            Binary uint8 tensor where elements are 1 if symbol > 0, else 0.
        """
        if not isinstance(symbols, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(symbols).__name__}")
        return (symbols > 0).to(torch.uint8)

    def embed_coefficients(
        self,
        hh_drop: torch.Tensor,
        hh_keep: torch.Tensor,
        mask: torch.Tensor,
        payload_bipolar: torch.Tensor,
    ) -> ACSEmbeddingResult:
        """Embeds bipolar payload into target subband coefficients using neural context and gain.

        Forward Equation:
            context = ContextPredictorCNN(hh_keep)
            gamma   = AdaptiveGainCNN(hh_keep)
            hh_drop' = (1 - mask) * hh_drop + mask * (context + gamma * payload_bipolar)

        Args:
            hh_drop: Target subband coefficient tensor of shape (B, 1, H, W) to modulate (Red-HH).
            hh_keep: Preserved subband coefficient tensor of shape (B, 2, H, W) (Green/Blue-HH).
            mask: Binary or floating mask of shape (B, 1, H, W) or broadcastable (1, 1, H, W).
            payload_bipolar: Modulating symbols in {-1.0, +1.0} of shape (B, 1, H, W).

        Returns:
            ACSEmbeddingResult containing modified coefficients and intermediate tensors.

        Raises:
            TypeError: If any input is not a torch.Tensor.
            ValueError: If tensor dimensions, channels, or shapes are incompatible.
        """
        self._validate_inputs(hh_drop, hh_keep, mask, payload_bipolar)

        # Ensure mask and payload are float tensors matching hh_drop dtype
        mask_f = mask.to(hh_drop.dtype)
        payload_f = payload_bipolar.to(hh_drop.dtype)

        # Neural forward prediction
        context = self.context_predictor(hh_keep)
        gamma = self.adaptive_gain(hh_keep)

        # Modulate coefficients
        stego_branch = context + gamma * payload_f
        hh_drop_stego = (1.0 - mask_f) * hh_drop + mask_f * stego_branch

        return ACSEmbeddingResult(
            hh_drop_stego=hh_drop_stego,
            context=context,
            gamma=gamma,
            mask=mask_f,
            payload_bipolar=payload_f,
        )

    def extract_coefficients(
        self,
        hh_drop_stego: torch.Tensor,
        hh_keep: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> ACSExtractionResult:
        """Extracts embedded symbols from modified subband coefficients.

        Extraction Equation:
            context_rec = ContextPredictorCNN(hh_keep)
            delta_rec   = hh_drop_stego - context_rec
            P_hat       = sign(delta_rec)
            b_hat       = 1 if delta_rec > 0 else 0

        Args:
            hh_drop_stego: Modulated target subband tensor of shape (B, 1, H, W).
            hh_keep: Preserved context subband tensor of shape (B, 2, H, W).
            mask: Optional mask tensor of shape (B, 1, H, W).

        Returns:
            ACSExtractionResult with extracted binary bits, bipolar symbols, and residuals.
        """
        if not isinstance(hh_drop_stego, torch.Tensor) or not isinstance(hh_keep, torch.Tensor):
            raise TypeError("hh_drop_stego and hh_keep must be torch.Tensors")
        if hh_drop_stego.ndim != 4 or hh_keep.ndim != 4:
            raise ValueError("Tensors must be 4D (B, C, H, W)")
        if hh_drop_stego.shape[0] != hh_keep.shape[0]:
            raise ValueError(f"Batch size mismatch: {hh_drop_stego.shape[0]} vs {hh_keep.shape[0]}")
        if hh_drop_stego.shape[1] != 1:
            raise ValueError(f"Expected hh_drop_stego to have 1 channel, got {hh_drop_stego.shape[1]}")
        if hh_keep.shape[1] != 2:
            raise ValueError(f"Expected hh_keep to have 2 channels, got {hh_keep.shape[1]}")
        if hh_drop_stego.shape[2:] != hh_keep.shape[2:]:
            raise ValueError(f"Spatial dimension mismatch: {hh_drop_stego.shape[2:]} vs {hh_keep.shape[2:]}")
        if not torch.isfinite(hh_drop_stego).all() or not torch.isfinite(hh_keep).all():
            raise ValueError("Inputs contain NaN or Inf")

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"Expected mask to be torch.Tensor, got {type(mask).__name__}")
            if mask.ndim != 4:
                raise ValueError(f"Expected mask to be 4D tensor (B, C, H, W), got shape {mask.shape}")
            b, _, h, w = hh_drop_stego.shape
            if mask.shape not in [(b, 1, h, w), (1, 1, h, w)]:
                raise ValueError(f"Incompatible mask shape {mask.shape} for spatial dimensions ({h}, {w})")
            if ((mask < 0.0) | (mask > 1.0)).any():
                raise ValueError("Mask values must be in range [0, 1]")

        context_rec = self.context_predictor(hh_keep)
        delta_rec = hh_drop_stego - context_rec

        extracted_bipolar = torch.sign(delta_rec)
        extracted_binary = self.bipolar_decode(delta_rec)

        return ACSExtractionResult(
            extracted_binary=extracted_binary,
            extracted_bipolar=extracted_bipolar,
            delta_rec=delta_rec,
            context_rec=context_rec,
        )

    def _validate_inputs(
        self,
        hh_drop: torch.Tensor,
        hh_keep: torch.Tensor,
        mask: torch.Tensor,
        payload_bipolar: torch.Tensor,
    ) -> None:
        """Validates input tensor dimensions, types, channels, and numerical stability."""
        for name, t in [
            ("hh_drop", hh_drop),
            ("hh_keep", hh_keep),
            ("mask", mask),
            ("payload_bipolar", payload_bipolar),
        ]:
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Expected {name} to be torch.Tensor, got {type(t).__name__}")
            if t.ndim != 4:
                raise ValueError(f"Expected {name} to be 4D tensor (B, C, H, W), got shape {t.shape}")
            if not torch.isfinite(t).all():
                raise ValueError(f"Input {name} contains NaN or Inf values")

        b, c_drop, h, w = hh_drop.shape
        if c_drop != 1:
            raise ValueError(f"Expected hh_drop to have 1 channel, got {c_drop}")

        if hh_keep.shape != (b, 2, h, w):
            raise ValueError(f"Expected hh_keep shape {(b, 2, h, w)}, got {hh_keep.shape}")

        if mask.shape not in [(b, 1, h, w), (1, 1, h, w)]:
            raise ValueError(f"Incompatible mask shape {mask.shape} for spatial dimensions ({h}, {w})")

        # Mask values must be bounded in [0, 1]
        if ((mask < 0.0) | (mask > 1.0)).any():
            raise ValueError("Mask values must be in range [0, 1]")

        if payload_bipolar.shape != (b, 1, h, w):
            raise ValueError(
                f"Expected payload_bipolar shape {(b, 1, h, w)}, got {payload_bipolar.shape}"
            )
