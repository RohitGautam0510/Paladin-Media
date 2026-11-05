"""Adaptive Gain Convolutional Neural Network for ACS-Sub-INN v3.

Implements the AdaptiveGainCNN that predicts a spatially varying, strictly positive
modulation gain field gamma(x, y) >= gamma_min from preserved context coefficients (HH_keep).
Enforces the positivity constraint via Softplus activation plus a strictly positive floor gamma_min.

DESIGN NOTE:
Setting gamma_min = 1.0 provides a strictly positive perturbation baseline matching
the +/-1 step size of the Phase 2 classical integer parity embedder. It prevents the
optimizer from collapsing gain to zero (which would minimize distortion while destroying
payload recovery). However, gamma_min = 1.0 does NOT analytically guarantee exact bit
recovery after spatial IDWT, 8-bit rounding, dynamic range clipping, or JPEG degradation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neural.context_predictor import ResidualBlock


class AdaptiveGainCNN(nn.Module):
    """Predicts strictly positive, spatially varying embedding gain gamma from HH_keep.

    Contract:
        Input:  (B, in_channels, H_sub, W_sub) float32/float64 tensor.
                Default in_channels = 2 (Green-HH and Blue-HH).
        Output: (B, out_channels, H_sub, W_sub) float tensor.
                Default out_channels = 1.
                Activation: gamma = Softplus(raw_gain) + gamma_min.
                Guaranteed: gamma(x, y) >= gamma_min > 0 everywhere.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        gamma_min: float = 1.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if num_blocks < 0:
            raise ValueError(f"num_blocks cannot be negative, got {num_blocks}")
        if gamma_min <= 0.0:
            raise ValueError(f"gamma_min must be strictly positive (> 0), got {gamma_min}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.gamma_min = float(gamma_min)
        self.negative_slope = negative_slope

        # Input projection layer
        self.entry_conv = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1, bias=True
        )
        self.entry_act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        # Residual backbone
        blocks = [
            ResidualBlock(hidden_channels, kernel_size=3, negative_slope=negative_slope)
            for _ in range(num_blocks)
        ]
        self.body = nn.Sequential(*blocks)

        # Output projection layer (linear before Softplus)
        self.exit_conv = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

        self._init_network()

    def _init_network(self) -> None:
        """Deterministic initialization for gain predictor."""
        nn.init.kaiming_normal_(self.entry_conv.weight, a=self.negative_slope, mode="fan_in")
        nn.init.zeros_(self.entry_conv.bias)

        # Initialize output projection with small weights so initial softplus starts smoothly near 0 + gamma_min
        nn.init.xavier_normal_(self.exit_conv.weight, gain=0.01)
        nn.init.zeros_(self.exit_conv.bias)

    def forward(self, hh_keep: torch.Tensor) -> torch.Tensor:
        """Computes spatially varying adaptive gain gamma.

        Args:
            hh_keep: Batched tensor of shape (B, in_channels, H, W).

        Returns:
            Spatially varying gain tensor of shape (B, out_channels, H, W)
            where all elements strictly satisfy gamma >= gamma_min > 0.

        Raises:
            TypeError: If input is not a torch.Tensor.
            ValueError: If input tensor dimensionality, channel count, or numerical values are invalid.
        """
        if not isinstance(hh_keep, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(hh_keep).__name__}")
        if hh_keep.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got ndim={hh_keep.ndim} with shape {hh_keep.shape}")
        if hh_keep.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels (got {hh_keep.shape[1]} in shape {hh_keep.shape})"
            )
        if not torch.isfinite(hh_keep).all():
            raise ValueError("Input tensor contains NaN or Inf values")

        x = self.entry_act(self.entry_conv(hh_keep))
        x = self.body(x)
        raw_gain = self.exit_conv(x)

        # Enforce strict positivity: Softplus(x) + gamma_min
        gamma = F.softplus(raw_gain) + self.gamma_min
        return gamma

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"hidden_channels={self.hidden_channels}, num_blocks={self.num_blocks}, "
            f"gamma_min={self.gamma_min}"
        )
