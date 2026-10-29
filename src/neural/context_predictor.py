"""Context Predictor Convolutional Neural Network for ACS-Sub-INN v3.

Implements the compact ContextPredictorCNN that predicts the unmodulated Red-HH
subband context from the preserved Green and Blue HH subband coefficients (HH_keep).
Performs dimension-preserving fully convolutional mapping with residual blocks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """2-layer convolutional residual block with LeakyReLU activation."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, bias=True
        )
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, bias=True
        )

        self._init_weights(negative_slope)

    def _init_weights(self, negative_slope: float) -> None:
        gain = nn.init.calculate_gain("leaky_relu", negative_slope)
        nn.init.kaiming_normal_(self.conv1.weight, a=negative_slope, mode="fan_in")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, a=negative_slope, mode="fan_in")
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class ContextPredictorCNN(nn.Module):
    """Predicts high-frequency subband context from preserved context channels (HH_keep).

    Contract:
        Input:  (B, in_channels, H_sub, W_sub) float32/float64 tensor.
                Default in_channels = 2 (Green-HH and Blue-HH).
        Output: (B, out_channels, H_sub, W_sub) float tensor.
                Default out_channels = 1 (Predicted Red-HH context).
                Linear activation: unconstrained real-valued predictions.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        hidden_channels: int = 32,
        num_blocks: int = 3,
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
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

        # Output projection layer (linear, unbounded context prediction)
        self.exit_conv = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

        self._init_network()

    def _init_network(self) -> None:
        """Deterministic Kaiming normal initialization for conv layers."""
        nn.init.kaiming_normal_(self.entry_conv.weight, a=self.negative_slope, mode="fan_in")
        nn.init.zeros_(self.entry_conv.bias)

        # Initialize output projection with small weights to center initial context prediction
        nn.init.xavier_normal_(self.exit_conv.weight, gain=0.1)
        nn.init.zeros_(self.exit_conv.bias)

    def forward(self, hh_keep: torch.Tensor) -> torch.Tensor:
        """Executes forward context prediction.

        Args:
            hh_keep: Batched tensor of shape (B, in_channels, H, W).

        Returns:
            Predicted context tensor of shape (B, out_channels, H, W).

        Raises:
            TypeError: If input is not a torch.Tensor.
            ValueError: If input tensor dimensionality or channel count is invalid.
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
        out = self.exit_conv(x)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"hidden_channels={self.hidden_channels}, num_blocks={self.num_blocks}"
        )
