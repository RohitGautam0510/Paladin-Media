"""Straight-Through Estimator (STE) Quantization Components.

Provides differentiable quantization primitives for Phase 3 neural steganography:
- RoundSTEFunction / round_ste: Forward integer rounding with backward pass-through.
- SpatialQuantizeSTEFunction / spatial_quantize_ste: Clamping to [min_val, max_val]
  followed by integer rounding, with backward identity gradient propagation.
- RoundSTE / SpatialQuantizeSTE: nn.Module wrappers.

IMPORTANT THEORETICAL DISCLAIMER:
Straight-Through Estimators (STE) approximate the gradient of non-differentiable
step functions (rounding and clamping) during backpropagation by substituting an
identity gradient. STE does NOT make spatial integer rounding or boundary clipping
mathematically invertible. Non-linear truncation and quantization errors occur on the
forward pass and represent irreversible information loss unless bounded by the
adaptive gain margin.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RoundSTEFunction(torch.autograd.Function):
    """Autograd function implementing Straight-Through Estimator for integer rounding.

    Forward Pass:
        y = round(x)
    Backward Pass:
        dL/dx = dL/dy (identity gradient pass-through)
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
        return torch.round(x)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        # Identity gradient flow
        return grad_output


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Applies integer rounding with Straight-Through Estimator gradient pass-through.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of rounded values with identical shape and dtype.
    """
    return RoundSTEFunction.apply(x)


class RoundSTE(nn.Module):
    """Module wrapper for RoundSTEFunction."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return round_ste(x)


class SpatialQuantizeSTEFunction(torch.autograd.Function):
    """Autograd function for spatial image quantization: clamp to [min_val, max_val] and round.

    Forward Pass:
        y = clamp(round(x), min_val, max_val)
    Backward Pass:
        If clip_grad is False:
            dL/dx = dL/dy (unconditional straight-through pass-through)
        If clip_grad is True:
            dL/dx = dL/dy where min_val <= x <= max_val; 0 otherwise.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        min_val: float = 0.0,
        max_val: float = 255.0,
        clip_grad: bool = False,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be strictly less than max_val ({max_val})")

        ctx.save_for_backward(x)
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.clip_grad = clip_grad

        rounded = torch.round(x)
        clamped = torch.clamp(rounded, min_val, max_val)
        return clamped

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        (x,) = ctx.saved_tensors
        if ctx.clip_grad:
            # Mask gradients outside the dynamic range
            valid_mask = (x >= ctx.min_val) & (x <= ctx.max_val)
            grad_input = grad_output * valid_mask.to(grad_output.dtype)
        else:
            grad_input = grad_output

        return grad_input, None, None, None


def spatial_quantize_ste(
    x: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 255.0,
    clip_grad: bool = False,
) -> torch.Tensor:
    """Quantizes spatial pixel values to integer range [min_val, max_val] using STE.

    Args:
        x: Input tensor (e.g., reconstructed spatial image in range [0, 255]).
        min_val: Lower clamping bound (default 0.0).
        max_val: Upper clamping bound (default 255.0).
        clip_grad: If True, zeroes gradients for inputs outside [min_val, max_val].
                   If False, passes gradients straight through unconditionally (default).

    Returns:
        Clamped and rounded tensor.
    """
    return SpatialQuantizeSTEFunction.apply(x, min_val, max_val, clip_grad)


class SpatialQuantizeSTE(nn.Module):
    """Module wrapper for SpatialQuantizeSTEFunction."""

    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 255.0,
        clip_grad: bool = False,
    ) -> None:
        super().__init__()
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be strictly less than max_val ({max_val})")
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.clip_grad = bool(clip_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return spatial_quantize_ste(x, self.min_val, self.max_val, self.clip_grad)

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}, max_val={self.max_val}, clip_grad={self.clip_grad}"
