"""Phase 5A ONNX Opset 14 Export Infrastructure and Parity Evaluation.

This module defines deployable PyTorch inference wrappers for ACS-Sub-INN v3:
1. ACSSubINNEmbeddingGraph: Core coefficient-domain embedding graph.
2. ACSSubINNExtractionGraph: Core coefficient-domain extraction graph.
3. NeuralSpatialEmbeddingGraph: End-to-end spatial-to-spatial embedding graph.
4. NeuralSpatialExtractionGraph: End-to-end spatial-to-binary extraction graph.

Enforces:
- Opset 14 compliance.
- Explicit input and output tensor contracts.
- Deterministic inference mode (eval(), torch.no_grad()).
- Graceful detection of environment dependencies (onnx, onnxruntime).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.spatial_bridge import NeuralSpatialBridge, haar_dwt2d, haar_idwt2d

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Inference Graph Wrappers for ONNX Export
# =============================================================================

class ACSSubINNEmbeddingGraph(nn.Module):
    """Deployable coefficient-domain neural embedding graph.

    Input Contract:
        - hh_drop: (B, 1, H, W) float32, Red-HH target subband.
        - hh_keep: (B, 2, H, W) float32, Green/Blue-HH context subbands.
        - mask: (B, 1, H, W) float32, binary or continuous embedding mask in [0, 1].
        - payload_bipolar: (B, 1, H, W) float32, bipolar symbols in {-1.0, +1.0}.

    Output Contract:
        - hh_drop_stego: (B, 1, H, W) float32, modulated Red-HH subband.
        - context: (B, 1, H, W) float32, predicted spatial context.
        - gamma: (B, 1, H, W) float32, adaptive modulation gain (>= gamma_min).
    """

    def __init__(self, acs_model: ACSSubINN) -> None:
        super().__init__()
        self.context_predictor = acs_model.context_predictor
        self.adaptive_gain = acs_model.adaptive_gain
        self.gamma_min = acs_model.gamma_min
        self.eval()

    def forward(
        self,
        hh_drop: torch.Tensor,
        hh_keep: torch.Tensor,
        mask: torch.Tensor,
        payload_bipolar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context_predictor(hh_keep)
        gamma = self.adaptive_gain(hh_keep)
        stego_branch = context + gamma * payload_bipolar
        hh_drop_stego = (1.0 - mask) * hh_drop + mask * stego_branch
        return hh_drop_stego, context, gamma


class ACSSubINNExtractionGraph(nn.Module):
    """Deployable coefficient-domain neural extraction graph.

    Input Contract:
        - hh_drop_stego: (B, 1, H, W) float32, received Red-HH subband.
        - hh_keep: (B, 2, H, W) float32, received Green/Blue-HH subband.

    Output Contract:
        - delta_rec: (B, 1, H, W) float32, extraction residual (hh_drop_stego - context).
        - context_rec: (B, 1, H, W) float32, reconstructed context.
        - extracted_binary: (B, 1, H, W) float32, binary bits in {0.0, 1.0} (delta_rec > 0).
    """

    def __init__(self, acs_model: ACSSubINN) -> None:
        super().__init__()
        self.context_predictor = acs_model.context_predictor
        self.eval()

    def forward(
        self,
        hh_drop_stego: torch.Tensor,
        hh_keep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_rec = self.context_predictor(hh_keep)
        delta_rec = hh_drop_stego - context_rec
        extracted_binary = (delta_rec > 0.0).float()
        return delta_rec, context_rec, extracted_binary


class NeuralSpatialEmbeddingGraph(nn.Module):
    """End-to-end spatial-to-spatial neural embedding graph.

    Input Contract:
        - carrier: (B, 3, H, W) float32, RGB image in [0.0, 255.0].
        - mask: (B, 1, H/2, W/2) float32, subband embedding mask.
        - payload_bipolar: (B, 1, H/2, W/2) float32, bipolar symbols in {-1.0, +1.0}.

    Output Contract:
        - stego_image: (B, 3, H, W) float32, clamped spatial stego image in [0.0, 255.0].
        - context: (B, 1, H/2, W/2) float32, predicted context.
        - gamma: (B, 1, H/2, W/2) float32, adaptive gain.
    """

    def __init__(self, bridge: NeuralSpatialBridge) -> None:
        super().__init__()
        self.context_predictor = bridge.acs_model.context_predictor
        self.adaptive_gain = bridge.acs_model.adaptive_gain
        self.eval()

    def forward(
        self,
        carrier: torch.Tensor,
        mask: torch.Tensor,
        payload_bipolar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ll, lh, hl, hh = haar_dwt2d(carrier)
        hh_drop = hh[:, 0:1, :, :]
        hh_keep = hh[:, 1:3, :, :]

        context = self.context_predictor(hh_keep)
        gamma = self.adaptive_gain(hh_keep)
        stego_drop = (1.0 - mask) * hh_drop + mask * (context + gamma * payload_bipolar)
        hh_stego = torch.cat([stego_drop, hh_keep], dim=1)

        stego_raw = haar_idwt2d(ll, lh, hl, hh_stego)
        stego_image = torch.clamp(stego_raw, 0.0, 255.0)
        return stego_image, context, gamma


class NeuralSpatialExtractionGraph(nn.Module):
    """End-to-end spatial-to-binary neural extraction graph.

    Input Contract:
        - stego_carrier: (B, 3, H, W) float32, RGB stego image in [0.0, 255.0].

    Output Contract:
        - extracted_binary: (B, 1, H/2, W/2) float32, binary bits in {0.0, 1.0}.
        - delta_rec: (B, 1, H/2, W/2) float32, extraction residual.
    """

    def __init__(self, bridge: NeuralSpatialBridge) -> None:
        super().__init__()
        self.context_predictor = bridge.acs_model.context_predictor
        self.eval()

    def forward(
        self,
        stego_carrier: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, _, hh_rec = haar_dwt2d(stego_carrier)
        hh_drop_rec = hh_rec[:, 0:1, :, :]
        hh_keep_rec = hh_rec[:, 1:3, :, :]

        context_rec = self.context_predictor(hh_keep_rec)
        delta_rec = hh_drop_rec - context_rec
        extracted_binary = (delta_rec > 0.0).float()
        return extracted_binary, delta_rec


# =============================================================================
# 2. Dependency Audit and Parity Evaluation Utilities
# =============================================================================

@dataclass(frozen=True)
class DependencyStatus:
    torch_version: str
    onnx_available: bool
    onnx_version: Optional[str]
    onnxruntime_available: bool
    onnxruntime_version: Optional[str]
    onnxruntime_providers: List[str]


def check_onnx_dependencies() -> DependencyStatus:
    """Inspects installed packages for ONNX export and runtime availability without modifying environment."""
    onnx_avail = False
    onnx_ver = None
    try:
        import onnx
        onnx_avail = True
        onnx_ver = getattr(onnx, "__version__", "unknown")
    except ImportError:
        pass

    ort_avail = False
    ort_ver = None
    ort_providers: List[str] = []
    try:
        import onnxruntime as ort
        ort_avail = True
        ort_ver = getattr(ort, "__version__", "unknown")
        ort_providers = ort.get_available_providers()
    except ImportError:
        pass

    return DependencyStatus(
        torch_version=torch.__version__,
        onnx_available=onnx_avail,
        onnx_version=onnx_ver,
        onnxruntime_available=ort_avail,
        onnxruntime_version=ort_ver,
        onnxruntime_providers=ort_providers,
    )


def compute_numerical_parity(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Computes exact numerical discrepancy metrics between PyTorch and ONNX outputs."""
    if pytorch_output.shape != onnx_output.shape:
        return {
            "shape_match": False,
            "pytorch_shape": list(pytorch_output.shape),
            "onnx_shape": list(onnx_output.shape),
            "passed": False,
        }

    abs_diff = np.abs(pytorch_output.astype(np.float64) - onnx_output.astype(np.float64))
    max_abs_err = float(np.max(abs_diff))
    mean_abs_err = float(np.mean(abs_diff))

    # Safe relative error avoiding division by zero
    nonzero_mask = np.abs(pytorch_output) > 1e-8
    if np.any(nonzero_mask):
        rel_diff = abs_diff[nonzero_mask] / np.abs(pytorch_output[nonzero_mask])
        max_rel_err = float(np.max(rel_diff))
    else:
        max_rel_err = 0.0

    finite_check = bool(np.all(np.isfinite(onnx_output)))
    allclose = bool(np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol))

    return {
        "shape_match": True,
        "shape": list(pytorch_output.shape),
        "dtype": str(onnx_output.dtype),
        "finite_values": finite_check,
        "max_absolute_error": max_abs_err,
        "mean_absolute_error": mean_abs_err,
        "max_relative_error": max_rel_err,
        "configured_rtol": rtol,
        "configured_atol": atol,
        "passed": allclose and finite_check,
    }
