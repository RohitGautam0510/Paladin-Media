"""Discrete Wavelet Transform (DWT) and Inverse DWT (IDWT) for PaladinMedia.

Implements one-level 2D Haar DWT and IDWT for 512x512 RGB carrier images per
PaladinMedia Architecture Specification §11 and Sprint-0 IRS §9.

Subband output shapes for (512, 512, 3) RGB:
- LL: (3, 256, 256) float64
- LH: (3, 256, 256) float64
- HL: (3, 256, 256) float64
- HH: (3, 256, 256) float64
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

__all__ = [
    "DWTSubbands",
    "HaarDWT2D",
    "visualize_subbands",
    "subband_to_image",
]


@dataclass(frozen=True)
class DWTSubbands:
    """Container for 1-level 2D Haar subbands with shape (3, H/2, W/2).

    Attributes:
        ll: Approximation subband (3, 256, 256), range [0, 510]
        lh: Horizontal detail subband (3, 256, 256), range [-255, 255]
        hl: Vertical detail subband (3, 256, 256), range [-255, 255]
        hh: Diagonal detail subband (3, 256, 256), range [-255, 255]
    """
    ll: np.ndarray
    lh: np.ndarray
    hl: np.ndarray
    hh: np.ndarray

    def __post_init__(self) -> None:
        for name, arr in [("ll", self.ll), ("lh", self.lh), ("hl", self.hl), ("hh", self.hh)]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Subband {name} must be a numpy.ndarray, got {type(arr).__name__}")
            if arr.ndim != 3:
                raise ValueError(f"Subband {name} must have shape (3, H, W), got ndim={arr.ndim}")
            if arr.shape[0] != 3:
                raise ValueError(f"Subband {name} first dimension must be 3 channels, got {arr.shape[0]}")


class HaarDWT2D:
    """One-level 2D Haar Discrete Wavelet Transform and Inverse Transform.

    Operates on 512x512 RGB images with explicit channel-first (3, 256, 256)
    subband representation.
    """

    EXPECTED_HEIGHT: int = 512
    EXPECTED_WIDTH: int = 512
    EXPECTED_CHANNELS: int = 3

    @classmethod
    def forward(cls, image: Union[np.ndarray, Image.Image]) -> DWTSubbands:
        """Computes 1-level 2D Haar DWT.

        Formulation for 2x2 spatial block [a, b; c, d]:
            LL = (a + b + c + d) / 2.0
            LH = (a - b + c - d) / 2.0
            HL = (a + b - c - d) / 2.0
            HH = (a - b - c + d) / 2.0

        Args:
            image: PIL Image or numpy array of shape (512, 512, 3), dtype uint8 or float.

        Returns:
            DWTSubbands with LL, LH, HL, HH arrays of shape (3, 256, 256) float64.
        """
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"), dtype=np.float64)
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != cls.EXPECTED_CHANNELS:
                raise ValueError(
                    f"Expected image array with shape (H, W, 3), got {image.shape}"
                )
            arr = image.astype(np.float64)
        else:
            raise TypeError(f"Unsupported image type: {type(image).__name__}")

        h, w, c = arr.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                f"Expected even image dimensions, got ({h}, {w})"
            )

        # Spatial 2x2 blocks:
        # a: top-left (even, even)
        # b: top-right (even, odd)
        # c: bottom-left (odd, even)
        # d: bottom-right (odd, odd)
        a = arr[0::2, 0::2, :]
        b = arr[0::2, 1::2, :]
        c = arr[1::2, 0::2, :]
        d = arr[1::2, 1::2, :]

        # Compute 2D Haar subbands
        ll = (a + b + c + d) / 2.0
        lh = (a - b + c - d) / 2.0
        hl = (a + b - c - d) / 2.0
        hh = (a - b - c + d) / 2.0

        # Transpose from (256, 256, 3) to (3, 256, 256) for strict specification compliance
        return DWTSubbands(
            ll=np.ascontiguousarray(np.transpose(ll, (2, 0, 1))),
            lh=np.ascontiguousarray(np.transpose(lh, (2, 0, 1))),
            hl=np.ascontiguousarray(np.transpose(hl, (2, 0, 1))),
            hh=np.ascontiguousarray(np.transpose(hh, (2, 0, 1))),
        )

    @classmethod
    def inverse(cls, subbands: DWTSubbands) -> np.ndarray:
        """Computes inverse 2D Haar DWT (IDWT) back to (512, 512, 3) float64.

        Formulation:
            a = (LL + LH + HL + HH) / 2.0
            b = (LL - LH + HL - HH) / 2.0
            c = (LL + LH - HL - HH) / 2.0
            d = (LL - LH - HL + HH) / 2.0

        Args:
            subbands: DWTSubbands with shape (3, 256, 256) each.

        Returns:
            Reconstructed array of shape (512, 512, 3) float64.
        """
        # Transpose back from (3, 256, 256) to (256, 256, 3)
        ll = np.transpose(subbands.ll, (1, 2, 0))
        lh = np.transpose(subbands.lh, (1, 2, 0))
        hl = np.transpose(subbands.hl, (1, 2, 0))
        hh = np.transpose(subbands.hh, (1, 2, 0))

        a = (ll + lh + hl + hh) / 2.0
        b = (ll - lh + hl - hh) / 2.0
        c = (ll + lh - hl - hh) / 2.0
        d = (ll - lh - hl + hh) / 2.0

        h_half, w_half, c_dim = ll.shape
        reconstructed = np.zeros((h_half * 2, w_half * 2, c_dim), dtype=np.float64)

        reconstructed[0::2, 0::2, :] = a
        reconstructed[0::2, 1::2, :] = b
        reconstructed[1::2, 0::2, :] = c
        reconstructed[1::2, 1::2, :] = d

        return reconstructed


def subband_to_image(subband: np.ndarray, is_ll: bool = False) -> Image.Image:
    """Normalizes an individual subband (3, 256, 256) to a viewable 8-bit RGB PIL Image.

    For LL: Values are in [0, 510], scaled by 0.5 to [0, 255].
    For Detail (LH, HL, HH): Values in [-255, 255] are centered at 128:
        val_norm = (subband / 2.0) + 128.0.

    Args:
        subband: Array of shape (3, 256, 256).
        is_ll: True if approximation band (LL), False for detail bands.

    Returns:
        PIL Image in RGB format (256x256).
    """
    # Transpose to (256, 256, 3)
    arr = np.transpose(subband, (1, 2, 0))

    if is_ll:
        scaled = np.clip(np.round(arr / 2.0), 0, 255).astype(np.uint8)
    else:
        # Map [-255, 255] -> [0, 255] with 0 mapped to 128 gray level
        scaled = np.clip(np.round((arr / 2.0) + 128.0), 0, 255).astype(np.uint8)

    return Image.fromarray(scaled, mode="RGB")


def visualize_subbands(subbands: DWTSubbands) -> Dict[str, Image.Image]:
    """Generates normalized visualizer images for all four subbands.

    Returns:
        Dict mapping 'LL', 'LH', 'HL', 'HH' to 256x256 RGB PIL Images.
    """
    return {
        "LL": subband_to_image(subbands.ll, is_ll=True),
        "LH": subband_to_image(subbands.lh, is_ll=False),
        "HL": subband_to_image(subbands.hl, is_ll=False),
        "HH": subband_to_image(subbands.hh, is_ll=False),
    }
