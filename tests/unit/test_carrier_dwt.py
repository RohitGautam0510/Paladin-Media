"""Unit tests for 1-level 2D Haar DWT and IDWT (T-011, T-012)."""

import numpy as np
import pytest
from PIL import Image

from src.carrier.dwt import HaarDWT2D, DWTSubbands, visualize_subbands


def test_t011_dwt_subband_shapes():
    """T-011: Verify 1-level 2D Haar DWT produces explicit (3, 256, 256) subbands."""
    # Test with random 512x512 RGB image array
    rng = np.random.RandomState(42)
    img_arr = rng.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    subbands = HaarDWT2D.forward(img_arr)
    assert isinstance(subbands, DWTSubbands)

    expected_shape = (3, 256, 256)
    assert subbands.ll.shape == expected_shape
    assert subbands.lh.shape == expected_shape
    assert subbands.hl.shape == expected_shape
    assert subbands.hh.shape == expected_shape

    assert subbands.ll.dtype == np.float64
    assert subbands.lh.dtype == np.float64
    assert subbands.hl.dtype == np.float64
    assert subbands.hh.dtype == np.float64

    # Also test with PIL Image
    pil_img = Image.fromarray(img_arr, mode="RGB")
    subbands_pil = HaarDWT2D.forward(pil_img)
    np.testing.assert_array_equal(subbands.ll, subbands_pil.ll)


def test_t012_dwt_idwt_reconstruction():
    """T-012: Verify mathematical invertibility and numerical reconstruction error of DWT->IDWT."""
    rng = np.random.RandomState(1337)
    img_arr = rng.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    subbands = HaarDWT2D.forward(img_arr)
    reconstructed = HaarDWT2D.inverse(subbands)

    # Invertibility on integer coordinates must have 0.0 floating-point reconstruction error
    max_error = np.max(np.abs(img_arr.astype(np.float64) - reconstructed))
    assert max_error == 0.0

    # Uint8 rounded array must be bit-for-bit identical to original
    rec_uint8 = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(img_arr, rec_uint8)


def test_dwt_visualizations():
    """Verify subband visualization images are 256x256 RGB PIL Images."""
    img = Image.new("RGB", (512, 512), color=(120, 80, 200))
    subbands = HaarDWT2D.forward(img)
    visuals = visualize_subbands(subbands)

    assert set(visuals.keys()) == {"LL", "LH", "HL", "HH"}
    for key, img_vis in visuals.items():
        assert isinstance(img_vis, Image.Image)
        assert img_vis.size == (256, 256)
        assert img_vis.mode == "RGB"
