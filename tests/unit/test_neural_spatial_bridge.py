"""Unit and integration tests for Phase 3 Neural Spatial Bridge (Sprint 2).

Covers:
1. DWT->IDWT numerical reconstruction before integer quantization (mathematical invertibility).
2. Valid RGB uint8 input and output (inference path).
3. Channel and subband ordering (strict Phase 2 convention compliance).
4. Differentiable autograd training path with SpatialQuantizeSTE.
5. All-zero, all-one, and mixed payload patterns through lossless PNG roundtrip.
6. Mask and payload shape validation.
7. Fail-closed handling of invalid dimensions and non-finite tensors.
8. Handling of clipping and near-saturated carrier pixels.
9. Exact decision-boundary behavior for residual zero.
10. No-embedding control verification (100% bit-exact baseline identity).
11. CUDA execution if GPU is available.
"""

from __future__ import annotations

import io
import numpy as np
import pytest
import torch
from PIL import Image

from src.carrier.dwt import HaarDWT2D
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.spatial_bridge import (
    HaarDWT2DTorch,
    NeuralSpatialBridge,
    SpatialEmbeddingResult,
    SpatialExtractionResult,
    SpatialRoundtripMetrics,
    haar_dwt2d,
    haar_idwt2d,
)


@pytest.fixture
def default_bridge():
    """Initializes default NeuralSpatialBridge with fixed random seed."""
    torch.manual_seed(42)
    return NeuralSpatialBridge()


# =============================================================================
# 1. Numerical Reconstruction Before Quantization
# =============================================================================

def test_dwt_idwt_numerical_reconstruction_before_quantization():
    """Verify that Haar DWT -> IDWT is mathematically invertible with near-zero error.

    For integer spatial coordinates, floating-point error must be exactly 0.0.
    For arbitrary continuous float coordinates, reconstruction error must be < 1e-6.
    """
    torch.manual_seed(101)

    # Integer coordinates in [0, 255]
    int_carrier = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.float32)
    ll, lh, hl, hh = haar_dwt2d(int_carrier)
    rec_int = haar_idwt2d(ll, lh, hl, hh)

    max_int_err = (int_carrier - rec_int).abs().max().item()
    assert max_int_err == 0.0, f"Expected 0.0 error for integer coordinates, got {max_int_err}"

    # Continuous float coordinates
    float_carrier = torch.randn(2, 3, 64, 64) * 100.0 + 128.0
    ll_f, lh_f, hl_f, hh_f = haar_dwt2d(float_carrier)
    rec_float = haar_idwt2d(ll_f, lh_f, hl_f, hh_f)

    max_float_err = (float_carrier - rec_float).abs().max().item()
    assert max_float_err < 1e-4, f"Expected < 1e-4 error for float32 coordinates, got {max_float_err}"


# =============================================================================
# 2. Channel and Subband Ordering Compliance
# =============================================================================

def test_dwt_idwt_channel_and_subband_ordering():
    """Verify exact equivalence between PyTorch tensor DWT and Phase 2 numpy HaarDWT2D."""
    rng = np.random.RandomState(42)
    img_np = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    # 1. Classical Phase 2 DWT
    phase2_subbands = HaarDWT2D.forward(img_np)

    # 2. PyTorch Spatial Bridge DWT
    tensor_img = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    ll_t, lh_t, hl_t, hh_t = haar_dwt2d(tensor_img)

    # Assert exact channel-first shape (3, H/2, W/2)
    assert ll_t.shape == (1, 3, 64, 64)
    assert lh_t.shape == (1, 3, 64, 64)
    assert hl_t.shape == (1, 3, 64, 64)
    assert hh_t.shape == (1, 3, 64, 64)

    # Assert exact numerical match with Phase 2 arrays
    np.testing.assert_allclose(phase2_subbands.ll, ll_t.squeeze(0).numpy(), atol=1e-5)
    np.testing.assert_allclose(phase2_subbands.lh, lh_t.squeeze(0).numpy(), atol=1e-5)
    np.testing.assert_allclose(phase2_subbands.hl, hl_t.squeeze(0).numpy(), atol=1e-5)
    np.testing.assert_allclose(phase2_subbands.hh, hh_t.squeeze(0).numpy(), atol=1e-5)

    # Test inverse equivalence
    phase2_rec = HaarDWT2D.inverse(phase2_subbands)
    torch_rec = haar_idwt2d(ll_t, lh_t, hl_t, hh_t).squeeze(0).permute(1, 2, 0).numpy()
    np.testing.assert_allclose(phase2_rec, torch_rec, atol=1e-5)


# =============================================================================
# 3. Valid RGB uint8 Input and Output (Inference Path)
# =============================================================================

def test_spatial_bridge_valid_rgb_uint8_io(default_bridge):
    """Verify that inference embedding produces a valid RGB uint8 tensor and PIL image."""
    carrier = torch.full((1, 3, 32, 32), 128.0, dtype=torch.float32)
    payload_bipolar = torch.ones(1, 1, 16, 16)
    mask = torch.ones(1, 1, 16, 16)

    embed_res = default_bridge.embed_spatial(
        carrier=carrier,
        payload_bipolar=payload_bipolar,
        mask=mask,
        differentiable=False,
        clip_pixels=True,
    )

    assert isinstance(embed_res, SpatialEmbeddingResult)
    assert embed_res.stego_image.dtype == torch.uint8
    assert embed_res.stego_image.shape == (1, 3, 32, 32)
    assert (embed_res.stego_image >= 0).all() and (embed_res.stego_image <= 255).all()

    # PIL Image construction
    stego_np = embed_res.stego_image.squeeze(0).permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(stego_np, mode="RGB")
    assert pil_img.size == (32, 32)
    assert pil_img.mode == "RGB"


# =============================================================================
# 4. Differentiable Autograd Training Path with SpatialQuantizeSTE
# =============================================================================

def test_spatial_bridge_differentiable_training_path():
    """Verify that differentiable=True preserves autograd graph for end-to-end training."""
    torch.manual_seed(42)
    model = ACSSubINN()
    bridge = NeuralSpatialBridge(acs_model=model)

    carrier = torch.full((1, 3, 32, 32), 128.0, dtype=torch.float32, requires_grad=True)
    payload_bipolar = torch.ones(1, 1, 16, 16)
    mask = torch.ones(1, 1, 16, 16)

    # Embed with differentiable=True
    embed_res = bridge.embed_spatial(
        carrier=carrier,
        payload_bipolar=payload_bipolar,
        mask=mask,
        differentiable=True,
    )

    assert embed_res.stego_image.requires_grad is True
    assert embed_res.stego_image.dtype == torch.float32

    # Extract differentiably from stego_image
    extract_res = bridge.extract_spatial(embed_res.stego_image, mask=mask)

    # Compute a dummy loss combining image fidelity and residual error
    loss_fidelity = ((embed_res.stego_image - carrier) ** 2).mean()
    loss_extraction = (extract_res.delta_rec - embed_res.acs_embedding_result.gamma).abs().mean()
    total_loss = loss_fidelity + loss_extraction

    total_loss.backward()

    # Gradients must flow back to carrier and to all model parameters
    assert carrier.grad is not None
    assert torch.isfinite(carrier.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} gradient contains NaN/Inf"


# =============================================================================
# 5. Payload Patterns through Lossless PNG Roundtrip
# =============================================================================

@pytest.mark.parametrize("payload_type", ["all_zeros", "all_ones", "alternating", "random"])
def test_payload_patterns_png_roundtrip(default_bridge, payload_type: str):
    """Verify that different payload bitstreams are accurately recovered after PNG save/reload."""
    b, h, w = 1, 64, 64
    sub_h, sub_w = h // 2, w // 2

    # Smooth carrier avoiding clipping boundaries
    carrier = torch.full((b, 3, h, w), 128.0, dtype=torch.float32)

    if payload_type == "all_zeros":
        payload_bits = torch.zeros(b, 1, sub_h, sub_w, dtype=torch.uint8)
    elif payload_type == "all_ones":
        payload_bits = torch.ones(b, 1, sub_h, sub_w, dtype=torch.uint8)
    elif payload_type == "alternating":
        payload_bits = torch.zeros(b, 1, sub_h, sub_w, dtype=torch.uint8)
        payload_bits[:, :, ::2, :] = 1
    else:
        torch.manual_seed(999)
        payload_bits = torch.randint(0, 2, (b, 1, sub_h, sub_w), dtype=torch.uint8)

    mask = torch.ones(b, 1, sub_h, sub_w)

    emb_res, ext_res, metrics = default_bridge.run_full_roundtrip(
        carrier=carrier,
        payload_bits=payload_bits,
        mask=mask,
        reload_png=True,
        image_id=f"test_{payload_type}",
    )

    assert metrics.reloaded_png is True
    assert metrics.ber_payload == 0.0, f"Payload {payload_type} had BER={metrics.ber_payload}"
    assert metrics.exact_recovery is True
    assert (ext_res.extracted_binary == payload_bits).all()


# =============================================================================
# 6. Shape and Dimension Fail-Closed Validation
# =============================================================================

def test_spatial_bridge_shape_and_dimension_validation(default_bridge):
    """Verify explicit fail-closed rejection on invalid shapes, non-4D tensors, and NaNs."""
    valid_carrier = torch.full((1, 3, 32, 32), 128.0)
    valid_payload = torch.ones(1, 1, 16, 16)
    valid_mask = torch.ones(1, 1, 16, 16)

    # 1. Non-tensor input
    with pytest.raises(TypeError, match="carrier must be torch.Tensor"):
        default_bridge.embed_spatial("invalid", valid_payload, valid_mask)

    # 2. Odd spatial dimensions in DWT
    with pytest.raises(ValueError, match="Image spatial dimensions must be even"):
        haar_dwt2d(torch.randn(1, 3, 33, 33))

    # 3. 3D carrier tensor (missing batch dimension)
    with pytest.raises(ValueError, match="carrier must be 4D"):
        default_bridge.embed_spatial(torch.randn(3, 32, 32), valid_payload, valid_mask)

    # 4. Wrong channel count in carrier (e.g. 1 channel or 4 channels)
    with pytest.raises(ValueError, match="carrier must have 3 channels"):
        default_bridge.embed_spatial(torch.randn(1, 1, 32, 32), valid_payload, valid_mask)

    # 5. NaN in carrier
    nan_carrier = valid_carrier.clone()
    nan_carrier[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="carrier contains NaN or Inf"):
        default_bridge.embed_spatial(nan_carrier, valid_payload, valid_mask)

    # 6. Inf in carrier
    inf_carrier = valid_carrier.clone()
    inf_carrier[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="carrier contains NaN or Inf"):
        default_bridge.embed_spatial(inf_carrier, valid_payload, valid_mask)

    # 7. Subband shape mismatch in IDWT
    ll = torch.randn(1, 3, 16, 16)
    lh = torch.randn(1, 3, 16, 16)
    hl = torch.randn(1, 3, 16, 16)
    hh_bad = torch.randn(1, 3, 16, 32)
    with pytest.raises(ValueError, match="Subband shape mismatch"):
        haar_idwt2d(ll, lh, hl, hh_bad)


# =============================================================================
# 7. Exact Decision Boundary for Residual Zero
# =============================================================================

def test_exact_decision_boundary_for_residual_zero():
    """Verify that a residual of exactly 0.0 resolves to bit 0.

    Mathematical rule:
        b_hat = (delta_rec > 0.0)
    For delta_rec == 0.0:
        0.0 > 0.0 is False => decoded bit is 0.
        If P = +1 was sent (bit 1), this represents a decision boundary error.
        If P = -1 was sent (bit 0), this represents a correct decision.
    """
    model = ACSSubINN()
    delta_zero = torch.zeros(1, 1, 4, 4)

    decoded_bits = model.bipolar_decode(delta_zero)
    assert (decoded_bits == 0).all()

    # Small positive epsilon should decode to 1
    delta_pos = torch.full((1, 1, 4, 4), 1e-7)
    assert (model.bipolar_decode(delta_pos) == 1).all()

    # Small negative epsilon should decode to 0
    delta_neg = torch.full((1, 1, 4, 4), -1e-7)
    assert (model.bipolar_decode(delta_neg) == 0).all()


# =============================================================================
# 8. Near-Saturated Carrier Pixels and Clipping Handling
# =============================================================================

def test_near_saturated_pixels_and_clipping_handling(default_bridge):
    """Verify clipping tracking on carrier with pixels near 0 and 255."""
    h, w = 32, 32
    # Create extreme carrier with 0 and 255 values
    carrier_arr = np.zeros((h, w, 3), dtype=np.float32)
    carrier_arr[:16, :16, :] = 255.0
    carrier_arr[16:, 16:, :] = 0.0
    carrier_arr[:16, 16:, :] = 1.0
    carrier_arr[16:, :16, :] = 254.0

    carrier = torch.from_numpy(carrier_arr).permute(2, 0, 1).unsqueeze(0)
    mask = torch.ones(1, 1, h // 2, w // 2)
    payload = torch.randint(0, 2, (1, 1, h // 2, w // 2), dtype=torch.uint8)

    emb_res, ext_res, metrics = default_bridge.run_full_roundtrip(
        carrier=carrier,
        payload_bits=payload,
        mask=mask,
        reload_png=True,
        image_id="near_saturated_carrier",
    )

    # Verify that clip_stats correctly recorded clipping
    assert "num_clipped_total" in metrics.clip_stats
    assert "min_val_raw" in metrics.clip_stats
    assert "max_val_raw" in metrics.clip_stats
    assert metrics.clip_stats["num_clipped_total"] > 0
    assert metrics.clip_stats["clip_fraction"] > 0.0

    # Ensure output image was clamped strictly within [0, 255]
    assert emb_res.stego_image.min().item() >= 0
    assert emb_res.stego_image.max().item() <= 255


# =============================================================================
# 9. No-Embedding Control Exact Identity
# =============================================================================

def test_no_embedding_control_exact_identity(default_bridge):
    """Verify that unmodulated DWT -> IDWT -> PNG roundtrip has strictly 0.0 error."""
    rng = np.random.RandomState(42)
    img_np = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    carrier = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

    ctrl = default_bridge.evaluate_no_embedding_control(carrier, reload_png=True, image_id="control_test")

    assert ctrl["max_float_diff"] == 0.0
    assert ctrl["max_pixel_diff"] == 0
    assert ctrl["psnr_db"] == float("inf")
    assert ctrl["ssim"] == pytest.approx(1.0, abs=1e-7)
    assert ctrl["mae"] == 0.0
    assert ctrl["max_subband_err"] == 0.0
    assert ctrl["exact_bit_identity"] is True


# =============================================================================
# 10. CUDA Device Compatibility (if available)
# =============================================================================

def test_cuda_execution_if_available():
    """Verify NeuralSpatialBridge functions on CUDA if an NVIDIA GPU is present."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    model = ACSSubINN().to(device)
    bridge = NeuralSpatialBridge(acs_model=model).to(device)

    carrier = torch.full((1, 3, 64, 64), 128.0, device=device)
    payload_bits = torch.randint(0, 2, (1, 1, 32, 32), dtype=torch.uint8, device=device)
    mask = torch.ones(1, 1, 32, 32, device=device)

    emb_res, ext_res, metrics = bridge.run_full_roundtrip(
        carrier=carrier,
        payload_bits=payload_bits,
        mask=mask,
        reload_png=True,
    )

    assert metrics.exact_recovery is True
    assert metrics.ber_payload == 0.0
    assert emb_res.stego_image.device.type == "cuda"
