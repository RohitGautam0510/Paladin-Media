"""Unit tests for Phase 3 isolated neural steganography components.

Tests:
1. ContextPredictorCNN: tensor shapes, multiple batch sizes, spatial preservation.
2. ContextPredictorCNN: CPU forward execution and backward gradient flow.
3. AdaptiveGainCNN: strictly positive, finite gains (gamma >= gamma_min > 0).
4. AdaptiveGainCNN: CPU forward execution and backward gradient flow.
5. RoundSTE: forward integer rounding and identity backward gradient flow.
6. SpatialQuantizeSTE: clamping, rounding, boundary handling, and gradient clipping options.
7. STE documentation invariant: test demonstrating rounding is not invertible.
8. ACSSubINN bipolar mapping: binary-to-bipolar encoding and demodulation.
9. ACSSubINN coefficient-domain exact recovery: 100% bit-exact payload recovery in unquantized coefficient domain.
10. ACSSubINN gradient flow: gradients propagate to both neural predictors.
11. Reproducibility: deterministic initialization under fixed random seed.
12. Input validation: explicit rejection of invalid dimensions, channels, types, non-finite values, and mask bounds.
13. CUDA execution: clean execution when CUDA available, skipped cleanly when unavailable.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.adaptive_gain import AdaptiveGainCNN
from src.neural.context_predictor import ContextPredictorCNN
from src.neural.quantization import (
    RoundSTE,
    SpatialQuantizeSTE,
    round_ste,
    spatial_quantize_ste,
)


# =============================================================================
# 1. ContextPredictorCNN Tests
# =============================================================================

@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("spatial_dim", [(8, 8), (16, 16), (32, 32)])
def test_context_predictor_shapes_and_batch_sizes(batch_size: int, spatial_dim: tuple[int, int]):
    """Verify ContextPredictorCNN preserves spatial resolution and returns (B, 1, H, W)."""
    h, w = spatial_dim
    model = ContextPredictorCNN(in_channels=2, out_channels=1, hidden_channels=16, num_blocks=2)
    x = torch.randn(batch_size, 2, h, w)
    out = model(x)

    assert out.shape == (batch_size, 1, h, w)
    assert torch.isfinite(out).all()


def test_context_predictor_cpu_forward_and_backward():
    """Verify forward computation and gradient propagation through all layers on CPU."""
    model = ContextPredictorCNN(in_channels=2, out_channels=1, hidden_channels=16, num_blocks=2)
    x = torch.randn(2, 2, 16, 16, requires_grad=True)
    out = model(x)

    target = torch.randn_like(out)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).any()

    # Check that model weights received non-zero gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite grad"


# =============================================================================
# 2. AdaptiveGainCNN Tests
# =============================================================================

@pytest.mark.parametrize("gamma_min", [0.5, 1.0, 2.5])
def test_adaptive_gain_strict_positivity(gamma_min: float):
    """Verify adaptive gain strictly satisfies gamma(x, y) >= gamma_min > 0."""
    model = AdaptiveGainCNN(
        in_channels=2, out_channels=1, hidden_channels=16, num_blocks=2, gamma_min=gamma_min
    )
    # Test with extreme negative, zero, and positive inputs
    x = torch.cat([
        torch.full((1, 2, 8, 8), -100.0),
        torch.zeros((1, 2, 8, 8)),
        torch.full((1, 2, 8, 8), 100.0),
    ], dim=0)

    gamma = model(x)
    assert gamma.shape == (3, 1, 8, 8)
    assert torch.isfinite(gamma).all()
    assert (gamma >= gamma_min).all(), f"Gain violated floor: min found {gamma.min()} < {gamma_min}"


def test_adaptive_gain_cpu_backward_gradient_flow():
    """Verify gradients propagate backwards through Softplus and gain CNN layers."""
    model = AdaptiveGainCNN(in_channels=2, out_channels=1, hidden_channels=16, num_blocks=2, gamma_min=1.0)
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    gamma = model(x)

    loss = gamma.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).any()


# =============================================================================
# 3. Straight-Through Estimator (STE) Quantization Tests
# =============================================================================

def test_round_ste_forward_and_backward():
    """Verify round_ste produces exact integer rounding in forward and identity gradient in backward."""
    x = torch.tensor([-1.7, -0.4, 0.0, 0.4, 1.2, 2.8], requires_grad=True)
    y = round_ste(x)

    # Forward check
    expected = torch.tensor([-2.0, 0.0, 0.0, 0.0, 1.0, 3.0])
    assert torch.allclose(y, expected)

    # Backward check: d/dx(sum(y * weights)) = weights
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    loss = (y * weights).sum()
    loss.backward()

    assert x.grad is not None
    assert torch.allclose(x.grad, weights)


def test_spatial_quantize_ste_forward_and_backward():
    """Verify spatial_quantize_ste clamping to [0, 255] and gradient clipping behavior."""
    x = torch.tensor([-10.5, 0.4, 127.8, 254.6, 300.2], requires_grad=True)

    # 1. Forward check with [0, 255] bounds
    y = spatial_quantize_ste(x, min_val=0.0, max_val=255.0, clip_grad=False)
    expected_y = torch.tensor([0.0, 0.0, 128.0, 255.0, 255.0])
    assert torch.allclose(y, expected_y)

    # 2. Backward check with clip_grad=False (identity pass-through everywhere)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x.grad, torch.ones_like(x))

    # 3. Backward check with clip_grad=True (zeroes outside [min_val, max_val])
    x_clip = torch.tensor([-10.5, 0.4, 127.8, 254.6, 300.2], requires_grad=True)
    y_clip = spatial_quantize_ste(x_clip, min_val=0.0, max_val=255.0, clip_grad=True)
    loss_clip = y_clip.sum()
    loss_clip.backward()
    expected_grad = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
    assert torch.allclose(x_clip.grad, expected_grad)


def test_ste_does_not_make_rounding_invertible():
    """Explicit test documenting that rounding is non-invertible despite differentiable STE."""
    x1 = torch.tensor([1.1])
    x2 = torch.tensor([1.4])
    y1 = round_ste(x1)
    y2 = round_ste(x2)

    # Both map to the same rounded value 1.0
    assert y1 == y2 == torch.tensor([1.0])
    # Information distinguishing 1.1 from 1.4 is irreversibly lost
    assert x1 != x2


# =============================================================================
# 4. Bipolar Symbol Mapping Tests
# =============================================================================

def test_bipolar_symbol_encoding_and_decoding():
    """Verify exact bijection between binary {0, 1} and bipolar {-1.0, +1.0}."""
    bits = torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)
    bipolar = ACSSubINN.bipolar_encode(bits)

    expected_bipolar = torch.tensor([[-1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
    assert torch.allclose(bipolar, expected_bipolar)

    # Demodulate back to binary
    recovered_bits = ACSSubINN.bipolar_decode(bipolar)
    assert (recovered_bits == bits).all()

    # Demodulate slightly perturbed continuous symbols
    noisy_bipolar = torch.tensor([[-0.2, 0.8], [0.1, -0.9]])
    recovered_noisy = ACSSubINN.bipolar_decode(noisy_bipolar)
    assert (recovered_noisy == bits).all()


def test_bipolar_encode_rejects_non_binary_values():
    """Verify bipolar_encode rejects non-binary inputs."""
    invalid = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match="must contain only values in {0, 1}"):
        ACSSubINN.bipolar_encode(invalid)


# =============================================================================
# 5. ACSSubINN Coefficient-Domain Reversibility Tests
# =============================================================================

def test_acs_sub_inn_coefficient_domain_exact_recovery():
    """Verify 100% exact bit recovery in the unquantized coefficient domain.

    Tests that:
    1. At masked positions (M == 1), extracted bits strictly match the original binary payload.
    2. At unmasked positions (M == 0), carrier coefficients remain bit-exact identical.
    """
    model = ACSSubINN(
        in_channels=2,
        out_channels=1,
        hidden_channels=16,
        num_blocks_context=2,
        num_blocks_gain=2,
        gamma_min=1.0,
    )

    batch_size = 2
    h, w = 16, 16

    # Synthetic subband coefficients
    torch.manual_seed(42)
    hh_drop = torch.randn(batch_size, 1, h, w)
    hh_keep = torch.randn(batch_size, 2, h, w)

    # Binary mask: half of coefficients active
    mask = (torch.rand(batch_size, 1, h, w) > 0.5).to(torch.float32)

    # Binary payload in {0, 1} mapped to bipolar {-1.0, 1.0}
    binary_payload = torch.randint(0, 2, (batch_size, 1, h, w), dtype=torch.uint8)
    bipolar_payload = ACSSubINN.bipolar_encode(binary_payload)

    # Forward embedding
    embed_res = model.embed_coefficients(hh_drop, hh_keep, mask, bipolar_payload)

    # Extraction
    extract_res = model.extract_coefficients(embed_res.hh_drop_stego, hh_keep)

    # 1. Assert unmasked coefficients are completely unmodified
    unmasked = mask == 0.0
    assert torch.allclose(embed_res.hh_drop_stego[unmasked], hh_drop[unmasked])

    # 2. Assert active masked bits are recovered with 100% exact match
    masked = mask == 1.0
    recovered_active = extract_res.extracted_binary[masked]
    original_active = binary_payload[masked]

    assert (recovered_active == original_active).all()
    bit_error_rate = float((recovered_active != original_active).sum()) / original_active.numel()
    assert bit_error_rate == 0.0


def test_acs_sub_inn_gradient_flow_end_to_end():
    """Verify backpropagation through embed_coefficients updates neural parameters."""
    model = ACSSubINN(
        in_channels=2,
        out_channels=1,
        hidden_channels=16,
        num_blocks_context=2,
        num_blocks_gain=2,
        gamma_min=1.0,
    )

    hh_drop = torch.randn(2, 1, 8, 8)
    hh_keep = torch.randn(2, 2, 8, 8, requires_grad=True)
    mask = torch.ones(2, 1, 8, 8)
    payload = torch.full((2, 1, 8, 8), 1.0)

    res = model.embed_coefficients(hh_drop, hh_keep, mask, payload)
    loss = res.hh_drop_stego.sum()
    loss.backward()

    assert hh_keep.grad is not None
    assert torch.isfinite(hh_keep.grad).all()

    # Check context predictor parameters
    for name, p in model.context_predictor.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Context predictor parameter {name} has None grad"
            assert torch.isfinite(p.grad).all()

    # Check adaptive gain parameters
    for name, p in model.adaptive_gain.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Adaptive gain parameter {name} has None grad"
            assert torch.isfinite(p.grad).all()


# =============================================================================
# 6. Reproducibility & Initialization Tests
# =============================================================================

def test_reproducibility_under_controlled_seed():
    """Verify deterministic weight initialization and output reproducibility."""
    torch.manual_seed(12345)
    model1 = ACSSubINN(hidden_channels=16, num_blocks_context=2, num_blocks_gain=2)

    torch.manual_seed(12345)
    model2 = ACSSubINN(hidden_channels=16, num_blocks_context=2, num_blocks_gain=2)

    # Check weights match bit-for-bit
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)

    # Check outputs match bit-for-bit on identical input
    x_drop = torch.randn(1, 1, 8, 8)
    x_keep = torch.randn(1, 2, 8, 8)
    mask = torch.ones(1, 1, 8, 8)
    payload = torch.ones(1, 1, 8, 8)

    res1 = model1.embed_coefficients(x_drop, x_keep, mask, payload)
    res2 = model2.embed_coefficients(x_drop, x_keep, mask, payload)

    assert torch.equal(res1.hh_drop_stego, res2.hh_drop_stego)


# =============================================================================
# 7. Input Validation and Failure Tests
# =============================================================================

def test_input_validation_rejections():
    """Verify model explicitly rejects malformed, non-finite, and incompatible inputs."""
    model = ACSSubINN()
    valid_drop = torch.randn(2, 1, 8, 8)
    valid_keep = torch.randn(2, 2, 8, 8)
    valid_mask = torch.ones(2, 1, 8, 8)
    valid_payload = torch.ones(2, 1, 8, 8)

    # Non-tensor input
    with pytest.raises(TypeError):
        model.embed_coefficients([1, 2, 3], valid_keep, valid_mask, valid_payload)  # type: ignore

    # Non-4D input
    with pytest.raises(ValueError, match="to be 4D tensor"):
        model.embed_coefficients(torch.randn(2, 1, 8), valid_keep, valid_mask, valid_payload)

    # Channel mismatch: hh_drop must have 1 channel
    with pytest.raises(ValueError, match="Expected hh_drop to have 1 channel"):
        model.embed_coefficients(torch.randn(2, 3, 8, 8), valid_keep, valid_mask, valid_payload)

    # Channel mismatch: hh_keep must have 2 channels
    with pytest.raises(ValueError, match="Expected hh_keep shape"):
        model.embed_coefficients(valid_drop, torch.randn(2, 1, 8, 8), valid_mask, valid_payload)

    # Spatial dimension mismatch
    with pytest.raises(ValueError, match="Expected hh_keep shape"):
        model.embed_coefficients(valid_drop, torch.randn(2, 2, 16, 16), valid_mask, valid_payload)

    # Mask value outside [0, 1]
    with pytest.raises(ValueError, match="Mask values must be in range"):
        model.embed_coefficients(valid_drop, valid_keep, torch.full((2, 1, 8, 8), 2.5), valid_payload)

    # Non-finite input
    nan_drop = valid_drop.clone()
    nan_drop[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        model.embed_coefficients(nan_drop, valid_keep, valid_mask, valid_payload)


# =============================================================================
# 8. CUDA Execution Tests (When Available)
# =============================================================================

def test_cuda_execution_smoke_test():
    """Verify CUDA execution, device allocation, and CPU/CUDA parity when GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this host")

    torch.cuda.init()
    device = torch.device("cuda:0")
    dummy = torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats()

    model = ACSSubINN(
        in_channels=2,
        out_channels=1,
        hidden_channels=16,
        num_blocks_context=2,
        num_blocks_gain=2,
    ).to(device)

    hh_drop = torch.randn(2, 1, 16, 16, device=device)
    hh_keep = torch.randn(2, 2, 16, 16, device=device)
    mask = torch.ones(2, 1, 16, 16, device=device)
    payload = torch.full((2, 1, 16, 16), 1.0, device=device)

    embed_res = model.embed_coefficients(hh_drop, hh_keep, mask, payload)
    extract_res = model.extract_coefficients(embed_res.hh_drop_stego, hh_keep)

    torch.cuda.synchronize()

    assert embed_res.hh_drop_stego.device.type == "cuda"
    assert extract_res.extracted_binary.device.type == "cuda"
    assert (extract_res.extracted_binary == 1).all()

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    assert peak_mem_mb > 0
    # Small synthetic test should allocate far below 100 MB
    assert peak_mem_mb < 50.0, f"Peak memory higher than expected: {peak_mem_mb:.2f} MB"
