"""Targeted unit tests for Phase 3 coefficient-domain extraction contract (Sprint 1.1).

Covers:
1. Zero and one payload bits: all-0 and all-1 payload embedding and extraction.
2. Positive floor margin: extraction behavior when adaptive gain is near gamma_min floor.
3. Exact recovery with unchanged context: 100% bit-exact recovery under invariant context.
4. Controlled context perturbations: testing the exact analytical decision margin gamma.
   - Sub-margin perturbations (|err| < gamma) maintain 100% correct bit decisions.
   - Super-margin perturbations (|err| >= gamma in opposing direction) cross decision boundary.
5. Unmasked coefficient integrity: unmasked positions remain bit-exact identical.
6. Mask and payload shape/range validation: explicit rejection of malformed or invalid inputs.
"""

from __future__ import annotations

import pytest
import torch

from src.neural.acs_sub_inn import ACSSubINN


@pytest.fixture
def acs_model():
    """Initializes standard ACSSubINN model."""
    torch.manual_seed(42)
    return ACSSubINN(
        in_channels=2,
        out_channels=1,
        hidden_channels=16,
        num_blocks_context=2,
        num_blocks_gain=2,
        gamma_min=1.0,
    )


# =============================================================================
# 1. Zero and One Payload Bits
# =============================================================================

def test_extraction_all_zeros_payload(acs_model):
    """Verify embedding and extracting an all-zero payload (b=0 -> P=-1.0)."""
    batch_size = 2
    h, w = 16, 16
    hh_drop = torch.randn(batch_size, 1, h, w)
    hh_keep = torch.randn(batch_size, 2, h, w)
    mask = torch.ones(batch_size, 1, h, w)

    # All-zero binary payload
    payload_zeros = torch.zeros(batch_size, 1, h, w, dtype=torch.uint8)
    bipolar_zeros = ACSSubINN.bipolar_encode(payload_zeros)
    assert (bipolar_zeros == -1.0).all()

    embed_res = acs_model.embed_coefficients(hh_drop, hh_keep, mask, bipolar_zeros)
    extract_res = acs_model.extract_coefficients(embed_res.hh_drop_stego, hh_keep, mask=mask)

    # Delta should be exactly -gamma < 0
    assert (extract_res.delta_rec < 0.0).all()
    assert (extract_res.extracted_binary == 0).all()
    assert torch.allclose(extract_res.delta_rec, -embed_res.gamma, atol=1e-5)


def test_extraction_all_ones_payload(acs_model):
    """Verify embedding and extracting an all-one payload (b=1 -> P=+1.0)."""
    batch_size = 2
    h, w = 16, 16
    hh_drop = torch.randn(batch_size, 1, h, w)
    hh_keep = torch.randn(batch_size, 2, h, w)
    mask = torch.ones(batch_size, 1, h, w)

    # All-one binary payload
    payload_ones = torch.ones(batch_size, 1, h, w, dtype=torch.uint8)
    bipolar_ones = ACSSubINN.bipolar_encode(payload_ones)
    assert (bipolar_ones == 1.0).all()

    embed_res = acs_model.embed_coefficients(hh_drop, hh_keep, mask, bipolar_ones)
    extract_res = acs_model.extract_coefficients(embed_res.hh_drop_stego, hh_keep, mask=mask)

    # Delta should be exactly +gamma > 0
    assert (extract_res.delta_rec > 0.0).all()
    assert (extract_res.extracted_binary == 1).all()
    assert torch.allclose(extract_res.delta_rec, embed_res.gamma, atol=1e-5)


# =============================================================================
# 2. Gain Values Near Configured Positive Floor
# =============================================================================

@pytest.mark.parametrize("gamma_floor", [0.1, 0.5, 1.0, 2.0])
def test_extraction_near_positive_floor(gamma_floor: float):
    """Verify extraction accuracy when gain is at or very close to the positive floor gamma_min."""
    model = ACSSubINN(
        in_channels=2,
        out_channels=1,
        hidden_channels=16,
        num_blocks_context=2,
        num_blocks_gain=2,
        gamma_min=gamma_floor,
    )

    # Configure output layer to push raw gain strongly negative so Softplus approaches 0
    with torch.no_grad():
        model.adaptive_gain.exit_conv.weight.zero_()
        model.adaptive_gain.exit_conv.bias.fill_(-20.0)

    batch_size = 1
    h, w = 8, 8
    hh_drop = torch.randn(batch_size, 1, h, w)
    hh_keep = torch.randn(batch_size, 2, h, w)
    mask = torch.ones(batch_size, 1, h, w)

    payload_bits = torch.tensor([[[[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]]]], dtype=torch.uint8)
    payload_bits = payload_bits.repeat(1, 1, 2, 2)  # expand to 8x8
    bipolar_payload = ACSSubINN.bipolar_encode(payload_bits)

    embed_res = model.embed_coefficients(hh_drop, hh_keep, mask, bipolar_payload)
    extract_res = model.extract_coefficients(embed_res.hh_drop_stego, hh_keep)

    # Gain should sit strictly at or within 1e-6 of the floor gamma_min
    assert (embed_res.gamma >= gamma_floor).all()
    assert torch.allclose(embed_res.gamma, torch.tensor(gamma_floor), atol=1e-5), "Gain did not sit at floor"

    # Extraction must remain bit-exact at the floor
    assert (extract_res.extracted_binary == payload_bits).all()


# =============================================================================
# 3. Exact Recovery with Unchanged Context
# =============================================================================

@pytest.mark.parametrize("seed", [101, 202, 303])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_exact_recovery_with_unchanged_context(seed: int, batch_size: int):
    """Verify that bit error rate is strictly 0.000000 when context is unchanged."""
    torch.manual_seed(seed)
    model = ACSSubINN(
        in_channels=2,
        out_channels=1,
        hidden_channels=16,
        num_blocks_context=2,
        num_blocks_gain=2,
        gamma_min=1.0,
    )

    h, w = 16, 16
    hh_drop = torch.randn(batch_size, 1, h, w)
    hh_keep = torch.randn(batch_size, 2, h, w)

    # Random binary mask
    mask = (torch.rand(batch_size, 1, h, w) > 0.4).to(torch.float32)

    # Random binary payload
    payload_bits = torch.randint(0, 2, (batch_size, 1, h, w), dtype=torch.uint8)
    bipolar_payload = ACSSubINN.bipolar_encode(payload_bits)

    embed_res = model.embed_coefficients(hh_drop, hh_keep, mask, bipolar_payload)
    extract_res = model.extract_coefficients(embed_res.hh_drop_stego, hh_keep)

    active_mask = mask == 1.0
    recovered_active = extract_res.extracted_binary[active_mask]
    original_active = payload_bits[active_mask]

    assert (recovered_active == original_active).all()
    ber = float((recovered_active != original_active).sum()) / original_active.numel()
    assert ber == 0.0


# =============================================================================
# 4. Controlled Context Perturbations & Symbol Decision Margin Crossing
# =============================================================================

def test_controlled_context_perturbation_margin_crossing(acs_model):
    """Verify the exact mathematical decision boundary of the sign demodulator.

    Under received context perturbation:
        Delta_rec = gamma * P - delta_context
    For P = +1 (bit 1):
        - Sub-margin: delta_context < gamma  => Delta_rec > 0 => Decoded bit is 1 (CORRECT)
        - Boundary:   delta_context = gamma  => Delta_rec = 0 => Decoded bit is 0 (DECISION ERROR)
        - Super-margin: delta_context > gamma => Delta_rec < 0 => Decoded bit is 0 (BIT FLIP ERROR)
    For P = -1 (bit 0):
        - Sub-margin: delta_context > -gamma => Delta_rec < 0 => Decoded bit is 0 (CORRECT)
        - Boundary:   delta_context = -gamma => Delta_rec = 0 => Decoded bit is 0 (CORRECT at boundary)
        - Super-margin: delta_context < -gamma => Delta_rec > 0 => Decoded bit is 1 (BIT FLIP ERROR)
    """
    batch_size = 1
    h, w = 4, 4
    hh_drop = torch.zeros(batch_size, 1, h, w)
    hh_keep = torch.zeros(batch_size, 2, h, w)
    mask = torch.ones(batch_size, 1, h, w)

    # Test bit 1 (P = +1)
    payload_ones = torch.ones(batch_size, 1, h, w, dtype=torch.uint8)
    bipolar_ones = ACSSubINN.bipolar_encode(payload_ones)
    embed_res = acs_model.embed_coefficients(hh_drop, hh_keep, mask, bipolar_ones)
    gamma = embed_res.gamma  # Known positive gain

    context_rec_clean = acs_model.context_predictor(hh_keep)

    # Case A: Sub-margin perturbation (delta = 0.5 * gamma)
    context_rec_sub = context_rec_clean + 0.5 * gamma
    delta_sub = embed_res.hh_drop_stego - context_rec_sub
    b_hat_sub = acs_model.bipolar_decode(delta_sub)
    assert (b_hat_sub == 1).all(), "Sub-margin perturbation should not flip bit 1"

    # Case B: Super-margin perturbation (delta = 1.5 * gamma)
    context_rec_super = context_rec_clean + 1.5 * gamma
    delta_super = embed_res.hh_drop_stego - context_rec_super
    b_hat_super = acs_model.bipolar_decode(delta_super)
    assert (b_hat_super == 0).all(), "Super-margin perturbation exceeding gamma must cross decision threshold"

    # Test bit 0 (P = -1)
    payload_zeros = torch.zeros(batch_size, 1, h, w, dtype=torch.uint8)
    bipolar_zeros = ACSSubINN.bipolar_encode(payload_zeros)
    embed_zeros = acs_model.embed_coefficients(hh_drop, hh_keep, mask, bipolar_zeros)

    # Case C: Sub-margin perturbation in negative direction (delta = -0.5 * gamma)
    context_rec_sub0 = context_rec_clean - 0.5 * gamma
    delta_sub0 = embed_zeros.hh_drop_stego - context_rec_sub0
    b_hat_sub0 = acs_model.bipolar_decode(delta_sub0)
    assert (b_hat_sub0 == 0).all(), "Sub-margin negative perturbation should not flip bit 0"

    # Case D: Super-margin perturbation in negative direction (delta = -1.5 * gamma)
    context_rec_super0 = context_rec_clean - 1.5 * gamma
    delta_super0 = embed_zeros.hh_drop_stego - context_rec_super0
    b_hat_super0 = acs_model.bipolar_decode(delta_super0)
    assert (b_hat_super0 == 1).all(), "Super-margin negative perturbation exceeding -gamma must flip bit 0"


# =============================================================================
# 5. Unmasked Coefficients Invariance
# =============================================================================

def test_unmasked_coefficients_strictly_unchanged(acs_model):
    """Verify that unmasked positions (M == 0) remain bit-for-bit identical."""
    batch_size = 2
    h, w = 16, 16
    hh_drop = torch.randn(batch_size, 1, h, w)
    hh_keep = torch.randn(batch_size, 2, h, w)

    # Checkerboard mask pattern
    checkerboard = torch.zeros(batch_size, 1, h, w)
    checkerboard[:, :, ::2, ::2] = 1.0
    checkerboard[:, :, 1::2, 1::2] = 1.0

    payload_bits = torch.randint(0, 2, (batch_size, 1, h, w), dtype=torch.uint8)
    bipolar_payload = ACSSubINN.bipolar_encode(payload_bits)

    embed_res = acs_model.embed_coefficients(hh_drop, hh_keep, checkerboard, bipolar_payload)

    unmasked = checkerboard == 0.0
    # Bit-for-bit identity check: maximum absolute difference must be exactly 0.0
    diff = (embed_res.hh_drop_stego[unmasked] - hh_drop[unmasked]).abs()
    assert (diff == 0.0).all()
    assert diff.max().item() == 0.0


# =============================================================================
# 6. Mask and Payload Shape & Range Validation
# =============================================================================

def test_mask_and_payload_shape_and_range_validation(acs_model):
    """Verify strict validation and explicit failure on malformed shapes or out-of-range masks."""
    b, h, w = 2, 8, 8
    hh_drop = torch.randn(b, 1, h, w)
    hh_keep = torch.randn(b, 2, h, w)
    valid_mask = torch.ones(b, 1, h, w)
    valid_payload = torch.ones(b, 1, h, w)

    # 1. Reject 3D mask
    with pytest.raises(ValueError, match="Expected mask to be 4D tensor"):
        acs_model.embed_coefficients(hh_drop, hh_keep, torch.ones(b, h, w), valid_payload)

    # 2. Reject mismatched spatial dimension in mask
    with pytest.raises(ValueError, match="Incompatible mask shape"):
        acs_model.embed_coefficients(hh_drop, hh_keep, torch.ones(b, 1, 16, 16), valid_payload)

    # 3. Reject mask values below 0.0
    neg_mask = valid_mask.clone()
    neg_mask[0, 0, 0, 0] = -0.1
    with pytest.raises(ValueError, match="Mask values must be in range"):
        acs_model.embed_coefficients(hh_drop, hh_keep, neg_mask, valid_payload)

    # 4. Reject mask values above 1.0
    over_mask = valid_mask.clone()
    over_mask[0, 0, 0, 0] = 1.05
    with pytest.raises(ValueError, match="Mask values must be in range"):
        acs_model.embed_coefficients(hh_drop, hh_keep, over_mask, valid_payload)

    # 5. Reject payload with wrong channel count
    with pytest.raises(ValueError, match="Expected payload_bipolar shape"):
        acs_model.embed_coefficients(hh_drop, hh_keep, valid_mask, torch.ones(b, 2, h, w))

    # 6. Reject mismatched batch size in payload
    with pytest.raises(ValueError, match="Expected payload_bipolar shape"):
        acs_model.embed_coefficients(hh_drop, hh_keep, valid_mask, torch.ones(b + 1, 1, h, w))

    # 7. In extract_coefficients, reject invalid mask shape if provided
    embed_res = acs_model.embed_coefficients(hh_drop, hh_keep, valid_mask, valid_payload)
    with pytest.raises(ValueError, match="Incompatible mask shape"):
        acs_model.extract_coefficients(embed_res.hh_drop_stego, hh_keep, mask=torch.ones(b, 1, 16, 16))

    # 8. In extract_coefficients, reject invalid channel count in hh_drop_stego
    with pytest.raises(ValueError, match="Expected hh_drop_stego to have 1 channel"):
        acs_model.extract_coefficients(torch.randn(b, 3, h, w), hh_keep)

    # 9. In extract_coefficients, reject invalid channel count in hh_keep
    with pytest.raises(ValueError, match="Expected hh_keep to have 2 channels"):
        acs_model.extract_coefficients(embed_res.hh_drop_stego, torch.randn(b, 3, h, w))
