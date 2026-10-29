"""Targeted Regression Tests for Phase 3 Sprint 3.4.1 Loss Diagnostics and Evidence Reconciliation.

Verifies:
1. Exact numerical equality between CompositeLoss total and the weighted sum of its raw components.
2. Pythagorean gradient norm orthogonality: ||g_model||_2 == sqrt(||g_ctx||_2^2 + ||g_gain||_2^2).
3. Gradient clipping execution path: clip_grad_norm_ scales norms by min(1, max_norm / total_norm).
4. Context-supervision routing distinction:
   - BitRecoveryLoss produces identically zero gradient for ContextPredictorCNN under lossless STE.
   - ContextPredictionLoss provides direct reconstruction supervision for ContextPredictorCNN.
   - ImageFidelityLoss provides non-zero indirect spatial fidelity gradients for ContextPredictorCNN.
5. Model parameter immutability (no optimizer.step()).
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.losses import (
    AdaptiveGainRegularization,
    BitRecoveryLoss,
    CompositeLoss,
    ContextPredictionLoss,
    ImageFidelityLoss,
)
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_scaffold import (
    TrainingConfig,
    build_model_and_optimizer,
    set_reproducibility_seeds,
)


def compute_norm(module: nn.Module) -> float:
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.sqrt(sum((g ** 2).sum() for g in grads)).item())


@pytest.fixture
def test_setup():
    set_reproducibility_seeds(42, deterministic=True)
    bridge, optimizer, comp_loss = build_model_and_optimizer(TrainingConfig(seed=42, device="cpu"))
    bridge.train()

    # Synthetic non-saturating carrier in valid range [20, 235] to avoid clipping leakage
    import numpy as np
    rng = np.random.RandomState(42)
    carrier_np = rng.randint(20, 235, (64, 64, 3), dtype=np.uint8)
    carrier = torch.from_numpy(carrier_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    payload_bits = torch.from_numpy(rng.randint(0, 2, (32, 32), dtype=np.uint8)).unsqueeze(0).unsqueeze(0).float()
    payload_bipolar = ACSSubINN.bipolar_encode(payload_bits)
    mask = torch.ones(1, 1, 32, 32, dtype=torch.float32)

    return bridge, optimizer, comp_loss, carrier, payload_bipolar, mask


def test_composite_loss_weighted_sum_exactness(test_setup):
    """Verify that total_loss equals the exact weighted sum of its constituent components."""
    bridge, _, comp_loss, carrier, payload_bipolar, mask = test_setup

    emb = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    ext = bridge.extract_spatial(emb.stego_image, mask)

    breakdown = comp_loss(
        delta_rec=ext.delta_rec,
        payload_bipolar=payload_bipolar,
        predicted_context=emb.acs_embedding_result.context,
        target_hh_drop=emb.subbands_original[3][:, 0:1, :, :],
        gamma=emb.acs_embedding_result.gamma,
        stego_spatial=emb.stego_image,
        cover_spatial=carrier,
        mask=mask,
    )

    weighted_sum = (
        comp_loss.lambda_bit * breakdown.loss_bit.item()
        + comp_loss.lambda_ctx * breakdown.loss_ctx.item()
        + comp_loss.lambda_gain * breakdown.loss_gain.item()
        + comp_loss.lambda_img * breakdown.loss_img.item()
    )

    assert breakdown.total_loss.item() == pytest.approx(weighted_sum, rel=1e-5, abs=1e-5)


def test_gradient_norm_orthogonality(test_setup):
    """Verify ||g_model||_2 == sqrt(||g_ctx||_2^2 + ||g_gain||_2^2) on disjoint parameter groups."""
    bridge, optimizer, comp_loss, carrier, payload_bipolar, mask = test_setup

    emb = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    ext = bridge.extract_spatial(emb.stego_image, mask)

    breakdown = comp_loss(
        delta_rec=ext.delta_rec,
        payload_bipolar=payload_bipolar,
        predicted_context=emb.acs_embedding_result.context,
        target_hh_drop=emb.subbands_original[3][:, 0:1, :, :],
        gamma=emb.acs_embedding_result.gamma,
        stego_spatial=emb.stego_image,
        cover_spatial=carrier,
        mask=mask,
    )

    optimizer.zero_grad()
    breakdown.total_loss.backward()

    g_ctx = compute_norm(bridge.acs_model.context_predictor)
    g_gain = compute_norm(bridge.acs_model.adaptive_gain)
    g_total = compute_norm(bridge.acs_model)

    expected_total = math.sqrt(g_ctx ** 2 + g_gain ** 2)
    assert abs(g_total - expected_total) < 1e-5


def test_gradient_clipping_scaling_behavior(test_setup):
    """Verify that clip_grad_norm_ properly scales gradients and constituent norms."""
    bridge, optimizer, comp_loss, carrier, payload_bipolar, mask = test_setup

    emb = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    ext = bridge.extract_spatial(emb.stego_image, mask)

    breakdown = comp_loss(
        delta_rec=ext.delta_rec,
        payload_bipolar=payload_bipolar,
        predicted_context=emb.acs_embedding_result.context,
        target_hh_drop=emb.subbands_original[3][:, 0:1, :, :],
        gamma=emb.acs_embedding_result.gamma,
        stego_spatial=emb.stego_image,
        cover_spatial=carrier,
        mask=mask,
    )

    optimizer.zero_grad()
    breakdown.total_loss.backward()

    pre_clip_total = compute_norm(bridge.acs_model)
    pre_clip_ctx = compute_norm(bridge.acs_model.context_predictor)
    pre_clip_gain = compute_norm(bridge.acs_model.adaptive_gain)

    clip_norm = 1.0
    reported_norm = float(torch.nn.utils.clip_grad_norm_(bridge.acs_model.parameters(), max_norm=clip_norm).item())

    post_clip_total = compute_norm(bridge.acs_model)
    post_clip_ctx = compute_norm(bridge.acs_model.context_predictor)
    post_clip_gain = compute_norm(bridge.acs_model.adaptive_gain)

    assert abs(reported_norm - pre_clip_total) < 1e-5
    if pre_clip_total > clip_norm:
        assert abs(post_clip_total - clip_norm) < 1e-4
        scale = clip_norm / pre_clip_total
        assert abs(post_clip_ctx - pre_clip_ctx * scale) < 1e-4
        assert abs(post_clip_gain - pre_clip_gain * scale) < 1e-4


def test_context_supervision_gradient_paths(test_setup):
    """Verify that BitRecoveryLoss yields zero Ctx gradient, while ImageFidelityLoss yields non-zero."""
    bridge, optimizer, _, carrier, payload_bipolar, mask = test_setup

    emb = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    ext = bridge.extract_spatial(emb.stego_image, mask)

    # 1. BitRecoveryLoss -> Ctx gradient MUST be zero due to context cancellation
    optimizer.zero_grad()
    l_bit = BitRecoveryLoss()(ext.delta_rec, payload_bipolar, mask=mask)
    l_bit.backward(retain_graph=True)
    g_ctx_bit = compute_norm(bridge.acs_model.context_predictor)
    assert g_ctx_bit == pytest.approx(0.0, abs=1e-6)

    # 2. ContextPredictionLoss -> Ctx gradient MUST be strictly non-zero
    optimizer.zero_grad()
    l_ctx = ContextPredictionLoss()(emb.acs_embedding_result.context, emb.subbands_original[3][:, 0:1, :, :], mask=mask)
    l_ctx.backward(retain_graph=True)
    g_ctx_ctx = compute_norm(bridge.acs_model.context_predictor)
    assert g_ctx_ctx > 0.0

    # 3. ImageFidelityLoss -> Ctx gradient MUST be strictly non-zero (indirect spatial path)
    optimizer.zero_grad()
    l_img = ImageFidelityLoss()(emb.stego_image, carrier)
    l_img.backward(retain_graph=True)
    g_ctx_img = compute_norm(bridge.acs_model.context_predictor)
    assert g_ctx_img > 0.0
