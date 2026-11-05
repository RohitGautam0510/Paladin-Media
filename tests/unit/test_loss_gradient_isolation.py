"""Unit tests for Phase 3 Sprint 3.1 Loss-Gradient Isolation and Boundary Audit.

Verifies:
1. Isolated backward passes through differentiable spatial bridge for each candidate loss:
   - BitRecoveryLoss: AdaptiveGainCNN receives non-zero gradient; ContextPredictorCNN has
     present (.grad is not None) but identically zero (L2 norm == 0) gradients under STE.
   - ContextPredictionLoss: ContextPredictorCNN receives non-zero gradient; AdaptiveGainCNN
     has strictly absent (.grad is None) gradients.
   - AdaptiveGainRegularization: AdaptiveGainCNN receives non-zero gradient; ContextPredictorCNN
     has strictly absent (.grad is None) gradients.
   - ImageFidelityLoss: Both networks receive non-zero gradients via IDWT and STE.
2. Formal distinction between absent gradients (.grad is None) and zero gradients (.grad == 0).
3. Context cancellation breakdown when STE dynamic range clipping is enforced (clip_grad=True).
4. Programmatic audit of the dataset boundary interface gap (absence of concrete Dataset/DataLoader).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.neural.losses import (
    AdaptiveGainRegularization,
    BitRecoveryLoss,
    CompositeLoss,
    ContextPredictionLoss,
    ImageFidelityLoss,
)
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_scaffold import (
    DatasetSplitManager,
    TrainingConfig,
    build_model_and_optimizer,
    execute_training_step,
    set_reproducibility_seeds,
)


@pytest.fixture
def synthetic_spatial_setup():
    """Sets up a reproducible small synthetic carrier and payload for gradient checks."""
    set_reproducibility_seeds(42, deterministic=True)
    bridge = NeuralSpatialBridge()
    bridge.train()

    # Synthetic 64x64 non-constant RGB carrier in valid range [20, 235]
    rng = np.random.RandomState(42)
    carrier_np = rng.randint(20, 235, (64, 64, 3), dtype=np.uint8)
    carrier = torch.from_numpy(carrier_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

    # 32x32 payload symbols
    payload_bits = torch.from_numpy(rng.randint(0, 2, (32, 32), dtype=np.uint8)).unsqueeze(0).unsqueeze(0)
    payload_bipolar = payload_bits.float() * 2.0 - 1.0

    # Mask active on first 500 positions
    mask = torch.zeros(1, 1, 32, 32)
    mask.view(-1)[:500] = 1.0

    return bridge, carrier, payload_bipolar, mask


def test_isolated_bit_recovery_loss_gradient_flow(synthetic_spatial_setup):
    """Proves that L_bit generates non-zero gradient for AdaptiveGainCNN but zero for ContextPredictor."""
    bridge, carrier, payload_bipolar, mask = synthetic_spatial_setup
    bridge.zero_grad()

    emb_res = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    ext_res = bridge.extract_spatial(emb_res.stego_image, mask)

    loss_fn = BitRecoveryLoss(formulation="logistic", temperature=1.0)
    loss = loss_fn(ext_res.delta_rec, payload_bipolar, mask)
    loss.backward()

    # 1. AdaptiveGainCNN: present, finite, strictly non-zero
    gain_grads = [p.grad for p in bridge.acs_model.adaptive_gain.parameters()]
    assert len(gain_grads) > 0
    assert all(g is not None for g in gain_grads), "AdaptiveGain gradients must be present"
    assert all(torch.isfinite(g).all() for g in gain_grads), "AdaptiveGain gradients must be finite"
    gain_l2 = torch.sqrt(sum((g ** 2).sum() for g in gain_grads)).item()
    assert gain_l2 > 1.0, f"AdaptiveGain L2 norm should be significant, got {gain_l2}"

    # 2. ContextPredictorCNN: present (.grad is not None) due to graph inclusion,
    # but identically zero due to STE context cancellation
    ctx_grads = [p.grad for p in bridge.acs_model.context_predictor.parameters()]
    assert len(ctx_grads) > 0
    assert all(g is not None for g in ctx_grads), "ContextPredictor gradients are present in the autograd graph"
    assert all(torch.isfinite(g).all() for g in ctx_grads), "ContextPredictor gradients must be finite"
    ctx_l2 = torch.sqrt(sum((g ** 2).sum() for g in ctx_grads)).item()
    assert ctx_l2 == pytest.approx(0.0, abs=1e-6), (
        f"ContextPredictor must have zero gradient under lossless/STE roundtrip, got {ctx_l2}"
    )


def test_isolated_context_prediction_loss_gradient_flow(synthetic_spatial_setup):
    """Proves that L_ctx generates gradients ONLY for ContextPredictor; AdaptiveGain is strictly ABSENT."""
    bridge, carrier, payload_bipolar, mask = synthetic_spatial_setup
    bridge.zero_grad()

    emb_res = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    hh_drop_orig = emb_res.subbands_original[3][:, 0:1, :, :]
    context_pred = emb_res.acs_embedding_result.context

    loss_fn = ContextPredictionLoss(formulation="smooth_l1", beta=1.0)
    loss = loss_fn(context_pred, hh_drop_orig)
    loss.backward()

    # 1. ContextPredictorCNN: present, finite, strictly non-zero
    ctx_grads = [p.grad for p in bridge.acs_model.context_predictor.parameters()]
    assert all(g is not None for g in ctx_grads), "ContextPredictor gradients must be present"
    assert all(torch.isfinite(g).all() for g in ctx_grads), "ContextPredictor gradients must be finite"
    ctx_l2 = torch.sqrt(sum((g ** 2).sum() for g in ctx_grads)).item()
    assert ctx_l2 > 1.0, f"ContextPredictor L2 norm should be significant, got {ctx_l2}"

    # 2. AdaptiveGainCNN: strictly ABSENT (not None vs zero)
    gain_grads = [p.grad for p in bridge.acs_model.adaptive_gain.parameters()]
    assert all(g is None for g in gain_grads), "AdaptiveGain gradients must be strictly ABSENT (None) under L_ctx"


def test_isolated_gain_regularization_gradient_flow(synthetic_spatial_setup):
    """Proves that L_gain generates gradients ONLY for AdaptiveGain; ContextPredictor is strictly ABSENT."""
    bridge, carrier, payload_bipolar, mask = synthetic_spatial_setup
    bridge.zero_grad()

    emb_res = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    gamma = emb_res.acs_embedding_result.gamma

    loss_fn = AdaptiveGainRegularization(formulation="excess_l1", gamma_min=1.0)
    loss = loss_fn(gamma, mask)
    loss.backward()

    # 1. AdaptiveGainCNN: present, finite, strictly non-zero
    gain_grads = [p.grad for p in bridge.acs_model.adaptive_gain.parameters()]
    assert all(g is not None for g in gain_grads), "AdaptiveGain gradients must be present"
    assert all(torch.isfinite(g).all() for g in gain_grads), "AdaptiveGain gradients must be finite"
    gain_l2 = torch.sqrt(sum((g ** 2).sum() for g in gain_grads)).item()
    assert gain_l2 > 1.0, f"AdaptiveGain L2 norm should be significant, got {gain_l2}"

    # 2. ContextPredictorCNN: strictly ABSENT (None)
    ctx_grads = [p.grad for p in bridge.acs_model.context_predictor.parameters()]
    assert all(g is None for g in ctx_grads), "ContextPredictor gradients must be strictly ABSENT (None) under L_gain"


def test_isolated_image_fidelity_loss_gradient_flow(synthetic_spatial_setup):
    """Proves that L_img backpropagates to BOTH submodules via differentiable IDWT and STE."""
    bridge, carrier, payload_bipolar, mask = synthetic_spatial_setup
    bridge.zero_grad()

    emb_res = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)

    loss_fn = ImageFidelityLoss(formulation="l1")
    loss = loss_fn(emb_res.stego_image, carrier)
    loss.backward()

    ctx_grads = [p.grad for p in bridge.acs_model.context_predictor.parameters()]
    gain_grads = [p.grad for p in bridge.acs_model.adaptive_gain.parameters()]

    assert all(g is not None for g in ctx_grads), "ContextPredictor gradients must be present under L_img"
    assert all(g is not None for g in gain_grads), "AdaptiveGain gradients must be present under L_img"

    ctx_l2 = torch.sqrt(sum((g ** 2).sum() for g in ctx_grads)).item()
    gain_l2 = torch.sqrt(sum((g ** 2).sum() for g in gain_grads)).item()

    assert ctx_l2 > 0.01, f"ContextPredictor L2 norm under L_img should be > 0.01, got {ctx_l2}"
    assert gain_l2 > 0.01, f"AdaptiveGain L2 norm under L_img should be > 0.01, got {gain_l2}"


def test_clipping_leakage_breaks_cancellation():
    """Proves that when STE dynamic range clipping is active (clip_grad=True) on clipped pixels,
    exact context cancellation is broken and non-zero gradient reaches ContextPredictor."""
    bridge = NeuralSpatialBridge()
    bridge.quantizer.clip_grad = True

    # Carrier at maximum value to trigger clipping when gamma is added
    carrier = torch.full((1, 3, 32, 32), 255.0)
    payload_bipolar = torch.ones(1, 1, 16, 16)
    mask = torch.ones(1, 1, 16, 16)

    emb_res = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    assert emb_res.clip_stats["num_clipped_total"] > 0, "Precondition: clipping must occur"

    ext_res = bridge.extract_spatial(emb_res.stego_image, mask)
    loss_fn = BitRecoveryLoss(formulation="logistic", temperature=1.0)
    loss = loss_fn(ext_res.delta_rec, payload_bipolar, mask)
    loss.backward()

    ctx_grads = [p.grad for p in bridge.acs_model.context_predictor.parameters() if p.grad is not None]
    ctx_l2 = torch.sqrt(sum((g ** 2).sum() for g in ctx_grads)).item()

    # The gradient is no longer exactly 0.0 due to clipping leakage
    assert ctx_l2 > 0.01, f"Expected non-zero context gradient leakage when clipping occurs, got {ctx_l2}"


def test_dataset_boundary_loader_absence_audit():
    """Audits and documents the architectural absence of a concrete Dataset/DataLoader in training_scaffold."""
    import src.neural.training_scaffold as ts

    # 1. DatasetSplitManager exists and rejects 'test' split
    mgr = ts.DatasetSplitManager()
    with pytest.raises(ValueError, match="Accessing the held-out 'test' split"):
        mgr.get_records_for_training("test")

    # 2. Check that no concrete Dataset or DataLoader is defined in training_scaffold
    exported_names = dir(ts)
    assert "DIV2KDataset" not in exported_names, "Concrete DIV2KDataset should not exist yet in Sprint 3 scaffold"
    assert "build_dataloader" not in exported_names, "DataLoader builder should not exist yet in Sprint 3 scaffold"

    # 3. Document that execute_training_step accepts raw unverified tensors
    # proving the gap that test protection is not yet enforced end-to-end at data loading boundary
    bridge, opt, _ = build_model_and_optimizer(TrainingConfig(learning_rate=1e-4))
    raw_carrier = torch.randn(1, 3, 64, 64)
    raw_payload = torch.randint(0, 2, (1, 1, 32, 32)).float()
    raw_mask = torch.ones(1, 1, 32, 32)
    comp_loss = CompositeLoss()
    # This executes successfully without knowing whether raw_carrier came from train, val, or test split
    step_res = execute_training_step(
        bridge=bridge,
        optimizer=opt,
        composite_loss=comp_loss,
        carrier=raw_carrier,
        payload_bits=raw_payload,
        mask=raw_mask,
        run_backward=True,
    )
    assert step_res.is_grad_finite is True
    assert step_res.optimizer_step_executed is False
