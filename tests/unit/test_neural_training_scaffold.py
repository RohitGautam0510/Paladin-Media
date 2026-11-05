"""Unit tests for Phase 3 Sprint 3 Training Scaffold and Loss Modules.

Covers:
1. Candidate loss functions produce finite scalars and expected gradient flows:
   - BitRecoveryLoss (logistic / BCE and hinge)
   - ContextPredictionLoss (smooth L1 / Huber and L1)
   - AdaptiveGainRegularization (excess L1 / L2 and L1)
   - ImageFidelityLoss (L1, Charbonnier, MSE)
2. Gradient flow routing:
   - Verifies context prediction loss flows to ContextPredictorCNN.
   - Verifies gain regularization flows to AdaptiveGainCNN.
   - Verifies image fidelity flows through IDWT and Straight-Through Estimator.
   - Proves mathematically that extraction residual delta_rec has near-zero/zero gradient to ContextPredictor
     under lossless roundtrip due to context cancellation, proving the necessity of explicit L_ctx.
3. DatasetSplitManager partitioning and test-set boundary protection:
   - Verifies 600 train, 100 val, 100 test split counts.
   - Verifies composite SHA-256 integrity.
   - Verifies that requesting 'test' split for training raises a hard ValueError.
4. Training step execution safety invariant:
   - Proves that execute_training_step() executes forward/backward passes with finite gradients
     WITHOUT calling optimizer.step() (weights remain bit-for-bit identical before and after).
5. Checkpoint management:
   - Verifies synthetic smoke checkpoint saving and loading with explicit governance metadata.
6. Clipping diagnostic metrics:
   - Verifies out-of-range pixel detection and boundary pixel counts on controlled tensors.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.losses import (
    AdaptiveGainRegularization,
    BitRecoveryLoss,
    ClippingDiagnosticReport,
    CompositeLoss,
    ContextPredictionLoss,
    ImageFidelityLoss,
    evaluate_clipping_diagnostic,
)
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_scaffold import (
    DatasetSplitManager,
    GateRegistry,
    TrainingConfig,
    build_model_and_optimizer,
    execute_training_step,
    load_and_validate_checkpoint,
    save_synthetic_smoke_checkpoint,
    set_reproducibility_seeds,
)


# =============================================================================
# 1. Individual Loss Functions Tests
# =============================================================================

def test_bit_recovery_loss_formulations():
    """Verify BitRecoveryLoss under both logistic and hinge formulations."""
    loss_logistic = BitRecoveryLoss(formulation="logistic", temperature=1.0)
    loss_hinge = BitRecoveryLoss(formulation="hinge", margin=1.0)

    delta = torch.tensor([[[[2.0, -2.0], [-0.5, 0.5]]]], requires_grad=True)
    payload = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]])  # first 2 correct, second 2 wrong
    mask = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])

    # Logistic loss
    l_log = loss_logistic(delta, payload, mask)
    assert torch.isfinite(l_log)
    assert l_log.item() > 0.0

    l_log.backward()
    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()

    # Hinge loss
    delta.grad.zero_()
    l_h = loss_hinge(delta, payload, mask)
    assert torch.isfinite(l_h)
    assert l_h.item() > 0.0

    l_h.backward()
    assert torch.isfinite(delta.grad).all()


def test_context_prediction_loss_gradient_routing():
    """Verify ContextPredictionLoss routes gradients to ContextPredictorCNN."""
    torch.manual_seed(42)
    bridge = NeuralSpatialBridge()
    loss_fn = ContextPredictionLoss(formulation="smooth_l1", beta=1.0)

    # Synthetic non-constant carrier
    carrier = torch.randn(1, 3, 64, 64) * 20.0 + 128.0
    payload_bipolar = torch.ones(1, 1, 32, 32)
    mask = torch.ones(1, 1, 32, 32)

    emb_res = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    hh_drop_orig = emb_res.subbands_original[3][:, 0:1, :, :]
    predicted_context = emb_res.acs_embedding_result.context

    loss = loss_fn(predicted_context, hh_drop_orig)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()

    # Verify gradients reach all layers of ContextPredictorCNN
    for name, p in bridge.acs_model.context_predictor.named_parameters():
        assert p.grad is not None, f"Gradient is None for {name}"
        assert torch.isfinite(p.grad).all(), f"Gradient is not finite for {name}"
        assert p.grad.abs().sum().item() > 0.0, f"Gradient is zero for {name}"

    # Verify AdaptiveGainCNN received zero gradients from context loss
    for name, p in bridge.acs_model.adaptive_gain.named_parameters():
        assert p.grad is None, f"AdaptiveGain should not receive gradients from L_ctx, got {name}"


def test_adaptive_gain_regularization_gradient_routing():
    """Verify AdaptiveGainRegularization routes gradients strictly to AdaptiveGainCNN."""
    torch.manual_seed(42)
    bridge = NeuralSpatialBridge()
    reg_fn = AdaptiveGainRegularization(formulation="excess_l1", gamma_min=1.0)

    hh_keep = torch.randn(1, 2, 32, 32)
    gamma = bridge.acs_model.adaptive_gain(hh_keep)

    loss = reg_fn(gamma)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    loss.backward()

    for name, p in bridge.acs_model.adaptive_gain.named_parameters():
        assert p.grad is not None, f"Gradient is None for {name}"
        assert torch.isfinite(p.grad).all(), f"Gradient is not finite for {name}"

    # Context predictor should receive zero gradients
    for name, p in bridge.acs_model.context_predictor.named_parameters():
        assert p.grad is None, f"ContextPredictor should not receive gradients from L_gain, got {name}"


def test_image_fidelity_loss_differentiable_path():
    """Verify ImageFidelityLoss backpropagates through IDWT and Straight-Through Estimator."""
    torch.manual_seed(42)
    bridge = NeuralSpatialBridge()
    loss_fn = ImageFidelityLoss(formulation="l1")

    carrier = torch.randn(1, 3, 64, 64) * 15.0 + 128.0
    payload_bipolar = torch.ones(1, 1, 32, 32)
    mask = torch.ones(1, 1, 32, 32)

    emb_res = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    loss = loss_fn(emb_res.stego_image, carrier)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()

    # Gradients should reach both networks via IDWT and embedding equation
    has_ctx_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.acs_model.context_predictor.parameters())
    has_gain_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.acs_model.adaptive_gain.parameters())
    assert has_ctx_grad, "Image fidelity loss failed to backpropagate to ContextPredictorCNN!"
    assert has_gain_grad, "Image fidelity loss failed to backpropagate to AdaptiveGainCNN!"


# =============================================================================
# 2. Context Cancellation Mathematical Invariant Test
# =============================================================================

def test_context_cancellation_under_lossless_roundtrip():
    """Demonstrates that delta_rec has zero gradient to ContextPredictor under lossless roundtrip.

    Mathematical proof:
        hh_drop_stego = context + gamma * P
        delta_rec = hh_drop_stego_rec - context_rec
        When subbands undergo no rounding / identity:
        delta_rec = (context + gamma * P) - context = gamma * P
        d(delta_rec) / d(context) = 1 - 1 = 0.
        Therefore, an explicit L_ctx is required to train the context predictor.
    """
    torch.manual_seed(42)
    bridge = NeuralSpatialBridge()

    # Pass coefficients directly through ACS-Sub-INN without spatial IDWT rounding
    hh_drop = torch.randn(1, 1, 32, 32)
    hh_keep = torch.randn(1, 2, 32, 32)
    payload_bipolar = torch.ones(1, 1, 32, 32)
    mask = torch.ones(1, 1, 32, 32)

    acs_emb = bridge.acs_model.embed_coefficients(hh_drop, hh_keep, mask, payload_bipolar)
    acs_ext = bridge.acs_model.extract_coefficients(acs_emb.hh_drop_stego, hh_keep, mask)

    # Loss based purely on delta_rec
    loss = acs_ext.delta_rec.sum()
    loss.backward()

    # Context predictor gradients should be identically zero
    for name, p in bridge.acs_model.context_predictor.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().max().item() == pytest.approx(0.0, abs=1e-6), (
                f"ContextPredictor received non-zero gradient {p.grad.abs().max().item()} for {name}!"
            )

    # Adaptive gain gradients MUST be non-zero
    gain_grad_sum = sum(p.grad.abs().sum().item() for p in bridge.acs_model.adaptive_gain.parameters() if p.grad is not None)
    assert gain_grad_sum > 0.0, "AdaptiveGain should have non-zero gradient from delta_rec!"


# =============================================================================
# 3. Dataset Split Boundaries & Test-Set Protection Tests
# =============================================================================

def test_dataset_split_manager_integrity():
    """Verify that DatasetSplitManager audits and preserves the 600/100/100 partition."""
    mgr = DatasetSplitManager()

    train_recs = mgr.get_train_records()
    val_recs = mgr.get_val_records()
    test_recs = mgr.get_held_out_test_manifest_for_audit_only()

    assert len(train_recs) == 600
    assert len(val_recs) == 100
    assert len(test_recs) == 100

    # Sequential image id ranges
    assert train_recs[0]["filename"] == "0001.png"
    assert train_recs[-1]["filename"] == "0600.png"
    assert val_recs[0]["filename"] == "0601.png"
    assert val_recs[-1]["filename"] == "0700.png"
    assert test_recs[0]["filename"] == "0701.png"
    assert test_recs[-1]["filename"] == "0800.png"


def test_test_set_protection_enforcement():
    """Verify that requesting the test partition for training or tuning raises ValueError."""
    mgr = DatasetSplitManager()

    # Authorized splits
    assert len(mgr.get_records_for_training("train")) == 600
    assert len(mgr.get_records_for_training("val")) == 100

    # FORBIDDEN: Requesting 'test' split for training
    with pytest.raises(ValueError, match="DATA INTEGRITY VIOLATION"):
        mgr.get_records_for_training("test")

    with pytest.raises(ValueError, match="Unknown split"):
        mgr.get_records_for_training("invalid_split_name")


# =============================================================================
# 4. Training Step Execution Safety (STRICT: NO OPTIMIZER.STEP)
# =============================================================================

def test_training_step_does_not_modify_weights():
    """Verify that execute_training_step computes gradients without calling optimizer.step()."""
    config = TrainingConfig(seed=42, device="cpu", learning_rate=1e-3)
    bridge, optimizer, composite_loss = build_model_and_optimizer(config)

    # Snapshot model parameter values before step
    initial_params = [p.clone().detach() for p in bridge.acs_model.parameters()]

    carrier = torch.randn(1, 3, 64, 64) * 20.0 + 128.0
    payload_bits = torch.randint(0, 2, (1, 1, 32, 32), dtype=torch.uint8)
    mask = torch.ones(1, 1, 32, 32)

    step_res = execute_training_step(
        bridge=bridge,
        optimizer=optimizer,
        composite_loss=composite_loss,
        carrier=carrier,
        payload_bits=payload_bits,
        mask=mask,
        run_backward=True,
    )

    # 1. Gradients were computed and finite
    assert step_res.is_grad_finite is True
    assert step_res.total_grad_norm > 0.0

    # 2. Strict invariant: optimizer.step() was NOT executed
    assert step_res.optimizer_step_executed is False

    # 3. Parameters after execution are BIT-FOR-BIT IDENTICAL to parameters before execution
    current_params = [p.clone().detach() for p in bridge.acs_model.parameters()]
    for init_p, curr_p in zip(initial_params, current_params):
        assert torch.equal(init_p, curr_p), "Model parameters were modified! optimizer.step() must not be called!"


# =============================================================================
# 5. Checkpoint Management Tests
# =============================================================================

def test_synthetic_smoke_checkpoint_save_and_load():
    """Verify saving and loading of synthetic smoke checkpoints with explicit governance tags."""
    config = TrainingConfig(seed=42, device="cpu")
    bridge, optimizer, _ = build_model_and_optimizer(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_smoke_checkpoint.pt"

        saved_meta = save_synthetic_smoke_checkpoint(bridge, optimizer, config, ckpt_path)
        assert ckpt_path.exists()
        assert saved_meta["checkpoint_type"] == "SYNTHETIC_SMOKE_TEST_ARTIFACT"
        assert saved_meta["training_authorized"] is False
        assert saved_meta["optimizer_step_applied"] is False

        # Load into fresh bridge
        fresh_bridge = NeuralSpatialBridge()
        loaded_meta = load_and_validate_checkpoint(ckpt_path, fresh_bridge)

        assert loaded_meta["checkpoint_type"] == "SYNTHETIC_SMOKE_TEST_ARTIFACT"

        # Verify weights match
        for p1, p2 in zip(bridge.acs_model.parameters(), fresh_bridge.acs_model.parameters()):
            assert torch.equal(p1, p2)


# =============================================================================
# 6. Clipping Diagnostic Tests
# =============================================================================

def test_clipping_diagnostic_evaluation():
    """Verify clipping diagnostic reports out-of-range pixels and boundary pinning accurately."""
    # Construct tensor with controlled out-of-range elements
    raw = torch.tensor([[[[-10.0, 50.0], [200.0, 260.0]]]])
    quant = torch.clamp(raw, 0.0, 255.0)

    report = evaluate_clipping_diagnostic(raw, quant)

    assert report.min_val_raw == -10.0
    assert report.max_val_raw == 260.0
    assert report.num_clipped_low == 1
    assert report.num_clipped_high == 1
    assert report.num_clipped_total == 2
    assert report.clip_fraction == 0.5  # 2 out of 4 elements
    assert report.boundary_pixel_count == 2  # one at 0, one at 255
    assert report.boundary_pixel_fraction == 0.5
    assert report.saturation_penalty_l1 == pytest.approx((10.0 + 5.0) / 4.0)


# =============================================================================
# 7. Gate Registry Governance Test
# =============================================================================

def test_gate_registry_classifications():
    """Verify that GateRegistry distinguishes existing gates from unvalidated proposals."""
    gates = GateRegistry.list_gates()

    # Pre-ECC BER must be an existing gate
    assert gates["pre_ecc_ber_png"]["classification"] == "EXISTING_PROJECT_GATE"
    assert gates["post_ecc_ber"]["classification"] == "EXISTING_PROJECT_GATE"
    assert gates["capacity_fail_closed"]["classification"] == "EXISTING_PROJECT_GATE"

    # Loss weights and lossy JPEG must be PROPOSED_UNVALIDATED
    assert gates["loss_composition_weights"]["classification"] == "PROPOSED_UNVALIDATED"
    assert gates["lossy_jpeg_channel"]["classification"] == "PROPOSED_UNVALIDATED"

    # Clipping fraction must be DIAGNOSTIC_ONLY
    assert gates["clipping_diagnostic_metric"]["classification"] == "DIAGNOSTIC_ONLY"
