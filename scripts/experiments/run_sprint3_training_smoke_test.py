"""Experiment Script: Phase 3 Sprint 3 Training Scaffold Synthetic Smoke Test.

EXECUTION CONTRACT (NO MODEL TRAINING):
This script executes a synthetic single-step smoke test for the Phase 3 ACS-Sub-INN v3
training scaffold. It demonstrates:
1. Instantiation of NeuralSpatialBridge, AdamW optimizer, and CompositeLoss.
2. Differentiable forward pass on a small synthetic carrier (64x64) with pseudo-random
   encrypted-like bit payloads (no plaintext enters the network).
3. Evaluation of all candidate loss components:
   - BitRecoveryLoss (logistic / BCE surrogate)
   - ContextPredictionLoss (smooth L1 / Huber)
   - AdaptiveGainRegularization (excess gain L1)
   - ImageFidelityLoss (spatial L1)
4. Evaluation of non-optimized ClippingDiagnostic metrics.
5. Execution of backward() to verify finite gradients across all trainable parameters.
6. STRICT INVARIANT: ZERO optimizer updates. No optimizer.step() is called;
   parameters before and after execution are bit-for-bit identical.
7. Creation and validation of a clearly labelled synthetic smoke-test checkpoint.
8. Output of versioned results to docs/experiments/sprint3_training_scaffold_results.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.losses import (
    AdaptiveGainRegularization,
    BitRecoveryLoss,
    ClippingDiagnosticReport,
    CompositeLoss,
    ContextPredictionLoss,
    ImageFidelityLoss,
    LossBreakdown,
    evaluate_clipping_diagnostic,
)
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_scaffold import (
    DatasetSplitManager,
    GateRegistry,
    TrainingConfig,
    TrainingStepResult,
    build_model_and_optimizer,
    execute_training_step,
    load_and_validate_checkpoint,
    save_synthetic_smoke_checkpoint,
    set_reproducibility_seeds,
)


def run_smoke_test() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 3 SPRINT 3: TRAINING SCAFFOLD SYNTHETIC SMOKE TEST (NO TRAINING)")
    print("=" * 80)

    # 1. Reproducibility setup
    seed_report = set_reproducibility_seeds(seed=42, deterministic=True)
    print(f"[1] Reproducibility seeds configured: {seed_report}")

    # 2. Audit Dataset Split Boundaries
    split_mgr = DatasetSplitManager()
    train_count = len(split_mgr.get_train_records())
    val_count = len(split_mgr.get_val_records())
    test_count = len(split_mgr.get_held_out_test_manifest_for_audit_only())
    print(f"[2] Dataset split verified: Train={train_count}, Val={val_count}, Held-out Test={test_count}")

    test_protection_verified = False
    try:
        split_mgr.get_records_for_training("test")
    except ValueError as exc:
        test_protection_verified = True
        print(f"    Held-out test protection verified: successfully caught unauthorized access ({exc})")

    # 3. Model, Optimizer, and Loss Construction
    config = TrainingConfig(
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        deterministic=True,
        batch_size=1,
        crop_size=64,
        learning_rate=1e-4,
        weight_decay=1e-5,
        grad_clip_norm=1.0,
        lambda_bit=1.0,
        lambda_ctx=0.5,
        lambda_gain=0.01,
        lambda_img=0.1,
        checkpoint_dir="checkpoints/sprint3_smoke",
    )
    print(f"[3] Configuration constructed on device: {config.device.upper()}")

    bridge, optimizer, composite_loss = build_model_and_optimizer(config)
    print(f"    NeuralSpatialBridge initialized (gamma_min={bridge.acs_model.gamma_min})")
    print(f"    CompositeLoss initialized: {composite_loss.STATUS}")

    # 4. Generate Synthetic Inputs (NO PLAINTEXT)
    device = torch.device(config.device)
    rng = np.random.RandomState(42)

    # Synthetic carrier (64x64, 3 channels)
    carrier_np = rng.randint(20, 235, (64, 64, 3), dtype=np.uint8)
    carrier = torch.from_numpy(carrier_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

    # Synthetic encrypted-like payload bits (32x32, 1 channel)
    payload_bits_np = rng.randint(0, 2, (32, 32), dtype=np.uint8)
    payload_bits = torch.from_numpy(payload_bits_np).unsqueeze(0).unsqueeze(0).to(device)

    # Active mask (occupying 500 positions)
    mask = torch.zeros(1, 1, 32, 32, device=device)
    mask.view(-1)[:500] = 1.0

    print(f"[4] Synthetic carrier generated: shape={tuple(carrier.shape)}, range=[{carrier.min().item():.1f}, {carrier.max().item():.1f}]")
    print(f"    Synthetic payload bits generated: shape={tuple(payload_bits.shape)}, active_mask={mask.sum().item():.0f} positions")

    # Snapshot parameter state before execution to prove NO optimizer updates
    initial_param_hashes = {
        name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
        for name, p in bridge.acs_model.named_parameters()
    }

    # 5. Execute Training Step (Forward + Differentiable Loss + Backward)
    print("\n[5] Executing forward and backward pass...")
    step_result: TrainingStepResult = execute_training_step(
        bridge=bridge,
        optimizer=optimizer,
        composite_loss=composite_loss,
        carrier=carrier,
        payload_bits=payload_bits,
        mask=mask,
        run_backward=True,
        clip_grad_norm=config.grad_clip_norm,
    )

    # 6. Verify Parameter Identity (STRICT SAFETY: NO OPTIMIZER STEP)
    current_param_hashes = {
        name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
        for name, p in bridge.acs_model.named_parameters()
    }

    parameters_unmodified = True
    for name, init_hash in initial_param_hashes.items():
        curr_hash = current_param_hashes[name]
        if init_hash != curr_hash:
            parameters_unmodified = False
            print(f"ERROR: Parameter {name} was modified!")

    print(f"[6] Parameter Invariant Verified:")
    print(f"    optimizer_step_executed: {step_result.optimizer_step_executed} (Expected: False)")
    print(f"    All parameters bit-for-bit identical before and after: {parameters_unmodified}")

    # 7. Detailed Loss & Gradient Analysis
    breakdown = step_result.loss_breakdown.to_dict()
    print("\n[7] Loss Component Breakdown:")
    print(f"    Total Composite Loss:   {breakdown['total_loss']:.6f}")
    print(f"    Bit Recovery Loss:      {breakdown['loss_bit']:.6f}  (weight={breakdown['lambda_bit']})")
    print(f"    Context Prediction Loss:{breakdown['loss_ctx']:.6f}  (weight={breakdown['lambda_ctx']})")
    print(f"    Adaptive Gain Reg:      {breakdown['loss_gain']:.6f}  (weight={breakdown['lambda_gain']})")
    print(f"    Image Fidelity Loss:    {breakdown['loss_img']:.6f}  (weight={breakdown['lambda_img']})")

    # Inspect gradient distributions per submodule
    ctx_grad_norms = []
    gain_grad_norms = []
    for name, p in bridge.acs_model.context_predictor.named_parameters():
        if p.grad is not None:
            ctx_grad_norms.append(float(p.grad.norm().item()))
    for name, p in bridge.acs_model.adaptive_gain.named_parameters():
        if p.grad is not None:
            gain_grad_norms.append(float(p.grad.norm().item()))

    ctx_norm_mean = float(np.mean(ctx_grad_norms)) if ctx_grad_norms else 0.0
    gain_norm_mean = float(np.mean(gain_grad_norms)) if gain_grad_norms else 0.0

    print("\n    Gradient Distribution Analysis:")
    print(f"    Total Clipped Grad Norm:     {step_result.total_grad_norm:.6f}")
    print(f"    Context Predictor Mean Norm: {ctx_norm_mean:.6f} (across {len(ctx_grad_norms)} layers)")
    print(f"    Adaptive Gain Mean Norm:     {gain_norm_mean:.6f} (across {len(gain_grad_norms)} layers)")
    print(f"    Max Param Gradient:          {step_result.max_param_grad:.6f}")
    print(f"    All Gradients Finite:        {step_result.is_grad_finite}")

    # 8. Clipping Diagnostic Metrics
    clip_diag = step_result.clipping_diagnostic
    print("\n[8] Clipping Diagnostic Metrics (Diagnostic Only):")
    print(f"    Pre-clamp Raw Range:      [{clip_diag.min_val_raw:.2f}, {clip_diag.max_val_raw:.2f}]")
    print(f"    Total Clipped Pixels:     {clip_diag.num_clipped_total} ({clip_diag.clip_fraction * 100.0:.3f}%)")
    print(f"    Boundary Pinned Pixels:   {clip_diag.boundary_pixel_count} ({clip_diag.boundary_pixel_fraction * 100.0:.3f}%)")
    print(f"    Saturation L1 Penalty:    {clip_diag.saturation_penalty_l1:.6f}")

    # 9. Synthetic Smoke Checkpoint Verification
    ckpt_dir = REPO_ROOT / config.checkpoint_dir
    ckpt_path = ckpt_dir / "synthetic_smoke_checkpoint.pt"
    save_meta = save_synthetic_smoke_checkpoint(bridge, optimizer, config, ckpt_path)
    print(f"\n[9] Synthetic smoke-test checkpoint saved to: {ckpt_path}")
    print(f"    Metadata Type: {save_meta['checkpoint_type']}")
    print(f"    Training Authorized: {save_meta['training_authorized']}")

    fresh_bridge = NeuralSpatialBridge()
    loaded_meta = load_and_validate_checkpoint(ckpt_path, fresh_bridge)
    checkpoint_verified = (loaded_meta["checkpoint_type"] == "SYNTHETIC_SMOKE_TEST_ARTIFACT")
    print(f"    Checkpoint reloaded and verified: {checkpoint_verified}")

    # 10. Compile Structured Results
    results = {
        "metadata": {
            "sprint": "Phase 3 Sprint 3 — Loss Function and Training Scaffold",
            "date": "2026-10-19",
            "training_authorized": False,
            "training_step_executed": True,
            "optimizer_step_executed": False,
            "model_architecture": "ACS-Sub-INN v3 (Frozen)",
            "gamma_min": bridge.acs_model.gamma_min,
            "device": config.device,
            "cuda_available": torch.cuda.is_available(),
        },
        "reproducibility": seed_report,
        "dataset_split_boundary_audit": {
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
            "held_out_test_protection_verified": test_protection_verified,
        },
        "loss_and_gradient_verification": {
            "loss_breakdown": breakdown,
            "total_grad_norm": step_result.total_grad_norm,
            "context_predictor_mean_grad_norm": ctx_norm_mean,
            "adaptive_gain_mean_grad_norm": gain_norm_mean,
            "max_param_grad": step_result.max_param_grad,
            "all_gradients_finite": step_result.is_grad_finite,
            "parameters_unmodified_verified": parameters_unmodified,
        },
        "clipping_diagnostic": {
            "min_val_raw": clip_diag.min_val_raw,
            "max_val_raw": clip_diag.max_val_raw,
            "num_clipped_total": clip_diag.num_clipped_total,
            "clip_fraction": clip_diag.clip_fraction,
            "boundary_pixel_fraction": clip_diag.boundary_pixel_fraction,
            "saturation_penalty_l1": clip_diag.saturation_penalty_l1,
        },
        "checkpoint_verification": {
            "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)),
            "checkpoint_type": save_meta["checkpoint_type"],
            "training_authorized": save_meta["training_authorized"],
            "reloaded_and_verified": checkpoint_verified,
        },
        "gate_governance_catalog": GateRegistry.list_gates(),
    }

    out_json = REPO_ROOT / "docs" / "experiments" / "sprint3_training_scaffold_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Versioned smoke-test results written to: {out_json}")
    print("\n" + "=" * 80)
    print("SMOKE TEST COMPLETE: FORWARD/BACKWARD VERIFIED WITH FINITE GRADIENTS.")
    print("TRAINING STATUS: NOT AUTHORIZED — awaiting human review of loss equations,")
    print("gradient paths, split protection, and smoke-test evidence.")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_smoke_test()
