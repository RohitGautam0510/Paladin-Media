"""Authorized Five-Epoch Pilot: Phase 4B Controlled ACS-Sub-INN v3 Training.

EXECUTION CONTRACT:
This script executes the authorized Phase 4B five-epoch training pilot:
1. Preflight verification of initial parameter digest, canonical manifest, and frozen architecture.
2. Comprehensive evaluation of the untrained baseline model on the canonical validation partition.
3. Execution of the 5-epoch training pilot via TrainingRunner with real optimizer updates.
4. Evaluation of the final and best-validation checkpoints against the baseline.
5. Verification of saved checkpoints and parameter state loading.
6. Structured JSON output to docs/experiments/phase4b_pilot_results.json.
7. Append-only telemetry to docs/experiments/phase4b_pilot_telemetry.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import (
    DEFAULT_DATASET_REL_PATH,
    DEFAULT_MANIFEST_REL_PATH,
    EXPECTED_COMPOSITE_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    build_stego_dataloader,
    validate_div2k_manifest,
)
from src.neural.losses import CompositeLoss
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_engine import (
    CHECKPOINT_SCHEMA_VERSION,
    PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
    TrainingAuthorizationToken,
    TrainingRunConfig,
    TrainingRunner,
    TrainingTelemetryLogger,
    ValidationMetrics,
    evaluate_validation_epoch,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.neural.training_scaffold import (
    TrainingConfig,
    build_model_and_optimizer,
    set_reproducibility_seeds,
)


def compute_param_digest(model: torch.nn.Module) -> str:
    """Computes a SHA-256 digest over all sorted named parameters in a PyTorch module."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def run_phase4b_pilot() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 4B: AUTHORIZED FIVE-EPOCH ACS-SUB-INN V3 PILOT")
    print("=" * 80)

    pilot_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "4B",
        "pilot_scope": "PHASE4B_PILOT",
        "training_authorized": True,
        "pilot_completed": False,
    }

    # -------------------------------------------------------------------------
    # 1. Preflight Verification
    # -------------------------------------------------------------------------
    print("\n[1] Preflight Verification...")
    seed_report = set_reproducibility_seeds(seed=42, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Execution Device: {device}")
    if torch.cuda.is_available():
        print(f"    GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"    Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")

    # Architecture instantiation
    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    bridge.to(device)

    # Initial parameter digest verification
    initial_digest = compute_param_digest(bridge.acs_model)
    expected_initial_digest = "fe7462957b0194d109ec32a78834d7bf44b9d1aa86510a40625f2e485fa40369"
    print(f"    Initial Model Parameter Digest: {initial_digest}")
    if initial_digest != expected_initial_digest:
        raise RuntimeError(
            f"PREFLIGHT BLOCKER: Initial parameter digest mismatch! "
            f"Expected {expected_initial_digest}, got {initial_digest}"
        )
    pilot_results["initial_model_parameter_digest"] = initial_digest

    # Canonical manifest verification
    print("    Verifying Canonical DIV2K Split Manifest...")
    manifest_audit = validate_div2k_manifest(
        manifest_path=DEFAULT_MANIFEST_REL_PATH,
        dataset_root=DEFAULT_DATASET_REL_PATH,
        enforce_file_existence=True,
    )
    counts = manifest_audit["partition_counts"]
    print(f"    Manifest Counts: {counts}")
    assert counts["train"] == 600, f"Expected 600 train, got {counts['train']}"
    assert counts["val"] == 100, f"Expected 100 val, got {counts['val']}"
    assert counts["test"] == 100, f"Expected 100 test, got {counts['test']}"
    pilot_results["manifest_verification"] = {
        "manifest_file_sha256": manifest_audit["manifest_file_sha256"],
        "composite_sha256": manifest_audit["composite_sha256"],
        "counts": counts,
        "status": "PASS",
    }

    # Frozen architecture verification
    assert bridge.acs_model.gamma_min == 1.0, f"gamma_min must be 1.0, got {bridge.acs_model.gamma_min}"
    print("    Frozen Architecture Verified: ACS-Sub-INN v3, gamma_min=1.0, Red-HH target.")

    # Unique output paths
    ckpt_dir = Path("checkpoints/phase4b_pilot")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = "docs/experiments/phase4b_pilot_telemetry.jsonl"
    print(f"    Checkpoint Destination: {ckpt_dir.resolve()}")
    print(f"    Telemetry Destination: {telemetry_path}")

    # Build canonical datasets and dataloaders
    print("\n[2] Building Canonical DataLoaders...")
    train_loader = build_stego_dataloader(
        split="train",
        purpose="train",
        batch_size=1,
        crop_size=512,
        shuffle=True,
        seed=42,
        num_workers=0,
    )
    val_loader = build_stego_dataloader(
        split="val",
        purpose="validation",
        batch_size=1,
        crop_size=512,
        shuffle=False,
        num_workers=0,
    )
    print(f"    Train Batches: {len(train_loader)} (Split='{train_loader.dataset_split}')")
    print(f"    Validation Batches: {len(val_loader)} (Split='{val_loader.dataset_split}')")

    # Optimizer and composite loss
    optimizer = torch.optim.AdamW(bridge.acs_model.parameters(), lr=1e-4, weight_decay=1e-5)
    comp_loss = CompositeLoss(
        lambda_bit=1.0,
        lambda_ctx=0.5,
        lambda_gain=0.01,
        lambda_img=0.1,
    )
    comp_loss.to(device)

    # -------------------------------------------------------------------------
    # 2. Establish Untrained Baseline
    # -------------------------------------------------------------------------
    print("\n[3] Evaluating Untrained Baseline on Canonical Validation Partition...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    base_start = time.time()

    baseline_metrics: ValidationMetrics = evaluate_validation_epoch(
        bridge=bridge,
        val_loader=val_loader,
        composite_loss=comp_loss,
        device=device,
        val_seed=99999,
    )
    base_duration = time.time() - base_start
    base_peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    print("    --- Baseline Metrics (Untrained Model) ---")
    print(f"    Total Loss:          {baseline_metrics.total_loss:.6f}")
    print(f"    Bit Loss (Raw):      {baseline_metrics.loss_bit_raw:.6f} (Weighted: {baseline_metrics.loss_bit_weighted:.6f})")
    print(f"    Context Loss (Raw):  {baseline_metrics.loss_ctx_raw:.6f} (Weighted: {baseline_metrics.loss_ctx_weighted:.6f})")
    print(f"    Gain Reg (Raw):      {baseline_metrics.loss_gain_raw:.6f} (Weighted: {baseline_metrics.loss_gain_weighted:.6f})")
    print(f"    Image Loss (Raw):    {baseline_metrics.loss_img_raw:.6f} (Weighted: {baseline_metrics.loss_img_weighted:.6f})")
    print(f"    Pre-ECC BER:         {baseline_metrics.pre_ecc_ber:.6f} ({baseline_metrics.pre_ecc_ber*100:.3f}%)")
    print(f"    Stego PSNR:          {baseline_metrics.stego_psnr_db:.2f} dB")
    print(f"    Stego SSIM:          {baseline_metrics.stego_ssim:.4f}")
    print(f"    Clipping Fraction:   {baseline_metrics.clip_fraction:.6f}")
    print(f"    Evaluation Duration: {base_duration:.2f} s")
    print(f"    Peak VRAM:           {base_peak_vram:.2f} MB")

    pilot_results["baseline_evaluation"] = {
        "metrics": baseline_metrics.to_dict(),
        "elapsed_seconds": base_duration,
        "peak_vram_mb": base_peak_vram,
    }

    # -------------------------------------------------------------------------
    # 3. Construct Training Authorization Token & Config
    # -------------------------------------------------------------------------
    print("\n[4] Constructing Training Authorization Token & Pilot Configuration...")
    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER_EXPLICIT_AUTHORIZATION",
        scope="PHASE4B_PILOT",
    )
    assert auth_token.is_valid(), "CRITICAL: Authorization token is invalid!"
    print("    Authorization Token: VALID (Scope='PHASE4B_PILOT')")

    run_config = TrainingRunConfig(
        seed=42,
        device=str(device),
        deterministic=True,
        epochs=5,
        batch_size=1,
        crop_size=512,
        grad_accum_steps=1,
        num_workers=0,
        learning_rate=1e-4,
        weight_decay=1e-5,
        clip_grad_norm=1.0,
        lambda_bit=1.0,
        lambda_ctx=0.5,
        lambda_gain=0.01,
        lambda_img=0.1,
        checkpoint_dir=str(ckpt_dir),
        save_every_n_epochs=1,
        early_stopping_patience=10,  # Will not trigger on 5 epochs
        early_stopping_min_delta=1e-4,
        telemetry_path=telemetry_path,
        run_id="phase4b_pilot_5epochs",
    )
    pilot_results["configuration"] = run_config.to_dict()

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=run_config,
        authorization_token=auth_token,
    )

    # -------------------------------------------------------------------------
    # 4. Execute Real Five-Epoch Training Pilot
    # -------------------------------------------------------------------------
    print("\n[5] Executing 5-Epoch Training Pilot...")
    pilot_start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    summary = runner.train()
    total_pilot_time = time.time() - pilot_start_time
    pilot_peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    print("\n    --- Training Pilot Execution Complete ---")
    print(f"    Completed Epochs:   {summary.completed_epochs}")
    print(f"    Total Global Steps: {summary.global_steps}")
    print(f"    Best Val Loss:      {summary.best_val_metric:.6f}")
    print(f"    Best Checkpoint:    {summary.best_checkpoint_path}")
    print(f"    Termination Reason: {summary.termination_reason}")
    print(f"    Total Duration:     {total_pilot_time:.2f} s")
    print(f"    Peak VRAM:          {pilot_peak_vram:.2f} MB")

    final_trained_digest = compute_param_digest(bridge.acs_model)
    print(f"    Final Model Parameter Digest: {final_trained_digest}")
    assert final_trained_digest != initial_digest, "CRITICAL ERROR: Model parameters did not update during training!"
    print("    Parameter Mutation Verified: Model parameters genuinely updated via optimizer.")

    pilot_results["training_summary"] = {
        "completed_epochs": summary.completed_epochs,
        "global_steps": summary.global_steps,
        "best_val_metric": summary.best_val_metric,
        "best_checkpoint_path": summary.best_checkpoint_path,
        "termination_reason": summary.termination_reason,
        "total_pilot_seconds": total_pilot_time,
        "peak_vram_mb": pilot_peak_vram,
        "final_trained_parameter_digest": final_trained_digest,
        "parameter_mutation_verified": True,
        "epoch_history": summary.history,
    }

    # -------------------------------------------------------------------------
    # 5. Post-Training Evaluation and Checkpoint Verification
    # -------------------------------------------------------------------------
    print("\n[6] Post-Training Checkpoint Verification & Final Evaluations...")

    # Evaluate final model (epoch 5)
    final_metrics = evaluate_validation_epoch(
        bridge=bridge,
        val_loader=val_loader,
        composite_loss=comp_loss,
        device=device,
        val_seed=99999,
    )
    pilot_results["final_epoch_evaluation"] = final_metrics.to_dict()

    # Load and evaluate best checkpoint in a fresh model instance
    best_ckpt_path = summary.best_checkpoint_path or (ckpt_dir / "best_model.pt")
    fresh_bridge, fresh_opt, fresh_loss = build_model_and_optimizer(TrainingConfig(seed=999, device=str(device)))
    fresh_bridge.to(device)

    restored = load_training_checkpoint(
        checkpoint_path=best_ckpt_path,
        bridge=fresh_bridge,
        optimizer=fresh_opt,
    )
    best_model_digest = compute_param_digest(fresh_bridge.acs_model)
    print(f"    Best Model Parameter Digest (Fresh Load): {best_model_digest}")

    best_metrics = evaluate_validation_epoch(
        bridge=fresh_bridge,
        val_loader=val_loader,
        composite_loss=fresh_loss,
        device=device,
        val_seed=99999,
    )
    pilot_results["best_model_evaluation"] = {
        "checkpoint_path": str(best_ckpt_path),
        "parameter_digest": best_model_digest,
        "restored_metadata": restored["metadata"],
        "metrics": best_metrics.to_dict(),
    }

    # -------------------------------------------------------------------------
    # 6. Baseline vs. Trained Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PILOT OUTCOME: UNTRAINED BASELINE VS. TRAINED BEST MODEL COMPARISON")
    print("=" * 80)

    comparison = {
        "total_loss": {
            "baseline": baseline_metrics.total_loss,
            "trained_best": best_metrics.total_loss,
            "delta": best_metrics.total_loss - baseline_metrics.total_loss,
            "relative_change_pct": (best_metrics.total_loss - baseline_metrics.total_loss) / baseline_metrics.total_loss * 100,
        },
        "loss_bit_raw": {
            "baseline": baseline_metrics.loss_bit_raw,
            "trained_best": best_metrics.loss_bit_raw,
            "delta": best_metrics.loss_bit_raw - baseline_metrics.loss_bit_raw,
        },
        "loss_ctx_raw": {
            "baseline": baseline_metrics.loss_ctx_raw,
            "trained_best": best_metrics.loss_ctx_raw,
            "delta": best_metrics.loss_ctx_raw - baseline_metrics.loss_ctx_raw,
        },
        "loss_gain_raw": {
            "baseline": baseline_metrics.loss_gain_raw,
            "trained_best": best_metrics.loss_gain_raw,
            "delta": best_metrics.loss_gain_raw - baseline_metrics.loss_gain_raw,
        },
        "loss_img_raw": {
            "baseline": baseline_metrics.loss_img_raw,
            "trained_best": best_metrics.loss_img_raw,
            "delta": best_metrics.loss_img_raw - baseline_metrics.loss_img_raw,
        },
        "pre_ecc_ber": {
            "baseline": baseline_metrics.pre_ecc_ber,
            "trained_best": best_metrics.pre_ecc_ber,
            "delta": best_metrics.pre_ecc_ber - baseline_metrics.pre_ecc_ber,
        },
        "stego_psnr_db": {
            "baseline": baseline_metrics.stego_psnr_db,
            "trained_best": best_metrics.stego_psnr_db,
            "delta": best_metrics.stego_psnr_db - baseline_metrics.stego_psnr_db,
        },
        "stego_ssim": {
            "baseline": baseline_metrics.stego_ssim,
            "trained_best": best_metrics.stego_ssim,
            "delta": best_metrics.stego_ssim - baseline_metrics.stego_ssim,
        },
        "clip_fraction": {
            "baseline": baseline_metrics.clip_fraction,
            "trained_best": best_metrics.clip_fraction,
            "delta": best_metrics.clip_fraction - baseline_metrics.clip_fraction,
        },
    }

    print(f"  Metric              | Baseline       | Trained Best   | Delta          | Direction")
    print(f"  --------------------+----------------+----------------+----------------+----------")
    for k, v in comparison.items():
        d = v["delta"]
        # In lower-is-better metrics: negative delta is improvement
        if k in ("total_loss", "loss_bit_raw", "loss_ctx_raw", "loss_gain_raw", "loss_img_raw", "pre_ecc_ber", "clip_fraction"):
            dir_str = "IMPROVED" if d < 0 else ("REGRESSED" if d > 0 else "UNCHANGED")
        else: # Higher is better (PSNR, SSIM)
            dir_str = "IMPROVED" if d > 0 else ("REGRESSED" if d < 0 else "UNCHANGED")
        print(f"  {k:<19} | {v['baseline']:<14.6f} | {v['trained_best']:<14.6f} | {d:<+14.6f} | {dir_str}")

    pilot_results["comparison"] = comparison
    pilot_results["environment"] = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    pilot_results["pilot_completed"] = True

    # Write results JSON
    out_json = REPO_ROOT / "docs" / "experiments" / "phase4b_pilot_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(pilot_results, indent=2), encoding="utf-8")
    print(f"\n[Pilot Complete] Results saved to: {out_json}")

    return pilot_results


if __name__ == "__main__":
    run_phase4b_pilot()
