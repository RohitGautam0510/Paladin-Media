"""Phase 4C Extended Training and End-to-End Recovery Validation Script.

EXECUTION CONTRACT:
1. Preflight verification of Phase 4B starting checkpoint (best_model.pt), parameter digest,
   canonical manifest, and frozen architecture.
2. Seed checkpoints/phase4c_extended_30ep/best_model.pt with a byte-for-byte copy of Phase 4B best_model.pt.
3. Resume training from epoch 6 up to epoch 30 (30-epoch total horizon, 25 additional epochs,
   approx 15,000 steps) using TrainingRunner with PHASE4C_EXTENDED authorization scope.
4. Record per-epoch telemetry to docs/experiments/phase4c_extended_telemetry.jsonl.
5. Post-training checkpoint acceptance: verify parameter mutation, reload best checkpoint in fresh
   instance, verify bit-exact digest and deterministic validation metrics.
6. Genuine end-to-end payload and Reed-Solomon recovery experiment using the trained checkpoint,
   real DIV2K carrier crop, and production MediaStegoPipeline / NeuralModulationEngine APIs.
7. Structured JSON results output to docs/experiments/phase4c_extended_results.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator
from src.carrier.modulation import NeuralModulationEngine
from src.core.media_pipeline import (
    MediaStegoPipeline,
    calculate_psnr,
    calculate_ssim,
)
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import (
    DEFAULT_DATASET_REL_PATH,
    DEFAULT_MANIFEST_REL_PATH,
    DIV2KStegoDataset,
    EXPECTED_COMPOSITE_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    build_stego_dataloader,
    validate_div2k_manifest,
)
from src.neural.losses import CompositeLoss
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_engine import (
    CHECKPOINT_SCHEMA_VERSION,
    PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
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
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


def compute_param_digest(model: torch.nn.Module) -> str:
    """Computes SHA-256 digest over sorted named parameters in a module."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def run_phase4c_extended_experiment() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 4C: EXTENDED TRAINING AND END-TO-END RECOVERY VALIDATION")
    print("=" * 80)

    experiment_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "4C",
        "scope": "PHASE4C_EXTENDED",
        "training_authorized": True,
        "completed": False,
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

    p4b_best_path = REPO_ROOT / "checkpoints" / "phase4b_pilot" / "best_model.pt"
    if not p4b_best_path.exists():
        raise RuntimeError(f"PREFLIGHT BLOCKER: Phase 4B checkpoint missing at {p4b_best_path}")

    # Inspect checkpoint before applying state
    raw_p4b = torch.load(p4b_best_path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" in raw_p4b, "Starting checkpoint lacks optimizer state!"
    assert raw_p4b.get("epoch") == 5, f"Expected epoch 5, got {raw_p4b.get('epoch')}"
    assert raw_p4b.get("global_step") == 3000, f"Expected step 3000, got {raw_p4b.get('global_step')}"
    assert raw_p4b.get("next_epoch") == 6, f"Expected next_epoch 6, got {raw_p4b.get('next_epoch')}"

    # Load through current fail-closed loader into temporary model
    pre_model = ACSSubINN(gamma_min=1.0)
    pre_bridge = NeuralSpatialBridge(acs_model=pre_model)
    pre_opt = torch.optim.AdamW(pre_model.parameters(), lr=1e-4)
    pre_restored = load_training_checkpoint(p4b_best_path, bridge=pre_bridge, optimizer=pre_opt)

    starting_digest = compute_param_digest(pre_bridge.acs_model)
    expected_p4b_digest = "8fd00c49455be04ae90d8474ad984e0f95f1a479690644421ed30a3daa8ef938"
    print(f"    Starting Parameter Digest: {starting_digest}")
    if starting_digest != expected_p4b_digest:
        raise RuntimeError(
            f"PREFLIGHT BLOCKER: Starting parameter digest mismatch! "
            f"Expected {expected_p4b_digest}, got {starting_digest}"
        )
    experiment_results["starting_checkpoint_verification"] = {
        "checkpoint_path": str(p4b_best_path),
        "parameter_digest": starting_digest,
        "epoch": raw_p4b.get("epoch"),
        "global_step": raw_p4b.get("global_step"),
        "best_val_metric": raw_p4b.get("best_val_metric"),
        "status": "PASS",
    }

    # Manifest verification
    print("    Verifying Canonical DIV2K Split Manifest...")
    manifest_audit = validate_div2k_manifest(
        manifest_path=DEFAULT_MANIFEST_REL_PATH,
        dataset_root=DEFAULT_DATASET_REL_PATH,
        enforce_file_existence=True,
    )
    counts = manifest_audit["partition_counts"]
    print(f"    Manifest Counts: {counts}")
    assert counts["train"] == 600 and counts["val"] == 100 and counts["test"] == 100
    experiment_results["manifest_verification"] = {
        "manifest_file_sha256": manifest_audit["manifest_file_sha256"],
        "composite_sha256": manifest_audit["composite_sha256"],
        "counts": counts,
        "status": "PASS",
    }

    # Verify Phase 4B historical evidence files exist
    p4b_report = REPO_ROOT / "docs" / "audits" / "PHASE4B_PILOT_REPORT.md"
    p4b_results_file = REPO_ROOT / "docs" / "experiments" / "phase4b_pilot_results.json"
    p4b_telemetry_file = REPO_ROOT / "docs" / "experiments" / "phase4b_pilot_telemetry.jsonl"
    assert p4b_report.exists(), "Phase 4B report missing!"
    assert p4b_results_file.exists(), "Phase 4B results JSON missing!"
    assert p4b_telemetry_file.exists(), "Phase 4B telemetry JSONL missing!"
    print("    Phase 4B Historical Artifacts Verified Intact.")

    # -------------------------------------------------------------------------
    # 2. Phase 4C Output Directory & Seeding
    # -------------------------------------------------------------------------
    print("\n[2] Setting Up Phase 4C Output Destinations...")
    p4c_ckpt_dir = REPO_ROOT / "checkpoints" / "phase4c_extended_30ep"
    p4c_ckpt_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = REPO_ROOT / "docs" / "experiments" / "phase4c_extended_telemetry.jsonl"
    results_path = REPO_ROOT / "docs" / "experiments" / "phase4c_extended_results.json"

    # Seed best_model.pt with byte-for-byte copy of Phase 4B best_model.pt ONLY if not present
    p4c_seeded_best = p4c_ckpt_dir / "best_model.pt"
    if not p4c_seeded_best.exists():
        shutil.copy2(p4b_best_path, p4c_seeded_best)
        print(f"    Seeded {p4c_seeded_best} with Phase 4B best_model.pt")
    else:
        print(f"    Existing best model found at {p4c_seeded_best}, preserving.")

    # Check for existing checkpoints to support seamless resumption across server restarts
    p4c_epoch_ckpts = sorted(p4c_ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if p4c_epoch_ckpts:
        resume_checkpoint_path = p4c_epoch_ckpts[-1]
        print(f"    Resuming from latest Phase 4C checkpoint: {resume_checkpoint_path}")
    else:
        resume_checkpoint_path = p4b_best_path
        print(f"    Resuming from Phase 4B best model: {resume_checkpoint_path}")

    # -------------------------------------------------------------------------
    # 3. Setup Model, Loaders, and Authorization
    # -------------------------------------------------------------------------
    print("\n[3] Setting Up Model, Dataloaders, and Authorization Token...")
    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    bridge.to(device)

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
    print(f"    Val Batches:   {len(val_loader)} (Split='{val_loader.dataset_split}')")

    optimizer = torch.optim.AdamW(bridge.acs_model.parameters(), lr=1e-4, weight_decay=1e-5)
    comp_loss = CompositeLoss(
        lambda_bit=1.0,
        lambda_ctx=0.5,
        lambda_gain=0.01,
        lambda_img=0.1,
    )
    comp_loss.to(device)

    auth_token = TrainingAuthorizationToken(
        passphrase=PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
        approver="USER_EXPLICIT_AUTHORIZATION",
        scope="PHASE4C_EXTENDED",
    )
    assert auth_token.is_valid(), "CRITICAL: Authorization token invalid!"
    print("    Authorization Token: VALID (Scope='PHASE4C_EXTENDED')")

    run_config = TrainingRunConfig(
        seed=42,
        device=str(device),
        deterministic=True,
        epochs=30,  # 30 total epochs horizon
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
        checkpoint_dir=str(p4c_ckpt_dir),
        save_every_n_epochs=1,
        early_stopping_patience=10,
        early_stopping_min_delta=1e-4,
        telemetry_path=str(telemetry_path),
        run_id="phase4c_extended_30ep",
    )
    experiment_results["configuration"] = run_config.to_dict()

    runner = TrainingRunner(
        bridge=bridge,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        composite_loss=comp_loss,
        config=run_config,
        authorization_token=auth_token,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    print(f"    Resumption Configured: Start Epoch = {runner.start_epoch}, Global Step = {runner.global_step}")
    print(f"    Resumed Best Validation Metric = {runner.best_val_metric:.6f}")
    assert runner.start_epoch >= 6, f"Expected start_epoch >= 6, got {runner.start_epoch}"
    assert runner.global_step >= 3000, f"Expected global_step >= 3000, got {runner.global_step}"

    # -------------------------------------------------------------------------
    # 4. Execute Extended Training Run (Epochs 6–30)
    # -------------------------------------------------------------------------
    print("\n[4] Executing Extended Training Run (Epochs 6-30)...")
    train_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    summary = runner.train()
    total_train_sec = time.time() - train_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    print("\n    --- Training Complete ---")
    print(f"    Additional Epochs Traversed: {summary.completed_epochs}")
    print(f"    Final Global Step:           {summary.global_steps}")
    print(f"    Best Validation Loss:        {summary.best_val_metric:.6f}")
    print(f"    Best Checkpoint Path:        {summary.best_checkpoint_path or str(p4c_seeded_best)}")
    print(f"    Termination Reason:          {summary.termination_reason}")
    print(f"    Training Elapsed Seconds:    {total_train_sec:.2f} s")
    print(f"    Peak VRAM:                   {peak_vram_mb:.2f} MB")

    final_digest = compute_param_digest(bridge.acs_model)
    print(f"    Final Trained Parameter Digest: {final_digest}")
    assert final_digest != starting_digest, "CRITICAL: Model parameters did not update during training!"

    # Load all records from telemetry_path to get complete Phase 4C history
    full_epoch_history = []
    if telemetry_path.exists():
        with open(telemetry_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    full_epoch_history.append(json.loads(line.strip()))

    experiment_results["training_summary"] = {
        "phase4b_initial_epoch": 5,
        "phase4b_initial_step": 3000,
        "target_horizon_epochs": 30,
        "total_phase4c_epochs_recorded": len(full_epoch_history),
        "final_global_step": summary.global_steps,
        "best_val_metric": summary.best_val_metric,
        "best_checkpoint_path": summary.best_checkpoint_path or str(p4c_seeded_best),
        "termination_reason": summary.termination_reason,
        "training_elapsed_sec": total_train_sec,
        "peak_vram_mb": peak_vram_mb,
        "starting_parameter_digest": starting_digest,
        "final_parameter_digest": final_digest,
        "epoch_history": full_epoch_history,
    }

    # -------------------------------------------------------------------------
    # 5. Post-Training Validation & Checkpoint Reload Acceptance
    # -------------------------------------------------------------------------
    print("\n[5] Post-Training Acceptance & Independent Reload Verification...")

    # Evaluate final model (end of training run)
    final_metrics = evaluate_validation_epoch(
        bridge=bridge,
        val_loader=val_loader,
        composite_loss=comp_loss,
        device=device,
        val_seed=99999,
    )
    experiment_results["final_epoch_evaluation"] = final_metrics.to_dict()

    # Load the overall best checkpoint across Phase 4B and Phase 4C into fresh model
    best_ckpt_file = Path(summary.best_checkpoint_path) if summary.best_checkpoint_path else p4c_seeded_best
    fresh_bridge, fresh_opt, fresh_loss = build_model_and_optimizer(TrainingConfig(seed=12345, device=str(device)))
    fresh_bridge.to(device)

    restored_best = load_training_checkpoint(
        checkpoint_path=best_ckpt_file,
        bridge=fresh_bridge,
        optimizer=fresh_opt,
    )
    best_reloaded_digest = compute_param_digest(fresh_bridge.acs_model)
    print(f"    Reloaded Best Model Digest: {best_reloaded_digest}")

    best_reloaded_metrics = evaluate_validation_epoch(
        bridge=fresh_bridge,
        val_loader=val_loader,
        composite_loss=fresh_loss,
        device=device,
        val_seed=99999,
    )
    print(f"    Reloaded Best Model Val Loss: {best_reloaded_metrics.total_loss:.6f}")
    print(f"    Reloaded Best Model Pre-ECC BER: {best_reloaded_metrics.pre_ecc_ber:.6f} ({best_reloaded_metrics.pre_ecc_ber*100:.4f}%)")
    print(f"    Reloaded Best Model Stego PSNR:  {best_reloaded_metrics.stego_psnr_db:.2f} dB")

    experiment_results["best_model_evaluation"] = {
        "checkpoint_path": str(best_ckpt_file),
        "parameter_digest": best_reloaded_digest,
        "restored_metadata": restored_best["metadata"],
        "metrics": best_reloaded_metrics.to_dict(),
    }

    # -------------------------------------------------------------------------
    # 6. Genuine End-to-End Payload and Reed-Solomon Recovery Experiment
    # -------------------------------------------------------------------------
    print("\n[6] Genuine End-to-End Payload and ECC Recovery Experiment...")
    print("    Instantiating production MediaStegoPipeline with NeuralModulationEngine...")

    # Move fresh_bridge to CPU for seamless interop with NumPy image arrays in NeuralModulationEngine
    fresh_bridge.to("cpu")
    neural_mod_engine = NeuralModulationEngine(spatial_bridge=fresh_bridge)
    prod_pipeline = MediaStegoPipeline(
        crypto_provider=AESGCMProvider(),
        ecc_encoder=ReedSolomonEncoder(ecc_symbols=32, block_size=255),
        modulation_engine=neural_mod_engine,
    )

    # Carrier preparation: clean 512x512 crop from validation partition (image 0601.png)
    val_dataset = DIV2KStegoDataset(split="val", crop_size=512)
    sample_val = val_dataset[0]  # Image ID 601
    cover_crop_np = (sample_val["carrier"].permute(1, 2, 0).numpy()).astype(np.uint8)
    print(f"    Carrier: Image ID {sample_val['image_id']} ({sample_val['filename']}), shape {cover_crop_np.shape}")

    # Generate 1,000 bytes deterministic secret payload safely within 8,176 B capacity
    payload_len = 1000
    secret_bytes = bytes([((i * 37 + 13) % 256) for i in range(payload_len)])
    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.TEXT,
        metadata={"filename": "phase4c_e2e_secret.txt"},
    )
    aes_key = hashlib.sha256(b"PaladinMedia-Phase4C-Key-Deterministic-2026").digest()
    aad = b"Phase4C-Session-42"
    session_counter = 42

    e2e_start = time.time()

    # Step 1-4: Protect and embed via production pipeline
    stego_result = prod_pipeline.protect_and_embed(
        secret_payload=secret_payload,
        cover_image=cover_crop_np,
        key=aes_key,
        aad=aad,
        session_counter=session_counter,
    )
    stego_png_bytes = stego_result.stego_png_bytes
    print(f"    Protected Payload Bytes: {len(prod_pipeline.security_pipeline.protect(secret_payload, aes_key, aad=aad))}")
    print(f"    Stego PNG Size:          {len(stego_png_bytes)} bytes")
    print(f"    Embedding PSNR:          {stego_result.psnr_db:.2f} dB")
    print(f"    Embedding SSIM:          {stego_result.ssim:.5f}")

    # Step 5: Save to disk and reload as fresh PNG bytes
    stego_png_file = p4c_ckpt_dir / "e2e_stego_carrier.png"
    stego_png_file.write_bytes(stego_png_bytes)
    reloaded_png_bytes = stego_png_file.read_bytes()
    reloaded_arr = np.array(Image.open(io.BytesIO(reloaded_png_bytes)).convert("RGB"), dtype=np.uint8)

    # Detailed inspection of extracted raw bits and per-codeword RS symbols
    bit_extractor = neural_mod_engine.begin_extraction(reloaded_arr, subband_dim=256)
    hdr_positions = DeterministicMaskGenerator.generate_header_positions(subband_dim=256)
    extracted_hdr_bits = bit_extractor.extract_bits(hdr_positions)
    extracted_hdr_bytes = Extractor.bits_to_bytes(extracted_hdr_bits)
    carrier_hdr = CarrierHeader.unpack(extracted_hdr_bytes)

    payload_positions = DeterministicMaskGenerator.generate_payload_positions(
        session_counter=carrier_hdr.session_counter,
        payload_length_bytes=carrier_hdr.payload_length,
        subband_dim=256,
    )
    extracted_payload_bits = bit_extractor.extract_bits(payload_positions)
    extracted_protected_bytes = Extractor.bits_to_bytes(extracted_payload_bits)

    # Compare against actual embedded protected bits from this specific embedding session
    actual_payload_bits = neural_mod_engine.last_payload_bits
    actual_protected_bytes = Extractor.bits_to_bytes(actual_payload_bits)
    bit_errors = int(np.sum(extracted_payload_bits != actual_payload_bits))
    pre_ecc_ber = float(bit_errors) / len(actual_payload_bits)
    print(f"    Raw Extracted Bit Errors: {bit_errors} / {len(actual_payload_bits)} (BER: {pre_ecc_ber:.6f})")

    # Inspect per-codeword Reed-Solomon correction
    rs_encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    rs_res = rs_encoder.decode_with_result(extracted_protected_bytes)
    print(f"    RS Blocks Count:         {rs_res.blocks_count}")
    print(f"    RS Total Corrected Syms: {rs_res.corrected_symbols}")
    print(f"    RS Prefix Recovered:     {rs_res.prefix_recovered}")

    # Inspect per-codeword symbol differences between extracted and actual embedded blocks
    k = rs_encoder.k
    block_len = k + rs_encoder.ecc_symbols
    offset = 4
    per_codeword_errors = []
    for blk_idx in range(rs_res.blocks_count):
        cur_len = block_len if (offset + block_len <= len(actual_protected_bytes)) else (len(actual_protected_bytes) - offset)
        ext_block = extracted_protected_bytes[offset : offset + cur_len]
        act_block = actual_protected_bytes[offset : offset + cur_len]
        diff_syms = sum(1 for a, b in zip(ext_block, act_block) if a != b)
        per_codeword_errors.append(diff_syms)
        offset += cur_len
    print(f"    Per-Codeword Symbol Errors (max t=16): {per_codeword_errors}")

    # Full receiver extraction and unprotection via production MediaStegoPipeline
    recovery_result = prod_pipeline.extract_and_unprotect(
        stego_carrier=reloaded_png_bytes,
        key=aes_key,
        aad=aad,
        reference_payload=secret_payload,
    )
    e2e_duration = time.time() - e2e_start

    exact_match = (recovery_result.recovered_payload.data == secret_bytes)
    print(f"    AES-GCM Authenticated:   True (Decryption succeeded without tampering error)")
    print(f"    Exact Byte Match:        {exact_match}")
    print(f"    Recovered Payload Size:  {len(recovery_result.recovered_payload.data)} bytes")
    print(f"    E2E Pipeline Duration:   {e2e_duration:.2f} s")
    assert exact_match, "CRITICAL ERROR: End-to-end recovered bytes do not match original secret!"

    experiment_results["end_to_end_recovery_experiment"] = {
        "carrier_filename": sample_val["filename"],
        "carrier_image_id": sample_val["image_id"],
        "payload_size_bytes": payload_len,
        "protected_frame_size_bytes": len(actual_protected_bytes),
        "raw_bit_errors": bit_errors,
        "total_payload_bits": len(actual_payload_bits),
        "raw_pre_ecc_ber": pre_ecc_ber,
        "rs_blocks_count": rs_res.blocks_count,
        "rs_total_corrected_symbols": rs_res.corrected_symbols,
        "rs_per_codeword_symbol_errors": per_codeword_errors,
        "rs_max_correctable_symbols_per_block": rs_encoder.t,
        "rs_prefix_recovered": rs_res.prefix_recovered,
        "aes_gcm_authenticated": True,
        "exact_byte_recovery": exact_match,
        "stego_psnr_db": stego_result.psnr_db,
        "stego_ssim": stego_result.ssim,
        "e2e_duration_sec": e2e_duration,
        "status": "PASS",
    }

    # -------------------------------------------------------------------------
    # 7. Baseline vs. Phase 4B vs. Phase 4C Final vs. Phase 4C Best Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 4C COMPREHENSIVE PERFORMANCE COMPARISON")
    print("=" * 80)

    # Load Phase 4B baseline metrics
    p4b_results = json.loads(p4b_results_file.read_text(encoding="utf-8"))
    baseline_metrics = p4b_results["baseline_evaluation"]["metrics"]
    p4b_best_metrics = p4b_results["best_model_evaluation"]["metrics"]

    comparison = {
        "total_loss": {
            "baseline_untrained": baseline_metrics["total_loss"],
            "phase4b_best": p4b_best_metrics["total_loss"],
            "phase4c_final": final_metrics.total_loss,
            "phase4c_best": best_reloaded_metrics.total_loss,
            "delta_vs_baseline": best_reloaded_metrics.total_loss - baseline_metrics["total_loss"],
            "delta_vs_phase4b": best_reloaded_metrics.total_loss - p4b_best_metrics["total_loss"],
        },
        "loss_bit_raw": {
            "baseline_untrained": baseline_metrics["loss_bit_raw"],
            "phase4b_best": p4b_best_metrics["loss_bit_raw"],
            "phase4c_final": final_metrics.loss_bit_raw,
            "phase4c_best": best_reloaded_metrics.loss_bit_raw,
            "delta_vs_baseline": best_reloaded_metrics.loss_bit_raw - baseline_metrics["loss_bit_raw"],
            "delta_vs_phase4b": best_reloaded_metrics.loss_bit_raw - p4b_best_metrics["loss_bit_raw"],
        },
        "loss_ctx_raw": {
            "baseline_untrained": baseline_metrics["loss_ctx_raw"],
            "phase4b_best": p4b_best_metrics["loss_ctx_raw"],
            "phase4c_final": final_metrics.loss_ctx_raw,
            "phase4c_best": best_reloaded_metrics.loss_ctx_raw,
            "delta_vs_baseline": best_reloaded_metrics.loss_ctx_raw - baseline_metrics["loss_ctx_raw"],
            "delta_vs_phase4b": best_reloaded_metrics.loss_ctx_raw - p4b_best_metrics["loss_ctx_raw"],
        },
        "loss_gain_raw": {
            "baseline_untrained": baseline_metrics["loss_gain_raw"],
            "phase4b_best": p4b_best_metrics["loss_gain_raw"],
            "phase4c_final": final_metrics.loss_gain_raw,
            "phase4c_best": best_reloaded_metrics.loss_gain_raw,
            "delta_vs_baseline": best_reloaded_metrics.loss_gain_raw - baseline_metrics["loss_gain_raw"],
            "delta_vs_phase4b": best_reloaded_metrics.loss_gain_raw - p4b_best_metrics["loss_gain_raw"],
        },
        "loss_img_raw": {
            "baseline_untrained": baseline_metrics["loss_img_raw"],
            "phase4b_best": p4b_best_metrics["loss_img_raw"],
            "phase4c_final": final_metrics.loss_img_raw,
            "phase4c_best": best_reloaded_metrics.loss_img_raw,
            "delta_vs_baseline": best_reloaded_metrics.loss_img_raw - baseline_metrics["loss_img_raw"],
            "delta_vs_phase4b": best_reloaded_metrics.loss_img_raw - p4b_best_metrics["loss_img_raw"],
        },
        "pre_ecc_ber": {
            "baseline_untrained": baseline_metrics["pre_ecc_ber"],
            "phase4b_best": p4b_best_metrics["pre_ecc_ber"],
            "phase4c_final": final_metrics.pre_ecc_ber,
            "phase4c_best": best_reloaded_metrics.pre_ecc_ber,
            "delta_vs_baseline": best_reloaded_metrics.pre_ecc_ber - baseline_metrics["pre_ecc_ber"],
            "delta_vs_phase4b": best_reloaded_metrics.pre_ecc_ber - p4b_best_metrics["pre_ecc_ber"],
        },
        "stego_psnr_db": {
            "baseline_untrained": baseline_metrics["stego_psnr_db"],
            "phase4b_best": p4b_best_metrics["stego_psnr_db"],
            "phase4c_final": final_metrics.stego_psnr_db,
            "phase4c_best": best_reloaded_metrics.stego_psnr_db,
            "delta_vs_baseline": best_reloaded_metrics.stego_psnr_db - baseline_metrics["stego_psnr_db"],
            "delta_vs_phase4b": best_reloaded_metrics.stego_psnr_db - p4b_best_metrics["stego_psnr_db"],
        },
        "stego_ssim": {
            "baseline_untrained": baseline_metrics["stego_ssim"],
            "phase4b_best": p4b_best_metrics["stego_ssim"],
            "phase4c_final": final_metrics.stego_ssim,
            "phase4c_best": best_reloaded_metrics.stego_ssim,
            "delta_vs_baseline": best_reloaded_metrics.stego_ssim - baseline_metrics["stego_ssim"],
            "delta_vs_phase4b": best_reloaded_metrics.stego_ssim - p4b_best_metrics["stego_ssim"],
        },
        "clip_fraction": {
            "baseline_untrained": baseline_metrics["clip_fraction"],
            "phase4b_best": p4b_best_metrics["clip_fraction"],
            "phase4c_final": final_metrics.clip_fraction,
            "phase4c_best": best_reloaded_metrics.clip_fraction,
            "delta_vs_baseline": best_reloaded_metrics.clip_fraction - baseline_metrics["clip_fraction"],
            "delta_vs_phase4b": best_reloaded_metrics.clip_fraction - p4b_best_metrics["clip_fraction"],
        },
    }

    print(f"  Metric              | Untrained Base | Phase 4B Best  | Phase 4C Final | Phase 4C Best  | Delta vs P4B")
    print(f"  --------------------+----------------+----------------+----------------+----------------+-------------")
    for k, v in comparison.items():
        print(f"  {k:<19} | {v['baseline_untrained']:<14.6f} | {v['phase4b_best']:<14.6f} | {v['phase4c_final']:<14.6f} | {v['phase4c_best']:<14.6f} | {v['delta_vs_phase4b']:<+12.6f}")

    experiment_results["comparison"] = comparison
    experiment_results["completed"] = True

    # Write results JSON
    results_path.write_text(json.dumps(experiment_results, indent=2), encoding="utf-8")
    print(f"\n[Phase 4C Complete] Results saved to: {results_path}")

    return experiment_results


if __name__ == "__main__":
    run_phase4c_extended_experiment()
