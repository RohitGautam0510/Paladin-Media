"""Experiment Script: Phase 3 Sprint 3.1 Loss-Gradient Isolation and Scale Audit.

Performs isolated backward passes for each candidate loss component:
1. BitRecoveryLoss only through differentiable spatial path.
2. ContextPredictionLoss only through differentiable spatial path.
3. AdaptiveGainRegularization only through differentiable spatial path.
4. ImageFidelityLoss only through differentiable spatial path.
5. Combined CompositeLoss objective.

Evaluates on:
- Small synthetic carrier (64x64) with pseudo-random payload.
- Real DIV2K carrier (0005.png, 512x512 center crop) with 5,592-bit payload.

Measures:
- Scalar loss values and weighted contributions.
- Per-module gradient presence (None vs not None), finiteness, L2 norms, and max absolute values.
- Compares spatial-bridge bit-recovery gradient leakage against ideal coefficient-domain context cancellation.

Outputs:
- docs/experiments/sprint3_1_gradient_isolation_results.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.neural.acs_sub_inn import ACSSubINN
from src.neural.losses import (
    AdaptiveGainRegularization,
    BitRecoveryLoss,
    CompositeLoss,
    ContextPredictionLoss,
    ImageFidelityLoss,
)
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_scaffold import set_reproducibility_seeds


def get_module_gradient_metrics(model: torch.nn.Module) -> Dict[str, Any]:
    """Calculates gradient presence, finiteness, L2 norm, and max abs value for a module."""
    grads = [p.grad for p in model.parameters()]
    present_count = sum(1 for g in grads if g is not None)
    total_count = len(grads)

    if present_count == 0:
        return {
            "present": False,
            "finite": True,
            "param_count_with_grad": 0,
            "total_param_count": total_count,
            "l2_norm": 0.0,
            "max_abs_grad": 0.0,
            "mean_abs_grad": 0.0,
        }

    valid_grads = [g for g in grads if g is not None]
    is_finite = all(torch.isfinite(g).all().item() for g in valid_grads)
    l2_norm = float(torch.sqrt(sum((g ** 2).sum() for g in valid_grads)).item())
    max_abs = float(max(g.abs().max().item() for g in valid_grads))
    mean_abs = float(np.mean([g.abs().mean().item() for g in valid_grads]))

    return {
        "present": True,
        "finite": is_finite,
        "param_count_with_grad": present_count,
        "total_param_count": total_count,
        "l2_norm": l2_norm,
        "max_abs_grad": max_abs,
        "mean_abs_grad": mean_abs,
    }


def evaluate_isolated_losses(
    bridge: NeuralSpatialBridge,
    carrier: torch.Tensor,
    payload_bipolar: torch.Tensor,
    mask: torch.Tensor,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Evaluates each candidate loss term in strict isolation, then combined."""
    # Step 1: Forward Differentiable Embedding & Extraction
    emb_res = bridge.embed_spatial(carrier, payload_bipolar, mask, differentiable=True)
    ext_res = bridge.extract_spatial(emb_res.stego_image, mask)

    hh_drop_orig = emb_res.subbands_original[3][:, 0:1, :, :]
    context_pred = emb_res.acs_embedding_result.context
    gamma = emb_res.acs_embedding_result.gamma

    # Individual loss modules
    loss_modules = {
        "bit_recovery": BitRecoveryLoss(formulation="logistic", temperature=1.0),
        "context_prediction": ContextPredictionLoss(formulation="smooth_l1", beta=1.0),
        "gain_regularization": AdaptiveGainRegularization(formulation="excess_l1", gamma_min=1.0),
        "image_fidelity": ImageFidelityLoss(formulation="l1"),
    }

    # Evaluate loss values
    raw_losses = {
        "bit_recovery": loss_modules["bit_recovery"](ext_res.delta_rec, payload_bipolar, mask),
        "context_prediction": loss_modules["context_prediction"](context_pred, hh_drop_orig),
        "gain_regularization": loss_modules["gain_regularization"](gamma, mask),
        "image_fidelity": loss_modules["image_fidelity"](emb_res.stego_image, carrier),
    }

    isolated_gradient_results = {}

    # Run separate backward passes for each individual loss
    for loss_name, loss_tensor in raw_losses.items():
        bridge.zero_grad()
        loss_tensor.backward(retain_graph=True)

        ctx_metrics = get_module_gradient_metrics(bridge.acs_model.context_predictor)
        gain_metrics = get_module_gradient_metrics(bridge.acs_model.adaptive_gain)

        w = weights.get(loss_name, 1.0)
        raw_val = float(loss_tensor.item())
        weighted_val = raw_val * w

        isolated_gradient_results[loss_name] = {
            "raw_loss": raw_val,
            "weight": w,
            "weighted_loss": weighted_val,
            "context_predictor": ctx_metrics,
            "adaptive_gain": gain_metrics,
        }

    # Run backward pass for Combined Composite Loss
    bridge.zero_grad()
    comp_loss_fn = CompositeLoss(
        lambda_bit=weights["bit_recovery"],
        lambda_ctx=weights["context_prediction"],
        lambda_gain=weights["gain_regularization"],
        lambda_img=weights["image_fidelity"],
    ).to(carrier.device)

    breakdown = comp_loss_fn(
        delta_rec=ext_res.delta_rec,
        payload_bipolar=payload_bipolar,
        predicted_context=context_pred,
        target_hh_drop=hh_drop_orig,
        gamma=gamma,
        stego_spatial=emb_res.stego_image,
        cover_spatial=carrier,
        mask=mask,
    )
    breakdown.total_loss.backward()

    combined_ctx_metrics = get_module_gradient_metrics(bridge.acs_model.context_predictor)
    combined_gain_metrics = get_module_gradient_metrics(bridge.acs_model.adaptive_gain)

    all_grads = [p.grad for p in bridge.acs_model.parameters() if p.grad is not None]
    combined_total_norm = float(torch.sqrt(sum((g ** 2).sum() for g in all_grads)).item())

    # Ideal coefficient-domain cancellation comparison test
    bridge.zero_grad()
    hh_drop_coef = hh_drop_orig.detach().clone()
    hh_keep_coef = emb_res.subbands_original[3][:, 1:3, :, :].detach().clone()
    acs_emb = bridge.acs_model.embed_coefficients(hh_drop_coef, hh_keep_coef, mask, payload_bipolar)
    acs_ext = bridge.acs_model.extract_coefficients(acs_emb.hh_drop_stego, hh_keep_coef, mask)
    ideal_loss = loss_modules["bit_recovery"](acs_ext.delta_rec, payload_bipolar, mask)
    ideal_loss.backward()

    ideal_ctx_metrics = get_module_gradient_metrics(bridge.acs_model.context_predictor)
    ideal_gain_metrics = get_module_gradient_metrics(bridge.acs_model.adaptive_gain)

    return {
        "isolated_losses": isolated_gradient_results,
        "combined_objective": {
            "total_scalar_loss": float(breakdown.total_loss.item()),
            "context_predictor": combined_ctx_metrics,
            "adaptive_gain": combined_gain_metrics,
            "total_grad_norm": combined_total_norm,
        },
        "context_cancellation_comparison": {
            "ideal_coefficient_domain": {
                "context_predictor_norm": ideal_ctx_metrics["l2_norm"],
                "context_predictor_max_abs": ideal_ctx_metrics["max_abs_grad"],
                "adaptive_gain_norm": ideal_gain_metrics["l2_norm"],
                "explanation": "delta_rec = (context + gamma*P) - context = gamma*P; d(delta)/d(context) == 0 exact.",
            },
            "spatial_bridge_path": {
                "context_predictor_norm": isolated_gradient_results["bit_recovery"]["context_predictor"]["l2_norm"],
                "context_predictor_max_abs": isolated_gradient_results["bit_recovery"]["context_predictor"]["max_abs_grad"],
                "adaptive_gain_norm": isolated_gradient_results["bit_recovery"]["adaptive_gain"]["l2_norm"],
                "explanation": (
                    "Through spatial IDWT, STE clamp/rounding, and DWT, hh_keep_rec undergoes small spatial rounding "
                    "noise eps_keep != 0. context_rec is evaluated at hh_keep_rec, breaking exact cancellation."
                ),
            },
        },
    }


def main() -> None:
    print("=" * 80)
    print("PHASE 3 SPRINT 3.1: LOSS-GRADIENT ISOLATION AND SCALE AUDIT")
    print("=" * 80)

    set_reproducibility_seeds(42, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device.type.upper()}")

    weights = {
        "bit_recovery": 1.0,
        "context_prediction": 0.5,
        "gain_regularization": 0.01,
        "image_fidelity": 0.1,
    }

    # -------------------------------------------------------------------------
    # Test A: Synthetic Carrier (64x64, 500 active bits)
    # -------------------------------------------------------------------------
    print("\n--- TEST A: SYNTHETIC CARRIER (64x64, 500 active positions) ---")
    bridge_syn = NeuralSpatialBridge().to(device)
    bridge_syn.train()

    rng = np.random.RandomState(42)
    syn_carrier_np = rng.randint(20, 235, (64, 64, 3), dtype=np.uint8)
    syn_carrier = torch.from_numpy(syn_carrier_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

    syn_payload_bits = torch.from_numpy(rng.randint(0, 2, (32, 32), dtype=np.uint8)).unsqueeze(0).unsqueeze(0).to(device)
    syn_payload_bipolar = syn_payload_bits.float() * 2.0 - 1.0
    syn_mask = torch.zeros(1, 1, 32, 32, device=device)
    syn_mask.view(-1)[:500] = 1.0

    synthetic_results = evaluate_isolated_losses(bridge_syn, syn_carrier, syn_payload_bipolar, syn_mask, weights)

    # Print Synthetic Summary
    print(f"{'Loss Term':<22} | {'Raw Loss':<10} | {'Weight':<8} | {'Weighted':<10} | {'Ctx Norm':<12} | {'Gain Norm':<12}")
    print("-" * 85)
    for name, r in synthetic_results["isolated_losses"].items():
        ctx_n = r["context_predictor"]["l2_norm"]
        gain_n = r["adaptive_gain"]["l2_norm"]
        print(
            f"{name:<22} | {r['raw_loss']:<10.4f} | {r['weight']:<8.2f} | {r['weighted_loss']:<10.4f} | "
            f"{ctx_n:<12.6f} | {gain_n:<12.6f}"
        )

    comb = synthetic_results["combined_objective"]
    print("-" * 85)
    print(
        f"{'Combined Objective':<22} | {comb['total_scalar_loss']:<10.4f} | {'N/A':<8} | {comb['total_scalar_loss']:<10.4f} | "
        f"{comb['context_predictor']['l2_norm']:<12.6f} | {comb['adaptive_gain']['l2_norm']:<12.6f}"
    )

    # -------------------------------------------------------------------------
    # Test B: Real DIV2K Carrier (0005.png, 512x512 crop, 5,592 bits)
    # -------------------------------------------------------------------------
    print("\n--- TEST B: REAL DIV2K CARRIER (0005.png, 512x512, 5,592 positions) ---")
    div2k_path = REPO_ROOT / "dataset" / "DIV2K" / "DIV2K_train_HR" / "DIV2K_train_HR" / "0005.png"
    if not div2k_path.exists():
        print(f"Error: {div2k_path} not found.")
        sys.exit(1)

    img = Image.open(div2k_path).convert("RGB")
    w, h = img.size
    crop = img.crop(((w - 512) // 2, (h - 512) // 2, (w - 512) // 2 + 512, (h - 512) // 2 + 512))
    div2k_carrier = torch.from_numpy(np.array(crop, dtype=np.uint8).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

    # 5,592 bits (corresponding to 500 B payload tier with RS overhead)
    rng_500 = np.random.RandomState(500)
    div2k_bits = torch.from_numpy(rng_500.randint(0, 2, (256, 256), dtype=np.uint8)).unsqueeze(0).unsqueeze(0).to(device)
    div2k_bipolar = div2k_bits.float() * 2.0 - 1.0
    div2k_mask = torch.zeros(1, 1, 256, 256, device=device)
    div2k_mask.view(-1)[:5592] = 1.0

    bridge_real = NeuralSpatialBridge().to(device)
    bridge_real.train()

    real_carrier_results = evaluate_isolated_losses(bridge_real, div2k_carrier, div2k_bipolar, div2k_mask, weights)

    print(f"{'Loss Term':<22} | {'Raw Loss':<10} | {'Weight':<8} | {'Weighted':<10} | {'Ctx Norm':<12} | {'Gain Norm':<12}")
    print("-" * 85)
    for name, r in real_carrier_results["isolated_losses"].items():
        ctx_n = r["context_predictor"]["l2_norm"]
        gain_n = r["adaptive_gain"]["l2_norm"]
        print(
            f"{name:<22} | {r['raw_loss']:<10.4f} | {r['weight']:<8.2f} | {r['weighted_loss']:<10.4f} | "
            f"{ctx_n:<12.6f} | {gain_n:<12.6f}"
        )

    comb_real = real_carrier_results["combined_objective"]
    print("-" * 85)
    print(
        f"{'Combined Objective':<22} | {comb_real['total_scalar_loss']:<10.4f} | {'N/A':<8} | {comb_real['total_scalar_loss']:<10.4f} | "
        f"{comb_real['context_predictor']['l2_norm']:<12.6f} | {comb_real['adaptive_gain']['l2_norm']:<12.6f}"
    )

    # Context cancellation summary
    print("\n--- CONTEXT CANCELLATION COMPARISON ---")
    canc_syn = synthetic_results["context_cancellation_comparison"]
    print(f"Synthetic Carrier:")
    print(f"  Ideal Coef Domain Context Norm: {canc_syn['ideal_coefficient_domain']['context_predictor_norm']:.8f}")
    print(f"  Spatial Bridge Context Norm:    {canc_syn['spatial_bridge_path']['context_predictor_norm']:.8f}")
    canc_real = real_carrier_results["context_cancellation_comparison"]
    print(f"Real DIV2K Carrier (0005.png):")
    print(f"  Ideal Coef Domain Context Norm: {canc_real['ideal_coefficient_domain']['context_predictor_norm']:.8f}")
    print(f"  Spatial Bridge Context Norm:    {canc_real['spatial_bridge_path']['context_predictor_norm']:.8f}")

    # -------------------------------------------------------------------------
    # Test C: Clipping Boundary Leakage Analysis (32x32 saturated carrier)
    # -------------------------------------------------------------------------
    print("\n--- TEST C: CLIPPING BOUNDARY LEAKAGE ANALYSIS ---")
    bridge_clip = NeuralSpatialBridge().to(device)
    bridge_clip.quantizer.clip_grad = True
    clip_carrier = torch.full((1, 3, 32, 32), 255.0, device=device)
    clip_payload = torch.ones(1, 1, 16, 16, device=device)
    clip_mask = torch.ones(1, 1, 16, 16, device=device)

    emb_clip = bridge_clip.embed_spatial(clip_carrier, clip_payload, clip_mask, differentiable=True)
    ext_clip = bridge_clip.extract_spatial(emb_clip.stego_image, clip_mask)
    bit_loss_fn = BitRecoveryLoss().to(device)
    loss_clip = bit_loss_fn(ext_clip.delta_rec, clip_payload, clip_mask)
    loss_clip.backward()

    ctx_clip_metrics = get_module_gradient_metrics(bridge_clip.acs_model.context_predictor)
    gain_clip_metrics = get_module_gradient_metrics(bridge_clip.acs_model.adaptive_gain)

    print(f"Clipped Pixels: {emb_clip.clip_stats['num_clipped_total']} / {emb_clip.stego_image.numel()} "
          f"({emb_clip.clip_stats['clip_fraction']*100:.2f}%)")
    print(f"ContextPredictor L2 Norm under Clipping Leakage: {ctx_clip_metrics['l2_norm']:.6f}")
    print(f"AdaptiveGain L2 Norm under Clipping Leakage:     {gain_clip_metrics['l2_norm']:.6f}")

    clipping_leakage_results = {
        "clip_stats": emb_clip.clip_stats,
        "clip_grad_enabled": True,
        "context_predictor": ctx_clip_metrics,
        "adaptive_gain": gain_clip_metrics,
        "explanation": (
            "When STE gradient clipping (clip_grad=True) is active on pixels clamped at 255.0, "
            "the spatial IDWT -> STE -> DWT path no longer inverts to Identity (dL/dx = 0 at clamped pixels). "
            "This breaks the exact coefficient-domain context cancellation, causing non-zero gradient leakage."
        ),
    }

    # Compile structured JSON artifact
    output_data = {
        "metadata": {
            "experiment": "Phase 3 Sprint 3.1 Loss-Gradient Isolation and Scale Audit",
            "date": "2026-10-19",
            "device": device.type,
            "weights_configuration": weights,
            "training_authorized": False,
        },
        "synthetic_carrier_results": synthetic_results,
        "real_div2k_carrier_results": real_carrier_results,
        "clipping_leakage_experiment": clipping_leakage_results,
    }

    out_file = REPO_ROOT / "docs" / "experiments" / "sprint3_1_gradient_isolation_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[OK] Versioned gradient isolation results saved to: {out_file}")


if __name__ == "__main__":
    main()
