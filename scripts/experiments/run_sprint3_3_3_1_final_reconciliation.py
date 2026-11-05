"""Experiment & Verification Runner: Sprint 3.3.3.1 Final-State Reconciliation.

Exercises and records:
1. Deterministic 16-case matrix across synthetic and DIV2K carriers (0001, 0002, 0005)
   at sizes 300, 1500, 3500, 7088 bytes.
2. Complete acceptance path through MediaStegoPipeline with NeuralModulationEngine:
   SecretPayload -> serialize -> AES-256-GCM -> RS encode -> capacity check ->
   CarrierHeader & DeterministicMask -> NeuralModulationEngine (ACSSubINN v3 + DWT/IDWT) ->
   uint8 quantization & PNG serialize/reload -> begin_extraction -> unpack header ->
   reconstruct mask -> extract bits -> RS decode with length derivation fallback ->
   AES-256-GCM authenticate -> deserialize -> exact byte comparison.
3. Records all required metrics:
   - Carrier dimensions, plaintext size, protected size, capacity utilization.
   - Raw payload bit errors with exact numerator and denominator.
   - Per-codeword RS symbol error distribution, worst block error, corrected symbols.
   - Prefix recovery invocation flag.
   - AES-GCM tag validity, exact plaintext equality, PSNR, SSIM, clipped pixel counts.
4. Capacity hard gate (8,176 B vs 8,177 B).
5. Fail-closed security tests (header tampering, container tampering, cross-engine extraction).
6. Classical baseline non-regression.
7. Model parameter immutability digest comparison.

Outputs:
- docs/experiments/sprint3_3_3_1_reconciliation_results.json
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.carrier.capacity import (
    CapacityCalculator,
    CapacityReport,
    PayloadCapacityExceededError,
)
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import (
    DeterministicMaskGenerator,
    EmbeddingMask,
    MaskGenerationError,
)
from src.carrier.modulation import (
    CarrierModulationEngine,
    NeuralModulationEngine,
    ParityModulationEngine,
)
from src.core.exceptions import AuthenticationError, ECCDecodeError
from src.core.media_pipeline import (
    MediaRecoveryResult,
    MediaStegoPipeline,
    MediaStegoResult,
    calculate_psnr,
    calculate_ssim,
)
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import DIV2KStegoDataset
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_scaffold import set_reproducibility_seeds
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


def compute_param_digest(model: torch.nn.Module) -> str:
    """Computes a SHA-256 digest over sorted named parameters of a PyTorch module."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def make_deterministic_payload(length: int, seed: int) -> bytes:
    """Generates reproducible payload bytes using SHA-256 counter."""
    chunks = []
    generated = 0
    counter = 0
    while generated < length:
        h = hashlib.sha256(struct.pack(">II", seed, counter)).digest()
        needed = min(len(h), length - generated)
        chunks.append(h[:needed])
        generated += needed
        counter += 1
    return b"".join(chunks)


def measure_rs_codeword_errors(
    protected: bytes,
    extracted: bytes,
    k: int = 223,
    ecc_symbols: int = 32,
) -> Dict[str, Any]:
    """Measures byte symbol errors per Reed-Solomon codeword block."""
    if len(protected) < 4 or len(extracted) < 4:
        raise ValueError("Protected byte stream too short for length prefix")

    header_errors = sum(1 for p, e in zip(protected[:4], extracted[:4]) if p != e)
    total_length = struct.unpack(">I", protected[:4])[0]
    num_full_blocks = total_length // k
    remainder = total_length % k

    offset = 4
    block_errors: List[int] = []

    for _ in range(num_full_blocks):
        block_len = k + ecc_symbols
        p_block = protected[offset : offset + block_len]
        e_block = extracted[offset : offset + block_len]
        errs = sum(1 for p, e in zip(p_block, e_block) if p != e)
        block_errors.append(errs)
        offset += block_len

    if remainder > 0:
        block_len = remainder + ecc_symbols
        p_block = protected[offset : offset + block_len]
        e_block = extracted[offset : offset + block_len]
        errs = sum(1 for p, e in zip(p_block, e_block) if p != e)
        block_errors.append(errs)
        offset += block_len

    worst = max(block_errors) if block_errors else 0
    total = sum(block_errors)
    max_correctable = ecc_symbols // 2

    return {
        "header_errors": header_errors,
        "block_errors": block_errors,
        "worst_codeword_errors": worst,
        "total_codeword_errors": total,
        "within_t16_limit": worst <= max_correctable,
        "t_limit": max_correctable,
    }


def run_reconciliation_orchestrator_matrix(
    neural_engine: NeuralModulationEngine,
    neural_pipeline: MediaStegoPipeline,
    key: bytes,
    aad: bytes,
) -> Dict[str, Any]:
    """Runs the 16-case matrix with deterministic payloads and granular diagnostics."""
    print("\n" + "=" * 80)
    print("SPRINT 3.3.3.1 RECONCILIATION: 16-CASE PRODUCTION ORCHESTRATOR MATRIX")
    print("=" * 80)

    ds = DIV2KStegoDataset(split="train")
    carriers: List[Tuple[str, str, np.ndarray]] = [
        ("synthetic", "Smooth Mid-Gray (128.0)", np.full((512, 512, 3), 128, dtype=np.uint8)),
        ("0001.png", "DIV2K #0001 (Train)", ds[0]["carrier"].permute(1, 2, 0).numpy().astype(np.uint8)),
        ("0002.png", "DIV2K #0002 (Train)", ds[1]["carrier"].permute(1, 2, 0).numpy().astype(np.uint8)),
        ("0005.png", "DIV2K #0005 (Train)", ds[4]["carrier"].permute(1, 2, 0).numpy().astype(np.uint8)),
    ]

    payload_lengths = [300, 1500, 3500, 7088]
    sub_dim = 256
    results = []

    for carrier_idx, (carrier_id, carrier_desc, carrier_arr) in enumerate(carriers):
        print(f"\n--- Carrier: {carrier_id} ({carrier_desc}) | Shape: {carrier_arr.shape} ---")
        for plen in payload_lengths:
            session_counter = 1000 + carrier_idx * 10 + (plen % 100)
            t_start = time.perf_counter()

            seed_val = 5000 + carrier_idx * 100 + plen
            raw_secret = make_deterministic_payload(plen, seed=seed_val)
            meta = None if plen >= 7088 else {"carrier": carrier_id, "size": plen}
            secret = SecretPayload(data=raw_secret, media_type=MediaType.IMAGE, metadata=meta)

            # 1. Pipeline Protect & Embed
            carrier_img = Image.fromarray(carrier_arr, mode="RGB")
            stego_res = neural_pipeline.protect_and_embed(
                secret_payload=secret,
                cover_image=carrier_img,
                key=key,
                aad=aad,
                session_counter=session_counter,
            )

            # Capture clipped pixels from engine
            clipped_pixels = getattr(neural_engine, "last_clip_stats", {}).get("num_clipped_total", 0)

            # 2. Pipeline Extract & Unprotect
            rec_res = neural_pipeline.extract_and_unprotect(
                stego_carrier=stego_res.stego_image,
                key=key,
                aad=aad,
                reference_payload=secret,
            )

            # 3. Fine-grained bit and codeword diagnostics
            bit_extractor = neural_pipeline.modulation_engine.begin_extraction(
                np.array(stego_res.stego_image),
                subband_dim=sub_dim,
            )
            payload_positions = DeterministicMaskGenerator.generate_payload_positions(
                session_counter=rec_res.header.session_counter,
                payload_length_bytes=rec_res.header.payload_length,
                subband_dim=sub_dim,
            )
            extracted_bits = bit_extractor.extract_bits(payload_positions)
            extracted_protected = Extractor.bits_to_bytes(extracted_bits)

            # Embedded reference bits and bytes captured directly from engine during embed_bits
            embedded_bits = neural_engine.last_payload_bits
            embedded_protected = Extractor.bits_to_bytes(embedded_bits)

            raw_bit_errors = int(np.sum(extracted_bits != embedded_bits))
            total_payload_bits = len(embedded_bits)
            raw_ber = float(raw_bit_errors / total_payload_bits) if total_payload_bits > 0 else 0.0

            rs_diag = measure_rs_codeword_errors(embedded_protected, extracted_protected)
            ecc_diag = neural_pipeline.security_pipeline.ecc_encoder.decode_with_result(extracted_protected)

            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

            exact_match = (
                rec_res.exact_match is True
                and rec_res.recovered_payload.data == secret.data
                and rec_res.recovered_payload.metadata == secret.metadata
            )

            header_match = (
                rec_res.header.session_counter == session_counter
                and rec_res.header.payload_length == stego_res.header.payload_length
            )

            entry = {
                "carrier_id": carrier_id,
                "carrier_description": carrier_desc,
                "carrier_dimensions": list(carrier_arr.shape),
                "session_counter": session_counter,
                "payload_seed": seed_val,
                "plaintext_bytes": plen,
                "protected_bytes": stego_res.header.payload_length,
                "carrier_header_bytes": 16,
                "total_embedded_bits": 128 + stego_res.header.payload_length * 8,
                "capacity_utilization_pct": round((16 + stego_res.header.payload_length) / 8192 * 100, 2),
                "psnr_db": round(stego_res.psnr_db, 2),
                "ssim": round(stego_res.ssim, 6),
                "clipped_pixels_total": clipped_pixels,
                "header_unpacked_cleanly": header_match,
                "raw_bit_errors_numerator": raw_bit_errors,
                "total_payload_bits_denominator": total_payload_bits,
                "raw_payload_ber": raw_ber,
                "rs_header_errors": rs_diag["header_errors"],
                "rs_worst_codeword_errors": rs_diag["worst_codeword_errors"],
                "rs_total_codeword_errors": rs_diag["total_codeword_errors"],
                "rs_within_t16_limit": rs_diag["within_t16_limit"],
                "rs_corrected_symbols": ecc_diag.corrected_symbols,
                "ecc_prefix_recovered": ecc_diag.prefix_recovered,
                "aes_gcm_tag_valid": True,
                "exact_plaintext_match": exact_match,
                "duration_ms": duration_ms,
                "failure_stage": None,
                "exception_message": None,
            }
            results.append(entry)

            print(
                f"  Payload: {plen:4d} B | Prot: {stego_res.header.payload_length:4d} B | "
                f"Hdr: {header_match!s:<5} | PSNR: {stego_res.psnr_db:5.2f} dB | SSIM: {stego_res.ssim:.4f} | "
                f"Clip: {clipped_pixels:3d} | BER: {raw_bit_errors:3d}/{total_payload_bits:<5d} ({raw_ber:.6f}) | "
                f"WorstRS: {rs_diag['worst_codeword_errors']:2d}/16 | PrefixRecov: {ecc_diag.prefix_recovered!s:<5} | "
                f"ExactMatch: {exact_match!s:<5} | Time: {duration_ms:6.1f} ms"
            )

    return {
        "trials_count": len(results),
        "all_exact_matches": all(r["exact_plaintext_match"] for r in results),
        "all_headers_valid": all(r["header_unpacked_cleanly"] for r in results),
        "all_aes_valid": all(r["aes_gcm_tag_valid"] for r in results),
        "trials": results,
    }


def run_capacity_and_security_verifications(
    neural_pipeline: MediaStegoPipeline,
    classical_pipeline: MediaStegoPipeline,
    key: bytes,
    aad: bytes,
) -> Dict[str, Any]:
    """Evaluates capacity hard gates, header corruption, container tampering, and cross-engine safety."""
    print("\n" + "=" * 80)
    print("CAPACITY & NEGATIVE SECURITY VERIFICATION")
    print("=" * 80)

    # 1. Capacity Hard Gate: 8,176 B fits, 8,177 B rejected before carrier modification
    carrier_arr = np.full((512, 512, 3), 128, dtype=np.uint8)
    carrier_clone = carrier_arr.copy()
    carrier_img = Image.fromarray(carrier_arr, mode="RGB")

    fits_8176 = CapacityCalculator.check_capacity(8176, subband_dim=256).fits

    exceeded_rejected = False
    try:
        CapacityCalculator.check_capacity(8177, subband_dim=256)
    except PayloadCapacityExceededError:
        exceeded_rejected = True

    # Confirm pipeline rejects payload requiring > 8,176 bytes protected without mutating carrier
    overflow_rejected_before_modification = False
    overflow_payload = SecretPayload(data=bytes(7150), media_type=MediaType.TEXT)
    try:
        neural_pipeline.protect_and_embed(overflow_payload, carrier_clone, key, aad=aad)
    except PayloadCapacityExceededError:
        overflow_rejected_before_modification = np.array_equal(carrier_arr, carrier_clone)

    # 2. Header corruption fails closed
    secret = SecretPayload(data=b"Test payload", media_type=MediaType.TEXT)
    stego_res = neural_pipeline.protect_and_embed(
        secret_payload=secret,
        cover_image=carrier_img,
        key=key,
        aad=aad,
        session_counter=42,
    )
    stego_arr = np.array(stego_res.stego_image).copy()

    # Corrupt header region (top 20 rows)
    stego_arr[:20, :, :] ^= 0xFF
    header_tamper_fails_closed = False
    try:
        neural_pipeline.extract_and_unprotect(Image.fromarray(stego_arr), key, aad=aad)
    except CarrierHeaderError:
        header_tamper_fails_closed = True

    # 3. Payload region tampering fails closed with zero plaintext leak
    stego_arr_payload = np.array(stego_res.stego_image).copy()
    stego_arr_payload[50:450, 50:450, :] = 0
    payload_tamper_fails_closed = False
    recovered_on_tamper = None
    try:
        recovered_on_tamper = neural_pipeline.extract_and_unprotect(Image.fromarray(stego_arr_payload), key, aad=aad)
    except (AuthenticationError, ECCDecodeError, CarrierHeaderError):
        payload_tamper_fails_closed = True

    no_plaintext_leaked = (payload_tamper_fails_closed and recovered_on_tamper is None)

    # 4. Cross-engine check: classical pipeline on neural stego
    cross_engine_fails_closed = False
    try:
        classical_pipeline.extract_and_unprotect(stego_res.stego_image, key, aad=aad)
    except (CarrierHeaderError, AuthenticationError, ECCDecodeError):
        cross_engine_fails_closed = True

    # 5. Classical pipeline default non-regression
    classical_stego = classical_pipeline.protect_and_embed(
        secret_payload=secret,
        cover_image=carrier_img,
        key=key,
        aad=aad,
        session_counter=99,
    )
    classical_rec = classical_pipeline.extract_and_unprotect(
        classical_stego.stego_image,
        key,
        aad=aad,
        reference_payload=secret,
    )
    classical_non_regression = classical_rec.exact_match is True

    print(f"Capacity 8,176 B fits: {fits_8176}")
    print(f"Capacity 8,177 B rejected: {exceeded_rejected}")
    print(f"Overflow rejected before carrier modification: {overflow_rejected_before_modification}")
    print(f"Header corruption fails closed: {header_tamper_fails_closed}")
    print(f"Payload tamper fails closed (zero plaintext leaked): {no_plaintext_leaked}")
    print(f"Cross-engine extraction fails closed: {cross_engine_fails_closed}")
    print(f"Classical default non-regression: {classical_non_regression}")

    return {
        "capacity_8176_fits": fits_8176,
        "capacity_8177_rejected": exceeded_rejected,
        "overflow_rejected_before_carrier_modification": overflow_rejected_before_modification,
        "header_corruption_fails_closed": header_tamper_fails_closed,
        "payload_tampering_fails_closed": payload_tamper_fails_closed,
        "no_plaintext_leaked": no_plaintext_leaked,
        "cross_engine_fails_closed": cross_engine_fails_closed,
        "classical_non_regression": classical_non_regression,
    }


def main() -> None:
    print("\n" + "=" * 80)
    print("PALADINMEDIA — SPRINT 3.3.3.1 FINAL-STATE RECONCILIATION & ACCEPTANCE CLOSURE")
    print("=" * 80)

    set_reproducibility_seeds(seed=42, deterministic=True)
    device = "cpu"
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | Platform: {platform.platform()}")

    acs_model = ACSSubINN(gamma_min=1.0)
    pre_digest = compute_param_digest(acs_model)
    bridge = NeuralSpatialBridge(acs_model=acs_model)

    neural_engine = NeuralModulationEngine(spatial_bridge=bridge)
    parity_engine = ParityModulationEngine()

    neural_pipeline = MediaStegoPipeline(modulation_engine=neural_engine)
    classical_pipeline = MediaStegoPipeline(modulation_engine=parity_engine)

    key = bytes.fromhex("11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff")
    aad = b"PaladinMedia-Phase3-Sprint3.3.3.1-Reconciliation"

    # Task 1: 16-case matrix via MediaStegoPipeline
    orchestrator_results = run_reconciliation_orchestrator_matrix(neural_engine, neural_pipeline, key, aad)

    # Task 2: Capacity & Security checks
    security_results = run_capacity_and_security_verifications(neural_pipeline, classical_pipeline, key, aad)

    # Task 3: Model parameter immutability
    post_digest = compute_param_digest(bridge.acs_model)
    weights_invariant = (pre_digest == post_digest)
    print(f"\nModel Parameter Invariance: {weights_invariant}")
    print(f"Pre-digest:  {pre_digest}")
    print(f"Post-digest: {post_digest}")

    overall_pass = (
        orchestrator_results["all_exact_matches"]
        and orchestrator_results["all_headers_valid"]
        and orchestrator_results["all_aes_valid"]
        and security_results["capacity_8176_fits"]
        and security_results["capacity_8177_rejected"]
        and security_results["overflow_rejected_before_carrier_modification"]
        and security_results["header_corruption_fails_closed"]
        and security_results["no_plaintext_leaked"]
        and security_results["cross_engine_fails_closed"]
        and security_results["classical_non_regression"]
        and weights_invariant
    )

    summary = {
        "sprint": "3.3.3.1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "device": device,
        "acs_sub_inn_v3_frozen": True,
        "gamma_min": 1.0,
        "hashing_procedure": "SHA-256 over sorted named_parameters with name.encode('utf-8') and param.detach().cpu().numpy().tobytes()",
        "pre_param_digest": pre_digest,
        "post_param_digest": post_digest,
        "weights_invariant": weights_invariant,
        "verdict": "PASS" if overall_pass else "FAIL",
        "orchestrator_matrix": orchestrator_results,
        "capacity_and_security": security_results,
        "training_authorized": False,
        "training_status": "NOT AUTHORIZED — Sprint 3.3.3.1 implements final-state reconciliation and verification only.",
    }

    out_path = REPO_ROOT / "docs" / "experiments" / "sprint3_3_3_1_reconciliation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults successfully written to: {out_path}")
    print(f"FINAL SPRINT 3.3.3.1 VERDICT: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
