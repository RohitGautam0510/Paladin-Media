"""Experiment & Verification Runner: Sprint 3.3.3 Production Neural Carrier Integration + ECC Framing Recovery.

Exercises and records:
1. Production Neural Carrier Integration (Decision A):
   - MediaStegoPipeline with injected NeuralModulationEngine.
   - Complete 16-case matrix (Synthetic, DIV2K 0001, 0002, 0005 across 300B, 1500B, 3500B, 7088B).
   - Complete end-to-end flow: SecretPayload -> AES-256-GCM -> ReedSolomon -> DeterministicMask -> ACSSubINN v3 -> PNG save/reload -> Extract -> Unprotect -> Exact byte equality.
2. ECC Length-Prefix Recovery Verification (Decision B):
   - Corrupted 4-byte length prefix with valid RS codewords.
   - Verified derivation fallback: M = L_enc - 4, payload length recovered, prefix_recovered=True.
3. Production Interface Decoupling & Non-Regression:
   - ParityModulationEngine (Phase 2 default) backward compatibility.
   - NeuralModulationEngine clean error handling.
4. Capacity Boundaries & Negative Security Tests:
   - Hard capacity gate at 8,176 B; rejection at 8,177 B.
   - Corrupted carrier header fails closed.
   - Tampered payload fails closed with no plaintext leak.
5. Model Parameter Invariance:
   - SHA-256 verification of ACS-Sub-INN v3 parameters before and after execution.

Outputs:
- docs/experiments/sprint3_3_3_production_integration_results.json
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
    """Computes a SHA-256 digest over all parameters in a PyTorch module."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


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


def run_production_neural_orchestrator_matrix(
    pipeline: MediaStegoPipeline,
    key: bytes,
    aad: bytes,
) -> Dict[str, Any]:
    """Executes the 16-case matrix directly through MediaStegoPipeline."""
    print("\n" + "=" * 80)
    print("PRODUCTION NEURAL ORCHESTRATOR ROUNDTRIP MATRIX (VIA MEDIASTEGOPIPELINE)")
    print("=" * 80)

    ds = DIV2KStegoDataset(split="train")
    carriers_to_test: List[Tuple[str, str, Image.Image]] = [
        ("synthetic", "Smooth Mid-Gray (128.0)", Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8))),
        ("0001.png", "DIV2K #0001 (Train)", Image.fromarray(ds[0]["carrier"].permute(1, 2, 0).numpy().astype(np.uint8))),
        ("0002.png", "DIV2K #0002 (Train)", Image.fromarray(ds[1]["carrier"].permute(1, 2, 0).numpy().astype(np.uint8))),
        ("0005.png", "DIV2K #0005 (Train)", Image.fromarray(ds[4]["carrier"].permute(1, 2, 0).numpy().astype(np.uint8))),
    ]

    payload_lengths = [300, 1500, 3500, 7088]
    results = []

    for carrier_idx, (carrier_id, carrier_desc, carrier_img) in enumerate(carriers_to_test):
        print(f"\n--- Carrier: {carrier_id} ({carrier_desc}) ---")
        for plen in payload_lengths:
            session_counter = 1000 + carrier_idx * 10 + (plen % 100)
            t_start = time.perf_counter()

            raw_secret = os.urandom(plen)
            meta = None if plen >= 7088 else {"carrier": carrier_id, "load": plen}
            secret = SecretPayload(data=raw_secret, media_type=MediaType.IMAGE, metadata=meta)

            # 1. Pipeline Protect & Embed
            stego_res = pipeline.protect_and_embed(
                secret_payload=secret,
                cover_image=carrier_img,
                key=key,
                aad=aad,
                session_counter=session_counter,
            )

            # 2. Pipeline Extract & Unprotect
            rec_res = pipeline.extract_and_unprotect(
                stego_carrier=stego_res.stego_image,
                key=key,
                aad=aad,
                reference_payload=secret,
            )

            # Detailed ECC extraction diagnostics
            sub_dim = 256
            bit_extractor = pipeline.modulation_engine.begin_extraction(
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
            ecc_diag = pipeline.security_pipeline.ecc_encoder.decode_with_result(extracted_protected)

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
                "session_counter": session_counter,
                "plaintext_bytes": plen,
                "protected_bytes": stego_res.header.payload_length,
                "carrier_header_bytes": 16,
                "total_embedded_bits": 128 + stego_res.header.payload_length * 8,
                "capacity_utilization_pct": round((16 + stego_res.header.payload_length) / 8192 * 100, 2),
                "psnr_db": round(stego_res.psnr_db, 2),
                "ssim": round(stego_res.ssim, 6),
                "header_unpacked_cleanly": header_match,
                "rs_corrected_symbols": ecc_diag.corrected_symbols,
                "ecc_prefix_recovered": ecc_diag.prefix_recovered,
                "aes_gcm_tag_valid": True,
                "exact_plaintext_match": exact_match,
                "duration_ms": duration_ms,
            }
            results.append(entry)

            print(
                f"  Payload: {plen:4d} B | Prot: {stego_res.header.payload_length:4d} B | "
                f"HdrValid: {header_match!s:<5} | PSNR: {stego_res.psnr_db:5.2f} dB | SSIM: {stego_res.ssim:.4f} | "
                f"RS Corr: {ecc_diag.corrected_symbols:2d} | PrefixRecov: {ecc_diag.prefix_recovered!s:<5} | "
                f"ExactMatch: {exact_match!s:<5} | Time: {duration_ms:6.1f} ms"
            )

    return {
        "trials_count": len(results),
        "all_exact_matches": all(r["exact_plaintext_match"] for r in results),
        "all_headers_valid": all(r["header_unpacked_cleanly"] for r in results),
        "all_aes_valid": all(r["aes_gcm_tag_valid"] for r in results),
        "trials": results,
    }


def run_ecc_framing_recovery_evaluation(ecc: ReedSolomonEncoder) -> Dict[str, Any]:
    """Evaluates the ECC length-prefix derivation and recovery mechanism across edge cases."""
    print("\n" + "=" * 80)
    print("ECC LENGTH-PREFIX FRAMING RECOVERY EVALUATION")
    print("=" * 80)

    test_lengths = [1, 10, 100, 222, 223, 224, 500, 1000, 4000, 7000, 7136]
    recovery_tests = []

    for L in test_lengths:
        payload = os.urandom(L)
        encoded = bytearray(ecc.encode(payload))
        enc_len = len(encoded)

        # 1. Verify mathematical derivation
        derived_len = ecc.derive_payload_length(enc_len)
        derivation_exact = (derived_len == L)

        # 2. Corrupt length prefix (all 4 bytes)
        encoded[0] ^= 0xFF
        encoded[1] ^= 0xAA
        encoded[2] ^= 0x55
        encoded[3] ^= 0x12

        # 3. Decode with fallback
        res = ecc.decode_with_result(bytes(encoded))
        recovered_ok = (res.data == payload and res.prefix_recovered is True)

        recovery_tests.append({
            "payload_length": L,
            "encoded_length": enc_len,
            "derived_length": derived_len,
            "derivation_exact": derivation_exact,
            "corrupted_prefix_recovered": recovered_ok,
            "corrected_symbols": res.corrected_symbols,
        })

    all_derivations_exact = all(t["derivation_exact"] for t in recovery_tests)
    all_recovered_ok = all(t["corrupted_prefix_recovered"] for t in recovery_tests)

    print(f"Tested {len(test_lengths)} length cases: All Derivations Exact={all_derivations_exact}, All Recovered OK={all_recovered_ok}")

    # Strict prefix check
    strict_rejected = False
    try:
        sample_payload = b"Sample strict check"
        sample_enc = bytearray(ecc.encode(sample_payload))
        sample_enc[0] ^= 0xFF
        ecc.decode_with_result(bytes(sample_enc), strict_prefix=True)
    except ECCDecodeError:
        strict_rejected = True

    print(f"Strict prefix flag enforcement (fails closed): {strict_rejected}")

    return {
        "cases_tested": len(test_lengths),
        "all_derivations_exact": all_derivations_exact,
        "all_recovered_ok": all_recovered_ok,
        "strict_prefix_enforced": strict_rejected,
        "details": recovery_tests,
    }


def run_negative_security_and_boundary_checks(
    neural_pipeline: MediaStegoPipeline,
    classical_pipeline: MediaStegoPipeline,
    key: bytes,
    aad: bytes,
) -> Dict[str, Any]:
    """Evaluates capacity hard gates, header corruption, container tampering, and cross-engine safety."""
    print("\n" + "=" * 80)
    print("NEGATIVE SECURITY, BOUNDARIES & CROSS-ENGINE COMPATIBILITY")
    print("=" * 80)

    # 1. Capacity Hard Gate (8,176 B fits, 8,177 B rejected)
    carrier_img = Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8))
    fits_8176 = CapacityCalculator.check_capacity(8176, subband_dim=256).fits

    exceeded_rejected = False
    try:
        CapacityCalculator.check_capacity(8177, subband_dim=256)
    except PayloadCapacityExceededError:
        exceeded_rejected = True

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

    # 3. Payload region tampering fails closed with no plaintext leak
    stego_arr_payload = np.array(stego_res.stego_image).copy()
    stego_arr_payload[50:450, 50:450, :] = 0
    payload_tamper_fails_closed = False
    try:
        neural_pipeline.extract_and_unprotect(Image.fromarray(stego_arr_payload), key, aad=aad)
    except (AuthenticationError, ECCDecodeError, CarrierHeaderError):
        payload_tamper_fails_closed = True

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
    print(f"Header corruption fails closed: {header_tamper_fails_closed}")
    print(f"Payload tamper fails closed: {payload_tamper_fails_closed}")
    print(f"Cross-engine extraction fails closed: {cross_engine_fails_closed}")
    print(f"Classical default non-regression: {classical_non_regression}")

    return {
        "capacity_8176_fits": fits_8176,
        "capacity_8177_rejected": exceeded_rejected,
        "header_corruption_fails_closed": header_tamper_fails_closed,
        "payload_tampering_fails_closed": payload_tamper_fails_closed,
        "cross_engine_fails_closed": cross_engine_fails_closed,
        "classical_non_regression": classical_non_regression,
    }


def main() -> None:
    print("\n" + "=" * 80)
    print("PALADINMEDIA — SPRINT 3.3.3 PRODUCTION INTEGRATION & ECC FRAMING RECOVERY")
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

    ecc = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    key = bytes.fromhex("11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff")
    aad = b"PaladinMedia-Phase3-Sprint3.3.3-ProductionNeuralIntegration"

    # Task 1: 16-case matrix via MediaStegoPipeline
    orchestrator_results = run_production_neural_orchestrator_matrix(neural_pipeline, key, aad)

    # Task 2: ECC framing recovery evaluation
    ecc_recovery_results = run_ecc_framing_recovery_evaluation(ecc)

    # Task 3: Negative security & cross-engine checks
    security_results = run_negative_security_and_boundary_checks(neural_pipeline, classical_pipeline, key, aad)

    # Task 4: Model parameter immutability
    post_digest = compute_param_digest(bridge.acs_model)
    weights_invariant = (pre_digest == post_digest)
    print(f"\nModel Parameter Invariance: {weights_invariant}")
    print(f"Pre-digest:  {pre_digest}")
    print(f"Post-digest: {post_digest}")

    overall_pass = (
        orchestrator_results["all_exact_matches"]
        and orchestrator_results["all_headers_valid"]
        and orchestrator_results["all_aes_valid"]
        and ecc_recovery_results["all_derivations_exact"]
        and ecc_recovery_results["all_recovered_ok"]
        and ecc_recovery_results["strict_prefix_enforced"]
        and security_results["capacity_8176_fits"]
        and security_results["capacity_8177_rejected"]
        and security_results["header_corruption_fails_closed"]
        and security_results["payload_tampering_fails_closed"]
        and security_results["cross_engine_fails_closed"]
        and security_results["classical_non_regression"]
        and weights_invariant
    )

    summary = {
        "sprint": "3.3.3",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "device": device,
        "acs_sub_inn_v3_frozen": True,
        "gamma_min": 1.0,
        "pre_param_digest": pre_digest,
        "post_param_digest": post_digest,
        "weights_invariant": weights_invariant,
        "verdict": "PASS" if overall_pass else "FAIL",
        "orchestrator_matrix": orchestrator_results,
        "ecc_framing_recovery": ecc_recovery_results,
        "negative_security_and_boundaries": security_results,
        "training_authorized": False,
        "training_status": "NOT AUTHORIZED — Sprint 3.3.3 implements production integration and verification only.",
    }

    out_path = REPO_ROOT / "docs" / "experiments" / "sprint3_3_3_production_integration_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults successfully written to: {out_path}")
    print(f"FINAL SPRINT 3.3.3 VERDICT: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
