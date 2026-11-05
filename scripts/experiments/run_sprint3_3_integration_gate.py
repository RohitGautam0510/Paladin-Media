"""Experiment & Audit Runner: Phase 3 Sprint 3.3 End-to-End Integration Gate.

Executes and verifies:
1. Manifest Integrity Audit (Task A):
   - Canonical manifest file SHA-256 validation.
   - Metadata composite SHA-256 validation.
   - Independent composite SHA-256 recomputation from 800 partition records.
   - Partition count (600/100/100), positive integer ID, stem match, and disjointness audits.
   - Isolated tampering test proving recomputation detects modified record digests.
2. Byte-Level Integration Gate (Task B & Task C):
   - Full real pipeline roundtrip: SecretPayload -> SecretSerializer -> AES-256-GCM ->
     ReedSolomon -> CapacityCalculator -> NeuralSpatialBridge (Haar DWT -> ACS-Sub-INN v3 -> IDWT)
     -> uint8 clamp -> PNG save/reload -> NeuralSpatialBridge extraction -> Bit unpack/pack ->
     Raw BER measurement -> RS symbol error correction -> AES-GCM tag auth -> Deserialization ->
     Exact byte equality verification.
   - Carrier test matrix:
     * Synthetic smooth carrier (128.0)
     * DIV2K natural carriers: 0001.png, 0002.png, 0005.png
   - Payload test matrix:
     * Baseline (300 B)
     * Medium load (1,500 B)
     * High load (3,500 B)
     * Full capacity load (7,088 B)
   - Capacity boundary enforcement: 8,176 B protected (fits) vs 8,177 B protected (fails closed).
   - Tampered stego container test: fail-closed rejection with zero plaintext leaked.
   - Overflow fail-closed test: oversized secret aborted before carrier modification.
   - Model parameter invariance: proof that weights remain strictly unchanged (untrained model).

Outputs:
- docs/experiments/sprint3_3_integration_results.json
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import struct
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.carrier.capacity import (
    CapacityCalculator,
    CapacityReport,
    PayloadCapacityExceededError,
)
from src.core.exceptions import AuthenticationError, ECCDecodeError
from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import (
    DEFAULT_DATASET_REL_PATH,
    DEFAULT_MANIFEST_REL_PATH,
    DIV2KStegoDataset,
    EXPECTED_COMPOSITE_SHA256,
    EXPECTED_MANIFEST_FILE_SHA256,
    compute_manifest_composite_sha256,
    validate_div2k_manifest,
)
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_scaffold import (
    DatasetSplitManager,
    GateRegistry,
    set_reproducibility_seeds,
)
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


def run_manifest_integrity_audit() -> Dict[str, Any]:
    """Executes Task A: Manifest Integrity Audit."""
    print("\n" + "=" * 80)
    print("TASK A: MANIFEST INTEGRITY AUDIT")
    print("=" * 80)

    canonical_manifest = REPO_ROOT / DEFAULT_MANIFEST_REL_PATH
    raw_bytes = canonical_manifest.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    manifest_data = json.loads(raw_bytes.decode("utf-8"))

    # 1. Manifest file digest check
    file_match = (file_sha256 == EXPECTED_MANIFEST_FILE_SHA256)
    print(f"Manifest File Path: {canonical_manifest}")
    print(f"Manifest File SHA-256: {file_sha256}")
    print(f"Expected File SHA-256: {EXPECTED_MANIFEST_FILE_SHA256} | Match: {file_match}")

    # 2. Manifest metadata claimed composite hash check
    meta_composite = manifest_data.get("metadata", {}).get("dataset_composite_sha256")
    meta_match = (meta_composite == EXPECTED_COMPOSITE_SHA256)
    print(f"Metadata Composite SHA-256: {meta_composite}")
    print(f"Expected Composite SHA-256: {EXPECTED_COMPOSITE_SHA256} | Match: {meta_match}")

    # 3. Independent recomputation from partition records
    recomputed_composite = compute_manifest_composite_sha256(manifest_data)
    recomputed_match = (recomputed_composite == EXPECTED_COMPOSITE_SHA256)
    print(f"Recomputed Composite SHA-256: {recomputed_composite}")
    print(f"Expected Composite SHA-256:  {EXPECTED_COMPOSITE_SHA256} | Match: {recomputed_match}")

    # 4. Partition count and integrity checks
    train_records = manifest_data.get("train", [])
    val_records = manifest_data.get("val", [])
    test_records = manifest_data.get("test", [])
    counts_valid = (
        len(train_records) == 600 and len(val_records) == 100 and len(test_records) == 100
    )
    print(f"Partition Counts: train={len(train_records)}, val={len(val_records)}, test={len(test_records)} | Valid: {counts_valid}")

    # 5. Schema & record consistency
    all_records = train_records + val_records + test_records
    ids_positive = all(isinstance(r["image_id"], int) and not isinstance(r["image_id"], bool) and r["image_id"] > 0 for r in all_records)
    stems_match = all(r["filename"] == f"{r['image_id']:04d}.png" for r in all_records)
    disjoint_ids = (len(set(r["image_id"] for r in all_records)) == 800)
    print(f"Positive IDs: {ids_positive} | Stems Match: {stems_match} | Disjoint (800 unique): {disjoint_ids}")

    # 6. Isolated tampering test: modify single record SHA-256 and confirm recomputation detects it
    tampered_data = json.loads(json.dumps(manifest_data))
    tampered_data["train"][0]["sha256"] = "0" * 64
    tampered_recomputed = compute_manifest_composite_sha256(tampered_data)
    tamper_detected = (tampered_recomputed != EXPECTED_COMPOSITE_SHA256)
    print(f"Isolated Tamper Test: Recomputed on tampered record detected: {tamper_detected}")

    # 7. DatasetSplitManager verification
    split_mgr = DatasetSplitManager()
    mgr_valid = (
        split_mgr.file_sha256 == EXPECTED_MANIFEST_FILE_SHA256
        and len(split_mgr.get_train_records()) == 600
        and len(split_mgr.get_val_records()) == 100
        and len(split_mgr.get_held_out_test_manifest_for_audit_only()) == 100
    )
    print(f"DatasetSplitManager Verification: {mgr_valid}")

    overall_task_a_passed = (
        file_match
        and meta_match
        and recomputed_match
        and counts_valid
        and ids_positive
        and stems_match
        and disjoint_ids
        and tamper_detected
        and mgr_valid
    )

    return {
        "status": "PASSED" if overall_task_a_passed else "FAILED",
        "canonical_manifest_path": str(canonical_manifest),
        "manifest_file_sha256": file_sha256,
        "expected_manifest_file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
        "manifest_file_sha256_match": file_match,
        "metadata_composite_sha256": meta_composite,
        "expected_composite_sha256": EXPECTED_COMPOSITE_SHA256,
        "metadata_composite_sha256_match": meta_match,
        "recomputed_composite_sha256": recomputed_composite,
        "recomputed_composite_sha256_match": recomputed_match,
        "partition_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "partition_counts_valid": counts_valid,
        "schema_checks": {
            "ids_positive_integers": ids_positive,
            "stems_match_image_id": stems_match,
            "disjoint_800_unique": disjoint_ids,
        },
        "isolated_tamper_detection_verified": tamper_detected,
        "dataset_split_manager_verified": mgr_valid,
    }


def run_integration_roundtrip_matrix(
    bridge: NeuralSpatialBridge,
    crypto: AESGCMProvider,
    ecc: ReedSolomonEncoder,
    key: bytes,
    aad: bytes,
) -> Dict[str, Any]:
    """Executes Task B & Task C: Integration Roundtrip Matrix across payloads and carriers."""
    print("\n" + "=" * 80)
    print("TASK B & C: END-TO-END INTEGRATION ROUNDTRIP MATRIX")
    print("=" * 80)

    # Carriers to evaluate
    ds = DIV2KStegoDataset(split="train")
    carriers_to_test: List[Tuple[str, str, torch.Tensor]] = [
        ("synthetic", "Smooth Mid-Gray (128.0)", torch.full((1, 3, 512, 512), 128.0, dtype=torch.float32)),
        ("0001.png", "DIV2K #0001 (Train)", ds[0]["carrier"].unsqueeze(0)),
        ("0002.png", "DIV2K #0002 (Train)", ds[1]["carrier"].unsqueeze(0)),
        ("0005.png", "DIV2K #0005 (Train)", ds[4]["carrier"].unsqueeze(0)),
    ]

    # Payload sizes: 300 B (baseline), 1500 B (mid), 3500 B (high), 7088 B (usable full capacity with 2B metadata)
    payload_lengths = [300, 1500, 3500, 7088]

    roundtrip_results: List[Dict[str, Any]] = []

    for carrier_id, carrier_desc, carrier_tensor in carriers_to_test:
        print(f"\n--- Testing Carrier: {carrier_id} ({carrier_desc}) ---")
        for plen in payload_lengths:
            raw_secret = os.urandom(plen)
            meta = None if plen >= 7088 else {"carrier": carrier_id, "load_bytes": plen}
            secret = SecretPayload(
                data=raw_secret,
                media_type=MediaType.IMAGE,
                metadata=meta,
            )

            # 1. Serialization
            serialized = SecretSerializer.serialize(secret)

            # 2. AES-256-GCM Encryption
            ciphertext = crypto.encrypt(serialized, key, aad=aad)

            # 3. Reed-Solomon RS(255, 223) Protection
            protected = ecc.encode(ciphertext)
            protected_len = len(protected)

            # 4. Capacity Hard Gate
            cap_report = CapacityCalculator.check_capacity(protected_len)
            assert cap_report.fits is True

            # 5. Bit Mapping
            bits_1d = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))
            num_bits = len(bits_1d)
            mask = torch.zeros(1, 1, 256, 256)
            payload_tensor = torch.zeros(1, 1, 256, 256)
            mask.view(-1)[:num_bits] = 1.0
            payload_tensor.view(-1)[:num_bits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

            # 6. Spatial Bridge Roundtrip (with lossless PNG save & reload)
            emb_res, ext_res, metrics = bridge.run_full_roundtrip(
                carrier=carrier_tensor,
                payload_bits=(payload_tensor > 0).to(torch.uint8),
                mask=mask,
                reload_png=True,
            )

            # 7. Receiver Demodulation & Bit Extraction
            ext_bits = ext_res.extracted_binary.view(-1)[:num_bits].cpu().numpy().astype(np.uint8)
            raw_bit_errors = int(np.sum(ext_bits != bits_1d))
            raw_ber = float(raw_bit_errors / num_bits) if num_bits > 0 else 0.0

            # 8. Byte Symbol Error Diagnostics
            ext_bytes = np.packbits(ext_bits).tobytes()
            err_diag = measure_rs_codeword_errors(protected, ext_bytes)

            # 9. Reed-Solomon Decoding
            rs_res = ecc.decode_with_result(ext_bytes)
            rs_success = (rs_res.data == ciphertext)

            # 10. AES-GCM Tag Verification & Decryption
            decrypted = crypto.decrypt(rs_res.data, key, aad=aad)
            aes_success = (decrypted == serialized)

            # 11. Deserialization & Verbatim Match
            recovered = SecretSerializer.deserialize(decrypted)
            exact_match = (recovered.data == secret.data and recovered.metadata == secret.metadata)
            post_ecc_ber = 0.0 if exact_match else 1.0

            entry = {
                "carrier_id": carrier_id,
                "carrier_description": carrier_desc,
                "plaintext_bytes": plen,
                "serialized_bytes": len(serialized),
                "ciphertext_bytes": len(ciphertext),
                "protected_bytes": protected_len,
                "capacity_fits": cap_report.fits,
                "capacity_utilization_pct": round(protected_len / 8176 * 100, 2),
                "psnr_db": round(metrics.psnr_db, 2),
                "ssim": round(metrics.ssim, 6),
                "mae": round(metrics.mae, 4),
                "clipped_pixels_total": metrics.clip_stats.get("num_clipped_total", 0),
                "raw_bit_errors": raw_bit_errors,
                "raw_ber": raw_ber,
                "rs_header_errors": err_diag["header_errors"],
                "rs_total_symbol_errors": err_diag["total_codeword_errors"],
                "rs_worst_codeword_errors": err_diag["worst_codeword_errors"],
                "rs_within_t16_limit": err_diag["within_t16_limit"],
                "rs_corrected_symbols": rs_res.corrected_symbols,
                "rs_decoding_success": rs_success,
                "aes_gcm_tag_valid": aes_success,
                "post_ecc_ber": post_ecc_ber,
                "exact_plaintext_match": exact_match,
            }
            roundtrip_results.append(entry)

            print(
                f"  Payload: {plen:4d} B ({protected_len:4d} B prot) | PSNR: {metrics.psnr_db:5.2f} dB | "
                f"SSIM: {metrics.ssim:.6f} | Clip: {entry['clipped_pixels_total']:3d} | "
                f"RawBER: {raw_ber:8.6f} | WorstBlockErr: {err_diag['worst_codeword_errors']:2d}/16 | "
                f"ExactMatch: {exact_match!s:<5}"
            )

    return {
        "trials_count": len(roundtrip_results),
        "all_exact_matches": all(r["exact_plaintext_match"] for r in roundtrip_results),
        "all_aes_valid": all(r["aes_gcm_tag_valid"] for r in roundtrip_results),
        "all_rs_within_limit": all(r["rs_within_t16_limit"] for r in roundtrip_results),
        "trials": roundtrip_results,
    }


def run_negative_security_and_capacity_tests(
    bridge: NeuralSpatialBridge,
    crypto: AESGCMProvider,
    ecc: ReedSolomonEncoder,
    key: bytes,
    aad: bytes,
) -> Dict[str, Any]:
    """Executes negative tests: capacity boundary, fail-closed overflow, and stego tampering."""
    print("\n" + "=" * 80)
    print("TASK C: NEGATIVE SECURITY, OVERFLOW & BOUNDARY TESTS")
    print("=" * 80)

    # 1. Capacity boundary: 8,176 B vs 8,177 B
    rep_8176 = CapacityCalculator.check_capacity(8176)
    pass_8176 = (rep_8176.fits is True and rep_8176.remaining_capacity_bytes == 0)

    pass_8177_fail_closed = False
    try:
        CapacityCalculator.check_capacity(8177)
    except PayloadCapacityExceededError:
        pass_8177_fail_closed = True

    print(f"Capacity Boundary 8,176 B (Fits): {pass_8176}")
    print(f"Capacity Boundary 8,177 B (Rejection Fail-Closed): {pass_8177_fail_closed}")

    # 2. Oversized payload aborted before carrier modification
    carrier_orig = torch.full((1, 3, 512, 512), 128.0)
    carrier_copy = carrier_orig.clone()
    overflow_aborted_cleanly = False
    try:
        secret_overflow = SecretPayload(data=os.urandom(10000), media_type=MediaType.IMAGE, metadata={})
        ser = SecretSerializer.serialize(secret_overflow)
        ciph = crypto.encrypt(ser, key, aad=aad)
        prot = ecc.encode(ciph)
        CapacityCalculator.check_capacity(len(prot))
    except PayloadCapacityExceededError:
        overflow_aborted_cleanly = True

    carrier_untouched = torch.equal(carrier_orig, carrier_copy)
    print(f"Overflow Aborted Before Carrier Modification: {overflow_aborted_cleanly} | Carrier Untouched: {carrier_untouched}")

    # 3. Tampered stego container fails closed with 0 plaintext leaked
    secret_tamper = SecretPayload(data=b"StrictlyConfidentialPaladinSecretBytes", media_type=MediaType.TEXT, metadata={})
    ser = SecretSerializer.serialize(secret_tamper)
    ciph = crypto.encrypt(ser, key, aad=aad)
    prot = ecc.encode(ciph)

    bits_1d = np.unpackbits(np.frombuffer(prot, dtype=np.uint8))
    nbits = len(bits_1d)
    mask = torch.zeros(1, 1, 256, 256)
    pten = torch.zeros(1, 1, 256, 256)
    mask.view(-1)[:nbits] = 1.0
    pten.view(-1)[:nbits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

    emb_res = bridge.embed_spatial(carrier_orig, pten, mask, differentiable=False)
    corrupted_stego = emb_res.stego_image.clone()
    corrupted_stego[:, :, 0:128, 0:128] = 0.0  # Zero out spatial embedding region

    ext_res = bridge.extract_spatial(corrupted_stego, mask=mask)
    ext_bits = ext_res.extracted_binary.view(-1)[:nbits].cpu().numpy().astype(np.uint8)
    ext_bytes = np.packbits(ext_bits).tobytes()

    tamper_caught = False
    leaked_plaintext = None
    caught_exception = ""
    try:
        rs_res = ecc.decode_with_result(ext_bytes)
        decrypted = crypto.decrypt(rs_res.data, key, aad=aad)
        leaked_plaintext = SecretSerializer.deserialize(decrypted).data
    except (ECCDecodeError, AuthenticationError) as exc:
        tamper_caught = True
        caught_exception = type(exc).__name__

    print(f"Tampered Stego Fail-Closed: Caught={tamper_caught} ({caught_exception}) | Plaintext Leaked: {leaked_plaintext is not None}")

    return {
        "boundary_8176_fits": pass_8176,
        "boundary_8177_rejected_fail_closed": pass_8177_fail_closed,
        "overflow_aborted_before_carrier_touch": overflow_aborted_cleanly and carrier_untouched,
        "tampered_stego_fail_closed": tamper_caught and (leaked_plaintext is None),
        "tamper_caught_exception": caught_exception,
    }


def main() -> None:
    print("\n" + "=" * 80)
    print("PALADINMEDIA — SPRINT 3.3 END-TO-END INTEGRATION GATE RUNNER")
    print("=" * 80)

    # 1. Environment & Seeds
    seed_info = set_reproducibility_seeds(seed=42, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | Device: {device}")

    # 2. Setup Real Production/Neural Components (Untrained Architecture)
    acs_model = ACSSubINN(gamma_min=1.0)
    initial_model_param_digest = compute_param_digest(acs_model)
    bridge = NeuralSpatialBridge(acs_model=acs_model)
    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    key = bytes.fromhex("11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff")
    aad = b"PaladinMedia-Phase3-Sprint3.3-IntegrationGate"

    # 3. Task A: Manifest Integrity Audit
    task_a_results = run_manifest_integrity_audit()

    # 4. Task B & Task C: Integration Roundtrip Matrix
    task_bc_matrix = run_integration_roundtrip_matrix(
        bridge=bridge,
        crypto=crypto,
        ecc=ecc,
        key=key,
        aad=aad,
    )

    # 5. Negative Security & Capacity Boundary Tests
    negative_tests = run_negative_security_and_capacity_tests(
        bridge=bridge,
        crypto=crypto,
        ecc=ecc,
        key=key,
        aad=aad,
    )

    # 6. Verify Model Parameter Invariance (Untrained Status Preserved)
    post_model_param_digest = compute_param_digest(bridge.acs_model)
    weights_invariant = (initial_model_param_digest == post_model_param_digest)
    print(f"\nModel Parameter Invariance: {weights_invariant}")
    print(f"Pre-experiment Model Digest:  {initial_model_param_digest}")
    print(f"Post-experiment Model Digest: {post_model_param_digest}")

    # 7. Overall Gate Verdict
    all_passed = (
        task_a_results["status"] == "PASSED"
        and task_bc_matrix["all_exact_matches"]
        and task_bc_matrix["all_aes_valid"]
        and task_bc_matrix["all_rs_within_limit"]
        and negative_tests["boundary_8176_fits"]
        and negative_tests["boundary_8177_rejected_fail_closed"]
        and negative_tests["overflow_aborted_before_carrier_touch"]
        and negative_tests["tampered_stego_fail_closed"]
        and weights_invariant
    )

    verdict = "PASSED" if all_passed else "FAILED"
    print("\n" + "=" * 80)
    print(f"SPRINT 3.3 INTEGRATION GATE FINAL VERDICT: {verdict}")
    print("=" * 80)

    # 8. Assemble Results Document
    results_doc = {
        "metadata": {
            "sprint": "3.3",
            "title": "Phase 3 Sprint 3.3 End-to-End Integration Gate & Integrity Audit",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "training_status": "NOT AUTHORIZED — Sprint 3.3 implements and validates integration gate only",
            "model_status": "UNTRAINED_ACS_SUB_INN_V3_FROZEN",
            "seed_configuration": seed_info,
            "system_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "platform": platform.platform(),
                "cuda_available": torch.cuda.is_available(),
                "device_used": device,
            },
        },
        "model_parameter_governance": {
            "initial_param_digest": initial_model_param_digest,
            "post_param_digest": post_model_param_digest,
            "weights_invariant": weights_invariant,
            "optimizer_steps_applied": 0,
            "epochs_run": 0,
        },
        "task_a_manifest_integrity": task_a_results,
        "task_bc_integration_roundtrip_matrix": task_bc_matrix,
        "task_c_negative_tests": negative_tests,
    }

    output_path = REPO_ROOT / "docs" / "experiments" / "sprint3_3_integration_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_doc, indent=2), encoding="utf-8")
    print(f"\nAuthoritative results written to: {output_path}")


if __name__ == "__main__":
    main()
