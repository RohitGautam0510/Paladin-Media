"""Experiment E10: Compression Before Encryption Evaluation.

Evaluates lossless compression (zlib Level 9) applied before AES-256-GCM encryption:
1. Secret A: tests/fixtures/secret_small.png (24x24 RGB PNG, 152 bytes)
2. Secret B: Synthetic icon / logo (64x64 RGB flat graphic, 12,288 bytes raw uncompressed)
3. Secret C: Structured JSON / text payload (10,000 bytes)
4. Secret D: Incompressible random byte fixture (1,000 bytes high entropy)

Records:
- Original secret size
- Compressed secret size
- Compression ratio
- Serialized size
- AES-GCM output size
- Reed-Solomon output size
- Carrier capacity fit
- Processing time
- Exact recovered-byte equality
"""

import io
import json
import os
import sys
import time
import zlib
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.carrier.capacity import CapacityCalculator
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


def evaluate_compression_pipeline(secret_data: bytes, media_type: MediaType, metadata: dict, cover_arr: np.ndarray, key: bytes, aad: bytes):
    orig_size = len(secret_data)

    # --- Branch 1: Baseline Without Compression ---
    t0 = time.perf_counter()
    p_uncompressed = SecretPayload(data=secret_data, media_type=media_type, metadata=metadata)
    ser_uncomp = SecretSerializer.serialize(p_uncompressed)
    crypto = AESGCMProvider()
    enc_uncomp = crypto.encrypt(ser_uncomp, key, aad=aad)
    ecc = ReedSolomonEncoder()
    prot_uncomp = ecc.encode(enc_uncomp)
    t_uncomp = time.perf_counter() - t0

    fits_uncomp = len(prot_uncomp) <= CapacityCalculator.MAX_PROTECTED_BYTES

    # --- Branch 2: With Lossless Compression Before Encryption ---
    t0 = time.perf_counter()
    compressed_data = zlib.compress(secret_data, level=9)
    comp_size = len(compressed_data)
    comp_ratio = orig_size / comp_size if comp_size > 0 else 0.0

    comp_metadata = dict(metadata)
    comp_metadata["compressed"] = "zlib_9"
    comp_metadata["orig_len"] = orig_size

    p_compressed = SecretPayload(data=compressed_data, media_type=media_type, metadata=comp_metadata)
    ser_comp = SecretSerializer.serialize(p_compressed)
    enc_comp = crypto.encrypt(ser_comp, key, aad=aad)
    prot_comp = ecc.encode(enc_comp)
    t_comp_protect = time.perf_counter() - t0

    fits_comp = len(prot_comp) <= CapacityCalculator.MAX_PROTECTED_BYTES

    # End-to-end roundtrip if compressed fits
    exact_recovery = False
    t_roundtrip = 0.0
    if fits_comp:
        t0 = time.perf_counter()
        embed_result = BaselineEmbedder.embed(cover_arr, prot_comp, session_counter=101)

        # PNG save / reload
        png_buf = io.BytesIO()
        embed_result.stego_image.save(png_buf, format="PNG")
        reloaded = Image.open(io.BytesIO(png_buf.getvalue())).convert("RGB")

        # Extraction
        header, ex_prot = Extractor.extract(reloaded)
        dec_enc = ecc.decode(ex_prot)
        dec_ser = crypto.decrypt(dec_enc, key, aad=aad)
        recovered_comp = SecretSerializer.deserialize(dec_ser)

        # Decompress
        decompressed_data = zlib.decompress(recovered_comp.data)
        exact_recovery = (decompressed_data == secret_data)
        t_roundtrip = time.perf_counter() - t0

    return {
        "original_bytes": orig_size,
        "uncompressed_branch": {
            "serialized_bytes": len(ser_uncomp),
            "encrypted_bytes": len(enc_uncomp),
            "protected_bytes": len(prot_uncomp),
            "fits_carrier": fits_uncomp,
            "processing_time_ms": round(t_uncomp * 1000, 2),
        },
        "compressed_branch": {
            "compressed_bytes": comp_size,
            "compression_ratio": round(comp_ratio, 3),
            "serialized_bytes": len(ser_comp),
            "encrypted_bytes": len(enc_comp),
            "protected_bytes": len(prot_comp),
            "fits_carrier": fits_comp,
            "capacity_saved_bytes": len(prot_uncomp) - len(prot_comp),
            "protect_time_ms": round(t_comp_protect * 1000, 2),
            "roundtrip_time_ms": round(t_roundtrip * 1000, 2),
            "exact_recovery": exact_recovery,
        }
    }


def run_e10():
    cover_path = os.path.join("MVP 0 Dataset", "0001x2.png")
    with Image.open(cover_path) as img:
        cover_arr = np.array(img.convert("RGB"), dtype=np.uint8)

    key = b"01234567890123456789012345678901"
    aad = b"PALADIN_E10_COMPRESSION"

    # Fixture A: Existing small secret image
    with open(os.path.join("tests", "fixtures", "secret_small.png"), "rb") as f:
        secret_a = f.read()

    # Fixture B: Synthetic graphic logo (64x64 flat color blocks, uncompressed raw RGB)
    logo_arr = np.zeros((64, 64, 3), dtype=np.uint8)
    logo_arr[:32, :32, :] = [255, 0, 0]
    logo_arr[32:, :32, :] = [0, 255, 0]
    logo_arr[:32, 32:, :] = [0, 0, 255]
    logo_arr[32:, 32:, :] = [255, 255, 0]
    secret_b = logo_arr.tobytes()  # 64 * 64 * 3 = 12,288 bytes raw

    # Fixture C: Structured telemetry / JSON (10,000 bytes)
    json_doc = {"records": [{"id": i, "status": "nominal", "reading": 42.0 + i} for i in range(250)]}
    secret_c = json.dumps(json_doc).encode("utf-8")  # ~10 KB

    # Fixture D: Incompressible random bytes (1,000 bytes)
    rng = np.random.RandomState(42)
    secret_d = bytes(rng.randint(0, 256, size=1000, dtype=np.uint8))

    fixtures = [
        {"name": "Fixture A (Small PNG Fixture)", "data": secret_a, "type": MediaType.IMAGE, "meta": {"desc": "secret_small.png"}},
        {"name": "Fixture B (Synthetic 64x64 Raw Image)", "data": secret_b, "type": MediaType.IMAGE, "meta": {"desc": "logo_raw_64x64"}},
        {"name": "Fixture C (Structured JSON Telemetry)", "data": secret_c, "type": MediaType.JSON, "meta": {"desc": "telemetry_json"}},
        {"name": "Fixture D (Random High-Entropy Bytes)", "data": secret_d, "type": MediaType.RAW_BYTES, "meta": {"desc": "random_bytes"}},
    ]

    all_results = {}
    for fix in fixtures:
        print(f"Evaluating {fix['name']} ({len(fix['data'])} bytes)...")
        res = evaluate_compression_pipeline(fix["data"], fix["type"], fix["meta"], cover_arr, key, aad)
        all_results[fix["name"]] = res

    out_file = os.path.join("docs", "experiments", "e10_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== E10 COMPRESSION BEFORE ENCRYPTION RESULTS ===")
    for name, r in all_results.items():
        cb = r["compressed_branch"]
        ub = r["uncompressed_branch"]
        print(f"[{name}] Orig: {r['original_bytes']} B -> Comp: {cb['compressed_bytes']} B (Ratio: {cb['compression_ratio']}x)")
        print(f"   Uncompressed: Prot = {ub['protected_bytes']} B (Fits: {ub['fits_carrier']})")
        print(f"   Compressed:   Prot = {cb['protected_bytes']} B (Fits: {cb['fits_carrier']}, Recovered: {cb['exact_recovery']})")

    print(f"Saved E10 results to {out_file}")
    return all_results


if __name__ == "__main__":
    run_e10()
