"""Qualification script evaluating neural_spatial_onnx vs acs_sub_inn_onnx across test carriers."""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization import SecretPayload, MediaType
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder
from src.neural.dataset import DIV2KStegoDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_JSON = REPO_ROOT / "docs" / "experiments" / "ACS_SUB_INN_QUALIFICATION_RESULTS.json"

def main():
    ds = DIV2KStegoDataset(split="test", random_crop=False)
    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder(parity_symbols=48)
    key = b"01234567890123456789012345678901"

    payload_data = b"PaladinMedia Post-Quantum ACS-Sub-INN Qualification Payload " * 16
    payload = SecretPayload(data=payload_data, media_type=MediaType.TEXT, metadata={"source": "qualification"})

    # Evaluate on first 25 held-out test carriers
    num_eval = 25
    results = {
        "timestamp": time.time(),
        "evaluated_carriers_count": num_eval,
        "payload_bytes": len(payload_data),
        "ecc_profile": "rs_2t_48",
        "backends": {
            "neural_spatial_onnx": {
                "exact_recovery_count": 0,
                "total_carriers": num_eval,
                "recovery_rate": 0.0,
                "avg_psnr_db": 0.0,
                "avg_ssim": 0.0,
                "avg_embed_latency_ms": 0.0,
                "avg_extract_latency_ms": 0.0,
                "carrier_metrics": [],
            },
            "acs_sub_inn_onnx": {
                "exact_recovery_count": 0,
                "total_carriers": num_eval,
                "recovery_rate": 0.0,
                "avg_psnr_db": 0.0,
                "avg_ssim": 0.0,
                "avg_embed_latency_ms": 0.0,
                "avg_extract_latency_ms": 0.0,
                "carrier_metrics": [],
            },
        },
    }

    pipe_spatial = MediaStegoPipeline(crypto_provider=crypto, ecc_encoder=ecc, backend="neural_spatial_onnx")
    pipe_subband = MediaStegoPipeline(crypto_provider=crypto, ecc_encoder=ecc, backend="acs_sub_inn_onnx")

    pipelines = {
        "neural_spatial_onnx": pipe_spatial,
        "acs_sub_inn_onnx": pipe_subband,
    }

    for bname, pipe in pipelines.items():
        print(f"Evaluating {bname} on {num_eval} carriers...")
        psnr_list, ssim_list, emb_lat_list, ext_lat_list = [], [], [], []
        recovery_cnt = 0

        for i in range(num_eval):
            item = ds[i]
            img_id = item["image_id"]
            carrier_np = item["carrier"].permute(1, 2, 0).numpy().astype(np.uint8)

            t0 = time.perf_counter()
            stego_res = pipe.protect_and_embed(payload, carrier_np, key, session_counter=i + 1)
            t_emb = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            try:
                rec_res = pipe.extract_and_unprotect(stego_res.stego_png_bytes, key, reference_payload=payload)
                t_ext = (time.perf_counter() - t1) * 1000.0
                is_match = bool(rec_res.exact_match and rec_res.recovered_payload.data == payload_data)
                err_str = None
            except Exception as e:
                t_ext = (time.perf_counter() - t1) * 1000.0
                is_match = False
                err_str = str(e)

            if is_match:
                recovery_cnt += 1

            psnr_list.append(stego_res.psnr_db)
            ssim_list.append(stego_res.ssim)
            emb_lat_list.append(t_emb)
            ext_lat_list.append(t_ext)

            results["backends"][bname]["carrier_metrics"].append({
                "image_id": img_id,
                "psnr_db": round(stego_res.psnr_db, 2),
                "ssim": round(stego_res.ssim, 5),
                "exact_match": is_match,
                "error": err_str,
                "embed_latency_ms": round(t_emb, 2),
                "extract_latency_ms": round(t_ext, 2),
            })

        results["backends"][bname]["exact_recovery_count"] = recovery_cnt
        results["backends"][bname]["recovery_rate"] = recovery_cnt / num_eval
        results["backends"][bname]["avg_psnr_db"] = round(float(np.mean(psnr_list)), 2)
        results["backends"][bname]["avg_ssim"] = round(float(np.mean(ssim_list)), 5)
        results["backends"][bname]["avg_embed_latency_ms"] = round(float(np.mean(emb_lat_list)), 2)
        results["backends"][bname]["avg_extract_latency_ms"] = round(float(np.mean(ext_lat_list)), 2)
        print(f"  {bname}: Recovery {recovery_cnt}/{num_eval} ({results['backends'][bname]['recovery_rate']*100:.1f}%), Avg PSNR={results['backends'][bname]['avg_psnr_db']} dB, Avg SSIM={results['backends'][bname]['avg_ssim']}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote qualification results to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
