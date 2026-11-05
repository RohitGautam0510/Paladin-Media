"""Green computing telemetry benchmark across PaladinMedia steganography engines."""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from src.api.telemetry import telemetry_tracker, SustainabilityTelemetryTracker
from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization import SecretPayload, MediaType
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder
from src.neural.dataset import DIV2KStegoDataset

OUTPUT_JSON = REPO_ROOT / "docs" / "experiments" / "GREEN_COMPUTING_RESULTS.json"

def main():
    print("Running Green Computing Telemetry Benchmark...")
    ds = DIV2KStegoDataset(split="test", random_crop=False)
    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder(parity_symbols=48)
    key = b"01234567890123456789012345678901"
    payload = SecretPayload(data=b"Green Computing Bench" * 20, media_type=MediaType.TEXT)

    engines = [
        "parity_classical",
        "neural_spatial_onnx",
        "acs_sub_inn_onnx",
    ]

    benchmark_results = {
        "timestamp": time.time(),
        "telemetry_source": "codecarbon" if SustainabilityTelemetryTracker.is_codecarbon_available() else "estimated_hardware_model",
        "is_estimated": not SustainabilityTelemetryTracker.is_codecarbon_available(),
        "power_model_assumptions": {
            "cpu_tdp_watts": SustainabilityTelemetryTracker.DEFAULT_CPU_TDP_WATTS,
            "cpu_idle_watts": SustainabilityTelemetryTracker.DEFAULT_CPU_IDLE_WATTS,
            "ram_w_per_gb": SustainabilityTelemetryTracker.DEFAULT_RAM_W_PER_GB,
            "grid_carbon_intensity_g_per_kwh": SustainabilityTelemetryTracker.GRID_CARBON_INTENSITY_G_PER_KWH,
        },
        "engine_benchmarks": {},
    }

    carriers = [ds[i]["carrier"].permute(1, 2, 0).numpy().astype(np.uint8) for i in range(5)]

    for eng in engines:
        print(f"Profiling {eng} across {len(carriers)} runs...")
        pipe = MediaStegoPipeline(crypto_provider=crypto, ecc_encoder=ecc, backend=eng)
        runs_metrics = []

        for i, carrier in enumerate(carriers):
            run_id = f"bench_{eng}_{i}"
            telemetry_tracker.start_tracking(run_id)

            stego_res = pipe.protect_and_embed(payload, carrier, key, session_counter=i+1)
            rec_res = pipe.extract_and_unprotect(stego_res.stego_png_bytes, key, reference_payload=payload)

            m = telemetry_tracker.stop_tracking(run_id)
            runs_metrics.append(m.to_dict())

        avg_lat = float(np.mean([m["execution_time_ms"] for m in runs_metrics]))
        avg_cpu = float(np.mean([m["cpu_utilization_percent"] for m in runs_metrics]))
        avg_ram = float(np.mean([m["memory_rss_mb"] for m in runs_metrics]))
        avg_power = float(np.mean([m["estimated_power_watts"] for m in runs_metrics]))
        avg_energy = float(np.mean([m["energy_kwh"] for m in runs_metrics]))
        avg_co2 = float(np.mean([m["co2_emissions_grams"] for m in runs_metrics]))

        benchmark_results["engine_benchmarks"][eng] = {
            "runs_count": len(carriers),
            "avg_latency_ms": round(avg_lat, 2),
            "avg_cpu_percent": round(avg_cpu, 1),
            "avg_ram_mb": round(avg_ram, 1),
            "avg_power_watts": round(avg_power, 2),
            "avg_energy_kwh": float(f"{avg_energy:.8f}"),
            "avg_co2_grams": float(f"{avg_co2:.6f}"),
            "detailed_runs": runs_metrics,
        }
        print(f"  {eng}: Latency={avg_lat:.1f}ms, Energy={avg_energy:.8f} kWh, CO2={avg_co2:.6f} g")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Wrote green computing results to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
