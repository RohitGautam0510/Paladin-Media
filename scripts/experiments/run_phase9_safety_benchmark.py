"""PaladinMedia - Phase 9 Public Demo Safety & Reliability Benchmark.

Empirically assesses:
- Carrier dimension scaling memory profile (512x512, 768x768, 1024x1024, 2048x2048) in fresh subprocesses.
- Real FastAPI workflow (upload -> /api/embed -> /api/extract and exact match).
- Concurrency scaling (1, 2, 3 concurrent requests on live uvicorn server).
- Upload and input safety matrix (Task B).
- Security, rate limiting, and CORS invariants (Task C).
- Sustainability telemetry validation (Task D).
"""

import asyncio
import base64
import concurrent.futures
import io
import json
import os
import signal
import struct
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_JSON = REPO_ROOT / "docs" / "experiments" / "phase9_safety_benchmark_results.json"


def generate_png_carrier(width: int, height: int) -> bytes:
    """Generates a deterministic RGB PNG carrier."""
    rng = np.random.RandomState(42)
    x = np.linspace(40, 215, width, dtype=np.uint8)
    y = np.linspace(40, 215, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    arr = np.stack([xx, yy, (xx // 2 + yy // 2).astype(np.uint8)], axis=-1)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =============================================================================
# Subprocess Worker for Single Dimension & Single Engine Memory Benchmark
# =============================================================================

WORKER_CODE = """
import io
import os
import sys
import time
import json
import psutil
from PIL import Image
import numpy as np

os.environ["PALADINMEDIA_MODE"] = "local"
os.environ["PALADINMEDIA_SECRET_KEY"] = ""

proc = psutil.Process()
baseline_rss = proc.memory_info().rss / (1024 * 1024)

# 1. App Init
t0 = time.perf_counter()
from src.api.server import create_app, session_store
from fastapi.testclient import TestClient
app = create_app()
client = TestClient(app)
app_init_rss = proc.memory_info().rss / (1024 * 1024)

# 2. Model Loaded
from src.carrier.modulation import create_modulation_engine
model_name = sys.argv[1]
dim = int(sys.argv[2])
engine = create_modulation_engine(model_name)
model_loaded_rss = proc.memory_info().rss / (1024 * 1024)

# 3. Create Carrier & Secret
rng = np.random.RandomState(42)
x = np.linspace(40, 215, dim, dtype=np.uint8)
y = np.linspace(40, 215, dim, dtype=np.uint8)
xx, yy = np.meshgrid(x, y)
arr = np.stack([xx, yy, (xx // 2 + yy // 2).astype(np.uint8)], axis=-1)
img = Image.fromarray(arr, mode="RGB")
buf = io.BytesIO()
img.save(buf, format="PNG")
carrier_bytes = buf.getvalue()

secret_bytes = b"PaladinMedia Phase 9 Qualification Secret Payload " * 8

# 4. Upload Peak RSS
r_cov = client.post("/api/upload", files={"file": ("cover.png", carrier_bytes, "image/png")})
r_sec = client.post("/api/upload", files={"file": ("secret.txt", secret_bytes, "application/octet-stream")})
upload_peak_rss = proc.memory_info().rss / (1024 * 1024)

cov_id = r_cov.json()["session_id"]
sec_id = r_sec.json()["session_id"]
key_hex = "01" * 32

# 5. Embed Peak RSS
t_embed_start = time.perf_counter()
r_emb = client.post("/api/embed", json={
    "secret_session_id": sec_id,
    "cover_session_id": cov_id,
    "key_hex": key_hex,
    "session_counter": 1,
    "model_name": model_name,
    "version": 2
})
t_embed = time.perf_counter() - t_embed_start
embed_peak_rss = proc.memory_info().rss / (1024 * 1024)

if r_emb.status_code != 200:
    res = {
        "status": "error",
        "error": r_emb.text,
        "dim": dim,
        "model_name": model_name,
        "baseline_rss": baseline_rss,
        "app_init_rss": app_init_rss,
        "model_loaded_rss": model_loaded_rss,
        "upload_peak_rss": upload_peak_rss,
        "embed_peak_rss": embed_peak_rss,
        "idwt_peak_rss": embed_peak_rss,
        "extract_peak_rss": embed_peak_rss,
        "recovery_peak_rss": embed_peak_rss,
        "max_rss": max(baseline_rss, app_init_rss, model_loaded_rss, upload_peak_rss, embed_peak_rss),
        "exact_match": False,
        "latency_s": t_embed,
    }
    print(json.dumps(res))
    sys.exit(0)

stego_session_id = r_emb.json()["stego_session_id"]

# 6. Extract Peak RSS
t_extract_start = time.perf_counter()
r_ext = client.post("/api/extract", json={
    "stego_session_id": stego_session_id,
    "key_hex": key_hex,
    "model_name": model_name,
    "reference_secret_session_id": sec_id
})
t_extract = time.perf_counter() - t_extract_start
extract_peak_rss = proc.memory_info().rss / (1024 * 1024)

exact_match = False
recovered_size = 0
if r_ext.status_code == 200:
    ext_data = r_ext.json()
    rec_sec_id = ext_data.get("recovered_session_id")
    rec_bytes = session_store.get(rec_sec_id) if rec_sec_id else b""
    recovered_size = len(rec_bytes) if rec_bytes else 0
    if rec_bytes == secret_bytes:
        exact_match = True

recovery_peak_rss = proc.memory_info().rss / (1024 * 1024)
max_rss = max(baseline_rss, app_init_rss, model_loaded_rss, upload_peak_rss, embed_peak_rss, extract_peak_rss, recovery_peak_rss)

res = {
    "status": "ok",
    "dim": dim,
    "model_name": model_name,
    "baseline_rss": round(baseline_rss, 2),
    "app_init_rss": round(app_init_rss, 2),
    "model_loaded_rss": round(model_loaded_rss, 2),
    "upload_peak_rss": round(upload_peak_rss, 2),
    "embed_peak_rss": round(embed_peak_rss, 2),
    "idwt_peak_rss": round(embed_peak_rss, 2),
    "extract_peak_rss": round(extract_peak_rss, 2),
    "recovery_peak_rss": round(recovery_peak_rss, 2),
    "max_rss": round(max_rss, 2),
    "embed_latency_s": round(t_embed, 3),
    "extract_latency_s": round(t_extract, 3),
    "total_latency_s": round(t_embed + t_extract, 3),
    "exact_match": exact_match,
    "recovered_size": recovered_size,
    "error": None
}
print(json.dumps(res))
"""


def run_dimension_benchmark(dim: int, model_name: str) -> Dict[str, Any]:
    """Runs isolated memory benchmark for a dimension and model in a fresh subprocess."""
    print(f"Benchmarking {model_name} at {dim}x{dim} in isolated subprocess...")
    proc = subprocess.run(
        [sys.executable, "-c", WORKER_CODE, model_name, str(dim)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "dim": dim,
            "model_name": model_name,
            "status": "crashed",
            "error": proc.stderr.strip() or proc.stdout.strip(),
            "max_rss": 0.0,
            "exact_match": False,
        }
    try:
        lines = [ln.strip() for ln in proc.stdout.strip().split("\n") if ln.strip().startswith("{")]
        if lines:
            return json.loads(lines[-1])
        return {
            "dim": dim,
            "model_name": model_name,
            "status": "parse_error",
            "error": proc.stdout.strip(),
            "max_rss": 0.0,
            "exact_match": False,
        }
    except Exception as e:
        return {
            "dim": dim,
            "model_name": model_name,
            "status": "exception",
            "error": str(e),
            "max_rss": 0.0,
            "exact_match": False,
        }


# =============================================================================
# Concurrency Benchmark on Live FastAPI Operational Server
# =============================================================================

def run_concurrency_benchmark(server_port: int = 8019) -> List[Dict[str, Any]]:
    """Runs live concurrency benchmarks (1, 2, 3 streams) on a background uvicorn server."""
    print("\n--- Running Live Concurrency Benchmark on FastAPI Server ---")
    env = os.environ.copy()
    env["PALADINMEDIA_MODE"] = "demo"
    env["PALADINMEDIA_SECRET_KEY"] = ""

    server_cmd = [
        sys.executable, "-m", "uvicorn", "src.api.server:app",
        "--host", "127.0.0.1",
        "--port", str(server_port),
        "--log-level", "warning",
    ]

    server_proc = subprocess.Popen(server_cmd, cwd=str(REPO_ROOT), env=env)
    server_psutil = psutil.Process(server_proc.pid)

    base_url = f"http://127.0.0.1:{server_port}"
    # Wait for server to become healthy
    ready = False
    for _ in range(50):
        try:
            req = urllib.request.Request(f"{base_url}/api/health")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.1)

    if not ready:
        server_proc.terminate()
        raise RuntimeError("Failed to start background test server within 5 seconds")

    initial_rss = server_psutil.memory_info().rss / (1024 * 1024)
    print(f"Server ready on port {server_port}. Initial RSS: {initial_rss:.1f} MB")

    concurrency_levels = [1, 2, 3]
    results = []

    carrier_bytes = generate_png_carrier(512, 512)
    secret_bytes = b"PaladinMedia Concurrency Qualification Secret " * 8

    def execute_single_stream(stream_idx: int, model_name: str) -> Dict[str, Any]:
        """Uploads and streams media pipeline via HTTP."""
        t_start = time.perf_counter()
        # Upload carrier
        boundary = "----PaladinBoundary" + str(int(time.time() * 1000))
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="carrier.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + carrier_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req_up = urllib.request.Request(
            f"{base_url}/api/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req_up, timeout=10.0) as resp:
            cov_id = json.loads(resp.read().decode("utf-8"))["session_id"]

        # Upload secret
        body_sec = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="secret.txt"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + secret_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req_sec = urllib.request.Request(
            f"{base_url}/api/upload",
            data=body_sec,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req_sec, timeout=10.0) as resp:
            sec_id = json.loads(resp.read().decode("utf-8"))["session_id"]

        # Call stream
        stream_payload = json.dumps({
            "secret_session_id": sec_id,
            "cover_session_id": cov_id,
            "key_hex": "02" * 32,
            "model_name": model_name,
            "session_counter": 100 + stream_idx,
            "version": 2
        }).encode("utf-8")

        req_stream = urllib.request.Request(
            f"{base_url}/api/stream",
            data=stream_payload,
            headers={"Content-Type": "application/json"},
        )
        event_count = 0
        exact_match = False
        with urllib.request.urlopen(req_stream, timeout=15.0) as resp:
            status_code = resp.status
            for line in resp:
                decoded = line.decode("utf-8", errors="ignore")
                if decoded.startswith("data:"):
                    event_count += 1
                    try:
                        ev_json = json.loads(decoded[5:].strip())
                        if ev_json.get("stage") == "EXACT_MATCH_VERIFY":
                            exact_match = bool(ev_json.get("metadata", {}).get("exact_match", False))
                    except Exception:
                        pass

        latency = time.perf_counter() - t_start
        return {
            "stream_idx": stream_idx,
            "model_name": model_name,
            "status_code": status_code,
            "event_count": event_count,
            "exact_match": exact_match,
            "latency_s": round(latency, 3),
        }

    for num_concurrent in concurrency_levels:
        peak_rss = initial_rss
        stop_sampling = False

        def memory_sampler():
            nonlocal peak_rss
            while not stop_sampling:
                try:
                    rss = server_psutil.memory_info().rss / (1024 * 1024)
                    if rss > peak_rss:
                        peak_rss = rss
                except Exception:
                    break
                time.sleep(0.02)

        sampler_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        sampler_future = sampler_thread.submit(memory_sampler)

        models = ["neural_spatial_onnx", "acs_sub_inn_onnx", "neural_spatial_onnx"]
        t_batch_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(execute_single_stream, i, models[i % len(models)])
                for i in range(num_concurrent)
            ]
            stream_results = [f.result() for f in futures]

        batch_latency = time.perf_counter() - t_batch_start
        stop_sampling = True
        sampler_thread.shutdown(wait=True)

        all_matches = all(s["exact_match"] for s in stream_results)
        all_200 = all(s["status_code"] == 200 for s in stream_results)
        headroom = 512.0 - peak_rss

        record = {
            "concurrent_requests": num_concurrent,
            "all_success": all_200 and all_matches,
            "all_exact_matches": all_matches,
            "batch_latency_s": round(batch_latency, 3),
            "peak_rss_mb": round(peak_rss, 1),
            "headroom_mb": round(headroom, 1),
            "headroom_pct": round((headroom / 512.0) * 100.0, 1),
            "streams": stream_results,
        }
        results.append(record)
        print(f"  Concurrency {num_concurrent}: Peak RSS={peak_rss:.1f} MB (Headroom={headroom:.1f} MB, {record['headroom_pct']}%), Match={all_matches}, Latency={batch_latency:.2f}s")

    # Clean up server
    server_proc.terminate()
    try:
        server_proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    return results


# =============================================================================
# Task B: Upload & Input Safety Test Matrix
# =============================================================================

def run_input_safety_matrix() -> Dict[str, Any]:
    """Tests input validation across limits, corruptions, and unsupported formats."""
    print("\n--- Running Input Safety & Boundary Enforcement Matrix (Task B) ---")
    os.environ["PALADINMEDIA_MODE"] = "demo"
    from src.api.server import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    matrix_results = {}

    # 1. Max upload size in demo mode (2 MB ceiling)
    oversized_data = b"X" * (2 * 1024 * 1024 + 1024)
    r1 = client.post("/api/upload", files={"file": ("big.dat", oversized_data, "application/octet-stream")})
    matrix_results["oversized_upload_2mb"] = {
        "status_code": r1.status_code,
        "rejected": r1.status_code == 413,
        "detail": r1.json().get("detail"),
    }
    print(f"  Oversized upload (>2MB): Status {r1.status_code} (Expected 413) -> {'PASS' if r1.status_code == 413 else 'FAIL'}")

    # 2. Maximum dimensions in demo mode (512x512 ceiling; 514x514 must be rejected)
    carrier_514 = generate_png_carrier(514, 514)
    r2 = client.post("/api/upload", files={"file": ("cover514.png", carrier_514, "image/png")})
    matrix_results["oversized_dimension_514x514_demo"] = {
        "status_code": r2.status_code,
        "rejected": r2.status_code == 413,
        "detail": r2.json().get("detail"),
    }
    print(f"  Oversized dimension (514x514): Status {r2.status_code} (Expected 413) -> {'PASS' if r2.status_code == 413 else 'FAIL'}")

    # 3. Valid dimensions in demo mode (512x512 must be accepted)
    carrier_512 = generate_png_carrier(512, 512)
    r3 = client.post("/api/upload", files={"file": ("cover512.png", carrier_512, "image/png")})
    matrix_results["valid_carrier_512x512"] = {
        "status_code": r3.status_code,
        "accepted": r3.status_code == 200,
        "session_id": r3.json().get("session_id"),
    }
    print(f"  Valid dimension (512x512): Status {r3.status_code} (Expected 200) -> {'PASS' if r3.status_code == 200 else 'FAIL'}")

    # 4. Odd dimensions (511x512 must be rejected with 422 in carrier validation)
    carrier_odd = generate_png_carrier(511, 512)
    # Upload accepts (upload checks pixel count/dimension), but carrier resolution / capacity check rejects
    r4_cap = client.get("/api/carrier/capacity?width=511&height=512")
    matrix_results["odd_dimensions_capacity"] = {
        "status_code": r4_cap.status_code,
        "rejected": r4_cap.status_code == 422,
        "detail": r4_cap.json().get("detail"),
    }
    print(f"  Odd dimensions capacity query: Status {r4_cap.status_code} (Expected 422) -> {'PASS' if r4_cap.status_code == 422 else 'FAIL'}")

    # 5. Malformed image bytes (random garbage with image/png content type)
    malformed_bytes = b"NOT_A_VALID_PNG_JUST_RANDOM_GARBAGE" * 10
    r5 = client.post("/api/upload", files={"file": ("bad.png", malformed_bytes, "image/png")})
    matrix_results["malformed_png"] = {
        "status_code": r5.status_code,
        "rejected": r5.status_code in (400, 422),
        "detail": r5.json().get("detail"),
    }
    print(f"  Malformed PNG upload: Status {r5.status_code} (Expected 400/422) -> {'PASS' if r5.status_code in (400, 422) else 'FAIL'}")

    # 6. Truncated PNG (only PNG header without data)
    truncated_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    r6 = client.post("/api/upload", files={"file": ("trunc.png", truncated_png, "image/png")})
    matrix_results["truncated_png"] = {
        "status_code": r6.status_code,
        "rejected": r6.status_code in (400, 422),
        "detail": r6.json().get("detail"),
    }
    print(f"  Truncated PNG upload: Status {r6.status_code} (Expected 400/422) -> {'PASS' if r6.status_code in (400, 422) else 'FAIL'}")

    # 7. Unsupported formats (JPEG carrier)
    jpeg_buf = io.BytesIO()
    Image.new("RGB", (256, 256), color=(100, 150, 200)).save(jpeg_buf, format="JPEG")
    r7_up = client.post("/api/upload", files={"file": ("cover.jpg", jpeg_buf.getvalue(), "image/jpeg")})
    jpeg_sess_id = r7_up.json().get("session_id")
    # Embedding with JPEG carrier must fail closed with 422
    sec_up = client.post("/api/upload", files={"file": ("sec.txt", b"secret payload", "application/octet-stream")})
    sec_id = sec_up.json()["session_id"]
    r7_emb = client.post("/api/embed", json={
        "secret_session_id": sec_id,
        "cover_session_id": jpeg_sess_id,
        "key_hex": "01" * 32,
    })
    matrix_results["unsupported_jpeg_carrier"] = {
        "status_code": r7_emb.status_code,
        "rejected": r7_emb.status_code == 422,
        "detail": r7_emb.json().get("detail"),
    }
    print(f"  Unsupported JPEG carrier embed: Status {r7_emb.status_code} (Expected 422) -> {'PASS' if r7_emb.status_code == 422 else 'FAIL'}")

    # 8. Capacity gate rejection before neural inference
    # A 512x512 carrier has usable capacity of ~3,900 bytes. Uploading 10,000 bytes secret payload must fail closed with 413
    huge_secret = b"A" * 10000
    r8_sec = client.post("/api/upload", files={"file": ("huge_sec.txt", huge_secret, "application/octet-stream")})
    huge_sec_id = r8_sec.json()["session_id"]
    r8_emb = client.post("/api/embed", json={
        "secret_session_id": huge_sec_id,
        "cover_session_id": r3.json()["session_id"],
        "key_hex": "01" * 32,
    })
    matrix_results["payload_capacity_gate"] = {
        "status_code": r8_emb.status_code,
        "rejected": r8_emb.status_code in (413, 422),
        "detail": r8_emb.json().get("detail"),
    }
    print(f"  Payload capacity gate (> usable capacity): Status {r8_emb.status_code} (Expected 413/422) -> {'PASS' if r8_emb.status_code in (413, 422) else 'FAIL'}")

    return matrix_results


# =============================================================================
# Task C & D: Security, Rate Limiting, and Telemetry Validation
# =============================================================================

def run_security_and_telemetry_validation() -> Dict[str, Any]:
    """Validates rate limiting, CORS configuration, and green computing telemetry."""
    print("\n--- Running Security, Rate Limiting & Telemetry Validation (Tasks C & D) ---")
    os.environ["PALADINMEDIA_MODE"] = "demo"
    from src.api.server import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    results = {}

    # 1. Telemetry inspection (call BEFORE rate limiter test exhausts quota)
    r_caps = client.get("/api/capabilities")
    caps = r_caps.json()
    telem_caps = caps.get("telemetry", {})
    telem_valid = (
        telem_caps.get("available") is True
        and telem_caps.get("is_estimated") is True
        and telem_caps.get("source") == "estimated_hardware_model"
    )
    results["telemetry"] = {
        "available": telem_caps.get("available"),
        "is_estimated": telem_caps.get("is_estimated"),
        "source": telem_caps.get("source"),
        "metrics_tracked": telem_caps.get("metrics_tracked"),
        "valid": telem_valid,
    }
    print(f"  Sustainability Telemetry: is_estimated={telem_caps.get('is_estimated')}, source={telem_caps.get('source')} -> {'PASS' if telem_valid else 'FAIL'}")

    # 2. CORS headers
    # Allowed origin: https://paladinmedia.vercel.app
    r_cors_good = client.options(
        "/api/stream",
        headers={
            "Origin": "https://paladinmedia.vercel.app",
            "Access-Control-Request-Method": "POST",
        },
    )
    cors_allowed = r_cors_good.headers.get("access-control-allow-origin") == "https://paladinmedia.vercel.app"

    # Malicious origin: https://evil-site.com
    r_cors_bad = client.options(
        "/api/stream",
        headers={
            "Origin": "https://evil-site.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    cors_blocked = r_cors_bad.headers.get("access-control-allow-origin") != "https://evil-site.com"

    results["cors"] = {
        "allowed_origin_matched": cors_allowed,
        "unauthorized_origin_blocked": cors_blocked,
    }
    print(f"  CORS Allowed Origin: {'PASS' if cors_allowed else 'FAIL'} | Unauthorized Origin Blocked: {'PASS' if cors_blocked else 'FAIL'}")

    # 3. Rate limiter test (60 req/min per IP)
    status_codes = []
    for _ in range(65):
        resp = client.get("/api/config")
        status_codes.append(resp.status_code)

    has_429 = 429 in status_codes
    count_200 = status_codes.count(200)
    count_429 = status_codes.count(429)
    results["rate_limiting"] = {
        "requests_sent": 65,
        "count_200": count_200,
        "count_429": count_429,
        "enforced": has_429,
    }
    print(f"  Rate limiting (65 reqs): {count_200} x 200 OK, {count_429} x 429 Too Many Requests -> {'PASS' if has_429 else 'FAIL'}")

    return results


def main():
    print("=================================================================")
    print("PaladinMedia Phase 9 Public Demo Safety & Reliability Benchmark")
    print("=================================================================")

    # 1. Dimension Benchmarks in isolated subprocesses
    dims = [512, 768, 1024, 2048]
    models = ["neural_spatial_onnx", "acs_sub_inn_onnx"]
    dim_results = []
    for dim in dims:
        for model in models:
            res = run_dimension_benchmark(dim, model)
            dim_results.append(res)
            print(f"  Result: {dim}x{dim} {model} -> Max RSS: {res.get('max_rss')} MB, Match: {res.get('exact_match')}, Latency: {res.get('total_latency_s')}s")

    # 2. Concurrency Benchmark on live server
    conc_results = run_concurrency_benchmark(server_port=8021)

    # 3. Task B: Input Safety Matrix
    safety_matrix = run_input_safety_matrix()

    # 4. Task C & D: Security, Rate Limiting & Telemetry
    sec_results = run_security_and_telemetry_validation()

    final_payload = {
        "timestamp": time.time(),
        "dimension_benchmarks": dim_results,
        "concurrency_benchmarks": conc_results,
        "safety_matrix": safety_matrix,
        "security_and_telemetry": sec_results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[DONE] Saved Phase 9 qualification results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
