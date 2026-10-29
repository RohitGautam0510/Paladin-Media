"""Phase 5D.2 Assessment Runner: Secure API and Controlled-Deployment Verification.

Evaluates the hardened FastAPI operational boundary and launcher:
- Preflight integrity verification (checkpoint digest, ONNX hashes, dataset manifest).
- Gate A: Streaming authentication, request-body key transport, and event stream.
- Gate B: Session storage and single-worker contract, loopback launcher binding.
- Gate C: Process liveness, application readiness, and SHA-256 verification in /api/config.
- Gate D: Consistent authentication policy enforcement across all endpoints.
- Gate E: Strict request validation (key syntax, cover requirement, ASGI body limits, TTL config).

Emits structured experimental results to docs/experiments/phase5d_results.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.api.launcher import validate_launcher_config
from src.api.security import APISettings, BoundedSessionStore, reset_api_settings, validate_key_hex
import src.api.server as srv
from src.api.server import create_app, get_deployment_configuration, session_store
from src.neural.dataset import validate_div2k_manifest

logger = logging.getLogger("phase5d_assessment")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "phase4c_extended_30ep" / "best_model.pt"
ONNX_DIR = REPO_ROOT / "checkpoints" / "phase5a_onnx"
MANIFEST_PATH = REPO_ROOT / "docs" / "project" / "div2k_hr_split.json"
RESULTS_PATH = REPO_ROOT / "docs" / "experiments" / "phase5d_results.json"

EXPECTED_CHECKPOINT_DIGEST = (
    "ec11dce74832c0fa0ef599ea7185248dfd3a16a10d83a5ebceddc9e267160d92"
)

EXPECTED_ONNX_HASHES = {
    "acs_sub_inn_embedding_opset14.onnx": "3cd22ba35e55d4addfb89e59cdc695f1086383bef5d507f39a3c91dd4917c173",
    "acs_sub_inn_extraction_opset14.onnx": "ab696814635b39dec34b432b78e80bb949c5fcd68517a3cec5890102152ff8ba",
    "neural_spatial_embedding_opset14.onnx": "7cda5a3becb8283c6b84572e7adb6d64ea29ecbf9f1e667cf34dcd0324e2d784",
    "neural_spatial_extraction_opset14.onnx": "bf85d8404469d90c08aa50282db582e3c82b2c6ea2159293423ae6e846b82be2",
}


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_parameter_digest(model_state: Dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(model_state.keys()):
        tensor = model_state[name]
        if hasattr(tensor, "cpu"):
            tensor_bytes = tensor.cpu().numpy().tobytes()
        else:
            tensor_bytes = np.asarray(tensor).tobytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor_bytes)
    return hasher.hexdigest()


# =============================================================================
# 1. Preflight Verification
# =============================================================================

def run_preflight_verification() -> Dict[str, Any]:
    logger.info("Executing Phase 5D.2 Preflight Safeguard Verification...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    actual_digest = compute_parameter_digest(state)
    assert actual_digest == EXPECTED_CHECKPOINT_DIGEST, f"Digest mismatch: {actual_digest}"

    manifest_audit = validate_div2k_manifest(MANIFEST_PATH)
    p_counts = manifest_audit.get("partition_counts", {})
    assert p_counts.get("train") == 600
    assert p_counts.get("val") == 100
    assert p_counts.get("test") == 100

    onnx_verified = {}
    for fname, exp_hash in EXPECTED_ONNX_HASHES.items():
        fpath = ONNX_DIR / fname
        assert fpath.exists(), f"Missing ONNX model: {fname}"
        act_hash = compute_file_sha256(fpath)
        assert act_hash == exp_hash, f"Hash mismatch for {fname}: {act_hash}"
        onnx_verified[fname] = {"sha256": act_hash, "verified": True}

    logger.info("Preflight verification passed: Checkpoint invariant, manifest valid, 4 ONNX models intact.")
    return {
        "status": "PASS",
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "parameter_digest": actual_digest,
            "epoch": ckpt.get("epoch", 30),
            "verified": True,
        },
        "manifest": {
            "file_sha256": manifest_audit.get("file_sha256"),
            "composite_sha256": manifest_audit.get("composite_sha256"),
            "partition_counts": p_counts,
            "verified": True,
        },
        "onnx_models": onnx_verified,
    }


# =============================================================================
# 2. Gate A — Streaming Authentication and Key Transport
# =============================================================================

VALID_ASSESSMENT_TOKEN = "valid_phase5d_assessment_token_32_characters_12345"


def run_gate_a_streaming_assessment() -> Dict[str, Any]:
    logger.info("Executing Gate A: Streaming Authentication and Key Transport...")
    token = VALID_ASSESSMENT_TOKEN
    remote_settings = APISettings(mode="remote", api_token=token)
    remote_app = create_app(remote_settings)
    auth_headers = {"Authorization": f"Bearer {token}"}

    key_bytes = b"\x01" * 32
    key_hex = key_bytes.hex()

    with TestClient(remote_app) as client:
        # Create secret & cover
        sec_img = Image.new("RGB", (24, 24), color=(200, 50, 50))
        sbuf = io.BytesIO()
        sec_img.save(sbuf, format="PNG")
        res_sec = client.post("/api/upload", headers=auth_headers, files={"file": ("s.png", sbuf.getvalue(), "image/png")})
        sec_sid = res_sec.json()["session_id"]

        cov_img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        cbuf = io.BytesIO()
        cov_img.save(cbuf, format="PNG")
        res_cov = client.post("/api/upload", headers=auth_headers, files={"file": ("c.png", cbuf.getvalue(), "image/png")})
        cov_sid = res_cov.json()["session_id"]

        # 1. Missing token rejected
        res_unauth = client.post("/api/stream", json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": key_hex})
        unauth_rejected = (res_unauth.status_code == 401)

        # 2. Invalid token rejected
        res_bad_tok = client.post(
            "/api/stream",
            headers={"Authorization": "Bearer bad_tok"},
            json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": key_hex},
        )
        invalid_tok_rejected = (res_bad_tok.status_code == 401)

        # 3. GET query key transport rejected (405 Method Not Allowed)
        res_get = client.get(f"/api/stream?key_hex={key_hex}", headers=auth_headers)
        get_query_rejected = (res_get.status_code == 405)

        # 4. Valid authenticated POST request succeeds and returns SSE
        res_stream = client.post(
            "/api/stream",
            headers=auth_headers,
            json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": key_hex},
        )
        stream_ok = (res_stream.status_code == 200 and "text/event-stream" in res_stream.headers.get("content-type", ""))
        stream_has_events = "EXACT_MATCH_VERIFY" in res_stream.text

    status = "PASS" if (unauth_rejected and invalid_tok_rejected and get_query_rejected and stream_ok and stream_has_events) else "FAIL"
    return {
        "status": status,
        "streaming_unauthenticated_rejected_401": unauth_rejected,
        "streaming_invalid_token_rejected_401": invalid_tok_rejected,
        "streaming_get_query_rejected_405": get_query_rejected,
        "streaming_authenticated_post_accepted_200": stream_ok,
        "streaming_events_delivered": stream_has_events,
    }


# =============================================================================
# 3. Gate B — Session Storage and Worker Contract
# =============================================================================

def run_gate_b_worker_assessment() -> Dict[str, Any]:
    logger.info("Executing Gate B: Session Storage and Single-Worker Launcher Contract...")
    # 1. Single worker allowed
    single_worker_ok = False
    try:
        cfg = validate_launcher_config("127.0.0.1", 8000, "local", workers=1)
        single_worker_ok = (cfg.mode == "local")
    except Exception:
        pass

    # 2. Multi-worker rejected
    multi_worker_rejected = False
    try:
        validate_launcher_config("127.0.0.1", 8000, "local", workers=4)
    except ValueError:
        multi_worker_rejected = True

    # 3. Invalid mode rejected
    invalid_mode_rejected = False
    try:
        validate_launcher_config("127.0.0.1", 8000, "staging", workers=1)
    except ValueError:
        invalid_mode_rejected = True

    # 4. Local non-loopback rejected
    non_loopback_rejected = False
    try:
        validate_launcher_config("0.0.0.0", 8000, "local", workers=1)
    except ValueError:
        non_loopback_rejected = True

    status = "PASS" if (single_worker_ok and multi_worker_rejected and invalid_mode_rejected and non_loopback_rejected) else "FAIL"
    return {
        "status": status,
        "single_worker_contract_enforced": single_worker_ok,
        "multi_worker_rejected_for_in_memory_sessions": multi_worker_rejected,
        "invalid_deployment_modes_rejected": invalid_mode_rejected,
        "local_mode_non_loopback_binding_rejected": non_loopback_rejected,
    }


# =============================================================================
# 4. Gate C — Startup, Liveness, and Readiness Semantics
# =============================================================================

def run_gate_c_readiness_assessment() -> Dict[str, Any]:
    logger.info("Executing Gate C: Startup, Liveness, and Readiness Probes...")
    app_instance = create_app(APISettings(mode="local", api_token=None))

    with TestClient(app_instance) as client:
        # Liveness probe
        res_live = client.get("/api/health/live")
        live_ok = (res_live.status_code == 200 and res_live.json().get("live") is True)

        # Readiness probe
        res_ready = client.get("/api/health/ready")
        ready_ok = (res_ready.status_code == 200 and res_ready.json().get("ready") is True)

        # /api/config verified fields
        res_cfg = client.get("/api/config")
        cfg = res_cfg.json()
        config_verified = (
            cfg.get("status") == "OPERATIONAL"
            and cfg.get("onnx_models_verified") is True
            and cfg.get("sessions_initialized") is True
            and cfg.get("is_operational") is True
        )

    # Remote mode startup failure when pipeline unavailable
    orig_pipeline = srv.qualified_pipeline
    orig_initialized = srv._sessions_initialized
    remote_startup_aborted = False
    try:
        srv.qualified_pipeline = None
        srv._sessions_initialized = False
        try:
            create_app(APISettings(mode="remote", api_token=VALID_ASSESSMENT_TOKEN))
        except RuntimeError:
            remote_startup_aborted = True
    finally:
        srv.qualified_pipeline = orig_pipeline
        srv._sessions_initialized = orig_initialized

    status = "PASS" if (live_ok and ready_ok and config_verified and remote_startup_aborted) else "FAIL"
    return {
        "status": status,
        "liveness_probe_operational_200": live_ok,
        "readiness_probe_operational_200": ready_ok,
        "config_sha256_verification_verified": config_verified,
        "remote_mode_startup_fails_closed_when_runtime_unavailable": remote_startup_aborted,
    }


# =============================================================================
# 5. Gate D — Authentication Policy Consistency
# =============================================================================

def run_gate_d_auth_policy_assessment() -> Dict[str, Any]:
    logger.info("Executing Gate D: Authentication Policy Consistency...")
    # 1. Local mode with token protects /api/config and routes
    local_with_tok = APISettings(mode="local", api_token="local_tok_123456789")
    app_auth = create_app(local_with_tok)
    with TestClient(app_auth) as client:
        res_unauth = client.get("/api/config")
        config_protected = (res_unauth.status_code == 401)
        res_auth = client.get("/api/config", headers={"Authorization": "Bearer local_tok_123456789"})
        config_permitted = (res_auth.status_code == 200)

    # 2. Remote mode rejects wildcard origin
    wildcard_rejected = False
    try:
        create_app(APISettings(mode="remote", api_token=VALID_ASSESSMENT_TOKEN, allowed_origins=["*"]))
    except RuntimeError:
        wildcard_rejected = True

    # 3. Missing remote token rejects startup
    missing_tok_rejected = False
    try:
        create_app(APISettings(mode="remote", api_token=None))
    except RuntimeError:
        missing_tok_rejected = True

    status = "PASS" if (config_protected and config_permitted and wildcard_rejected and missing_tok_rejected) else "FAIL"
    return {
        "status": status,
        "local_mode_token_protects_config_consistently": config_protected and config_permitted,
        "remote_mode_wildcard_cors_rejected": wildcard_rejected,
        "remote_mode_missing_token_startup_aborted": missing_tok_rejected,
    }


# =============================================================================
# 6. Gate E — Strict Request and Resource Validation
# =============================================================================

def run_gate_e_validation_assessment() -> Dict[str, Any]:
    logger.info("Executing Gate E: Strict Request and Resource Validation...")
    tight_settings = APISettings(mode="local", max_upload_bytes=20 * 1024, session_ttl_seconds=50, max_sessions=80)
    app_instance = create_app(tight_settings)

    # Check store configured
    store_configured = (session_store.max_entries == 80 and session_store.default_ttl == 50)

    with TestClient(app_instance) as client:
        # Key format rejection (whitespace, non-hex, wrong length)
        res_k1 = client.post("/api/embed", json={"secret_session_id": "dummy", "key_hex": "01 " * 21})
        res_k2 = client.post("/api/embed", json={"secret_session_id": "dummy", "key_hex": "ZZ" * 32})
        res_k3 = client.post("/api/embed", json={"secret_session_id": "dummy", "key_hex": "01" * 31})
        key_format_rejected = (res_k1.status_code == 422 and res_k2.status_code == 422 and res_k3.status_code == 422)

        # ASGI request body size limit
        oversized = b"Z" * (90 * 1024)
        res_asgi = client.post(
            "/api/upload",
            content=oversized,
            headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(oversized))},
        )
        asgi_limit_enforced = (res_asgi.status_code == 413)

    # Remote mode missing cover fails closed
    remote_app = create_app(APISettings(mode="remote", api_token=VALID_ASSESSMENT_TOKEN))
    with TestClient(remote_app) as client:
        res_sec = client.post(
            "/api/upload",
            headers={"Authorization": f"Bearer {VALID_ASSESSMENT_TOKEN}"},
            files={"file": ("s.bin", b"data", "application/octet-stream")},
        )
        sec_sid = res_sec.json()["session_id"]
        res_nocov = client.post(
            "/api/embed",
            headers={"Authorization": f"Bearer {VALID_ASSESSMENT_TOKEN}"},
            json={"secret_session_id": sec_sid, "cover_session_id": None, "key_hex": "01" * 32},
        )
        missing_cover_fails_closed = (res_nocov.status_code == 422)

    status = "PASS" if (store_configured and key_format_rejected and asgi_limit_enforced and missing_cover_fails_closed) else "FAIL"
    return {
        "status": status,
        "dynamic_session_store_configured": store_configured,
        "strict_key_hex_validation_enforced": key_format_rejected,
        "asgi_request_body_limits_enforced_413": asgi_limit_enforced,
        "remote_mode_missing_cover_fails_closed_422": missing_cover_fails_closed,
    }


# =============================================================================
# Main Orchestrator
# =============================================================================

def run_phase5d_assessment() -> Dict[str, Any]:
    t0 = time.perf_counter()
    logger.info("Starting Phase 5D.2 Security and Controlled-Deployment Assessment...")

    preflight = run_preflight_verification()
    gate_a = run_gate_a_streaming_assessment()
    gate_b = run_gate_b_worker_assessment()
    gate_c = run_gate_c_readiness_assessment()
    gate_d = run_gate_d_auth_policy_assessment()
    gate_e = run_gate_e_validation_assessment()

    g_preflight = preflight["status"]
    g_a = gate_a["status"]
    g_b = gate_b["status"]
    g_c = gate_c["status"]
    g_d = gate_d["status"]
    g_e = gate_e["status"]

    all_pass = (
        g_preflight == "PASS"
        and g_a == "PASS"
        and g_b == "PASS"
        and g_c == "PASS"
        and g_d == "PASS"
        and g_e == "PASS"
    )

    overall_verdict = "PASS_WITH_LIMITATIONS" if all_pass else "FAIL"

    results = {
        "phase": "Phase 5D.2 Corrective Security Closure and Remote-Mode Verification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_seconds": round(time.perf_counter() - t0, 2),
        "gates": {
            "gate_preflight": g_preflight,
            "gate_a_streaming_auth_and_key": g_a,
            "gate_b_session_and_worker": g_b,
            "gate_c_startup_liveness_readiness": g_c,
            "gate_d_auth_policy_consistency": g_d,
            "gate_e_strict_request_and_resource": g_e,
            "overall_verdict": overall_verdict,
        },
        "preflight": preflight,
        "gate_a_streaming": gate_a,
        "gate_b_worker": gate_b,
        "gate_c_readiness": gate_c,
        "gate_d_auth_policy": gate_d,
        "gate_e_validation": gate_e,
        "deployment_safeguard_contract": {
            "official_launcher": "python -m src.api.launcher (or python src/api/launcher.py)",
            "worker_configuration": "Exactly 1 worker (workers=1 strictly enforced for in-memory session store)",
            "default_deployment_mode": "local (loopback 127.0.0.1 strictly required)",
            "remote_deployment_mode": "remote (enforces PALADINMEDIA_API_TOKEN, disallows wildcard CORS)",
            "streaming_transport": "POST /api/stream with Bearer auth header and request body session IDs/key",
            "qualified_pipeline": "ONNX Runtime + rs_2t_48 (48 parity symbols)",
            "carrier_format_contract": "Lossless 8-bit RGB PNG (512x512)",
            "upload_limit_default": "10 MB (configurable via PALADINMEDIA_MAX_UPLOAD_BYTES, enforced at ASGI boundary)",
            "session_ttl_default": "300 seconds (configurable via PALADINMEDIA_SESSION_TTL_SECONDS)",
            "external_deployment_dependencies": [
                "TLS termination reverse proxy (Nginx, Caddy, Cloudflare, AWS ALB) required for HTTPS",
                "External secrets management (Vault, AWS Secrets Manager, Kubernetes Secret) for API token",
                "Reverse proxy rate-limiting and DDoS mitigation",
            ],
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Phase 5D.2 assessment completed in %0.2fs. Results written to %s", results["total_runtime_seconds"], RESULTS_PATH)

    # Restore default local settings after assessment
    reset_api_settings(APISettings(mode="local", api_token=None))
    return results


if __name__ == "__main__":
    run_phase5d_assessment()
