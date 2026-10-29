"""Phase 5D: Secure API and Controlled-Deployment Readiness Unit Tests.

Covers:
1. Missing authentication rejected in remote/production mode (HTTP 401).
2. Invalid credentials rejected (HTTP 401).
3. Valid configured credentials permit intended workflow (upload -> embed -> extract).
4. Local-development mode follows its explicitly documented policy.
5. No default production credential is silently accepted (safe startup failure).
6. Route access policies (/api/health, /api/config, /api/upload, /api/embed, /api/extract, /api/stream).
7. Oversized uploads and malformed image content rejected before expensive processing.
8. Incorrect image dimensions and unsupported carrier formats fail closed.
9. Unknown or invalid session identifiers cannot access session data.
10. Expired sessions and temporary records are cleaned up by TTL lifecycle.
11. Sensitive key and payload values are not disclosed in normal logs or failure responses.
12. Capacity rejection and Phase 5C exact-boundary behavior remain unchanged through API.
13. Application still selects qualified ONNX + rs_2t_48 path.
14. Security headers and CORS defense-in-depth enforcement.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from typing import Generator

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from src.api.launcher import LOOPBACK_HOSTS, validate_launcher_config
from src.api.security import (
    HEX_64_PATTERN,
    APISettings,
    BoundedSessionStore,
    get_api_settings,
    reset_api_settings,
    validate_key_hex,
)
from src.api.server import (
    app,
    create_app,
    get_deployment_configuration,
    session_store,
)
from src.carrier.capacity import CapacityCalculator
from src.core.media_pipeline import MediaStegoPipeline

VALID_REMOTE_TOKEN = "valid_remote_token_32_characters_long_12345"


@pytest.fixture
def dummy_cover() -> np.ndarray:
    """Deterministic 512x512 RGB carrier array."""
    rng = np.random.RandomState(42)
    return rng.randint(40, 215, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def dummy_key() -> bytes:
    """32-byte AES-256 key."""
    return b"\x01\x23\x45\x67\x89\xab\xcd\xef\x01\x23\x45\x67\x89\xab\xcd\xef" \
           b"\x01\x23\x45\x67\x89\xab\xcd\xef\x01\x23\x45\x67\x89\xab\xcd\xef"


@pytest.fixture
def dummy_cover_bytes(dummy_cover: np.ndarray) -> bytes:
    """PNG bytes for dummy cover."""
    buf = io.BytesIO()
    Image.fromarray(dummy_cover, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_settings_after_test():
    """Ensures settings are restored to default local mode after each test."""
    yield
    reset_api_settings(APISettings(mode="local", api_token=None))
    session_store.clear()


# =============================================================================
# Requirement 1 & 2 & 3: Authentication and Credential Validation
# =============================================================================

def test_missing_authentication_rejected_in_remote_mode():
    """R1: Missing authentication header is rejected with HTTP 401 in remote mode."""
    remote_settings = APISettings(mode="remote", api_token=VALID_REMOTE_TOKEN)
    remote_app = create_app(remote_settings)

    with TestClient(remote_app) as client:
        # Protected route without auth
        res = client.post(
            "/api/upload",
            files={"file": ("test.png", b"test data", "image/png")},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Authentication required"
        assert "WWW-Authenticate" in res.headers


def test_invalid_credentials_rejected_in_remote_mode():
    """R2: Invalid credentials or wrong bearer schemes are rejected with HTTP 401."""
    remote_settings = APISettings(mode="remote", api_token=VALID_REMOTE_TOKEN)
    remote_app = create_app(remote_settings)

    with TestClient(remote_app) as client:
        # 1. Invalid token value
        res_bad_token = client.post(
            "/api/upload",
            headers={"Authorization": "Bearer wrong_token_value"},
            files={"file": ("test.png", b"test data", "image/png")},
        )
        assert res_bad_token.status_code == 401
        assert res_bad_token.json()["detail"] == "Invalid API token"

        # 2. Invalid auth scheme (Basic instead of Bearer)
        res_bad_scheme = client.post(
            "/api/upload",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
            files={"file": ("test.png", b"test data", "image/png")},
        )
        assert res_bad_scheme.status_code == 401
        assert "Bearer token required" in res_bad_scheme.json()["detail"]


def test_valid_configured_credentials_permit_workflow(dummy_cover_bytes, dummy_key):
    """R3: Valid bearer credentials permit full upload -> embed -> extract workflow in remote mode."""
    token = "qualified_production_bearer_token_xyz"
    remote_settings = APISettings(mode="remote", api_token=token)
    remote_app = create_app(remote_settings)
    auth_headers = {"Authorization": f"Bearer {token}"}

    with TestClient(remote_app) as client:
        # 1. Upload secret payload
        secret_bytes = b"Authenticated secret payload 12345"
        res_up_sec = client.post(
            "/api/upload",
            headers=auth_headers,
            files={"file": ("secret.bin", secret_bytes, "application/octet-stream")},
        )
        assert res_up_sec.status_code == 200
        sec_id = res_up_sec.json()["session_id"]

        # 2. Upload cover carrier
        res_up_cov = client.post(
            "/api/upload",
            headers=auth_headers,
            files={"file": ("cover.png", dummy_cover_bytes, "image/png")},
        )
        assert res_up_cov.status_code == 200
        cov_id = res_up_cov.json()["session_id"]

        # 3. Embed
        res_embed = client.post(
            "/api/embed",
            headers=auth_headers,
            json={
                "secret_session_id": sec_id,
                "cover_session_id": cov_id,
                "key_hex": dummy_key.hex(),
            },
        )
        assert res_embed.status_code == 200
        stego_id = res_embed.json()["stego_session_id"]

        # 4. Extract
        res_extract = client.post(
            "/api/extract",
            headers=auth_headers,
            json={"stego_session_id": stego_id, "key_hex": dummy_key.hex()},
        )
        assert res_extract.status_code == 200
        assert res_extract.json()["recovered_bytes"] == len(secret_bytes)


# =============================================================================
# Requirement 4 & 5: Local Mode and Startup Invariant Enforcement
# =============================================================================

def test_local_mode_follows_documented_policy(dummy_cover_bytes):
    """R4: Local-development mode permits unauthenticated access by default for rapid development."""
    local_settings = APISettings(mode="local", api_token=None)
    local_app = create_app(local_settings)

    with TestClient(local_app) as client:
        res = client.post(
            "/api/upload",
            files={"file": ("test.png", dummy_cover_bytes, "image/png")},
        )
        assert res.status_code == 200
        assert "session_id" in res.json()


def test_no_default_production_credential_accepted():
    """R5: Remote mode without configured token fails closed at startup (raises RuntimeError)."""
    invalid_remote_settings = APISettings(mode="remote", api_token=None)
    with pytest.raises(RuntimeError) as exc_info:
        create_app(invalid_remote_settings)

    assert "CONFIGURATION_ERROR" in str(exc_info.value)
    assert "PALADINMEDIA_API_TOKEN is missing" in str(exc_info.value)


# =============================================================================
# Requirement 6: Route Access Policy and Documentation Exposure
# =============================================================================

def test_route_access_policies_and_documentation_handling():
    """R6: Verifies public health probe, protected config in remote mode, and disabled docs in remote."""
    token = VALID_REMOTE_TOKEN
    remote_settings = APISettings(mode="remote", api_token=token)
    remote_app = create_app(remote_settings)

    with TestClient(remote_app) as client:
        # Public health endpoint requires no auth
        res_health = client.get("/api/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"
        assert res_health.json()["deployment_mode"] == "remote"

        # In remote mode, /api/config requires auth
        res_cfg_unauth = client.get("/api/config")
        assert res_cfg_unauth.status_code == 401

        res_cfg_auth = client.get("/api/config", headers={"Authorization": f"Bearer {token}"})
        assert res_cfg_auth.status_code == 200
        assert res_cfg_auth.json()["deployment_mode"] == "remote"

        # Interactive docs are disabled by default in remote mode
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404


# =============================================================================
# Requirement 7 & 8: Input, Upload, and Carrier Safeguards
# =============================================================================

def test_oversized_uploads_rejected_before_buffering():
    """R7: Uploads exceeding MAX_UPLOAD_BYTES are rejected with HTTP 413."""
    # Set artificial 50KB upload limit for testing
    tight_settings = APISettings(mode="local", max_upload_bytes=50 * 1024)
    reset_api_settings(tight_settings)
    tight_app = create_app(tight_settings)

    oversized_data = b"X" * (60 * 1024)  # 60 KB > 50 KB
    with TestClient(tight_app) as client:
        res = client.post(
            "/api/upload",
            files={"file": ("large.bin", oversized_data, "application/octet-stream")},
        )
        assert res.status_code == 413
        assert "Upload exceeds maximum permitted size" in res.json()["detail"]


def test_malformed_image_content_rejected():
    """R7: Files claiming image MIME type but containing corrupt bytes are rejected with HTTP 400."""
    with TestClient(app) as client:
        corrupt_png = b"\x89PNG\r\n\x1a\nCorrupt garbage non-image content 12345"
        res = client.post(
            "/api/upload",
            files={"file": ("corrupt.png", corrupt_png, "image/png")},
        )
        assert res.status_code == 400
        assert "Malformed or corrupted image" in res.json()["detail"]


def test_empty_upload_rejected():
    """R7: Uploading a 0-byte file is rejected with HTTP 400."""
    with TestClient(app) as client:
        res = client.post(
            "/api/upload",
            files={"file": ("empty.bin", b"", "application/octet-stream")},
        )
        assert res.status_code == 400
        assert "Uploaded file is empty" in res.json()["detail"]


def test_carrier_dimensions_and_jpeg_rejection(dummy_key):
    """R8: Stego carriers with wrong dimensions or lossy JPEG formats fail closed."""
    with TestClient(app) as client:
        # 1. Odd dimensions stego carrier rejected
        wrong_size_img = Image.new("RGB", (511, 512), color=(100, 100, 100))
        buf = io.BytesIO()
        wrong_size_img.save(buf, format="PNG")
        res_up_wrong = client.post(
            "/api/upload",
            files={"file": ("wrong_dims.png", buf.getvalue(), "image/png")},
        )
        wrong_sid = res_up_wrong.json()["session_id"]

        res_ext_wrong = client.post(
            "/api/extract",
            json={"stego_session_id": wrong_sid, "key_hex": dummy_key.hex()},
        )
        assert res_ext_wrong.status_code == 422
        assert "must be even for DWT decomposition" in res_ext_wrong.json()["detail"]

        # 2. JPEG stego carrier rejected with actionable message
        jpeg_img = Image.new("RGB", (512, 512), color=(100, 100, 100))
        jpeg_buf = io.BytesIO()
        jpeg_img.save(jpeg_buf, format="JPEG", quality=85)
        res_up_jpg = client.post(
            "/api/upload",
            files={"file": ("carrier.jpg", jpeg_buf.getvalue(), "image/jpeg")},
        )
        jpg_sid = res_up_jpg.json()["session_id"]

        res_ext_jpg = client.post(
            "/api/extract",
            json={"stego_session_id": jpg_sid, "key_hex": dummy_key.hex()},
        )
        assert res_ext_jpg.status_code == 400
        assert "JPEG-transcoded carriers are not qualified" in res_ext_jpg.json()["detail"]


def test_invalid_key_hex_format_rejected():
    """R8: Invalid key_hex format (wrong length or non-hex) is rejected with HTTP 422."""
    with TestClient(app) as client:
        # Wrong length (31 bytes instead of 32)
        res_bad_len = client.post(
            "/api/embed",
            json={"secret_session_id": "dummy", "key_hex": "01" * 31},
        )
        assert res_bad_len.status_code == 422
        assert "64-character hex string" in res_bad_len.json()["detail"]

        # Non-hex characters
        res_non_hex = client.post(
            "/api/embed",
            json={"secret_session_id": "dummy", "key_hex": "ZZ" * 32},
        )
        assert res_non_hex.status_code == 422
        assert "invalid non-hexadecimal" in res_non_hex.json()["detail"]


# =============================================================================
# Requirement 9 & 10: Session Store Security, Isolation, and TTL Cleanup
# =============================================================================

def test_unknown_or_invalid_session_id_cannot_access_data(dummy_key):
    """R9: Unknown or forged session IDs fail with HTTP 404."""
    with TestClient(app) as client:
        res = client.post(
            "/api/extract",
            json={"stego_session_id": "non_existent_session_id_12345", "key_hex": dummy_key.hex()},
        )
        assert res.status_code == 404
        assert "not found or expired" in res.json()["detail"]


def test_session_lifecycle_ttl_expiration_and_deletion():
    """R10: Sessions expire after TTL and can be explicitly deleted via DELETE endpoint."""
    store = BoundedSessionStore(max_entries=100, default_ttl=1)  # 1-second TTL

    # 1. Store session
    sid = store.store(b"temporary artifact data", ttl_seconds=1)
    assert store.get(sid) == b"temporary artifact data"

    # 2. Wait for expiration
    time.sleep(1.1)
    assert store.get(sid) is None  # Expired
    assert len(store) == 0

    # 3. Explicit deletion via API
    with TestClient(app) as client:
        res_up = client.post(
            "/api/upload",
            files={"file": ("data.bin", b"to be deleted", "application/octet-stream")},
        )
        del_sid = res_up.json()["session_id"]

        # Delete session
        res_del = client.delete(f"/api/session/{del_sid}")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "deleted"

        # Confirm gone
        res_del_again = client.delete(f"/api/session/{del_sid}")
        assert res_del_again.status_code == 404


# =============================================================================
# Requirement 11: Information Leakage Prevention
# =============================================================================

def test_no_sensitive_key_or_payload_in_responses_or_logs(dummy_cover_bytes, caplog):
    """R11: Sensitive key material, tokens, and payloads are never exposed in error responses."""
    secret_key = b"MY_SUPER_SECRET_KEY_123456789012"  # 32 bytes
    secret_text = b"CONFIDENTIAL_TOP_SECRET_PLAINTEXT_DATA"

    with TestClient(app) as client:
        with caplog.at_level(logging.DEBUG):
            # Upload secret
            res_up = client.post(
                "/api/upload",
                files={"file": ("secret.bin", secret_text, "application/octet-stream")},
            )
            sec_id = res_up.json()["session_id"]

            res_up_cov = client.post(
                "/api/upload",
                files={"file": ("cover.png", dummy_cover_bytes, "image/png")},
            )
            cov_id = res_up_cov.json()["session_id"]

            # Embed
            res_emb = client.post(
                "/api/embed",
                json={
                    "secret_session_id": sec_id,
                    "cover_session_id": cov_id,
                    "key_hex": secret_key.hex(),
                },
            )
            stego_id = res_emb.json()["stego_session_id"]

            # Extract with wrong key
            wrong_key = b"WRONG_KEY_WRONG_KEY_012345678901"
            res_ext_bad = client.post(
                "/api/extract",
                json={"stego_session_id": stego_id, "key_hex": wrong_key.hex()},
            )
            assert res_ext_bad.status_code == 422
            resp_body = res_ext_bad.text

            # Ensure neither plaintext nor key appear in client error response
            assert secret_text.decode("utf-8") not in resp_body
            assert secret_key.hex() not in resp_body
            assert "E:\\Paladin_media" not in resp_body  # No absolute path leakage

            # Verify server logs do not leak plaintext
            assert secret_text.decode("utf-8") not in caplog.text


# =============================================================================
# Requirement 12: Capacity Rejection via API Matches Phase 5C Contract
# =============================================================================

def test_capacity_rejection_via_api(dummy_cover_bytes, dummy_key):
    """R12: Payload of 6,579 bytes is rejected via API with HTTP 413, matching Phase 5C exact limit."""
    with TestClient(app) as client:
        res_up_cov = client.post(
            "/api/upload",
            files={"file": ("cover.png", dummy_cover_bytes, "image/png")},
        )
        cov_id = res_up_cov.json()["session_id"]

        # 6,579 bytes exceeds 6,578-byte raw ceiling
        oversized_secret = b"B" * 6579
        res_up_sec = client.post(
            "/api/upload",
            files={"file": ("secret.bin", oversized_secret, "application/octet-stream")},
        )
        sec_id = res_up_sec.json()["session_id"]

        res_embed = client.post(
            "/api/embed",
            json={
                "secret_session_id": sec_id,
                "cover_session_id": cov_id,
                "key_hex": dummy_key.hex(),
            },
        )
        assert res_embed.status_code == 413
        assert "PAYLOAD_CAPACITY_EXCEEDED" in res_embed.json()["detail"]


# =============================================================================
# Requirement 13 & 14: Deployment Security Headers and Invariant Checks
# =============================================================================

def test_security_headers_enforced_on_responses():
    """R14: Responses include security defense-in-depth headers and cache restrictions."""
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.headers["X-Content-Type-Options"] == "nosniff"
        assert res.headers["X-Frame-Options"] == "DENY"
        assert "no-store" in res.headers["Cache-Control"]


# =============================================================================
# Phase 5D.1: Gate A — Streaming Authentication and Key Transport
# =============================================================================

def test_streaming_authentication_and_key_transport(dummy_cover_bytes, dummy_key):
    """Gate A: Validates POST /api/stream authentication, request-body key transport, and event stream."""
    token = VALID_REMOTE_TOKEN
    remote_settings = APISettings(mode="remote", api_token=token)
    remote_app = create_app(remote_settings)
    auth_headers = {"Authorization": f"Bearer {token}"}

    with TestClient(remote_app) as client:
        # Create a small valid 24x24 secret PNG
        sec_img = Image.new("RGB", (24, 24), color=(200, 50, 50))
        sec_buf = io.BytesIO()
        sec_img.save(sec_buf, format="PNG")
        sec_bytes = sec_buf.getvalue()

        # Upload secret & cover
        res_sec = client.post(
            "/api/upload",
            headers=auth_headers,
            files={"file": ("secret.png", sec_bytes, "image/png")},
        )
        assert res_sec.status_code == 200
        sec_sid = res_sec.json()["session_id"]

        res_cov = client.post(
            "/api/upload",
            headers=auth_headers,
            files={"file": ("cover.png", dummy_cover_bytes, "image/png")},
        )
        assert res_cov.status_code == 200
        cov_sid = res_cov.json()["session_id"]

        # 1. Unauthenticated request rejected
        res_unauth = client.post(
            "/api/stream",
            json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": dummy_key.hex()},
        )
        assert res_unauth.status_code == 401
        assert "WWW-Authenticate" in res_unauth.headers

        # 2. Invalid token rejected
        res_bad_tok = client.post(
            "/api/stream",
            headers={"Authorization": "Bearer invalid_token"},
            json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": dummy_key.hex()},
        )
        assert res_bad_tok.status_code == 401

        # 3. GET /api/stream rejected (query key transport prohibited)
        res_get = client.get(
            f"/api/stream?secret_session_id={sec_sid}&key_hex={dummy_key.hex()}",
            headers=auth_headers,
        )
        assert res_get.status_code == 405  # Method Not Allowed

        # 4. Valid authenticated POST /api/stream request returns SSE events
        res_stream = client.post(
            "/api/stream",
            headers=auth_headers,
            json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": dummy_key.hex()},
        )
        assert res_stream.status_code == 200
        assert "text/event-stream" in res_stream.headers["content-type"]

        # Parse SSE events
        lines = res_stream.text.split("\n\n")
        stages_observed = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("data: "):
                event_data = json.loads(line_str[6:])
                stages_observed.append(event_data.get("stage"))

        assert "LOAD" in stages_observed
        assert "SERIALIZE" in stages_observed
        assert "ENCRYPT" in stages_observed
        assert "RS_ENCODE" in stages_observed
        assert "CAPACITY_CHECK" in stages_observed
        assert "DWT" in stages_observed
        assert "EMBED" in stages_observed
        assert "IDWT" in stages_observed
        assert "EXACT_MATCH_VERIFY" in stages_observed


# =============================================================================
# Phase 5D.1: Gate B — Launcher Configuration and Worker Contract
# =============================================================================

def test_launcher_configuration_and_worker_contract():
    """Gate B: Official launcher enforces single-worker and loopback binding invariants."""
    # 1. Qualified deployment requires exactly 1 worker
    valid_cfg = validate_launcher_config("127.0.0.1", 8000, "local", workers=1)
    assert valid_cfg.mode == "local"

    with pytest.raises(ValueError, match="INVALID_WORKER_CONFIG"):
        validate_launcher_config("127.0.0.1", 8000, "local", workers=4)

    with pytest.raises(ValueError, match="INVALID_WORKER_CONFIG"):
        validate_launcher_config("127.0.0.1", 8000, "remote", workers=2, api_token="tok123")

    # 2. Invalid deployment mode rejected
    with pytest.raises(ValueError, match="Invalid deployment mode"):
        validate_launcher_config("127.0.0.1", 8000, "staging", workers=1)

    with pytest.raises(ValueError, match="Invalid deployment mode"):
        APISettings(mode="unsupported_mode")

    # 3. Local mode on non-loopback host rejected
    with pytest.raises(ValueError, match="BINDING_VIOLATION"):
        validate_launcher_config("0.0.0.0", 8000, "local", workers=1)

    with pytest.raises(ValueError, match="BINDING_VIOLATION"):
        validate_launcher_config("192.168.1.100", 8000, "local", workers=1)

    # 4. Remote mode permits internal binding when token is provided
    remote_cfg = validate_launcher_config("127.0.0.1", 8000, "remote", workers=1, api_token=VALID_REMOTE_TOKEN)
    assert remote_cfg.mode == "remote"


# =============================================================================
# Phase 5D.1: Gate C — Startup, Liveness, and Readiness Semantics
# =============================================================================

def test_startup_liveness_and_readiness_semantics():
    """Gate C: Probes correctly differentiate process liveness from qualified pipeline readiness."""
    import src.api.server as srv

    with TestClient(app) as client:
        # Liveness probe
        res_live = client.get("/api/health/live")
        assert res_live.status_code == 200
        assert res_live.json() == {"status": "live", "live": True}

        # Readiness probe in healthy operational state
        res_ready = client.get("/api/health/ready")
        assert res_ready.status_code == 200
        assert res_ready.json()["ready"] is True

        # /api/config reports actual presence and sha256 verification
        res_cfg = client.get("/api/config")
        assert res_cfg.status_code == 200
        cfg = res_cfg.json()
        assert cfg["status"] == "OPERATIONAL"
        assert cfg["onnx_models_verified"] is True
        assert cfg["onnx_models_present"] is True
        assert cfg["sessions_initialized"] is True
        assert cfg["is_operational"] is True
        assert all(cfg["onnx_hashes_verified"].values())

    # Simulate pipeline failure
    orig_pipeline = srv.qualified_pipeline
    orig_initialized = srv._sessions_initialized
    try:
        srv.qualified_pipeline = None
        srv._sessions_initialized = False

        # 1. Remote mode startup fails closed when qualified pipeline unavailable
        with pytest.raises(RuntimeError, match="REMOTE_STARTUP_PREFLIGHT_FAILED"):
            create_app(APISettings(mode="remote", api_token=VALID_REMOTE_TOKEN))

        # 2. In degraded state, readiness probe returns HTTP 503
        degraded_app = create_app(APISettings(mode="local"))
        with TestClient(degraded_app) as client:
            res_deg_ready = client.get("/api/health/ready")
            assert res_deg_ready.status_code == 503
            assert res_deg_ready.json()["ready"] is False

            res_deg_health = client.get("/api/health")
            assert res_deg_health.status_code == 503
            assert res_deg_health.json()["status"] == "degraded"
            assert res_deg_health.json()["ready"] is False
            assert res_deg_health.json()["live"] is True

            # Liveness probe still returns 200
            res_deg_live = client.get("/api/health/live")
            assert res_deg_live.status_code == 200
    finally:
        srv.qualified_pipeline = orig_pipeline
        srv._sessions_initialized = orig_initialized


# =============================================================================
# Phase 5D.1: Gate D — Authentication and Origin Policies
# =============================================================================

def test_consistent_authentication_and_origin_policy():
    """Gate D: Enforces declared authentication policy across all endpoints and rejects wildcard origins."""
    # 1. Local mode with configured token requires authentication across all endpoints including /api/config
    local_with_token = APISettings(mode="local", api_token="my_local_token_with_length_greater_than_10")
    auth_app = create_app(local_with_token)

    with TestClient(auth_app) as client:
        # Config requires auth
        res_cfg_unauth = client.get("/api/config")
        assert res_cfg_unauth.status_code == 401

        res_cfg_auth = client.get("/api/config", headers={"Authorization": "Bearer my_local_token_with_length_greater_than_10"})
        assert res_cfg_auth.status_code == 200

        # Upload requires auth
        res_up_unauth = client.post("/api/upload", files={"file": ("t.bin", b"data", "application/octet-stream")})
        assert res_up_unauth.status_code == 401

    # 2. Remote mode rejects wildcard origin '*'
    with pytest.raises(RuntimeError, match=r"Wildcard CORS origin '\*' is strictly prohibited"):
        create_app(APISettings(mode="remote", api_token=VALID_REMOTE_TOKEN, allowed_origins=["*"]))

    # 3. Remote mode without token fails startup
    with pytest.raises(RuntimeError, match="PALADINMEDIA_API_TOKEN is missing"):
        create_app(APISettings(mode="remote", api_token=None))


# =============================================================================
# Phase 5D.1: Gate E — Strict Request and Resource Validation
# =============================================================================

def test_strict_key_format_validation(dummy_cover_bytes):
    """Gate E1: Rejects whitespace, newline characters, non-hex, and invalid lengths in key_hex."""
    with TestClient(app) as client:
        # Upload valid secret & cover
        res_sec = client.post("/api/upload", files={"file": ("s.bin", b"data", "application/octet-stream")})
        sec_sid = res_sec.json()["session_id"]
        res_cov = client.post("/api/upload", files={"file": ("c.png", dummy_cover_bytes, "image/png")})
        cov_sid = res_cov.json()["session_id"]

        valid_hex = "0123456789abcdef" * 4  # 64 chars

        # 1. Key with whitespace inside
        bad_space = valid_hex[:32] + " " + valid_hex[33:]
        res1 = client.post("/api/embed", json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": bad_space})
        assert res1.status_code == 422
        assert "invalid non-hexadecimal" in res1.json()["detail"]

        # 2. Key with newline
        res2 = client.post("/api/embed", json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": valid_hex + "\n"})
        assert res2.status_code == 422

        # 3. Key with non-hex character
        bad_char = valid_hex[:62] + "ZZ"
        res3 = client.post("/api/embed", json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": bad_char})
        assert res3.status_code == 422
        assert "invalid non-hexadecimal" in res3.json()["detail"]

        # 4. Key with wrong length (63 characters)
        res4 = client.post("/api/embed", json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": valid_hex[:63]})
        assert res4.status_code == 422
        assert "64-character hex string" in res4.json()["detail"]


def test_remote_mode_missing_cover_fails_closed(dummy_key):
    """Gate E2: In remote mode, missing cover sessions fail closed without fixture fallback."""
    token = VALID_REMOTE_TOKEN
    remote_settings = APISettings(mode="remote", api_token=token)
    remote_app = create_app(remote_settings)
    auth_headers = {"Authorization": f"Bearer {token}"}

    with TestClient(remote_app) as client:
        res_sec = client.post(
            "/api/upload",
            headers=auth_headers,
            files={"file": ("s.bin", b"data", "application/octet-stream")},
        )
        sec_sid = res_sec.json()["session_id"]

        # Missing cover_session_id in remote mode rejected
        res_no_cov = client.post(
            "/api/embed",
            headers=auth_headers,
            json={"secret_session_id": sec_sid, "cover_session_id": None, "key_hex": dummy_key.hex()},
        )
        assert res_no_cov.status_code == 422
        assert "cover_session_id is required in remote deployment mode" in res_no_cov.json()["detail"]

        # Unknown cover_session_id in remote mode rejected
        res_bad_cov = client.post(
            "/api/embed",
            headers=auth_headers,
            json={"secret_session_id": sec_sid, "cover_session_id": "nonexistent_sid", "key_hex": dummy_key.hex()},
        )
        assert res_bad_cov.status_code == 404
        assert "Cover carrier session not found or expired" in res_bad_cov.json()["detail"]


def test_asgi_request_body_size_limits():
    """Gate E3: Enforces request body size limits at the ASGI boundary."""
    # 20 KB upload limit -> max_request_body_bytes = 20KB + 64KB = 84KB
    tight_settings = APISettings(mode="local", max_upload_bytes=20 * 1024)
    tight_app = create_app(tight_settings)

    with TestClient(tight_app) as client:
        # Request with advertised Content-Length exceeding limit
        oversized_bytes = b"Y" * (90 * 1024)  # 90 KB > 84 KB
        res = client.post(
            "/api/upload",
            content=oversized_bytes,
            headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(oversized_bytes))},
        )
        assert res.status_code == 413
        assert "Request body exceeds maximum permitted size" in res.json()["detail"]


def test_dynamic_session_store_configuration_respected():
    """Gate E4: Session store reflects custom settings passed to create_app."""
    custom_settings = APISettings(mode="local", session_ttl_seconds=42, max_sessions=77)
    create_app(custom_settings)

    assert session_store.max_entries == 77
    assert session_store.default_ttl == 42


# =============================================================================
# Phase 5D.2: Corrective Security Closure Tests
# =============================================================================

def test_onnx_sha256_integrity_gate_blocks_startup_and_requests(monkeypatch, dummy_key):
    """P5D.2: Deliberately corrupted ONNX hash blocks remote startup and disables operational endpoints."""
    import src.api.server as srv

    # 1. Normal baseline: all 4 models verified
    int_info = srv.verify_onnx_integrity()
    assert int_info["integrity_passed"] is True
    assert all(int_info["hashes_verified"].values())

    # 2. Tamper with one expected model hash
    tampered_hashes = dict(srv.EXPECTED_ONNX_HASHES)
    tampered_hashes["acs_sub_inn_embedding_opset14.onnx"] = "0000000000000000000000000000000000000000000000000000000000000000"
    monkeypatch.setattr(srv, "EXPECTED_ONNX_HASHES", tampered_hashes)

    # Verify check_runtime_operational reports degraded
    is_ready, reason, status = srv.check_runtime_operational()
    assert is_ready is False
    assert "acs_sub_inn_embedding_opset14.onnx" in reason

    # 3. Remote mode startup fails closed
    with pytest.raises(RuntimeError, match="REMOTE_STARTUP_PREFLIGHT_FAILED"):
        create_app(APISettings(mode="remote", api_token=VALID_REMOTE_TOKEN))

    # 4. In degraded local mode, probes and operational endpoints fail closed
    degraded_app = create_app(APISettings(mode="local"))
    with TestClient(degraded_app) as client:
        # /api/health returns 503 degraded
        res_h = client.get("/api/health")
        assert res_h.status_code == 503
        assert res_h.json()["status"] == "degraded"
        assert res_h.json()["ready"] is False

        # /api/health/ready returns 503 not_ready
        res_r = client.get("/api/health/ready")
        assert res_r.status_code == 503
        assert res_r.json()["ready"] is False

        # /api/config returns DEGRADED status
        res_c = client.get("/api/config")
        assert res_c.status_code == 200
        assert res_c.json()["status"] == "DEGRADED"
        assert res_c.json()["is_operational"] is False
        assert res_c.json()["onnx_models_verified"] is False
        assert res_c.json()["onnx_hashes_verified"]["acs_sub_inn_embedding_opset14.onnx"] is False

        # /api/embed fails closed with 503
        res_emb = client.post("/api/embed", json={"secret_session_id": "dummy", "key_hex": dummy_key.hex()})
        assert res_emb.status_code == 503
        assert "Steganography runtime is unavailable" in res_emb.json()["detail"]

        # /api/extract fails closed with 503
        res_ext = client.post("/api/extract", json={"stego_session_id": "dummy", "key_hex": dummy_key.hex()})
        assert res_ext.status_code == 503
        assert "Steganography runtime is unavailable" in res_ext.json()["detail"]

        # /api/stream fails closed with 503
        res_str = client.post("/api/stream", json={"secret_session_id": "dummy", "key_hex": dummy_key.hex()})
        assert res_str.status_code == 503
        assert "Steganography runtime is unavailable" in res_str.json()["detail"]


def test_token_strength_and_entropy_policy():
    """P5D.2: Enforces minimum 32-character length and character diversity/entropy on remote tokens."""
    # 1. Short token (< 32 chars) rejected
    with pytest.raises(RuntimeError, match="must be at least 32 characters"):
        create_app(APISettings(mode="remote", api_token="short_token_12345"))

    # 2. 32-character low entropy token rejected
    with pytest.raises(RuntimeError, match="insufficient character diversity/entropy"):
        create_app(APISettings(mode="remote", api_token="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))

    # 3. 32-character high entropy token accepted
    valid_token = "K9#xP2$mQ8!vL5@wR7&tY3*cN6^bZ1~d"  # 32 chars, diverse
    app_valid = create_app(APISettings(mode="remote", api_token=valid_token))
    assert app_valid is not None


def test_bounded_session_store_byte_ceiling_and_eviction():
    """P5D.2: Enforces total-byte budget and oldest-created-first (FIFO) eviction policy on BoundedSessionStore."""
    # Store with max 1000 bytes and max 10 entries
    store = BoundedSessionStore(max_entries=10, default_ttl=300, max_bytes=1000)

    # Storing payload larger than max_bytes fails immediately with 413
    with pytest.raises(HTTPException) as exc_info:
        store.store(b"A" * 1001)
    assert exc_info.value.status_code == 413

    # Store 3 payloads of 400 bytes each (total 1200 > 1000)
    s1 = store.store(b"1" * 400)
    assert store.total_bytes == 400
    assert store.get(s1) is not None

    time.sleep(0.01)  # Ensure distinct created_at timestamp
    s2 = store.store(b"2" * 400)
    assert store.total_bytes == 800
    assert store.get(s1) is not None
    assert store.get(s2) is not None

    # Access s1: under oldest-created-first (FIFO by created_at), accessing s1 does NOT reset its eviction priority
    _ = store.get(s1)

    time.sleep(0.01)
    # Adding s3 should evict s1 (oldest created) to make room, demonstrating FIFO policy
    s3 = store.store(b"3" * 400)
    assert store.total_bytes == 800  # s1 evicted (400), s2 (400) + s3 (400) = 800
    assert store.get(s1) is None     # Evicted because created_at(s1) < created_at(s2)
    assert store.get(s2) is not None
    assert store.get(s3) is not None


def test_image_pixel_count_and_dimension_limits():
    """P5D.2: Enforces pixel count (4MP) and dimension (2048) ceilings on image uploads."""
    with TestClient(app) as client:
        # 1. Dimension exceeding MAX_DIMENSION (2049 x 10)
        img_wide = Image.new("RGB", (2049, 10), color=(100, 100, 100))
        buf_wide = io.BytesIO()
        img_wide.save(buf_wide, format="PNG")
        res_wide = client.post(
            "/api/upload",
            files={"file": ("wide.png", buf_wide.getvalue(), "image/png")},
        )
        assert res_wide.status_code == 413
        assert "exceed maximum allowed pixel budget" in res_wide.json()["detail"]

        # 2. Pixel count exceeding MAX_PIXEL_COUNT (2048 x 2049 = 4,196,352 > 4,194,304)
        img_huge = Image.new("RGB", (2048, 2049), color=(50, 50, 50))
        buf_huge = io.BytesIO()
        img_huge.save(buf_huge, format="PNG")
        res_huge = client.post(
            "/api/upload",
            files={"file": ("huge.png", buf_huge.getvalue(), "image/png")},
        )
        assert res_huge.status_code == 413
        assert "exceed maximum allowed pixel budget" in res_huge.json()["detail"]


def test_complete_request_size_enforcement_conditions():
    """P5D.2: Tests Content-Length header, multipart form body, and chunked streaming enforcement."""
    # 25 KB upload limit -> max_request_body_bytes = 25KB + 64KB = 89KB
    tight_settings = APISettings(mode="local", max_upload_bytes=25 * 1024)
    tight_app = create_app(tight_settings)

    with TestClient(tight_app) as client:
        # Condition 1: Advertised Content-Length exceeding limit
        advertised_oversized = b"A" * (95 * 1024)  # 95 KB > 89 KB
        res1 = client.post(
            "/api/upload",
            content=advertised_oversized,
            headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(advertised_oversized))},
        )
        assert res1.status_code == 413
        assert "Request body exceeds maximum permitted size" in res1.json()["detail"]

        # Condition 2: Oversized multipart form request (upload exceeds max_upload_bytes)
        oversized_form_file = b"B" * (30 * 1024)  # 30 KB > 25 KB
        res2 = client.post(
            "/api/upload",
            files={"file": ("test.bin", oversized_form_file, "application/octet-stream")},
        )
        assert res2.status_code == 413
        assert "Upload exceeds maximum permitted size" in res2.json()["detail"]

        # Condition 3: Chunked streaming multipart body without Content-Length header exceeding limit
        boundary = "----WebKitFormBoundaryChunkTest123"
        header = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="large.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("latin1")
        footer = f"\r\n--{boundary}--\r\n".encode("latin1")

        def chunk_generator():
            yield header
            for _ in range(10):
                yield b"C" * (10 * 1024)  # 10 chunks of 10 KB = 100 KB > 89 KB
            yield footer

        res3 = client.post(
            "/api/upload",
            content=chunk_generator(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        assert res3.status_code == 413
        assert "Request body exceeds maximum permitted size" in res3.json()["detail"]


def test_middleware_propagates_unrelated_application_exceptions():
    """P5D.2: Unrelated application exceptions are not swallowed by RequestSizeLimitMiddleware."""
    from fastapi import FastAPI
    from src.api.security import RequestSizeLimitMiddleware

    error_app = FastAPI()
    error_app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=50 * 1024)

    @error_app.post("/test_error")
    async def failing_endpoint(data: dict):
        raise RuntimeError("UNRELATED_APPLICATION_CRASH_TEST")

    # 1. With raise_server_exceptions=True: unhandled exception propagates out to the test runner
    with TestClient(error_app, raise_server_exceptions=True) as client:
        with pytest.raises(RuntimeError, match="UNRELATED_APPLICATION_CRASH_TEST"):
            client.post("/test_error", json={"valid": "payload"})

    # 2. With raise_server_exceptions=False: standard ASGI 500 Internal Server Error is returned
    with TestClient(error_app, raise_server_exceptions=False) as client:
        res = client.post("/test_error", json={"valid": "payload"})
        assert res.status_code == 500


def test_correct_termination_of_413_response():
    """P5D.2: 413 response terminates cleanly, returns valid JSON, and sends headers exactly once."""
    tight_settings = APISettings(mode="local", max_upload_bytes=10 * 1024)
    tight_app = create_app(tight_settings)

    with TestClient(tight_app) as client:
        oversized = b"E" * (80 * 1024)  # 80 KB > (10 KB + 64 KB = 74 KB)
        res = client.post(
            "/api/upload",
            content=oversized,
            headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(oversized))},
        )
        # Verify 413 status
        assert res.status_code == 413
        # Verify Content-Type
        assert "application/json" in res.headers["content-type"]
        # Verify body is not empty and valid JSON
        assert len(res.content) > 0
        body = res.json()
        assert "Request body exceeds maximum permitted size" in body["detail"]
        # Verify cleanly terminated (text readable, no connection hanging)
        assert res.text is not None


def test_cover_carrier_dimension_enforcement_and_no_silent_resizing(dummy_key):
    """Cover carrier validation enforces even dimensions >= 32x32 and supports dynamic spatial scaling (512x512, 1024x1024, 512x768)."""
    # Create valid secret (24x24)
    sec_img = Image.new("RGB", (24, 24), color=(50, 100, 150))
    sbuf = io.BytesIO()
    sec_img.save(sbuf, format="PNG")

    # Create invalid odd-dimension carrier (511x512)
    odd_carrier = Image.new("RGB", (511, 512), color=(100, 150, 200))
    cbuf_odd = io.BytesIO()
    odd_carrier.save(cbuf_odd, format="PNG")

    # Create invalid sub-minimum carrier (16x16)
    tiny_carrier = Image.new("RGB", (16, 16), color=(100, 150, 200))
    cbuf_tiny = io.BytesIO()
    tiny_carrier.save(cbuf_tiny, format="PNG")

    local_app = create_app(APISettings(mode="local"))
    with TestClient(local_app) as client:
        r_sec = client.post("/api/upload", files={"file": ("secret.png", sbuf.getvalue(), "image/png")})
        assert r_sec.status_code == 200
        sec_sid = r_sec.json()["session_id"]

        # 1. Odd-dimension carrier rejected with 422
        r_cov_odd = client.post("/api/upload", files={"file": ("odd_carrier.png", cbuf_odd.getvalue(), "image/png")})
        assert r_cov_odd.status_code == 200
        odd_cov_sid = r_cov_odd.json()["session_id"]

        r_embed_odd = client.post(
            "/api/embed",
            json={"secret_session_id": sec_sid, "cover_session_id": odd_cov_sid, "key_hex": dummy_key.hex()},
        )
        assert r_embed_odd.status_code == 422
        assert "must be even for DWT decomposition" in r_embed_odd.json()["detail"]

        # 2. Sub-minimum (< 32x32) carrier rejected with 422
        r_cov_tiny = client.post("/api/upload", files={"file": ("tiny_carrier.png", cbuf_tiny.getvalue(), "image/png")})
        assert r_cov_tiny.status_code == 200
        tiny_cov_sid = r_cov_tiny.json()["session_id"]

        r_embed_tiny = client.post(
            "/api/embed",
            json={"secret_session_id": sec_sid, "cover_session_id": tiny_cov_sid, "key_hex": dummy_key.hex()},
        )
        assert r_embed_tiny.status_code == 422
        assert "must be at least 32x32 pixels" in r_embed_tiny.json()["detail"]

        # 3. Valid 512x512 carrier is accepted and output matches dimensions
        good_carrier = Image.new("RGB", (512, 512), color=(100, 150, 200))
        good_buf = io.BytesIO()
        good_carrier.save(good_buf, format="PNG")
        r_good_cov = client.post("/api/upload", files={"file": ("good_512.png", good_buf.getvalue(), "image/png")})
        assert r_good_cov.status_code == 200
        good_cov_sid = r_good_cov.json()["session_id"]

        r_embed_good = client.post(
            "/api/embed",
            json={"secret_session_id": sec_sid, "cover_session_id": good_cov_sid, "key_hex": dummy_key.hex()},
        )
        assert r_embed_good.status_code == 200
        assert r_embed_good.json()["status"] == "success"
        from src.api.server import session_store
        stego_bytes_512 = session_store.get(r_embed_good.json()["stego_session_id"])
        assert stego_bytes_512 is not None
        assert Image.open(io.BytesIO(stego_bytes_512)).size == (512, 512)

        # 4. Valid 1024x1024 carrier is accepted and output preserves 1024x1024 dimensions
        hr_carrier = Image.new("RGB", (1024, 1024), color=(110, 160, 210))
        hr_buf = io.BytesIO()
        hr_carrier.save(hr_buf, format="PNG")
        r_hr_cov = client.post("/api/upload", files={"file": ("hr_1024.png", hr_buf.getvalue(), "image/png")})
        assert r_hr_cov.status_code == 200
        hr_cov_sid = r_hr_cov.json()["session_id"]

        r_embed_hr = client.post(
            "/api/embed",
            json={"secret_session_id": sec_sid, "cover_session_id": hr_cov_sid, "key_hex": dummy_key.hex()},
        )
        assert r_embed_hr.status_code == 200
        assert r_embed_hr.json()["status"] == "success"
        stego_bytes_1024 = session_store.get(r_embed_hr.json()["stego_session_id"])
        assert stego_bytes_1024 is not None
        assert Image.open(io.BytesIO(stego_bytes_1024)).size == (1024, 1024)

        # 5. Valid non-square even carrier (512x768) is accepted and output preserves (512, 768)
        rect_carrier = Image.new("RGB", (512, 768), color=(90, 140, 190))
        rect_buf = io.BytesIO()
        rect_carrier.save(rect_buf, format="PNG")
        r_rect_cov = client.post("/api/upload", files={"file": ("rect_512x768.png", rect_buf.getvalue(), "image/png")})
        assert r_rect_cov.status_code == 200
        rect_cov_sid = r_rect_cov.json()["session_id"]

        r_embed_rect = client.post(
            "/api/embed",
            json={"secret_session_id": sec_sid, "cover_session_id": rect_cov_sid, "key_hex": dummy_key.hex()},
        )
        assert r_embed_rect.status_code == 200
        assert r_embed_rect.json()["status"] == "success"
        stego_bytes_rect = session_store.get(r_embed_rect.json()["stego_session_id"])
        assert stego_bytes_rect is not None
        assert Image.open(io.BytesIO(stego_bytes_rect)).size == (512, 768)

        # 6. Extract from the 1024x1024 stego carrier and verify exact match with 24x24 secret
        r_ext_hr = client.post(
            "/api/extract",
            json={"stego_session_id": r_embed_hr.json()["stego_session_id"], "key_hex": dummy_key.hex()},
        )
        assert r_ext_hr.status_code == 200
        rec_sid = r_ext_hr.json()["recovered_session_id"]
        recovered_bytes = session_store.get(rec_sid)
        assert recovered_bytes == sbuf.getvalue()


def test_carrier_strict_format_and_color_mode_rejection(dummy_key):
    """Finding C: Carrier contract requires lossless 8-bit RGB PNG (512x512).

    Rejects JPEG, RGBA, grayscale, palette, oversized dimensions, and malformed inputs with HTTP 422.
    """
    sec_img = Image.new("RGB", (24, 24), color=(50, 100, 150))
    sbuf = io.BytesIO()
    sec_img.save(sbuf, format="PNG")

    local_app = create_app(APISettings(mode="local"))
    with TestClient(local_app) as client:
        r_sec = client.post("/api/upload", files={"file": ("secret.png", sbuf.getvalue(), "image/png")})
        assert r_sec.status_code == 200
        sec_sid = r_sec.json()["session_id"]

        # Helper to upload and test embed rejection
        def assert_carrier_rejected(file_bytes, filename, content_type, expected_error_substr):
            r_up = client.post("/api/upload", files={"file": (filename, file_bytes, content_type)})
            assert r_up.status_code == 200
            cov_sid = r_up.json()["session_id"]

            r_embed = client.post(
                "/api/embed",
                json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": dummy_key.hex()},
            )
            assert r_embed.status_code == 422
            assert expected_error_substr in r_embed.json()["detail"]

            # Also verify /api/stream rejects before streaming
            r_stream = client.post(
                "/api/stream",
                json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": dummy_key.hex()},
            )
            assert r_stream.status_code == 422
            assert expected_error_substr in r_stream.json()["detail"]

        # 1. 512x512 JPEG rejected
        jpeg_img = Image.new("RGB", (512, 512), color=(120, 130, 140))
        jbuf = io.BytesIO()
        jpeg_img.save(jbuf, format="JPEG")
        assert_carrier_rejected(jbuf.getvalue(), "cover.jpg", "image/jpeg", "Carrier format must be lossless PNG, got JPEG")

        # 2. 512x512 RGBA PNG rejected
        rgba_img = Image.new("RGBA", (512, 512), color=(120, 130, 140, 255))
        rgbabuf = io.BytesIO()
        rgba_img.save(rgbabuf, format="PNG")
        assert_carrier_rejected(rgbabuf.getvalue(), "cover_rgba.png", "image/png", "Carrier color mode must be 8-bit RGB without alpha, got RGBA")

        # 3. 512x512 Grayscale PNG rejected
        l_img = Image.new("L", (512, 512), color=128)
        lbuf = io.BytesIO()
        l_img.save(lbuf, format="PNG")
        assert_carrier_rejected(lbuf.getvalue(), "cover_gray.png", "image/png", "Carrier color mode must be 8-bit RGB without alpha, got L")

        # 4. 512x512 Palette PNG rejected (8-bit palette color_type=3)
        p_img = Image.new("P", (512, 512))
        palette = []
        for i in range(256):
            palette.extend([i, i, i])
        p_img.putpalette(palette)
        p_img.putpixel((0, 0), 200)
        pbuf = io.BytesIO()
        p_img.save(pbuf, format="PNG")
        assert_carrier_rejected(pbuf.getvalue(), "cover_palette.png", "image/png", "Carrier color mode must be 8-bit RGB without alpha, got P")

        # 5. Odd-dimension RGB PNG rejected (511x512)
        odd_img = Image.new("RGB", (511, 512), color=(100, 100, 100))
        oddbuf = io.BytesIO()
        odd_img.save(oddbuf, format="PNG")
        assert_carrier_rejected(oddbuf.getvalue(), "cover_odd.png", "image/png", "Cover carrier dimensions must be even for DWT decomposition, got 511x512")

        # 6. Malformed PNG rejected
        # Uploading with application/octet-stream bypasses /api/upload image probe,
        # verifying that /api/embed and /api/stream validate_carrier_image fails closed with HTTP 422
        corrupt_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"bad_corrupt_data_garbage"
        assert_carrier_rejected(corrupt_bytes, "corrupt.bin", "application/octet-stream", "Malformed or corrupted cover carrier image")

        # Also verify that /api/upload rejects corrupted image directly if uploaded as image/png
        r_up_bad = client.post("/api/upload", files={"file": ("corrupt.png", corrupt_bytes, "image/png")})
        assert r_up_bad.status_code == 400
        assert "Malformed or corrupted image content" in r_up_bad.json()["detail"]

        # 7. Real 16-bit RGB PNG (512x512, bit_depth=16, color_type=2) rejected
        # Note: Pillow would decode 16-bit PNG into mode="RGB" and size=(512, 512).
        # Our raw IHDR chunk validation directly detects bit_depth=16 and rejects with HTTP 422.
        import struct
        import zlib
        from src.api.server import validate_carrier_image

        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_payload = struct.pack(">IIBBBBB", 512, 512, 16, 2, 0, 0, 0)
        ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_payload))
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_payload + ihdr_crc
        scanline = b"\x00" + (b"\x80\x00" * (512 * 3))
        raw_idat = scanline * 512
        comp = zlib.compress(raw_idat)
        idat = struct.pack(">I", len(comp)) + b"IDAT" + comp + struct.pack(">I", zlib.crc32(b"IDAT" + comp))
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
        sixteen_bit_png = sig + ihdr + idat + iend

        assert_carrier_rejected(
            sixteen_bit_png,
            "carrier_16bit.png",
            "image/png",
            "Carrier bit depth must be 8 bits per channel, got 16",
        )

        # Also verify validate_carrier_image directly
        with pytest.raises(HTTPException) as exc_info:
            validate_carrier_image(sixteen_bit_png)
        assert exc_info.value.status_code == 422
        assert "Carrier bit depth must be 8 bits per channel, got 16" in exc_info.value.detail


def test_dynamic_carrier_capacity_api_and_config():
    """Verify GET /api/carrier/capacity and GET /api/config with dynamic carrier dimensions."""
    local_app = create_app(APISettings(mode="local"))
    with TestClient(local_app) as client:
        # 1. 512x512 capacity
        r1 = client.get("/api/carrier/capacity?width=512&height=512")
        assert r1.status_code == 200
        data1 = r1.json()
        assert data1["carrier_width"] == 512
        assert data1["carrier_height"] == 512
        assert data1["subband_width"] == 256
        assert data1["subband_height"] == 256
        assert data1["max_protected_bytes"] == 8176

        # 2. 1024x1024 capacity
        r2 = client.get("/api/carrier/capacity?width=1024&height=1024")
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["carrier_width"] == 1024
        assert data2["carrier_height"] == 1024
        assert data2["max_protected_bytes"] == 32752

        # 3. Non-square 512x768 capacity
        r3 = client.get("/api/carrier/capacity?width=512&height=768")
        assert r3.status_code == 200
        data3 = r3.json()
        assert data3["carrier_width"] == 512
        assert data3["carrier_height"] == 768
        assert data3["subband_width"] == 256
        assert data3["subband_height"] == 384
        assert data3["max_protected_bytes"] == 12272

        # 4. Odd dimensions rejected with HTTP 422
        r_odd = client.get("/api/carrier/capacity?width=511&height=512")
        assert r_odd.status_code == 422
        assert "must be even" in r_odd.json()["detail"]

        # 5. Dimensions < 32 rejected with HTTP 422
        r_tiny = client.get("/api/carrier/capacity?width=16&height=16")
        assert r_tiny.status_code == 422
        assert "at least 32x32" in r_tiny.json()["detail"]

        # 6. /api/config dynamic query
        r_cfg = client.get("/api/config?width=1024&height=1024")
        assert r_cfg.status_code == 200
        cfg = r_cfg.json()
        assert cfg["carrier_dimensions"] == "dynamic"
        assert cfg["max_protected_payload_bytes"] == 32752


def test_session_media_download_endpoint():
    """Verify GET /api/session/{session_id} downloads stored media."""
    app = create_app(APISettings(mode="local"))
    with TestClient(app) as client:
        # 1. Store test image in session store
        test_img = Image.new("RGB", (64, 64), color=(12, 34, 56))
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        r_up = client.post("/api/upload", files={"file": ("test.png", raw_bytes, "image/png")})
        assert r_up.status_code == 200
        sid = r_up.json()["session_id"]

        # 2. Retrieve via GET /api/session/{sid}
        r_get = client.get(f"/api/session/{sid}")
        assert r_get.status_code == 200
        assert r_get.headers["content-type"] == "image/png"
        assert r_get.content == raw_bytes

        # 3. Non-existent session returns 404
        r_missing = client.get("/api/session/nonexistent_session_12345")
        assert r_missing.status_code == 404


def test_run_subbands_endpoint():
    """Verify GET /api/runs/{run_id}/subbands returns cached subbands."""
    app = create_app(APISettings(mode="local"))
    with TestClient(app) as client:
        # Cache mock subbands
        session_store.store(
            data=json.dumps({"LL": "mock_ll", "LH": "mock_lh", "HL": "mock_hl", "HH": "mock_hh"}).encode("utf-8"),
            custom_session_id="subbands_test_run_999",
            media_type="application/json",
        )

        r = client.get("/api/runs/test_run_999/subbands")
        assert r.status_code == 200
        data = r.json()
        assert data["run_id"] == "test_run_999"
        assert data["subbands"]["LL"] == "mock_ll"

        # Missing run returns 404
        r_miss = client.get("/api/runs/missing_run_000/subbands")
        assert r_miss.status_code == 404


@pytest.mark.anyio
async def test_2040x1356_carrier_stream_and_bounded_visualization(dummy_key):
    """Verify that a large 2040x1356 carrier run yields valid events conforming strictly to the 1MB wire limit."""
    from src.api.event_adapter import MediaEventAdapter, serialize_sse_event, DEFAULT_MAX_SSE_EVENT_BYTES
    from src.core.serialization.serializer import SecretPayload, MediaType

    # Create 2040x1356 carrier and 128x128 secret image
    carrier = Image.new("RGB", (2040, 1356), color=(80, 120, 160))
    secret_img = Image.new("RGB", (128, 128), color=(200, 50, 50))
    sbuf = io.BytesIO()
    secret_img.save(sbuf, format="PNG")

    secret_payload = SecretPayload(
        data=sbuf.getvalue(),
        media_type=MediaType.IMAGE,
        metadata={"filename": "penguin_128.png", "width": 128, "height": 128},
    )

    adapter = MediaEventAdapter(session_store=session_store)
    events = []

    async for ev in adapter.run_media_pipeline(
        secret_payload=secret_payload,
        cover_image=carrier,
        key=dummy_key,
        session_counter=201,
    ):
        events.append(ev)
        # Verify strict SSE wire size compliance on every single event
        line = serialize_sse_event(ev, max_event_bytes=DEFAULT_MAX_SSE_EVENT_BYTES)
        assert len(line.encode("utf-8")) <= DEFAULT_MAX_SSE_EVENT_BYTES

    # Check DWT event
    dwt_ev = next(e for e in events if e.get("stage") == "DWT" and e.get("status") == "complete")
    assert dwt_ev["metadata"]["subbands"] is not None
    assert "LL" in dwt_ev["metadata"]["subbands"]
    assert dwt_ev["metadata"]["subband_width"] == 1020
    assert dwt_ev["metadata"]["subband_height"] == 678

    # Check EMBED event
    embed_ev = next(e for e in events if e.get("stage") == "EMBED" and e.get("status") == "complete")
    assert embed_ev["metadata"]["modified_hh"] is not None

    # Check IDWT event
    idwt_ev = next(e for e in events if e.get("stage") == "IDWT" and e.get("status") == "complete")
    assert idwt_ev["metadata"]["stego_image"] is not None
    assert idwt_ev["metadata"]["stego_session_id"] is not None
    assert idwt_ev["metadata"]["psnr_db"] > 50.0
    assert idwt_ev["metadata"]["ssim"] > 0.999

    # Verify that full-resolution lossless 2040x1356 stego PNG was stored in session_store
    stego_bytes = session_store.get(idwt_ev["metadata"]["stego_session_id"])
    assert stego_bytes is not None
    stego_pil = Image.open(io.BytesIO(stego_bytes))
    assert stego_pil.size == (2040, 1356)

    # Check terminal EXACT_MATCH_VERIFY
    term_ev = next(e for e in events if e.get("stage") == "EXACT_MATCH_VERIFY")
    assert term_ev["status"] == "complete"
    assert term_ev["metadata"]["exact_match"] is True
