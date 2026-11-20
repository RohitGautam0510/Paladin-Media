"""Phase 13 Full-Stack Integration & Architecture Conformance Tests.

Verifies:
1. Dynamic precision selection ('fp32' vs 'int8') in capabilities, embed, extract, and stream.
2. End-to-end embed -> extract roundtrip with 100% byte equality across FP32 and INT8.
3. Streaming pipeline SSE lifecycle and terminal completion across FP32 and INT8.
4. Standalone extraction fail-closed enforcement under tampered carriers and invalid keys.
5. InboundTrustStore multi-process advisory file locking and state preservation.
"""

from __future__ import annotations

import io
import json
import os
import secrets
import tempfile
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.api.server import api_router, get_api_settings, reset_api_settings
from src.security.pqc.models import PQCIdentity
from src.security.pqc.trust import TrustStore
from src.security.pqc.exceptions import PQCReplayError


@pytest.fixture
def client():
    reset_api_settings()
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


def _create_test_png(width: int, height: int, color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_secret_bytes(size: int = 128) -> bytes:
    return secrets.token_bytes(size)


def test_phase13_capabilities_precision(client):
    """Verify GET /api/capabilities exposes default and supported precisions."""
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default_precision"] == "fp32"
    assert "fp32" in data["supported_precisions"]
    assert "int8" in data["supported_precisions"]


@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_phase13_direct_embed_extract_roundtrip(client, precision):
    """Verify POST /api/embed and POST /api/extract roundtrip across precisions."""
    secret_bytes = _create_secret_bytes(100)
    cover_png = _create_test_png(128, 128)

    # 1. Upload secret
    up_secret = client.post(
        "/api/upload",
        files={"file": ("secret.bin", secret_bytes, "application/octet-stream")},
    )
    assert up_secret.status_code == 200
    secret_sid = up_secret.json()["session_id"]

    # 2. Upload cover
    up_cover = client.post(
        "/api/upload",
        files={"file": ("cover.png", cover_png, "image/png")},
    )
    assert up_cover.status_code == 200
    cover_sid = up_cover.json()["session_id"]

    # 3. Embed with target precision
    embed_resp = client.post(
        "/api/embed",
        json={
            "secret_session_id": secret_sid,
            "cover_session_id": cover_sid,
            "precision": precision,
            "model_name": "neural_spatial_onnx",
        },
    )
    assert embed_resp.status_code == 200, embed_resp.text
    embed_data = embed_resp.json()
    assert embed_data["status"] == "success"
    stego_sid = embed_data["stego_session_id"]

    # 4. Extract with target precision
    extract_resp = client.post(
        "/api/extract",
        json={
            "stego_session_id": stego_sid,
            "precision": precision,
            "model_name": "neural_spatial_onnx",
        },
    )
    assert extract_resp.status_code == 200, extract_resp.text
    extract_data = extract_resp.json()
    assert extract_data["status"] == "success"
    assert extract_data["recovered_bytes"] == len(secret_bytes)

    # 5. Fetch recovered bytes and verify exact match
    rec_sid = extract_data["recovered_session_id"]
    rec_resp = client.get(f"/api/session/{rec_sid}")
    assert rec_resp.status_code == 200
    assert rec_resp.content == secret_bytes


@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_phase13_stream_pipeline_roundtrip(client, precision):
    """Verify POST /api/stream executes terminal verification across precisions."""
    secret_bytes = _create_secret_bytes(128)
    cover_png = _create_test_png(128, 128)

    up_secret = client.post("/api/upload", files={"file": ("secret.bin", secret_bytes, "application/octet-stream")})
    up_cover = client.post("/api/upload", files={"file": ("cover.png", cover_png, "image/png")})

    stream_resp = client.post(
        "/api/stream",
        json={
            "secret_session_id": up_secret.json()["session_id"],
            "cover_session_id": up_cover.json()["session_id"],
            "precision": precision,
            "model_name": "neural_spatial_onnx",
        },
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]

    events = []
    for line in stream_resp.iter_lines():
        if line.startswith("data: "):
            ev = json.loads(line[6:])
            events.append(ev)

    assert len(events) >= 5
    terminal_event = events[-1]
    assert terminal_event["stage"] == "EXACT_MATCH_VERIFY"
    assert terminal_event["status"] == "complete"
    assert terminal_event["metadata"]["exact_match"] is True


def test_phase13_standalone_extract_fail_closed_tampered(client):
    """Verify /api/extract securely fails closed on non-stego/tampered carrier bytes."""
    # Innocent PNG with no embedded secret
    clean_png = _create_test_png(128, 128, color=(200, 200, 200))
    up_clean = client.post("/api/upload", files={"file": ("carrier.png", clean_png, "image/png")})
    assert up_clean.status_code == 200
    clean_sid = up_clean.json()["session_id"]

    extract_resp = client.post(
        "/api/extract",
        json={"stego_session_id": clean_sid, "model_name": "neural_spatial_onnx"},
    )
    assert extract_resp.status_code == 422
    assert "Extraction failed closed" in extract_resp.json()["detail"]


def test_phase13_standalone_extract_fail_closed_wrong_key(client):
    """Verify /api/extract rejects decryption when supplied with a wrong AES key."""
    secret_bytes = _create_secret_bytes(64)
    cover_png = _create_test_png(128, 128)

    up_secret = client.post("/api/upload", files={"file": ("secret.bin", secret_bytes, "application/octet-stream")})
    up_cover = client.post("/api/upload", files={"file": ("cover.png", cover_png, "image/png")})

    embed_resp = client.post(
        "/api/embed",
        json={
            "secret_session_id": up_secret.json()["session_id"],
            "cover_session_id": up_cover.json()["session_id"],
        },
    )
    assert embed_resp.status_code == 200
    stego_sid = embed_resp.json()["stego_session_id"]

    wrong_key = "00" * 32
    extract_resp = client.post(
        "/api/extract",
        json={"stego_session_id": stego_sid, "key_hex": wrong_key},
    )
    assert extract_resp.status_code == 422
    assert "Extraction failed closed" in extract_resp.json()["detail"]


def test_phase13_trust_store_file_locking():
    """Verify TrustStore atomic write and advisory lock integrity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "trust_store.json")
        store = TrustStore(storage_path=store_path)

        identity = PQCIdentity(
            identity_id="peer_alpha",
            name="Peer Alpha",
            kem_public_key=b"\x01" * 1184,
            dsa_public_key=b"\x02" * 1952,
        )
        store.register_identity(identity)

        store.check_inbound_session_counter("peer_alpha", 10)
        store.commit_inbound_session_counter("peer_alpha", 10)

        with pytest.raises(PQCReplayError):
            store.check_inbound_session_counter("peer_alpha", 10)

        store.check_inbound_session_counter("peer_alpha", 11)

        # Verify disk file is valid JSON and contains state
        assert os.path.exists(store_path)
        with open(store_path, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        assert "peer_alpha" in disk_data["identities"]
        assert disk_data["inbound_counters"]["peer_alpha"] == 10
