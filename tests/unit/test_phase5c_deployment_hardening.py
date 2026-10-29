"""Phase 5C: Deployment Hardening and Operational Safeguards Regression Tests.

Verifies:
1. Operational entry point selects ONNX Runtime and rs_2t_48.
2. Legacy default constructor still uses rs_2t_32.
3. Sender emits correct profile-identifying header flags (0x01 for rs_2t_48).
4. Fresh receiver instance adapts to the header without sender-side in-memory state.
5. Qualified lossless PNG roundtrip recovers exact payload.
6. JPEG input, when identifiable, is explicitly rejected as unsupported.
7. Unreadable or corrupted carrier fails closed without returning plaintext.
8. Payloads exceeding the selected profile's capacity are rejected before carrier modification.
9. Session reuse remains intact.
10. No sensitive payload or cryptographic key data appears in operational logs or errors.
11. Operational API server endpoints expose non-sensitive config and enforce qualified policies.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import struct
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.event_adapter import MediaEventAdapter
from src.api.server import app, get_deployment_configuration
from src.carrier.capacity import CapacityCalculator, PayloadCapacityExceededError
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.modulation import ParityModulationEngine
from src.core.exceptions import (
    AuthenticationError,
    ECCDecodeError,
    UnsupportedCarrierFormatError,
)
from src.core.media_pipeline import (
    MediaStegoPipeline,
    create_qualified_deployment_pipeline,
)
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.ecc.encoder import ReedSolomonEncoder


@pytest.fixture
def dummy_cover() -> np.ndarray:
    """Deterministic 512x512 RGB carrier array."""
    rng = np.random.RandomState(42)
    return rng.randint(40, 215, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def dummy_key() -> bytes:
    """32-byte AES-256 key."""
    return b"\x01" * 32


@pytest.fixture
def api_client() -> Generator[TestClient, None, None]:
    """TestClient for PaladinMedia API server."""
    with TestClient(app) as client:
        yield client


# =============================================================================
# Requirement 1: Operational Entry Point Selects ONNX Runtime and rs_2t_48
# =============================================================================

def test_operational_entry_point_selects_onnx_and_rs_2t_48():
    """R1: The qualified deployment factory and server entry point select ONNX and rs_2t_48."""
    # 1. Pipeline factory
    pipe = MediaStegoPipeline.create_qualified_deployment()
    assert pipe.security_pipeline.ecc_encoder.parity_symbols == 48
    assert getattr(pipe.modulation_engine, "backend", "") == "onnx"

    # 2. Module level convenience factory
    pipe_mod = create_qualified_deployment_pipeline()
    assert pipe_mod.security_pipeline.ecc_encoder.parity_symbols == 48
    assert getattr(pipe_mod.modulation_engine, "backend", "") == "onnx"

    # 3. Server deployment configuration view
    cfg = get_deployment_configuration()
    assert cfg["inference_backend"] == "onnx"
    assert cfg["ecc_profile"] == "rs_2t_48"
    assert cfg["ecc_parity_symbols"] == 48
    assert cfg["is_qualified_deployment"] is True
    assert "Lossless" in cfg["carrier_format_policy"]
    assert cfg["raw_secret_capacity_bytes"] == 6578
    assert cfg["reference_metadata_capacity_bytes"] == 6576


# =============================================================================
# Requirement 2: Legacy Default Constructor Still Uses rs_2t_32
# =============================================================================

def test_legacy_default_constructor_still_uses_rs_2t_32():
    """R2: Default constructor MediaStegoPipeline() retains legacy classical rs_2t_32 behavior."""
    legacy_pipe = MediaStegoPipeline()
    assert legacy_pipe.security_pipeline.ecc_encoder.parity_symbols == 32
    assert isinstance(legacy_pipe.modulation_engine, ParityModulationEngine)

    cap_info = legacy_pipe.get_capacity_limits(subband_dim=256)
    assert cap_info["ecc_profile"] == "rs_2t_32"
    assert cap_info["ecc_parity_symbols"] == 32
    assert cap_info["raw_secret_capacity_bytes"] == 7090
    assert cap_info["reference_metadata_capacity_bytes"] == 7088
    assert cap_info["is_qualified_deployment"] is False


# =============================================================================
# Requirement 3: Sender Emits Correct Profile-Identifying Header Flags
# =============================================================================

def test_sender_emits_correct_profile_identifying_header_flags(dummy_cover, dummy_key):
    """R3: Qualified deployment pipeline sets header flags byte to 0x01 (rs_2t_48)."""
    pipe = MediaStegoPipeline.create_qualified_deployment()
    payload = SecretPayload(media_type=MediaType.TEXT, data=b"Profile flag test payload", metadata={})

    result = pipe.protect_and_embed(payload, dummy_cover, dummy_key)

    assert result.header.flags == 0x01
    assert result.header.ecc_parity == 48
    assert result.carrier_format == "PNG"
    assert "Lossless" in result.output_format_policy


# =============================================================================
# Requirement 4: Fresh Receiver Adapts to Header Without In-Memory State
# =============================================================================

def test_fresh_receiver_adapts_to_header_without_in_memory_state(dummy_cover, dummy_key):
    """R4: Fresh receiver instance adapts to header without sender-side in-memory state.

    First extracts with reference_payload=None (the true independent receiver operational path).
    Verifies exact payload-byte recovery and that the ECC profile is derived from the carrier header.
    Then independently constructs expected reference payload to confirm exact byte and metadata equality.
    """
    sender = MediaStegoPipeline.create_qualified_deployment()
    payload = SecretPayload(
        media_type=MediaType.TEXT,
        data=b"Dynamic receiver adaptation test bytes 1234567890",
        metadata={"key": "val", "origin": "sender_side"},
    )

    result = sender.protect_and_embed(payload, dummy_cover, dummy_key)

    # Receiver is a fresh instance with constructor default (rs_2t_32) and NO shared in-memory state
    receiver = MediaStegoPipeline(backend="onnx")
    assert receiver.security_pipeline.ecc_encoder.parity_symbols == 32  # constructor default

    # 1. Operational path: extraction with reference_payload=None (independent receiver)
    recovery_independent = receiver.extract_and_unprotect(
        stego_carrier=result.stego_png_bytes,
        key=dummy_key,
        reference_payload=None,
    )

    # Verify recovery succeeded without reference_payload
    assert recovery_independent.recovered_payload is not None
    assert recovery_independent.recovered_payload.data == b"Dynamic receiver adaptation test bytes 1234567890"
    assert recovery_independent.recovered_payload.metadata == {"key": "val", "origin": "sender_side"}
    assert recovery_independent.recovered_payload.media_type == MediaType.TEXT
    # Verify ECC profile was dynamically derived from carrier header flags (0x01 -> rs_2t_48)
    assert recovery_independent.header.flags == 0x01
    assert recovery_independent.header.ecc_parity == 48
    assert recovery_independent.exact_match is False  # reference_payload was None, so exact_match is False

    # 2. Verify against an independently constructed reference payload (without accessing sender object)
    independent_reference = SecretPayload(
        media_type=MediaType.TEXT,
        data=b"Dynamic receiver adaptation test bytes 1234567890",
        metadata={"key": "val", "origin": "sender_side"},
    )
    assert recovery_independent.recovered_payload == independent_reference

    # 3. Optional comparison path when an independent reference is supplied to the receiver
    recovery_with_ref = receiver.extract_and_unprotect(
        stego_carrier=result.stego_png_bytes,
        key=dummy_key,
        reference_payload=independent_reference,
    )
    assert recovery_with_ref.exact_match is True


# =============================================================================
# Requirement 5: Qualified Lossless PNG Roundtrip Recovers Exact Payload
# =============================================================================

def test_qualified_lossless_png_roundtrip_recovers_exact_payload(dummy_cover, dummy_key):
    """R5: Full roundtrip over lossless PNG preserves exact byte payload for both text and image."""
    pipe = MediaStegoPipeline.create_qualified_deployment()

    # Text payload
    p_text = SecretPayload(
        media_type=MediaType.TEXT,
        data=b"Qualified ONNX rs_2t_48 end-to-end exact recovery verified." * 10,
        metadata={"category": "audit_test", "index": 42},
    )
    s_res, r_res = pipe.run_full_roundtrip(p_text, dummy_cover, dummy_key)
    assert r_res.exact_match is True
    assert r_res.recovered_payload.data == p_text.data

    # Binary media payload
    p_img = SecretPayload(
        media_type=MediaType.IMAGE,
        data=bytes(i % 256 for i in range(1500)),
        metadata={"filename": "test_pattern.bin"},
    )
    s_res_img, r_res_img = pipe.run_full_roundtrip(p_img, dummy_cover, dummy_key)
    assert r_res_img.exact_match is True
    assert r_res_img.recovered_payload.data == p_img.data


# =============================================================================
# Requirement 6: JPEG Input Explicitly Rejected as Unsupported
# =============================================================================

def test_jpeg_input_explicitly_rejected(dummy_cover, dummy_key):
    """R6: Identifiable JPEG carriers are rejected with an actionable UnsupportedCarrierFormatError."""
    pipe = MediaStegoPipeline.create_qualified_deployment()

    # Create dummy JPEG bytes
    pil_img = Image.fromarray(dummy_cover, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    jpeg_bytes = buf.getvalue()

    with pytest.raises(UnsupportedCarrierFormatError) as exc_info:
        pipe.extract_and_unprotect(stego_carrier=jpeg_bytes, key=dummy_key)
    assert "JPEG-transcoded carriers are not qualified" in str(exc_info.value)
    assert "PNG" in str(exc_info.value)

    # PIL Image with JPEG format
    buf.seek(0)
    jpeg_pil = Image.open(buf)
    with pytest.raises(UnsupportedCarrierFormatError) as exc_info_pil:
        pipe.extract_and_unprotect(stego_carrier=jpeg_pil, key=dummy_key)
    assert "JPEG-transcoded carriers are not qualified" in str(exc_info_pil.value)


# =============================================================================
# Requirement 7: Unreadable or Corrupted Carrier Fails Closed
# =============================================================================

def test_unreadable_or_corrupted_carrier_fails_closed(dummy_cover, dummy_key):
    """R7: Corrupted or tampered carrier fails closed with CarrierHeaderError and zero plaintext leak."""
    pipe = MediaStegoPipeline.create_qualified_deployment()
    payload = SecretPayload(media_type=MediaType.TEXT, data=b"Secret top-security payload bytes", metadata={})
    res = pipe.protect_and_embed(payload, dummy_cover, dummy_key)

    # Corrupt stego carrier array (unspecified format: raw array)
    stego_arr = np.array(Image.open(io.BytesIO(res.stego_png_bytes)), dtype=np.uint8)
    # Zero out top rows containing anchor header positions
    stego_arr[:32, :32] = 0

    with pytest.raises(CarrierHeaderError) as exc_info:
        pipe.extract_and_unprotect(stego_carrier=stego_arr, key=dummy_key)
    # Does NOT falsely claim JPEG; reports invalid/corrupted header
    err_str = str(exc_info.value)
    assert "Invalid carrier header magic" in err_str
    assert "lossless" in err_str.lower()
    assert payload.data not in err_str.encode("utf-8")


# =============================================================================
# Requirement 8: Exact Capacity-Boundary Regression & Fail-Closed Enforcement
# =============================================================================

def test_exact_capacity_boundary_end_to_end_recovery_and_rejection(dummy_cover, dummy_key):
    """R8: Deterministic exact capacity-boundary regression for rs_2t_48 on 512x512 carrier.

    Verifies:
    1. A payload at the documented 6,578-byte raw ceiling completes end-to-end embedding
       and exact recovery with a fresh receiver (reference_payload=None).
    2. exact_match is False when reference_payload=None, while payload recovery is 100% exact.
    3. A 6,579-byte payload is rejected before carrier modification.
    4. Production capacity calculators and embedding/recovery paths are exercised directly.
    5. Metadata semantics:
       - Capacity is calculated using actual serialized metadata byte length M.
       - Empty mapping metadata={} serializes to M=0 bytes (0 metadata bytes in wire framing).
       - Reference baseline with M=2 yields 6,576 bytes.
       - Actual non-empty mappings enforce ceiling 6,578 - M bytes:
         * 6-byte minimal non-empty metadata {"": 0} (M=6) enforces ceiling 6,572 B (6,572 B succeeds; 6,573 B rejected).
         * 7-byte metadata {"a": 1} (M=7) enforces ceiling 6,571 B (6,571 B succeeds; 6,572 B rejected).
    """
    pipe = MediaStegoPipeline.create_qualified_deployment()
    cap_info = pipe.get_capacity_limits(subband_dim=256)
    raw_limit = cap_info["raw_secret_capacity_bytes"]
    ref_limit = cap_info["reference_metadata_capacity_bytes"]
    assert raw_limit == 6578
    assert ref_limit == 6576

    cover_copy = dummy_cover.copy()

    # 1. Payload at exact documented raw limit (6,578 B, M=0 metadata)
    p_at_limit = SecretPayload(data=b"B" * raw_limit, metadata={})
    res_at_limit = pipe.protect_and_embed(p_at_limit, cover_copy, dummy_key)

    # Verify stego carrier differs from original cover (transform-domain embedding applied)
    stego_arr_at_limit = np.array(res_at_limit.stego_image)
    assert not np.array_equal(stego_arr_at_limit, dummy_cover)
    assert res_at_limit.psnr_db > 0.0

    # Verify fresh receiver extracts exact payload with reference_payload=None
    receiver = MediaStegoPipeline(backend="onnx")
    rec_at_limit = receiver.extract_and_unprotect(
        stego_carrier=res_at_limit.stego_png_bytes,
        key=dummy_key,
        reference_payload=None,
    )
    assert rec_at_limit.recovered_payload.data == b"B" * raw_limit
    assert len(rec_at_limit.recovered_payload.data) == raw_limit
    assert rec_at_limit.header.ecc_parity == 48
    # Exact match is False because reference_payload=None, but payload recovery succeeded
    assert rec_at_limit.exact_match is False

    # 2. Over-limit payload (6,579 B, M=0 metadata): must be rejected before carrier modification
    cover_for_over = dummy_cover.copy()
    p_over = SecretPayload(data=b"X" * (raw_limit + 1), metadata={})

    with pytest.raises(PayloadCapacityExceededError) as exc_info:
        pipe.protect_and_embed(p_over, cover_for_over, dummy_key)

    assert "PAYLOAD_CAPACITY_EXCEEDED" in str(exc_info.value)
    # Carrier pixels must remain bit-for-bit identical
    assert np.array_equal(cover_for_over, dummy_cover)

    # 3. Metadata capacity accounting semantics and wire representation
    # Empty metadata mapping ({}) serializes to M=0 on the wire (not 2 bytes)
    s_bytes_empty = SecretSerializer.serialize(SecretPayload(data=b"test", metadata={}))
    meta_len_field = struct.unpack(">H", s_bytes_empty[8:10])[0]
    assert meta_len_field == 0  # Proof: empty dict does not produce 2 bytes on wire

    # Reference capacity baseline (M=2 theoretical baseline) = 6,576 B
    calc_cap_ref = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=2, subband_dim=256, rs_parity=48
    )
    assert calc_cap_ref == 6576
    calc_cap_raw = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=0, subband_dim=256, rs_parity=48
    )
    assert calc_cap_raw == 6578

    # 4. End-to-end qualification of actual non-empty metadata boundary
    # Sample metadata: {"a": 1} serializes to b'{"a":1}' (M = 7 bytes)
    meta_sample = {"a": 1}
    meta_json = json.dumps(meta_sample, sort_keys=True, separators=(",", ":")).encode("utf-8")
    m_bytes = len(meta_json)
    assert m_bytes == 7

    usable_with_sample_meta = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=m_bytes, subband_dim=256, rs_parity=48
    )
    assert usable_with_sample_meta == 6578 - 7 == 6571

    # End-to-end at limit with actual metadata (6,571 B data + 7 B metadata)
    cover_meta = dummy_cover.copy()
    p_meta_at_limit = SecretPayload(data=b"M" * usable_with_sample_meta, metadata=meta_sample)
    res_meta = pipe.protect_and_embed(p_meta_at_limit, cover_meta, dummy_key)
    rec_meta = receiver.extract_and_unprotect(
        stego_carrier=res_meta.stego_png_bytes,
        key=dummy_key,
        reference_payload=None,
    )
    assert rec_meta.recovered_payload.data == b"M" * usable_with_sample_meta
    assert rec_meta.recovered_payload.metadata == meta_sample
    assert rec_meta.exact_match is False

    # Over limit with actual metadata (6,572 B data + 7 B metadata): rejected before modification
    cover_meta_over = dummy_cover.copy()
    p_meta_over = SecretPayload(data=b"M" * (usable_with_sample_meta + 1), metadata=meta_sample)
    with pytest.raises(PayloadCapacityExceededError) as exc_meta:
        pipe.protect_and_embed(p_meta_over, cover_meta_over, dummy_key)
    assert "PAYLOAD_CAPACITY_EXCEEDED" in str(exc_meta.value)
    assert np.array_equal(cover_meta_over, dummy_cover)

    # 5. Focused regression test for 6-byte minimal non-empty metadata case ({"": 0})
    # Proves capacity is calculated using actual serialized metadata byte length M
    meta_6b = {"": 0}
    meta_6b_json = json.dumps(meta_6b, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert len(meta_6b_json) == 6  # Compact JSON b'{"":0}' is 6 bytes
    usable_6b = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=6, subband_dim=256, rs_parity=48
    )
    assert usable_6b == 6578 - 6 == 6572

    # Verify actual production preflight accepts 6,572 B data with 6 B metadata
    cover_6b = dummy_cover.copy()
    p_6b_ok = SecretPayload(data=b"Z" * usable_6b, metadata=meta_6b)
    res_6b = pipe.protect_and_embed(p_6b_ok, cover_6b, dummy_key)
    rec_6b = receiver.extract_and_unprotect(
        stego_carrier=res_6b.stego_png_bytes,
        key=dummy_key,
        reference_payload=None,
    )
    assert rec_6b.recovered_payload.data == b"Z" * usable_6b
    assert rec_6b.recovered_payload.metadata == meta_6b
    assert rec_6b.exact_match is False

    # Verify actual production preflight rejects 6,573 B data (one above limit) before modification
    cover_6b_over = dummy_cover.copy()
    p_6b_over = SecretPayload(data=b"Z" * (usable_6b + 1), metadata=meta_6b)
    with pytest.raises(PayloadCapacityExceededError) as exc_6b:
        pipe.protect_and_embed(p_6b_over, cover_6b_over, dummy_key)
    assert "PAYLOAD_CAPACITY_EXCEEDED" in str(exc_6b.value)
    assert np.array_equal(cover_6b_over, dummy_cover)


# =============================================================================
# Requirement 9: ONNX Session Reuse Remains Intact
# =============================================================================

def test_onnx_session_reuse_remains_intact(dummy_cover, dummy_key):
    """R9: Multiple embeds and extractions reuse the same cached ONNX InferenceSession."""
    pipe = MediaStegoPipeline.create_qualified_deployment()

    sess_embed_1 = pipe.modulation_engine.onnx_sess_embed
    sess_extract_1 = pipe.modulation_engine.onnx_sess_extract
    assert sess_embed_1 is not None
    assert sess_extract_1 is not None

    p1 = SecretPayload(data=b"Session reuse trial 1", metadata={})
    res1 = pipe.protect_and_embed(p1, dummy_cover, dummy_key)
    pipe.extract_and_unprotect(res1.stego_png_bytes, dummy_key)

    p2 = SecretPayload(data=b"Session reuse trial 2", metadata={})
    res2 = pipe.protect_and_embed(p2, dummy_cover, dummy_key)
    pipe.extract_and_unprotect(res2.stego_png_bytes, dummy_key)

    # Identical session instances reused
    assert pipe.modulation_engine.onnx_sess_embed is sess_embed_1
    assert pipe.modulation_engine.onnx_sess_extract is sess_extract_1


# =============================================================================
# Requirement 10: No Sensitive Payload or Key Data in Logs or Errors
# =============================================================================

def test_no_sensitive_payload_or_key_data_in_logs_or_errors(dummy_cover, caplog):
    """R10: Error messages and log records never expose secret plaintexts or raw key material."""
    pipe = MediaStegoPipeline.create_qualified_deployment()
    secret_key = b"TOP_SECRET_CRYPTO_KEY_0123456789"  # exactly 32 bytes
    secret_text = b"SUPER_SENSITIVE_PLAINTEXT_PAYLOAD_CONTENT"
    payload = SecretPayload(data=secret_text, metadata={"confidential": True})

    with caplog.at_level(logging.DEBUG):
        # 1. Normal run
        res = pipe.protect_and_embed(payload, dummy_cover, secret_key)
        pipe.extract_and_unprotect(res.stego_png_bytes, secret_key)

        # 2. Tampered run (wrong key)
        wrong_key = b"WRONG_KEY_WRONG_KEY_012345678901"
        with pytest.raises(AuthenticationError) as auth_exc:
            pipe.extract_and_unprotect(res.stego_png_bytes, wrong_key)

        err_msg = str(auth_exc.value)
        assert secret_text.decode("utf-8") not in err_msg
        assert secret_key.decode("utf-8") not in err_msg

    # Verify captured logs
    all_logs = caplog.text
    assert secret_text.decode("utf-8") not in all_logs
    assert secret_key.decode("utf-8") not in all_logs


# =============================================================================
# Requirement 11: API Server Operational Endpoints & Observability
# =============================================================================

def test_api_server_endpoints(api_client, dummy_cover, dummy_key):
    """R11: Tests FastAPI operational endpoints (/api/config, /api/embed, /api/extract)."""
    # 1. Configuration endpoint
    r_cfg = api_client.get("/api/config")
    assert r_cfg.status_code == 200
    cfg = r_cfg.json()
    assert cfg["inference_backend"] == "onnx"
    assert cfg["ecc_profile"] == "rs_2t_48"
    assert cfg["is_qualified_deployment"] is True
    assert cfg["raw_secret_capacity_bytes"] == 6578

    # 2. Upload secret & cover
    r_up_sec = api_client.post(
        "/api/upload",
        files={"file": ("test_secret.txt", b"API endpoint operational secret", "text/plain")},
    )
    assert r_up_sec.status_code == 200
    sec_id = r_up_sec.json()["session_id"]

    pil_cov = Image.fromarray(dummy_cover, mode="RGB")
    cov_buf = io.BytesIO()
    pil_cov.save(cov_buf, format="PNG")
    cov_bytes = cov_buf.getvalue()

    r_up_cov = api_client.post(
        "/api/upload",
        files={"file": ("cover.png", cov_bytes, "image/png")},
    )
    assert r_up_cov.status_code == 200
    cov_id = r_up_cov.json()["session_id"]

    # 3. Operational Embed
    key_hex = dummy_key.hex()
    r_embed = api_client.post(
        "/api/embed",
        json={"secret_session_id": sec_id, "cover_session_id": cov_id, "key_hex": key_hex},
    )
    assert r_embed.status_code == 200
    stego_json = r_embed.json()
    assert stego_json["status"] == "success"
    assert stego_json["carrier_format"] == "PNG"
    assert stego_json["ecc_profile"] == "rs_2t_48"
    stego_id = stego_json["stego_session_id"]

    # 4. Operational Extract
    r_extract = api_client.post(
        "/api/extract",
        json={"stego_session_id": stego_id, "key_hex": key_hex},
    )
    assert r_extract.status_code == 200
    ext_json = r_extract.json()
    assert ext_json["status"] == "success"
    assert ext_json["recovered_bytes"] == len(b"API endpoint operational secret")
    assert ext_json["ecc_parity"] == 48

    # 5. Operational Extract with JPEG rejection
    jpeg_buf = io.BytesIO()
    pil_cov.save(jpeg_buf, format="JPEG")
    r_up_jpg = api_client.post(
        "/api/upload",
        files={"file": ("carrier.jpg", jpeg_buf.getvalue(), "image/jpeg")},
    )
    jpg_stego_id = r_up_jpg.json()["session_id"]

    r_ext_jpg = api_client.post(
        "/api/extract",
        json={"stego_session_id": jpg_stego_id, "key_hex": key_hex},
    )
    assert r_ext_jpg.status_code == 400
    assert "JPEG-transcoded carriers are not qualified" in r_ext_jpg.json()["detail"]
