"""
Phase 11 Promotion Integration Test Suite — Wire Protocol v3-MR.

Validates end-to-end integration of Wire Protocol v3 Maximum Resilience (v3-MR, RS 2t=64)
into production source modules (`src/`) under strict governance constraints:
  - Production Wire Protocol v2 remains the active default.
  - v3-MR is accessible only via explicit opt-in and server-side feature gate (`PALADIN_ENABLE_V3_MR=true`).
  - Strict fail-closed error handling (HTTP 403 on disabled, HTTP 413 on capacity exceeded, HTTP 422 on unsupported).
  - Dual-anchor probing (Row 32 v3 primary, Row 0 v2 fallback).
  - Dual-valid header ambiguity conflict resolution fails closed with CarrierHeaderError.
  - Exact bit-for-bit payload recovery up to maximum usable capacity (6,066 B / 6,064 B).
  - Deterministic symbol interleaver invertibility and boundary bypasses.
"""

import io
import json
import math
import os
import secrets
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.carrier.capacity import CapacityCalculator, PayloadCapacityExceededError
from src.carrier.dwt import HaarDWT2D, DWTSubbands
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.interleaver import DeterministicSymbolInterleaver
from src.carrier.mask import DeterministicMaskGenerator
from src.core.serialization import SecretPayload, MediaType
from src.core.media_pipeline import MediaStegoPipeline
from src.api.security import APISettings
from src.api.server import create_app


@pytest.fixture
def clean_env():
    """Ensure clean environment state for feature gating tests."""
    old_val = os.environ.get("PALADIN_ENABLE_V3_MR")
    yield
    if old_val is not None:
        os.environ["PALADIN_ENABLE_V3_MR"] = old_val
    else:
        os.environ.pop("PALADIN_ENABLE_V3_MR", None)


@pytest.fixture
def synthetic_carrier():
    """Deterministic 512x512 RGB carrier image."""
    rng = np.random.RandomState(42)
    arr = rng.randint(40, 215, size=(512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def shared_key():
    return b"\x01" * 32


# -----------------------------------------------------------------------------
# 1. Deterministic Symbol Interleaver Invertibility & Boundary Bypasses
# -----------------------------------------------------------------------------

class TestDeterministicSymbolInterleaver:
    def test_interleaver_small_stream_bypass(self):
        """Payload streams <= 4 bytes bypass interleaving bit-for-bit unchanged."""
        for size in [0, 1, 2, 3, 4]:
            raw = secrets.token_bytes(size)
            interleaved = DeterministicSymbolInterleaver.interleave(raw, session_counter=100)
            assert interleaved == raw
            deinterleaved = DeterministicSymbolInterleaver.deinterleave(interleaved, session_counter=100)
            assert deinterleaved == raw

    def test_interleaver_various_lengths_invertibility(self):
        """Payload streams > 4 bytes must be deterministically permuted and strictly invertible."""
        for length in [5, 16, 70, 71, 100, 255, 1000, 6066, 8168]:
            data = os.urandom(length)
            interleaved = DeterministicSymbolInterleaver.interleave(data, session_counter=42)
            assert len(interleaved) == length
            if length >= 6:
                # When payload after 4-byte prefix is >= 2 bytes, it must permute
                assert interleaved != data
            recovered = DeterministicSymbolInterleaver.deinterleave(interleaved, session_counter=42)
            assert recovered == data

    def test_interleaver_stride_coprime_rule(self):
        """Stride must be >= 71 and coprime with length N."""
        for n in range(5, 500):
            stride = DeterministicSymbolInterleaver.get_coprime_step(n, preferred_step=71)
            assert stride >= 71
            assert math.gcd(stride, n) == 1


# -----------------------------------------------------------------------------
# 2. Carrier Header Relocation & Encoding for v3-MR
# -----------------------------------------------------------------------------

class TestCarrierHeaderV3MR:
    def test_header_v3_creation_and_fields(self):
        hdr = CarrierHeader.create(
            version=3,
            session_counter=12345,
            payload_length=4000,
            flags=CarrierHeader.FLAG_RELOCATED_HEADER | CarrierHeader.FLAG_SYMBOL_INTERLEAVED,
        )
        assert hdr.version == 3
        assert hdr.row_offset == CarrierHeader.HEADER_ROW_OFFSET_V3  # 32
        assert hdr.relocated_header is True
        assert hdr.symbol_interleaved is True
        assert hdr.header_size == 24

        packed = hdr.pack()
        assert len(packed) == 24

        # Gated unpack: allow_v3=True allows unpacking
        unpacked = CarrierHeader.unpack(packed, allow_v3=True)
        assert unpacked.version == 3
        assert unpacked.session_counter == 12345
        assert unpacked.payload_length == 4000
        assert unpacked.row_offset == 32
        assert unpacked.relocated_header is True
        assert unpacked.symbol_interleaved is True

    def test_header_v3_rejected_when_v3_disabled(self, monkeypatch):
        monkeypatch.delenv("PALADIN_ENABLE_V3_MR", raising=False)
        hdr = CarrierHeader.create(version=3, session_counter=1, payload_length=100)
        packed = hdr.pack()
        with pytest.raises(CarrierHeaderError, match="Wire Protocol v3 is disabled"):
            CarrierHeader.unpack(packed, allow_v3=False)


# -----------------------------------------------------------------------------
# 3. Mask Generation & Spatial Separation for v3-MR
# -----------------------------------------------------------------------------

class TestDeterministicMaskV3MR:
    def test_mask_excludes_row_32_for_version_3(self):
        mask_gen = DeterministicMaskGenerator()
        mask_v3 = mask_gen.generate_mask(
            session_counter=42,
            payload_length_bytes=1000,
            subband_dim=(256, 256),
            version=3,
        )
        # Check that header positions are all at Row 32
        assert all(r == 32 for r, c in mask_v3.header_positions)
        # Check that payload positions do NOT overlap with header positions
        header_set = set(mask_v3.header_positions)
        payload_set = set(mask_v3.payload_positions)
        assert len(header_set.intersection(payload_set)) == 0

    def test_mask_v2_preserves_bit_for_bit_compatibility(self):
        mask_gen = DeterministicMaskGenerator()
        mask_v2_explicit = mask_gen.generate_mask(
            session_counter=99,
            payload_length_bytes=500,
            subband_dim=(256, 256),
            version=2,
        )
        mask_v2_default = mask_gen.generate_mask(
            session_counter=99,
            payload_length_bytes=500,
            subband_dim=(256, 256),
        )
        assert mask_v2_explicit.header_positions == mask_v2_default.header_positions
        assert mask_v2_explicit.payload_positions == mask_v2_default.payload_positions


# -----------------------------------------------------------------------------
# 4. Usable Capacity Hard Gates for v3-MR
# -----------------------------------------------------------------------------

class TestCapacityHardGatesV3MR:
    def test_exact_capacity_values_512(self):
        """Verify exact capacity calculation for 512x512 carrier (256x256 subband)."""
        sub_dim = (256, 256)
        # v3-MR: 2t=64, 24B header
        cap_0b = CapacityCalculator.calculate_usable_secret_capacity(
            metadata_len=0, subband_dim=sub_dim, rs_parity=64, version=3
        )
        assert cap_0b == 6066

        cap_2b = CapacityCalculator.calculate_usable_secret_capacity(
            metadata_len=2, subband_dim=sub_dim, rs_parity=64, version=3
        )
        assert cap_2b == 6064

        # v2: 2t=48, 24B header
        cap_v2_0b = CapacityCalculator.calculate_usable_secret_capacity(
            metadata_len=0, subband_dim=sub_dim, rs_parity=48, version=2
        )
        assert cap_v2_0b == 6578

    def test_capacity_check_fails_closed_on_overflow(self):
        sub_dim = (256, 256)
        # 8168 protected bytes fits (8192 total - 24 header)
        report_ok = CapacityCalculator.check_capacity(
            protected_payload_bytes=8168, subband_dim=sub_dim, rs_parity=64, version=3
        )
        assert report_ok.fits is True

        # 8169 protected bytes exceeds capacity -> raises PayloadCapacityExceededError
        with pytest.raises(PayloadCapacityExceededError):
            CapacityCalculator.check_capacity(
                protected_payload_bytes=8169, subband_dim=sub_dim, rs_parity=64, version=3
            )


# -----------------------------------------------------------------------------
# 5. End-to-End MediaStegoPipeline (Protect -> Embed -> Extract -> Unprotect)
# -----------------------------------------------------------------------------

class TestMediaStegoPipelineV3MR:
    def test_v3_mr_deployment_factory(self, synthetic_carrier, shared_key):
        pipeline = MediaStegoPipeline.create_v3_mr_deployment()
        assert pipeline.enable_v3_mr is True

        payload_bytes = b"PaladinMedia Post-Quantum v3-MR End-to-End Verification Payload"
        secret = SecretPayload(data=payload_bytes, media_type=MediaType.TEXT, metadata={"author": "Phase 11"})

        result = pipeline.protect_and_embed(
            secret_payload=secret,
            cover_image=synthetic_carrier,
            key=shared_key,
            session_counter=101,
            version=3,
        )
        assert result.stego_image.size == synthetic_carrier.size

        recovery = pipeline.extract_and_unprotect(
            stego_carrier=result.stego_image,
            key=shared_key,
        )
        assert recovery.recovered_payload.data == payload_bytes
        assert recovery.recovered_payload.metadata["author"] == "Phase 11"
        assert recovery.header.row_offset == 32
        assert recovery.header.version == 3
        assert recovery.header.ecc_parity == 64

    def test_v3_mr_various_payload_sizes(self, synthetic_carrier, shared_key):
        """Test exact byte recovery across multiple payload sizes up to max capacity."""
        pipeline = MediaStegoPipeline.create_v3_mr_deployment()

        sizes_to_test = [
            1,      # Single byte
            10,     # Small
            256,    # Subband block size
            1024,   # 1 KiB
            6064,   # Max usable with 2B metadata ({})
        ]

        for sz in sizes_to_test:
            meta = {}
            data = os.urandom(sz)
            secret = SecretPayload(data=data, media_type=MediaType.RAW_BYTES, metadata=meta)
            result = pipeline.protect_and_embed(
                secret_payload=secret,
                cover_image=synthetic_carrier,
                key=shared_key,
                session_counter=200 + sz,
                version=3,
            )
            recovery = pipeline.extract_and_unprotect(
                stego_carrier=result.stego_image,
                key=shared_key,
            )
            assert recovery.recovered_payload.data == data, f"Mismatch for payload size {sz} B"

    def test_v3_mr_zero_payload_edge_case(self, synthetic_carrier, shared_key):
        """Test zero-payload data with valid metadata."""
        pipeline = MediaStegoPipeline.create_v3_mr_deployment()
        secret = SecretPayload(data=b"", media_type=MediaType.TEXT, metadata={"empty": "true"})
        result = pipeline.protect_and_embed(
            secret_payload=secret,
            cover_image=synthetic_carrier,
            key=shared_key,
            session_counter=555,
            version=3,
        )
        recovery = pipeline.extract_and_unprotect(
            stego_carrier=result.stego_image,
            key=shared_key,
        )
        assert recovery.recovered_payload.data == b""
        assert recovery.recovered_payload.metadata["empty"] == "true"

    def test_v3_mr_oversized_payload_rejected_pre_embed(self, synthetic_carrier, shared_key):
        """Payload exceeding usable capacity fails closed before embedding."""
        pipeline = MediaStegoPipeline.create_v3_mr_deployment()
        # With metadata={}, meta_len is 0, so maximum usable capacity is exactly 6066 B.
        # 6067 B payload must fail closed with PayloadCapacityExceededError before embedding.
        oversized = os.urandom(6067)
        secret = SecretPayload(data=oversized, media_type=MediaType.RAW_BYTES, metadata={})
        with pytest.raises(PayloadCapacityExceededError):
            pipeline.protect_and_embed(
                secret_payload=secret,
                cover_image=synthetic_carrier,
                key=shared_key,
                session_counter=999,
                version=3,
            )

    def test_deterministic_reproducibility(self, synthetic_carrier, shared_key):
        """Pipeline extraction deterministically and exactly recovers original secret."""
        pipeline = MediaStegoPipeline.create_v3_mr_deployment()
        secret = SecretPayload(data=b"Reproducible determinism test", media_type=MediaType.TEXT, metadata={"test": "repro"})

        res1 = pipeline.protect_and_embed(
            secret_payload=secret,
            cover_image=synthetic_carrier,
            key=shared_key,
            session_counter=777,
            version=3,
        )
        res2 = pipeline.protect_and_embed(
            secret_payload=secret,
            cover_image=synthetic_carrier,
            key=shared_key,
            session_counter=777,
            version=3,
        )
        rec1 = pipeline.extract_and_unprotect(res1.stego_image, key=shared_key, reference_payload=secret)
        rec2 = pipeline.extract_and_unprotect(res2.stego_image, key=shared_key, reference_payload=secret)
        assert rec1.recovered_payload.data == secret.data
        assert rec2.recovered_payload.data == secret.data
        assert rec1.exact_match is True
        assert rec2.exact_match is True


# -----------------------------------------------------------------------------
# 6. Dual-Anchor Probing & Ambiguity Conflict Resolution
# -----------------------------------------------------------------------------

class TestDualAnchorReceiver:
    def test_dual_anchor_extracts_v2_carrier(self, synthetic_carrier, shared_key):
        """Dual-anchor receiver extracts v2 carrier anchored at Row 0."""
        pipe_v2 = MediaStegoPipeline.create_qualified_deployment()
        secret_v2 = SecretPayload(data=b"Wire Protocol v2 Payload", media_type=MediaType.TEXT, metadata={})
        res_v2 = pipe_v2.protect_and_embed(
            secret_payload=secret_v2,
            cover_image=synthetic_carrier,
            key=shared_key,
            session_counter=10,
            version=2,
        )

        # Extract using dual-anchor enabled pipeline
        pipe_dual = MediaStegoPipeline.create_v3_mr_deployment()
        recovery = pipe_dual.extract_and_unprotect(
            stego_carrier=res_v2.stego_image,
            key=shared_key,
        )
        assert recovery.recovered_payload.data == b"Wire Protocol v2 Payload"
        assert recovery.header.row_offset == 0
        assert recovery.header.version == 2

    def test_dual_anchor_extracts_v3_carrier(self, synthetic_carrier, shared_key):
        """Dual-anchor receiver extracts v3 carrier anchored at Row 32."""
        pipe_v3 = MediaStegoPipeline.create_v3_mr_deployment()
        secret_v3 = SecretPayload(data=b"Wire Protocol v3-MR Payload", media_type=MediaType.TEXT, metadata={})
        res_v3 = pipe_v3.protect_and_embed(
            secret_payload=secret_v3,
            cover_image=synthetic_carrier,
            key=shared_key,
            session_counter=20,
            version=3,
        )

        recovery = pipe_v3.extract_and_unprotect(
            stego_carrier=res_v3.stego_image,
            key=shared_key,
        )
        assert recovery.recovered_payload.data == b"Wire Protocol v3-MR Payload"
        assert recovery.header.row_offset == 32
        assert recovery.header.version == 3

    def test_production_v2_receiver_rejects_v3_carrier_fail_closed(self, synthetic_carrier, shared_key, clean_env):
        """Pure v2 receiver (without allow_v3) must reject v3 carrier fail-closed."""
        os.environ.pop("PALADIN_ENABLE_V3_MR", None)
        pipe_v3 = MediaStegoPipeline.create_v3_mr_deployment()
        secret = SecretPayload(data=b"Stego v3 Carrier", media_type=MediaType.TEXT, metadata={})
        res_v3 = pipe_v3.protect_and_embed(
            secret_payload=secret,
            cover_image=synthetic_carrier,
            key=shared_key,
            session_counter=30,
            version=3,
        )

        pipe_v2 = MediaStegoPipeline.create_qualified_deployment()
        assert pipe_v2.enable_v3_mr is False
        with pytest.raises(CarrierHeaderError):
            pipe_v2.extract_and_unprotect(
                stego_carrier=res_v3.stego_image,
                key=shared_key,
            )

    def test_dual_valid_header_ambiguity_conflict_fails_closed(self):
        """
        When both Row 32 and Row 0 decode valid headers with conflicting metadata,
        extraction MUST abort fail-closed with CarrierHeaderError.
        """
        from src.carrier.embedding import BaselineEmbedder
        arr = np.full((512, 512, 3), 128, dtype=np.uint8)

        # 1. Embed valid v2 header into Row 0
        hdr_v2 = CarrierHeader.create(version=2, session_counter=10, payload_length=100, flags=0)
        bits_v2 = BaselineEmbedder.bytes_to_bits(hdr_v2.pack())
        for col in range(192):
            r_px, c_px = 0, col * 2
            a = int(arr[r_px, c_px, 0])
            b = int(arr[r_px, c_px + 1, 0])
            c = int(arr[r_px + 1, c_px, 0])
            d = int(arr[r_px + 1, c_px + 1, 0])
            two_hh = a - b - c + d
            if (two_hh % 2) != bits_v2[col]:
                arr[r_px, c_px, 0] += 1

        # 2. Embed valid v3 header into Row 32 (conflicting counter and length)
        hdr_v3 = CarrierHeader.create(
            version=3,
            session_counter=999,
            payload_length=500,
            flags=CarrierHeader.FLAG_RELOCATED_HEADER,
        )
        bits_v3 = BaselineEmbedder.bytes_to_bits(hdr_v3.pack())
        for col in range(192):
            r_px, c_px = 32 * 2, col * 2
            a = int(arr[r_px, c_px, 0])
            b = int(arr[r_px, c_px + 1, 0])
            c = int(arr[r_px + 1, c_px, 0])
            d = int(arr[r_px + 1, c_px + 1, 0])
            two_hh = a - b - c + d
            if (two_hh % 2) != bits_v3[col]:
                arr[r_px, c_px, 0] += 1

        dual_stego = Image.fromarray(arr)

        # Verify dual-anchor extraction raises CarrierHeaderError with ambiguity message
        extractor = Extractor()
        with pytest.raises(CarrierHeaderError, match="Ambiguous carrier header detected"):
            extractor.extract(dual_stego, allow_v3=True)


# -----------------------------------------------------------------------------
# 7. Fast-API HTTP API Hard Gate Enforcement
# -----------------------------------------------------------------------------

class TestAPIHardGatesV3MR:
    def test_api_v3_mr_feature_gate_disabled_by_default(self, synthetic_carrier, clean_env):
        os.environ.pop("PALADIN_ENABLE_V3_MR", None)
        app = create_app(APISettings(mode="local", enable_v3_mr=False))
        client = TestClient(app)

        # 1. Capacity endpoint rejects v3 with 403
        r_cap = client.get("/api/carrier/capacity?width=512&height=512&protocol_version=3")
        assert r_cap.status_code == 403
        assert "disabled on this server" in r_cap.json()["detail"]

        # 2. Embed endpoint rejects v3_mr with 403
        cov_buf = io.BytesIO()
        synthetic_carrier.save(cov_buf, format="PNG")
        cov_bytes = cov_buf.getvalue()

        r_cov = client.post("/api/upload", files={"file": ("cover.png", cov_bytes, "image/png")})
        assert r_cov.status_code == 200
        cov_sid = r_cov.json()["session_id"]

        r_sec = client.post("/api/upload", files={"file": ("sec.bin", b"small payload", "application/octet-stream")})
        assert r_sec.status_code == 200
        sec_sid = r_sec.json()["session_id"]

        key_hex = "01" * 32
        r_embed = client.post(
            "/api/embed",
            json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": key_hex, "protocol": "v3_mr"},
        )
        assert r_embed.status_code == 403

        # 3. Stream endpoint rejects v3_mr with 403
        r_stream = client.post(
            "/api/stream",
            json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": key_hex, "protocol": "v3_mr"},
        )
        assert r_stream.status_code == 403

        # 4. Embed endpoint rejects unsupported v3_standard with HTTP 422
        r_unsupp = client.post(
            "/api/embed",
            json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": key_hex, "protocol": "v3_standard"},
        )
        assert r_unsupp.status_code == 422

    def test_api_v3_mr_enabled_capacity_and_embed(self, synthetic_carrier, clean_env):
        os.environ["PALADIN_ENABLE_V3_MR"] = "true"
        app = create_app(APISettings(mode="local", enable_v3_mr=True))
        client = TestClient(app)

        # 1. Capacity endpoint returns exact v3-MR capacity
        r_cap = client.get("/api/carrier/capacity?width=512&height=512&protocol_version=3")
        assert r_cap.status_code == 200
        data_cap = r_cap.json()
        assert data_cap["protocol_version"] == 3
        assert data_cap["raw_secret_capacity_bytes"] == 6066
        assert data_cap["reference_metadata_capacity_bytes"] == 6064
        assert data_cap["ecc_parity_symbols"] == 64

        # 2. Embed endpoint enforces capacity hard gate (6064 B max with metadata)
        cov_buf = io.BytesIO()
        synthetic_carrier.save(cov_buf, format="PNG")
        r_cov = client.post("/api/upload", files={"file": ("cover.png", cov_buf.getvalue(), "image/png")})
        cov_sid = r_cov.json()["session_id"]

        # Oversized payload (6065 bytes)
        r_sec_over = client.post("/api/upload", files={"file": ("oversized.bin", os.urandom(6065), "application/octet-stream")})
        sec_over_sid = r_sec_over.json()["session_id"]

        key_hex = "01" * 32
        r_embed_over = client.post(
            "/api/embed",
            json={"secret_session_id": sec_over_sid, "cover_session_id": cov_sid, "key_hex": key_hex, "protocol": "v3_mr"},
        )
        assert r_embed_over.status_code == 413
        assert "PAYLOAD_CAPACITY_EXCEEDED" in r_embed_over.json()["detail"]

        # Permitted payload (100 bytes) succeeds
        r_sec_ok = client.post("/api/upload", files={"file": ("sec.bin", os.urandom(100), "application/octet-stream")})
        sec_ok_sid = r_sec_ok.json()["session_id"]

        r_embed_ok = client.post(
            "/api/embed",
            json={"secret_session_id": sec_ok_sid, "cover_session_id": cov_sid, "key_hex": key_hex, "protocol": "v3_mr"},
        )
        assert r_embed_ok.status_code == 200
        stego_sid = r_embed_ok.json()["stego_session_id"]
        assert stego_sid is not None
