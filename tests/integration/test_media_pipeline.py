"""End-to-End Media-to-Media Integration Tests for PaladinMedia Phase 2 (T-019, T-020)."""

import os
import pytest
from PIL import Image

from src.core.media_pipeline import MediaStegoPipeline, MediaStegoResult, MediaRecoveryResult
from src.core.serialization.serializer import MediaType, SecretPayload
from src.carrier.capacity import PayloadCapacityExceededError


@pytest.fixture
def secret_image_fixture() -> bytes:
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "fixtures", "secret_small.png"
    )
    with open(fixture_path, "rb") as f:
        return f.read()


@pytest.fixture
def cover_image_fixture() -> Image.Image:
    # Use real dataset carrier if available, else generate 512x512 RGB
    carrier_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "MVP 0 Dataset", "0001x2.png"
    )
    if os.path.exists(carrier_path):
        return Image.open(carrier_path).convert("RGB")
    return Image.new("RGB", (512, 512), color=(128, 128, 128))


def test_t019_full_media_to_media_mvp(secret_image_fixture, cover_image_fixture):
    """T-019: Real small secret image embedded into real 512x512 cover image."""
    pipeline = MediaStegoPipeline()
    key = bytes.fromhex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")

    secret_payload = SecretPayload(
        data=secret_image_fixture,
        media_type=MediaType.IMAGE,
        metadata={"filename": "secret_small.png", "width": 24, "height": 24},
    )

    stego_res, rec_res = pipeline.run_full_roundtrip(
        secret_payload=secret_payload,
        cover_image=cover_image_fixture,
        key=key,
        session_counter=101,
    )

    # Validate stego result artifacts
    assert isinstance(stego_res, MediaStegoResult)
    assert stego_res.stego_image.size == (512, 512)
    assert len(stego_res.stego_png_bytes) > 0
    assert stego_res.capacity_report.fits is True
    assert stego_res.psnr_db > 50.0  # High fidelity stego carrier
    assert stego_res.ssim > 0.999
    assert set(stego_res.dwt_visualizations.keys()) == {"LL", "LH", "HL", "HH"}
    assert stego_res.modified_hh_image.size == (256, 256)

    # Validate recovery result
    assert isinstance(rec_res, MediaRecoveryResult)
    assert rec_res.exact_match is True
    assert rec_res.recovered_payload.media_type == MediaType.IMAGE
    assert rec_res.recovered_image is not None
    assert rec_res.recovered_image.size == (24, 24)


def test_t020_exact_byte_equality(secret_image_fixture, cover_image_fixture):
    """T-020: Verbatim byte equality between original and recovered secret."""
    pipeline = MediaStegoPipeline()
    key = bytes.fromhex("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210")
    test_aad = b"PaladinMedia-Phase2-VerificationContext"

    secret_payload = SecretPayload(
        data=secret_image_fixture,
        media_type=MediaType.IMAGE,
        metadata={"format": "PNG", "hash": "sha256_fixture_check"},
    )

    stego_res = pipeline.protect_and_embed(
        secret_payload=secret_payload,
        cover_image=cover_image_fixture,
        key=key,
        aad=test_aad,
        session_counter=555,
    )

    rec_res = pipeline.extract_and_unprotect(
        stego_carrier=stego_res.stego_png_bytes,
        key=key,
        aad=test_aad,
        reference_payload=secret_payload,
    )

    # Assert exact byte match
    assert rec_res.exact_match is True
    assert rec_res.recovered_payload.data == secret_payload.data
    assert rec_res.recovered_payload.media_type == secret_payload.media_type
    assert rec_res.recovered_payload.metadata == secret_payload.metadata


def test_capacity_exceeded_hard_gate(cover_image_fixture):
    """Negative test: Large payload exceeding capacity is rejected before embedding."""
    pipeline = MediaStegoPipeline()
    key = bytes(32)

    # Create oversized payload (e.g. 10 KB > 7,088 B)
    large_data = b"X" * 10000
    oversized_payload = SecretPayload(data=large_data, media_type=MediaType.RAW_BYTES)

    with pytest.raises(PayloadCapacityExceededError, match="PAYLOAD_CAPACITY_EXCEEDED"):
        pipeline.protect_and_embed(
            secret_payload=oversized_payload,
            cover_image=cover_image_fixture,
            key=key,
        )
