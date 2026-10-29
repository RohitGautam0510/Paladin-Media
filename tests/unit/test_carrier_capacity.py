"""Unit tests for Carrier Capacity Calculator and Hard Gate (T-015, T-016)."""

import pytest

from src.carrier.capacity import (
    CapacityCalculator,
    CapacityReport,
    PayloadCapacityExceededError,
)


def test_t015_capacity_calculation():
    """T-015: Verify accurate capacity accounting for 1 HH channel."""
    assert CapacityCalculator.RAW_CAPACITY_BYTES == 8192
    assert CapacityCalculator.CARRIER_HEADER_BYTES == 16
    assert CapacityCalculator.MAX_PROTECTED_BYTES == 8176

    # 512x512 carrier:
    # metadata_len=2 (standard reference minimal JSON dict): exactly 7,088 B
    usable_ref = CapacityCalculator.calculate_usable_secret_capacity()
    assert usable_ref == 7088

    # metadata_len=0 (unconstrained mathematical ceiling): exactly 7,090 B
    usable_zero_meta = CapacityCalculator.calculate_usable_secret_capacity(metadata_len=0)
    assert usable_zero_meta == 7090

    # 1024x1024 carrier (subband_dim=512):
    # Full-block floor baseline:
    usable_1024_ref = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=2, subband_dim=512, include_shortened_blocks=False
    )
    assert usable_1024_ref == 28496

    usable_1024_zero = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=0, subband_dim=512, include_shortened_blocks=False
    )
    assert usable_1024_zero == 28498

    # Actual shortened RS encoder capability:
    usable_1024_shortened_ref = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=2, subband_dim=512, include_shortened_blocks=True
    )
    assert usable_1024_shortened_ref == 28572

    usable_1024_shortened_zero = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=0, subband_dim=512, include_shortened_blocks=True
    )
    assert usable_1024_shortened_zero == 28574

    report = CapacityCalculator.evaluate(4000)
    assert isinstance(report, CapacityReport)
    assert report.fits is True
    assert report.status == "FITS"
    assert report.raw_capacity_bytes == 8192
    assert report.carrier_header_bytes == 16
    assert report.max_protected_bytes == 8176
    assert report.protected_payload_bytes == 4000
    assert report.remaining_capacity_bytes == (8192 - 4000 - 16)


def test_t016_capacity_exceeded_rejection():
    """T-016: Hard-gate check must strictly reject payloads exceeding usable capacity."""
    # 512x512: Exceeding 8,176 bytes
    report = CapacityCalculator.evaluate(8177)
    assert report.fits is False
    assert report.status == "EXCEEDS_CAPACITY"

    with pytest.raises(PayloadCapacityExceededError, match="PAYLOAD_CAPACITY_EXCEEDED") as exc_512:
        CapacityCalculator.check_capacity(8177)
    assert "8176 B" in str(exc_512.value)
    assert "7090 B" in str(exc_512.value)

    # 512x512: Exactly 8,176 bytes passes
    report_edge = CapacityCalculator.check_capacity(8176)
    assert report_edge.fits is True

    # 1024x1024: 32,752 bytes passes, 32,753 bytes fails
    report_1024 = CapacityCalculator.check_capacity(32752, subband_dim=512)
    assert report_1024.fits is True
    with pytest.raises(PayloadCapacityExceededError, match="PAYLOAD_CAPACITY_EXCEEDED") as exc_1024:
        CapacityCalculator.check_capacity(32753, subband_dim=512)
    assert "32752 B" in str(exc_1024.value)
    assert "28574 B" in str(exc_1024.value)

    # Massive payload (like the 791 KB example)
    with pytest.raises(PayloadCapacityExceededError, match="PAYLOAD_CAPACITY_EXCEEDED"):
        CapacityCalculator.check_capacity(791251)


def test_capacity_geometry_scaling_formula():
    """Verify that square carrier protected capacity matches (size * size // 32 - 16)."""
    for size, expected_subband, expected_protected in [(512, 256, 8176), (1024, 512, 32752)]:
        subband_dim = size // 2
        assert subband_dim == expected_subband

        # 1 HH channel has subband_dim * subband_dim bits
        raw_bits = subband_dim * subband_dim
        raw_bytes = raw_bits // 8
        assert raw_bytes == (size * size) // 32

        # Less 16-byte carrier header
        max_protected = raw_bytes - CapacityCalculator.CARRIER_HEADER_BYTES
        assert max_protected == (size * size) // 32 - 16
        assert max_protected == expected_protected

        # Evaluated via CapacityCalculator
        report = CapacityCalculator.evaluate(max_protected, subband_dim=subband_dim)
        assert report.fits is True
        assert report.max_protected_bytes == expected_protected
        assert report.raw_capacity_bytes == raw_bytes


def test_capacity_boundary_real_encoder_and_serializer():
    """Verify exact capacity boundaries under real SecretSerializer, AES-GCM, and ReedSolomonEncoder."""
    from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
    from src.security.crypto.provider import AESGCMProvider
    from src.security.ecc.encoder import ReedSolomonEncoder

    key = b"01234567890123456789012345678901"
    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder()

    # --- 512x512 Carrier Boundary Checks ---
    # 1. Zero metadata: exact ceiling is 7,090 bytes
    p_7090 = SecretPayload(data=b"A" * 7090, media_type=MediaType.RAW_BYTES, metadata=None)
    enc_7090 = crypto.encrypt(SecretSerializer.serialize(p_7090), key)
    prot_7090 = ecc.encode(enc_7090)
    assert len(prot_7090) == 8164 <= 8176
    CapacityCalculator.check_capacity(len(prot_7090), subband_dim=256)

    # 1 byte over: 7,091 bytes requires 33rd RS block (8,197 bytes > 8,176)
    p_7091 = SecretPayload(data=b"A" * 7091, media_type=MediaType.RAW_BYTES, metadata=None)
    enc_7091 = crypto.encrypt(SecretSerializer.serialize(p_7091), key)
    prot_7091 = ecc.encode(enc_7091)
    assert len(prot_7091) == 8197 > 8176
    with pytest.raises(PayloadCapacityExceededError):
        CapacityCalculator.check_capacity(len(prot_7091), subband_dim=256)

    # 2. Explicit metadata: M = 7 (e.g. {"k": 1} -> b'{"k":1}')
    meta_7 = {"k": 1}
    max_for_meta_7 = CapacityCalculator.calculate_usable_secret_capacity(metadata_len=7, subband_dim=256)
    assert max_for_meta_7 == 7083  # 7090 - 7 = 7083

    p_7083 = SecretPayload(data=b"A" * 7083, media_type=MediaType.RAW_BYTES, metadata=meta_7)
    enc_7083 = crypto.encrypt(SecretSerializer.serialize(p_7083), key)
    prot_7083 = ecc.encode(enc_7083)
    assert len(prot_7083) == 8164 <= 8176
    CapacityCalculator.check_capacity(len(prot_7083), subband_dim=256)

    # 1 byte over with M=7: 7,084 bytes
    p_7084 = SecretPayload(data=b"A" * 7084, media_type=MediaType.RAW_BYTES, metadata=meta_7)
    enc_7084 = crypto.encrypt(SecretSerializer.serialize(p_7084), key)
    prot_7084 = ecc.encode(enc_7084)
    assert len(prot_7084) == 8197 > 8176
    with pytest.raises(PayloadCapacityExceededError):
        CapacityCalculator.check_capacity(len(prot_7084), subband_dim=256)

    # --- 1024x1024 Carrier Boundary Checks ---
    # 3. Zero metadata under shortened RS encoder: exact ceiling is 28,574 bytes
    p_28574 = SecretPayload(data=b"B" * 28574, media_type=MediaType.RAW_BYTES, metadata=None)
    enc_28574 = crypto.encrypt(SecretSerializer.serialize(p_28574), key)
    prot_28574 = ecc.encode(enc_28574)
    assert len(prot_28574) == 32752 <= 32752
    CapacityCalculator.check_capacity(len(prot_28574), subband_dim=512)

    # 1 byte over: 28,575 bytes produces 32,753 bytes > 32,752
    p_28575 = SecretPayload(data=b"B" * 28575, media_type=MediaType.RAW_BYTES, metadata=None)
    enc_28575 = crypto.encrypt(SecretSerializer.serialize(p_28575), key)
    prot_28575 = ecc.encode(enc_28575)
    assert len(prot_28575) == 32753 > 32752
    with pytest.raises(PayloadCapacityExceededError):
        CapacityCalculator.check_capacity(len(prot_28575), subband_dim=512)

    # 4. Explicit metadata for high-load test: M = 48
    # Metadata: {"note":"High-Load 1024x1024 test","size":20000} (48 bytes)
    meta_48 = {"note": "High-Load 1024x1024 test", "size": 20000}
    max_for_meta_48 = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=48, subband_dim=512, include_shortened_blocks=True
    )
    assert max_for_meta_48 == 28526  # 28574 - 48 = 28526

    p_28526 = SecretPayload(data=b"C" * 28526, media_type=MediaType.RAW_BYTES, metadata=meta_48)
    enc_28526 = crypto.encrypt(SecretSerializer.serialize(p_28526), key)
    prot_28526 = ecc.encode(enc_28526)
    assert len(prot_28526) == 32752 <= 32752
    CapacityCalculator.check_capacity(len(prot_28526), subband_dim=512)

    # 1 byte over with M=48: 28,527 bytes
    p_28527 = SecretPayload(data=b"C" * 28527, media_type=MediaType.RAW_BYTES, metadata=meta_48)
    enc_28527 = crypto.encrypt(SecretSerializer.serialize(p_28527), key)
    prot_28527 = ecc.encode(enc_28527)
    assert len(prot_28527) == 32753 > 32752
    with pytest.raises(PayloadCapacityExceededError):
        CapacityCalculator.check_capacity(len(prot_28527), subband_dim=512)


def test_non_square_even_capacity_scaling():
    """Verify dynamic capacity calculation and hard gate for non-square even carriers (e.g. 512x768)."""
    # Carrier (512, 768) -> subband (256, 384)
    # raw_bits = 256 * 384 = 98,304 bits -> raw_bytes = 12,288 B
    # max_protected = 12,288 - 16 = 12,272 B
    subband_dim = (256, 384)
    report_edge = CapacityCalculator.check_capacity(12272, subband_dim=subband_dim)
    assert report_edge.fits is True
    assert report_edge.raw_capacity_bytes == 12288
    assert report_edge.max_protected_bytes == 12272
    assert report_edge.remaining_capacity_bytes == 0

    # 1 byte over: 12,273 B fails closed
    with pytest.raises(PayloadCapacityExceededError, match="PAYLOAD_CAPACITY_EXCEEDED") as exc:
        CapacityCalculator.check_capacity(12273, subband_dim=subband_dim)
    assert "12272 B" in str(exc.value)

    # Carrier (768, 512) -> subband (384, 256)
    # Symmetric area
    report_transposed = CapacityCalculator.check_capacity(12272, subband_dim=(384, 256))
    assert report_transposed.fits is True
    assert report_transposed.raw_capacity_bytes == 12288
    assert report_transposed.max_protected_bytes == 12272


def test_resolve_subband_dims_and_non_square_mask():
    """Verify coordinate resolution and non-square mask collision resistance."""
    from src.carrier.capacity import resolve_subband_dims
    from src.carrier.mask import DeterministicMaskGenerator

    assert resolve_subband_dims(256) == (256, 256)
    assert resolve_subband_dims((256, 384)) == (256, 384)
    assert resolve_subband_dims([384, 256]) == (384, 256)

    # Generate non-square mask
    mask = DeterministicMaskGenerator.generate_mask(
        session_counter=10,
        payload_length_bytes=1000,
        subband_dim=(256, 384),
    )
    assert len(mask.header_positions) == 128
    assert len(mask.payload_positions) == 8000

    # Check bounds
    for r, c in mask.header_positions:
        assert 0 <= r < 256
        assert 0 <= c < 384
    for r, c in mask.payload_positions:
        assert 0 <= r < 256
        assert 0 <= c < 384

    # Check zero overlap
    hdr_set = set(mask.header_positions)
    pld_set = set(mask.payload_positions)
    assert len(hdr_set & pld_set) == 0
