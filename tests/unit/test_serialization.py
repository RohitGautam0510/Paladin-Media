"""Unit tests for SecretSerializer (PDM-S1-005, T-001).

Tests canonical binary wire framing, deterministic JSON metadata serialization,
boundary conditions (0-byte, 1-byte, ~7KB capacity), and negative validation
(corrupted magic, unsupported version, truncated payload, CRC-32 mismatch).
"""

import json
import struct
import zlib
import pytest

from src.core.exceptions import SerializationError
from tests.conftest import get_serialization_components
from tests.fixtures.vectors import (
    PAYLOAD_EMPTY,
    PAYLOAD_SINGLE_BYTE,
    PAYLOAD_TEXT,
    PAYLOAD_JSON,
    PAYLOAD_IMAGE_SNIPPET,
    PAYLOAD_CAPACITY,
    flip_bit,
    append_garbage,
    truncate_data,
)

# Dynamic import resolution
SecretPayload, MediaType, SecretSerializer = get_serialization_components()


@pytest.fixture(autouse=True)
def check_implementation():
    global SecretPayload, MediaType, SecretSerializer
    if SecretSerializer is None:
        SecretPayload, MediaType, SecretSerializer = get_serialization_components()
    if SecretSerializer is None:
        pytest.skip("SecretSerializer implementation not yet landed")


# =============================================================================
# Tier 1: Feature Coverage (Canonical Framing & Roundtrip)
# =============================================================================

def test_canonical_wire_framing_layout():
    """Verify exact wire framing layout and byte offsets.

    Layout:
      0x00..0x03: Magic b'PLDS' (4B)
      0x04: Version 0x01 (1B)
      0x05: Media Type (1B)
      0x06..0x07: Flags 0x0000 (2B, uint16 BE)
      0x08..0x09: Metadata Length M (2B, uint16 BE)
      0x0A..0x0A+M-1: Metadata UTF-8 JSON (MB)
      0x0A+M..0x0D+M: Payload Length L (4B, uint32 BE)
      0x0E+M..0x0D+M+L: Payload Data (LB)
      0x0E+M+L..0x11+M+L: CRC-32 Checksum (4B, uint32 BE)
    """
    metadata = {"key": "value"}
    data = b"Hello, Paladin!"
    payload = SecretPayload(data=data, media_type=MediaType.TEXT, metadata=metadata)

    wire = SecretSerializer.serialize(payload)

    # 1. Magic bytes
    assert wire[:4] == b"PLDS", f"Expected b'PLDS', got {wire[:4]}"

    # 2. Version
    assert wire[4] == 0x01

    # 3. Media Type
    assert wire[5] == int(MediaType.TEXT)

    # 4. Flags (2 bytes, big-endian)
    flags = struct.unpack(">H", wire[6:8])[0]
    assert flags == 0x0000

    # 5. Metadata Length (2 bytes, big-endian)
    meta_len = struct.unpack(">H", wire[8:10])[0]
    expected_meta_json = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert meta_len == len(expected_meta_json)

    # 6. Metadata JSON
    meta_bytes = wire[10:10 + meta_len]
    assert meta_bytes == expected_meta_json

    # 7. Payload Length (4 bytes, big-endian)
    payload_len_offset = 10 + meta_len
    payload_len = struct.unpack(">I", wire[payload_len_offset:payload_len_offset + 4])[0]
    assert payload_len == len(data)

    # 8. Payload Data
    payload_data_offset = payload_len_offset + 4
    extracted_data = wire[payload_data_offset:payload_data_offset + payload_len]
    assert extracted_data == data

    # 9. CRC-32 Checksum
    crc_offset = payload_data_offset + payload_len
    expected_crc = zlib.crc32(wire[:crc_offset]) & 0xFFFFFFFF
    actual_crc = struct.unpack(">I", wire[crc_offset:crc_offset + 4])[0]
    assert actual_crc == expected_crc

    # Total length
    assert len(wire) == crc_offset + 4


def test_roundtrip_all_media_types():
    """Verify lossless serialization and deserialization across all MediaType enum values."""
    media_types = [
        (MediaType.RAW_BYTES, b"\x00\x01\x02\x03\xff"),
        (MediaType.TEXT, b"Plain text payload with UTF-8: \xc3\xa9\xc3\xa0"),
        (MediaType.IMAGE, PAYLOAD_IMAGE_SNIPPET),
        (MediaType.AUDIO, b"\x52\x49\x46\x46" + b"\x00" * 32),  # RIFF header stub
        (MediaType.JSON, PAYLOAD_JSON),
    ]

    for media_type, data in media_types:
        meta = {"media_code": int(media_type), "tag": "roundtrip_test"}
        original = SecretPayload(data=data, media_type=media_type, metadata=meta)

        serialized = SecretSerializer.serialize(original)
        recovered = SecretSerializer.deserialize(serialized)

        assert recovered.data == original.data, f"Data mismatch for {media_type.name}"
        assert recovered.media_type == original.media_type, f"Media type mismatch for {media_type.name}"
        assert recovered.metadata == original.metadata, f"Metadata mismatch for {media_type.name}"


def test_deterministic_serialization():
    """Verify that serializing the same payload twice produces identical byte streams."""
    payload = SecretPayload(
        data=b"Deterministic test vector bytes",
        media_type=MediaType.TEXT,
        metadata={"alpha": 1, "beta": 2, "gamma": [1, 2, 3]},
    )
    wire1 = SecretSerializer.serialize(payload)
    wire2 = SecretSerializer.serialize(payload)
    assert wire1 == wire2


def test_deterministic_json_metadata_sorting():
    """Verify that metadata dictionary key insertion order does not affect serialized output."""
    meta1 = {"z": 100, "a": 200, "m": 300}
    meta2 = {"a": 200, "m": 300, "z": 100}

    p1 = SecretPayload(data=b"data", media_type=MediaType.RAW_BYTES, metadata=meta1)
    p2 = SecretPayload(data=b"data", media_type=MediaType.RAW_BYTES, metadata=meta2)

    wire1 = SecretSerializer.serialize(p1)
    wire2 = SecretSerializer.serialize(p2)
    assert wire1 == wire2


# =============================================================================
# Tier 2: Boundary & Corner Cases
# =============================================================================

def test_boundary_empty_payload_no_metadata():
    """Verify 0-byte payload with None metadata serializes and deserializes cleanly."""
    payload = SecretPayload(data=PAYLOAD_EMPTY, media_type=MediaType.RAW_BYTES, metadata=None)

    wire = SecretSerializer.serialize(payload)
    # Expected size: 4 (PLDS) + 1 (ver) + 1 (type) + 2 (flags) + 2 (meta_len=0) + 4 (data_len=0) + 4 (crc) = 18B
    assert len(wire) == 18

    recovered = SecretSerializer.deserialize(wire)
    assert recovered.data == b""
    assert recovered.media_type == MediaType.RAW_BYTES
    assert recovered.metadata is None or recovered.metadata == {}


def test_boundary_empty_payload_with_empty_dict_metadata():
    """Verify 0-byte payload with empty dict metadata."""
    payload = SecretPayload(data=b"", media_type=MediaType.TEXT, metadata={})
    wire = SecretSerializer.serialize(payload)
    recovered = SecretSerializer.deserialize(wire)
    assert recovered.data == b""
    assert recovered.media_type == MediaType.TEXT


def test_boundary_single_byte_payload():
    """Verify 1-byte payload boundary."""
    payload = SecretPayload(data=PAYLOAD_SINGLE_BYTE, media_type=MediaType.RAW_BYTES, metadata=None)
    wire = SecretSerializer.serialize(payload)
    assert len(wire) == 19  # 18B base framing + 1B data
    recovered = SecretSerializer.deserialize(wire)
    assert recovered.data == b"\x42"


def test_boundary_capacity_payload():
    """Verify maximum usable capacity payload (~7,088 bytes single sub-band)."""
    payload = SecretPayload(
        data=PAYLOAD_CAPACITY,
        media_type=MediaType.RAW_BYTES,
        metadata={"capacity_test": True, "size": len(PAYLOAD_CAPACITY)},
    )
    wire = SecretSerializer.serialize(payload)
    recovered = SecretSerializer.deserialize(wire)
    assert recovered.data == PAYLOAD_CAPACITY
    assert recovered.metadata["size"] == 7088


def test_boundary_large_payload_64kb():
    """Verify handling of 64KB multi-block payload."""
    large_data = bytes([i % 256 for i in range(65536)])
    payload = SecretPayload(data=large_data, media_type=MediaType.RAW_BYTES, metadata={"len": 65536})
    wire = SecretSerializer.serialize(payload)
    recovered = SecretSerializer.deserialize(wire)
    assert len(recovered.data) == 65536
    assert recovered.data == large_data


def test_boundary_unicode_and_nested_metadata():
    """Verify metadata containing Unicode text, emojis, deeply nested objects."""
    complex_meta = {
        "title": "Secret Directives \u2708 \u2694 \U0001F600",
        "nested": {
            "level1": {
                "level2": ["tagA", 42, 3.14159, True, None]
            }
        },
        "japanese": "\u6a5f\u5bc6\u6307\u4ee4",
    }
    payload = SecretPayload(data=b"Unicode test", media_type=MediaType.TEXT, metadata=complex_meta)
    wire = SecretSerializer.serialize(payload)
    recovered = SecretSerializer.deserialize(wire)
    assert recovered.metadata == complex_meta


# =============================================================================
# Tier 3: Negative & Error Cases (All must raise SerializationError)
# =============================================================================

def test_negative_invalid_magic_bytes():
    """Verify corrupted magic bytes trigger SerializationError."""
    payload = SecretPayload(data=b"Test", media_type=MediaType.TEXT)
    valid_wire = SecretSerializer.serialize(payload)

    # Replace b"PLDS" with b"XXXX"
    corrupt_wire = b"XXXX" + valid_wire[4:]
    with pytest.raises(SerializationError):
        SecretSerializer.deserialize(corrupt_wire)


def test_negative_unsupported_version():
    """Verify unsupported version byte triggers SerializationError."""
    payload = SecretPayload(data=b"Test", media_type=MediaType.TEXT)
    valid_wire = SecretSerializer.serialize(payload)

    # Replace version 0x01 with 0x02
    corrupt_wire = valid_wire[:4] + b"\x02" + valid_wire[5:]
    with pytest.raises(SerializationError):
        SecretSerializer.deserialize(corrupt_wire)


def test_negative_truncated_header():
    """Verify truncated header (< 16 bytes) triggers SerializationError."""
    for length in [0, 4, 8, 12, 15]:
        truncated = b"PLDS\x01\x00\x00\x00"[:length]
        with pytest.raises(SerializationError):
            SecretSerializer.deserialize(truncated)


def test_negative_truncated_payload():
    """Verify truncated payload body triggers SerializationError."""
    payload = SecretPayload(data=b"1234567890ABCDEF", media_type=MediaType.RAW_BYTES)
    valid_wire = SecretSerializer.serialize(payload)

    # Truncate by 1 byte, 5 bytes
    for cut in [1, 5]:
        truncated = truncate_data(valid_wire, cut)
        with pytest.raises(SerializationError):
            SecretSerializer.deserialize(truncated)


def test_negative_extraneous_trailing_garbage():
    """Verify extraneous trailing bytes trigger SerializationError."""
    payload = SecretPayload(data=b"Sample", media_type=MediaType.TEXT)
    valid_wire = SecretSerializer.serialize(payload)

    tampered = append_garbage(valid_wire, b"\x00\x01\x02")
    with pytest.raises(SerializationError):
        SecretSerializer.deserialize(tampered)


def test_negative_crc32_mismatch_payload_tampered():
    """Verify tampering with a payload byte causes CRC32 mismatch and raises SerializationError."""
    payload = SecretPayload(data=b"CriticalSecretData", media_type=MediaType.TEXT)
    valid_wire = SecretSerializer.serialize(payload)

    # Mutate a byte in the payload area
    # Payload is right before trailing 4-byte CRC
    tampered = flip_bit(valid_wire, byte_pos=len(valid_wire) - 6, bit_pos=0)
    with pytest.raises(SerializationError):
        SecretSerializer.deserialize(tampered)


def test_negative_crc32_mismatch_metadata_tampered():
    """Verify tampering with metadata bytes causes CRC32 mismatch and raises SerializationError."""
    payload = SecretPayload(
        data=b"Data",
        media_type=MediaType.TEXT,
        metadata={"sensitive": "flag"}
    )
    valid_wire = SecretSerializer.serialize(payload)

    # Flip bit in metadata region (offset 12)
    tampered = flip_bit(valid_wire, byte_pos=12, bit_pos=0)
    with pytest.raises(SerializationError):
        SecretSerializer.deserialize(tampered)


def test_negative_crc32_checksum_trailer_tampered():
    """Verify flipping a bit in the CRC-32 trailer itself raises SerializationError."""
    payload = SecretPayload(data=b"Valid Data", media_type=MediaType.RAW_BYTES)
    valid_wire = SecretSerializer.serialize(payload)

    # Flip bit in the very last byte (part of CRC32)
    tampered = flip_bit(valid_wire, byte_pos=len(valid_wire) - 1, bit_pos=0)
    with pytest.raises(SerializationError):
        SecretSerializer.deserialize(tampered)


# =============================================================================
# Tier 4: Real-World Workloads
# =============================================================================

def test_real_world_secret_text_directive():
    """Verify realistic secret intelligence directive payload roundtrip."""
    directive_text = PAYLOAD_TEXT
    metadata = {
        "classification": "TOP SECRET",
        "sender": "Agent-007",
        "timestamp_utc": "2026-09-22T17:42:00Z",
        "channel": "DWT-HH-Stego",
    }
    payload = SecretPayload(data=directive_text, media_type=MediaType.TEXT, metadata=metadata)

    wire = SecretSerializer.serialize(payload)
    recovered = SecretSerializer.deserialize(wire)

    assert recovered.data == directive_text
    assert recovered.media_type == MediaType.TEXT
    assert recovered.metadata == metadata


def test_real_world_image_pixel_data():
    """Verify synthetic 256-byte image pixel snippet roundtrip."""
    metadata = {"width": 16, "height": 16, "channels": 1, "format": "L"}
    payload = SecretPayload(data=PAYLOAD_IMAGE_SNIPPET, media_type=MediaType.IMAGE, metadata=metadata)

    wire = SecretSerializer.serialize(payload)
    recovered = SecretSerializer.deserialize(wire)

    assert recovered.data == PAYLOAD_IMAGE_SNIPPET
    assert recovered.media_type == MediaType.IMAGE
    assert recovered.metadata == metadata
