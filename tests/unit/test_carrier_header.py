"""Unit tests for 16-byte Carrier Header (T-013)."""

import pytest

from src.carrier.header import CarrierHeader, CarrierHeaderError


def test_t013_header_round_trip():
    """T-013: Verify carrier header packing, unpack roundtrip, and framing validation."""
    header = CarrierHeader(
        session_counter=42,
        payload_length=4096,
        version=1,
        flags=0,
        reserved=0,
    )
    raw = header.pack()
    assert len(raw) == 16
    assert raw[:4] == b"PLDH"

    unpacked = CarrierHeader.unpack(raw)
    assert unpacked.session_counter == 42
    assert unpacked.payload_length == 4096
    assert unpacked.version == 1
    assert unpacked.flags == 0
    assert unpacked.reserved == 0
    assert unpacked == header


def test_t013_header_validation_errors():
    """T-013 negative tests: Truncated header, invalid magic, invalid version."""
    # Truncated
    with pytest.raises(CarrierHeaderError, match="truncated"):
        CarrierHeader.unpack(b"PLDH\x01\x00")

    # Invalid Magic
    bad_magic = b"ABCD" + b"\x00" * 12
    with pytest.raises(CarrierHeaderError, match="Invalid carrier header magic"):
        CarrierHeader.unpack(bad_magic)

    # Invalid Version
    bad_version = b"PLDH\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x05"
    with pytest.raises(CarrierHeaderError, match="Unsupported carrier header version"):
        CarrierHeader.unpack(bad_version)
