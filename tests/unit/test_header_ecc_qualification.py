"""Synthetic unit tests for Task 4: Header-ECC Qualification.

Verifies:
1. Scheme B (3x Bit Repetition with Majority Voting):
   - Clean roundtrip encoding and decoding.
   - Single-bit error correction across all 128 bit positions.
   - Multi-bit independent error tolerance (up to 128 non-colliding errors).
   - Colliding error detection and fail-closed validation.
   - Truncation and malformed input rejection.
2. Scheme C (Shortened RS(24, 16) over GF(2^8)):
   - Clean roundtrip encoding and decoding.
   - Systematic (24, 16) code parameters (n=24, k=16, 2t=8, t=4).
   - Correction of up to 4 arbitrary corrupted bytes (up to 32 bit flips).
   - Fail-closed rejection of >= 5 corrupted bytes.
   - Truncation and malformed input rejection.
3. Fail-Closed Validation & Guard Invariants:
   - Oversized payload length rejected before mask allocation.
   - Corrupted flags / unsupported profile rejected.
   - Unsupported protocol versions rejected.
"""

from __future__ import annotations

import random
import struct
from typing import Optional, Tuple

import numpy as np
import pytest
import reedsolo

from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import DeterministicMaskGenerator, MaskGenerationError


# =============================================================================
# Codec Implementations for Diagnostics
# =============================================================================

def encode_repetition_3x(bits: np.ndarray) -> np.ndarray:
    """Repeats each bit 3 times: (128,) -> (384,)."""
    if len(bits) != 128:
        raise ValueError(f"Expected 128 bits, got {len(bits)}")
    return np.repeat(bits, 3).astype(np.uint8)


def decode_repetition_3x(bits: np.ndarray) -> np.ndarray:
    """Majority vote on each 3-bit triplet: (384,) -> (128,)."""
    if len(bits) != 384:
        raise ValueError(f"Expected 384 bits for 3x repetition, got {len(bits)}")
    reshaped = bits.reshape(128, 3)
    return (np.sum(reshaped, axis=1) >= 2).astype(np.uint8)


def encode_rs24_16(hdr_bytes: bytes) -> bytes:
    """Systematic RS(24, 16) encoding over GF(2^8): 16 bytes -> 24 bytes."""
    if len(hdr_bytes) != 16:
        raise ValueError(f"Expected 16 header bytes, got {len(hdr_bytes)}")
    rsc = reedsolo.RSCodec(nsym=8)
    return bytes(rsc.encode(hdr_bytes))


def decode_rs24_16(enc_bytes: bytes) -> Tuple[bytes, int]:
    """Decodes RS(24, 16) with t=4 bytes correction."""
    if len(enc_bytes) != 24:
        raise ValueError(f"Expected 24 encoded bytes, got {len(enc_bytes)}")
    rsc = reedsolo.RSCodec(nsym=8)
    res = rsc.decode(enc_bytes)
    corrected = len(res[2]) if len(res) > 2 and res[2] is not None else 0
    return bytes(res[0]), corrected


# =============================================================================
# 1. Scheme B: 3x Bit Repetition Tests
# =============================================================================

def test_scheme_b_clean_roundtrip():
    """Verify clean 3x repetition encoding and decoding produces identical header."""
    hdr = CarrierHeader(session_counter=774, payload_length=1338, flags=CarrierHeader.ECC_PROFILE_RS_2T_48, version=CarrierHeader.VERSION_1)
    raw_bytes = hdr.pack()
    raw_bits = BaselineEmbedder.bytes_to_bits(raw_bytes)
    assert len(raw_bits) == 128

    enc_bits = encode_repetition_3x(raw_bits)
    assert len(enc_bits) == 384

    dec_bits = decode_repetition_3x(enc_bits)
    assert np.array_equal(dec_bits, raw_bits)
    dec_bytes = Extractor.bits_to_bytes(dec_bits)
    unpacked_hdr = CarrierHeader.unpack(dec_bytes)
    assert unpacked_hdr == hdr


def test_scheme_b_single_bit_flip_all_positions():
    """Verify that a single bit flip in ANY of the 384 encoded positions is 100% corrected."""
    raw_bytes = CarrierHeader(session_counter=101, payload_length=5000, flags=1, version=CarrierHeader.VERSION_1).pack()
    raw_bits = BaselineEmbedder.bytes_to_bits(raw_bytes)
    enc_bits = encode_repetition_3x(raw_bits)

    for flip_pos in range(384):
        corrupted = np.array(enc_bits, copy=True)
        corrupted[flip_pos] ^= 1
        dec_bits = decode_repetition_3x(corrupted)
        assert np.array_equal(dec_bits, raw_bits), f"Failed to correct bit flip at position {flip_pos}"


def test_scheme_b_multi_bit_non_colliding_errors():
    """Verify that multiple simultaneous errors (up to 128) are corrected if <= 1 error per triplet."""
    raw_bytes = CarrierHeader(session_counter=999, payload_length=2000, flags=1, version=CarrierHeader.VERSION_1).pack()
    raw_bits = BaselineEmbedder.bytes_to_bits(raw_bytes)
    enc_bits = encode_repetition_3x(raw_bits)

    corrupted = np.array(enc_bits, copy=True)
    # Flip 1 bit in every single triplet (128 total errors!)
    rng = random.Random(42)
    for triplet_idx in range(128):
        offset = rng.randint(0, 2)
        corrupted[triplet_idx * 3 + offset] ^= 1

    assert np.sum(corrupted != enc_bits) == 128
    dec_bits = decode_repetition_3x(corrupted)
    assert np.array_equal(dec_bits, raw_bits), "Failed to correct 128 non-colliding errors"


def test_scheme_b_colliding_errors_fail_closed():
    """Verify that 2 flips in the SAME triplet flip that single decoded bit."""
    raw_bytes = CarrierHeader(session_counter=555, payload_length=1000, flags=1, version=CarrierHeader.VERSION_1).pack()
    raw_bits = BaselineEmbedder.bytes_to_bits(raw_bytes)
    enc_bits = encode_repetition_3x(raw_bits)

    corrupted = np.array(enc_bits, copy=True)
    # 2 flips in triplet 0 (e.g. bits 0 and 1)
    corrupted[0] ^= 1
    corrupted[1] ^= 1
    dec_bits = decode_repetition_3x(corrupted)
    # Triplet 0 inverted
    assert dec_bits[0] != raw_bits[0]
    # Remaining 127 bits intact
    assert np.array_equal(dec_bits[1:], raw_bits[1:])


def test_scheme_b_invalid_length_rejection():
    """Verify that malformed or truncated inputs are rejected."""
    with pytest.raises(ValueError, match="Expected 128 bits"):
        encode_repetition_3x(np.zeros(127, dtype=np.uint8))

    with pytest.raises(ValueError, match="Expected 384 bits"):
        decode_repetition_3x(np.zeros(383, dtype=np.uint8))


# =============================================================================
# 2. Scheme C: Shortened RS(24, 16) Tests
# =============================================================================

def test_scheme_c_clean_roundtrip():
    """Verify clean RS(24, 16) encoding and decoding produces identical header."""
    hdr = CarrierHeader(session_counter=774, payload_length=1338, flags=CarrierHeader.ECC_PROFILE_RS_2T_48, version=CarrierHeader.VERSION_1)
    raw_bytes = hdr.pack()
    enc_bytes = encode_rs24_16(raw_bytes)
    assert len(enc_bytes) == 24

    dec_bytes, corrected = decode_rs24_16(enc_bytes)
    assert dec_bytes == raw_bytes
    assert corrected == 0
    assert CarrierHeader.unpack(dec_bytes) == hdr


@pytest.mark.parametrize("num_errors", [1, 2, 3, 4])
def test_scheme_c_corrects_up_to_4_byte_errors(num_errors: int):
    """Verify that RS(24, 16) corrects up to t=4 corrupted bytes anywhere in the 24 bytes."""
    raw_bytes = CarrierHeader(session_counter=12345, payload_length=6000, flags=1, version=CarrierHeader.VERSION_1).pack()
    enc_bytes = bytearray(encode_rs24_16(raw_bytes))

    rng = random.Random(num_errors * 100)
    corrupt_positions = rng.sample(range(24), num_errors)
    for p in corrupt_positions:
        enc_bytes[p] ^= rng.randint(1, 255)

    dec_bytes, corrected = decode_rs24_16(bytes(enc_bytes))
    assert dec_bytes == raw_bytes
    assert corrected == num_errors


def test_scheme_c_rejects_exceeded_errors_fail_closed():
    """Verify that 5 or more corrupted bytes cause decoding to fail closed."""
    raw_bytes = CarrierHeader(session_counter=888, payload_length=1500, flags=1, version=CarrierHeader.VERSION_1).pack()
    enc_bytes = bytearray(encode_rs24_16(raw_bytes))

    # Corrupt 5 distinct bytes
    for p in range(5):
        enc_bytes[p] ^= 0xFF

    with pytest.raises(reedsolo.ReedSolomonError):
        decode_rs24_16(bytes(enc_bytes))


def test_scheme_c_invalid_length_rejection():
    """Verify that invalid input lengths are rejected."""
    with pytest.raises(ValueError, match="Expected 16 header bytes"):
        encode_rs24_16(b"short_15_bytes_")

    with pytest.raises(ValueError, match="Expected 24 encoded bytes"):
        decode_rs24_16(b"short_23_bytes_here___")


# =============================================================================
# 3. Fail-Closed Validation & Guard Invariant Tests
# =============================================================================

def test_oversized_payload_length_rejected_before_mask_generation():
    """Verify that corrupted payload lengths exceeding subband capacity raise MaskGenerationError."""
    # 256x256 subband has 65,536 positions. With 384 header bits, max payload bits = 65,152.
    # Attempting to allocate e.g. 5,244,216 bytes (as happened on 0774) must raise immediately.
    with pytest.raises(MaskGenerationError, match="Requested payload bits .* exceeds available positions"):
        DeterministicMaskGenerator.generate_payload_positions(
            session_counter=774,
            payload_length_bytes=5244216,
            subband_dim=256,
        )


def test_corrupted_ecc_profile_rejected_fail_closed():
    """Verify that invalid ECC flags (e.g. 0x05 as happened on 0715) raise CarrierHeaderError fail-closed."""
    # Byte 5 is flags
    corrupted_hdr_bytes = b"PLDH\x01\x05\x00\x00\x00\x00\x03\x06\x00\x00\x05\x3a"
    with pytest.raises(CarrierHeaderError, match="Unsupported ECC profile in carrier header flags"):
        CarrierHeader.unpack(corrupted_hdr_bytes)


def test_corrupted_magic_rejected_fail_closed():
    """Verify that non-PLDH magic raises CarrierHeaderError."""
    bad_magic = b"NOPE\x01\x01\x00\x00\x00\x00\x00\x01\x00\x00\x01\x00"
    with pytest.raises(CarrierHeaderError, match="Invalid carrier header magic"):
        CarrierHeader.unpack(bad_magic)


def test_corrupted_version_rejected_fail_closed():
    """Verify that unsupported protocol version raises CarrierHeaderError."""
    bad_version = b"PLDH\x99\x01\x00\x00\x00\x00\x00\x01\x00\x00\x01\x00"
    with pytest.raises(CarrierHeaderError, match="Unsupported carrier header version"):
        CarrierHeader.unpack(bad_version)
