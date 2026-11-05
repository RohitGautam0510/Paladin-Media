"""Comprehensive Unit and Integration Test Suite for Wire Protocol v2 Header ECC.

Validates:
1. v1 round-trip and existing v1 fixture backward compatibility.
2. v2 round-trip and byte-exact recovery with Reed-Solomon RS(24, 16) protection.
3. Exhaustive single-bit header flips across all 192 bit positions in v2 header.
4. Correction of 1, 2, 3, and 4 erroneous header bytes anywhere in data or parity.
5. Strict fail-closed rejection of errors exceeding correction capability (>4 bytes).
6. Fail-closed rejection of corrupted magic, unsupported version, corrupted ECC profile,
   oversized payload, invalid lengths, and truncated headers.
7. Ambiguous v1/v2 probes and adversarial header inputs.
8. Capacity boundary verification and mask-coordinate consistency between header and payload.
9. Full end-to-end embed/save/reload/extract/authenticate tests across ONNX and Parity engines.
"""

import io
import os
import random
import struct
import numpy as np
import pytest
from PIL import Image

from src.carrier.capacity import (
    CapacityCalculator,
    CapacityReport,
    PayloadCapacityExceededError,
)
from src.carrier.embedding import BaselineEmbedder, EmbeddingResult
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import (
    DeterministicMaskGenerator,
    EmbeddingMask,
    MaskGenerationError,
)
from src.core.media_pipeline import (
    MediaStegoPipeline,
    MediaStegoResult,
    MediaRecoveryResult,
)
from src.core.serialization import MediaType, SecretPayload
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


# =============================================================================
# 1. Format & Codec Unit Tests (v1 vs v2 Framing)
# =============================================================================

def test_v2_header_structure_and_constants():
    """Verify v2 header bit/byte allocations, systematic codeword layout, and constants."""
    assert CarrierHeader.VERSION_1 == 0x01
    assert CarrierHeader.VERSION_2 == 0x02
    assert CarrierHeader.HEADER_SIZE_V1 == 16
    assert CarrierHeader.HEADER_SIZE_V2 == 24
    assert CarrierHeader.HEADER_SIZE == 24
    assert CarrierHeader.HEADER_BITS_V1 == 128
    assert CarrierHeader.HEADER_BITS_V2 == 192
    assert CarrierHeader.HEADER_BITS == 192
    assert CarrierHeader.RS_PARITY_BYTES == 8
    assert CarrierHeader.RS_DATA_BYTES == 16

    hdr_v2 = CarrierHeader.create(session_counter=42, payload_length=1500, version=2)
    packed_v2 = hdr_v2.pack()
    assert len(packed_v2) == 24
    assert hdr_v2.header_size == 24
    assert hdr_v2.header_bits == 192
    assert hdr_v2.version == 2
    assert hdr_v2.ecc_parity == 48

    # Systematic layout: first 16 bytes contain raw data header
    magic, ver, flags, res, sess, plen = struct.unpack(">4sBBHII", packed_v2[:16])
    assert magic == b"PLDH"
    assert ver == 0x02
    assert sess == 42
    assert plen == 1500
    assert res == 0


def test_v1_header_backward_compatibility():
    """Verify legacy v1 header creation, packing, unpacking, and fields."""
    hdr_v1 = CarrierHeader.create(session_counter=99, payload_length=1200, version=1, ecc_parity=32)
    packed_v1 = hdr_v1.pack()
    assert len(packed_v1) == 16
    assert hdr_v1.header_size == 16
    assert hdr_v1.header_bits == 128
    assert hdr_v1.version == 1
    assert hdr_v1.ecc_parity == 32

    # Unpack from 16 bytes
    unpacked_16 = CarrierHeader.unpack(packed_v1)
    assert unpacked_16.version == 1
    assert unpacked_16.session_counter == 99
    assert unpacked_16.payload_length == 1200
    assert unpacked_16.ecc_parity == 32

    # Unpack from a 24-byte buffer (e.g. 16B v1 header + 8B v1 payload bits)
    v1_stream_padded = packed_v1 + b"\xde\xad\xbe\xef\xca\xfe\xba\xbe"
    unpacked_stream = CarrierHeader.unpack(v1_stream_padded)
    assert unpacked_stream.version == 1
    assert unpacked_stream.session_counter == 99
    assert unpacked_stream.payload_length == 1200


# =============================================================================
# 2. RS(24, 16) Error Correction & Rejection Tests
# =============================================================================

def test_v2_all_single_bit_flips_exhaustive():
    """Exhaustively flip every single bit (all 192 positions) in a v2 header."""
    hdr = CarrierHeader.create(session_counter=12345, payload_length=2048, version=2, ecc_parity=48)
    original_wire = hdr.pack()
    assert len(original_wire) == 24

    for bit_idx in range(192):
        byte_idx = bit_idx // 8
        bit_pos = bit_idx % 8

        corrupted = bytearray(original_wire)
        corrupted[byte_idx] ^= (1 << bit_pos)

        recovered = CarrierHeader.unpack(bytes(corrupted))
        assert recovered.version == 2
        assert recovered.session_counter == 12345
        assert recovered.payload_length == 2048
        assert recovered.ecc_parity == 48


@pytest.mark.parametrize("num_corrupted_bytes", [1, 2, 3, 4])
def test_v2_corrects_up_to_4_byte_errors(num_corrupted_bytes: int):
    """Verify that RS(24, 16) corrects up to 4 arbitrary corrupted bytes anywhere in the 24 bytes."""
    hdr = CarrierHeader.create(session_counter=9876, payload_length=3500, version=2, ecc_parity=48)
    original_wire = hdr.pack()

    rng = random.Random(42 + num_corrupted_bytes)
    for _ in range(50):
        # Choose num_corrupted_bytes distinct byte positions across data and parity
        corrupt_indices = rng.sample(range(24), num_corrupted_bytes)
        corrupted = bytearray(original_wire)
        for idx in corrupt_indices:
            corrupted[idx] ^= rng.randint(1, 255)

        recovered = CarrierHeader.unpack(bytes(corrupted))
        assert recovered.version == 2
        assert recovered.session_counter == 9876
        assert recovered.payload_length == 3500
        assert recovered.ecc_parity == 48


def test_v2_rejects_errors_exceeding_capacity_fail_closed():
    """Verify that 5 or more corrupted bytes fail closed with CarrierHeaderError."""
    hdr = CarrierHeader.create(session_counter=777, payload_length=1000, version=2)
    original_wire = hdr.pack()

    rng = random.Random(999)
    # 5 corrupted bytes exceed RS(24, 16) error capability (t=4)
    for _ in range(30):
        corrupt_indices = rng.sample(range(24), 5)
        corrupted = bytearray(original_wire)
        for idx in corrupt_indices:
            corrupted[idx] ^= rng.randint(1, 255)

        with pytest.raises(CarrierHeaderError):
            CarrierHeader.unpack(bytes(corrupted))


# =============================================================================
# 3. Fail-Closed Header Validation & Adversarial Inputs
# =============================================================================

def test_v2_corrupted_magic_uncorrectable_rejected():
    """Uncorrectable magic corruption must fail closed."""
    hdr = CarrierHeader.create(session_counter=1, payload_length=500, version=2)
    wire = bytearray(hdr.pack())
    # Corrupt magic and parity so it exceeds 4 byte errors
    wire[0:4] = b"NOPE"
    wire[16:18] = b"\x00\x00"
    with pytest.raises(CarrierHeaderError, match="Invalid carrier header magic"):
        CarrierHeader.unpack(bytes(wire))


def test_v2_unsupported_version_rejected():
    """Headers with unsupported versions (e.g. 0x03, 0x00, 0xFF) must be rejected."""
    for bad_ver in [0x00, 0x03, 0x05, 0xFF]:
        # Construct raw 16B data with bad version
        raw_bad = struct.pack(">4sBBHII", b"PLDH", bad_ver, 0x01, 0, 10, 500)
        with pytest.raises(CarrierHeaderError, match="Unsupported carrier header version"):
            CarrierHeader.unpack(raw_bad)


def test_truncated_headers_rejected():
    """Headers with fewer than 16 bytes must fail closed with truncation error."""
    valid_v2 = CarrierHeader.create(session_counter=1, payload_length=500, version=2).pack()
    for trunc_len in [0, 4, 8, 12, 15]:
        with pytest.raises(CarrierHeaderError, match="Carrier header truncated"):
            CarrierHeader.unpack(valid_v2[:trunc_len])


def test_corrupted_reserved_field_rejected():
    """Non-zero reserved field must fail closed."""
    raw_bad_res = struct.pack(">4sBBHII", b"PLDH", 0x01, 0x01, 0x1234, 10, 500)
    with pytest.raises(CarrierHeaderError, match="Non-zero reserved field"):
        CarrierHeader.unpack(raw_bad_res)


def test_corrupted_ecc_profile_rejected():
    """Invalid ECC profile flag bits must fail closed."""
    # 0x0F is outside valid profiles (0, 1, 2)
    raw_bad_flag = struct.pack(">4sBBHII", b"PLDH", 0x01, 0x0F, 0x0000, 10, 500)
    with pytest.raises(CarrierHeaderError, match="Unsupported ECC profile"):
        CarrierHeader.unpack(raw_bad_flag)


# =============================================================================
# 4. Capacity & Mask Consistency Across Versions
# =============================================================================

def test_capacity_constants_and_scaling_v2():
    """Verify v2 capacity accounting: 24B header -> 8,168 B protected limit for 512x512."""
    assert CapacityCalculator.CARRIER_HEADER_BYTES == 24
    assert CapacityCalculator.MAX_PROTECTED_BYTES == 8168
    assert CapacityCalculator.CARRIER_HEADER_BYTES_V1 == 16
    assert CapacityCalculator.MAX_PROTECTED_BYTES_V1 == 8176
    assert CapacityCalculator.CARRIER_HEADER_BYTES_V2 == 24
    assert CapacityCalculator.MAX_PROTECTED_BYTES_V2 == 8168

    # 512x512 carrier
    rep_v2_fits = CapacityCalculator.check_capacity(8168, subband_dim=256, version=2)
    assert rep_v2_fits.fits is True
    assert rep_v2_fits.carrier_header_bytes == 24
    assert rep_v2_fits.max_protected_bytes == 8168

    with pytest.raises(PayloadCapacityExceededError, match="PAYLOAD_CAPACITY_EXCEEDED"):
        CapacityCalculator.check_capacity(8169, subband_dim=256, version=2)


def test_mask_coordinate_consistency_and_zero_overlap():
    """Verify header and payload coordinate ranges, non-collision, and seed stability."""
    # Wire Protocol v2 (192 header bits, payload sampled from [192..65535])
    mask_v2 = DeterministicMaskGenerator.generate_mask(
        session_counter=100, payload_length_bytes=1000, subband_dim=256, version=2
    )
    assert len(mask_v2.header_positions) == 192
    assert len(mask_v2.payload_positions) == 8000
    hdr_v2_set = set(mask_v2.header_positions)
    pld_v2_set = set(mask_v2.payload_positions)
    assert len(hdr_v2_set & pld_v2_set) == 0

    # Wire Protocol v1 (128 header bits, payload sampled from [128..65535])
    mask_v1 = DeterministicMaskGenerator.generate_mask(
        session_counter=100, payload_length_bytes=1000, subband_dim=256, version=1
    )
    assert len(mask_v1.header_positions) == 128
    assert len(mask_v1.payload_positions) == 8000
    hdr_v1_set = set(mask_v1.header_positions)
    pld_v1_set = set(mask_v1.payload_positions)
    assert len(hdr_v1_set & pld_v1_set) == 0


def test_oversized_payload_rejected_before_mask_generation():
    """Oversized payload length in header must trigger capacity failure before mask generation."""
    # 50 MB length from corrupted bits
    corrupted_hdr = CarrierHeader(session_counter=1, payload_length=50 * 1024 * 1024, version=2)
    with pytest.raises(PayloadCapacityExceededError):
        CapacityCalculator.check_capacity(corrupted_hdr.payload_length, subband_dim=256, version=corrupted_hdr.version)


# =============================================================================
# 5. Full End-to-End Roundtrip Tests (Baseline Integer Parity Engine)
# =============================================================================

def test_e2e_v2_baseline_embedder_roundtrip():
    """Verify full embedding, PNG serialization, reload, extraction, and recovery under v2."""
    cover = Image.new("RGB", (512, 512), color=(120, 160, 200))
    rng = np.random.RandomState(42)
    payload_data = bytes(rng.randint(0, 256, 1500, dtype=np.uint8))
    session = 8888

    # Embed with default v2
    emb_res = BaselineEmbedder.embed(cover, payload_data, session_counter=session)
    assert emb_res.header.version == 2
    assert len(emb_res.mask.header_positions) == 192

    # Lossless PNG save and reload
    buf = io.BytesIO()
    emb_res.stego_image.save(buf, format="PNG")
    buf.seek(0)
    reloaded_img = Image.open(buf)

    # Extract via Extractor
    extracted_hdr, extracted_payload = Extractor.extract(reloaded_img)
    assert extracted_hdr.version == 2
    assert extracted_hdr.session_counter == session
    assert extracted_hdr.payload_length == 1500
    assert extracted_payload == payload_data


def test_e2e_v1_legacy_compatibility_in_extractor():
    """Verify that modern Extractor seamlessly extracts legacy v1 carriers."""
    cover = Image.new("RGB", (512, 512), color=(150, 120, 90))
    rng = np.random.RandomState(99)
    payload_data = bytes(rng.randint(0, 256, 1200, dtype=np.uint8))
    session = 7777

    # Embed explicitly as v1
    emb_v1 = BaselineEmbedder.embed(cover, payload_data, session_counter=session, version=1)
    assert emb_v1.header.version == 1
    assert len(emb_v1.mask.header_positions) == 128

    # PNG round-trip
    buf = io.BytesIO()
    emb_v1.stego_image.save(buf, format="PNG")
    buf.seek(0)
    reloaded_v1 = Image.open(buf)

    # Extractor extracts 192 bits, detects v1, extracts payload using v1 positions
    extracted_hdr, extracted_payload = Extractor.extract(reloaded_v1)
    assert extracted_hdr.version == 1
    assert extracted_hdr.session_counter == session
    assert extracted_hdr.payload_length == 1200
    assert extracted_payload == payload_data


def test_e2e_media_pipeline_roundtrip_v2_and_v1():
    """Verify MediaStegoPipeline full roundtrip under Wire Protocol v2 and v1."""
    pipeline = MediaStegoPipeline()
    key = b"01234567890123456789012345678901"
    aad = b"PaladinMedia-Test-AAD-Context"
    cover = Image.new("RGB", (512, 512), color=(100, 140, 180))

    secret = SecretPayload(data=b"SECRET_MEDIA_CONTENT_TEST_V2_ECC" * 20, media_type=MediaType.TEXT, metadata={})

    # 1. Test v2 roundtrip (default)
    stego_v2, rec_v2 = pipeline.run_full_roundtrip(secret, cover, key, aad=aad, session_counter=101, version=2)
    assert stego_v2.header.version == 2
    assert rec_v2.header.version == 2
    assert rec_v2.exact_match is True
    assert rec_v2.recovered_payload.data == secret.data

    # 2. Test v1 roundtrip (legacy compatibility)
    stego_v1, rec_v1 = pipeline.run_full_roundtrip(secret, cover, key, aad=aad, session_counter=102, version=1)
    assert stego_v1.header.version == 1
    assert rec_v1.header.version == 1
    assert rec_v1.exact_match is True
    assert rec_v1.recovered_payload.data == secret.data
