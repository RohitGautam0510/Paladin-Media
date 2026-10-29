"""Unit tests for ECCEncoder / ReedSolomonEncoder (PDM-S1-004, T-004).

Tests Reed-Solomon GF(2^8) chunked encoding/decoding, symbol-level error correction
up to t = (n-k)/2 = 16 byte symbols per block, burst error resilience, and detection
of uncorrectable errors (> t symbol corruptions) raising ECCDecodeError.
"""

import struct
import pytest

from src.core.exceptions import ECCDecodeError, ECCError
from tests.conftest import get_ecc_components
from tests.fixtures.vectors import (
    PAYLOAD_EMPTY,
    PAYLOAD_SINGLE_BYTE,
    PAYLOAD_TEXT,
    PAYLOAD_JSON,
    PAYLOAD_CAPACITY,
    PAYLOAD_MULTI_BLOCK,
    inject_symbol_errors,
    inject_burst_error,
    corrupt_byte,
    truncate_data,
)

# Dynamic import resolution
ECCEncoder, ReedSolomonEncoder = get_ecc_components()


@pytest.fixture(autouse=True)
def check_implementation():
    global ECCEncoder, ReedSolomonEncoder
    if ReedSolomonEncoder is None:
        ECCEncoder, ReedSolomonEncoder = get_ecc_components()
    if ReedSolomonEncoder is None:
        pytest.skip("ReedSolomonEncoder implementation not yet landed")


# =============================================================================
# Tier 1: Feature Coverage (Clean Roundtrip, Framing, Chunking)
# =============================================================================

def test_rs_encode_decode_clean_roundtrip():
    """Verify clean roundtrip encoding and decoding without errors."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"PaladinMedia Reed-Solomon clean roundtrip verification payload."

    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)

    assert decoded == data


def test_rs_wire_framing_length_prefix():
    """Verify that the first 4 bytes represent the original data length in big-endian."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"Testing length prefix framing in RS stream"

    encoded = encoder.encode(data)
    assert len(encoded) >= 4

    declared_len = struct.unpack(">I", encoded[:4])[0]
    assert declared_len == len(data)


def test_rs_block_chunking_exact_multiples():
    """Verify chunking when data length is an exact multiple of chunk size k (223 bytes)."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    k = 255 - 32  # 223 bytes

    # 2 full blocks = 446 bytes
    data = bytes([(i % 256) for i in range(2 * k)])
    encoded = encoder.encode(data)

    # Expected wire size: 4 (prefix) + 2 * 255 (blocks) = 514 bytes
    assert len(encoded) == 4 + 2 * 255

    decoded = encoder.decode(encoded)
    assert decoded == data


def test_rs_block_chunking_arbitrary_sizes():
    """Verify chunking with non-multiple data lengths."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    k = 223

    # 500 bytes -> 2 full blocks (223 + 223) + 1 partial block (54 bytes)
    data = bytes([(i * 7) % 256 for i in range(500)])
    encoded = encoder.encode(data)

    # Expected size: 4 + 255 + 255 + (54 + 32) = 4 + 510 + 86 = 600 bytes
    assert len(encoded) == 4 + 2 * 255 + (500 - 2 * k + 32)

    decoded = encoder.decode(encoded)
    assert decoded == data


# =============================================================================
# Tier 2: Boundary & Corner Cases
# =============================================================================

def test_boundary_empty_data():
    """Verify handling of 0-byte input data."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    encoded = encoder.encode(PAYLOAD_EMPTY)

    # Length prefix must declare 0
    declared_len = struct.unpack(">I", encoded[:4])[0]
    assert declared_len == 0

    decoded = encoder.decode(encoded)
    assert decoded == b""


def test_boundary_single_byte():
    """Verify handling of 1-byte input data."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    encoded = encoder.encode(PAYLOAD_SINGLE_BYTE)

    # 4B prefix + 1B data + 32B parity = 37 bytes
    assert len(encoded) == 4 + 1 + 32

    decoded = encoder.decode(encoded)
    assert decoded == PAYLOAD_SINGLE_BYTE


def test_boundary_chunk_size_minus_one():
    """Verify data length exactly k - 1 (222 bytes, single shortened block)."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"A" * 222
    encoded = encoder.encode(data)
    assert len(encoded) == 4 + 222 + 32
    assert encoder.decode(encoded) == data


def test_boundary_chunk_size_exact():
    """Verify data length exactly k (223 bytes, single full block)."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"B" * 223
    encoded = encoder.encode(data)
    assert len(encoded) == 4 + 255
    assert encoder.decode(encoded) == data


def test_boundary_chunk_size_plus_one():
    """Verify data length k + 1 (224 bytes, 1 full block + 1 1-byte block)."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"C" * 224
    encoded = encoder.encode(data)
    # Block 1: 223 + 32 = 255. Block 2: 1 + 32 = 33. Total = 4 + 255 + 33 = 292
    assert len(encoded) == 4 + 255 + 33
    assert encoder.decode(encoded) == data


def test_boundary_capacity_payload():
    """Verify 7,088-byte capacity payload roundtrip."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    encoded = encoder.encode(PAYLOAD_CAPACITY)
    decoded = encoder.decode(encoded)
    assert decoded == PAYLOAD_CAPACITY


def test_boundary_multi_block_10000_bytes():
    """Verify 10,000-byte multi-block payload roundtrip (~45 blocks)."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    encoded = encoder.encode(PAYLOAD_MULTI_BLOCK)
    decoded = encoder.decode(encoded)
    assert decoded == PAYLOAD_MULTI_BLOCK


# =============================================================================
# Tier 3: Symbol-Level Correction & Negative Uncorrectable Tests
# =============================================================================

def test_symbol_error_correction_single_error():
    """Verify correction of 1 corrupted byte symbol in a block."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"Single symbol error correction test vector for PaladinMedia RS"
    encoded = encoder.encode(data)

    # Corrupt 1 byte in the block area (offset 10)
    corrupted = corrupt_byte(encoded, byte_pos=10)
    assert corrupted != encoded

    decoded = encoder.decode(corrupted)
    assert decoded == data


def test_symbol_error_correction_half_bound():
    """Verify correction of t/2 = 8 corrupted byte symbols in a block."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"Half-bound error correction test payload with plenty of bytes." * 3
    encoded = encoder.encode(data)

    # Inject 8 symbol errors in block 1 (offset 4 to len(encoded))
    corrupted, indices = inject_symbol_errors(
        encoded, count=8, block_offset=4, block_length=len(encoded) - 4, seed=42
    )
    assert len(indices) == 8

    decoded = encoder.decode(corrupted)
    assert decoded == data


def test_symbol_error_correction_exact_bound_t():
    """Verify correction of exactly t = 16 corrupted byte symbols in a single block.

    This is the theoretical maximum correction bound for 2t = 32 parity bytes.
    """
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    # Create single full block (223 bytes data -> 255 bytes block)
    data = bytes([(i * 13) % 256 for i in range(223)])
    encoded = encoder.encode(data)
    assert len(encoded) == 4 + 255

    # Inject exactly t=16 symbol corruptions into the 255-byte block
    corrupted, indices = inject_symbol_errors(
        encoded, count=16, block_offset=4, block_length=255, seed=12345
    )
    assert len(indices) == 16

    decoded = encoder.decode(corrupted)
    assert decoded == data, "Failed to recover data at maximum theoretical bound t=16!"


def test_symbol_error_correction_shortened_block():
    """Verify that a shortened RS block (e.g. 76 bytes data + 32 parity) corrects up to t=16 errors and rejects t+1=17 errors."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    # 76 bytes data (matches 1024x1024 partial block size)
    data = bytes([(i * 19) % 256 for i in range(76)])
    encoded = encoder.encode(data)
    # Wire format: 4 (prefix) + 76 (data) + 32 (parity) = 112 bytes
    assert len(encoded) == 4 + 76 + 32

    # 1. Inject t=16 errors into the 108-byte shortened block (offset 4)
    corrupted_16, indices_16 = inject_symbol_errors(
        encoded, count=16, block_offset=4, block_length=108, seed=4242
    )
    assert len(indices_16) == 16
    decoded_16 = encoder.decode(corrupted_16)
    assert decoded_16 == data

    # 2. Inject t+1=17 errors into the 108-byte shortened block -> must raise ECCDecodeError
    corrupted_17, indices_17 = inject_symbol_errors(
        encoded, count=17, block_offset=4, block_length=108, seed=4242
    )
    assert len(indices_17) == 17
    with pytest.raises(ECCDecodeError):
        encoder.decode(corrupted_17)


def test_symbol_error_correction_across_multiple_blocks():
    """Verify that multiple blocks can each independently correct up to t errors."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    k = 223
    # 2 full blocks = 446 bytes
    data = bytes([(i * 17) % 256 for i in range(2 * k)])
    encoded = encoder.encode(data)

    # Block 1 starts at offset 4 (size 255)
    # Block 2 starts at offset 4 + 255 = 259 (size 255)
    corrupted_b1, idx1 = inject_symbol_errors(encoded, count=15, block_offset=4, block_length=255, seed=101)
    corrupted_b1_b2, idx2 = inject_symbol_errors(corrupted_b1, count=15, block_offset=259, block_length=255, seed=202)

    decoded = encoder.decode(corrupted_b1_b2)
    assert decoded == data


def test_burst_error_all_8_bits_in_single_symbol():
    """Verify that flipping all 8 bits in a single byte counts as only 1 symbol error."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"Burst error in single byte symbol test"
    encoded = encoder.encode(data)

    # Flip all 8 bits in byte at index 15
    corrupted = bytearray(encoded)
    corrupted[15] ^= 0xFF

    decoded = encoder.decode(bytes(corrupted))
    assert decoded == data


def test_burst_error_contiguous_16_bytes():
    """Verify contiguous burst error of 16 bytes (128 bit errors) is corrected within 1 block."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = bytes([(i * 3) % 256 for i in range(223)])
    encoded = encoder.encode(data)

    # Inject contiguous 16-byte burst (16 symbol errors <= t=16) starting at offset 20
    corrupted = inject_burst_error(encoded, start_byte=20, burst_byte_len=16)

    decoded = encoder.decode(corrupted)
    assert decoded == data


def test_negative_uncorrectable_t_plus_one_errors():
    """Verify that injecting t + 1 = 17 symbol errors into a block raises ECCDecodeError."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = bytes([(i * 5) % 256 for i in range(223)])
    encoded = encoder.encode(data)

    # Inject 17 symbol errors in the 255-byte block
    corrupted, indices = inject_symbol_errors(
        encoded, count=17, block_offset=4, block_length=255, seed=999
    )
    assert len(indices) == 17

    with pytest.raises(ECCDecodeError):
        encoder.decode(corrupted)


def test_negative_uncorrectable_t_plus_five_errors():
    """Verify that injecting t + 5 = 21 symbol errors into a block raises ECCDecodeError."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = bytes([(i * 9) % 256 for i in range(223)])
    encoded = encoder.encode(data)

    corrupted, _ = inject_symbol_errors(
        encoded, count=21, block_offset=4, block_length=255, seed=888
    )

    with pytest.raises(ECCDecodeError):
        encoder.decode(corrupted)


def test_negative_multi_block_one_uncorrectable():
    """Verify that if one block among many exceeds t errors, ECCDecodeError is raised."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    k = 223
    data = bytes([(i * 11) % 256 for i in range(3 * k)])  # 3 blocks
    encoded = encoder.encode(data)

    # Block 1: 5 errors (correctable)
    c1, _ = inject_symbol_errors(encoded, count=5, block_offset=4, block_length=255, seed=1)
    # Block 2: 18 errors (uncorrectable!)
    c2, _ = inject_symbol_errors(c1, count=18, block_offset=4 + 255, block_length=255, seed=2)

    with pytest.raises(ECCDecodeError):
        encoder.decode(c2)


def test_negative_truncated_encoded_stream():
    """Verify that truncating the encoded stream raises ECCDecodeError or ECCError."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"Truncation test data payload"
    encoded = encoder.encode(data)

    for cut in [1, 5, 16, 32]:
        truncated = truncate_data(encoded, cut)
        with pytest.raises((ECCDecodeError, ECCError)):
            encoder.decode(truncated)


def test_negative_corrupted_length_prefix():
    """Verify that corrupting the 4-byte length prefix raises ECCDecodeError when strict_prefix=True,
    or when the stream length itself is structurally invalid."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"Short payload"
    encoded = encoder.encode(data)

    # 1. Under strict_prefix=True, declared length mismatch raises ECCDecodeError
    corrupted = struct.pack(">I", 10_000_000) + encoded[4:]
    with pytest.raises((ECCDecodeError, ECCError)):
        encoder.decode(corrupted, strict_prefix=True)

    # 2. When stream length itself is invalid, raises ECCDecodeError even with recovery
    invalid_stream = encoded[:-5]
    with pytest.raises((ECCDecodeError, ECCError)):
        encoder.decode(invalid_stream)


# =============================================================================
# Tier 4: Real-World Workloads
# =============================================================================

def test_real_world_secret_payload_ecc_under_channel_noise():
    """Verify that real-world payloads withstand simulated channel symbol noise."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)

    for payload in [PAYLOAD_TEXT, PAYLOAD_JSON]:
        encoded = encoder.encode(payload)

        # Inject 10 random symbol errors in the encoded stream (safe within t=16)
        corrupted, _ = inject_symbol_errors(
            encoded, count=10, block_offset=4, block_length=len(encoded) - 4, seed=777
        )

        decoded = encoder.decode(corrupted)
        assert decoded == payload
