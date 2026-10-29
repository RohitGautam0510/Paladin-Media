"""Focused Deterministic Unit Tests for Sprint 3.3.3: ECC Framing Recovery.

Covers Phase C requirements:
1. Empty input handling (L=0).
2. Boundary lengths: immediately below, at, and above block boundaries (k-1, k, k+1, 2k-1, 2k, 2k+1).
3. Valid full-block and shortened-block streams.
4. Intact prefix agreement with derived length.
5. Prefix corruption: 1 bit, 1 byte, multiple bytes, all 4 bytes.
6. Structurally invalid stream lengths (tail < 33 bytes).
7. Truncated and oversized frames.
8. Arbitrary binary payloads.
9. Correction boundary (t=16 vs t=17).
10. Fail-closed security: zero plaintext release on uncorrectable errors or authentication failure.
"""

import hashlib
import struct
import pytest

from src.core.exceptions import ECCDecodeError, ECCError
from src.security.ecc.encoder import ReedSolomonEncoder


def make_deterministic_bytes(length: int, seed: int = 42) -> bytes:
    """Generates a reproducible byte sequence of arbitrary length using SHA-256 counter."""
    chunks = []
    generated = 0
    counter = 0
    while generated < length:
        h = hashlib.sha256(struct.pack(">II", seed, counter)).digest()
        needed = min(len(h), length - generated)
        chunks.append(h[:needed])
        generated += needed
        counter += 1
    return b"".join(chunks)


# =============================================================================
# 1. Block Boundary & Framing Derivation Tests
# =============================================================================

@pytest.mark.parametrize(
    "payload_len",
    [
        0,       # Empty input
        1,       # Single byte
        222,     # k - 1 (shortened block)
        223,     # k (single full block)
        224,     # k + 1 (1 full + 1 shortened of 1 byte)
        445,     # 2k - 1 (1 full + 1 shortened of 222 bytes)
        446,     # 2k (2 full blocks)
        447,     # 2k + 1 (2 full + 1 shortened of 1 byte)
        1000,    # Multi-block
        7088,    # Near-capacity baseline
    ],
)
def test_ecc_framing_length_derivation_roundtrip(payload_len: int):
    """Verifies that derive_payload_length perfectly recovers the exact length
    for empty, boundary, and multi-block payloads."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = make_deterministic_bytes(payload_len, seed=100 + payload_len)
    encoded = encoder.encode(data)

    derived = encoder.derive_payload_length(len(encoded))
    assert derived == payload_len

    decoded = encoder.decode(encoded)
    assert decoded == data


def test_ecc_empty_input_handling():
    """Verifies that 0-byte payload encodes to 4-byte prefix and decodes cleanly."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    encoded = encoder.encode(b"")
    assert len(encoded) == 4
    assert struct.unpack(">I", encoded[:4])[0] == 0

    res = encoder.decode_with_result(encoded)
    assert res.data == b""
    assert res.blocks_count == 0
    assert res.corrected_symbols == 0
    assert res.prefix_recovered is False


# =============================================================================
# 2. Length-Prefix Corruption Recovery Tests
# =============================================================================

def test_ecc_prefix_corruption_single_bit():
    """Verifies recovery when exactly 1 bit flips in the 4-byte prefix."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = make_deterministic_bytes(500, seed=1)
    encoded = bytearray(encoder.encode(data))

    # Flip 1 bit in byte 0
    encoded[0] ^= 0x01
    corrupted = bytes(encoded)

    res = encoder.decode_with_result(corrupted)
    assert res.data == data
    assert res.prefix_recovered is True


def test_ecc_prefix_corruption_single_byte():
    """Verifies recovery when 1 entire byte is corrupted in the 4-byte prefix."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = make_deterministic_bytes(1500, seed=2)
    encoded = bytearray(encoder.encode(data))

    # Corrupt byte 2 completely
    encoded[2] = 0xAA
    corrupted = bytes(encoded)

    res = encoder.decode_with_result(corrupted)
    assert res.data == data
    assert res.prefix_recovered is True


def test_ecc_prefix_corruption_all_four_bytes():
    """Verifies recovery when all 4 bytes of the prefix are corrupted to an impossible value."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = make_deterministic_bytes(3500, seed=3)
    encoded = encoder.encode(data)

    # Set declared length to 99,999,999 bytes
    corrupted = struct.pack(">I", 99_999_999) + encoded[4:]

    res = encoder.decode_with_result(corrupted)
    assert res.data == data
    assert res.prefix_recovered is True


def test_ecc_strict_prefix_mode_rejects_corruption():
    """Verifies that strict_prefix=True rejects prefix corruption."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = b"Strict prefix mode verification"
    encoded = encoder.encode(data)

    corrupted = struct.pack(">I", 9999) + encoded[4:]
    with pytest.raises(ECCDecodeError, match="Encoded data length mismatch"):
        encoder.decode(corrupted, strict_prefix=True)


# =============================================================================
# 3. Structurally Invalid & Malformed Framing Tests
# =============================================================================

@pytest.mark.parametrize("tail_rem", [1, 10, 20, 31, 32])
def test_ecc_structurally_invalid_stream_lengths_rejected(tail_rem: int):
    """Verifies that stream lengths with tail in [1..32] (fewer than 32 parity + 1 data)
    are rejected as structurally invalid."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    # Stream length: 4 (prefix) + 255 (full block) + tail_rem
    invalid_len = 4 + 255 + tail_rem
    fake_stream = bytes(invalid_len)

    with pytest.raises(ECCDecodeError, match="Invalid encoded stream length"):
        encoder.decode(fake_stream)


def test_ecc_truncated_stream_rejected():
    """Verifies that truncated streams fail closed with ECCDecodeError."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = make_deterministic_bytes(400, seed=5)
    encoded = encoder.encode(data)

    # Truncate by 1 byte
    with pytest.raises(ECCDecodeError):
        encoder.decode(encoded[:-1])

    # Truncate by 15 bytes
    with pytest.raises(ECCDecodeError):
        encoder.decode(encoded[:-15])


def test_ecc_oversized_stream_rejected():
    """Verifies that appending garbage bytes resulting in an invalid RS framing length fails closed."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    # 223 bytes data encodes to exactly 4 + 255 = 259 bytes (1 full block)
    data = make_deterministic_bytes(223, seed=6)
    encoded = encoder.encode(data)
    assert len(encoded) == 259

    # Appending 1 to 32 bytes causes M % 255 to fall in [1..32], which is structurally invalid
    for extra_bytes in [1, 5, 16, 32]:
        oversized = encoded + bytes(extra_bytes)
        with pytest.raises(ECCDecodeError, match="Invalid encoded stream length"):
            encoder.decode(oversized)


# =============================================================================
# 4. Error Correction Boundary & Fail-Closed Tests
# =============================================================================

def test_ecc_correction_within_t16_boundary():
    """Verifies that exactly t=16 symbol corruptions in a block are corrected."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = make_deterministic_bytes(223, seed=7)
    encoded = bytearray(encoder.encode(data))

    # Corrupt exactly 16 symbols in block 0 (indices 4..19)
    for i in range(4, 20):
        encoded[i] ^= 0xFF

    res = encoder.decode_with_result(bytes(encoded))
    assert res.data == data
    assert res.corrected_symbols == 16


def test_ecc_uncorrectable_t17_fails_closed():
    """Verifies that t=17 symbol corruptions in a block fail closed with ECCDecodeError."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    data = make_deterministic_bytes(223, seed=8)
    encoded = bytearray(encoder.encode(data))

    # Corrupt 17 symbols in block 0 (indices 4..20)
    for i in range(4, 21):
        encoded[i] ^= 0xFF

    with pytest.raises(ECCDecodeError, match="uncorrectable symbol errors exceed capacity"):
        encoder.decode(bytes(encoded))


def test_ecc_arbitrary_binary_payloads_with_corrupted_prefix():
    """Verifies clean roundtrip for arbitrary binary payloads containing null bytes,
    high-entropy bytes, and inverted bits, even with corrupted prefix."""
    encoder = ReedSolomonEncoder(ecc_symbols=32, block_size=255)
    test_vectors = [
        bytes([i % 256 for i in range(500)]),
        b"\x00" * 300,
        b"\xFF" * 300,
        b"\x00\xFF\x00\xFF" * 100,
    ]

    for vec in test_vectors:
        encoded = encoder.encode(vec)
        # Corrupt prefix
        corrupted = struct.pack(">I", 0xDEADBEEF) + encoded[4:]
        res = encoder.decode_with_result(corrupted)
        assert res.data == vec
        assert res.prefix_recovered is True


def test_ecc_end_to_end_fail_closed_zero_plaintext_leak_on_excess_corruption():
    """Verifies that when error corruption exceeds RS capacity, the pipeline fails closed
    either at ECC decode (ECCDecodeError) or at AES-GCM authentication (AuthenticationError),
    with zero plaintext ever returned or leaked."""
    from src.core.pipeline import SecurityPipeline
    from src.core.serialization import SecretPayload, MediaType
    from src.security.crypto.provider import AESGCMProvider
    from src.core.exceptions import AuthenticationError

    pipe = SecurityPipeline(AESGCMProvider(), ReedSolomonEncoder(ecc_symbols=32, block_size=255))
    key = bytes(32)
    payload = SecretPayload(data=b"Confidential payload bytes", media_type=MediaType.TEXT)
    protected = bytearray(pipe.protect(payload, key))

    # Introduce burst corruption of 20 symbols into block 0 (indices 4..23)
    for i in range(4, 24):
        protected[i] ^= 0xAA

    with pytest.raises((ECCDecodeError, AuthenticationError)):
        pipe.unprotect(bytes(protected), key)
