"""Known test vectors, payload fixtures, and error injection helpers for PaladinMedia.

This module provides:
1. NIST SP 800-38D AES-256-GCM test vectors.
2. Representative secret payload fixtures across media types.
3. Symbol-level and bit-level error injection helper utilities.
"""

from dataclasses import dataclass
import json
import random
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AESGCMVector:
    """NIST SP 800-38D AES-256-GCM Test Vector representation."""
    description: str
    key: bytes
    iv: bytes
    plaintext: bytes
    aad: bytes
    ciphertext: bytes
    tag: bytes

    @property
    def packed_payload(self) -> bytes:
        """Returns canonical packed wire format: IV (12B) || Ciphertext || Tag (16B)."""
        return self.iv + self.ciphertext + self.tag


# -----------------------------------------------------------------------------
# Official NIST SP 800-38D Test Vectors for AES-256-GCM
# -----------------------------------------------------------------------------
NIST_AES_256_GCM_VECTORS: List[AESGCMVector] = [
    # Vector 1: 0-byte plaintext, 0-byte AAD
    AESGCMVector(
        description="NIST SP 800-38D Case 1 (Empty Plaintext, Empty AAD, 256-bit Key)",
        key=bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000"),
        iv=bytes.fromhex("000000000000000000000000"),
        plaintext=b"",
        aad=b"",
        ciphertext=b"",
        tag=bytes.fromhex("530f8afbc74536b9a963b4f1c4cb738b"),
    ),
    # Vector 2: 16-byte plaintext (single AES block), 0-byte AAD
    AESGCMVector(
        description="NIST SP 800-38D Case 2 (16B Zero Plaintext, Empty AAD, 256-bit Key)",
        key=bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000"),
        iv=bytes.fromhex("000000000000000000000000"),
        plaintext=bytes.fromhex("00000000000000000000000000000000"),
        aad=b"",
        ciphertext=bytes.fromhex("cea7403d4d606b6e074ec5d3baf39d18"),
        tag=bytes.fromhex("d0d1c8a799996bf0265b98b5d48ab919"),
    ),
    # Vector 3: 64-byte plaintext, 0-byte AAD
    AESGCMVector(
        description="NIST SP 800-38D Case 4 (64B Plaintext, Empty AAD, 256-bit Key)",
        key=bytes.fromhex("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308"),
        iv=bytes.fromhex("cafebabefacedbaddecaf888"),
        plaintext=bytes.fromhex(
            "d9313225f88406e5a55909c5aff5269a"
            "86a7a9531534f7da2e4c303d8a318a72"
            "1c3c0c95956809532fcf0e2449a6b525"
            "b16aedf5aa0de657ba637b391aafd255"
        ),
        aad=b"",
        ciphertext=bytes.fromhex(
            "522dc1f099567d07f47f37a32a84427d"
            "643a8cdcbfe5c0c97598a2bd2555d1aa"
            "8cb08e48590dbb3da7b08b1056828838"
            "c5f61e6393ba7a0abcc9f662898015ad"
        ),
        tag=bytes.fromhex("b094dac5d93471bdec1a502270e3cc6c"),
    ),
    # Vector 4: 64-byte plaintext, 20-byte AAD
    AESGCMVector(
        description="NIST SP 800-38D Case 5 (64B Plaintext, 20B AAD, 256-bit Key)",
        key=bytes.fromhex("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308"),
        iv=bytes.fromhex("cafebabefacedbaddecaf888"),
        plaintext=bytes.fromhex(
            "d9313225f88406e5a55909c5aff5269a"
            "86a7a9531534f7da2e4c303d8a318a72"
            "1c3c0c95956809532fcf0e2449a6b525"
            "b16aedf5aa0de657ba637b391aafd255"
        ),
        aad=bytes.fromhex("feedfacedeadbeeffeedfacedeadbeefabaddaca"),
        ciphertext=bytes.fromhex(
            "522dc1f099567d07f47f37a32a84427d"
            "643a8cdcbfe5c0c97598a2bd2555d1aa"
            "8cb08e48590dbb3da7b08b1056828838"
            "c5f61e6393ba7a0abcc9f662898015ad"
        ),
        tag=bytes.fromhex("8956dada5beee07d28b113fcfb43605d"),
    ),
]


# -----------------------------------------------------------------------------
# Representative Secret Payloads
# -----------------------------------------------------------------------------

PAYLOAD_EMPTY = b""

PAYLOAD_SINGLE_BYTE = b"\x42"

PAYLOAD_TEXT_STR = (
    "CLASSIFIED OPERATIONAL DIRECTIVE // PALADIN MEDIA MVP 0 // SPRINT 1\n"
    "Target: Secure Cryptographic Foundation & Reed-Solomon Protected Bitstream.\n"
    "Invariant: Fail-closed verification with 0 unauthenticated plaintext leak.\n"
    "Metadata hash binding active."
)
PAYLOAD_TEXT = PAYLOAD_TEXT_STR.encode("utf-8")

PAYLOAD_JSON_DICT: Dict[str, Any] = {
    "operation": "BLACK_BRIAR",
    "version": 1,
    "coordinates": {"lat": 37.7749, "lon": -122.4194, "alt_m": 42.5},
    "targets": ["node_alpha", "relay_beta", "sentinel_gamma"],
    "active": True,
    "auth_tokens": [101, 202, 303, 404],
}
PAYLOAD_JSON = json.dumps(PAYLOAD_JSON_DICT, sort_keys=True, separators=(",", ":")).encode("utf-8")

# Synthetic 64-byte raw image snippet (e.g. 4x4 RGBA raw pixel block)
PAYLOAD_IMAGE_SNIPPET = bytes(
    [(i * 37 + (i % 7) * 19) % 256 for i in range(256)]
)

# Reference usable capacity payload (~7,088 bytes per Architecture §10.2)
def generate_capacity_payload(length: int = 7088, seed: int = 0x504C4453) -> bytes:
    """Generates a deterministic pseudo-random payload of specified length."""
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(length))

PAYLOAD_CAPACITY = generate_capacity_payload(7088)

# Multi-chunk payload spanning ~45 Reed-Solomon blocks (10,000 bytes)
PAYLOAD_MULTI_BLOCK = generate_capacity_payload(10000, seed=0xDEADBEEF)


# -----------------------------------------------------------------------------
# Error Injection Helpers
# -----------------------------------------------------------------------------

def flip_bit(data: bytes, byte_pos: int, bit_pos: int = 0) -> bytes:
    """Flips a single bit at the specified byte position and bit index (0-7).

    Args:
        data: Input byte sequence.
        byte_pos: 0-indexed byte position.
        bit_pos: Bit position within byte (0=LSB, 7=MSB).

    Returns:
        New bytes object with exactly one bit inverted.
    """
    if not (0 <= byte_pos < len(data)):
        raise IndexError(f"byte_pos {byte_pos} out of range [0, {len(data)})")
    if not (0 <= bit_pos <= 7):
        raise ValueError(f"bit_pos {bit_pos} must be between 0 and 7")

    byte_arr = bytearray(data)
    byte_arr[byte_pos] ^= (1 << bit_pos)
    return bytes(byte_arr)


def flip_random_bit(data: bytes, rng: Optional[random.Random] = None) -> Tuple[bytes, int, int]:
    """Flips a randomly chosen bit in data.

    Returns:
        Tuple of (corrupted_bytes, byte_pos, bit_pos).
    """
    if not data:
        raise ValueError("Cannot flip bit in empty data")
    if rng is None:
        rng = random.Random()
    byte_pos = rng.randrange(len(data))
    bit_pos = rng.randrange(8)
    return flip_bit(data, byte_pos, bit_pos), byte_pos, bit_pos


def corrupt_byte(data: bytes, byte_pos: int, new_val: Optional[int] = None) -> bytes:
    """Corrupts an entire byte symbol at byte_pos to ensure new_val != original.

    Args:
        data: Input byte sequence.
        byte_pos: 0-indexed byte position.
        new_val: Optional replacement value. If None, XORs with 0xFF.

    Returns:
        New bytes object with specified byte altered.
    """
    if not (0 <= byte_pos < len(data)):
        raise IndexError(f"byte_pos {byte_pos} out of range [0, {len(data)})")

    byte_arr = bytearray(data)
    if new_val is None:
        byte_arr[byte_pos] ^= 0xFF
    else:
        if new_val == byte_arr[byte_pos]:
            byte_arr[byte_pos] ^= 0x01
        else:
            byte_arr[byte_pos] = new_val & 0xFF
    return bytes(byte_arr)


def inject_symbol_errors(
    data: bytes,
    count: int,
    block_offset: int = 0,
    block_length: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[bytes, List[int]]:
    """Injects exactly `count` distinct symbol (byte) corruptions within a specified block.

    Args:
        data: Input byte sequence.
        count: Number of distinct byte errors to inject.
        block_offset: Starting byte index of the target block.
        block_length: Length of target block. Defaults to len(data) - block_offset.
        seed: Optional RNG seed for deterministic error placement.

    Returns:
        Tuple of (corrupted_bytes, sorted_list_of_corrupted_indices).
    """
    if block_length is None:
        block_length = len(data) - block_offset

    if block_offset + block_length > len(data):
        raise ValueError("Block range exceeds data length")
    if count > block_length:
        raise ValueError(f"Cannot inject {count} errors in block of size {block_length}")

    rng = random.Random(seed)
    target_indices = rng.sample(range(block_offset, block_offset + block_length), count)
    target_indices.sort()

    byte_arr = bytearray(data)
    for idx in target_indices:
        # Corrupt byte to a guaranteed different value
        orig = byte_arr[idx]
        delta = rng.randint(1, 255)
        byte_arr[idx] = (orig ^ delta) & 0xFF

    return bytes(byte_arr), target_indices


def inject_burst_error(data: bytes, start_byte: int, burst_byte_len: int) -> bytes:
    """Injects a contiguous burst error spanning `burst_byte_len` consecutive bytes.

    All bits in each byte in the burst are inverted (8 bit errors per symbol).
    """
    if start_byte + burst_byte_len > len(data):
        raise ValueError("Burst exceeds data length")

    byte_arr = bytearray(data)
    for i in range(start_byte, start_byte + burst_byte_len):
        byte_arr[i] ^= 0xFF  # Invert all 8 bits
    return bytes(byte_arr)


def truncate_data(data: bytes, num_bytes: int = 1) -> bytes:
    """Truncates `num_bytes` from the end of data."""
    if num_bytes >= len(data):
        return b""
    return data[:-num_bytes]


def append_garbage(data: bytes, garbage: bytes = b"\xde\xad\xbe\xef") -> bytes:
    """Appends trailing garbage bytes to data."""
    return data + garbage
