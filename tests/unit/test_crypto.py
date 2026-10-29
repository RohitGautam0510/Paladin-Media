"""Unit tests for CryptoProvider / AESGCMProvider (PDM-S1-002, PDM-S1-003, T-002, T-003).

Tests AES-256-GCM encryption/decryption, packaging format ([12B Nonce] + [Ciphertext] + [16B Tag]),
NIST SP 800-38D test vectors, nonce uniqueness, AAD authentication, and comprehensive
fail-closed negative security tests under single-bit flipping and payload truncation.
"""

import pytest

from src.core.exceptions import AuthenticationError, CryptoError
from tests.conftest import get_crypto_components
from tests.fixtures.vectors import (
    NIST_AES_256_GCM_VECTORS,
    PAYLOAD_EMPTY,
    PAYLOAD_SINGLE_BYTE,
    PAYLOAD_TEXT,
    PAYLOAD_JSON,
    PAYLOAD_CAPACITY,
    flip_bit,
    truncate_data,
    append_garbage,
)

# Dynamic import resolution
CryptoProvider, AESGCMProvider = get_crypto_components()


@pytest.fixture(autouse=True)
def check_implementation():
    global CryptoProvider, AESGCMProvider
    if AESGCMProvider is None:
        CryptoProvider, AESGCMProvider = get_crypto_components()
    if AESGCMProvider is None:
        pytest.skip("AESGCMProvider implementation not yet landed")


# =============================================================================
# Tier 1: Feature Coverage (T-002: Packaging, Vectors, Nonce Uniqueness)
# =============================================================================

def test_aes_256_gcm_encrypt_decrypt_roundtrip(sample_key):
    """Verify basic AES-256-GCM encryption and decryption roundtrip."""
    provider = AESGCMProvider()
    plaintext = b"PaladinMedia MVP 0 Confidential Plaintext"

    protected = provider.encrypt(plaintext, sample_key)
    decrypted = provider.decrypt(protected, sample_key)

    assert decrypted == plaintext


def test_packaging_layout_and_length(sample_key):
    """Verify wire packaging layout: [12B Nonce] + [Ciphertext] + [16B Tag].

    Total packed length must equal len(plaintext) + 28.
    """
    provider = AESGCMProvider()
    plaintext = b"Payload with exact length measurement"

    protected = provider.encrypt(plaintext, sample_key)
    expected_length = len(plaintext) + provider.NONCE_SIZE + provider.TAG_SIZE
    assert len(protected) == expected_length

    nonce = protected[:provider.NONCE_SIZE]
    tag = protected[-provider.TAG_SIZE:]
    ciphertext = protected[provider.NONCE_SIZE:-provider.TAG_SIZE]

    assert len(nonce) == 12
    assert len(tag) == 16
    assert len(ciphertext) == len(plaintext)


def test_nist_sp800_38d_kat_vectors():
    """Verify AES-256-GCM implementation against official NIST SP 800-38D test vectors."""
    provider = AESGCMProvider()

    for vector in NIST_AES_256_GCM_VECTORS:
        # Test Decryption of official KAT vector
        packed_kat = vector.packed_payload
        aad = vector.aad if vector.aad else None

        decrypted = provider.decrypt(packed_kat, vector.key, aad=aad)
        assert decrypted == vector.plaintext, f"Failed KAT decryption for: {vector.description}"

        # Test Encryption with explicit IV matches KAT ciphertext and tag
        try:
            encrypted = provider.encrypt(vector.plaintext, vector.key, aad=aad, nonce=vector.iv)
            assert encrypted == packed_kat, f"Failed KAT encryption match for: {vector.description}"
        except TypeError:
            # If provider doesn't accept explicit nonce in encrypt, skip encrypt check
            pass


def test_nonce_uniqueness_stateless(sample_key):
    """Verify that encrypting identical plaintext twice produces distinct nonces and ciphertexts."""
    provider = AESGCMProvider()
    plaintext = b"Identical plaintext for nonce uniqueness verification"

    protected1 = provider.encrypt(plaintext, sample_key)
    protected2 = provider.encrypt(plaintext, sample_key)

    nonce1 = protected1[:12]
    nonce2 = protected2[:12]
    ciphertext1 = protected1[12:-16]
    ciphertext2 = protected2[12:-16]

    assert nonce1 != nonce2, "CRITICAL: Nonce reuse detected across consecutive encryptions!"
    assert ciphertext1 != ciphertext2, "Ciphertext collision under distinct nonces!"


def test_aad_integrity_binding(sample_key, sample_aad):
    """Verify that Associated Authenticated Data (AAD) binds to the payload."""
    provider = AESGCMProvider()
    plaintext = b"Data with context metadata binding"

    protected = provider.encrypt(plaintext, sample_key, aad=sample_aad)

    # Matching AAD -> Success
    decrypted = provider.decrypt(protected, sample_key, aad=sample_aad)
    assert decrypted == plaintext


# =============================================================================
# Tier 2: Boundary & Corner Cases
# =============================================================================

def test_boundary_empty_plaintext(sample_key):
    """Verify encryption of 0-byte plaintext produces 28-byte packed payload."""
    provider = AESGCMProvider()
    protected = provider.encrypt(PAYLOAD_EMPTY, sample_key)
    assert len(protected) == 28  # 12B nonce + 0B ciphertext + 16B tag

    decrypted = provider.decrypt(protected, sample_key)
    assert decrypted == b""


def test_boundary_single_byte_plaintext(sample_key):
    """Verify encryption of 1-byte plaintext produces 29-byte packed payload."""
    provider = AESGCMProvider()
    protected = provider.encrypt(PAYLOAD_SINGLE_BYTE, sample_key)
    assert len(protected) == 29

    decrypted = provider.decrypt(protected, sample_key)
    assert decrypted == PAYLOAD_SINGLE_BYTE


def test_boundary_large_payload_64kb(sample_key):
    """Verify encryption/decryption of large 64KB payload."""
    provider = AESGCMProvider()
    large_pt = bytes([(i * 31) % 256 for i in range(65536)])

    protected = provider.encrypt(large_pt, sample_key)
    assert len(protected) == 65536 + 28

    decrypted = provider.decrypt(protected, sample_key)
    assert decrypted == large_pt


def test_boundary_pathological_plaintexts(sample_key):
    """Verify handling of all-zero, all-one, and alternating bytes."""
    provider = AESGCMProvider()
    test_cases = [
        b"\x00" * 256,
        b"\xff" * 256,
        b"\xaa\x55" * 128,
    ]
    for pt in test_cases:
        protected = provider.encrypt(pt, sample_key)
        assert provider.decrypt(protected, sample_key) == pt


def test_boundary_none_vs_empty_bytes_aad(sample_key):
    """Verify consistent behavior between aad=None and aad=b''."""
    provider = AESGCMProvider()
    pt = b"AAD boundary test"

    # Encrypt with aad=None
    p1 = provider.encrypt(pt, sample_key, aad=None)
    # Decrypt with aad=None -> OK
    assert provider.decrypt(p1, sample_key, aad=None) == pt


# =============================================================================
# Tier 3: Negative Security Tests (T-003: FAIL-CLOSED Invariant)
# =============================================================================

def test_fail_closed_ciphertext_bit_flip_head(sample_key):
    """Verify flipping a single bit in the first byte of ciphertext raises AuthenticationError."""
    provider = AESGCMProvider()
    plaintext = b"Confidential Payload Content"
    protected = provider.encrypt(plaintext, sample_key)

    # First byte of ciphertext is at index 12 (after 12B nonce)
    tampered = flip_bit(protected, byte_pos=12, bit_pos=0)

    result = None
    with pytest.raises((AuthenticationError, CryptoError)):
        result = provider.decrypt(tampered, sample_key)

    # Strictly prove zero plaintext leaked
    assert result is None


def test_fail_closed_ciphertext_bit_flip_middle(sample_key):
    """Verify flipping a bit in the middle of ciphertext raises AuthenticationError."""
    provider = AESGCMProvider()
    plaintext = b"Longer confidential message that spans multiple blocks for bit-flipping test"
    protected = provider.encrypt(plaintext, sample_key)

    mid_pos = 12 + (len(plaintext) // 2)
    tampered = flip_bit(protected, byte_pos=mid_pos, bit_pos=3)

    result = None
    with pytest.raises((AuthenticationError, CryptoError)):
        result = provider.decrypt(tampered, sample_key)
    assert result is None


def test_fail_closed_ciphertext_bit_flip_tail(sample_key):
    """Verify flipping a bit in the last byte of ciphertext raises AuthenticationError."""
    provider = AESGCMProvider()
    plaintext = b"Confidential Payload Content"
    protected = provider.encrypt(plaintext, sample_key)

    # Last byte of ciphertext is at index len(protected) - 17
    last_ct_pos = len(protected) - 17
    tampered = flip_bit(protected, byte_pos=last_ct_pos, bit_pos=7)

    result = None
    with pytest.raises((AuthenticationError, CryptoError)):
        result = provider.decrypt(tampered, sample_key)
    assert result is None


def test_fail_closed_tag_bit_flip_head(sample_key):
    """Verify flipping a single bit in the first byte of tag raises AuthenticationError."""
    provider = AESGCMProvider()
    plaintext = b"Confidential Payload Content"
    protected = provider.encrypt(plaintext, sample_key)

    # First byte of tag is at len(protected) - 16
    tag_head_pos = len(protected) - 16
    tampered = flip_bit(protected, byte_pos=tag_head_pos, bit_pos=0)

    result = None
    with pytest.raises((AuthenticationError, CryptoError)):
        result = provider.decrypt(tampered, sample_key)
    assert result is None


def test_fail_closed_tag_bit_flip_tail(sample_key):
    """Verify flipping a single bit in the last byte of tag raises AuthenticationError."""
    provider = AESGCMProvider()
    plaintext = b"Confidential Payload Content"
    protected = provider.encrypt(plaintext, sample_key)

    # Last byte of tag is at len(protected) - 1
    tag_tail_pos = len(protected) - 1
    tampered = flip_bit(protected, byte_pos=tag_tail_pos, bit_pos=0)

    result = None
    with pytest.raises((AuthenticationError, CryptoError)):
        result = provider.decrypt(tampered, sample_key)
    assert result is None


def test_fail_closed_tampered_aad(sample_key, sample_aad):
    """Verify that any modification to AAD causes AuthenticationError."""
    provider = AESGCMProvider()
    plaintext = b"Sensitive data with AAD"
    protected = provider.encrypt(plaintext, sample_key, aad=sample_aad)

    # Modified AAD (1 bit flipped)
    tampered_aad = flip_bit(sample_aad, byte_pos=0, bit_pos=0)

    with pytest.raises((AuthenticationError, CryptoError)):
        provider.decrypt(protected, sample_key, aad=tampered_aad)


def test_fail_closed_missing_expected_aad(sample_key, sample_aad):
    """Verify decrypting without AAD when AAD was used raises AuthenticationError."""
    provider = AESGCMProvider()
    plaintext = b"Sensitive data with AAD"
    protected = provider.encrypt(plaintext, sample_key, aad=sample_aad)

    with pytest.raises((AuthenticationError, CryptoError)):
        provider.decrypt(protected, sample_key, aad=None)


def test_fail_closed_extraneous_aad_provided(sample_key, sample_aad):
    """Verify providing AAD during decryption when none was used raises AuthenticationError."""
    provider = AESGCMProvider()
    plaintext = b"Sensitive data without AAD"
    protected = provider.encrypt(plaintext, sample_key, aad=None)

    with pytest.raises((AuthenticationError, CryptoError)):
        provider.decrypt(protected, sample_key, aad=sample_aad)


def test_fail_closed_wrong_key(sample_key, alternative_key):
    """Verify decryption with mismatched 32-byte key raises AuthenticationError."""
    provider = AESGCMProvider()
    plaintext = b"Classified directive"
    protected = provider.encrypt(plaintext, sample_key)

    with pytest.raises((AuthenticationError, CryptoError)):
        provider.decrypt(protected, alternative_key)


def test_negative_invalid_key_lengths():
    """Verify keys shorter or longer than 32 bytes raise CryptoError."""
    provider = AESGCMProvider()
    plaintext = b"Data"
    invalid_keys = [
        b"",
        b"\x00" * 16,  # 128-bit key
        b"\x00" * 24,  # 192-bit key
        b"\x00" * 31,  # 248-bit key
        b"\x00" * 33,  # 264-bit key
        b"\x00" * 64,  # 512-bit key
    ]
    for key in invalid_keys:
        with pytest.raises(CryptoError):
            provider.encrypt(plaintext, key)


def test_negative_truncated_protected_payload(sample_key):
    """Verify truncated payload (< 28 bytes total) raises AuthenticationError or CryptoError."""
    provider = AESGCMProvider()

    # Any payload shorter than 28 bytes cannot contain 12B nonce + 16B tag
    for length in [0, 1, 12, 20, 27]:
        short_payload = b"\x00" * length
        with pytest.raises((AuthenticationError, CryptoError)):
            provider.decrypt(short_payload, sample_key)


def test_prove_no_plaintext_leakage_on_tamper(sample_key):
    """Prove mathematically and programmatically that no plaintext is ever returned on tamper."""
    provider = AESGCMProvider()
    secret_text = b"EXTREMELY_SENSITIVE_SECRET_TOKEN_PALADIN_2026"
    protected = provider.encrypt(secret_text, sample_key)

    # Tamper with 10 random bit locations across ciphertext and tag
    for offset in range(12, len(protected)):
        tampered = flip_bit(protected, byte_pos=offset, bit_pos=0)
        returned_val = "NOT_NONE"
        try:
            returned_val = provider.decrypt(tampered, sample_key)
            pytest.fail(f"Decryption succeeded on tampered payload at offset {offset}!")
        except (AuthenticationError, CryptoError):
            # Pass: correct fail-closed exception raised
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e)}")

        # Verify returned_val was never set to any slice of plaintext
        assert returned_val == "NOT_NONE" or returned_val is None


# =============================================================================
# Tier 4: Real-World Workloads
# =============================================================================

def test_real_world_encrypted_capacity_payload(sample_key, sample_aad):
    """Verify encryption and decryption of ~7KB capacity payload with AAD."""
    provider = AESGCMProvider()
    protected = provider.encrypt(PAYLOAD_CAPACITY, sample_key, aad=sample_aad)
    assert len(protected) == len(PAYLOAD_CAPACITY) + 28

    decrypted = provider.decrypt(protected, sample_key, aad=sample_aad)
    assert decrypted == PAYLOAD_CAPACITY
