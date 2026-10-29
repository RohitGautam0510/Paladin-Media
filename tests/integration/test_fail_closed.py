"""Integration tests for fail-closed security invariants across the PaladinMedia pipeline.

Architectural Invariant INV-03 Mandate:
Any uncorrectable error, cryptographic tampering, tag mismatch, key mismatch,
or AAD discrepancy MUST cause terminal rejection and leak ZERO plaintext.

Tests:
  - Uncorrectable channel noise (> t symbol errors) halts at ECC layer (ECCDecodeError).
  - Post-ECC tampered ciphertext halts at Crypto layer (AuthenticationError).
  - Tampered authentication tag halts at Crypto layer (AuthenticationError).
  - AAD tampering / mismatch halts at Crypto layer (AuthenticationError).
  - Wrong decryption key halts at Crypto layer (AuthenticationError).
  - Mathematical verification of zero plaintext leakage across all failure modes.
"""

import pytest

from src.core.exceptions import AuthenticationError, ECCDecodeError
from tests.conftest import (
    get_serialization_components,
    get_crypto_components,
    get_ecc_components,
    get_pipeline_component,
)
from tests.fixtures.vectors import (
    PAYLOAD_TEXT,
    PAYLOAD_CAPACITY,
    inject_symbol_errors,
    flip_bit,
)

SecretPayload, MediaType, SecretSerializer = get_serialization_components()
CryptoProvider, AESGCMProvider = get_crypto_components()
ECCEncoder, ReedSolomonEncoder = get_ecc_components()
SecurityPipeline = get_pipeline_component()


@pytest.fixture(autouse=True)
def check_pipeline_dependencies():
    global SecretPayload, MediaType, SecretSerializer
    global CryptoProvider, AESGCMProvider
    global ECCEncoder, ReedSolomonEncoder
    global SecurityPipeline

    if SecretSerializer is None:
        SecretPayload, MediaType, SecretSerializer = get_serialization_components()
    if AESGCMProvider is None:
        CryptoProvider, AESGCMProvider = get_crypto_components()
    if ReedSolomonEncoder is None:
        ECCEncoder, ReedSolomonEncoder = get_ecc_components()
    if SecurityPipeline is None:
        SecurityPipeline = get_pipeline_component()

    if None in (SecretSerializer, AESGCMProvider, ReedSolomonEncoder):
        pytest.skip("Pipeline dependencies not fully landed")


# =============================================================================
# Fail-Closed Pipeline Tests
# =============================================================================

def test_integration_uncorrectable_channel_noise_halts_at_ecc(sample_key, sample_aad):
    """Verify that channel noise exceeding t=16 symbol errors per block raises ECCDecodeError.

    The pipeline must abort immediately at the ECC decoder, preventing corrupted
    data from ever reaching the decryption engine.
    """
    payload = SecretPayload(data=PAYLOAD_TEXT, media_type=MediaType.TEXT)

    serialized = SecretSerializer.serialize(payload)
    crypto = AESGCMProvider()
    protected = crypto.encrypt(serialized, sample_key, aad=sample_aad)
    ecc = ReedSolomonEncoder()
    encoded = ecc.encode(protected)

    # Inject 20 symbol errors in the first block (exceeds t=16 bound)
    corrupted, indices = inject_symbol_errors(
        encoded, count=20, block_offset=4, block_length=min(255, len(encoded) - 4), seed=9999
    )
    assert len(indices) == 20

    # Receiver pipeline execution
    result = "NOT_NONE"
    with pytest.raises(ECCDecodeError):
        # Must halt here!
        recovered_protected = ecc.decode(corrupted)
        result = crypto.decrypt(recovered_protected, sample_key, aad=sample_aad)

    # Zero plaintext returned
    assert result == "NOT_NONE"


def test_integration_post_ecc_tampered_ciphertext_halts_fail_closed(sample_key, sample_aad):
    """Verify that if an adversary modifies ciphertext after ECC or re-encodes forged ciphertext,
    AES-GCM detects tag mismatch and raises AuthenticationError fail-closed.
    """
    payload = SecretPayload(data=PAYLOAD_TEXT, media_type=MediaType.TEXT)

    serialized = SecretSerializer.serialize(payload)
    crypto = AESGCMProvider()
    protected = crypto.encrypt(serialized, sample_key, aad=sample_aad)

    # Adversary flips 1 bit in ciphertext (offset 15) and re-encodes with valid RS parity
    tampered_protected = flip_bit(protected, byte_pos=15, bit_pos=0)

    ecc = ReedSolomonEncoder()
    encoded_tampered = ecc.encode(tampered_protected)

    # Receiver pipeline:
    # 1. ECC decode succeeds (parity matches tampered payload)
    decoded_protected = ecc.decode(encoded_tampered)
    assert decoded_protected == tampered_protected

    # 2. Crypto decrypt MUST fail closed
    result = "NOT_NONE"
    with pytest.raises(AuthenticationError):
        result = crypto.decrypt(decoded_protected, sample_key, aad=sample_aad)

    assert result == "NOT_NONE"


def test_integration_post_ecc_tampered_tag_halts_fail_closed(sample_key, sample_aad):
    """Verify that tampering with the GHASH authentication tag causes fail-closed AuthenticationError."""
    payload = SecretPayload(data=PAYLOAD_TEXT, media_type=MediaType.TEXT)

    serialized = SecretSerializer.serialize(payload)
    crypto = AESGCMProvider()
    protected = crypto.encrypt(serialized, sample_key, aad=sample_aad)

    # Flip bit in the authentication tag (last byte of protected)
    tampered_protected = flip_bit(protected, byte_pos=len(protected) - 1, bit_pos=0)

    ecc = ReedSolomonEncoder()
    encoded_tampered = ecc.encode(tampered_protected)

    decoded_protected = ecc.decode(encoded_tampered)

    result = "NOT_NONE"
    with pytest.raises(AuthenticationError):
        result = crypto.decrypt(decoded_protected, sample_key, aad=sample_aad)

    assert result == "NOT_NONE"


def test_integration_aad_mismatch_halts_fail_closed(sample_key, sample_aad):
    """Verify that providing altered AAD during pipeline unprotect causes AuthenticationError."""
    payload = SecretPayload(data=PAYLOAD_TEXT, media_type=MediaType.TEXT)

    serialized = SecretSerializer.serialize(payload)
    crypto = AESGCMProvider()
    protected = crypto.encrypt(serialized, sample_key, aad=sample_aad)
    ecc = ReedSolomonEncoder()
    encoded = ecc.encode(protected)

    # Clean transmission through ECC
    decoded_protected = ecc.decode(encoded)

    # Wrong AAD provided to receiver
    wrong_aad = b"Mismatched-Context-AAD"
    with pytest.raises(AuthenticationError):
        crypto.decrypt(decoded_protected, sample_key, aad=wrong_aad)


def test_integration_wrong_key_halts_fail_closed(sample_key, alternative_key):
    """Verify that providing an incorrect 32-byte key causes AuthenticationError."""
    payload = SecretPayload(data=PAYLOAD_TEXT, media_type=MediaType.TEXT)

    serialized = SecretSerializer.serialize(payload)
    crypto = AESGCMProvider()
    protected = crypto.encrypt(serialized, sample_key)
    ecc = ReedSolomonEncoder()
    encoded = ecc.encode(protected)

    decoded_protected = ecc.decode(encoded)

    with pytest.raises(AuthenticationError):
        crypto.decrypt(decoded_protected, alternative_key)


def test_prove_zero_plaintext_leakage_systematic(sample_key, sample_aad, alternative_key):
    """Systematically prove across all error vectors that no plaintext is ever returned or leaked."""
    secret_bytes = b"SECRET_TOP_SECRET_DIRECTIVE_2026_DO_NOT_LEAK"
    payload = SecretPayload(data=secret_bytes, media_type=MediaType.TEXT)

    serialized = SecretSerializer.serialize(payload)
    crypto = AESGCMProvider()
    protected = crypto.encrypt(serialized, sample_key, aad=sample_aad)
    ecc = ReedSolomonEncoder()
    encoded = ecc.encode(protected)

    failure_scenarios = [
        # Scenario 1: excessive noise in encoded stream (> t errors)
        ("uncorrectable_noise", lambda: ecc.decode(
            inject_symbol_errors(encoded, 18, block_offset=4, block_length=min(255, len(encoded)-4), seed=1)[0]
        )),
        # Scenario 2: tampered ciphertext bit
        ("tampered_ct", lambda: crypto.decrypt(
            flip_bit(protected, 14, 0), sample_key, aad=sample_aad
        )),
        # Scenario 3: tampered tag bit
        ("tampered_tag", lambda: crypto.decrypt(
            flip_bit(protected, len(protected) - 1, 0), sample_key, aad=sample_aad
        )),
        # Scenario 4: tampered nonce bit
        ("tampered_nonce", lambda: crypto.decrypt(
            flip_bit(protected, 0, 0), sample_key, aad=sample_aad
        )),
        # Scenario 5: mismatched AAD
        ("mismatched_aad", lambda: crypto.decrypt(
            protected, sample_key, aad=b"wrong_aad"
        )),
        # Scenario 6: wrong key
        ("wrong_key", lambda: crypto.decrypt(
            protected, alternative_key, aad=sample_aad
        )),
    ]

    for name, op in failure_scenarios:
        caught = False
        result = None
        try:
            result = op()
        except (AuthenticationError, ECCDecodeError):
            caught = True
        except Exception as exc:
            pytest.fail(f"Scenario '{name}' raised unexpected exception: {type(exc)}: {exc}")

        assert caught, f"Scenario '{name}' did not raise expected fail-closed exception!"
        assert result is None, f"Scenario '{name}' leaked data: {result}"
