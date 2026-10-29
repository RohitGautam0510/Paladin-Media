"""Integration tests for the complete PaladinMedia protected payload pipeline.

Tests the full unidirectional pipeline sequence:
  SecretSerializer.serialize -> CryptoProvider.encrypt -> ECCEncoder.encode
And exact symmetric reversal:
  ECCEncoder.decode -> CryptoProvider.decrypt -> SecretSerializer.deserialize

Covers:
  - Clean channel roundtrip across diverse payload types and metadata.
  - Correctable channel noise injection (within Reed-Solomon t=16 symbol budget).
  - SecurityPipeline wrapper integration (when available).
  - Multi-block large payloads (~7KB capacity and 10KB streams).
"""

import pytest

from tests.conftest import (
    get_serialization_components,
    get_crypto_components,
    get_ecc_components,
    get_pipeline_component,
)
from tests.fixtures.vectors import (
    PAYLOAD_EMPTY,
    PAYLOAD_SINGLE_BYTE,
    PAYLOAD_TEXT,
    PAYLOAD_JSON,
    PAYLOAD_IMAGE_SNIPPET,
    PAYLOAD_CAPACITY,
    PAYLOAD_MULTI_BLOCK,
    inject_symbol_errors,
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
        pytest.skip("Pipeline dependencies (Serializer, Crypto, ECC) not all landed")


# Helper for direct step-by-step pipeline execution
def run_full_pipeline_roundtrip(payload, key, aad=None, noise_injector=None):
    """Executes full step-by-step pipeline with optional noise injection in the channel."""
    # 1. Serialize
    serialized = SecretSerializer.serialize(payload)

    # 2. Encrypt
    crypto = AESGCMProvider()
    protected = crypto.encrypt(serialized, key, aad=aad)

    # 3. ECC Encode
    ecc = ReedSolomonEncoder()
    encoded = ecc.encode(protected)

    # [Channel Simulation]
    channel_output = encoded
    if noise_injector is not None:
        channel_output = noise_injector(encoded)

    # 4. ECC Decode
    recovered_protected = ecc.decode(channel_output)

    # 5. Decrypt
    decrypted = crypto.decrypt(recovered_protected, key, aad=aad)

    # 6. Deserialize
    recovered_payload = SecretSerializer.deserialize(decrypted)
    return recovered_payload


# =============================================================================
# Tier 1: Clean Channel Roundtrip Tests
# =============================================================================

def test_pipeline_clean_roundtrip_text(sample_key, sample_aad):
    """Verify clean channel pipeline roundtrip with UTF-8 text directive."""
    payload = SecretPayload(
        data=PAYLOAD_TEXT,
        media_type=MediaType.TEXT,
        metadata={"classification": "SECRET", "priority": "CRITICAL"},
    )

    recovered = run_full_pipeline_roundtrip(payload, sample_key, sample_aad)

    assert recovered.data == payload.data
    assert recovered.media_type == MediaType.TEXT
    assert recovered.metadata == payload.metadata


def test_pipeline_clean_roundtrip_all_media_types(sample_key):
    """Verify clean channel pipeline roundtrip across all media types."""
    test_cases = [
        SecretPayload(data=b"\x01\x02\x03\x04", media_type=MediaType.RAW_BYTES),
        SecretPayload(data=PAYLOAD_TEXT, media_type=MediaType.TEXT),
        SecretPayload(data=PAYLOAD_IMAGE_SNIPPET, media_type=MediaType.IMAGE),
        SecretPayload(data=b"AUDIO_DATA_STUB", media_type=MediaType.AUDIO),
        SecretPayload(data=PAYLOAD_JSON, media_type=MediaType.JSON),
    ]
    for original in test_cases:
        recovered = run_full_pipeline_roundtrip(original, sample_key)
        assert recovered.data == original.data
        assert recovered.media_type == original.media_type


# =============================================================================
# Tier 2: Boundary & Capacity Tests
# =============================================================================

def test_pipeline_boundary_empty_payload(sample_key):
    """Verify clean pipeline roundtrip with 0-byte payload."""
    payload = SecretPayload(data=PAYLOAD_EMPTY, media_type=MediaType.RAW_BYTES, metadata=None)
    recovered = run_full_pipeline_roundtrip(payload, sample_key)
    assert recovered.data == b""
    assert recovered.media_type == MediaType.RAW_BYTES


def test_pipeline_boundary_single_byte(sample_key):
    """Verify clean pipeline roundtrip with 1-byte payload."""
    payload = SecretPayload(data=PAYLOAD_SINGLE_BYTE, media_type=MediaType.RAW_BYTES, metadata=None)
    recovered = run_full_pipeline_roundtrip(payload, sample_key)
    assert recovered.data == PAYLOAD_SINGLE_BYTE


def test_pipeline_boundary_capacity_payload(sample_key, sample_aad):
    """Verify clean pipeline roundtrip with ~7KB capacity payload."""
    payload = SecretPayload(
        data=PAYLOAD_CAPACITY,
        media_type=MediaType.RAW_BYTES,
        metadata={"subband": "HH", "size": len(PAYLOAD_CAPACITY)},
    )
    recovered = run_full_pipeline_roundtrip(payload, sample_key, sample_aad)
    assert recovered.data == PAYLOAD_CAPACITY
    assert recovered.metadata["size"] == 7088


def test_pipeline_boundary_multi_block_10kb(sample_key):
    """Verify multi-block 10KB payload through complete pipeline."""
    payload = SecretPayload(
        data=PAYLOAD_MULTI_BLOCK,
        media_type=MediaType.RAW_BYTES,
        metadata={"blocks": "multi"},
    )
    recovered = run_full_pipeline_roundtrip(payload, sample_key)
    assert recovered.data == PAYLOAD_MULTI_BLOCK


# =============================================================================
# Tier 3: Correctable Channel Noise Tests
# =============================================================================

def test_pipeline_correctable_noise_single_block(sample_key, sample_aad):
    """Verify pipeline recovers 100% exact bytes when channel injects 12 symbol errors (<= 16)."""
    payload = SecretPayload(
        data=b"Protected payload enduring channel perturbations and self-healing completely!",
        media_type=MediaType.TEXT,
        metadata={"noise_test": True},
    )

    def channel_noise(stream: bytes) -> bytes:
        # Inject 12 symbol errors within block (offset 4 to len)
        corrupted, _ = inject_symbol_errors(
            stream, count=12, block_offset=4, block_length=len(stream) - 4, seed=54321
        )
        return corrupted

    recovered = run_full_pipeline_roundtrip(payload, sample_key, sample_aad, noise_injector=channel_noise)
    assert recovered.data == payload.data
    assert recovered.metadata == payload.metadata


def test_pipeline_correctable_noise_multi_block(sample_key):
    """Verify pipeline recovers multi-block payload with symbol errors distributed in each block."""
    payload = SecretPayload(
        data=PAYLOAD_CAPACITY,  # ~7KB spans ~32 RS blocks
        media_type=MediaType.RAW_BYTES,
        metadata={"multi_block_noise": True},
    )

    def distributed_channel_noise(stream: bytes) -> bytes:
        # Inject 8 symbol errors in block 1 (offset 4, len 255)
        # and 8 symbol errors in block 2 (offset 259, len 255)
        # and 8 symbol errors in block 3 (offset 514, len 255)
        s = stream
        s, _ = inject_symbol_errors(s, count=8, block_offset=4, block_length=255, seed=11)
        s, _ = inject_symbol_errors(s, count=8, block_offset=259, block_length=255, seed=22)
        s, _ = inject_symbol_errors(s, count=8, block_offset=514, block_length=255, seed=33)
        return s

    recovered = run_full_pipeline_roundtrip(payload, sample_key, noise_injector=distributed_channel_noise)
    assert recovered.data == PAYLOAD_CAPACITY


# =============================================================================
# Tier 4: SecurityPipeline High-Level Wrapper (if implemented)
# =============================================================================

def test_security_pipeline_class_wrapper(sample_key, sample_aad):
    """Verify the SecurityPipeline class interface from src.core.pipeline if implemented."""
    if SecurityPipeline is None:
        pytest.skip("SecurityPipeline class not implemented yet")

    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder()
    pipeline = SecurityPipeline(crypto_provider=crypto, ecc_encoder=ecc)

    original = SecretPayload(data=PAYLOAD_TEXT, media_type=MediaType.TEXT, metadata={"env": "prod"})

    protected_bytes = pipeline.protect(original, key=sample_key, aad=sample_aad)
    recovered = pipeline.unprotect(protected_bytes, key=sample_key, aad=sample_aad)

    assert recovered.data == original.data
    assert recovered.media_type == original.media_type
    assert recovered.metadata == original.metadata
