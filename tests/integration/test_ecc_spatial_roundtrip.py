"""Integration tests for Sprint 2.1: Spatial Bridge and ECC Gate Validation.

Validates:
1. Reed-Solomon correction boundary at t=16 and fail-closed rejection at t=17.
2. End-to-end real security pipeline roundtrip:
   SecretPayload -> SecretSerializer -> AESGCMProvider -> ReedSolomonEncoder ->
   NeuralSpatialBridge -> IDWT -> uint8 -> PNG reload -> Extractor ->
   RS decode -> AES decrypt -> Deserializer -> exact byte equality.
3. Clipping causality verification: unclipped carriers have strictly 0 bit errors.
"""

from __future__ import annotations

import os
import numpy as np
import pytest
import torch
from PIL import Image

from src.core.exceptions import ECCDecodeError
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


@pytest.fixture
def crypto_provider():
    return AESGCMProvider()


@pytest.fixture
def ecc_encoder():
    return ReedSolomonEncoder(ecc_symbols=32, block_size=255)


@pytest.fixture
def spatial_bridge():
    torch.manual_seed(42)
    return NeuralSpatialBridge()


# =============================================================================
# 1. Reed-Solomon Correction Boundary Tests
# =============================================================================

def test_rs_correction_limits_at_and_above_t16(ecc_encoder):
    """Verify exact boundary behavior of Reed-Solomon RS(255, 223) codec.

    - Under/at limit (<= 16 symbol errors): 100% exact correction.
    - Above limit (>= 17 symbol errors): terminal fail-closed ECCDecodeError.
    """
    data = os.urandom(200)
    encoded = bytearray(ecc_encoder.encode(data))

    # Test at limit (16 symbol errors in the codeword body)
    corrupted_16 = bytearray(encoded)
    for i in range(16):
        corrupted_16[4 + i] ^= 0xAA

    res_16 = ecc_encoder.decode_with_result(bytes(corrupted_16))
    assert res_16.data == data
    assert res_16.corrected_symbols == 16

    # Test above limit (17 symbol errors in the codeword body)
    corrupted_17 = bytearray(encoded)
    for i in range(17):
        corrupted_17[4 + i] ^= 0xAA

    with pytest.raises(ECCDecodeError, match="uncorrectable symbol errors exceed capacity"):
        ecc_encoder.decode(bytes(corrupted_17))


def test_rs_framing_prefix_tampering_fail_closed(ecc_encoder):
    """Verify that corruption in length prefix is recovered via derivation or rejected under strict_prefix."""
    data = os.urandom(100)
    encoded = bytearray(ecc_encoder.encode(data))

    # Corrupt length prefix byte
    encoded[0] ^= 0x80
    # Under Sprint 3.3.3 Decision B, length derivation recovers payload
    recovered = ecc_encoder.decode(bytes(encoded))
    assert recovered == data

    # Strict prefix mode enforces fail-closed rejection
    with pytest.raises(ECCDecodeError):
        ecc_encoder.decode(bytes(encoded), strict_prefix=True)


# =============================================================================
# 2. Real Security Pipeline Roundtrip Through Spatial Bridge
# =============================================================================

@pytest.mark.parametrize("payload_bytes_len", [300, 1500, 3500])
def test_end_to_end_payload_roundtrip_through_spatial_bridge(
    crypto_provider,
    ecc_encoder,
    spatial_bridge,
    payload_bytes_len: int,
):
    """Exercises:

    SecretPayload -> Serialize -> AES-256-GCM -> RS -> SpatialBridge ->
    PNG roundtrip -> Extract -> RS decode -> AES decrypt -> Deserialize.
    """
    torch.manual_seed(payload_bytes_len)
    np.random.seed(payload_bytes_len)

    raw_secret = os.urandom(payload_bytes_len)
    secret_payload = SecretPayload(
        data=raw_secret,
        media_type=MediaType.IMAGE,
        metadata={"test": "integration_e2e", "len": payload_bytes_len},
    )

    key = bytes.fromhex("11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff")
    aad = b"PaladinMedia-IntegrationAAD-Context"

    # 1. Sender Security Pipeline
    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, key, aad=aad)
    protected = ecc_encoder.encode(ciphertext)

    # 2. Bit Mapping
    bits_1d = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))
    num_bits = len(bits_1d)

    sub_dim = 256
    mask = torch.zeros(1, 1, sub_dim, sub_dim)
    payload_tensor = torch.zeros(1, 1, sub_dim, sub_dim)
    mask.view(-1)[:num_bits] = 1.0
    payload_tensor.view(-1)[:num_bits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

    # 3. Carrier (Smooth carrier in mid-range to prevent clipping)
    carrier = torch.full((1, 3, 512, 512), 128.0, dtype=torch.float32)

    # 4. Spatial Bridge Roundtrip with PNG Save/Reload
    emb_res, ext_res, metrics = spatial_bridge.run_full_roundtrip(
        carrier=carrier,
        payload_bits=(payload_tensor > 0).to(torch.uint8),
        mask=mask,
        reload_png=True,
    )

    # 5. Receiver Extraction & Security Unprotection
    ext_bits = ext_res.extracted_binary.view(-1)[:num_bits].cpu().numpy().astype(np.uint8)
    ext_bytes = np.packbits(ext_bits).tobytes()

    # RS Decoding
    rs_res = ecc_encoder.decode_with_result(ext_bytes)
    assert rs_res.data == ciphertext

    # AES-GCM Decryption
    decrypted = crypto_provider.decrypt(rs_res.data, key, aad=aad)
    assert decrypted == serialized

    # Deserialization
    recovered = SecretSerializer.deserialize(decrypted)
    assert recovered.data == secret_payload.data
    assert recovered.metadata == secret_payload.metadata
    assert recovered.media_type == secret_payload.media_type


# =============================================================================
# 3. Clipping Causality Invariant Test
# =============================================================================

def test_clipping_causality_zero_errors_on_unclipped_carrier(spatial_bridge):
    """Verify that an unclipped carrier achieves strictly 0 bit errors and 0 symbol errors."""
    carrier = torch.full((1, 3, 512, 512), 128.0, dtype=torch.float32)
    mask = torch.ones(1, 1, 256, 256)
    torch.manual_seed(999)
    payload_bits = torch.randint(0, 2, (1, 1, 256, 256), dtype=torch.uint8)

    emb_res, ext_res, metrics = spatial_bridge.run_full_roundtrip(
        carrier=carrier,
        payload_bits=payload_bits,
        mask=mask,
        reload_png=True,
    )

    # Assert 0 clipped pixels
    assert metrics.clip_stats["num_clipped_total"] == 0
    # Assert strictly 0 bit errors
    assert metrics.ber_payload == 0.0
    assert metrics.exact_recovery is True
    assert (ext_res.extracted_binary == payload_bits).all()
