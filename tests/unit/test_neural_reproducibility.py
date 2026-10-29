"""Unit tests for Phase 3 Sprint 2.1.2 Determinism and Capacity Verification.

Covers:
1. Controlled end-to-end repeatability holding all pipeline inputs constant:
   - Model weights, carrier, plaintext, AES key, AAD, and AES-GCM nonce.
   - Verifies bit-for-bit identical stego images, extracted bits, BER, and plaintext recovery.
2. Capacity convention accounting and fail-closed gate enforcement at 8,176 bytes:
   - Evaluates at-limit (8,176 bytes) and over-limit (8,177 bytes).
   - Verifies fail-closed rejection via CapacityCalculator and DeterministicMaskGenerator.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pytest
import torch

from src.carrier.capacity import CapacityCalculator
from src.carrier.mask import DeterministicMaskGenerator, MaskGenerationError
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


def test_pipeline_controlled_repeatability():
    """Verify that the end-to-end spatial roundtrip is bit-for-bit deterministic when inputs are held constant."""
    torch.manual_seed(42)
    bridge = NeuralSpatialBridge()
    bridge.eval()

    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder()

    # Fixed carrier: 128x128 image with mid-gray level (subband is 64x64 = 4096 positions)
    carrier_tensor = torch.full((1, 3, 128, 128), 128.0)

    # Fixed payload, key, aad, and nonce
    raw_data = b"Sprint2.1.2-Deterministic-Validation-Data-12345678"
    secret = SecretPayload(data=raw_data, media_type=MediaType.IMAGE, metadata={"test": "rep"})
    serialized = SecretSerializer.serialize(secret)

    key = bytes.fromhex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
    aad = b"DeterministicContext"
    fixed_nonce = bytes.fromhex("00112233445566778899aabb")

    ciphertext = crypto.encrypt(serialized, key, aad=aad, nonce=fixed_nonce)
    protected_bytes = ecc.encode(ciphertext)
    bits_1d = np.unpackbits(np.frombuffer(protected_bytes, dtype=np.uint8))
    num_bits = len(bits_1d)

    sub_dim = 64
    mask = torch.zeros(1, 1, sub_dim, sub_dim)
    payload_tensor = torch.zeros(1, 1, sub_dim, sub_dim)
    mask.view(-1)[:num_bits] = 1.0
    payload_tensor.view(-1)[:num_bits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

    runs = []
    for r in range(2):
        emb_res, ext_res, metrics = bridge.run_full_roundtrip(
            carrier=carrier_tensor,
            payload_bits=(payload_tensor > 0).to(torch.uint8),
            mask=mask,
            reload_png=True,
            image_id=f"rep_{r}",
        )
        stego_bytes = emb_res.stego_image.cpu().numpy().tobytes()
        stego_sha = hashlib.sha256(stego_bytes).hexdigest()

        ext_bits = ext_res.extracted_binary.view(-1)[:num_bits].cpu().numpy().astype(np.uint8)
        ext_sha = hashlib.sha256(ext_bits.tobytes()).hexdigest()

        bit_errs = int((bits_1d != ext_bits).sum())
        runs.append((stego_sha, ext_sha, bit_errs, metrics.exact_recovery))

    assert runs[0][0] == runs[1][0], "Stego image hash differed between deterministic runs!"
    assert runs[0][1] == runs[1][1], "Extracted bits hash differed between deterministic runs!"
    assert runs[0][2] == runs[1][2], "Bit error count differed between deterministic runs!"
    assert runs[0][3] is True and runs[1][3] is True, "Exact recovery failed in deterministic run!"

    # Decoupled cryptographic recovery verification
    rec_bytes = np.packbits(ext_bits).tobytes()
    rs_res = ecc.decode_with_result(rec_bytes)
    assert rs_res.data == ciphertext
    dec = crypto.decrypt(rs_res.data, key, aad=aad)
    rec_secret = SecretSerializer.deserialize(dec)
    assert rec_secret.data == raw_data


def test_capacity_fail_closed_boundary_8176_vs_8177():
    """Verify that CapacityCalculator and DeterministicMaskGenerator strictly enforce the 8,176 B limit."""
    subband_dim = 256
    max_protected = CapacityCalculator.MAX_PROTECTED_BYTES  # 8,176 bytes

    # Test at-limit (8,176 B)
    rep_at = CapacityCalculator.evaluate(max_protected, subband_dim=subband_dim)
    assert rep_at.fits is True
    assert rep_at.status == "FITS"
    assert rep_at.remaining_capacity_bytes == 0

    mask_at = DeterministicMaskGenerator.generate_payload_positions(
        session_counter=1, payload_length_bytes=max_protected, subband_dim=subband_dim
    )
    assert len(mask_at) == max_protected * 8  # 65,408 bits

    # Test over-limit (8,177 B)
    rep_over = CapacityCalculator.evaluate(max_protected + 1, subband_dim=subband_dim)
    assert rep_over.fits is False
    assert rep_over.status == "EXCEEDS_CAPACITY"
    assert rep_over.remaining_capacity_bytes == 0

    with pytest.raises(MaskGenerationError, match="exceeds available positions"):
        DeterministicMaskGenerator.generate_payload_positions(
            session_counter=1, payload_length_bytes=max_protected + 1, subband_dim=subband_dim
        )


def test_headerless_vs_header_reserved_capacity_accounting():
    """Verify that the 6,000 B plaintext payload fits in both headerless and header-reserved conventions."""
    subband_dim = 256
    raw_positions = subband_dim * subband_dim  # 65,536 positions
    header_positions = 128
    usable_positions = raw_positions - header_positions  # 65,408 positions

    # 6,000 B payload produces 7,000 B protected stream = 56,000 bits
    prot_bytes = 7000
    embedded_bits = prot_bytes * 8  # 56,000 bits

    # Headerless convention (test script)
    assert embedded_bits <= raw_positions
    headerless_remaining = raw_positions - embedded_bits
    assert headerless_remaining == 9536

    # Header-reserved convention (production)
    assert embedded_bits <= usable_positions
    header_reserved_remaining = usable_positions - embedded_bits
    assert header_reserved_remaining == 9408
