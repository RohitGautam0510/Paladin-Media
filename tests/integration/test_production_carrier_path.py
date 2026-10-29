"""Focused Integration Tests for Sprint 3.3.1: Production Carrier Path Verification.

Exercises PaladinMedia's real production carrier interfaces with the neural spatial bridge:
1. Production Carrier Framing & Mask Integration:
   - SecretPayload & SecretSerializer (binary wire serialization)
   - AESGCMProvider (AES-256-GCM authenticated encryption)
   - ReedSolomonEncoder (RS(255, 223), t=16 error correction)
   - CapacityCalculator (production capacity hard gate: 8,176 B protected limit)
   - CarrierHeader (16-byte fixed header framing at fixed anchor positions 0..127)
   - DeterministicMaskGenerator (deterministic pseudo-random coordinate generation from seed)
   - NeuralSpatialBridge (Haar DWT -> ACS-Sub-INN v3 Red-HH embedding -> IDWT)
   - Lossless PNG serialization and reload
   - Neural extraction, header unpack, payload coordinate reconstruction, and exact byte recovery.
2. Capacity & Framing Boundary Agreement:
   - CapacityCalculator vs DeterministicMaskGenerator vs CarrierHeader (8,176 B fits, 8,177 B rejected)
3. Header & Extraction Fail-Closed Validation:
   - Corrupted header magic, version, or truncated headers raise CarrierHeaderError.
4. Stego Container Tampering Fail-Closed:
   - Zero plaintext leaked on corrupted stego input.
5. Production Interface Coupling / Architectural Gap Audit:
   - Tests and documents that classical BaselineEmbedder and Extractor are hardcoded to
     spatial integer parity ((a - b - c + d) mod 2) and fail on neural stego images.
6. Model Parameter Immutability:
   - Verifies zero weight updates (untrained model state strictly preserved).
"""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from src.carrier.capacity import (
    CapacityCalculator,
    CapacityReport,
    PayloadCapacityExceededError,
)
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import (
    DeterministicMaskGenerator,
    EmbeddingMask,
    MaskGenerationError,
)
from src.core.exceptions import AuthenticationError, ECCDecodeError
from src.core.media_pipeline import MediaStegoPipeline, calculate_psnr, calculate_ssim
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import DIV2KStegoDataset
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


# =============================================================================
# Helper Functions
# =============================================================================

def compute_model_param_digest(model: torch.nn.Module) -> str:
    """Computes a SHA-256 digest over all parameters in a PyTorch module."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def measure_rs_codeword_errors(
    protected: bytes,
    extracted: bytes,
    k: int = 223,
    ecc_symbols: int = 32,
) -> Dict[str, Any]:
    """Measures byte symbol errors per Reed-Solomon codeword block."""
    if len(protected) < 4 or len(extracted) < 4:
        raise ValueError("Protected byte stream too short for length prefix")

    header_errors = sum(1 for p, e in zip(protected[:4], extracted[:4]) if p != e)
    total_length = struct.unpack(">I", protected[:4])[0]
    num_full_blocks = total_length // k
    remainder = total_length % k

    offset = 4
    block_errors: List[int] = []

    for _ in range(num_full_blocks):
        block_len = k + ecc_symbols
        p_block = protected[offset : offset + block_len]
        e_block = extracted[offset : offset + block_len]
        errs = sum(1 for p, e in zip(p_block, e_block) if p != e)
        block_errors.append(errs)
        offset += block_len

    if remainder > 0:
        block_len = remainder + ecc_symbols
        p_block = protected[offset : offset + block_len]
        e_block = extracted[offset : offset + block_len]
        errs = sum(1 for p, e in zip(p_block, e_block) if p != e)
        block_errors.append(errs)
        offset += block_len

    worst = max(block_errors) if block_errors else 0
    total = sum(block_errors)
    max_correctable = ecc_symbols // 2

    return {
        "header_errors": header_errors,
        "block_errors": block_errors,
        "worst_codeword_errors": worst,
        "total_codeword_errors": total,
        "within_t16_limit": worst <= max_correctable,
        "t_limit": max_correctable,
    }


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def crypto_provider() -> AESGCMProvider:
    return AESGCMProvider()


@pytest.fixture
def ecc_encoder() -> ReedSolomonEncoder:
    return ReedSolomonEncoder(ecc_symbols=32, block_size=255)


@pytest.fixture
def spatial_bridge() -> NeuralSpatialBridge:
    torch.manual_seed(42)
    acs = ACSSubINN(gamma_min=1.0)
    return NeuralSpatialBridge(acs_model=acs)


@pytest.fixture
def fixed_key() -> bytes:
    return bytes.fromhex("11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff")


@pytest.fixture
def fixed_aad() -> bytes:
    return b"PaladinMedia-Phase3-Sprint3.3.1-ProductionCarrier"


# =============================================================================
# 1. Production Carrier Path Roundtrip Tests (Header + Mask + Neural Bridge)
# =============================================================================

def test_production_carrier_path_synthetic_carrier(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    spatial_bridge: NeuralSpatialBridge,
    fixed_key: bytes,
    fixed_aad: bytes,
):
    """Verifies complete end-to-end roundtrip on synthetic carrier using real production framing:
    - CarrierHeader (16B packed into 128 anchor bits)
    - DeterministicMaskGenerator (generating non-colliding coordinates from seed)
    - NeuralSpatialBridge (continuous wavelet modulation + IDWT + PNG save/reload)
    - Receiver extracts header at anchor positions, unpacks CarrierHeader, derives payload coordinates,
      and recovers exact plaintext bytes.
    """
    sub_dim = 256
    session_counter = 42
    payload_len = 500
    raw_secret = os.urandom(payload_len)
    secret_payload = SecretPayload(
        data=raw_secret,
        media_type=MediaType.IMAGE,
        metadata={"carrier": "synthetic", "load": payload_len},
    )

    # 1. Serialize -> Encrypt -> RS Encode
    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)

    # 2. Production Capacity Hard Gate
    cap_report = CapacityCalculator.check_capacity(len(protected), subband_dim=sub_dim)
    assert cap_report.fits is True

    # 3. Production Carrier Header
    header = CarrierHeader(session_counter=session_counter, payload_length=len(protected))
    header_bytes = header.pack()
    assert len(header_bytes) == 16
    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    payload_bits = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))

    # 4. Production Deterministic Mask Generation
    mask = DeterministicMaskGenerator.generate_mask(
        session_counter=session_counter,
        payload_length_bytes=len(protected),
        subband_dim=sub_dim,
    )
    assert len(mask.header_positions) == 128
    assert len(mask.payload_positions) == len(protected) * 8

    # 5. Populate 2D tensors strictly using production mask coordinates
    mask_tensor = torch.zeros(1, 1, sub_dim, sub_dim)
    payload_tensor = torch.zeros(1, 1, sub_dim, sub_dim)

    for bit, (r, c) in zip(header_bits, mask.header_positions):
        mask_tensor[0, 0, r, c] = 1.0
        payload_tensor[0, 0, r, c] = 1.0 if bit == 1 else -1.0

    for bit, (r, c) in zip(payload_bits, mask.payload_positions):
        mask_tensor[0, 0, r, c] = 1.0
        payload_tensor[0, 0, r, c] = 1.0 if bit == 1 else -1.0

    # 6. Embed through NeuralSpatialBridge and perform PNG roundtrip
    carrier = torch.full((1, 3, 512, 512), 128.0, dtype=torch.float32)
    emb_res = spatial_bridge.embed_spatial(carrier, payload_tensor, mask_tensor, differentiable=False)
    reloaded_stego = spatial_bridge.save_and_reload_png(emb_res.stego_image)

    # 7. Receiver Extraction: Extract through NeuralSpatialBridge
    ext_res = spatial_bridge.extract_spatial(reloaded_stego)

    # 8. Receiver Header Interpretation (from fixed anchor coordinates)
    extracted_hdr_bits = [int(ext_res.extracted_binary[0, 0, r, c].item()) for r, c in mask.header_positions]
    extracted_hdr_bytes = Extractor.bits_to_bytes(extracted_hdr_bits)
    rec_header = CarrierHeader.unpack(extracted_hdr_bytes)
    assert rec_header.session_counter == session_counter
    assert rec_header.payload_length == len(protected)

    # 9. Receiver Payload Interpretation (from regenerated mask coordinates)
    rec_payload_positions = DeterministicMaskGenerator.generate_payload_positions(
        session_counter=rec_header.session_counter,
        payload_length_bytes=rec_header.payload_length,
        subband_dim=sub_dim,
    )
    extracted_payload_bits = [int(ext_res.extracted_binary[0, 0, r, c].item()) for r, c in rec_payload_positions]
    extracted_protected = Extractor.bits_to_bytes(extracted_payload_bits)

    # 10. Reed-Solomon Decode -> AES-GCM Decrypt -> Deserialize
    rs_res = ecc_encoder.decode_with_result(extracted_protected)
    assert rs_res.data == ciphertext
    assert rs_res.corrected_symbols == 0

    decrypted = crypto_provider.decrypt(rs_res.data, fixed_key, aad=fixed_aad)
    recovered = SecretSerializer.deserialize(decrypted)

    # 11. Assert exact byte equality
    assert recovered.data == secret_payload.data
    assert recovered.metadata == secret_payload.metadata


@pytest.mark.parametrize("carrier_idx, img_name", [(0, "0001.png"), (1, "0002.png"), (4, "0005.png")])
def test_production_carrier_path_div2k_carrier(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    spatial_bridge: NeuralSpatialBridge,
    fixed_key: bytes,
    fixed_aad: bytes,
    carrier_idx: int,
    img_name: str,
):
    """Verifies complete production framing and mask roundtrip on native DIV2K carriers."""
    from src.neural.dataset import DEFAULT_DATASET_REL_PATH
    repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / DEFAULT_DATASET_REL_PATH).is_dir():
        pytest.skip(
            f"DIV2K dataset root missing at '{repo_root / DEFAULT_DATASET_REL_PATH}'. "
            f"Download DIV2K_train_HR per README to run dataset integration tests."
        )
    ds = DIV2KStegoDataset(split="train")
    carrier = ds[carrier_idx]["carrier"].unsqueeze(0)

    sub_dim = 256
    session_counter = 777 + carrier_idx
    payload_len = 1500
    raw_secret = os.urandom(payload_len)
    secret_payload = SecretPayload(
        data=raw_secret,
        media_type=MediaType.IMAGE,
        metadata={"image": img_name, "session": session_counter},
    )

    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)

    cap_report = CapacityCalculator.check_capacity(len(protected), subband_dim=sub_dim)
    assert cap_report.fits is True

    header = CarrierHeader(session_counter=session_counter, payload_length=len(protected))
    header_bytes = header.pack()
    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    payload_bits = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))

    mask = DeterministicMaskGenerator.generate_mask(
        session_counter=session_counter,
        payload_length_bytes=len(protected),
        subband_dim=sub_dim,
    )

    mask_tensor = torch.zeros(1, 1, sub_dim, sub_dim)
    payload_tensor = torch.zeros(1, 1, sub_dim, sub_dim)

    for bit, (r, c) in zip(header_bits, mask.header_positions):
        mask_tensor[0, 0, r, c] = 1.0
        payload_tensor[0, 0, r, c] = 1.0 if bit == 1 else -1.0

    for bit, (r, c) in zip(payload_bits, mask.payload_positions):
        mask_tensor[0, 0, r, c] = 1.0
        payload_tensor[0, 0, r, c] = 1.0 if bit == 1 else -1.0

    emb_res = spatial_bridge.embed_spatial(carrier, payload_tensor, mask_tensor, differentiable=False)
    reloaded_stego = spatial_bridge.save_and_reload_png(emb_res.stego_image)

    ext_res = spatial_bridge.extract_spatial(reloaded_stego)

    extracted_hdr_bits = [int(ext_res.extracted_binary[0, 0, r, c].item()) for r, c in mask.header_positions]
    extracted_hdr_bytes = Extractor.bits_to_bytes(extracted_hdr_bits)
    rec_header = CarrierHeader.unpack(extracted_hdr_bytes)
    assert rec_header.session_counter == session_counter
    assert rec_header.payload_length == len(protected)

    rec_payload_positions = DeterministicMaskGenerator.generate_payload_positions(
        session_counter=rec_header.session_counter,
        payload_length_bytes=rec_header.payload_length,
        subband_dim=sub_dim,
    )
    extracted_payload_bits = [int(ext_res.extracted_binary[0, 0, r, c].item()) for r, c in rec_payload_positions]
    extracted_protected = Extractor.bits_to_bytes(extracted_payload_bits)

    err_diag = measure_rs_codeword_errors(protected, extracted_protected)
    assert err_diag["within_t16_limit"] is True

    rs_res = ecc_encoder.decode_with_result(extracted_protected)
    assert rs_res.data == ciphertext

    decrypted = crypto_provider.decrypt(rs_res.data, fixed_key, aad=fixed_aad)
    recovered = SecretSerializer.deserialize(decrypted)
    assert recovered.data == secret_payload.data
    assert recovered.metadata == secret_payload.metadata


# =============================================================================
# 2. Capacity and Framing Boundary Agreement Tests
# =============================================================================

def test_production_capacity_and_framing_boundaries_agreement():
    """Verifies that CapacityCalculator, CarrierHeader, and DeterministicMaskGenerator
    agree exactly on capacity boundaries and semantics."""
    subband_dim = 256
    raw_subband_bits = subband_dim * subband_dim  # 65,536
    raw_subband_bytes = raw_subband_bits // 8     # 8,192
    header_bytes = CarrierHeader.HEADER_SIZE      # 16
    header_bits = header_bytes * 8               # 128
    max_protected_bytes = raw_subband_bytes - header_bytes  # 8,176
    max_payload_bits = max_protected_bytes * 8              # 65,408

    # Check constant agreement
    assert CapacityCalculator.RAW_CAPACITY_BYTES == raw_subband_bytes
    assert CapacityCalculator.CARRIER_HEADER_BYTES == header_bytes
    assert CapacityCalculator.MAX_PROTECTED_BYTES == max_protected_bytes

    assert DeterministicMaskGenerator.HEADER_BITS == header_bits
    assert DeterministicMaskGenerator.TOTAL_POSITIONS == raw_subband_bits
    assert DeterministicMaskGenerator.MAX_PAYLOAD_BITS == max_payload_bits

    # Boundary: 8,176 B fits
    rep_8176 = CapacityCalculator.check_capacity(max_protected_bytes, subband_dim=subband_dim)
    assert rep_8176.fits is True
    assert rep_8176.remaining_capacity_bytes == 0

    coords_8176 = DeterministicMaskGenerator.generate_payload_positions(
        session_counter=1, payload_length_bytes=max_protected_bytes, subband_dim=subband_dim
    )
    assert len(coords_8176) == max_payload_bits

    # Boundary: 8,177 B rejected
    with pytest.raises(PayloadCapacityExceededError):
        CapacityCalculator.check_capacity(max_protected_bytes + 1, subband_dim=subband_dim)

    with pytest.raises(MaskGenerationError):
        DeterministicMaskGenerator.generate_payload_positions(
            session_counter=1, payload_length_bytes=max_protected_bytes + 1, subband_dim=subband_dim
        )


def test_overflow_rejected_before_carrier_modification(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    fixed_key: bytes,
    fixed_aad: bytes,
):
    """Verifies that an overflow payload is rejected before carrier modification."""
    carrier = torch.full((1, 3, 512, 512), 128.0)
    carrier_clone = carrier.clone()

    oversized = SecretPayload(data=os.urandom(9000), media_type=MediaType.IMAGE, metadata={})
    serialized = SecretSerializer.serialize(oversized)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)

    with pytest.raises(PayloadCapacityExceededError):
        CapacityCalculator.check_capacity(len(protected))

    assert torch.equal(carrier, carrier_clone)


# =============================================================================
# 3. Header & Extraction Fail-Closed Validation Tests
# =============================================================================

def test_corrupted_carrier_header_framing_rejected():
    """Verifies that corrupted magic, version, or length prefixes in CarrierHeader fail closed."""
    valid_hdr = CarrierHeader(session_counter=10, payload_length=500).pack()
    assert CarrierHeader.unpack(valid_hdr).payload_length == 500

    # 1. Corrupted Magic
    bad_magic = b"XXXX" + valid_hdr[4:]
    with pytest.raises(CarrierHeaderError, match="Invalid carrier header magic"):
        CarrierHeader.unpack(bad_magic)

    # 2. Unsupported Version (0x02)
    bad_version = valid_hdr[:4] + b"\x02" + valid_hdr[5:]
    with pytest.raises(CarrierHeaderError, match="Unsupported carrier header version"):
        CarrierHeader.unpack(bad_version)

    # 3. Truncated Header
    truncated = valid_hdr[:12]
    with pytest.raises(CarrierHeaderError, match="Carrier header truncated"):
        CarrierHeader.unpack(truncated)


def test_stego_container_header_corruption_fails_closed(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    spatial_bridge: NeuralSpatialBridge,
    fixed_key: bytes,
    fixed_aad: bytes,
):
    """Verifies that spatial corruption in the header anchor positions fails closed."""
    session_counter = 88
    payload_len = 300
    secret_payload = SecretPayload(data=os.urandom(payload_len), media_type=MediaType.TEXT, metadata={})
    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)

    header = CarrierHeader(session_counter=session_counter, payload_length=len(protected))
    header_bytes = header.pack()
    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    payload_bits = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))

    mask = DeterministicMaskGenerator.generate_mask(session_counter, len(protected), subband_dim=256)

    mask_tensor = torch.zeros(1, 1, 256, 256)
    payload_tensor = torch.zeros(1, 1, 256, 256)

    for bit, (r, c) in zip(header_bits, mask.header_positions):
        mask_tensor[0, 0, r, c] = 1.0
        payload_tensor[0, 0, r, c] = 1.0 if bit == 1 else -1.0

    for bit, (r, c) in zip(payload_bits, mask.payload_positions):
        mask_tensor[0, 0, r, c] = 1.0
        payload_tensor[0, 0, r, c] = 1.0 if bit == 1 else -1.0

    carrier = torch.full((1, 3, 512, 512), 128.0)
    emb_res = spatial_bridge.embed_spatial(carrier, payload_tensor, mask_tensor, differentiable=False)
    corrupted_stego = emb_res.stego_image.clone()

    # Corrupt spatial pixels in top-left region covering header anchor positions
    corrupted_stego[:, :, 0:10, 0:10] = 0.0

    ext_res = spatial_bridge.extract_spatial(corrupted_stego)
    extracted_hdr_bits = [int(ext_res.extracted_binary[0, 0, r, c].item()) for r, c in mask.header_positions]
    extracted_hdr_bytes = Extractor.bits_to_bytes(extracted_hdr_bits)

    # Extraction must fail closed via CarrierHeaderError
    with pytest.raises(CarrierHeaderError):
        CarrierHeader.unpack(extracted_hdr_bytes)


# =============================================================================
# 4. Stego Container Tampering & Plaintext Leakage Prevention
# =============================================================================

def test_tampered_payload_region_fails_closed_no_plaintext_leak(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    spatial_bridge: NeuralSpatialBridge,
    fixed_key: bytes,
    fixed_aad: bytes,
):
    """Verifies that tampering with the payload region fails closed (ECC or AES tag)
    and leaks zero plaintext bytes."""
    session_counter = 99
    secret_payload = SecretPayload(data=b"TOP_SECRET_CLASSIFIED_PAYLOAD_BYTES", media_type=MediaType.TEXT, metadata={})
    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)

    header = CarrierHeader(session_counter=session_counter, payload_length=len(protected))
    header_bytes = header.pack()
    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    payload_bits = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))

    mask = DeterministicMaskGenerator.generate_mask(session_counter, len(protected), subband_dim=256)

    mask_tensor = torch.zeros(1, 1, 256, 256)
    payload_tensor = torch.zeros(1, 1, 256, 256)

    for bit, (r, c) in zip(header_bits, mask.header_positions):
        mask_tensor[0, 0, r, c] = 1.0
        payload_tensor[0, 0, r, c] = 1.0 if bit == 1 else -1.0

    for bit, (r, c) in zip(payload_bits, mask.payload_positions):
        mask_tensor[0, 0, r, c] = 1.0
        payload_tensor[0, 0, r, c] = 1.0 if bit == 1 else -1.0

    carrier = torch.full((1, 3, 512, 512), 128.0)
    emb_res = spatial_bridge.embed_spatial(carrier, payload_tensor, mask_tensor, differentiable=False)
    corrupted_stego = emb_res.stego_image.clone()

    # Corrupt spatial pixels across payload region, exceeding t=16 error capacity while leaving header intact
    corrupted_stego[:, :, 50:450, 50:450] = 255.0

    ext_res = spatial_bridge.extract_spatial(corrupted_stego)

    extracted_hdr_bits = [int(ext_res.extracted_binary[0, 0, r, c].item()) for r, c in mask.header_positions]
    extracted_hdr_bytes = Extractor.bits_to_bytes(extracted_hdr_bits)
    rec_header = CarrierHeader.unpack(extracted_hdr_bytes)

    rec_payload_positions = DeterministicMaskGenerator.generate_payload_positions(
        session_counter=rec_header.session_counter,
        payload_length_bytes=rec_header.payload_length,
        subband_dim=256,
    )
    extracted_payload_bits = [int(ext_res.extracted_binary[0, 0, r, c].item()) for r, c in rec_payload_positions]
    extracted_protected = Extractor.bits_to_bytes(extracted_payload_bits)

    tamper_caught = False
    recovered_plaintext = None
    try:
        rs_res = ecc_encoder.decode_with_result(extracted_protected)
        decrypted = crypto_provider.decrypt(rs_res.data, fixed_key, aad=fixed_aad)
        recovered_plaintext = SecretSerializer.deserialize(decrypted)
    except (ECCDecodeError, AuthenticationError):
        tamper_caught = True

    assert tamper_caught is True
    assert recovered_plaintext is None


# =============================================================================
# 5. Production Interface Coupling / Architectural Gap Audit Test
# =============================================================================

def test_direct_production_carrier_interface_incompatibility_defect():
    """Documents and proves the exact architectural gap between Phase 2's
    classical BaselineEmbedder/Extractor and Phase 3's NeuralSpatialBridge:

    1. BaselineEmbedder in src/carrier/embedding.py hardcodes classical integer-parity
       modulation on spatial pixels ((a - b - c + d) mod 2). It has no interface to
       execute NeuralSpatialBridge or ACSSubINN.
    2. Extractor in src/carrier/extraction.py hardcodes classical integer-parity extraction.
    3. When a neural stego carrier is passed to Extractor.extract(), it raises CarrierHeaderError
       because continuous wavelet modulation does not preserve integer parity.
    4. MediaStegoPipeline in src/core/media_pipeline.py is hardcoded to BaselineEmbedder and Extractor.
    """
    bridge = NeuralSpatialBridge()
    carrier = torch.full((1, 3, 512, 512), 128.0)
    payload_tensor = torch.zeros((1, 1, 256, 256))
    mask_tensor = torch.zeros((1, 1, 256, 256))

    # Embed a dummy header via neural bridge
    header_bytes = CarrierHeader(session_counter=1, payload_length=100).pack()
    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    header_positions = DeterministicMaskGenerator.generate_header_positions(subband_dim=256)
    for bit, (r, c) in zip(header_bits, header_positions):
        mask_tensor[0, 0, r, c] = 1.0
        payload_tensor[0, 0, r, c] = 1.0 if bit == 1 else -1.0

    emb_res = bridge.embed_spatial(carrier, payload_tensor, mask_tensor, differentiable=False)
    stego_pil = Image.fromarray(emb_res.stego_image.squeeze(0).permute(1, 2, 0).numpy())

    # Passing neural stego image to classical Extractor.extract fails with CarrierHeaderError
    with pytest.raises(CarrierHeaderError, match="Invalid carrier header magic"):
        Extractor.extract(stego_pil)

    # Passing neural stego image to MediaStegoPipeline.extract_and_unprotect fails with CarrierHeaderError
    pipeline = MediaStegoPipeline()
    with pytest.raises(CarrierHeaderError, match="Invalid carrier header magic"):
        pipeline.extract_and_unprotect(stego_pil, key=bytes(32))


# =============================================================================
# 6. Model Parameter Immutability Test
# =============================================================================

def test_model_parameter_immutability(spatial_bridge: NeuralSpatialBridge):
    """Verifies that model parameters remain strictly invariant after all operations."""
    initial_digest = compute_model_param_digest(spatial_bridge.acs_model)

    carrier = torch.full((1, 3, 512, 512), 128.0)
    payload = torch.ones((1, 1, 256, 256))
    mask = torch.ones((1, 1, 256, 256))

    emb = spatial_bridge.embed_spatial(carrier, payload, mask, differentiable=False)
    ext = spatial_bridge.extract_spatial(emb.stego_image)

    post_digest = compute_model_param_digest(spatial_bridge.acs_model)
    assert initial_digest == post_digest, "Model weights drifted during inference!"
