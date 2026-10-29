"""End-to-End Integration Gate & Integrity Tests for Phase 3 Sprint 3.3.

Validates the full byte-level roundtrip across the integrated architecture:
Plaintext (SecretPayload)
    -> SecretSerializer (binary serialization + framing)
    -> AESGCMProvider (AES-256-GCM authenticated encryption)
    -> ReedSolomonEncoder (RS(255, 223), t=16 symbol error correction)
    -> CapacityCalculator (fail-closed capacity gate)
    -> NeuralSpatialBridge (Haar DWT -> ACS-Sub-INN v3 Red-HH embedding -> IDWT)
    -> Spatial Quantization (uint8 clamping)
    -> Lossless PNG save and reload (in-memory or filesystem)
    -> NeuralSpatialBridge Extractor (DWT -> ACS-Sub-INN v3 extraction branch)
    -> Bit Unpacking / Packing & Raw BER Measurement
    -> ReedSolomonEncoder (RS decoding, symbol error correction & headroom)
    -> AESGCMProvider (GHASH tag authentication & decryption)
    -> SecretSerializer (deserialization & checksum)
    -> Plaintext byte-for-byte exact equality verification.

CRITICAL SAFETY CONTRACT:
- Model weights are strictly UNTRAINED (initial state).
- NO optimizer updates, NO optimizer.step(), NO learning.
- 100% fail-closed on tampering and overflow.
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
from src.core.exceptions import AuthenticationError, ECCDecodeError
from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.dataset import DIV2KStegoDataset
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


# =============================================================================
# Helper Functions
# =============================================================================

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

    # Full blocks
    for _ in range(num_full_blocks):
        block_len = k + ecc_symbols
        p_block = protected[offset : offset + block_len]
        e_block = extracted[offset : offset + block_len]
        errs = sum(1 for p, e in zip(p_block, e_block) if p != e)
        block_errors.append(errs)
        offset += block_len

    # Final shortened block
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


def compute_model_param_digest(model: torch.nn.Module) -> str:
    """Computes a SHA-256 digest over all parameters in a PyTorch module."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


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
    return bytes.fromhex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")


@pytest.fixture
def fixed_aad() -> bytes:
    return b"PaladinMedia-Phase3-Sprint3.3-AAD"


@pytest.fixture
def synthetic_carrier() -> torch.Tensor:
    """Mid-gray 512x512 synthetic carrier (no clipping baseline)."""
    return torch.full((1, 3, 512, 512), 128.0, dtype=torch.float32)


@pytest.fixture
def div2k_carrier() -> torch.Tensor:
    """Canonical DIV2K 0001.png center crop (512x512)."""
    from src.neural.dataset import DEFAULT_DATASET_REL_PATH
    repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / DEFAULT_DATASET_REL_PATH).is_dir():
        pytest.skip(
            f"DIV2K dataset root missing at '{repo_root / DEFAULT_DATASET_REL_PATH}'. "
            f"Download DIV2K_train_HR per README to run dataset integration tests."
        )
    ds = DIV2KStegoDataset(split="train")
    return ds[0]["carrier"].unsqueeze(0)


# =============================================================================
# 1. Baseline Roundtrip Tests (Task B & Task C: Baseline)
# =============================================================================

def test_end_to_end_baseline_synthetic_carrier(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    spatial_bridge: NeuralSpatialBridge,
    fixed_key: bytes,
    fixed_aad: bytes,
    synthetic_carrier: torch.Tensor,
):
    """Verifies baseline 300 B payload roundtrip on smooth synthetic carrier.

    Expectations:
    - 0 clipped pixels.
    - Raw pre-ECC BER == 0.000000.
    - RS corrected symbols == 0 (0 codeword errors).
    - Valid AES-GCM tag.
    - 100% exact plaintext recovery.
    - Visual fidelity: PSNR >= 60 dB, SSIM >= 0.9995.
    """
    payload_len = 300
    raw_secret = os.urandom(payload_len)
    secret_payload = SecretPayload(
        data=raw_secret,
        media_type=MediaType.IMAGE,
        metadata={"sprint": "3.3", "carrier": "synthetic", "bytes": payload_len},
    )

    # 1. Serialize -> Encrypt -> RS Encode -> Capacity Check
    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)
    cap_report = CapacityCalculator.check_capacity(len(protected))
    assert cap_report.fits is True

    # 2. Bit Mapping
    bits_1d = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))
    num_bits = len(bits_1d)
    mask = torch.zeros(1, 1, 256, 256)
    payload_tensor = torch.zeros(1, 1, 256, 256)
    mask.view(-1)[:num_bits] = 1.0
    payload_tensor.view(-1)[:num_bits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

    # 3. Spatial Bridge Roundtrip (with lossless PNG save & reload)
    emb_res, ext_res, metrics = spatial_bridge.run_full_roundtrip(
        carrier=synthetic_carrier,
        payload_bits=(payload_tensor > 0).to(torch.uint8),
        mask=mask,
        reload_png=True,
    )

    # 4. Receiver Extraction & Decoding
    ext_bits = ext_res.extracted_binary.view(-1)[:num_bits].cpu().numpy().astype(np.uint8)
    raw_ber = float(np.mean(ext_bits != bits_1d))
    assert raw_ber == 0.0, f"Expected 0.0 BER on synthetic carrier, got {raw_ber}"

    ext_bytes = np.packbits(ext_bits).tobytes()
    err_diag = measure_rs_codeword_errors(protected, ext_bytes)
    assert err_diag["worst_codeword_errors"] == 0
    assert err_diag["total_codeword_errors"] == 0

    rs_res = ecc_encoder.decode_with_result(ext_bytes)
    assert rs_res.data == ciphertext
    assert rs_res.corrected_symbols == 0

    decrypted = crypto_provider.decrypt(rs_res.data, fixed_key, aad=fixed_aad)
    assert decrypted == serialized

    recovered = SecretSerializer.deserialize(decrypted)
    assert recovered.data == secret_payload.data
    assert recovered.metadata == secret_payload.metadata
    assert recovered.media_type == secret_payload.media_type

    # High fidelity on synthetic carrier
    assert metrics.psnr_db >= 60.0, f"PSNR too low: {metrics.psnr_db} dB"
    assert metrics.ssim >= 0.9995, f"SSIM too low: {metrics.ssim}"


def test_end_to_end_baseline_div2k_carrier(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    spatial_bridge: NeuralSpatialBridge,
    fixed_key: bytes,
    fixed_aad: bytes,
    div2k_carrier: torch.Tensor,
):
    """Verifies baseline 300 B payload roundtrip on natural DIV2K 0001.png carrier.

    Expectations:
    - Untrained model may experience clipping / low raw BER.
    - Reed-Solomon absorbs errors (worst codeword errors <= 16).
    - Post-ECC BER == 0.000000.
    - Valid AES-GCM tag.
    - 100% exact plaintext recovery.
    """
    payload_len = 300
    raw_secret = os.urandom(payload_len)
    secret_payload = SecretPayload(
        data=raw_secret,
        media_type=MediaType.IMAGE,
        metadata={"sprint": "3.3", "carrier": "0001.png", "bytes": payload_len},
    )

    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)
    cap_report = CapacityCalculator.check_capacity(len(protected))
    assert cap_report.fits is True

    bits_1d = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))
    num_bits = len(bits_1d)
    mask = torch.zeros(1, 1, 256, 256)
    payload_tensor = torch.zeros(1, 1, 256, 256)
    mask.view(-1)[:num_bits] = 1.0
    payload_tensor.view(-1)[:num_bits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

    emb_res, ext_res, metrics = spatial_bridge.run_full_roundtrip(
        carrier=div2k_carrier,
        payload_bits=(payload_tensor > 0).to(torch.uint8),
        mask=mask,
        reload_png=True,
    )

    ext_bits = ext_res.extracted_binary.view(-1)[:num_bits].cpu().numpy().astype(np.uint8)
    raw_ber = float(np.mean(ext_bits != bits_1d))

    ext_bytes = np.packbits(ext_bits).tobytes()
    err_diag = measure_rs_codeword_errors(protected, ext_bytes)
    assert err_diag["within_t16_limit"] is True, f"Worst codeword errors {err_diag['worst_codeword_errors']} exceeds 16"

    rs_res = ecc_encoder.decode_with_result(ext_bytes)
    assert rs_res.data == ciphertext

    decrypted = crypto_provider.decrypt(rs_res.data, fixed_key, aad=fixed_aad)
    recovered = SecretSerializer.deserialize(decrypted)
    assert recovered.data == secret_payload.data
    assert recovered.metadata == secret_payload.metadata


# =============================================================================
# 2. Higher-Load Roundtrip Tests (Task C: Higher-Load)
# =============================================================================

@pytest.mark.parametrize("payload_len", [1500, 3500])
@pytest.mark.parametrize("carrier_type", ["synthetic", "div2k"])
def test_end_to_end_higher_load_roundtrip(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    spatial_bridge: NeuralSpatialBridge,
    fixed_key: bytes,
    fixed_aad: bytes,
    synthetic_carrier: torch.Tensor,
    request: pytest.FixtureRequest,
    payload_len: int,
    carrier_type: str,
):
    """Verifies higher-load roundtrips (1,500 B and 3,500 B) across synthetic and natural DIV2K carriers."""
    carrier = synthetic_carrier if carrier_type == "synthetic" else request.getfixturevalue("div2k_carrier")

    raw_secret = os.urandom(payload_len)
    secret_payload = SecretPayload(
        data=raw_secret,
        media_type=MediaType.IMAGE,
        metadata={"load": payload_len, "carrier": carrier_type},
    )

    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)

    cap_report = CapacityCalculator.check_capacity(len(protected))
    assert cap_report.fits is True

    bits_1d = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))
    num_bits = len(bits_1d)
    mask = torch.zeros(1, 1, 256, 256)
    payload_tensor = torch.zeros(1, 1, 256, 256)
    mask.view(-1)[:num_bits] = 1.0
    payload_tensor.view(-1)[:num_bits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

    emb_res, ext_res, metrics = spatial_bridge.run_full_roundtrip(
        carrier=carrier,
        payload_bits=(payload_tensor > 0).to(torch.uint8),
        mask=mask,
        reload_png=True,
    )

    ext_bits = ext_res.extracted_binary.view(-1)[:num_bits].cpu().numpy().astype(np.uint8)
    ext_bytes = np.packbits(ext_bits).tobytes()

    err_diag = measure_rs_codeword_errors(protected, ext_bytes)
    assert err_diag["within_t16_limit"] is True

    rs_res = ecc_encoder.decode_with_result(ext_bytes)
    assert rs_res.data == ciphertext

    decrypted = crypto_provider.decrypt(rs_res.data, fixed_key, aad=fixed_aad)
    recovered = SecretSerializer.deserialize(decrypted)
    assert recovered.data == secret_payload.data
    assert recovered.metadata == secret_payload.metadata


# =============================================================================
# 3. Capacity Boundary Enforcement Tests (Task C: Capacity Boundary)
# =============================================================================

def test_capacity_boundary_enforcement():
    """Verifies exact 8,176 B vs 8,177 B capacity boundary enforcement.

    Accounting (512x512 carrier, 256x256 subband):
    - Raw subband capacity: 256 * 256 = 65,536 bits = 8,192 bytes.
    - Carrier header: 16 bytes.
    - Maximum protected payload: 8,192 - 16 = 8,176 bytes.
    - 8,176 bytes: fits.
    - 8,177 bytes: rejected fail-closed with PayloadCapacityExceededError.
    """
    # 1. 8,176 B protected payload passes
    report_valid = CapacityCalculator.check_capacity(8176)
    assert report_valid.fits is True
    assert report_valid.status == "FITS"
    assert report_valid.protected_payload_bytes == 8176
    assert report_valid.remaining_capacity_bytes == 0

    # 2. 8,177 B protected payload raises PayloadCapacityExceededError
    with pytest.raises(PayloadCapacityExceededError, match="PAYLOAD_CAPACITY_EXCEEDED: Protected payload .*8177.* exceeds maximum carrier capacity .*8176"):
        CapacityCalculator.check_capacity(8177)

    # 3. Maximum usable plaintext capacity derivation
    usable_0 = CapacityCalculator.calculate_usable_secret_capacity(metadata_len=0)
    assert usable_0 == 7090

    usable_2 = CapacityCalculator.calculate_usable_secret_capacity(metadata_len=2)
    assert usable_2 == 7088


def test_fail_closed_overflow_aborts_before_embedding(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    synthetic_carrier: torch.Tensor,
    fixed_key: bytes,
    fixed_aad: bytes,
):
    """Verifies that an oversized payload fails closed before any carrier modification."""
    # Construct an oversized 10,000 B payload
    oversized_data = os.urandom(10000)
    secret_payload = SecretPayload(
        data=oversized_data,
        media_type=MediaType.IMAGE,
        metadata={},
    )
    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)

    # Snapshot carrier before check
    carrier_copy = synthetic_carrier.clone()

    # Capacity check MUST fail closed
    with pytest.raises(PayloadCapacityExceededError):
        CapacityCalculator.check_capacity(len(protected))

    # Carrier tensor remains completely untouched
    assert torch.equal(synthetic_carrier, carrier_copy)


# =============================================================================
# 4. Tampered Stego Input Tests (Task C: Tampered Stego)
# =============================================================================

def test_tampered_stego_input_fails_closed(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    spatial_bridge: NeuralSpatialBridge,
    fixed_key: bytes,
    fixed_aad: bytes,
    synthetic_carrier: torch.Tensor,
):
    """Verifies that tampering with the stego container fails closed, leaking 0 plaintext."""
    payload_len = 500
    secret_payload = SecretPayload(
        data=b"HighlyConfidentialPaladinSecretBytes",
        media_type=MediaType.TEXT,
        metadata={"classification": "TOP_SECRET"},
    )

    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)

    bits_1d = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))
    num_bits = len(bits_1d)
    mask = torch.zeros(1, 1, 256, 256)
    payload_tensor = torch.zeros(1, 1, 256, 256)
    mask.view(-1)[:num_bits] = 1.0
    payload_tensor.view(-1)[:num_bits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

    # Embed stego carrier
    emb_res = spatial_bridge.embed_spatial(
        carrier=synthetic_carrier,
        payload_bipolar=payload_tensor,
        mask=mask,
        differentiable=False,
    )

    # Tamper with stego spatial image (corrupt coefficients in embedding region)
    corrupted_stego = emb_res.stego_image.clone()
    corrupted_stego[:, :, 0:128, 0:128] = 0.0

    # Extract from corrupted stego
    ext_res = spatial_bridge.extract_spatial(corrupted_stego, mask=mask)
    ext_bits = ext_res.extracted_binary.view(-1)[:num_bits].cpu().numpy().astype(np.uint8)
    ext_bytes = np.packbits(ext_bits).tobytes()

    recovered_secret = None
    tamper_caught = False

    try:
        rs_res = ecc_encoder.decode_with_result(ext_bytes)
        decrypted = crypto_provider.decrypt(rs_res.data, fixed_key, aad=fixed_aad)
        recovered_secret = SecretSerializer.deserialize(decrypted)
    except (ECCDecodeError, AuthenticationError) as exc:
        tamper_caught = True

    assert tamper_caught is True, "Pipeline failed to detect stego container tampering!"
    assert recovered_secret is None, "Plaintext was leaked despite stego container corruption!"


# =============================================================================
# 5. Untrained Model Invariance Test (Task C: Untrained Status)
# =============================================================================

def test_untrained_model_weights_invariance(
    crypto_provider: AESGCMProvider,
    ecc_encoder: ReedSolomonEncoder,
    spatial_bridge: NeuralSpatialBridge,
    fixed_key: bytes,
    fixed_aad: bytes,
    div2k_carrier: torch.Tensor,
):
    """Verifies that neural model weights remain strictly invariant during inference roundtrips.

    Ensures that:
    - Initialized state has not been trained (gamma_min=1.0).
    - No gradients are retained or updated during inference.
    - Parameter bit-digest is identical before and after inference.
    """
    initial_digest = compute_model_param_digest(spatial_bridge.acs_model)

    # Run roundtrip
    payload_len = 1000
    secret_payload = SecretPayload(data=os.urandom(payload_len), media_type=MediaType.IMAGE, metadata={})
    serialized = SecretSerializer.serialize(secret_payload)
    ciphertext = crypto_provider.encrypt(serialized, fixed_key, aad=fixed_aad)
    protected = ecc_encoder.encode(ciphertext)

    bits_1d = np.unpackbits(np.frombuffer(protected, dtype=np.uint8))
    num_bits = len(bits_1d)
    mask = torch.zeros(1, 1, 256, 256)
    payload_tensor = torch.zeros(1, 1, 256, 256)
    mask.view(-1)[:num_bits] = 1.0
    payload_tensor.view(-1)[:num_bits] = torch.from_numpy(bits_1d.astype(np.float32) * 2.0 - 1.0)

    emb_res, ext_res, metrics = spatial_bridge.run_full_roundtrip(
        carrier=div2k_carrier,
        payload_bits=(payload_tensor > 0).to(torch.uint8),
        mask=mask,
        reload_png=True,
    )

    post_digest = compute_model_param_digest(spatial_bridge.acs_model)
    assert initial_digest == post_digest, "Model weights mutated during inference roundtrip!"
