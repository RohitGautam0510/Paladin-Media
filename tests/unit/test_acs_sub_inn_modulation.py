"""Unit tests for ACSSubINNModulationEngine and modulation engine factory."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.carrier.capacity import CapacityCalculator
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator
from src.carrier.modulation import (
    ACSSubINNModulationEngine,
    NeuralModulationEngine,
    ParityModulationEngine,
    create_modulation_engine,
)
from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization import MediaType, SecretPayload
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


@pytest.fixture
def reproducible_cover():
    """Generates reproducible 512x512 RGB synthetic carrier."""
    rng = np.random.RandomState(42)
    return rng.randint(40, 215, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def secret_payload():
    return SecretPayload(
        data=b"ACSSubINN Subband Modulation Test Payload 1234567890",
        media_type=MediaType.TEXT,
        metadata={"test": "subband_onnx"},
    )


def test_create_modulation_engine_factory():
    """Verifies that create_modulation_engine instantiates correct classes for all valid keys."""
    eng_spatial = create_modulation_engine("neural_spatial_onnx")
    assert isinstance(eng_spatial, NeuralModulationEngine)
    assert eng_spatial.backend == "onnx"

    eng_subband = create_modulation_engine("acs_sub_inn_onnx")
    assert isinstance(eng_subband, ACSSubINNModulationEngine)

    eng_subband_via_neural = NeuralModulationEngine(backend="acs_sub_inn_onnx")
    assert eng_subband_via_neural.backend == "subband_onnx"
    assert eng_subband_via_neural._subband_engine is not None

    eng_parity = create_modulation_engine("parity_classical")
    assert isinstance(eng_parity, ParityModulationEngine)

    with pytest.raises(ValueError, match="Unknown modulation engine"):
        create_modulation_engine("invalid_engine_name")


def test_acs_sub_inn_modulation_roundtrip(reproducible_cover, secret_payload):
    """Verifies end-to-end embedding and extraction using ACSSubINNModulationEngine with MediaStegoPipeline."""
    crypto = AESGCMProvider()
    ecc = ReedSolomonEncoder(parity_symbols=48)
    pipeline = MediaStegoPipeline(
        crypto_provider=crypto,
        ecc_encoder=ecc,
        backend="acs_sub_inn_onnx",
    )

    key = b"01234567890123456789012345678901"
    stego_res = pipeline.protect_and_embed(
        secret_payload=secret_payload,
        cover_image=reproducible_cover,
        key=key,
        session_counter=42,
    )

    assert stego_res.stego_image is not None
    assert stego_res.psnr_db > 30.0
    assert stego_res.ssim > 0.90

    rec_res = pipeline.extract_and_unprotect(
        stego_carrier=stego_res.stego_png_bytes,
        key=key,
        reference_payload=secret_payload,
    )

    assert rec_res.exact_match is True
    assert rec_res.recovered_payload.data == secret_payload.data
    assert rec_res.recovered_payload.media_type == secret_payload.media_type
