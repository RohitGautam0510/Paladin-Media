"""Focused unit tests for Phase 5A neural inference graph boundaries and contracts."""

from __future__ import annotations

import hashlib
import io
import pytest
import torch
import numpy as np
from PIL import Image

from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator
from src.carrier.modulation import NeuralModulationEngine
from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization import MediaType, SecretPayload
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.onnx_export import (
    ACSSubINNEmbeddingGraph,
    ACSSubINNExtractionGraph,
    NeuralSpatialEmbeddingGraph,
    NeuralSpatialExtractionGraph,
    check_onnx_dependencies,
    compute_numerical_parity,
)
from src.neural.spatial_bridge import NeuralSpatialBridge
from src.neural.training_engine import load_training_checkpoint
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


@pytest.fixture(scope="module")
def loaded_bridge():
    """Loads Phase 4C accepted checkpoint for inference boundary testing."""
    ckpt_path = "checkpoints/phase4c_extended_30ep/best_model.pt"
    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    load_training_checkpoint(ckpt_path, bridge=bridge)
    bridge.to("cpu")
    bridge.eval()
    return bridge


def test_dependency_checker_handles_missing_packages():
    """Confirms dependency audit runs safely without exceptions when packages are missing."""
    status = check_onnx_dependencies()
    assert isinstance(status.torch_version, str)
    assert isinstance(status.onnx_available, bool)
    assert isinstance(status.onnxruntime_available, bool)
    assert isinstance(status.onnxruntime_providers, list)


def test_coefficient_embedding_graph_contract(loaded_bridge):
    """Verifies ACSSubINNEmbeddingGraph input/output contract and finite outputs."""
    graph = ACSSubINNEmbeddingGraph(loaded_bridge.acs_model)
    b, c, h, w = 1, 1, 64, 64
    hh_drop = torch.randn(b, 1, h, w)
    hh_keep = torch.randn(b, 2, h, w)
    mask = torch.ones(b, 1, h, w)
    payload_bipolar = torch.ones(b, 1, h, w)

    with torch.no_grad():
        hh_drop_stego, context, gamma = graph(hh_drop, hh_keep, mask, payload_bipolar)

    assert hh_drop_stego.shape == (b, 1, h, w)
    assert context.shape == (b, 1, h, w)
    assert gamma.shape == (b, 1, h, w)
    assert torch.isfinite(hh_drop_stego).all()
    assert (gamma >= 1.0).all(), "gamma must be >= gamma_min (1.0)"


def test_coefficient_extraction_graph_contract(loaded_bridge):
    """Verifies ACSSubINNExtractionGraph input/output contract and binary output values."""
    graph = ACSSubINNExtractionGraph(loaded_bridge.acs_model)
    b, h, w = 1, 64, 64
    hh_drop_stego = torch.randn(b, 1, h, w)
    hh_keep = torch.randn(b, 2, h, w)

    with torch.no_grad():
        delta_rec, context_rec, extracted_binary = graph(hh_drop_stego, hh_keep)

    assert delta_rec.shape == (b, 1, h, w)
    assert context_rec.shape == (b, 1, h, w)
    assert extracted_binary.shape == (b, 1, h, w)
    assert torch.isfinite(delta_rec).all()
    # Check binary values are strictly in {0.0, 1.0}
    is_binary = ((extracted_binary == 0.0) | (extracted_binary == 1.0)).all()
    assert is_binary.item() is True


def test_spatial_inference_graphs_dynamic_shapes(loaded_bridge):
    """Verifies spatial inference graphs across dynamic spatial dimensions (256x256 and 512x512)."""
    embed_graph = NeuralSpatialEmbeddingGraph(loaded_bridge)
    extract_graph = NeuralSpatialExtractionGraph(loaded_bridge)

    for spatial_dim in [256, 512]:
        sub_dim = spatial_dim // 2
        carrier = torch.randint(0, 256, (1, 3, spatial_dim, spatial_dim)).float()
        mask = torch.ones(1, 1, sub_dim, sub_dim)
        payload_bipolar = torch.ones(1, 1, sub_dim, sub_dim)

        with torch.no_grad():
            stego_img, ctx, gamma = embed_graph(carrier, mask, payload_bipolar)
            assert stego_img.shape == (1, 3, spatial_dim, spatial_dim)
            assert (stego_img >= 0.0).all() and (stego_img <= 255.0).all()

            extracted_bin, delta = extract_graph(stego_img)
            assert extracted_bin.shape == (1, 1, sub_dim, sub_dim)
            assert delta.shape == (1, 1, sub_dim, sub_dim)


def test_independent_receiver_recovery_contract(loaded_bridge):
    """Verifies that MediaStegoPipeline.extract_and_unprotect functions independently without reference_payload."""
    stego_png_path = "checkpoints/phase4c_extended_30ep/e2e_stego_carrier.png"
    stego_bytes = open(stego_png_path, "rb").read()

    pipeline = MediaStegoPipeline(
        crypto_provider=AESGCMProvider(),
        ecc_encoder=ReedSolomonEncoder(ecc_symbols=32, block_size=255),
        modulation_engine=NeuralModulationEngine(loaded_bridge),
    )

    aes_key = hashlib.sha256(b"PaladinMedia-Phase4C-Key-Deterministic-2026").digest()
    aad = b"Phase4C-Session-42"
    expected_secret_bytes = bytes([((i * 37 + 13) % 256) for i in range(1000)])

    # Independent recovery call: reference_payload is explicitly None
    res = pipeline.extract_and_unprotect(stego_bytes, key=aes_key, aad=aad, reference_payload=None)
    assert res.recovered_payload.data == expected_secret_bytes
    assert res.exact_match is False  # exact_match is False because reference_payload was not provided for comparison
    assert len(res.recovered_payload.data) == 1000


def test_numerical_parity_comparator():
    """Verifies compute_numerical_parity helper with known synthetic arrays."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.000001, 2.000002, 3.000003], dtype=np.float32)

    res = compute_numerical_parity(a, b, rtol=1e-4, atol=1e-5)
    assert res["passed"] is True
    assert res["shape_match"] is True
    assert res["max_absolute_error"] < 1e-5

    c = np.array([1.1, 2.0, 3.0], dtype=np.float32)
    res_fail = compute_numerical_parity(a, c, rtol=1e-4, atol=1e-5)
    assert res_fail["passed"] is False


def test_exported_onnx_models_runtime_parity(loaded_bridge):
    """Verifies that exported ONNX models load in ONNX Runtime and achieve bit-exact extraction."""
    from pathlib import Path
    import onnxruntime as ort

    onnx_dir = Path("checkpoints/phase5a_onnx")
    required_models = [
        "acs_sub_inn_embedding_opset14.onnx",
        "acs_sub_inn_extraction_opset14.onnx",
        "neural_spatial_embedding_opset14.onnx",
        "neural_spatial_extraction_opset14.onnx",
    ]

    for model_name in required_models:
        model_path = onnx_dir / model_name
        assert model_path.exists(), f"Expected exported ONNX model {model_path} to exist"

    # Test spatial extraction in ONNX Runtime vs PyTorch
    sp_ext_path = onnx_dir / "neural_spatial_extraction_opset14.onnx"
    sess = ort.InferenceSession(str(sp_ext_path), providers=["CPUExecutionProvider"])

    test_carrier = torch.randint(0, 256, (1, 3, 256, 256)).float()
    pt_extract_graph = NeuralSpatialExtractionGraph(loaded_bridge)

    with torch.no_grad():
        pt_binary, pt_delta = pt_extract_graph(test_carrier)

    ort_binary, ort_delta = sess.run(None, {"stego_carrier": test_carrier.numpy()})

    # Assert bit-exact agreement
    assert np.array_equal(pt_binary.numpy(), ort_binary), "Extracted binary bits must be bit-exact"
    assert np.all(np.isfinite(ort_delta)), "Extraction delta must be finite"
    assert np.all(np.isfinite(ort_binary)), "Extracted binary must be finite"
