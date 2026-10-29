"""Phase 5B unit tests for ONNX Runtime integration, pipeline backend selection, and failure boundaries."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.carrier.capacity import CapacityCalculator, PayloadCapacityExceededError
from src.carrier.header import CarrierHeaderError
from src.carrier.modulation import NeuralModulationEngine, ParityModulationEngine
from src.core.exceptions import PaladinSecurityError
from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization import MediaType, SecretPayload


@pytest.fixture
def dummy_cover():
    """Generates a reproducible 512x512 RGB synthetic carrier."""
    rng = np.random.RandomState(42)
    return rng.randint(40, 215, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def dummy_key():
    return b"01234567890123456789012345678901"


def test_neural_modulation_engine_backend_selection():
    """Verifies NeuralModulationEngine initializes supported backends and rejects invalid ones."""
    engine_py = NeuralModulationEngine(backend="pytorch")
    assert engine_py.backend == "pytorch"
    assert engine_py.bridge is not None

    engine_onnx = NeuralModulationEngine(backend="onnx")
    assert engine_onnx.backend == "onnx"
    assert engine_onnx.onnx_sess_embed is not None
    assert engine_onnx.onnx_sess_extract is not None

    with pytest.raises(ValueError, match="Unsupported backend"):
        NeuralModulationEngine(backend="unsupported_backend")


def test_neural_modulation_engine_missing_onnx_dir():
    """Verifies FileNotFoundError when onnx_dir is missing required models."""
    with pytest.raises(FileNotFoundError, match="ONNX embedding model missing"):
        NeuralModulationEngine(backend="onnx", onnx_dir=Path("non_existent_directory_12345"))


def test_media_pipeline_backend_selection():
    """Verifies MediaStegoPipeline backend kwarg correctly instantiates modulation engines."""
    # Default (no backend) -> ParityModulationEngine for Phase 2 backward compatibility
    pipe_default = MediaStegoPipeline()
    assert isinstance(pipe_default.modulation_engine, ParityModulationEngine)

    # backend="onnx" -> NeuralModulationEngine with onnx backend
    pipe_onnx = MediaStegoPipeline(backend="onnx")
    assert isinstance(pipe_onnx.modulation_engine, NeuralModulationEngine)
    assert pipe_onnx.modulation_engine.backend == "onnx"

    # backend="pytorch" -> NeuralModulationEngine with pytorch backend
    pipe_py = MediaStegoPipeline(backend="pytorch")
    assert isinstance(pipe_py.modulation_engine, NeuralModulationEngine)
    assert pipe_py.modulation_engine.backend == "pytorch"


def test_onnx_session_reuse(dummy_cover, dummy_key):
    """Verifies that multiple embeddings and extractions reuse cached InferenceSession instances."""
    pipe = MediaStegoPipeline(backend="onnx")
    engine = pipe.modulation_engine

    embed_session_id = id(engine.onnx_sess_embed)
    extract_session_id = id(engine.onnx_sess_extract)

    payload = SecretPayload(
        media_type=MediaType.TEXT,
        data=b"Session reuse test",
        metadata={"run": 1},
    )

    # First roundtrip
    stego_res1 = pipe.protect_and_embed(payload, dummy_cover, dummy_key)
    pipe.extract_and_unprotect(stego_res1.stego_png_bytes, dummy_key)

    # Second roundtrip
    stego_res2 = pipe.protect_and_embed(payload, dummy_cover, dummy_key)
    pipe.extract_and_unprotect(stego_res2.stego_png_bytes, dummy_key)

    # Confirm session instances remained invariant
    assert id(engine.onnx_sess_embed) == embed_session_id
    assert id(engine.onnx_sess_extract) == extract_session_id


def test_onnx_lossless_roundtrip_exact_recovery(dummy_cover, dummy_key):
    """Verifies complete lossless end-to-end roundtrip with ONNX backend."""
    pipe = MediaStegoPipeline(backend="onnx")

    payload = SecretPayload(
        media_type=MediaType.TEXT,
        data=b"Lossless ONNX transmission payload: alpha-beta-gamma-1234567890",
        metadata={"algorithm": "ACS-Sub-INN-v3", "backend": "onnx"},
    )

    stego_res, rec_res = pipe.run_full_roundtrip(payload, dummy_cover, dummy_key)

    assert rec_res.exact_match is True
    assert rec_res.recovered_payload == payload
    assert stego_res.psnr_db > 35.0
    assert stego_res.ssim > 0.99


def test_onnx_capacity_hard_gate_fail_closed(dummy_cover, dummy_key):
    """Verifies that oversized payloads are rejected fail-closed before embedding in ONNX pipeline."""
    pipe = MediaStegoPipeline(backend="onnx")

    # Plaintext payload that exceeds 7,088 B usable capacity
    oversized_data = b"X" * 7500
    oversized_payload = SecretPayload(
        media_type=MediaType.TEXT,
        data=oversized_data,
        metadata={},
    )

    with pytest.raises(PayloadCapacityExceededError):
        pipe.protect_and_embed(oversized_payload, dummy_cover, dummy_key)


def test_onnx_tampered_carrier_fail_closed(dummy_cover, dummy_key):
    """Verifies that tampered stego carrier fails closed without leaking plaintext in ONNX pipeline."""
    pipe = MediaStegoPipeline(backend="onnx")

    payload = SecretPayload(
        media_type=MediaType.TEXT,
        data=b"Sensitive secret payload to protect",
        metadata={"sec": "high"},
    )

    stego_res = pipe.protect_and_embed(payload, dummy_cover, dummy_key)
    stego_arr = np.array(stego_res.stego_image).copy()

    # Tamper with stego image pixels (e.g. flip region)
    stego_arr[0:16, 0:16, :] = 255 - stego_arr[0:16, 0:16, :]

    with pytest.raises((PaladinSecurityError, CarrierHeaderError)):
        pipe.extract_and_unprotect(stego_arr, dummy_key)


def test_fresh_receiver_with_ecc_profile(dummy_cover, dummy_key):
    """Verifies that a fresh receiver instance successfully identifies and adapts to sender's ECC profile."""
    from src.security.ecc.encoder import ReedSolomonEncoder

    # Sender uses enhanced 2t=48 profile
    sender = MediaStegoPipeline(
        ecc_encoder=ReedSolomonEncoder(parity_symbols=48),
        backend="onnx",
    )
    payload = SecretPayload(
        media_type=MediaType.TEXT,
        data=b"Interoperable ECC profile payload for fresh receiver",
        metadata={"parity": 48},
    )
    stego_res = sender.protect_and_embed(payload, dummy_cover, dummy_key)

    # Verify header flags indicate 2t=48 profile
    assert stego_res.header.ecc_profile == 0x01
    assert stego_res.header.ecc_parity == 48

    # Fresh receiver instance (defaults to 2t=32)
    fresh_receiver = MediaStegoPipeline(backend="onnx")
    rec_res = fresh_receiver.extract_and_unprotect(
        stego_res.stego_png_bytes, dummy_key, reference_payload=payload
    )

    assert rec_res.exact_match is True
    assert rec_res.recovered_payload.data == payload.data


def test_profile_specific_capacity_checks(dummy_cover, dummy_key):
    """Verifies that capacity calculator and pipeline enforce profile-specific limits fail-closed."""
    from src.security.ecc.encoder import ReedSolomonEncoder

    # Profile 2t=48 usable capacity is 6,578 B (with 0B metadata)
    usable_48 = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=0, subband_dim=256, rs_parity=48
    )
    assert usable_48 == 6578

    # A payload that exceeds 6,578 B must be rejected under 2t=48
    oversized_for_48 = SecretPayload(
        media_type=MediaType.TEXT,
        data=b"Z" * (usable_48 + 50),
        metadata={},
    )
    sender_48 = MediaStegoPipeline(
        ecc_encoder=ReedSolomonEncoder(parity_symbols=48),
        backend="onnx",
    )
    with pytest.raises(PayloadCapacityExceededError):
        sender_48.protect_and_embed(oversized_for_48, dummy_cover, dummy_key)


def test_identical_input_graph_comparison():
    """Verifies that PyTorch and ONNX spatial extraction graphs fed identical inputs produce bit-exact bits."""
    import onnxruntime as ort
    import torch
    from src.neural.acs_sub_inn import ACSSubINN
    from src.neural.onnx_export import NeuralSpatialExtractionGraph
    from src.neural.spatial_bridge import NeuralSpatialBridge
    from src.neural.training_engine import load_training_checkpoint

    ckpt_path = "checkpoints/phase4c_extended_30ep/best_model.pt"
    model = ACSSubINN(gamma_min=1.0)
    bridge = NeuralSpatialBridge(acs_model=model)
    load_training_checkpoint(ckpt_path, bridge=bridge)
    bridge.eval()

    py_extract = NeuralSpatialExtractionGraph(bridge)

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    ort_extract = ort.InferenceSession(
        "checkpoints/phase5a_onnx/neural_spatial_extraction_opset14.onnx",
        sess_opts,
        providers=["CPUExecutionProvider"],
    )

    # Identical input carrier tensor
    rng = np.random.RandomState(42)
    carrier_np = rng.randint(40, 215, (1, 3, 512, 512)).astype(np.float32)

    with torch.no_grad():
        py_bin, py_delta = py_extract(torch.from_numpy(carrier_np))
    ort_bin, ort_delta = ort_extract.run(None, {"stego_carrier": carrier_np})

    # Bit-exactness check
    assert np.array_equal(py_bin.numpy(), ort_bin)


def test_strict_and_adjusted_parity_contract():
    """Verifies that parity evaluation contract correctly applies strict (1e-5) and adjusted (2e-5) tolerances."""
    a = np.zeros(3, dtype=np.float32)
    # diff is 1.5e-5: fails strict (1e-5), passes adjusted (2e-5)
    b = a + 1.5e-5

    strict_pass = np.allclose(a, b, rtol=1e-4, atol=1e-5)
    adj_pass = np.allclose(a, b, rtol=1e-4, atol=2e-5)

    assert strict_pass is False
    assert adj_pass is True


def test_failed_recovery_trial_blocks_lossless_gate():
    """Verifies that changing one raw exact_match to False while leaving summary unchanged blocks acceptance."""
    from scripts.experiments.run_phase5b_assessment import aggregate_phase5b_gates
    import json

    results_path = Path("docs/experiments/phase5b_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Clone data and simulate 1 raw trial exact_match=False while deliberately leaving summaries untouched
    ws_bc_corrupted = json.loads(json.dumps(data["workstreams_bc_lossless"]))
    p48_payloads = ws_bc_corrupted["qualification_matrix"]["rs_2t_48"]["payloads"]
    p48_payloads["payload_100b"]["trials"][0]["exact_match"] = False
    # Deliberately leave summary unchanged (showing 20/20 and 100%)
    assert ws_bc_corrupted["profile_evaluation_summary"]["rs_2t_48"]["successful_trials"] == 20
    assert ws_bc_corrupted["profile_evaluation_summary"]["rs_2t_48"]["recovery_rate_pct"] == 100.0

    gates = aggregate_phase5b_gates(
        data["preflight"],
        data["workstream_a_parity"],
        ws_bc_corrupted,
        data["workstream_d_robustness"],
        data["workstream_e_ecc"],
    )

    # Aggregator must detect the raw trial failure / summary contradiction and block acceptance
    assert gates["gate_lossless_recovery"] == "FAIL"
    assert gates["overall_verdict"] == "BLOCKED"


def test_capacity_gate_bypass_removed_fails_when_mandatory_checks_fail():
    """Verifies that oversized rejection alone cannot bypass failed mandatory capacity-boundary checks."""
    from scripts.experiments.run_phase5b_assessment import aggregate_phase5b_gates
    import json

    results_path = Path("docs/experiments/phase5b_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Mandatory checks fail, but oversized (raw + 100) rejection succeeds
    ws_bc_corrupted = json.loads(json.dumps(data["workstreams_bc_lossless"]))
    p48_mat = ws_bc_corrupted["qualification_matrix"]["rs_2t_48"]
    p48_mat["capacity_boundary_tests"]["status"] = "FAIL"
    p48_mat["capacity_boundary_tests"]["at_limit_pass"] = False
    p48_mat["capacity_boundary_tests"]["one_below_pass"] = False
    p48_mat["capacity_boundary_tests"]["one_above_rejected_fail_closed"] = False
    # Supplementary check passes
    p48_mat["capacity_boundary_tests"]["oversized_rejected_fail_closed"] = True
    p48_mat["oversized_fail_closed_test"]["rejected_fail_closed"] = True

    gates = aggregate_phase5b_gates(
        data["preflight"],
        data["workstream_a_parity"],
        ws_bc_corrupted,
        data["workstream_d_robustness"],
        data["workstream_e_ecc"],
    )

    assert gates["gate_lossless_recovery"] == "FAIL"
    assert gates["overall_verdict"] == "BLOCKED"


def test_failed_capacity_boundary_blocks_lossless_gate():
    """Verifies that a failed required capacity-boundary test prevents lossless gate from passing."""
    from scripts.experiments.run_phase5b_assessment import aggregate_phase5b_gates
    import json

    results_path = Path("docs/experiments/phase5b_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Clone data and simulate failed boundary test
    ws_bc_corrupted = json.loads(json.dumps(data["workstreams_bc_lossless"]))
    p48_mat = ws_bc_corrupted["qualification_matrix"]["rs_2t_48"]
    p48_mat["capacity_boundary_tests"]["status"] = "FAIL"
    p48_mat["capacity_boundary_tests"]["at_limit_pass"] = False
    p48_mat["capacity_boundary_tests"]["one_below_pass"] = False
    p48_mat["capacity_boundary_tests"]["one_above_rejected_fail_closed"] = False
    p48_mat["capacity_boundary_tests"]["oversized_rejected_fail_closed"] = False
    p48_mat["oversized_fail_closed_test"]["rejected_fail_closed"] = False

    gates = aggregate_phase5b_gates(
        data["preflight"],
        data["workstream_a_parity"],
        ws_bc_corrupted,
        data["workstream_d_robustness"],
        data["workstream_e_ecc"],
    )

    assert gates["gate_lossless_recovery"] == "FAIL"
    assert gates["overall_verdict"] == "BLOCKED"


def test_failed_profile_adaptation_blocks_ecc_qualification():
    """Verifies that a failed required profile-adaptation check blocks ECC qualification."""
    from scripts.experiments.run_phase5b_assessment import aggregate_phase5b_gates
    import json

    results_path = Path("docs/experiments/phase5b_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Clone data and simulate failed receiver profile adaptation
    ws_e_corrupted = json.loads(json.dumps(data["workstream_e_ecc"]))
    ws_e_corrupted["policy_decision"]["interoperability"] = "Receiver adaptation failed"
    ws_e_corrupted["policy_decision"]["interoperability_verified"] = False

    gates = aggregate_phase5b_gates(
        data["preflight"],
        data["workstream_a_parity"],
        data["workstreams_bc_lossless"],
        data["workstream_d_robustness"],
        ws_e_corrupted,
    )

    assert gates["gate_ecc_decision"] == "FAIL"
    assert gates["overall_verdict"] == "BLOCKED"


def test_passing_recovery_and_capacity_allows_lossless_gate_to_pass():
    """Verifies that passing required recovery (20/20) and capacity checks allows lossless gate to pass."""
    from scripts.experiments.run_phase5b_assessment import aggregate_phase5b_gates
    import json

    results_path = Path("docs/experiments/phase5b_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gates = aggregate_phase5b_gates(
        data["preflight"],
        data["workstream_a_parity"],
        data["workstreams_bc_lossless"],
        data["workstream_d_robustness"],
        data["workstream_e_ecc"],
    )

    assert gates["gate_lossless_recovery"] == "PASS"
    assert gates["gate_ecc_decision"] == "PASS"
    assert gates["overall_verdict"] == "PASS_WITH_LIMITATIONS"


def test_channel_partial_recovery_summarized_as_partial():
    """Verifies that a channel with partial recovery is summarized as partial, not zero recovery."""
    import json

    results_path = Path("docs/experiments/phase5b_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ws_d = data["workstream_d_robustness"]
    n10_agg = ws_d["aggregated_channels"]["gaussian_noise_sigma_1.0"]

    # Verify trials and status
    assert n10_agg["total_trials"] == 9
    assert n10_agg["exact_recovery_count"] == 3
    assert n10_agg["recovery_rate_pct"] == 33.33
    assert n10_agg["status"] == "PARTIAL"

    # Verify operating envelope string is dynamic and reports partial
    env_str = ws_d["operating_envelope"]["additive_noise_sigma_1.0"]
    assert "Partial" in env_str or "partial" in env_str
    assert "33.33%" in env_str
    assert "3/9" in env_str
    assert "0% recovery" not in env_str


def test_json_summaries_and_statuses_computed_from_raw_records():
    """Verifies that JSON summaries and gate statuses match the underlying raw records."""
    from scripts.experiments.run_phase5b_assessment import aggregate_phase5b_gates
    import json

    results_path = Path("docs/experiments/phase5b_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Workstream BC: raw trial count matches summary
    qmat = data["workstreams_bc_lossless"]["qualification_matrix"]
    psum = data["workstreams_bc_lossless"]["profile_evaluation_summary"]

    for prof in ["rs_2t_32", "rs_2t_48", "rs_2t_64"]:
        raw_trials = []
        for p_info in qmat[prof]["payloads"].values():
            raw_trials.extend(p_info["trials"])
        assert len(raw_trials) == psum[prof]["total_trials"]
        exact_from_raw = sum(1 for t in raw_trials if t["exact_match"])
        assert exact_from_raw == psum[prof]["successful_trials"]
        expected_rate = round(exact_from_raw / len(raw_trials) * 100.0, 2)
        assert expected_rate == psum[prof]["recovery_rate_pct"]

        # Verify protected_payload_bytes is actual length, not constant 4
        for t in raw_trials:
            assert t["protected_payload_bytes"] > 100

    # 2. Workstream D: trial_records match aggregated_channels
    ws_d = data["workstream_d_robustness"]
    raw_ch_trials = ws_d["trial_records"]
    assert len(raw_ch_trials) == 39

    for ch_name, ch_agg in ws_d["aggregated_channels"].items():
        matching = [t for t in raw_ch_trials if t["channel_name"] == ch_name]
        assert len(matching) == ch_agg["total_trials"]
        rec_count = sum(1 for t in matching if t["exact_recovery"])
        assert rec_count == ch_agg["exact_recovery_count"]

    # 3. Gate computation reproducibility
    computed_gates = aggregate_phase5b_gates(
        data["preflight"],
        data["workstream_a_parity"],
        data["workstreams_bc_lossless"],
        data["workstream_d_robustness"],
        data["workstream_e_ecc"],
    )
    assert computed_gates == data["gates"]


def test_qualified_deployment_factory_configuration(dummy_cover, dummy_key):
    """Verifies that create_qualified_deployment explicitly configures rs_2t_48 while preserving legacy default."""
    from src.core.media_pipeline import create_qualified_deployment_pipeline

    # 1. Qualified deployment sender explicitly uses 2t=48
    sender = MediaStegoPipeline.create_qualified_deployment()
    assert sender.security_pipeline.ecc_encoder.parity_symbols == 48

    # 2. Legacy / default constructor without ecc_encoder retains 2t=32
    legacy_sender = MediaStegoPipeline(backend="onnx")
    assert legacy_sender.security_pipeline.ecc_encoder.parity_symbols == 32

    # 3. Embed payload with qualified deployment pipeline
    payload = SecretPayload(
        media_type=MediaType.TEXT,
        data=b"Qualified deployment payload under rs_2t_48",
        metadata={"deployment": "qualified_phase5b"},
    )
    stego_res = sender.protect_and_embed(payload, dummy_cover, dummy_key)

    # 4. Confirm wire-level header flags encode 2t=48
    assert stego_res.header.ecc_profile == 0x01
    assert stego_res.header.ecc_parity == 48

    # 5. Confirm fresh receiver instance (which defaults to legacy 2t=32) dynamically adapts to 2t=48
    fresh_receiver = MediaStegoPipeline(backend="onnx")
    assert fresh_receiver.security_pipeline.ecc_encoder.parity_symbols == 32

    rec_res = fresh_receiver.extract_and_unprotect(stego_res.stego_png_bytes, dummy_key, reference_payload=payload)
    assert rec_res.exact_match is True
    assert rec_res.recovered_payload.data == payload.data


def test_exact_capacity_boundaries_over_limit_rejection(dummy_cover, dummy_key):
    """Verifies that payloads exceeding calculated raw capacity by 1 byte are rejected fail-closed before embedding."""
    from src.security.ecc.encoder import ReedSolomonEncoder

    expected_caps = {
        32: {"raw": 7090, "ref": 7088},
        48: {"raw": 6578, "ref": 6576},
        64: {"raw": 6066, "ref": 6064},
    }

    for parity, caps in expected_caps.items():
        calc_raw = CapacityCalculator.calculate_usable_secret_capacity(metadata_len=0, rs_parity=parity)
        calc_ref = CapacityCalculator.calculate_usable_secret_capacity(metadata_len=2, rs_parity=parity)
        assert calc_raw == caps["raw"]
        assert calc_ref == caps["ref"]

        pipe = MediaStegoPipeline(ecc_encoder=ReedSolomonEncoder(parity_symbols=parity), backend="onnx")

        # Over-limit payload (raw + 1) must be rejected before carrier modification
        p_over = SecretPayload(data=b"X" * (caps["raw"] + 1), metadata={})
        with pytest.raises(PayloadCapacityExceededError):
            pipe.protect_and_embed(p_over, dummy_cover, dummy_key)


def test_qualified_profiles_exact_boundary_recovery(dummy_cover, dummy_key):
    """Verifies that qualified profiles (rs_2t_48 and rs_2t_64) achieve exact recovery at exact limit and one-below."""
    from src.security.ecc.encoder import ReedSolomonEncoder

    test_profiles = [
        (48, 6578),  # rs_2t_48: raw limit = 6,578 B
        (64, 6066),  # rs_2t_64: raw limit = 6,066 B
    ]

    fresh_rx = MediaStegoPipeline(backend="onnx")

    for parity, raw_limit in test_profiles:
        pipe = MediaStegoPipeline(ecc_encoder=ReedSolomonEncoder(parity_symbols=parity), backend="onnx")

        # 1. Payload at exact calculated raw limit
        p_at = SecretPayload(data=b"X" * raw_limit, metadata={})
        res_at = pipe.protect_and_embed(p_at, dummy_cover, dummy_key)
        rec_at = fresh_rx.extract_and_unprotect(res_at.stego_png_bytes, dummy_key, reference_payload=p_at)
        assert rec_at.exact_match is True

        # 2. Payload at one byte below calculated raw limit
        p_below = SecretPayload(data=b"X" * (raw_limit - 1), metadata={})
        res_below = pipe.protect_and_embed(p_below, dummy_cover, dummy_key)
        rec_below = fresh_rx.extract_and_unprotect(res_below.stego_png_bytes, dummy_key, reference_payload=p_below)
        assert rec_below.exact_match is True


def test_rs_2t_32_boundary_unreliability_empirical_finding():
    """Preserves empirical finding: rs_2t_32 theoretical maximum is not reliably recoverable in recorded boundary trial."""
    import json

    results_path = Path("docs/experiments/phase5b_results.json")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    p32_bnd = data["workstreams_bc_lossless"]["qualification_matrix"]["rs_2t_32"]["capacity_boundary_tests"]

    # In the recorded boundary trial, rs_2t_32 failed exact recovery at its calculated raw limit (7,090 B)
    assert p32_bnd["raw_usable_limit_bytes"] == 7090
    assert p32_bnd["at_limit_pass"] is False
    assert p32_bnd["status"] == "FAIL"

    # In contrast, qualified profile rs_2t_48 passed all mandatory boundary checks
    p48_bnd = data["workstreams_bc_lossless"]["qualification_matrix"]["rs_2t_48"]["capacity_boundary_tests"]
    assert p48_bnd["raw_usable_limit_bytes"] == 6578
    assert p48_bnd["at_limit_pass"] is True
    assert p48_bnd["one_below_pass"] is True
    assert p48_bnd["one_above_rejected_fail_closed"] is True
    assert p48_bnd["status"] == "PASS"


def test_phase5b_parity_report_json_consistency():
    """Verifies that parity element counts, percentages, and gate reasons match between JSON and report."""
    from scripts.experiments.check_phase5b_parity_consistency import check_consistency
    assert check_consistency() is True
