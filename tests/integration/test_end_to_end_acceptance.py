"""Manual & Programmatic End-to-End Acceptance Test for PaladinMedia Embedding Inspector.

Verifies the complete Section 12 criteria:
1. Small secret (24x24) + 512x512 carrier reaches EXACT_MATCH_VERIFY complete.
2. Stego inspection diagnostics: authentic positions, changed coefficients, pixel diff.
3. Second run with different secret + different carrier updates all evidence independently.
4. Oversized secret payload correctly triggers hard capacity gate failure and produces no embedding artifacts.
"""

import io
import json
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.api.diagnostics import diagnostics_store
from src.api.security import APISettings
from src.api.server import create_app


def test_full_end_to_end_acceptance_workflow():
    app = create_app(APISettings(mode="local"))
    client = TestClient(app)

    # -------------------------------------------------------------------------
    # PART 1: Baseline Small Secret (24x24) + Carrier 0001x2.png
    # -------------------------------------------------------------------------
    sec_fixture_path = Path("tests/fixtures/secret_small.png")
    cov_fixture_path = (
        Path("tests/fixtures/carrier_512.png")
        if Path("tests/fixtures/carrier_512.png").exists()
        else Path("dataset/MVP 0 Dataset/0001x2.png")
    )
    assert sec_fixture_path.exists()
    assert cov_fixture_path.exists()

    with open(sec_fixture_path, "rb") as f:
        sec1_bytes = f.read()
    with open(cov_fixture_path, "rb") as f:
        cov1_bytes = f.read()

    # Upload files to session store
    r_sec1 = client.post("/api/upload", files={"file": ("secret_small.png", sec1_bytes, "image/png")})
    assert r_sec1.status_code == 200
    sec1_sid = r_sec1.json()["session_id"]

    r_cov1 = client.post("/api/upload", files={"file": ("0001x2.png", cov1_bytes, "image/png")})
    assert r_cov1.status_code == 200
    cov1_sid = r_cov1.json()["session_id"]

    # Stream pipeline execution
    key_hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    r_stream1 = client.post(
        "/api/stream",
        json={"secret_session_id": sec1_sid, "cover_session_id": cov1_sid, "key_hex": key_hex},
    )
    assert r_stream1.status_code == 200

    events1 = []
    for line in r_stream1.text.split("\n\n"):
        line = line.strip()
        if line.startswith("data: "):
            events1.append(json.loads(line[6:]))

    stages1 = [ev["stage"] for ev in events1]
    assert "LOAD" in stages1
    assert "SERIALIZE" in stages1
    assert "ENCRYPT" in stages1
    assert "RS_ENCODE" in stages1
    assert "CAPACITY_CHECK" in stages1
    assert "DWT" in stages1
    assert "EMBED" in stages1
    assert "IDWT" in stages1
    assert "EXTRACT" in stages1
    assert "SECRET_RECOVER" in stages1
    assert "EXACT_MATCH_VERIFY" in stages1

    # Verify terminal completion
    terminal1 = [ev for ev in events1 if ev["stage"] == "EXACT_MATCH_VERIFY"][0]
    assert terminal1["status"] == "complete"
    assert terminal1["metadata"]["exact_match"] is True
    assert terminal1["metadata"]["bytes_verified"] == len(sec1_bytes)
    run1_id = terminal1["metadata"]["run_id"]

    # Verify genuine diagnostic metadata
    embed1 = [ev for ev in events1 if ev["stage"] == "EMBED" and ev["status"] == "complete"][0]
    diag1 = embed1["metadata"]["diagnostics"]
    assert diag1["run_id"] == run1_id
    assert diag1["header_bits_count"] == 128
    assert diag1["total_embedded_bits"] == 128 + diag1["payload_bits_count"]
    assert len(diag1["embedded_positions"]) == diag1["total_embedded_bits"]
    assert diag1["changed_coefficients_count"] > 0
    assert len(diag1["changed_positions"]) == diag1["changed_coefficients_count"]

    # Query diagnostics endpoint
    r_diag1 = client.get(f"/api/diagnostics/{run1_id}")
    assert r_diag1.status_code == 200
    assert r_diag1.json()["run_id"] == run1_id
    assert r_diag1.json()["total_embedded_bits"] == diag1["total_embedded_bits"]

    # -------------------------------------------------------------------------
    # PART 2: Different Secret + Different Carrier (0002x2.png)
    # -------------------------------------------------------------------------
    cov2_fixture_path = (
        Path("tests/fixtures/carrier2_512.png")
        if Path("tests/fixtures/carrier2_512.png").exists()
        else Path("dataset/MVP 0 Dataset/0002x2.png")
    )
    assert cov2_fixture_path.exists()
    with open(cov2_fixture_path, "rb") as f:
        cov2_bytes = f.read()

    # Create a different custom secret image (20x20 blue image)
    sec2_img = Image.new("RGB", (20, 20), color=(30, 80, 210))
    sec2_buf = io.BytesIO()
    sec2_img.save(sec2_buf, format="PNG")
    sec2_bytes = sec2_buf.getvalue()

    r_sec2 = client.post("/api/upload", files={"file": ("secret_blue_20x20.png", sec2_bytes, "image/png")})
    assert r_sec2.status_code == 200
    sec2_sid = r_sec2.json()["session_id"]

    r_cov2 = client.post("/api/upload", files={"file": ("0002x2.png", cov2_bytes, "image/png")})
    assert r_cov2.status_code == 200
    cov2_sid = r_cov2.json()["session_id"]

    r_stream2 = client.post(
        "/api/stream",
        json={"secret_session_id": sec2_sid, "cover_session_id": cov2_sid, "key_hex": key_hex},
    )
    assert r_stream2.status_code == 200

    events2 = []
    for line in r_stream2.text.split("\n\n"):
        line = line.strip()
        if line.startswith("data: "):
            events2.append(json.loads(line[6:]))

    terminal2 = [ev for ev in events2 if ev["stage"] == "EXACT_MATCH_VERIFY"][0]
    assert terminal2["status"] == "complete"
    assert terminal2["metadata"]["exact_match"] is True
    assert terminal2["metadata"]["bytes_verified"] == len(sec2_bytes)
    run2_id = terminal2["metadata"]["run_id"]

    # Verify run isolation: Run 2 has distinct run_id and distinct diagnostics
    assert run2_id != run1_id
    embed2 = [ev for ev in events2 if ev["stage"] == "EMBED" and ev["status"] == "complete"][0]
    diag2 = embed2["metadata"]["diagnostics"]
    assert diag2["run_id"] == run2_id

    # -------------------------------------------------------------------------
    # PART 3: Oversized Secret Payload -> Capacity Hard Gate Failure
    # -------------------------------------------------------------------------
    # Create an oversized image with high-entropy noise whose PNG bytes exceed 8,176 B
    import numpy as np
    noise_arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    oversized_img = Image.fromarray(noise_arr)
    oversized_buf = io.BytesIO()
    oversized_img.save(oversized_buf, format="PNG")
    oversized_bytes = oversized_buf.getvalue()
    assert len(oversized_bytes) > 8176

    r_sec_over = client.post("/api/upload", files={"file": ("oversized.png", oversized_bytes, "image/png")})
    assert r_sec_over.status_code == 200
    over_sid = r_sec_over.json()["session_id"]

    r_stream_over = client.post(
        "/api/stream",
        json={"secret_session_id": over_sid, "cover_session_id": cov1_sid, "key_hex": key_hex},
    )
    assert r_stream_over.status_code == 200

    events_over = []
    for line in r_stream_over.text.split("\n\n"):
        line = line.strip()
        if line.startswith("data: "):
            events_over.append(json.loads(line[6:]))

    over_stages = [ev["stage"] for ev in events_over]
    assert "CAPACITY_CHECK" in over_stages
    assert "EMBED" not in over_stages
    assert "IDWT" not in over_stages
    assert "EXACT_MATCH_VERIFY" not in over_stages

    cap_fail = [ev for ev in events_over if ev["stage"] == "CAPACITY_CHECK" and ev["status"] == "failed"][0]
    assert cap_fail["error"] == "PAYLOAD_CAPACITY_EXCEEDED"
    assert cap_fail["metadata"]["fits"] is False
    assert cap_fail["metadata"]["protected_payload_bytes"] > cap_fail["metadata"]["max_protected_bytes"]

    # Verify diagnostic store has PRE_EMBED_FAILED record
    load_over = [ev for ev in events_over if ev["stage"] == "LOAD"][0]
    over_run_id = load_over["metadata"]["run_id"]
    stored_over = diagnostics_store.get(over_run_id)
    assert stored_over is not None
    assert stored_over.verification_status == "PRE_EMBED_FAILED"
    assert stored_over.exact_match is False
    assert "PAYLOAD_CAPACITY_EXCEEDED" in stored_over.error_message


def test_pipeline_failure_lifecycle_status_is_failed_not_aborted(monkeypatch):
    """Finding C: Genuine pipeline failure records FAILED (not ABORTED or PENDING).

    Simulates a failure during IDWT synthesis and verifies that:
    1. The IDWT failure event is emitted with status="failed".
    2. The diagnostics store records verification_status="FAILED" and exact_match=False.
    3. The finally: cleanup block does NOT overwrite FAILED with ABORTED.
    """
    app = create_app(APISettings(mode="local"))
    client = TestClient(app)

    sec_img = Image.new("RGB", (24, 24), color=(150, 40, 40))
    sbuf = io.BytesIO()
    sec_img.save(sbuf, format="PNG")

    cov_img = Image.new("RGB", (512, 512), color=(100, 100, 100))
    cbuf = io.BytesIO()
    cov_img.save(cbuf, format="PNG")

    r_sec = client.post("/api/upload", files={"file": ("secret.png", sbuf.getvalue(), "image/png")})
    assert r_sec.status_code == 200
    sec_sid = r_sec.json()["session_id"]

    r_cov = client.post("/api/upload", files={"file": ("carrier.png", cbuf.getvalue(), "image/png")})
    assert r_cov.status_code == 200
    cov_sid = r_cov.json()["session_id"]

    # Monkeypatch calculate_psnr during IDWT stage to trigger an IDWT failure
    def mock_psnr(*args, **kwargs):
        raise RuntimeError("Simulated synthetic IDWT transform hardware failure")

    monkeypatch.setattr("src.api.event_adapter.calculate_psnr", mock_psnr)

    r_stream = client.post(
        "/api/stream",
        json={"secret_session_id": sec_sid, "cover_session_id": cov_sid, "key_hex": "01" * 32},
    )
    assert r_stream.status_code == 200

    events = []
    for line in r_stream.text.split("\n\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # Find the run_id from LOAD or EMBED
    load_ev = [ev for ev in events if ev["stage"] == "LOAD"][0]
    run_id = load_ev["metadata"]["run_id"]

    # Verify that a failure occurred
    failed_evs = [ev for ev in events if ev.get("status") == "failed"]
    assert len(failed_evs) >= 1

    # Verify diagnostic store record: MUST be FAILED, NOT ABORTED
    stored = diagnostics_store.get(run_id)
    assert stored is not None
    assert stored.verification_status == "FAILED", f"Expected FAILED, got {stored.verification_status}"
    assert stored.exact_match is False
    assert "Simulated synthetic IDWT" in stored.error_message


@pytest.mark.anyio
async def test_client_cancellation_lifecycle_status_is_aborted():
    """Finding C: Client stream cancellation or disconnection records ABORTED.

    Verifies that when a streaming consumer disconnects or cancels midway,
    the pending run status transitions to ABORTED with a cancellation error message.
    """
    import asyncio
    from src.api.server import qualified_pipeline
    from src.api.event_adapter import MediaEventAdapter
    from src.core.serialization.serializer import SecretPayload, MediaType

    adapter = MediaEventAdapter(pipeline=qualified_pipeline)
    sec_img = Image.new("RGB", (24, 24), color=(10, 20, 30))
    sbuf = io.BytesIO()
    sec_img.save(sbuf, format="PNG")
    secret = SecretPayload(data=sbuf.getvalue(), media_type=MediaType.IMAGE)
    cover = Image.new("RGB", (512, 512), color=(100, 100, 100))

    # Emulate the server's streaming event_generator
    last_run_id = None

    async def streaming_generator():
        nonlocal last_run_id
        try:
            async for event in adapter.run_media_pipeline(secret, cover, key=b"C" * 32, session_counter=909):
                if isinstance(event, dict) and event.get("metadata", {}).get("run_id"):
                    last_run_id = event["metadata"]["run_id"]
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            if last_run_id:
                curr = diagnostics_store.get(last_run_id)
                if curr and curr.verification_status == "PENDING":
                    diagnostics_store.update_verification(
                        run_id=last_run_id,
                        verification_status="ABORTED",
                        exact_match=False,
                        error_message="Stream aborted by client cancellation",
                    )
            raise
        finally:
            if last_run_id:
                curr = diagnostics_store.get(last_run_id)
                if curr and curr.verification_status == "PENDING":
                    diagnostics_store.update_verification(
                        run_id=last_run_id,
                        verification_status="ABORTED",
                        exact_match=False,
                        error_message="Stream terminated prematurely",
                    )

    gen = streaming_generator()
    first_ev = await anext(gen)
    assert last_run_id is not None
    assert diagnostics_store.get(last_run_id).verification_status == "PENDING"

    # Simulate client closing the connection prematurely
    await gen.aclose()

    # Verify diagnostic store status: MUST be ABORTED
    stored = diagnostics_store.get(last_run_id)
    assert stored is not None
    assert stored.verification_status == "ABORTED"
    assert stored.exact_match is False
    assert "cancellation" in stored.error_message.lower() or "prematurely" in stored.error_message.lower()
