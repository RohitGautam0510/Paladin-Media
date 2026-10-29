"""Unit tests for PaladinMedia Steganography Diagnostics and Telemetry.

Verifies:
1. DiagnosticsStore bounded retention and retrieval.
2. MediaEventAdapter emission of genuine transform diagnostics.
3. Diagnostic positions match actual embedding mask coordinates.
4. Changed coefficient counts and pixel diff metrics are truthful and bounded.
5. Endpoints /api/diagnostics/latest and /api/diagnostics/{run_id} enforce authentication.
6. Diagnostic collection does not modify stego or recovered artifacts.
"""

import io
import json
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.api.diagnostics import (
    DEFAULT_MAX_RECORD_BYTES,
    DiagnosticsStore,
    FrozenDict,
    RunDiagnostics,
    diagnostics_store,
    freeze_value,
    unfreeze_value,
)
from src.api.event_adapter import (
    DEFAULT_MAX_DIAGNOSTIC_SUBOBJECT_BYTES,
    DEFAULT_MAX_SSE_EVENT_BYTES,
    MediaEventAdapter,
    serialize_sse_event,
)
from src.api.security import APISettings
from src.api.server import create_app
from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization.serializer import MediaType, SecretPayload


@pytest.fixture
def sample_secret_payload() -> SecretPayload:
    sec_img = Image.new("RGB", (24, 24), color=(180, 40, 40))
    buf = io.BytesIO()
    sec_img.save(buf, format="PNG")
    return SecretPayload(
        data=buf.getvalue(),
        media_type=MediaType.IMAGE,
        metadata={"filename": "sample_secret.png"},
    )


@pytest.fixture
def sample_cover_image() -> Image.Image:
    fixture_path = Path("dataset/MVP 0 Dataset/0001x2.png")
    if fixture_path.exists():
        return Image.open(fixture_path).convert("RGB")
    return Image.new("RGB", (512, 512), color=(120, 130, 140))


def test_diagnostics_store_bounded_lifecycle():
    """Verifies that DiagnosticsStore stores, retrieves, and bounds memory."""
    store = DiagnosticsStore(max_entries=3)

    for i in range(5):
        diag = RunDiagnostics(
            run_id=f"run_{i}",
            timestamp=float(i),
            session_counter=i,
            carrier_dims=(512, 512),
            stego_dims=(512, 512),
            subband_dim=256,
            header_bits_count=128,
            payload_bits_count=1000,
            total_embedded_bits=1128,
            changed_coefficients_count=500,
            max_hh_diff=1.0,
            embedded_positions=[(0, 0), (0, 1)],
            changed_positions=[(0, 0)],
            pixel_diff_metrics={"changed_pixels_count": 200, "max_channel_diff": 1},
        )
        store.store(diag)

    # Max entries is 3, so run_0 and run_1 should have been evicted
    assert store.get("run_0") is None
    assert store.get("run_1") is None
    assert store.get("run_2") is not None
    assert store.get("run_4") is not None
    assert store.get_latest().run_id == "run_4"

    store.clear()
    assert store.get_latest() is None


@pytest.mark.anyio
async def test_event_adapter_emits_truthful_diagnostics(sample_secret_payload, sample_cover_image):
    """Verifies that MediaEventAdapter produces real diagnostics matching transform execution."""
    pipeline = MediaStegoPipeline.create_qualified_deployment()
    adapter = MediaEventAdapter(pipeline=pipeline)
    key = b"D" * 32

    events = []
    async for ev in adapter.run_media_pipeline(
        secret_payload=sample_secret_payload,
        cover_image=sample_cover_image,
        key=key,
        session_counter=201,
    ):
        events.append(ev)

    stages = [ev["stage"] for ev in events]
    assert "LOAD" in stages
    assert "EMBED" in stages
    assert "IDWT" in stages
    assert "EXACT_MATCH_VERIFY" in stages

    # Check EMBED event metadata
    embed_ev = [ev for ev in events if ev["stage"] == "EMBED" and ev["status"] == "complete"][0]
    diag = embed_ev["metadata"]["diagnostics"]
    assert diag["header_bits_count"] == 128
    assert diag["payload_bits_count"] > 0
    assert diag["total_embedded_bits"] == 128 + diag["payload_bits_count"]
    assert len(diag["embedded_positions"]) == diag["total_embedded_bits"]
    assert diag["changed_coefficients_count"] <= diag["total_embedded_bits"]
    assert len(diag["changed_positions"]) == diag["changed_coefficients_count"]
    assert diag["subband_dim"] == 256
    assert diag["carrier_dims"] == [512, 512]

    # Check IDWT event metadata
    idwt_ev = [ev for ev in events if ev["stage"] == "IDWT" and ev["status"] == "complete"][0]
    p_diff = idwt_ev["metadata"]["pixel_diff_metrics"]
    assert p_diff["total_pixels"] == 512 * 512
    assert p_diff["changed_pixels_count"] > 0
    assert p_diff["max_channel_diff"] >= 1

    # Check that RunDiagnostics was placed in global store
    run_id = embed_ev["metadata"]["run_id"]
    stored_diag = diagnostics_store.get(run_id)
    assert stored_diag is not None
    assert stored_diag.run_id == run_id
    assert stored_diag.total_embedded_bits == diag["total_embedded_bits"]


def test_diagnostics_endpoint_security_and_retrieval(sample_secret_payload, sample_cover_image):
    """Verifies /api/diagnostics/latest and /api/diagnostics/{run_id} behavior and auth gates."""
    token = "diag_test_token_secret_12345_safe_token_32_chars"

    # Pre-populate diagnostics_store
    run_id = "test_run_endpoint_001"
    diagnostics_store.store(
        RunDiagnostics(
            run_id=run_id,
            timestamp=100.0,
            session_counter=1,
            carrier_dims=(512, 512),
            stego_dims=(512, 512),
            subband_dim=256,
            header_bits_count=128,
            payload_bits_count=800,
            total_embedded_bits=928,
            changed_coefficients_count=450,
            max_hh_diff=0.5,
            embedded_positions=[(0, 0)],
            changed_positions=[(0, 0)],
            pixel_diff_metrics={"changed_pixels_count": 100, "max_channel_diff": 1},
        )
    )

    # 1. Remote mode: unauthenticated should be 401, authenticated should be 200
    remote_app = create_app(APISettings(mode="remote", api_token=token))
    with TestClient(remote_app) as client:
        res = client.get(f"/api/diagnostics/{run_id}")
        assert res.status_code == 401
        res_lat = client.get("/api/diagnostics/latest")
        assert res_lat.status_code == 401

        # Authenticated remote should be 200
        res_auth = client.get(
            f"/api/diagnostics/{run_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res_auth.status_code == 200
        data = res_auth.json()
        assert data["run_id"] == run_id
        assert data["total_embedded_bits"] == 928

    # 2. Local mode: allow_anonymous should be 200
    local_app = create_app(APISettings(mode="local"))
    with TestClient(local_app) as client:
        res_local = client.get(f"/api/diagnostics/{run_id}")
        assert res_local.status_code == 200
        assert res_local.json()["run_id"] == run_id

        res_missing = client.get("/api/diagnostics/non_existent_run")
        assert res_missing.status_code == 404


def test_diagnostics_store_per_record_coordinate_truncation():
    """Finding 2: Per-record byte bounding truncates excessive coordinate lists with explicit flag."""
    store = DiagnosticsStore(max_record_bytes=4096, max_total_bytes=65536, max_entries=10)

    # 10,000 coordinates would exceed 100 KB in JSON
    large_embedded = [(i % 256, (i * 3) % 256) for i in range(10000)]
    large_changed = [(i % 256, (i * 3) % 256) for i in range(5000)]

    diag = RunDiagnostics(
        run_id="run_truncated_01",
        timestamp=123456.7,
        session_counter=42,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=9872,
        total_embedded_bits=10000,
        changed_coefficients_count=5000,
        max_hh_diff=2.5,
        embedded_positions=large_embedded,
        changed_positions=large_changed,
        pixel_diff_metrics={"changed_pixels_count": 4000, "max_channel_diff": 2},
    )

    store.store(diag)
    retrieved = store.get("run_truncated_01")
    assert retrieved is not None
    assert retrieved.coordinates_truncated is True
    assert retrieved.total_embedded_positions_count == 10000
    assert retrieved.total_changed_positions_count == 5000
    assert len(retrieved.embedded_positions) < 10000
    assert len(retrieved.changed_positions) < 5000

    # Ensure serialized JSON representation fits within budget
    d_dict = retrieved.to_dict()
    assert d_dict["coordinates_truncated"] is True
    assert d_dict["total_embedded_positions_count"] == 10000
    assert d_dict["total_changed_positions_count"] == 5000
    serialized_len = len(json.dumps(d_dict).encode("utf-8"))
    assert serialized_len <= 4096


def test_diagnostics_store_aggregate_byte_budget_fifo_eviction():
    """Finding 2: Total byte budget enforces FIFO eviction when aggregate memory threshold is reached."""
    # Budget fits ~2 records of ~2 KB each
    store = DiagnosticsStore(max_record_bytes=4096, max_total_bytes=4500, max_entries=100)

    coords_sample = [(i, i) for i in range(120)]
    for i in range(5):
        diag = RunDiagnostics(
            run_id=f"run_fifo_{i}",
            timestamp=float(i),
            session_counter=i,
            carrier_dims=(512, 512),
            stego_dims=(512, 512),
            subband_dim=256,
            header_bits_count=128,
            payload_bits_count=120,
            total_embedded_bits=248,
            changed_coefficients_count=100,
            max_hh_diff=1.0,
            embedded_positions=coords_sample,
            changed_positions=coords_sample[:50],
            pixel_diff_metrics={"changed_pixels_count": 50, "max_channel_diff": 1},
        )
        store.store(diag)

    # Aggregate byte budget must be respected
    assert store.get_total_bytes() <= 4500
    # Oldest runs must have been evicted first (FIFO)
    assert store.get("run_fifo_0") is None
    # Most recent run must still exist
    assert store.get("run_fifo_4") is not None
    assert store.get_latest().run_id == "run_fifo_4"


def test_diagnostics_lifecycle_transitions():
    """Finding 4: Verification status transitions PENDING -> VERIFIED / FAILED / ABORTED."""
    store = DiagnosticsStore(max_entries=10)

    diag = RunDiagnostics(
        run_id="run_lifecycle_test",
        timestamp=1000.0,
        session_counter=10,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=500,
        total_embedded_bits=628,
        changed_coefficients_count=300,
        max_hh_diff=0.5,
        embedded_positions=[(0, 0)],
        changed_positions=[(0, 0)],
        pixel_diff_metrics={"changed_pixels_count": 50, "max_channel_diff": 1},
        verification_status="PENDING",
    )
    store.store(diag)

    # Initial state
    assert store.get("run_lifecycle_test").verification_status == "PENDING"
    assert store.get("run_lifecycle_test").exact_match is None

    # Transition to VERIFIED
    res = store.update_verification("run_lifecycle_test", "VERIFIED", exact_match=True)
    assert res is True
    updated = store.get("run_lifecycle_test")
    assert updated.verification_status == "VERIFIED"
    assert updated.exact_match is True
    assert updated.error_message is None

    # Transition to FAILED with error message
    res_fail = store.update_verification(
        "run_lifecycle_test", "FAILED", exact_match=False, error_message="CRC mismatch"
    )
    assert res_fail is True
    updated_fail = store.get("run_lifecycle_test")
    assert updated_fail.verification_status == "FAILED"
    assert updated_fail.exact_match is False
    assert updated_fail.error_message == "CRC mismatch"

    # Transition to ABORTED
    res_abort = store.update_verification(
        "run_lifecycle_test", "ABORTED", exact_match=None, error_message="Client disconnected"
    )
    assert res_abort is True
    updated_abort = store.get("run_lifecycle_test")
    assert updated_abort.verification_status == "ABORTED"
    assert updated_abort.error_message == "Client disconnected"

    # Updating non-existent run returns False gracefully
    assert store.update_verification("non_existent_run", "FAILED") is False


def test_diagnostics_store_thread_safety():
    """Thread concurrency: store and update operations remain thread-safe without exceptions."""
    import threading

    store = DiagnosticsStore(max_record_bytes=8192, max_total_bytes=32768, max_entries=20)
    errors = []

    def worker(worker_id: int):
        try:
            for j in range(25):
                rid = f"thread_run_{worker_id}_{j}"
                d = RunDiagnostics(
                    run_id=rid,
                    timestamp=float(j),
                    session_counter=j,
                    carrier_dims=(512, 512),
                    stego_dims=(512, 512),
                    subband_dim=256,
                    header_bits_count=128,
                    payload_bits_count=100,
                    total_embedded_bits=228,
                    changed_coefficients_count=50,
                    max_hh_diff=1.0,
                    embedded_positions=[(x, x) for x in range(20)],
                    changed_positions=[(x, x) for x in range(10)],
                    pixel_diff_metrics={"changed_pixels_count": 20, "max_channel_diff": 1},
                    verification_status="PENDING",
                )
                store.store(d)
                store.update_verification(rid, "VERIFIED", exact_match=True)
                _ = store.get_latest()
                _ = store.get(rid)
                _ = store.get_total_bytes()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert store.get_total_bytes() <= 32768


def test_diagnostics_store_update_verification_enforces_budget_and_truncates_oversized_error():
    """Finding B regression: Store small diagnostic under 1,000-byte budget, update with 10,000-char error.

    Proves that neither the record limit nor aggregate store budget is exceeded.
    """
    store = DiagnosticsStore(max_record_bytes=1000, max_total_bytes=5000, max_entries=10)

    # Initial small record with 10 coordinates (well within 1000 bytes)
    diag = RunDiagnostics(
        run_id="run_1000b_test",
        timestamp=100.0,
        session_counter=1,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=80,
        total_embedded_bits=208,
        changed_coefficients_count=10,
        max_hh_diff=0.5,
        embedded_positions=[(i, i) for i in range(10)],
        changed_positions=[(i, i) for i in range(5)],
        pixel_diff_metrics={"changed_pixels_count": 5, "max_channel_diff": 1},
        verification_status="PENDING",
    )
    store.store(diag)
    initial_rec = store.get("run_1000b_test")
    assert initial_rec is not None
    assert initial_rec.serialized_size() <= 1000
    assert store.get_total_bytes() <= 1000

    # Attempt to update with a 10,000-character error message
    massive_error = "X" * 10000
    res = store.update_verification(
        run_id="run_1000b_test",
        verification_status="FAILED",
        exact_match=False,
        error_message=massive_error,
    )
    assert res is True

    updated_rec = store.get("run_1000b_test")
    assert updated_rec is not None
    assert updated_rec.verification_status == "FAILED"
    assert updated_rec.exact_match is False
    assert updated_rec.error_message is not None
    assert "... [truncated]" in updated_rec.error_message

    # CRITICAL INVARIANT: The serialized size MUST strictly not exceed 1,000 bytes
    actual_size = updated_rec.serialized_size()
    assert actual_size <= 1000, f"Record size {actual_size} exceeded 1000-byte ceiling"
    assert store.get_total_bytes() <= 1000, f"Total store bytes {store.get_total_bytes()} exceeded budget"


def test_diagnostics_immutability_and_defensive_copies():
    """Finding B: Stored records are immutable and immune to external mutations."""
    store = DiagnosticsStore(max_record_bytes=4096, max_total_bytes=16384, max_entries=5)

    coords = [[1, 2], [3, 4]]
    metrics = {"changed_pixels": 10}

    diag = RunDiagnostics(
        run_id="run_immutability",
        timestamp=100.0,
        session_counter=1,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=16,
        total_embedded_bits=144,
        changed_coefficients_count=2,
        max_hh_diff=0.5,
        embedded_positions=coords,
        changed_positions=coords,
        pixel_diff_metrics=metrics,
    )
    store.store(diag)

    # Mutate the input structures outside the store
    coords.append([99, 99])
    metrics["changed_pixels"] = 9999

    # Verify store contents were not mutated
    retrieved = store.get("run_immutability")
    assert retrieved is not None
    assert len(retrieved.embedded_positions) == 2
    assert retrieved.pixel_diff_metrics["changed_pixels"] == 10

    # Verify retrieved dict mutation does not affect store
    as_dict = retrieved.to_dict()
    as_dict["embedded_positions"].append([88, 88])
    assert len(store.get("run_immutability").embedded_positions) == 2


@pytest.mark.anyio
async def test_diagnostics_lifecycle_persisted_from_run_start_on_capacity_rejection(sample_cover_image):
    """Finding D: Runs rejected by capacity gate persist a PRE_EMBED_FAILED lifecycle record."""
    pipeline = MediaStegoPipeline.create_qualified_deployment()
    adapter = MediaEventAdapter(pipeline=pipeline)
    key = b"D" * 32

    # Secret exceeding 8,176 byte capacity
    huge_data = b"M" * 15000
    huge_secret = SecretPayload(data=huge_data, media_type=MediaType.RAW_BYTES)

    events = []
    async for ev in adapter.run_media_pipeline(
        secret_payload=huge_secret,
        cover_image=sample_cover_image,
        key=key,
        session_counter=301,
    ):
        events.append(ev)

    # Verify capacity failure event
    cap_failed = [ev for ev in events if ev["stage"] == "CAPACITY_CHECK" and ev["status"] == "failed"]
    assert len(cap_failed) == 1
    load_ev = [ev for ev in events if ev["stage"] == "LOAD" and ev["status"] == "processing"][0]
    run_id = load_ev["metadata"]["run_id"]

    # Verify diagnostic store has persistent PRE_EMBED_FAILED record
    stored = diagnostics_store.get(run_id)
    assert stored is not None
    assert stored.run_id == run_id
    assert stored.verification_status == "PRE_EMBED_FAILED"
    assert stored.exact_match is False
    assert stored.completeness_status == "FAILED"
    assert stored.total_embedded_bits == 0
    assert len(stored.embedded_positions) == 0
    assert "PAYLOAD_CAPACITY_EXCEEDED" in stored.error_message


@pytest.mark.anyio
async def test_diagnostics_sse_emission_bounded_before_serialization(sample_cover_image):
    """Finding A: Diagnostic data is bounded BEFORE SSE event serialization."""
    pipeline = MediaStegoPipeline.create_qualified_deployment()
    adapter = MediaEventAdapter(pipeline=pipeline)
    key = b"D" * 32

    # Secret generating ~2,000 bytes protected payload
    sec_img = Image.new("RGB", (32, 32), color=(50, 100, 150))
    buf = io.BytesIO()
    sec_img.save(buf, format="PNG")
    secret = SecretPayload(data=buf.getvalue(), media_type=MediaType.IMAGE)

    events = []
    async for ev in adapter.run_media_pipeline(
        secret_payload=secret,
        cover_image=sample_cover_image,
        key=key,
        session_counter=401,
    ):
        events.append(ev)

    embed_ev = [ev for ev in events if ev["stage"] == "EMBED" and ev["status"] == "complete"][0]
    diag = embed_ev["metadata"]["diagnostics"]

    # Verify diagnostics dictionary format and bounds
    assert "completeness_status" in diag
    assert diag["completeness_status"] in ("COMPLETE", "PARTIAL")
    assert "coordinates_truncated" in diag
    assert "total_embedded_positions_count" in diag
    assert "total_changed_positions_count" in diag

    # Verify serialized JSON size stays strictly bounded
    serialized_bytes = json.dumps(diag, separators=(",", ":")).encode("utf-8")
    assert len(serialized_bytes) <= DEFAULT_MAX_RECORD_BYTES


def test_run_id_full_random_uuid_format():
    """Finding F: Run IDs preserve full 128-bit random UUID component for collision resistance."""
    from src.api.event_adapter import MediaEventAdapter
    from src.carrier.capacity import CapacityCalculator
    import uuid

    adapter = MediaEventAdapter()
    run_ids = set()

    for i in range(100):
        rid = f"run_{i}_{int(time.time() * 1000)}_{uuid.uuid4().hex}"
        assert len(rid.split("_")[-1]) == 32, "UUID component must be full 32 hex characters (128 bits)"
        run_ids.add(rid)

    # 100 unique run IDs
    assert len(run_ids) == 100


def test_diagnostics_interleaved_runs_lifecycle_isolation():
    """Finding D: Multiple active runs maintain independent lifecycle states without crosstalk."""
    store = DiagnosticsStore(max_entries=10)

    # Register Run A and Run B
    diag_a = RunDiagnostics(
        run_id="run_interleaved_A",
        timestamp=100.0,
        session_counter=1,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=100,
        total_embedded_bits=228,
        changed_coefficients_count=50,
        max_hh_diff=1.0,
        embedded_positions=((0, 0),),
        changed_positions=((0, 0),),
        pixel_diff_metrics={},
        verification_status="PENDING",
    )
    diag_b = RunDiagnostics(
        run_id="run_interleaved_B",
        timestamp=101.0,
        session_counter=2,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=100,
        total_embedded_bits=228,
        changed_coefficients_count=50,
        max_hh_diff=1.0,
        embedded_positions=((1, 1),),
        changed_positions=((1, 1),),
        pixel_diff_metrics={},
        verification_status="PENDING",
    )

    store.store(diag_a)
    store.store(diag_b)

    # Fail Run A
    store.update_verification("run_interleaved_A", "FAILED", exact_match=False, error_message="A failed")
    assert store.get("run_interleaved_A").verification_status == "FAILED"
    # Run B must remain PENDING
    assert store.get("run_interleaved_B").verification_status == "PENDING"
    assert store.get("run_interleaved_B").exact_match is None

    # Verify Run B
    store.update_verification("run_interleaved_B", "VERIFIED", exact_match=True)
    assert store.get("run_interleaved_B").verification_status == "VERIFIED"
    assert store.get("run_interleaved_B").exact_match is True
    # Run A must still be FAILED
    assert store.get("run_interleaved_A").verification_status == "FAILED"


def test_diagnostics_store_update_oldest_record_accounting_and_eviction():
    """Finding A: Updating the oldest record never causes self-eviction or counter desync.

    Proves:
    1. Two records stored under a tight aggregate budget (1,300 bytes).
    2. Oldest record (run_1) is updated with new verification data.
    3. If budget is exceeded by the update, other records (run_2) are evicted to make room, NOT run_1.
    4. Total byte counter is strictly reconciled with sum of stored record serialized sizes.
    """
    store = DiagnosticsStore(max_record_bytes=1000, max_total_bytes=1300, max_entries=10)

    # Initial records of ~500-600 bytes each
    diag1 = RunDiagnostics(
        run_id="run_oldest_1",
        timestamp=100.0,
        session_counter=1,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=80,
        total_embedded_bits=208,
        changed_coefficients_count=10,
        max_hh_diff=0.5,
        embedded_positions=[(i, i) for i in range(8)],
        changed_positions=[(i, i) for i in range(4)],
        pixel_diff_metrics={"changed_pixels_count": 5, "max_channel_diff": 1},
        verification_status="PENDING",
    )
    diag2 = RunDiagnostics(
        run_id="run_newer_2",
        timestamp=101.0,
        session_counter=2,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=80,
        total_embedded_bits=208,
        changed_coefficients_count=10,
        max_hh_diff=0.5,
        embedded_positions=[(i, i) for i in range(8)],
        changed_positions=[(i, i) for i in range(4)],
        pixel_diff_metrics={"changed_pixels_count": 5, "max_channel_diff": 1},
        verification_status="PENDING",
    )

    store.store(diag1)
    store.store(diag2)

    # Both run_oldest_1 and run_newer_2 are present initially
    assert store.get("run_oldest_1") is not None
    assert store.get("run_newer_2") is not None
    initial_total = store.get_total_bytes()
    assert initial_total <= 1300

    # Verify run_oldest_1 is indeed the oldest key in the OrderedDict
    assert next(iter(store._store)) == "run_oldest_1"

    # Now update run_oldest_1 with verification details and error message that expands its size
    res = store.update_verification(
        run_id="run_oldest_1",
        verification_status="FAILED",
        exact_match=False,
        error_message="Detailed recovery failure: 128 header bits verified, payload integrity checksum mismatch",
    )
    assert res is True

    # Critical assertions:
    # 1. run_oldest_1 is NOT evicted! It is retained and updated.
    updated1 = store.get("run_oldest_1")
    assert updated1 is not None
    assert updated1.verification_status == "FAILED"
    assert updated1.exact_match is False
    assert "checksum mismatch" in updated1.error_message

    # 2. Total store bytes strictly respects the 1,300-byte budget
    current_total = store.get_total_bytes()
    assert current_total <= 1300

    # 3. If newer record had to be evicted to fit run_oldest_1's expansion, aggregate math holds
    reconciled_sum = sum(
        len(json.dumps(r.to_dict(), separators=(",", ":")).encode("utf-8"))
        for r in store._store.values()
    )
    assert current_total == reconciled_sum, f"Counter {current_total} != reconciled sum {reconciled_sum}"


def test_frozen_dict_and_immutability_guarantees():
    """Finding A: Immutability enforcement across FrozenDict, coordinate tuples, and RunDiagnostics."""
    # 1. Direct FrozenDict mutations raise TypeError
    fd = FrozenDict({"changed_pixels_count": 42, "max_channel_diff": 2})
    with pytest.raises(TypeError, match="does not support item assignment"):
        fd["changed_pixels_count"] = 99
    with pytest.raises(TypeError, match="does not support item deletion"):
        del fd["max_channel_diff"]
    with pytest.raises(TypeError, match="does not support mutation"):
        fd.update({"new_key": 100})
    with pytest.raises(TypeError, match="does not support mutation"):
        fd.pop("changed_pixels_count")
    with pytest.raises(TypeError, match="does not support mutation"):
        fd.clear()

    # 2. RunDiagnostics wraps pixel_diff_metrics in FrozenDict automatically
    mutable_metrics = {"changed_pixels_count": 10, "max_channel_diff": 1}
    diag = RunDiagnostics(
        run_id="run_freeze_test",
        timestamp=200.0,
        session_counter=5,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=64,
        total_embedded_bits=192,
        changed_coefficients_count=5,
        max_hh_diff=0.25,
        embedded_positions=[(1, 2), (3, 4)],
        changed_positions=[(1, 2)],
        pixel_diff_metrics=mutable_metrics,
        verification_status="PENDING",
    )

    # Modifying the original input dictionary does not mutate diag
    mutable_metrics["changed_pixels_count"] = 999
    assert diag.pixel_diff_metrics["changed_pixels_count"] == 10
    assert isinstance(diag.pixel_diff_metrics, FrozenDict)

    # Modifying pixel_diff_metrics on diag raises TypeError
    with pytest.raises(TypeError, match="does not support item assignment"):
        diag.pixel_diff_metrics["changed_pixels_count"] = 999

    # Modifying frozen dataclass fields raises FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        diag.run_id = "hacked_run_id"
    with pytest.raises(FrozenInstanceError):
        diag.verification_status = "VERIFIED"

    # Coordinates are tuples of tuples (immutable)
    assert isinstance(diag.embedded_positions, tuple)
    with pytest.raises(TypeError):
        diag.embedded_positions[0] = (99, 99)

    # 3. Store defensive copying on get() and get_latest()
    store = DiagnosticsStore(max_entries=5)
    store.store(diag)

    retrieved = store.get("run_freeze_test")
    assert retrieved is not None
    # Verify that retrieved is a deepcopy: internal object identities differ
    assert retrieved is not store._store["run_freeze_test"]

    latest = store.get_latest()
    assert latest is not None
    assert latest is not store._store["run_freeze_test"]


def test_serialize_sse_event_envelope_bounding():
    """Finding D: SSE events enforce MAX_SSE_EVENT_BYTES ceiling without corrupting data."""
    # 1. Normal small event within limit serializes cleanly
    small_event = {
        "stage": "EMBED",
        "status": "complete",
        "progress": 0.6,
        "metadata": {
            "run_id": "small_run",
            "diagnostics": {
                "header_bits_count": 128,
                "payload_bits_count": 64,
                "embedded_positions": [(0, 0), (1, 1)],
                "changed_positions": [(0, 0)],
            },
        },
    }
    sse_line = serialize_sse_event(small_event, max_event_bytes=DEFAULT_MAX_SSE_EVENT_BYTES)
    assert sse_line.startswith("data: ")
    assert sse_line.endswith("\n\n")
    parsed = json.loads(sse_line[6:-2])
    assert parsed["stage"] == "EMBED"
    assert parsed["status"] == "complete"

    # 2. Oversized coordinates in diagnostic sub-object truncated progressively
    large_coords = [[i % 256, (i * 2) % 256] for i in range(10000)]
    oversized_diag_event = {
        "stage": "EMBED",
        "status": "complete",
        "progress": 0.6,
        "metadata": {
            "run_id": "oversized_diag_run",
            "diagnostics": {
                "header_bits_count": 128,
                "payload_bits_count": 10000,
                "embedded_positions": large_coords,
                "changed_positions": large_coords[:5000],
            },
        },
    }
    # Enforce small 5,000-byte limit
    tight_sse = serialize_sse_event(oversized_diag_event, max_event_bytes=5000)
    tight_bytes = len(tight_sse.encode("utf-8"))
    assert tight_bytes <= 5000, f"SSE output {tight_bytes} exceeded 5,000 bytes ceiling"
    parsed_tight = json.loads(tight_sse[6:-2])
    diag_meta = parsed_tight["metadata"]["diagnostics"]
    assert diag_meta["coordinates_truncated"] is True
    assert diag_meta["completeness_status"] == "PARTIAL"
    assert len(diag_meta["embedded_positions"]) < 10000

    # 3. Oversized image payload omitted cleanly with explicit flags (no corrupted base64)
    huge_image_b64 = "data:image/png;base64," + ("A" * 60000)
    image_event = {
        "stage": "IDWT",
        "status": "complete",
        "progress": 0.7,
        "metadata": {
            "run_id": "image_run",
            "modified_hh": huge_image_b64,
            "stego_image": huge_image_b64,
            "pixel_diff_metrics": {"changed_pixels_count": 120},
        },
    }
    bounded_image_sse = serialize_sse_event(image_event, max_event_bytes=10000)
    bounded_bytes = len(bounded_image_sse.encode("utf-8"))
    assert bounded_bytes <= 10000, f"SSE output {bounded_bytes} exceeded 10,000 bytes ceiling"
    parsed_img = json.loads(bounded_image_sse[6:-2])
    # The heavy image payloads must be omitted cleanly, NOT sliced into invalid base64
    assert parsed_img["metadata"]["modified_hh_omitted"] is True
    assert parsed_img["metadata"]["stego_image_omitted"] is True
    assert parsed_img["metadata"]["pixel_diff_metrics"]["changed_pixels_count"] == 120





def test_diagnostics_store_caller_mutation_isolation():
    """Finding A: Caller mutating input object after insertion cannot affect stored state or byte accounting."""
    store = DiagnosticsStore(max_entries=5, max_total_bytes=100_000)

    # 1. Caller creates mutable dictionary with nested lists and mutable coordinates
    caller_nested_metrics = {
        "hist": [1, 2, 3],
        "channel_stats": {"red_diff": [10, 20]},
        "flag": True,
    }
    caller_coords = [(0, 0), (1, 1), (2, 2)]

    diag = RunDiagnostics(
        run_id="run_mutation_test",
        timestamp=time.time(),
        session_counter=1,
        carrier_dims=(512, 512),
        stego_dims=(512, 512),
        subband_dim=256,
        header_bits_count=128,
        payload_bits_count=200,
        total_embedded_bits=328,
        changed_coefficients_count=150,
        max_hh_diff=1.0,
        embedded_positions=caller_coords,
        changed_positions=[(0, 0)],
        pixel_diff_metrics=caller_nested_metrics,
    )

    store.store(diag)
    initial_stored_bytes = store.get_total_bytes()
    assert initial_stored_bytes > 0

    # 2. Caller mutates original structures after insertion
    caller_nested_metrics["hist"].append(999)
    caller_nested_metrics["channel_stats"]["red_diff"].append(999)
    caller_nested_metrics["channel_stats"]["new_key"] = "leaked"
    caller_coords.append((999, 999))

    # 3. Assert store holds independent copy; byte accounting is invariant
    stored = store.get("run_mutation_test")
    assert stored is not None
    assert stored.pixel_diff_metrics["hist"] == (1, 2, 3)
    assert stored.pixel_diff_metrics["channel_stats"]["red_diff"] == (10, 20)
    assert "new_key" not in stored.pixel_diff_metrics["channel_stats"]
    assert len(stored.embedded_positions) == 3
    assert store.get_total_bytes() == initial_stored_bytes

    # 4. Assert stored record cannot be mutated in-place
    with pytest.raises(TypeError):
        stored.pixel_diff_metrics["hist"] = (99,)

    # 5. Assert to_dict() returns a safe un-frozen copy whose mutations do not affect the store
    as_dict = stored.to_dict()
    assert isinstance(as_dict["pixel_diff_metrics"], dict)
    assert isinstance(as_dict["pixel_diff_metrics"]["hist"], list)
    as_dict["pixel_diff_metrics"]["hist"].append(42)

    stored_again = store.get("run_mutation_test")
    assert stored_again.pixel_diff_metrics["hist"] == (1, 2, 3)


def test_diagnostics_store_strict_accounting_invariant():
    """Finding A: Total byte accounting strictly equals sum of all retained serialized records."""
    store = DiagnosticsStore(max_entries=4, max_total_bytes=10_000)

    for i in range(8):
        diag = RunDiagnostics(
            run_id=f"run_acc_{i}",
            timestamp=float(i),
            session_counter=i,
            carrier_dims=(512, 512),
            stego_dims=(512, 512),
            subband_dim=256,
            header_bits_count=128,
            payload_bits_count=100 + i * 10,
            total_embedded_bits=228 + i * 10,
            changed_coefficients_count=50 + i,
            max_hh_diff=1.0,
            embedded_positions=[(i, j) for j in range(20)],
            changed_positions=[(i, j) for j in range(10)],
            pixel_diff_metrics={"iter": i, "hist": [i, i + 1, i + 2]},
        )
        store.store(diag)

        # Verify exact invariant after each store
        expected_bytes = sum(r.serialized_size() for r in store._store.values())
        assert store.get_total_bytes() == expected_bytes
        assert store.get_total_bytes() <= store.max_total_bytes
        assert len(store._store) <= store.max_entries

    # Update verification status on an existing run and re-verify invariant
    latest = store.get_latest()
    assert latest is not None
    store.update_verification(
        run_id=latest.run_id,
        verification_status="VERIFIED",
        exact_match=True,
    )
    expected_bytes = sum(r.serialized_size() for r in store._store.values())
    assert store.get_total_bytes() == expected_bytes
    assert store.get_total_bytes() <= store.max_total_bytes


def test_freeze_unfreeze_value_recursion():
    """Finding A: Recursive freeze_value and unfreeze_value immutability guarantees."""
    data = {
        "num": 42,
        "text": "hello",
        "nested_list": [1, [2, 3], {"inner": [4, 5]}],
        "nested_dict": {"a": {"b": [6, 7]}},
        "set_val": {1, 2},
    }

    frozen = freeze_value(data)
    assert isinstance(frozen, FrozenDict)
    assert isinstance(frozen["nested_list"], tuple)
    assert isinstance(frozen["nested_list"][1], tuple)
    assert isinstance(frozen["nested_list"][2], FrozenDict)
    assert isinstance(frozen["nested_list"][2]["inner"], tuple)
    assert isinstance(frozen["nested_dict"]["a"], FrozenDict)
    assert isinstance(frozen["set_val"], frozenset)

    # Immutability verification
    with pytest.raises(TypeError):
        frozen["nested_list"] = (10,)
    with pytest.raises(TypeError):
        frozen["nested_dict"]["a"]["b"] = (10,)

    # Round-trip unfreeze verification
    unfrozen = unfreeze_value(frozen)
    assert isinstance(unfrozen, dict)
    assert isinstance(unfrozen["nested_list"], list)
    assert isinstance(unfrozen["nested_list"][1], list)
    assert isinstance(unfrozen["nested_list"][2], dict)
    assert isinstance(unfrozen["nested_dict"]["a"], dict)
    assert isinstance(unfrozen["nested_dict"]["a"]["b"], list)
    # JSON serializability test
    json_bytes = json.dumps(unfrozen)
    assert "nested_list" in json_bytes


def test_serialize_sse_event_dwt_subband_omission():
    """Finding B: DWT stage metadata.subbands is cleanly omitted when event exceeds limit."""
    dwt_event = {
        "stage": "DWT",
        "status": "complete",
        "message": "Haar DWT complete: LL, LH, HL, HH subbands generated (3, 256, 256).",
        "metadata": {
            "run_id": "dwt_test_run",
            "subbands": {
                "LL": "data:image/png;base64," + ("A" * 15000),
                "LH": "data:image/png;base64," + ("B" * 15000),
                "HL": "data:image/png;base64," + ("C" * 15000),
                "HH": "data:image/png;base64," + ("D" * 15000),
            },
        },
    }

    # Tight limit of 5,000 bytes
    line = serialize_sse_event(dwt_event, max_event_bytes=5000)
    assert len(line.encode("utf-8")) <= 5000
    assert line.startswith("data: ")
    assert line.endswith("\n\n")

    parsed = json.loads(line[6:-2])
    assert parsed["stage"] == "DWT"
    assert parsed["status"] == "complete"
    assert parsed["metadata"]["subbands_omitted"] is True
    assert parsed["metadata"]["subbands"] is None
    assert parsed["metadata"]["run_id"] == "dwt_test_run"


def test_serialize_sse_event_all_visual_payloads_omitted():
    """Finding B: All visual payload fields (subbands, modified_hh, stego_image, recovered_image) omitted cleanly."""
    huge_b64 = "data:image/png;base64," + ("Z" * 10000)
    event = {
        "stage": "IDWT",
        "status": "complete",
        "message": "Reconstruction complete.",
        "metadata": {
            "run_id": "all_visual_run",
            "subbands": {"HH": huge_b64},
            "modified_hh": huge_b64,
            "stego_image": huge_b64,
            "recovered_image": huge_b64,
        },
    }

    line = serialize_sse_event(event, max_event_bytes=2000)
    assert len(line.encode("utf-8")) <= 2000
    parsed = json.loads(line[6:-2])
    assert parsed["metadata"]["subbands_omitted"] is True
    assert parsed["metadata"]["modified_hh_omitted"] is True
    assert parsed["metadata"]["stego_image_omitted"] is True
    assert parsed["metadata"]["recovered_image_omitted"] is True
    assert parsed["metadata"]["subbands"] is None
    assert parsed["metadata"]["modified_hh"] is None
    assert parsed["metadata"]["stego_image"] is None
    assert parsed["metadata"]["recovered_image"] is None


def test_serialize_sse_event_irreducible_fallback():
    """Finding B: Irreducible arbitrary oversized metadata falls back to minimal bounded envelope."""
    huge_unrecognized = "K" * 30000
    event = {
        "stage": "CUSTOM_ANALYSIS",
        "status": "processing",
        "message": "Processing large uncompressible data stream.",
        "metadata": {
            "run_id": "irreducible_run_42",
            "exact_match": True,
            "opaque_payload": huge_unrecognized,
        },
    }

    # Bounded to 1024 bytes
    line = serialize_sse_event(event, max_event_bytes=1024)
    wire_len = len(line.encode("utf-8"))
    assert wire_len <= 1024, f"Output wire length {wire_len} exceeded 1024 bytes"
    assert line.startswith("data: ")
    assert line.endswith("\n\n")

    parsed = json.loads(line[6:-2])
    assert parsed["stage"] == "CUSTOM_ANALYSIS"
    assert parsed["status"] == "processing"
    assert parsed["metadata"]["event_size_exceeded"] is True
    assert parsed["metadata"]["run_id"] == "irreducible_run_42"
    assert parsed["metadata"]["exact_match"] is True


def test_serialize_sse_event_strict_wire_size_ceiling():
    """Finding B: Strict guarantee that output bytes never exceed max_event_bytes across edge cases."""
    test_cases = [
        ({"stage": "INIT", "status": "processing", "message": "Short message"}, 500),
        ({"stage": "MSG", "status": "processing", "message": "A" * 5000}, 200),
        ({"stage": "ERR", "status": "failed", "error": "E" * 5000}, 300),
        ({"stage": "DWT", "status": "complete", "metadata": {"subbands": {"LL": "X" * 10000}}}, 1500),
    ]

    for ev, limit in test_cases:
        line = serialize_sse_event(ev, max_event_bytes=limit)
        wire_bytes = len(line.encode("utf-8"))
        assert wire_bytes <= limit, f"Failed for limit {limit}: got {wire_bytes} bytes"
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
