"""Unit tests for /api/capabilities and /api/telemetry endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import api_router, get_api_settings, reset_api_settings
from src.api.telemetry import telemetry_tracker
from fastapi import FastAPI


@pytest.fixture
def client():
    reset_api_settings()
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


def test_capabilities_endpoint(client):
    """Verifies that GET /api/capabilities exposes supported engines, protocol versions, and telemetry."""
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] in ("OPERATIONAL", "DEGRADED")
    assert data["active_engine"] == "neural_spatial_onnx"
    assert len(data["available_engines"]) >= 4

    engine_ids = [e["id"] for e in data["available_engines"]]
    assert "neural_spatial_onnx" in engine_ids
    assert "acs_sub_inn_onnx" in engine_ids
    assert "acs_sub_inn_pytorch" in engine_ids
    assert "parity_classical" in engine_ids

    assert data["supported_protocol_versions"] == [2]
    assert "telemetry" in data
    assert data["telemetry"]["available"] is True
    assert isinstance(data["telemetry"]["is_estimated"], bool)
    assert len(data["pipeline_stages"]) >= 14


def test_telemetry_endpoint(client):
    """Verifies that GET /api/telemetry/{run_id} returns 404 for missing runs and 200 for tracked runs."""
    resp_404 = client.get("/api/telemetry/non_existent_run_999")
    assert resp_404.status_code == 404

    # Record a test run
    run_id = "test_api_run_abc"
    telemetry_tracker.start_tracking(run_id)
    _ = telemetry_tracker.stop_tracking(run_id)

    resp_200 = client.get(f"/api/telemetry/{run_id}")
    assert resp_200.status_code == 200
    m = resp_200.json()
    assert m["run_id"] == run_id
    assert "energy_kwh" in m
    assert "co2_emissions_grams" in m
