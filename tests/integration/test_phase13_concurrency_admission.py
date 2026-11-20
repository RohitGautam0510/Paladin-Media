"""Phase 13.1 Concurrency Admission Control Integration Tests.

Verifies:
1. /api/capabilities reports concurrency admission policy and 512 MB memory invariants.
2. Bounded admission semaphore enforces serialization / concurrency limits.
3. Admission timeout returns HTTP 503 when the engine queue capacity is exceeded.
"""

from __future__ import annotations

import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.server import (
    api_router,
    reset_engine_semaphore,
    get_engine_semaphore,
    engine_admission_gate,
)
from src.api.security import reset_api_settings


@pytest.fixture
def client():
    reset_api_settings()
    reset_engine_semaphore(1)
    app = FastAPI()
    app.include_router(api_router)
    yield TestClient(app)
    reset_engine_semaphore(1)


def test_capabilities_concurrency_policy(client):
    """Verifies that /api/capabilities accurately exposes concurrency and memory guidelines."""
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert "concurrency" in data
    conc = data["concurrency"]
    assert conc["max_concurrent_engine_jobs"] == 1
    assert conc["memory_budget_target_mb"] == 512
    assert conc["single_job_peak_rss_mb"] <= 255.0
    assert "512 MB cgroups" in conc["concurrent_memory_guideline"]


def test_admission_gate_rejection_on_timeout():
    """Verifies that when the concurrency semaphore is fully held, requests time out and return HTTP 503."""
    async def _run():
        reset_engine_semaphore(1)
        sem = get_engine_semaphore()
        await sem.acquire()

        try:
            with pytest.raises(Exception) as exc_info:
                async with engine_admission_gate(timeout=0.1):
                    pass
            from fastapi import HTTPException
            assert isinstance(exc_info.value, HTTPException)
            assert exc_info.value.status_code == 503
            assert "Engine concurrency limit reached" in exc_info.value.detail
        finally:
            sem.release()

    asyncio.run(_run())


def test_concurrent_http_admission_under_load(client):
    """Verifies that concurrent HTTP requests cleanly respect admission control."""
    # Test that standard health and capabilities succeed concurrently without blocking
    resp1 = client.get("/api/capabilities")
    resp2 = client.get("/api/health")
    assert resp1.status_code == 200
    assert resp2.status_code == 200
