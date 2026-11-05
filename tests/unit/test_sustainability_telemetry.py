"""Unit tests for PaladinMedia Sustainability Telemetry Tracker."""

from __future__ import annotations

import time
import pytest

from src.api.telemetry import (
    CODECARBON_AVAILABLE,
    SustainabilityMetrics,
    SustainabilityTelemetryTracker,
    telemetry_tracker,
)


def test_telemetry_tracker_lifecycle():
    """Verifies that start_tracking and stop_tracking produce genuine non-zero metrics."""
    tracker = SustainabilityTelemetryTracker(max_records=10)
    run_id = "test_run_12345"

    tracker.start_tracking(run_id)
    # Simulate a small workload
    time.sleep(0.05)
    _ = [i ** 2 for i in range(10000)]

    metrics = tracker.stop_tracking(run_id)

    assert metrics.run_id == run_id
    assert metrics.execution_time_ms > 40.0  # At least 40 ms
    assert metrics.estimated_power_watts > 0.0
    assert metrics.energy_kwh > 0.0
    assert metrics.co2_emissions_grams > 0.0
    assert metrics.memory_rss_mb > 0.0

    if not CODECARBON_AVAILABLE:
        assert metrics.is_estimated is True
        assert metrics.telemetry_source == "estimated_hardware_model"

    # Verify history retrieval
    retrieved = tracker.get_metrics(run_id)
    assert retrieved is not None
    assert retrieved.run_id == run_id


def test_telemetry_tracker_unknown_run():
    """Verifies graceful handling of stop_tracking for unknown or unstarted run_id."""
    tracker = SustainabilityTelemetryTracker()
    metrics = tracker.stop_tracking("unknown_run_999")

    assert metrics.run_id == "unknown_run_999"
    assert metrics.telemetry_source == "unavailable"
