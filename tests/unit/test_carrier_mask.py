"""Unit tests for Deterministic Mask Generator (T-014)."""

import pytest

from src.carrier.mask import DeterministicMaskGenerator, MaskGenerationError


def test_t014_mask_determinism():
    """T-014: Same context + same payload length = exactly identical positions."""
    session_counter = 0x12345678
    payload_len = 1024  # 1024 bytes = 8192 bits

    mask1 = DeterministicMaskGenerator.generate_mask(session_counter, payload_len)
    mask2 = DeterministicMaskGenerator.generate_mask(session_counter, payload_len)

    assert mask1.header_positions == mask2.header_positions
    assert mask1.payload_positions == mask2.payload_positions
    assert len(mask1.header_positions) == 128
    assert len(mask1.payload_positions) == 8192


def test_t014_mask_different_context_yields_different_positions():
    """T-014: Different session counters produce different non-colliding positions."""
    mask_a = DeterministicMaskGenerator.generate_mask(1, 500)
    mask_b = DeterministicMaskGenerator.generate_mask(2, 500)

    assert mask_a.header_positions == mask_b.header_positions  # Fixed anchors
    assert mask_a.payload_positions != mask_b.payload_positions


def test_t014_mask_non_overlapping_and_within_bounds():
    """T-014: No duplicate coordinates; coordinates stay strictly within 256x256 grid."""
    mask = DeterministicMaskGenerator.generate_mask(99, 1000)

    # Check bounds
    for r, c in mask.header_positions + mask.payload_positions:
        assert 0 <= r < 256
        assert 0 <= c < 256

    # Header and payload positions must be disjoint
    header_set = set(mask.header_positions)
    payload_set = set(mask.payload_positions)
    assert len(header_set) == 128
    assert len(payload_set) == 8000
    assert len(header_set.intersection(payload_set)) == 0


def test_t014_mask_capacity_exceeded():
    """T-014: Exceeding maximum bit capacity raises MaskGenerationError."""
    with pytest.raises(MaskGenerationError, match="exceeds available positions"):
        DeterministicMaskGenerator.generate_payload_positions(1, 9000)  # > 8,176 B
