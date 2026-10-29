"""Deterministic Embedding Position Mask Generator for PaladinMedia Phase 2.

Implements the deterministic mask generator per Architecture §14 and Sprint-0 IRS §6.3:
- Reversible and non-circular: The 16-byte carrier header (128 bits) occupies fixed anchor
  positions (indices 0..127) so the receiver can extract the header without prior mask knowledge.
- The remaining payload positions are deterministically derived from:
    seed = H(b"PALADIN_MASK_SEED" || session_counter || payload_bit_count)
- Generates reproducible, non-colliding (row, col) coordinates within the 256x256 subband.
"""

from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

try:
    from src.core.exceptions import PaladinSecurityError
except ImportError:
    from core.exceptions import PaladinSecurityError  # type: ignore[no-redef]


def resolve_subband_dims(subband_dim: Union[int, Tuple[int, int], Sequence[int]]) -> Tuple[int, int]:
    """Resolves subband dimensions into a (subband_h, subband_w) tuple."""
    if isinstance(subband_dim, (tuple, list)):
        return int(subband_dim[0]), int(subband_dim[1])
    return int(subband_dim), int(subband_dim)


class MaskGenerationError(PaladinSecurityError, ValueError):
    """Raised when mask parameters or position generation fails."""
    pass


@dataclass(frozen=True)
class EmbeddingMask:
    """Encapsulates deterministic 2D coordinates for header and payload bits."""

    header_positions: Tuple[Tuple[int, int], ...]
    payload_positions: Tuple[Tuple[int, int], ...]

    @property
    def total_positions(self) -> int:
        return len(self.header_positions) + len(self.payload_positions)


class DeterministicMaskGenerator:
    """Generates non-circular, deterministic 2D coordinates for carrier embedding."""

    SUBBAND_DIM: int = 256
    TOTAL_POSITIONS: int = SUBBAND_DIM * SUBBAND_DIM  # 65,536 positions
    HEADER_BITS: int = 16 * 8  # 128 bits for 16-byte carrier header
    MAX_PAYLOAD_BITS: int = TOTAL_POSITIONS - HEADER_BITS  # 65,408 bits (8,176 bytes)

    @classmethod
    def index_to_coord(
        cls, index: int, subband_dim: Union[int, Tuple[int, int], Sequence[int]] = SUBBAND_DIM
    ) -> Tuple[int, int]:
        """Converts linear index to (row, col) within subband grid."""
        _, sub_w = resolve_subband_dims(subband_dim)
        return divmod(index, sub_w)

    @classmethod
    def generate_header_positions(
        cls, subband_dim: Union[int, Tuple[int, int], Sequence[int]] = SUBBAND_DIM
    ) -> Tuple[Tuple[int, int], ...]:
        """Returns the fixed anchor coordinates for the 128 header bits (indices 0..127)."""
        return tuple(cls.index_to_coord(i, subband_dim) for i in range(cls.HEADER_BITS))

    @classmethod
    def generate_payload_positions(
        cls,
        session_counter: int,
        payload_length_bytes: int,
        subband_dim: Union[int, Tuple[int, int], Sequence[int]] = SUBBAND_DIM,
    ) -> Tuple[Tuple[int, int], ...]:
        """Derives deterministic, non-colliding coordinates for the payload bits.

        Args:
            session_counter: 32-bit session/file counter from carrier header.
            payload_length_bytes: Exact byte length of the Reed-Solomon protected stream.
            subband_dim: Dimension of the subband (int for square or (sub_h, sub_w) tuple).

        Returns:
            Tuple of (row, col) coordinate pairs of length payload_length_bytes * 8.

        Raises:
            MaskGenerationError: If payload length exceeds maximum capacity.
        """
        sub_h, sub_w = resolve_subband_dims(subband_dim)
        total_positions = sub_h * sub_w
        max_payload_bits = total_positions - cls.HEADER_BITS

        payload_bits = payload_length_bytes * 8
        if payload_bits > max_payload_bits:
            raise MaskGenerationError(
                f"Requested payload bits ({payload_bits}) exceeds available positions ({max_payload_bits})"
            )
        if payload_bits == 0:
            return ()

        # Deterministic seed derivation: SHA-256(domain || session_counter || payload_bits)
        hasher = hashlib.sha256()
        hasher.update(b"PALADIN_MASK_SEED_V1")
        hasher.update(struct.pack(">II", session_counter, payload_bits))
        seed_int = int.from_bytes(hasher.digest()[:8], "big")

        rng = random.Random(seed_int)

        # Sample exactly payload_bits without replacement from indices [128..total_positions-1]
        candidate_range = range(cls.HEADER_BITS, total_positions)
        selected_indices = rng.sample(candidate_range, payload_bits)

        return tuple(cls.index_to_coord(idx, (sub_h, sub_w)) for idx in selected_indices)

    @classmethod
    def generate_mask(
        cls,
        session_counter: int,
        payload_length_bytes: int,
        subband_dim: Union[int, Tuple[int, int], Sequence[int]] = SUBBAND_DIM,
    ) -> EmbeddingMask:
        """Generates the complete embedding mask containing header and payload coordinates."""
        header_pos = cls.generate_header_positions(subband_dim=subband_dim)
        payload_pos = cls.generate_payload_positions(
            session_counter, payload_length_bytes, subband_dim=subband_dim
        )
        return EmbeddingMask(header_positions=header_pos, payload_positions=payload_pos)
