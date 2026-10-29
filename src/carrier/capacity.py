"""Carrier capacity accounting and hard-gate verification for PaladinMedia Phase 2.

Implements explicit capacity accounting per Architecture §10.2 and Sprint-0 IRS §6.4:
- 512x512 RGB carrier with 1-level 2D Haar DWT has 256x256 HH subband per channel.
- MVP 0 allocation: exactly 1 HH channel (Red channel) at 1 bit per coefficient position.
- Raw capacity: 256 * 256 = 65,536 bits = 8,192 bytes.
- Carrier Header: 16 bytes.
- Maximum Protected Payload Capacity: 8,192 - 16 = 8,176 bytes.
- Maximum Usable Plaintext Capacity: ~7,088 bytes after RS parity (32 blocks), AES-GCM (28B),
  and SecretSerializer framing (18B).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

try:
    from src.core.exceptions import PaladinSecurityError
except ImportError:
    from core.exceptions import PaladinSecurityError  # type: ignore[no-redef]


def resolve_subband_dims(subband_dim: Union[int, Tuple[int, int], Sequence[int]]) -> Tuple[int, int]:
    """Resolves subband dimensions into a (subband_h, subband_w) tuple."""
    if isinstance(subband_dim, (tuple, list)):
        return int(subband_dim[0]), int(subband_dim[1])
    return int(subband_dim), int(subband_dim)


class PayloadCapacityExceededError(PaladinSecurityError, ValueError):
    """Raised when the protected payload size exceeds the usable carrier capacity."""
    pass


@dataclass(frozen=True)
class CapacityReport:
    """Detailed capacity accounting report.

    Attributes:
        raw_capacity_bytes: Total bit budget in bytes for 1 HH channel (8,192).
        carrier_header_bytes: Fixed steganographic carrier header size in bytes (16).
        max_protected_bytes: Maximum allowed Reed-Solomon protected payload (8,176).
        protected_payload_bytes: Actual size of the protected payload to be embedded.
        usable_capacity_bytes: Maximum application secret plaintext data capacity with 0 metadata bytes (7,090 B for 2t=32).
        remaining_capacity_bytes: Remaining space in the carrier in bytes.
        fits: True if protected payload fits within carrier capacity; False otherwise.
        status: "FITS" or "EXCEEDS_CAPACITY".
        reference_metadata_capacity_bytes: Reference usable application secret capacity with 2-byte minimal metadata (7,088 B for 2t=32).
    """
    raw_capacity_bytes: int
    carrier_header_bytes: int
    max_protected_bytes: int
    protected_payload_bytes: int
    usable_capacity_bytes: int
    remaining_capacity_bytes: int
    fits: bool
    status: str
    reference_metadata_capacity_bytes: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "raw_capacity_bytes": self.raw_capacity_bytes,
            "carrier_header_bytes": self.carrier_header_bytes,
            "max_protected_bytes": self.max_protected_bytes,
            "protected_payload_bytes": self.protected_payload_bytes,
            "usable_capacity_bytes": self.usable_capacity_bytes,
            "reference_metadata_capacity_bytes": self.reference_metadata_capacity_bytes,
            "remaining_capacity_bytes": self.remaining_capacity_bytes,
            "fits": self.fits,
            "status": self.status,
        }


class CapacityCalculator:
    """Calculates and enforces transform-domain carrier capacity."""

    # 256 * 256 positions at 1 bit per coefficient
    RAW_CAPACITY_BITS: int = 256 * 256  # 65,536 bits
    RAW_CAPACITY_BYTES: int = RAW_CAPACITY_BITS // 8  # 8,192 bytes
    CARRIER_HEADER_BYTES: int = 16  # 16 bytes = 128 bits
    MAX_PROTECTED_BYTES: int = RAW_CAPACITY_BYTES - CARRIER_HEADER_BYTES  # 8,176 bytes

    # Baseline reference constants for 1 HH channel
    RS_K: int = 223  # RS data symbols per block
    RS_PARITY: int = 32  # RS parity symbols per block
    RS_LENGTH_PREFIX: int = 4  # RS 4-byte length prefix
    AES_OVERHEAD: int = 12 + 16  # 12B nonce + 16B GHASH tag = 28 bytes
    SERIALIZER_BASE_FRAMING: int = 18  # 10B prefix + 4B len + 4B CRC32 = 18 bytes
    resolve_subband_dims = staticmethod(resolve_subband_dims)

    @classmethod
    def calculate_usable_secret_capacity(
        cls,
        metadata_len: Optional[int] = None,
        subband_dim: Union[int, Tuple[int, int], Sequence[int]] = 256,
        include_shortened_blocks: bool = True,
        rs_parity: int = 32,
    ) -> int:
        """Derives the maximum usable secret plaintext capacity.

        Accounting:
            sub_h, sub_w = resolve_subband_dims(subband_dim)
            raw_positions = sub_h * sub_w
            raw_bytes = raw_positions // 8 (8,192 B for 512x512; 32,768 B for 1024x1024)
            max_protected = raw_bytes - 16 (8,176 B for 512x512; 32,752 B for 1024x1024)
            avail_for_blocks = max_protected - 4 (8,172 B for 512x512; 32,748 B for 1024x1024)
            full_blocks = avail_for_blocks // 255 (32 for 512x512; 128 for 1024x1024)
            remainder = avail_for_blocks % 255 (12 for 512x512; 108 for 1024x1024)

        Args:
            metadata_len: Length in bytes of the JSON metadata string. If None, defaults to 2
                          (reference metadata baseline).
            subband_dim: Subband dimension (int for square, or (sub_h, sub_w) tuple for arbitrary even dimensions).
            include_shortened_blocks: If True (default), accounts for shortened RS blocks fitting in
                                      the carrier remainder. If False, counts full blocks only
                                      for conservative baseline parity.
            rs_parity: Reed-Solomon parity symbol count per block (2t; default 32).
        """
        # Default to 2 bytes (standard reference metadata) to match Sprint-0 baseline (7,088 B)
        meta_len = 2 if metadata_len is None else metadata_len
        sub_h, sub_w = resolve_subband_dims(subband_dim)
        raw_bits = sub_h * sub_w
        raw_bytes = raw_bits // 8
        max_protected = raw_bytes - cls.CARRIER_HEADER_BYTES
        avail_for_blocks = max_protected - cls.RS_LENGTH_PREFIX
        num_full_blocks = avail_for_blocks // 255
        remainder = avail_for_blocks % 255

        rs_k = 255 - rs_parity
        partial_data = max(0, remainder - rs_parity) if include_shortened_blocks else 0
        max_encrypted = num_full_blocks * rs_k + partial_data
        max_serialized = max_encrypted - cls.AES_OVERHEAD
        usable_secret = max_serialized - cls.SERIALIZER_BASE_FRAMING - meta_len
        return max(0, usable_secret)

    @classmethod
    def evaluate(
        cls,
        protected_payload_bytes: int,
        subband_dim: Union[int, Tuple[int, int], Sequence[int]] = 256,
        rs_parity: int = 32,
    ) -> CapacityReport:
        """Evaluates capacity without throwing an exception.

        Args:
            protected_payload_bytes: Total byte length of the Reed-Solomon protected stream.
            subband_dim: Subband dimension (int for square or (sub_h, sub_w) tuple).
            rs_parity: RS parity symbols per block (2t; default 32).

        Returns:
            CapacityReport detailing allocation and whether it fits.
        """
        sub_h, sub_w = resolve_subband_dims(subband_dim)
        raw_bits = sub_h * sub_w
        raw_bytes = raw_bits // 8
        max_protected = raw_bytes - cls.CARRIER_HEADER_BYTES
        total_required = protected_payload_bytes + cls.CARRIER_HEADER_BYTES
        remaining = raw_bytes - total_required
        fits = protected_payload_bytes <= max_protected
        usable_capacity = cls.calculate_usable_secret_capacity(
            metadata_len=0,
            subband_dim=subband_dim,
            include_shortened_blocks=True,
            rs_parity=rs_parity,
        )
        reference_capacity = cls.calculate_usable_secret_capacity(
            metadata_len=2,
            subband_dim=subband_dim,
            include_shortened_blocks=True,
            rs_parity=rs_parity,
        )

        return CapacityReport(
            raw_capacity_bytes=raw_bytes,
            carrier_header_bytes=cls.CARRIER_HEADER_BYTES,
            max_protected_bytes=max_protected,
            protected_payload_bytes=protected_payload_bytes,
            usable_capacity_bytes=usable_capacity,
            remaining_capacity_bytes=remaining if fits else 0,
            fits=fits,
            status="FITS" if fits else "EXCEEDS_CAPACITY",
            reference_metadata_capacity_bytes=reference_capacity,
        )

    @classmethod
    def check_capacity(
        cls,
        protected_payload_bytes: int,
        subband_dim: Union[int, Tuple[int, int], Sequence[int]] = 256,
        rs_parity: int = 32,
    ) -> CapacityReport:
        """Enforces capacity as a hard gate, raising PayloadCapacityExceededError on failure.

        Args:
            protected_payload_bytes: Total byte length of the Reed-Solomon protected stream.
            subband_dim: Subband dimension (int for square or (sub_h, sub_w) tuple).
            rs_parity: RS parity symbols per block (2t; default 32).

        Returns:
            CapacityReport if capacity check passes.

        Raises:
            PayloadCapacityExceededError: If protected payload exceeds maximum carrier capacity.
        """
        report = cls.evaluate(protected_payload_bytes, subband_dim=subband_dim, rs_parity=rs_parity)
        if not report.fits:
            usable_ref = max(0, report.usable_capacity_bytes - 2)
            raise PayloadCapacityExceededError(
                f"PAYLOAD_CAPACITY_EXCEEDED: Protected payload ({protected_payload_bytes} B) "
                f"exceeds maximum carrier capacity ({report.max_protected_bytes} B). "
                f"Usable capacity is {report.usable_capacity_bytes} B (with 0 metadata bytes, or "
                f"{usable_ref} B with 2B reference metadata) under RS parity {rs_parity}. Embedding aborted."
            )
        return report
