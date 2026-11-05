"""Carrier capacity accounting and hard-gate verification for PaladinMedia.

Implements explicit capacity accounting per Architecture §10.2 and Wire Protocol v2:
- 512x512 RGB carrier with 1-level 2D Haar DWT has 256x256 HH subband per channel.
- Allocation: 1 HH channel (Red channel) at 1 bit per coefficient position.
- Raw capacity: 256 * 256 = 65,536 bits = 8,192 bytes.
- Wire Protocol v2 Carrier Header: 24 bytes (16 data + 8 RS parity).
- Maximum Protected Payload Capacity (v2): 8,192 - 24 = 8,168 bytes.
- Legacy Wire Protocol v1 Carrier Header: 16 bytes.
- Maximum Protected Payload Capacity (v1): 8,192 - 16 = 8,176 bytes.
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
        carrier_header_bytes: Steganographic carrier header size in bytes (24 for v2, 16 for v1).
        max_protected_bytes: Maximum allowed Reed-Solomon protected payload (8,168 for v2, 8,176 for v1).
        protected_payload_bytes: Actual size of the protected payload to be embedded.
        usable_capacity_bytes: Maximum application secret plaintext data capacity with 0 metadata bytes.
        remaining_capacity_bytes: Remaining space in the carrier in bytes.
        fits: True if protected payload fits within carrier capacity; False otherwise.
        status: "FITS" or "EXCEEDS_CAPACITY".
        reference_metadata_capacity_bytes: Reference usable application secret capacity with 2-byte minimal metadata.
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
    """Calculates and enforces transform-domain carrier capacity across wire versions."""

    # 256 * 256 positions at 1 bit per coefficient
    RAW_CAPACITY_BITS: int = 256 * 256  # 65,536 bits
    RAW_CAPACITY_BYTES: int = RAW_CAPACITY_BITS // 8  # 8,192 bytes

    # Header sizing
    CARRIER_HEADER_BYTES_V1: int = 16  # 16 bytes = 128 bits
    CARRIER_HEADER_BYTES_V2: int = 24  # 24 bytes = 192 bits (16 data + 8 RS parity)
    CARRIER_HEADER_BYTES: int = 24     # Wire Protocol v2 default

    MAX_PROTECTED_BYTES_V1: int = RAW_CAPACITY_BYTES - CARRIER_HEADER_BYTES_V1  # 8,176 bytes
    MAX_PROTECTED_BYTES_V2: int = RAW_CAPACITY_BYTES - CARRIER_HEADER_BYTES_V2  # 8,168 bytes
    MAX_PROTECTED_BYTES: int = MAX_PROTECTED_BYTES_V2                          # 8,168 bytes

    # Baseline reference constants for 1 HH channel
    RS_K: int = 223  # RS data symbols per block (2t=32)
    RS_PARITY: int = 32  # RS parity symbols per block
    RS_LENGTH_PREFIX: int = 4  # RS 4-byte length prefix
    AES_OVERHEAD: int = 12 + 16  # 12B nonce + 16B GHASH tag = 28 bytes
    SERIALIZER_BASE_FRAMING: int = 18  # 10B prefix + 4B len + 4B CRC32 = 18 bytes
    resolve_subband_dims = staticmethod(resolve_subband_dims)

    @classmethod
    def get_header_bytes(cls, version: int = 2) -> int:
        """Returns header wire byte allocation for the requested wire version."""
        return cls.CARRIER_HEADER_BYTES_V2 if version == 2 else cls.CARRIER_HEADER_BYTES_V1

    @classmethod
    def calculate_usable_secret_capacity(
        cls,
        metadata_len: Optional[int] = None,
        subband_dim: Union[int, Tuple[int, int], Sequence[int]] = 256,
        include_shortened_blocks: bool = True,
        rs_parity: int = 32,
        version: int = 2,
        header_bytes: Optional[int] = None,
    ) -> int:
        """Derives the maximum usable secret plaintext capacity.

        Args:
            metadata_len: Length in bytes of the JSON metadata string. If None, defaults to 2
                          (reference metadata baseline).
            subband_dim: Subband dimension (int for square, or (sub_h, sub_w) tuple).
            include_shortened_blocks: If True (default), accounts for shortened RS blocks fitting in
                                      the carrier remainder. If False, counts full blocks only.
            rs_parity: Reed-Solomon parity symbol count per block (2t; default 32).
            version: Wire protocol version (2 for v2; 1 for v1).
            header_bytes: Optional explicit header byte size override.
        """
        meta_len = 2 if metadata_len is None else metadata_len
        sub_h, sub_w = resolve_subband_dims(subband_dim)
        raw_bits = sub_h * sub_w
        raw_bytes = raw_bits // 8
        hdr_bytes = cls.get_header_bytes(version=version) if header_bytes is None else header_bytes
        max_protected = max(0, raw_bytes - hdr_bytes)
        avail_for_blocks = max_protected - cls.RS_LENGTH_PREFIX
        if avail_for_blocks <= 0:
            return 0
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
        version: int = 2,
        header_bytes: Optional[int] = None,
    ) -> CapacityReport:
        """Evaluates capacity without throwing an exception.

        Args:
            protected_payload_bytes: Total byte length of the Reed-Solomon protected stream.
            subband_dim: Subband dimension (int for square or (sub_h, sub_w) tuple).
            rs_parity: RS parity symbols per block (2t; default 32).
            version: Wire protocol version (2 for v2; 1 for v1).
            header_bytes: Optional explicit header byte size override.

        Returns:
            CapacityReport detailing allocation and whether it fits.
        """
        sub_h, sub_w = resolve_subband_dims(subband_dim)
        raw_bits = sub_h * sub_w
        raw_bytes = raw_bits // 8
        hdr_bytes = cls.get_header_bytes(version=version) if header_bytes is None else header_bytes
        max_protected = max(0, raw_bytes - hdr_bytes)
        total_required = protected_payload_bytes + hdr_bytes
        remaining = raw_bytes - total_required
        fits = protected_payload_bytes <= max_protected
        usable_capacity = cls.calculate_usable_secret_capacity(
            metadata_len=0,
            subband_dim=subband_dim,
            include_shortened_blocks=True,
            rs_parity=rs_parity,
            version=version,
            header_bytes=hdr_bytes,
        )
        reference_capacity = cls.calculate_usable_secret_capacity(
            metadata_len=2,
            subband_dim=subband_dim,
            include_shortened_blocks=True,
            rs_parity=rs_parity,
            version=version,
            header_bytes=hdr_bytes,
        )

        return CapacityReport(
            raw_capacity_bytes=raw_bytes,
            carrier_header_bytes=hdr_bytes,
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
        version: int = 2,
        header_bytes: Optional[int] = None,
    ) -> CapacityReport:
        """Enforces capacity as a hard gate, raising PayloadCapacityExceededError on failure.

        Args:
            protected_payload_bytes: Total byte length of the Reed-Solomon protected stream.
            subband_dim: Subband dimension (int for square or (sub_h, sub_w) tuple).
            rs_parity: RS parity symbols per block (2t; default 32).
            version: Wire protocol version (2 for v2; 1 for v1).
            header_bytes: Optional explicit header byte size override.

        Returns:
            CapacityReport if capacity check passes.

        Raises:
            PayloadCapacityExceededError: If protected payload exceeds maximum carrier capacity.
        """
        report = cls.evaluate(
            protected_payload_bytes,
            subband_dim=subband_dim,
            rs_parity=rs_parity,
            version=version,
            header_bytes=header_bytes,
        )
        if not report.fits:
            usable_ref = max(0, report.usable_capacity_bytes - 2)
            raise PayloadCapacityExceededError(
                f"PAYLOAD_CAPACITY_EXCEEDED: Protected payload ({protected_payload_bytes} B) "
                f"exceeds maximum carrier capacity ({report.max_protected_bytes} B). "
                f"Usable capacity is {report.usable_capacity_bytes} B (with 0 metadata bytes, or "
                f"{usable_ref} B with 2B reference metadata) under RS parity {rs_parity} and Wire Protocol v{version}. Embedding aborted."
            )
        return report
