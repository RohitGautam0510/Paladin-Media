"""Steganographic Carrier Header Codec for PaladinMedia Phase 2.

Implements the 16-byte fixed carrier header per Architecture §8.3/§10.1 and
Sprint-0 IRS §6.2:

Wire Format Specification (16 bytes = 128 bits):
+-----------------+---------+---------+-----------------------------------+
| Field Name      | Type    | Size    | Description                       |
+-----------------+---------+---------+-----------------------------------+
| Magic           | bytes   | 4 bytes | Fixed ASCII: b"PLDH" (0x504C4448) |
| Protocol Version| uint8   | 1 byte  | 0x01 (MVP 0 / Phase 2)            |
| Flags           | uint8   | 1 byte  | 0x00 (Reserved)                   |
| Reserved        | uint16  | 2 bytes | Big-endian 0x0000                 |
| Session Counter | uint32  | 4 bytes | Big-endian session/file context   |
| Payload Length  | uint32  | 4 bytes | Big-endian length of RS payload   |
+-----------------+---------+---------+-----------------------------------+
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar

try:
    from src.core.exceptions import PaladinSecurityError
except ImportError:
    from core.exceptions import PaladinSecurityError  # type: ignore[no-redef]


class CarrierHeaderError(PaladinSecurityError, ValueError):
    """Raised when carrier header framing or values are invalid."""
    pass


@dataclass(frozen=True)
class CarrierHeader:
    """Represents the decoded carrier metadata."""

    session_counter: int
    payload_length: int
    version: int = 0x01
    flags: int = 0x00
    reserved: int = 0x0000

    MAGIC: ClassVar[bytes] = b"PLDH"
    FORMAT: ClassVar[str] = ">4sBBHII"
    HEADER_SIZE: ClassVar[int] = 16  # 16 bytes = 128 bits

    # ECC Profile Flags (Bits 0-3 of flags byte)
    FLAG_ECC_MASK: ClassVar[int] = 0x0F
    ECC_PROFILE_RS_2T_32: ClassVar[int] = 0x00  # RS(255, 223), 2t=32, baseline default
    ECC_PROFILE_RS_2T_48: ClassVar[int] = 0x01  # RS(255, 207), 2t=48, enhanced resilience
    ECC_PROFILE_RS_2T_64: ClassVar[int] = 0x02  # RS(255, 191), 2t=64, maximum resilience

    ECC_PARITY_TO_PROFILE: ClassVar[dict[int, int]] = {
        32: 0x00,
        48: 0x01,
        64: 0x02,
    }

    ECC_PROFILE_TO_PARITY: ClassVar[dict[int, int]] = {
        0x00: 32,
        0x01: 48,
        0x02: 64,
    }

    @property
    def ecc_profile(self) -> int:
        """Returns the 4-bit ECC profile identifier from header flags."""
        return self.flags & self.FLAG_ECC_MASK

    @property
    def ecc_parity(self) -> int:
        """Resolves the Reed-Solomon parity symbol count (2t) from header flags."""
        prof = self.ecc_profile
        if prof not in self.ECC_PROFILE_TO_PARITY:
            raise CarrierHeaderError(f"Unsupported ECC profile in carrier header flags: {prof:#04x}")
        return self.ECC_PROFILE_TO_PARITY[prof]

    @classmethod
    def create(
        cls,
        session_counter: int,
        payload_length: int,
        ecc_parity: int = 32,
        flags: int = 0x00,
        reserved: int = 0x0000,
    ) -> CarrierHeader:
        """Constructs a CarrierHeader with encoded ECC profile in flags."""
        if ecc_parity not in cls.ECC_PARITY_TO_PROFILE:
            raise CarrierHeaderError(f"Unsupported ecc_parity {ecc_parity}. Must be 32, 48, or 64.")
        prof_bits = cls.ECC_PARITY_TO_PROFILE[ecc_parity]
        combined_flags = (flags & ~cls.FLAG_ECC_MASK) | prof_bits
        return cls(
            session_counter=session_counter,
            payload_length=payload_length,
            flags=combined_flags,
            reserved=reserved,
        )

    def pack(self) -> bytes:
        """Serializes the carrier header into canonical 16-byte binary format."""
        if not (0 <= self.session_counter <= 0xFFFFFFFF):
            raise CarrierHeaderError(
                f"Session counter out of uint32 range: {self.session_counter}"
            )
        if not (0 <= self.payload_length <= 0xFFFFFFFF):
            raise CarrierHeaderError(
                f"Payload length out of uint32 range: {self.payload_length}"
            )

        return struct.pack(
            self.FORMAT,
            self.MAGIC,
            self.version,
            self.flags,
            self.reserved,
            self.session_counter,
            self.payload_length,
        )

    @classmethod
    def unpack(cls, raw: bytes) -> CarrierHeader:
        """Deserializes and validates a 16-byte carrier header."""
        if len(raw) < cls.HEADER_SIZE:
            raise CarrierHeaderError(
                f"Carrier header truncated: expected at least {cls.HEADER_SIZE} bytes, got {len(raw)}"
            )

        header_bytes = raw[: cls.HEADER_SIZE]
        try:
            magic, version, flags, reserved, session_counter, payload_length = struct.unpack(
                cls.FORMAT, header_bytes
            )
        except struct.error as e:
            raise CarrierHeaderError(f"Malformed carrier header framing: {e}") from e

        if magic != cls.MAGIC:
            raise CarrierHeaderError(
                f"Invalid carrier header magic: expected {cls.MAGIC!r}, got {magic!r}. "
                "The carrier is invalid, corrupted, or stored in an unsupported format; "
                "please use the original lossless 8-bit RGB PNG carrier."
            )
        if version != 0x01:
            raise CarrierHeaderError(
                f"Unsupported carrier header version: {version:#02x}"
            )

        return cls(
            session_counter=session_counter,
            payload_length=payload_length,
            version=version,
            flags=flags,
            reserved=reserved,
        )
