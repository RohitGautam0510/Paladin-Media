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
                f"Invalid carrier header magic: expected {cls.MAGIC!r}, got {magic!r}"
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
