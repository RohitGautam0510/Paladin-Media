"""Steganographic Carrier Header Codec for PaladinMedia.

Implements the fixed carrier header per Architecture §8.3/§10.1 and Wire Protocol v2
with shortened Reed-Solomon RS(24, 16) error-correction coding:

Wire Protocol v1 Specification (16 bytes = 128 bits, Legacy Unprotected):
+-----------------+---------+---------+-----------------------------------+
| Field Name      | Type    | Size    | Description                       |
+-----------------+---------+---------+-----------------------------------+
| Magic           | bytes   | 4 bytes | Fixed ASCII: b"PLDH" (0x504C4448) |
| Protocol Version| uint8   | 1 byte  | 0x01 (Legacy Phase 2)             |
| Flags           | uint8   | 1 byte  | ECC profile in bits 0-3           |
| Reserved        | uint16  | 2 bytes | Big-endian 0x0000                 |
| Session Counter | uint32  | 4 bytes | Big-endian session/file context   |
| Payload Length  | uint32  | 4 bytes | Big-endian length of RS payload   |
+-----------------+---------+---------+-----------------------------------+

Wire Protocol v2 Specification (24 bytes = 192 bits, Shortened RS(24, 16)):
+-----------------+---------+---------+-----------------------------------+
| Field Name      | Type    | Size    | Description                       |
+-----------------+---------+---------+-----------------------------------+
| Magic           | bytes   | 4 bytes | Fixed ASCII: b"PLDH" (0x504C4448) |
| Protocol Version| uint8   | 1 byte  | 0x02 (Qualified Protocol v2)      |
| Flags           | uint8   | 1 byte  | ECC profile in bits 0-3           |
| Reserved        | uint16  | 2 bytes | Big-endian 0x0000                 |
| Session Counter | uint32  | 4 bytes | Big-endian session/file context   |
| Payload Length  | uint32  | 4 bytes | Big-endian length of RS payload   |
| RS Parity       | bytes   | 8 bytes | Shortened RS(24, 16) GF(2^8) (t=4)|
+-----------------+---------+---------+-----------------------------------+
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar

import reedsolo

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
    version: int = 0x02
    flags: int = 0x00
    reserved: int = 0x0000

    MAGIC: ClassVar[bytes] = b"PLDH"
    FORMAT: ClassVar[str] = ">4sBBHII"

    # Version Constants
    VERSION_1: ClassVar[int] = 0x01
    VERSION_2: ClassVar[int] = 0x02

    # Wire Sizing
    HEADER_SIZE_V1: ClassVar[int] = 16  # 16 bytes = 128 bits
    HEADER_SIZE_V2: ClassVar[int] = 24  # 24 bytes = 192 bits (16 data + 8 RS parity)
    HEADER_SIZE: ClassVar[int] = 24     # Default Wire Protocol v2 size

    HEADER_BITS_V1: ClassVar[int] = 128
    HEADER_BITS_V2: ClassVar[int] = 192
    HEADER_BITS: ClassVar[int] = 192

    RS_DATA_BYTES: ClassVar[int] = 16
    RS_PARITY_BYTES: ClassVar[int] = 8  # 8 parity symbols -> t=4 correctable bytes

    # ECC Profile Flags (Bits 0-3 of flags byte)
    FLAG_ECC_MASK: ClassVar[int] = 0x0F
    ECC_PROFILE_RS_2T_32: ClassVar[int] = 0x00  # RS(255, 223), 2t=32, baseline default
    ECC_PROFILE_RS_2T_48: ClassVar[int] = 0x01  # RS(255, 207), 2t=48, qualified deployment
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

    @property
    def header_size(self) -> int:
        """Returns wire byte size for this header's version."""
        return self.HEADER_SIZE_V2 if self.version == self.VERSION_2 else self.HEADER_SIZE_V1

    @property
    def header_bits(self) -> int:
        """Returns wire bit count for this header's version."""
        return self.header_size * 8

    @classmethod
    def create(
        cls,
        session_counter: int,
        payload_length: int,
        ecc_parity: int = 48,
        flags: int = 0x00,
        reserved: int = 0x0000,
        version: int = 0x02,
    ) -> CarrierHeader:
        """Constructs a CarrierHeader with encoded ECC profile in flags."""
        if ecc_parity not in cls.ECC_PARITY_TO_PROFILE:
            raise CarrierHeaderError(f"Unsupported ecc_parity {ecc_parity}. Must be 32, 48, or 64.")
        if version not in (cls.VERSION_1, cls.VERSION_2):
            raise CarrierHeaderError(f"Unsupported carrier header version: {version:#02x}. Must be 0x01 or 0x02.")
        prof_bits = cls.ECC_PARITY_TO_PROFILE[ecc_parity]
        combined_flags = (flags & ~cls.FLAG_ECC_MASK) | prof_bits
        return cls(
            session_counter=session_counter,
            payload_length=payload_length,
            version=version,
            flags=combined_flags,
            reserved=reserved,
        )

    def pack(self) -> bytes:
        """Serializes the carrier header into canonical binary format.

        - Wire Protocol v1: 16 bytes raw unprotected header.
        - Wire Protocol v2: 24 bytes systematic shortened RS(24, 16) codeword.
        """
        if not (0 <= self.session_counter <= 0xFFFFFFFF):
            raise CarrierHeaderError(
                f"Session counter out of uint32 range: {self.session_counter}"
            )
        if not (0 <= self.payload_length <= 0xFFFFFFFF):
            raise CarrierHeaderError(
                f"Payload length out of uint32 range: {self.payload_length}"
            )
        if self.version not in (self.VERSION_1, self.VERSION_2):
            raise CarrierHeaderError(
                f"Unsupported carrier header version: {self.version:#02x}. Must be 0x01 or 0x02."
            )

        raw_data = struct.pack(
            self.FORMAT,
            self.MAGIC,
            self.version,
            self.flags,
            self.reserved,
            self.session_counter,
            self.payload_length,
        )

        if self.version == self.VERSION_1:
            return raw_data

        # Wire Protocol v2: Systematic shortened RS(24, 16) over GF(2^8)
        rsc = reedsolo.RSCodec(self.RS_PARITY_BYTES)
        codeword = bytes(rsc.encode(raw_data))
        if len(codeword) != self.HEADER_SIZE_V2:
            raise CarrierHeaderError(
                f"RS(24, 16) encoding failed: expected {self.HEADER_SIZE_V2} bytes, got {len(codeword)}"
            )
        return codeword

    @classmethod
    def unpack(cls, raw: bytes) -> CarrierHeader:
        """Deserializes and validates a carrier header with robust v2/v1 auto-detection.

        Safety & Validation Invariants:
        1. Multi-Stage Probe: Attempts v2 RS(24, 16) decode if at least 24 bytes are provided.
        2. Corrects up to 4 erroneous bytes in v2 header across data and parity.
        3. Backward Compatibility: Seamlessly accepts legacy v1 16-byte headers.
        4. Rejects invalid magic, version, ECC profiles, reserved bits, and uncorrectable noise fail-closed.
        """
        if len(raw) < cls.HEADER_SIZE_V1:
            raise CarrierHeaderError(
                f"Carrier header truncated: expected at least {cls.HEADER_SIZE_V1} bytes, got {len(raw)}"
            )

        # -----------------------------------------------------------------
        # Stage 1: Probe Wire Protocol v2 (24 bytes systematic RS(24, 16))
        # -----------------------------------------------------------------
        if len(raw) >= cls.HEADER_SIZE_V2:
            rsc = reedsolo.RSCodec(cls.RS_PARITY_BYTES)
            try:
                dec = rsc.decode(raw[:cls.HEADER_SIZE_V2])
                dec_data = bytes(dec[0])
                magic, version, flags, reserved, session_counter, payload_length = struct.unpack(
                    cls.FORMAT, dec_data
                )
                if magic == cls.MAGIC and version == cls.VERSION_2:
                    if reserved != 0:
                        raise CarrierHeaderError(f"Non-zero reserved field in carrier header: {reserved:#06x}")
                    if flags & ~cls.FLAG_ECC_MASK != 0:
                        raise CarrierHeaderError(f"Invalid flag bits in carrier header: {flags:#04x}")
                    prof = flags & cls.FLAG_ECC_MASK
                    if prof not in cls.ECC_PROFILE_TO_PARITY:
                        raise CarrierHeaderError(f"Unsupported ECC profile in carrier header flags: {prof:#04x}")
                    return cls(
                        session_counter=session_counter,
                        payload_length=payload_length,
                        version=cls.VERSION_2,
                        flags=flags,
                        reserved=reserved,
                    )
            except reedsolo.ReedSolomonError:
                # If raw bytes clearly declared v2 framing (magic and version 0x02) but errors
                # exceed RS(24, 16) capacity (t=4 bytes), fail closed immediately.
                if len(raw) >= 5 and raw[:4] == cls.MAGIC and raw[4] == cls.VERSION_2:
                    raise CarrierHeaderError(
                        "Uncorrectable Wire Protocol v2 carrier header: errors exceed RS(24, 16) correction capability (t=4 bytes)"
                    )

        # -----------------------------------------------------------------
        # Stage 2: Probe Legacy Wire Protocol v1 (16 bytes unprotected)
        # -----------------------------------------------------------------
        header_bytes_v1 = raw[:cls.HEADER_SIZE_V1]
        try:
            magic, version, flags, reserved, session_counter, payload_length = struct.unpack(
                cls.FORMAT, header_bytes_v1
            )
        except struct.error as e:
            raise CarrierHeaderError(f"Malformed carrier header framing: {e}") from e

        if magic == cls.MAGIC and version == cls.VERSION_1:
            if reserved != 0:
                raise CarrierHeaderError(f"Non-zero reserved field in carrier header: {reserved:#06x}")
            if flags & ~cls.FLAG_ECC_MASK != 0:
                raise CarrierHeaderError(f"Invalid flag bits in carrier header: {flags:#04x}")
            prof = flags & cls.FLAG_ECC_MASK
            if prof not in cls.ECC_PROFILE_TO_PARITY:
                raise CarrierHeaderError(f"Unsupported ECC profile in carrier header flags: {prof:#04x}")
            return cls(
                session_counter=session_counter,
                payload_length=payload_length,
                version=cls.VERSION_1,
                flags=flags,
                reserved=reserved,
            )

        # -----------------------------------------------------------------
        # Stage 3: Fail-Closed Terminal Rejection
        # -----------------------------------------------------------------
        if magic != cls.MAGIC:
            raise CarrierHeaderError(
                f"Invalid carrier header magic: expected {cls.MAGIC!r}, got {magic!r}. "
                "The carrier is invalid, corrupted, or stored in an unsupported format; "
                "please use the original lossless 8-bit RGB PNG carrier."
            )
        raise CarrierHeaderError(
            f"Unsupported carrier header version: {version:#02x}. Only Wire Protocol v1 (0x01) and v2 (0x02) are supported."
        )
