"""Secret payload data models and canonical wire format serialization.

Implements PDM-S1-005: Canonical binary framing, deterministic JSON metadata encoding,
and CRC-32 integrity verification for steganographic secret payloads.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

from src.core.exceptions import SerializationError

__all__ = [
    "MediaType",
    "SecretPayload",
    "SecretSerializer",
    "SerializationError",
]


class MediaType(IntEnum):
    """Enumeration of supported secret payload media types."""
    RAW_BYTES = 0x00
    TEXT = 0x01
    IMAGE = 0x02
    AUDIO = 0x03
    JSON = 0x04


@dataclass(frozen=True)
class SecretPayload:
    """Immutable representation of a secret payload with associated metadata and type.

    Attributes:
        data: Verbatim secret binary content.
        media_type: MediaType enum indicating the semantic format of the payload data.
        metadata: Optional dictionary containing arbitrary metadata (e.g., filename,
            dimensions, timestamp). Serialized to deterministic compact JSON.
    """
    data: bytes
    media_type: MediaType = MediaType.RAW_BYTES
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray)):
            raise TypeError(
                f"Payload data must be bytes or bytearray, got {type(self.data).__name__}"
            )
        if isinstance(self.data, bytearray):
            object.__setattr__(self, "data", bytes(self.data))

        if not isinstance(self.media_type, MediaType):
            try:
                object.__setattr__(self, "media_type", MediaType(self.media_type))
            except (ValueError, TypeError) as err:
                raise TypeError(f"Invalid media_type: {self.media_type}") from err

        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise TypeError(
                    f"Payload metadata must be a dictionary or None, got {type(self.metadata).__name__}"
                )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SecretPayload):
            return False
        # Normalize empty dict and None metadata for roundtrip equivalence
        self_meta = None if not self.metadata else self.metadata
        other_meta = None if not other.metadata else other.metadata
        return (
            self.data == other.data
            and self.media_type == other.media_type
            and self_meta == other_meta
        )

    def __hash__(self) -> int:
        meta_hashable: Optional[Tuple[Any, ...]] = None
        if self.metadata:
            try:
                meta_hashable = tuple(sorted(self.metadata.items()))
            except Exception:
                meta_hashable = (str(sorted(self.metadata.keys())),)
        return hash((self.data, self.media_type, meta_hashable))


class SecretSerializer:
    """Serializer and deserializer for the canonical PaladinMedia binary wire format.

    Wire Format Specification:
    +------------------+---------+---------+-----------------------------------+
    | Field Name       | Type    | Size    | Description                       |
    +------------------+---------+---------+-----------------------------------+
    | Magic Bytes      | bytes   | 4 bytes | Fixed ASCII: b"PLDS" (0x504C4453) |
    | Protocol Version | uint8   | 1 byte  | Format Version: 0x01 (MVP 0)      |
    | Media Type       | uint8   | 1 byte  | 0x00=RAW_BYTES .. 0x04=JSON       |
    | Flags            | uint16  | 2 bytes | Big-endian: 0x0000 (reserved)     |
    | Metadata Length  | uint16  | 2 bytes | Big-endian length M of JSON meta  |
    | Metadata JSON    | bytes   | M bytes | Compact UTF-8 JSON (sorted keys)  |
    | Payload Length   | uint32  | 4 bytes | Big-endian length L of Data       |
    | Payload Data     | bytes   | L bytes | Raw binary payload content        |
    | CRC-32 Checksum  | uint32  | 4 bytes | Big-endian CRC-32 over all bytes  |
    |                  |         |         | from Magic through Payload Data   |
    +------------------+---------+---------+-----------------------------------+
    """

    MAGIC: bytes = b"PLDS"
    VERSION: int = 0x01
    FLAGS: int = 0x0000
    MIN_HEADER_SIZE: int = 16  # 12B header + 4B CRC32 minimum check
    PREFIX_HEADER_SIZE: int = 10  # Magic (4) + Version (1) + MediaType (1) + Flags (2) + MetaLen (2)
    MIN_WIRE_SIZE: int = 18  # Prefix (10) + PayloadLen (4) + CRC32 (4) when M=0 and L=0

    @staticmethod
    def serialize(payload: SecretPayload) -> bytes:
        """Serializes a SecretPayload into canonical wire format with CRC-32.

        Args:
            payload: The SecretPayload to serialize.

        Returns:
            bytes: The canonical binary wire format byte stream.

        Raises:
            TypeError: If payload is not an instance of SecretPayload or data is invalid.
            SerializationError: If metadata cannot be serialized to JSON or exceeds 16-bit capacity.
        """
        if not isinstance(payload, SecretPayload):
            raise TypeError(f"Expected SecretPayload, got {type(payload).__name__}")

        # Encode metadata to deterministic compact UTF-8 JSON with sorted keys
        if payload.metadata:
            try:
                metadata_bytes = json.dumps(
                    payload.metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as err:
                raise SerializationError(f"Failed to serialize metadata to JSON: {err}") from err

            metadata_len = len(metadata_bytes)
            if metadata_len > 0xFFFF:
                raise SerializationError(
                    f"Metadata length exceeds 16-bit capacity: {metadata_len} > 65535"
                )
        else:
            metadata_bytes = b""
            metadata_len = 0

        # Validate payload data
        data_bytes = payload.data
        if not isinstance(data_bytes, (bytes, bytearray)):
            raise TypeError(f"Payload data must be bytes, got {type(data_bytes).__name__}")

        payload_len = len(data_bytes)
        if payload_len > 0xFFFFFFFF:
            raise SerializationError(
                f"Payload length exceeds 32-bit capacity: {payload_len} > 4294967295"
            )

        # Assemble fixed prefix header: Magic (4B) + Version (1B) + Media Type (1B) + Flags (2B) + MetaLen (2B)
        prefix_header = struct.pack(
            ">4sBBHH",
            SecretSerializer.MAGIC,
            SecretSerializer.VERSION,
            int(payload.media_type),
            SecretSerializer.FLAGS,
            metadata_len,
        )

        # Payload Length (4B uint32 big-endian)
        payload_len_bytes = struct.pack(">I", payload_len)

        # Assemble content over which CRC-32 is calculated (Magic through end of Payload Data)
        content = prefix_header + metadata_bytes + payload_len_bytes + bytes(data_bytes)

        # Compute IEEE 802.3 CRC-32 checksum
        checksum = zlib.crc32(content) & 0xFFFFFFFF
        checksum_bytes = struct.pack(">I", checksum)

        return content + checksum_bytes

    @staticmethod
    def deserialize(raw_bytes: bytes) -> SecretPayload:
        """Deserializes canonical wire format into a SecretPayload with strict integrity checks.

        Args:
            raw_bytes: Raw binary bytes from carrier or transport stream.

        Returns:
            SecretPayload: Restored secret payload object.

        Raises:
            SerializationError: If buffer is truncated, magic is invalid, version is unsupported,
                CRC-32 integrity fails, or metadata JSON is malformed.
        """
        if isinstance(raw_bytes, memoryview):
            raw = raw_bytes.tobytes()
        elif isinstance(raw_bytes, bytearray):
            raw = bytes(raw_bytes)
        elif isinstance(raw_bytes, bytes):
            raw = raw_bytes
        else:
            raise SerializationError(f"Expected bytes-like object, got {type(raw_bytes).__name__}")

        # 1. Validate minimum header size (16 bytes minimum: 12B header + 4B CRC32)
        if len(raw) < SecretSerializer.MIN_HEADER_SIZE:
            raise SerializationError(
                f"Buffer too short: minimum size is {SecretSerializer.MIN_HEADER_SIZE} bytes, got {len(raw)}"
            )

        # 2. Validate Magic equals b"PLDS"
        magic = raw[:4]
        if magic != SecretSerializer.MAGIC:
            raise SerializationError("Invalid magic bytes")

        # 3. Validate Version equals 0x01
        version = raw[4]
        if version != SecretSerializer.VERSION:
            raise SerializationError("Unsupported version")

        # 4. Verify CRC-32 integrity check over all preceding bytes
        expected_crc = struct.unpack(">I", raw[-4:])[0]
        calculated_crc = zlib.crc32(raw[:-4]) & 0xFFFFFFFF
        if calculated_crc != expected_crc:
            raise SerializationError("CRC32 integrity check failed")

        # 5. Validate minimum buffer for prefix header and framing
        if len(raw) < SecretSerializer.MIN_WIRE_SIZE:
            raise SerializationError(
                f"Buffer truncated: minimum framing requires at least {SecretSerializer.MIN_WIRE_SIZE} bytes, got {len(raw)}"
            )

        # Parse Media Type
        media_type_byte = raw[5]
        try:
            media_type = MediaType(media_type_byte)
        except ValueError:
            raise SerializationError(f"Invalid media type: {media_type_byte:#02x}")

        # Parse Flags (reserved 0x0000)
        _flags = struct.unpack(">H", raw[6:8])[0]

        # Parse Metadata Length
        metadata_len = struct.unpack(">H", raw[8:10])[0]

        # Verify buffer accommodates metadata JSON + 4B payload length + 4B CRC
        min_required_len = SecretSerializer.PREFIX_HEADER_SIZE + metadata_len + 4 + 4
        if len(raw) < min_required_len:
            raise SerializationError(
                f"Buffer truncated: metadata length {metadata_len} requires at least {min_required_len} bytes, got {len(raw)}"
            )

        # Parse Metadata JSON
        metadata_start = SecretSerializer.PREFIX_HEADER_SIZE
        metadata_end = metadata_start + metadata_len
        if metadata_len > 0:
            metadata_raw = raw[metadata_start:metadata_end]
            try:
                metadata_str = metadata_raw.decode("utf-8")
                metadata = json.loads(metadata_str)
                if not isinstance(metadata, dict):
                    raise SerializationError("Malformed metadata: root JSON value must be an object/dict")
            except (UnicodeDecodeError, json.JSONDecodeError) as err:
                raise SerializationError(f"Malformed metadata JSON: {err}") from err
        else:
            metadata = None

        # Parse Payload Length
        payload_len_offset = metadata_end
        payload_len = struct.unpack(">I", raw[payload_len_offset : payload_len_offset + 4])[0]

        # Verify total buffer length matches header + metadata + payload + 4
        expected_total_len = SecretSerializer.PREFIX_HEADER_SIZE + metadata_len + 4 + payload_len + 4
        if len(raw) != expected_total_len:
            raise SerializationError(
                f"Buffer length mismatch: expected {expected_total_len} bytes (metadata={metadata_len}, payload={payload_len}), got {len(raw)}"
            )

        # Parse Payload Data
        payload_start = payload_len_offset + 4
        payload_end = payload_start + payload_len
        payload_data = raw[payload_start:payload_end]

        return SecretPayload(
            data=payload_data,
            media_type=media_type,
            metadata=metadata,
        )
