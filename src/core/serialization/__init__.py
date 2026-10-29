"""Payload serialization module for PaladinMedia.

Provides canonical binary serialization, deserialization, and data structures
for steganographic secret payloads with CRC-32 integrity validation.
"""

from src.core.exceptions import SerializationError
from src.core.serialization.serializer import (
    MediaType,
    SecretPayload,
    SecretSerializer,
)

__all__ = [
    "MediaType",
    "SecretPayload",
    "SecretSerializer",
    "SerializationError",
]
