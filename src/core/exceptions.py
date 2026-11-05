"""Core exception hierarchy for PaladinMedia.

This module defines the standard exception classes used across the
PaladinMedia cryptographic, serialization, and error correction pipelines.
"""

__all__ = [
    "PaladinSecurityError",
    "SerializationError",
    "CryptoError",
    "AuthenticationError",
    "ECCError",
    "ECCDecodeError",
    "UnsupportedCarrierFormatError",
]


class PaladinSecurityError(Exception):
    """Base exception for all PaladinMedia security and pipeline errors."""
    pass


class SerializationError(PaladinSecurityError):
    """Raised on serialization framing, magic mismatch, CRC mismatch, or deserialization failure."""
    pass


class CryptoError(PaladinSecurityError):
    """Base exception for cryptographic operations."""
    pass


class AuthenticationError(CryptoError):
    """Raised specifically on AES-GCM authentication failure, tag mismatch, or tampering.

    In accordance with fail-closed security invariants, when this error is raised,
    no unauthenticated plaintext should ever be returned or leaked.
    """
    pass


class ECCError(PaladinSecurityError):
    """Base exception for error correction operations."""
    pass


class ECCDecodeError(ECCError):
    """Raised when Reed-Solomon decoding fails due to uncorrectable errors exceeding correction capability t."""
    pass


class UnsupportedCarrierFormatError(PaladinSecurityError, ValueError):
    """Raised when an unsupported or unqualified carrier format (such as lossy JPEG) is provided."""
    pass
