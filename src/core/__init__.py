"""Core module for PaladinMedia."""

from src.core.exceptions import (
    AuthenticationError,
    CryptoError,
    ECCDecodeError,
    ECCError,
    PaladinSecurityError,
    SerializationError,
    UnsupportedCarrierFormatError,
)
__all__ = [
    "AuthenticationError",
    "CryptoError",
    "ECCDecodeError",
    "ECCError",
    "PaladinSecurityError",
    "SecurityPipeline",
    "SerializationError",
    "UnsupportedCarrierFormatError",
]


def __getattr__(name: str):
    if name == "SecurityPipeline":
        from src.core.pipeline import SecurityPipeline
        return SecurityPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
