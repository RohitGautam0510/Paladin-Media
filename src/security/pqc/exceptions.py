"""
Post-Quantum Cryptography & Authenticated Key Establishment Exception Hierarchy.

Defines fail-closed security exceptions for PQC operations in PaladinMedia.
"""

from src.core.exceptions import PaladinSecurityError, CryptoError, AuthenticationError


class PQCError(CryptoError):
    """Base exception for all Post-Quantum Cryptographic operations."""
    pass


class PQCProviderError(PQCError):
    """Raised when the PQC underlying algorithm provider encounters an internal fault."""
    pass


class PQCDisabledError(PQCError):
    """Raised when PQC operations are attempted while PALADIN_ENABLE_PQC is false."""
    pass


class PQCEnvelopeError(PQCError):
    """Raised when a PQC envelope header, format, or framing is corrupted or invalid."""
    pass


class PQCIdentityError(AuthenticationError, PQCError):
    """Raised when an unknown, unregistered, or mismatched identity is encountered."""
    pass


class PQCKeyExpiredError(PQCIdentityError):
    """Raised when an identity's public key certificate has expired."""
    pass


class PQCKeyRevokedError(PQCIdentityError):
    """Raised when an identity's public key has been revoked."""
    pass


class PQCSignatureVerificationError(AuthenticationError, PQCError):
    """Raised when ML-DSA digital signature verification fails."""
    pass


class PQCKeyConfirmationError(AuthenticationError, PQCError):
    """Raised when AKE transcript key confirmation tag validation fails."""
    pass


class PQCReplayError(AuthenticationError, PQCError):
    """Raised when session counter replay, non-monotonicity, or timestamp expiration is detected."""
    pass


class PQCDecapsulationError(PQCError):
    """Raised when ML-KEM decapsulation fails."""
    pass
