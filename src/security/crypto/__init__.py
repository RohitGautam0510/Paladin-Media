"""Cryptographic providers module for PaladinMedia.

Provides AES-256-GCM authenticated encryption conforming to NIST SP 800-38D
with strict fail-closed security invariants.
"""

from src.security.crypto.provider import (
    AESGCMProvider,
    CryptoError,
    CryptoProvider,
    CryptoResult,
    InvalidKeyError,
    InvalidNonceError,
    NonceReuseError,
)
from src.core.exceptions import AuthenticationError

__all__ = [
    "CryptoProvider",
    "AESGCMProvider",
    "CryptoResult",
    "CryptoError",
    "AuthenticationError",
    "InvalidKeyError",
    "InvalidNonceError",
    "NonceReuseError",
]
