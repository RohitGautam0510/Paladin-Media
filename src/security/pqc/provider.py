"""
Post-Quantum Cryptography Primitive Providers for PaladinMedia.

Implements clean provider abstractions for NIST FIPS 203 (ML-KEM-768) and NIST FIPS 204 (ML-DSA-65).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Tuple

from cryptography.hazmat.primitives.asymmetric import mlkem, mldsa
from src.security.pqc.exceptions import (
    PQCProviderError,
    PQCDecapsulationError,
    PQCSignatureVerificationError,
)


class PQCProvider(ABC):
    """Abstract provider interface for Post-Quantum Key Encapsulation and Digital Signatures."""

    # Exact NIST parameter sizes
    MLKEM768_PUBLIC_KEY_BYTES: int = 1184
    MLKEM768_CIPHERTEXT_BYTES: int = 1088
    MLKEM768_SHARED_SECRET_BYTES: int = 32

    MLDSA65_PUBLIC_KEY_BYTES: int = 1952
    MLDSA65_SIGNATURE_BYTES: int = 3309

    @abstractmethod
    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        """Generates an ML-KEM-768 keypair. Returns (public_key_bytes, private_key_bytes)."""
        ...

    @abstractmethod
    def encapsulate(self, recipient_public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulates a shared secret using recipient's ML-KEM-768 public key.

        Returns:
            Tuple of (shared_secret_32b, kem_ciphertext_1088b).
        """
        ...

    @abstractmethod
    def decapsulate(self, recipient_private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulates a shared secret from an ML-KEM-768 ciphertext.

        Returns:
            32-byte shared secret.
        """
        ...

    @abstractmethod
    def generate_dsa_keypair(self) -> Tuple[bytes, bytes]:
        """Generates an ML-DSA-65 keypair. Returns (public_key_bytes, private_key_bytes)."""
        ...

    @abstractmethod
    def sign(self, sender_private_key: bytes, message: bytes) -> bytes:
        """Signs message using sender's ML-DSA-65 private key. Returns 3,309-byte signature."""
        ...

    @abstractmethod
    def verify(self, sender_public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verifies ML-DSA-65 signature over message against sender's public key."""
        ...


class CryptographyPQCProvider(PQCProvider):
    """Production PQC provider backed by native OpenSSL/Rust implementations in cryptography."""

    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        try:
            priv = mlkem.MLKEM768PrivateKey.generate()
            pub = priv.public_key()
            return pub.public_bytes_raw(), priv.private_bytes_raw()
        except Exception as e:
            raise PQCProviderError(f"ML-KEM-768 keypair generation failed: {e}") from e

    def encapsulate(self, recipient_public_key: bytes) -> Tuple[bytes, bytes]:
        if len(recipient_public_key) != self.MLKEM768_PUBLIC_KEY_BYTES:
            raise PQCProviderError(
                f"Invalid ML-KEM-768 public key size: expected {self.MLKEM768_PUBLIC_KEY_BYTES} B, got {len(recipient_public_key)} B"
            )
        try:
            pub = mlkem.MLKEM768PublicKey.from_public_bytes(recipient_public_key)
            shared_secret, ciphertext = pub.encapsulate()
            assert len(shared_secret) == self.MLKEM768_SHARED_SECRET_BYTES
            assert len(ciphertext) == self.MLKEM768_CIPHERTEXT_BYTES
            return shared_secret, ciphertext
        except Exception as e:
            raise PQCProviderError(f"ML-KEM-768 encapsulation failed: {e}") from e

    def decapsulate(self, recipient_private_key: bytes, ciphertext: bytes) -> bytes:
        if len(ciphertext) != self.MLKEM768_CIPHERTEXT_BYTES:
            raise PQCDecapsulationError(
                f"Invalid ML-KEM-768 ciphertext size: expected {self.MLKEM768_CIPHERTEXT_BYTES} B, got {len(ciphertext)} B"
            )
        try:
            priv = mlkem.MLKEM768PrivateKey.from_seed_bytes(recipient_private_key)
            shared_secret = priv.decapsulate(ciphertext)
            if len(shared_secret) != self.MLKEM768_SHARED_SECRET_BYTES:
                raise PQCDecapsulationError("Decapsulated shared secret size mismatch")
            return shared_secret
        except PQCDecapsulationError:
            raise
        except Exception as e:
            raise PQCDecapsulationError(f"ML-KEM-768 decapsulation failed: {e}") from e

    def generate_dsa_keypair(self) -> Tuple[bytes, bytes]:
        try:
            priv = mldsa.MLDSA65PrivateKey.generate()
            pub = priv.public_key()
            return pub.public_bytes_raw(), priv.private_bytes_raw()
        except Exception as e:
            raise PQCProviderError(f"ML-DSA-65 keypair generation failed: {e}") from e

    def sign(self, sender_private_key: bytes, message: bytes) -> bytes:
        try:
            priv = mldsa.MLDSA65PrivateKey.from_seed_bytes(sender_private_key)
            sig = priv.sign(message)
            if len(sig) != self.MLDSA65_SIGNATURE_BYTES:
                raise PQCProviderError(f"Unexpected ML-DSA-65 signature size: {len(sig)} B")
            return sig
        except Exception as e:
            raise PQCProviderError(f"ML-DSA-65 signing failed: {e}") from e

    def verify(self, sender_public_key: bytes, message: bytes, signature: bytes) -> bool:
        if len(sender_public_key) != self.MLDSA65_PUBLIC_KEY_BYTES:
            raise PQCSignatureVerificationError(
                f"Invalid ML-DSA-65 public key size: expected {self.MLDSA65_PUBLIC_KEY_BYTES} B, got {len(sender_public_key)} B"
            )
        if len(signature) != self.MLDSA65_SIGNATURE_BYTES:
            raise PQCSignatureVerificationError(
                f"Invalid ML-DSA-65 signature size: expected {self.MLDSA65_SIGNATURE_BYTES} B, got {len(signature)} B"
            )
        try:
            pub = mldsa.MLDSA65PublicKey.from_public_bytes(sender_public_key)
            pub.verify(signature, message)
            return True
        except Exception as e:
            raise PQCSignatureVerificationError(f"ML-DSA-65 signature verification failed: {e}") from e


# Default system provider instance
default_pqc_provider = CryptographyPQCProvider()
