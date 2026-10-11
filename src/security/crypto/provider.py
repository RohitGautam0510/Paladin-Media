"""Cryptographic provider interfaces and AES-256-GCM implementation for PaladinMedia.

This module implements:
1. CryptoProvider: Abstract Base Class defining the encryption/decryption contract.
2. AESGCMProvider: Concrete AEAD cipher provider using AES-256 in Galois/Counter Mode (GCM),
   adhering to NIST SP 800-38D, with fail-closed security guarantees and 0 plaintext leakage.
3. CryptoResult: Structured representation of packed/unpacked cryptographic artifacts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.core.exceptions import AuthenticationError, CryptoError


class InvalidKeyError(CryptoError):
    """Raised when an invalid encryption or decryption key is provided."""
    pass


class InvalidNonceError(CryptoError):
    """Raised when an invalid nonce is provided."""
    pass


class NonceReuseError(CryptoError):
    """Raised when nonce reuse is detected under the same key."""
    pass


@dataclass(frozen=True)
class CryptoResult:
    """Structured representation of AES-256-GCM encrypted artifacts.

    Wire Packaging Format:
        [12-byte Nonce] + [Ciphertext] + [16-byte GHASH Tag]
    """
    ciphertext: bytes
    nonce: bytes
    tag: bytes
    aad: Optional[bytes] = None

    def pack(self) -> bytes:
        """Returns contiguous byte sequence: nonce (12B) || ciphertext || tag (16B)."""
        return self.nonce + self.ciphertext + self.tag

    @classmethod
    def unpack(cls, packed: bytes, aad: Optional[bytes] = None) -> "CryptoResult":
        """Parses contiguous byte sequence into CryptoResult.

        Args:
            packed: Contiguous byte sequence of at least 28 bytes (12B nonce + 16B tag).
            aad: Optional associated authenticated data.

        Returns:
            CryptoResult instance.

        Raises:
            AuthenticationError: If packed payload is shorter than 28 bytes.
        """
        if len(packed) < 28:
            raise AuthenticationError("Protected payload too short for AES-GCM (minimum 28 bytes)")
        nonce = packed[:12]
        tag = packed[-16:]
        ciphertext = packed[12:-16]
        return cls(ciphertext=ciphertext, nonce=nonce, tag=tag, aad=aad)


class CryptoProvider(ABC):
    """Abstract Base Class for cryptographic providers.

    All implementations must enforce fail-closed security invariants:
    if integrity or authentication fails during decryption, the implementation
    must terminate immediately with AuthenticationError and return/leak zero plaintext.
    """

    @abstractmethod
    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Encrypts plaintext using key and optional Associated Authenticated Data (AAD).

        Args:
            plaintext: Raw bytes to encrypt.
            key: Secret cryptographic key.
            aad: Optional Associated Authenticated Data to bind into authentication tag.

        Returns:
            Packaged protected payload bytes.
        """
        ...

    @abstractmethod
    def decrypt(
        self,
        protected_payload: bytes,
        key: bytes,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Decrypts and authenticates protected payload using key and optional AAD.

        Args:
            protected_payload: Packaged payload bytes containing nonce, ciphertext, and tag.
            key: Secret cryptographic key.
            aad: Optional Associated Authenticated Data.

        Returns:
            Decrypted and verified plaintext bytes.

        Raises:
            AuthenticationError: If authentication tag verification fails, payload is tampered,
                                 corrupted, or truncated (fail-closed, 0 plaintext).
            CryptoError: If key is invalid.
        """
        ...


class AESGCMProvider(CryptoProvider):
    """AES-256-GCM Authenticated Encryption with Associated Data (AEAD) provider.

    Adheres to:
    - NIST SP 800-38D (Recommendation for Block Cipher Modes of Operation: Galois/Counter Mode).
    - PaladinMedia Architectural Invariants:
      * INV-03 (Fail-Closed Security Property): 100% terminal rejection on tampering, 0 plaintext leaked.
      * INV-04 (Nonce Uniqueness Guarantee): 96-bit (12-byte) cryptographically secure random nonces.
      * INV-05 (AAD Binding): GHASH binding of associated authenticated data.

    Wire format:
      [12-byte Nonce] + [Ciphertext] + [16-byte GHASH Tag]
    """

    KEY_SIZE: int = 32      # 256 bits
    NONCE_SIZE: int = 12    # 96 bits
    TAG_SIZE: int = 16      # 128 bits
    MIN_PAYLOAD_SIZE: int = NONCE_SIZE + TAG_SIZE  # 28 bytes

    def __init__(self, enforce_nonce_uniqueness: bool = False):
        """Initializes the AESGCMProvider.

        Args:
            enforce_nonce_uniqueness: If True, tracks used (key, nonce) pairs in-memory
                                      and raises NonceReuseError on any reuse attempt.
                                      Default is False (stateless CSPRNG nonce generation).
        """
        self.enforce_nonce_uniqueness = enforce_nonce_uniqueness
        self._used_nonces: set[tuple[bytes, bytes]] = set()

    def reset_nonce_history(self) -> None:
        """Clears the recorded nonce history cache."""
        self._used_nonces.clear()

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        aad: Optional[bytes] = None,
        nonce: Optional[bytes] = None,
    ) -> bytes:
        """Encrypts plaintext using AES-256-GCM with a 256-bit key and 96-bit nonce.

        Args:
            plaintext: Bytes to encrypt.
            key: Exactly 32 bytes (256-bit AES key).
            aad: Optional associated authenticated data to bind in GHASH.
            nonce: Optional 12-byte nonce (used for vector verification).
                   If None, generates 12 cryptographically secure random bytes via os.urandom.

        Returns:
            Packaged bytes: [12B Nonce] + [Ciphertext] + [16B Tag].

        Raises:
            CryptoError: If key is not 32 bytes, or nonce is not 12 bytes.
            NonceReuseError: If enforce_nonce_uniqueness is True and (key, nonce) was already used.
            TypeError: If plaintext or key is not bytes-like.
        """
        # Validate key length
        if not isinstance(key, (bytes, bytearray)) or len(key) != self.KEY_SIZE:
            raise CryptoError("AES-256 key must be exactly 32 bytes")

        # Validate plaintext
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError("Plaintext must be bytes or bytearray")

        # Validate AAD if provided
        if aad is not None and not isinstance(aad, (bytes, bytearray)):
            raise TypeError("AAD must be bytes or bytearray if provided")

        # Handle nonce
        if nonce is None:
            nonce_bytes = os.urandom(self.NONCE_SIZE)
        else:
            if not isinstance(nonce, (bytes, bytearray)) or len(nonce) != self.NONCE_SIZE:
                raise CryptoError("Nonce must be exactly 12 bytes")
            nonce_bytes = bytes(nonce)

        # Enforce nonce uniqueness if configured
        if self.enforce_nonce_uniqueness:
            key_nonce_pair = (bytes(key), nonce_bytes)
            if key_nonce_pair in self._used_nonces:
                raise NonceReuseError(
                    f"Nonce reuse detected for key: nonce {nonce_bytes.hex()} has already been used"
                )
            self._used_nonces.add(key_nonce_pair)

        # Execute AES-256-GCM encryption
        # AESGCM.encrypt returns ciphertext + 16-byte tag
        aesgcm = AESGCM(bytes(key))
        ciphertext_and_tag = aesgcm.encrypt(
            nonce_bytes,
            bytes(plaintext),
            bytes(aad) if aad is not None else None,
        )

        # Return packaged wire format: [12B Nonce] + [Ciphertext] + [16B Tag]
        return nonce_bytes + ciphertext_and_tag

    def decrypt(
        self,
        protected_payload: Optional[bytes] = None,
        key: bytes = b"",
        aad: Optional[bytes] = None,
        *,
        ciphertext: Optional[bytes] = None,
        nonce: Optional[bytes] = None,
        tag: Optional[bytes] = None,
    ) -> bytes:
        """Decrypts and authenticates protected payload using AES-256-GCM.

        FAIL-CLOSED SECURITY INVARIANT:
        If tag verification fails or any decryption exception occurs, this method
        terminates immediately, raises AuthenticationError, and ensures ZERO plaintext
        is returned, retained, or leaked.

        Args:
            protected_payload: Packaged bytes: [12B Nonce] + [Ciphertext] + [16B Tag].
                               Must be at least 28 bytes.
            key: Exactly 32 bytes (256-bit AES key).
            aad: Optional associated authenticated data. Must match encryption AAD.
            ciphertext: Optional individual ciphertext bytes (if not passing protected_payload).
            nonce: Optional individual 12B nonce bytes.
            tag: Optional individual 16B tag bytes.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            AuthenticationError: On tag verification failure, tampering, wrong key, altered AAD,
                                 or truncated payload (fail-closed).
            CryptoError: If key is not exactly 32 bytes.
        """
        # Validate key length
        if not isinstance(key, (bytes, bytearray)) or len(key) != self.KEY_SIZE:
            raise CryptoError("AES-256 key must be exactly 32 bytes")

        # Support keyword arguments for individual components if protected_payload is not provided
        if protected_payload is None:
            if ciphertext is not None and nonce is not None and tag is not None:
                protected_payload = bytes(nonce) + bytes(ciphertext) + bytes(tag)
            else:
                raise AuthenticationError("Protected payload must be provided")

        if not isinstance(protected_payload, (bytes, bytearray)):
            raise AuthenticationError("Protected payload must be bytes or bytearray")

        # Validate minimum payload length (12B nonce + 16B tag)
        if len(protected_payload) < self.MIN_PAYLOAD_SIZE:
            raise AuthenticationError("Protected payload too short for AES-GCM")

        # Extract 12-byte nonce and remaining ciphertext + tag
        nonce_bytes = bytes(protected_payload[:self.NONCE_SIZE])
        ciphertext_and_tag = bytes(protected_payload[self.NONCE_SIZE:])
        aad_bytes = bytes(aad) if aad is not None else None

        # FAIL-CLOSED DECRYPTION
        try:
            aesgcm = AESGCM(bytes(key))
            plaintext = aesgcm.decrypt(nonce_bytes, ciphertext_and_tag, aad_bytes)
            return plaintext
        except Exception as exc:
            # Terminate immediately on any error: invalid tag, corrupted ciphertext, altered AAD
            # Guarantee zero plaintext is returned, retained, or leaked
            raise AuthenticationError(
                "AES-GCM authentication failure: payload tampered or invalid key/aad"
            ) from exc

    def encrypt_structured(
        self,
        plaintext: bytes,
        key: bytes,
        aad: Optional[bytes] = None,
        nonce: Optional[bytes] = None,
    ) -> CryptoResult:
        """Encrypts plaintext and returns structured CryptoResult."""
        packed = self.encrypt(plaintext, key, aad=aad, nonce=nonce)
        return CryptoResult.unpack(packed, aad=aad)

    def decrypt_structured(
        self,
        result: CryptoResult,
        key: bytes,
    ) -> bytes:
        """Decrypts a structured CryptoResult."""
        return self.decrypt(result.pack(), key, aad=result.aad)
