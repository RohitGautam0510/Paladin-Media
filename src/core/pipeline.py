"""Security Pipeline for PaladinMedia MVP 0 Sprint 1.

Integrates:
1. SecretSerializer: Serializes SecretPayload into canonical wire format with CRC-32 integrity.
2. CryptoProvider (AES-256-GCM): Authenticated symmetric encryption with random 96-bit nonce
   and 128-bit GHASH authentication tag, strictly fail-closed.
3. ECCEncoder (ReedSolomonEncoder): Error correction coding operating over GF(2^8)
   with systematic block chunking and length framing.

Enforces Architectural Invariants:
- INV-01: Strict unidirectional sequence: Serialize -> Encrypt -> ECC Encode.
- INV-02: ECC operates exclusively on protected representation (ciphertext), never on plaintext.
- INV-03: Fail-closed security: Any tampering in ciphertext, tag, or AAD causes terminal
  rejection and zero plaintext leakage.
"""

from __future__ import annotations

from typing import Optional
# pyrefly: ignore [missing-import]
from src.core.serialization import SecretPayload, SecretSerializer
# pyrefly: ignore [missing-import]
from src.security.crypto.provider import AESGCMProvider, CryptoProvider
# pyrefly: ignore [missing-import]
from src.security.ecc.encoder import ECCEncoder, ReedSolomonEncoder


class SecurityPipeline:
    """End-to-end security pipeline for protect and unprotect operations."""

    def __init__(
        self,
        crypto_provider: Optional[CryptoProvider] = None,
        ecc_encoder: Optional[ECCEncoder] = None,
    ) -> None:
        """Initializes SecurityPipeline with cryptographic and ECC providers.

        Args:
            crypto_provider: Cryptographic provider instance (defaults to AESGCMProvider()).
            ecc_encoder: Error correction encoder instance (defaults to ReedSolomonEncoder()).
        """
        self.crypto_provider: CryptoProvider = (
            crypto_provider if crypto_provider is not None else AESGCMProvider()
        )
        self.ecc_encoder: ECCEncoder = (
            ecc_encoder if ecc_encoder is not None else ReedSolomonEncoder()
        )

    def protect(
        self,
        payload: SecretPayload,
        key: bytes,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Protects a SecretPayload through Serialization -> Encryption -> ECC Encoding.

        Sequence:
            1. Canonical serialization with CRC-32 integrity check.
            2. Authenticated encryption (AES-256-GCM) with 96-bit nonce and 128-bit tag.
            3. Reed-Solomon error correction encoding over GF(2^8) with systematic chunking.

        Args:
            payload: SecretPayload instance containing secret data, media type, and metadata.
            key: Exactly 32 bytes (256-bit symmetric key).
            aad: Optional Associated Authenticated Data for cryptographic binding.

        Returns:
            Protected bytes ready for steganographic carrier embedding.
        """
        # Step 1: Serialize
        serialized = SecretSerializer.serialize(payload)

        # Step 2: Encrypt
        encrypted = self.crypto_provider.encrypt(serialized, key, aad=aad)

        # Step 3: ECC Encode
        protected = self.ecc_encoder.encode(encrypted)

        return protected

    def unprotect(
        self,
        protected_bytes: bytes,
        key: bytes,
        aad: Optional[bytes] = None,
        ecc_encoder: Optional[ECCEncoder] = None,
    ) -> SecretPayload:
        """Reverses protection through ECC Decoding -> Decryption -> Deserialization.

        Sequence:
            1. Reed-Solomon error correction decoding, repairing up to t symbol errors per block.
            2. Authenticated decryption (AES-256-GCM), verifying GHASH tag (fail-closed).
            3. Canonical deserialization, verifying magic bytes, version, and CRC-32 integrity.

        Args:
            protected_bytes: Protected byte stream extracted from carrier or channel.
            key: Exactly 32 bytes (256-bit symmetric key).
            aad: Optional Associated Authenticated Data matching encryption context.
            ecc_encoder: Optional ECCEncoder instance to override default decoder.

        Returns:
            Recovered SecretPayload with verbatim original data, media type, and metadata.

        Raises:
            ECCDecodeError: If uncorrectable errors exceed correction capability or framing is invalid.
            AuthenticationError: If ciphertext, tag, nonce, key, or AAD is tampered or invalid.
            SerializationError: If framing, version, or CRC-32 checksum is invalid.
        """
        # Step 1: ECC Decode using matching or specified decoder
        decoder = ecc_encoder if ecc_encoder is not None else self.ecc_encoder
        ecc_decoded = decoder.decode(protected_bytes)

        # Step 2: Decrypt
        decrypted = self.crypto_provider.decrypt(ecc_decoded, key, aad=aad)

        # Step 3: Deserialize
        payload = SecretSerializer.deserialize(decrypted)

        return payload
