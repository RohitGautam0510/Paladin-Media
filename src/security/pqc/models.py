"""
Data models and binary structures for Post-Quantum Cryptography in PaladinMedia.

Implements PQCIdentity, PQCHeaderB1 (104-byte uncompressed canonical header),
PQCHeaderA (16-byte detached-signature header), and PQCEnvelope.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.security.pqc.exceptions import PQCEnvelopeError, PQCIdentityError, PQCKeyExpiredError, PQCKeyRevokedError


CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256 = 0x0001

MAGIC_PQC_AKE = b"PQCA"
MAGIC_PQC_KEM = b"PQCK"
MAGIC_PQC_SIDECAR = b"PQCS"

VERSION_PQC_V1 = 0x01

# Flag bits
FLAG_DETACHED_SIGNATURE = 0x01
FLAG_EPHEMERAL_KEM = 0x02


@dataclass(frozen=True)
class PQCSidecar:
    """
    Authenticated Sidecar for Layout A Detached-Signature Delivery.
    
    Carries complete transcript association fields and detached ML-DSA-65 signature
    required to reconstruct the pre-confirmation transcript independently.
    
    Binary Layout:
      0x00 - 0x03 (4B): magic b"PQCS"
      0x04        (1B): version (0x01)
      0x05 - 0x06 (2B): ciphersuite_id (0x0001)
      0x07 - 0x0E (8B): session_counter uint64 BE
      0x0F - 0x16 (8B): session_timestamp_ms uint64 BE
      0x17 - 0x36 (32B): sender_dsa_fingerprint
      0x37 - 0x56 (32B): recipient_kem_fingerprint
      0x57 - 0x66 (16B): key_confirmation_tag
      0x67 - 0x86 (32B): kem_ciphertext_sha256 (carrier binding hash)
      0x87 - 0xD73 (3309B): signature
    Total binary size: 3,444 bytes.
    """

    STRUCT_HEADER_FORMAT = ">4sBHQQ32s32s16s32s"
    HEADER_SIZE = 135
    SIGNATURE_SIZE = 3309
    TOTAL_SIZE = HEADER_SIZE + SIGNATURE_SIZE  # 3,444 bytes

    session_counter: int
    session_timestamp_ms: int
    sender_dsa_fingerprint: bytes
    recipient_kem_fingerprint: bytes
    key_confirmation_tag: bytes
    kem_ciphertext_sha256: bytes
    signature: bytes
    version: int = VERSION_PQC_V1
    ciphersuite_id: int = CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256

    def __post_init__(self) -> None:
        if len(self.sender_dsa_fingerprint) != 32:
            raise PQCEnvelopeError(f"Invalid sender DSA fingerprint length: {len(self.sender_dsa_fingerprint)} != 32")
        if len(self.recipient_kem_fingerprint) != 32:
            raise PQCEnvelopeError(f"Invalid recipient KEM fingerprint length: {len(self.recipient_kem_fingerprint)} != 32")
        if len(self.key_confirmation_tag) != 16:
            raise PQCEnvelopeError(f"Invalid key confirmation tag length: {len(self.key_confirmation_tag)} != 16")
        if len(self.kem_ciphertext_sha256) != 32:
            raise PQCEnvelopeError(f"Invalid KEM ciphertext SHA-256 hash length: {len(self.kem_ciphertext_sha256)} != 32")
        if len(self.signature) != self.SIGNATURE_SIZE:
            raise PQCEnvelopeError(f"Invalid signature length: {len(self.signature)} != {self.SIGNATURE_SIZE}")

    def pack(self) -> bytes:
        hdr = struct.pack(
            self.STRUCT_HEADER_FORMAT,
            MAGIC_PQC_SIDECAR,
            self.version,
            self.ciphersuite_id,
            self.session_counter,
            self.session_timestamp_ms,
            self.sender_dsa_fingerprint,
            self.recipient_kem_fingerprint,
            self.key_confirmation_tag,
            self.kem_ciphertext_sha256,
        )
        return hdr + self.signature

    @classmethod
    def unpack(cls, data: bytes) -> PQCSidecar:
        if len(data) < cls.TOTAL_SIZE:
            raise PQCEnvelopeError(f"Insufficient bytes for PQCSidecar: got {len(data)}, expected {cls.TOTAL_SIZE}")
        magic, ver, cs, ctr, ts, s_fp, r_fp, tag, ct_hash = struct.unpack_from(cls.STRUCT_HEADER_FORMAT, data, 0)
        if magic != MAGIC_PQC_SIDECAR:
            raise PQCEnvelopeError(f"Invalid PQCSidecar magic: expected {MAGIC_PQC_SIDECAR}, got {magic}")
        if ver != VERSION_PQC_V1:
            raise PQCEnvelopeError(f"Unsupported PQCSidecar version: {ver}")
        sig = data[cls.HEADER_SIZE : cls.HEADER_SIZE + cls.SIGNATURE_SIZE]
        return cls(
            session_counter=ctr,
            session_timestamp_ms=ts,
            sender_dsa_fingerprint=s_fp,
            recipient_kem_fingerprint=r_fp,
            key_confirmation_tag=tag,
            kem_ciphertext_sha256=ct_hash,
            signature=sig,
            version=ver,
            ciphersuite_id=cs,
        )

    def to_dict(self) -> dict:
        return {
            "magic": MAGIC_PQC_SIDECAR.decode("ascii"),
            "version": self.version,
            "ciphersuite_id": self.ciphersuite_id,
            "session_counter": self.session_counter,
            "session_timestamp_ms": self.session_timestamp_ms,
            "sender_dsa_fingerprint": self.sender_dsa_fingerprint.hex(),
            "recipient_kem_fingerprint": self.recipient_kem_fingerprint.hex(),
            "key_confirmation_tag": self.key_confirmation_tag.hex(),
            "kem_ciphertext_sha256": self.kem_ciphertext_sha256.hex(),
            "signature": self.signature.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PQCSidecar:
        return cls(
            session_counter=int(d["session_counter"]),
            session_timestamp_ms=int(d["session_timestamp_ms"]),
            sender_dsa_fingerprint=bytes.fromhex(d["sender_dsa_fingerprint"]),
            recipient_kem_fingerprint=bytes.fromhex(d["recipient_kem_fingerprint"]),
            key_confirmation_tag=bytes.fromhex(d["key_confirmation_tag"]),
            kem_ciphertext_sha256=bytes.fromhex(d["kem_ciphertext_sha256"]),
            signature=bytes.fromhex(d["signature"]),
            version=int(d.get("version", VERSION_PQC_V1)),
            ciphersuite_id=int(d.get("ciphersuite_id", CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256)),
        )


@dataclass
class PQCIdentity:
    """Represents a validated cryptographic identity within the PaladinMedia trust domain."""

    identity_id: str
    name: str
    kem_public_key: bytes  # ML-KEM-768 public key (1,184 bytes)
    dsa_public_key: bytes  # ML-DSA-65 public key (1,952 bytes)
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 86400 * 365)  # 1 year default
    is_revoked: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.kem_public_key) != 1184:
            raise PQCIdentityError(f"Invalid ML-KEM-768 public key length: {len(self.kem_public_key)} != 1184")
        if len(self.dsa_public_key) != 1952:
            raise PQCIdentityError(f"Invalid ML-DSA-65 public key length: {len(self.dsa_public_key)} != 1952")

    @property
    def kem_fingerprint(self) -> bytes:
        """32-byte SHA-256 fingerprint of the ML-KEM-768 public key."""
        return hashlib.sha256(self.kem_public_key).digest()

    @property
    def dsa_fingerprint(self) -> bytes:
        """32-byte SHA-256 fingerprint of the ML-DSA-65 public key."""
        return hashlib.sha256(self.dsa_public_key).digest()

    def validate(self, now: Optional[float] = None) -> None:
        """Validates that the identity is not revoked and not expired."""
        if self.is_revoked:
            raise PQCKeyRevokedError(f"Identity '{self.identity_id}' ({self.name}) key has been revoked")
        current_time = now if now is not None else time.time()
        if current_time > self.expires_at:
            raise PQCKeyExpiredError(
                f"Identity '{self.identity_id}' ({self.name}) key expired at {self.expires_at} (current {current_time})"
            )


@dataclass(frozen=True)
class PQCHeaderB1:
    """
    Canonical Layout B.1 104-byte uncompressed PQC-AKE header.

    Binary layout (>4sBHBQQ32s32s16s):
      0x00 - 0x03 (4B): magic b"PQCA"
      0x04        (1B): version (0x01)
      0x05 - 0x06 (2B): ciphersuite_id uint16 big-endian (0x0001)
      0x07        (1B): flags uint8 bitmask (0x00)
      0x08 - 0x0F (8B): session_counter uint64 big-endian
      0x10 - 0x17 (8B): session_timestamp uint64 big-endian (ms epoch)
      0x18 - 0x37 (32B): sender_id_fingerprint (SHA-256 of sender DSA pk)
      0x38 - 0x57 (32B): recipient_id_fingerprint (SHA-256 of recipient KEM pk)
      0x58 - 0x67 (16B): key_confirmation_tag (truncated HMAC-SHA256)
    Total: exactly 104 bytes.
    """

    STRUCT_FORMAT = ">4sBHBQQ32s32s16s"
    HEADER_SIZE = 104

    session_counter: int
    session_timestamp_ms: int
    sender_dsa_fingerprint: bytes
    recipient_kem_fingerprint: bytes
    key_confirmation_tag: bytes
    version: int = VERSION_PQC_V1
    ciphersuite_id: int = CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256
    flags: int = 0

    def __post_init__(self) -> None:
        if len(self.sender_dsa_fingerprint) != 32:
            raise PQCEnvelopeError(f"Invalid sender DSA fingerprint length: {len(self.sender_dsa_fingerprint)} != 32")
        if len(self.recipient_kem_fingerprint) != 32:
            raise PQCEnvelopeError(f"Invalid recipient KEM fingerprint length: {len(self.recipient_kem_fingerprint)} != 32")
        if len(self.key_confirmation_tag) != 16:
            raise PQCEnvelopeError(f"Invalid key confirmation tag length: {len(self.key_confirmation_tag)} != 16")

    def pack(self) -> bytes:
        return struct.pack(
            self.STRUCT_FORMAT,
            MAGIC_PQC_AKE,
            self.version,
            self.ciphersuite_id,
            self.flags,
            self.session_counter,
            self.session_timestamp_ms,
            self.sender_dsa_fingerprint,
            self.recipient_kem_fingerprint,
            self.key_confirmation_tag,
        )

    @classmethod
    def unpack(cls, data: bytes) -> PQCHeaderB1:
        if len(data) < cls.HEADER_SIZE:
            raise PQCEnvelopeError(f"Insufficient bytes for PQCHeaderB1: got {len(data)}, expected {cls.HEADER_SIZE}")
        magic, ver, cs, flags, counter, ts, s_fp, r_fp, tag = struct.unpack_from(cls.STRUCT_FORMAT, data, 0)
        if magic != MAGIC_PQC_AKE:
            raise PQCEnvelopeError(f"Invalid PQCHeaderB1 magic: expected {MAGIC_PQC_AKE}, got {magic}")
        if ver != VERSION_PQC_V1:
            raise PQCEnvelopeError(f"Unsupported PQCHeaderB1 version: {ver}")
        return cls(
            session_counter=counter,
            session_timestamp_ms=ts,
            sender_dsa_fingerprint=s_fp,
            recipient_kem_fingerprint=r_fp,
            key_confirmation_tag=tag,
            version=ver,
            ciphersuite_id=cs,
            flags=flags,
        )


@dataclass(frozen=True)
class PQCHeaderA:
    """
    Canonical Layout A 16-byte detached-signature PQC header.

    Binary layout (>4sBHBQ):
      0x00 - 0x03 (4B): magic b"PQCK"
      0x04        (1B): version (0x01)
      0x05 - 0x06 (2B): ciphersuite_id uint16 big-endian (0x0001)
      0x07        (1B): flags uint8 bitmask (0x01 = detached signature)
      0x08 - 0x0F (8B): session_counter uint64 big-endian
    Total: exactly 16 bytes.
    """

    STRUCT_FORMAT = ">4sBHBQ"
    HEADER_SIZE = 16

    session_counter: int
    version: int = VERSION_PQC_V1
    ciphersuite_id: int = CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256
    flags: int = FLAG_DETACHED_SIGNATURE

    def pack(self) -> bytes:
        return struct.pack(
            self.STRUCT_FORMAT,
            MAGIC_PQC_KEM,
            self.version,
            self.ciphersuite_id,
            self.flags,
            self.session_counter,
        )

    @classmethod
    def unpack(cls, data: bytes) -> PQCHeaderA:
        if len(data) < cls.HEADER_SIZE:
            raise PQCEnvelopeError(f"Insufficient bytes for PQCHeaderA: got {len(data)}, expected {cls.HEADER_SIZE}")
        magic, ver, cs, flags, counter = struct.unpack_from(cls.STRUCT_FORMAT, data, 0)
        if magic != MAGIC_PQC_KEM:
            raise PQCEnvelopeError(f"Invalid PQCHeaderA magic: expected {MAGIC_PQC_KEM}, got {magic}")
        if ver != VERSION_PQC_V1:
            raise PQCEnvelopeError(f"Unsupported PQCHeaderA version: {ver}")
        return cls(
            session_counter=counter,
            version=ver,
            ciphersuite_id=cs,
            flags=flags,
        )


@dataclass
class PQCEnvelope:
    """
    Complete packed Post-Quantum Cryptographic Envelope.

    Option B (Inline):
      [104B PQCHeaderB1]
      [1088B ML-KEM-768 Ciphertext]
      [3309B ML-DSA-65 Signature]
      [12B Nonce + Ciphertext + 16B GHASH Tag] (AES-256-GCM payload)

    Option A (Detached Signature):
      [16B PQCHeaderA]
      [1088B ML-KEM-768 Ciphertext]
      [12B Nonce + Ciphertext + 16B GHASH Tag] (AES-256-GCM payload)
      Signature delivered out-of-band.
    """

    header: PQCHeaderB1 | PQCHeaderA
    kem_ciphertext: bytes  # 1,088 bytes
    aes_gcm_payload: bytes  # 12B nonce + ciphertext + 16B tag
    signature: Optional[bytes] = None  # 3,309 bytes for Option B, None for Option A

    def pack(self) -> bytes:
        """Serializes the envelope into raw Pre-RS carrier bytes."""
        hdr_bytes = self.header.pack()
        if isinstance(self.header, PQCHeaderB1):
            if self.signature is None or len(self.signature) != 3309:
                raise PQCEnvelopeError("Option B envelope requires a 3,309-byte ML-DSA-65 signature")
            return hdr_bytes + self.kem_ciphertext + self.signature + self.aes_gcm_payload
        else:
            return hdr_bytes + self.kem_ciphertext + self.aes_gcm_payload

    @classmethod
    def unpack(cls, data: bytes) -> PQCEnvelope:
        """Parses envelope bytes by detecting magic."""
        if len(data) < 4:
            raise PQCEnvelopeError("Data too short to identify PQC envelope magic")
        magic = data[:4]
        if magic == MAGIC_PQC_AKE:
            # Option B
            if len(data) < PQCHeaderB1.HEADER_SIZE + 1088 + 3309 + 28:
                raise PQCEnvelopeError(
                    f"Option B envelope data too short: {len(data)} < {PQCHeaderB1.HEADER_SIZE + 1088 + 3309 + 28}"
                )
            header = PQCHeaderB1.unpack(data[: PQCHeaderB1.HEADER_SIZE])
            offset = PQCHeaderB1.HEADER_SIZE
            kem_ciphertext = data[offset : offset + 1088]
            offset += 1088
            signature = data[offset : offset + 3309]
            offset += 3309
            aes_payload = data[offset:]
            return cls(
                header=header,
                kem_ciphertext=kem_ciphertext,
                signature=signature,
                aes_gcm_payload=aes_payload,
            )
        elif magic == MAGIC_PQC_KEM:
            # Option A
            if len(data) < PQCHeaderA.HEADER_SIZE + 1088 + 28:
                raise PQCEnvelopeError(
                    f"Option A envelope data too short: {len(data)} < {PQCHeaderA.HEADER_SIZE + 1088 + 28}"
                )
            header = PQCHeaderA.unpack(data[: PQCHeaderA.HEADER_SIZE])
            offset = PQCHeaderA.HEADER_SIZE
            kem_ciphertext = data[offset : offset + 1088]
            offset += 1088
            aes_payload = data[offset:]
            return cls(
                header=header,
                kem_ciphertext=kem_ciphertext,
                signature=None,
                aes_gcm_payload=aes_payload,
            )
        else:
            raise PQCEnvelopeError(f"Unrecognized PQC envelope magic: {magic}")
