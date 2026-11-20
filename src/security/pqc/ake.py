"""
Authenticated Key Establishment (AKE) Engine for Post-Quantum Steganography in PaladinMedia.

Implements:
  - Canonical 1,213-byte pre-confirmation transcript construction (T_pre)
  - Primary transcript hash (TH1 = SHA-256(T_pre))
  - RFC 5869 HKDF-SHA256 key schedule (PRK, K_AES, K_CONF)
  - Non-recursive key confirmation tag (Tag_CONF = HMAC-SHA256(K_CONF, ...))
  - ML-DSA-65 digital signature over binding transcript (M_sig)
  - Authenticated payload encryption via AES-256-GCM with envelope AAD binding
  - Fail-closed verification and recovery pipeline
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import time
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

from src.security.pqc.exceptions import (
    PQCDecapsulationError,
    PQCEnvelopeError,
    PQCIdentityError,
    PQCKeyConfirmationError,
    PQCSignatureVerificationError,
)
from src.security.pqc.models import (
    CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256,
    FLAG_DETACHED_SIGNATURE,
    MAGIC_PQC_AKE,
    MAGIC_PQC_KEM,
    MAGIC_PQC_SIDECAR,
    VERSION_PQC_V1,
    PQCEnvelope,
    PQCHeaderA,
    PQCHeaderB1,
    PQCIdentity,
    PQCSidecar,
)
from src.security.pqc.provider import PQCProvider, default_pqc_provider
from src.security.pqc.trust import TrustStore, default_trust_store


PROTO_ID = b"PaladinMedia-WireV3-PQC-v1"  # 26 bytes
INFO_AES_KEY = b"PaladinMedia-v1-AES-256-GCM-Key"
INFO_KEY_CONF = b"PaladinMedia-v1-Key-Confirmation"
CONFIRM_SENDER_PREFIX = b"PaladinMedia-Confirm-Sender"
SIG_BINDING_PREFIX = b"PaladinMedia-v1-Signature-Binding"
SID_DERIVATION_PREFIX = b"PaladinMedia-SID-v1"


class AKEEngine:
    """Post-Quantum Authenticated Key Establishment engine."""

    def __init__(
        self,
        provider: Optional[PQCProvider] = None,
        trust_store: Optional[TrustStore] = None,
    ) -> None:
        self.provider = provider or default_pqc_provider
        self.trust_store = trust_store or default_trust_store

    @staticmethod
    def derive_sid(session_counter: int, session_timestamp_ms: int, kem_ciphertext: bytes) -> bytes:
        """Deterministically derives the 16-byte session identifier (SID) from ephemeral context."""
        h = hashlib.sha256()
        h.update(SID_DERIVATION_PREFIX)
        h.update(struct.pack(">QQ", session_counter, session_timestamp_ms))
        h.update(kem_ciphertext)
        return h.digest()[:16]

    @staticmethod
    def build_pre_confirmation_transcript(
        version: int,
        ciphersuite_id: int,
        session_counter: int,
        session_timestamp_ms: int,
        sid: bytes,
        sender_dsa_fingerprint: bytes,
        recipient_kem_fingerprint: bytes,
        kem_ciphertext: bytes,
    ) -> bytes:
        """
        Builds the 1,213-byte unhashed pre-confirmation transcript T_pre:
          1. PROTO_ID (26B)
          2. Version (1B)
          3. CS_ID (2B)
          4. CTR (8B)
          5. TS (8B)
          6. SID (16B)
          7. FP_DSA_S (32B)
          8. FP_KEM_R (32B)
          9. c_KEM (1,088B)
        """
        if len(sid) != 16:
            raise PQCEnvelopeError(f"Invalid SID length: {len(sid)} != 16")
        if len(sender_dsa_fingerprint) != 32:
            raise PQCEnvelopeError(f"Invalid sender DSA fingerprint length: {len(sender_dsa_fingerprint)} != 32")
        if len(recipient_kem_fingerprint) != 32:
            raise PQCEnvelopeError(f"Invalid recipient KEM fingerprint length: {len(recipient_kem_fingerprint)} != 32")
        if len(kem_ciphertext) != 1088:
            raise PQCEnvelopeError(f"Invalid KEM ciphertext length: {len(kem_ciphertext)} != 1088")

        transcript = (
            PROTO_ID
            + struct.pack(">BHQQ", version, ciphersuite_id, session_counter, session_timestamp_ms)
            + sid
            + sender_dsa_fingerprint
            + recipient_kem_fingerprint
            + kem_ciphertext
        )
        assert len(transcript) == 1213, f"Expected 1213-byte transcript, got {len(transcript)}"
        return transcript

    @staticmethod
    def compute_transcript_hash(transcript: bytes) -> bytes:
        """Computes TH1 = SHA-256(T_pre) (32 bytes)."""
        return hashlib.sha256(transcript).digest()

    @staticmethod
    def derive_keys(shared_secret: bytes, transcript_hash: bytes) -> Tuple[bytes, bytes]:
        """
        Derives K_AES (32B) and K_CONF (32B) using RFC 5869 HKDF-SHA256:
          PRK = HKDF-Extract(salt=TH1, IKM=SS_PQC)
          K_AES = HKDF-Expand(PRK, info=b"PaladinMedia-v1-AES-256-GCM-Key" || TH1, L=32)
          K_CONF = HKDF-Expand(PRK, info=b"PaladinMedia-v1-Key-Confirmation" || TH1, L=32)
        """
        # Step 1: HKDF-Extract: PRK = HMAC-SHA256(key=salt, msg=IKM)
        prk = hmac.new(transcript_hash, shared_secret, hashlib.sha256).digest()

        # Step 2: HKDF-Expand for K_AES
        info_aes = INFO_AES_KEY + transcript_hash
        k_aes = HKDFExpand(algorithm=hashes.SHA256(), length=32, info=info_aes).derive(prk)

        # Step 3: HKDF-Expand for K_CONF
        info_conf = INFO_KEY_CONF + transcript_hash
        k_conf = HKDFExpand(algorithm=hashes.SHA256(), length=32, info=info_conf).derive(prk)

        return k_aes, k_conf

    @staticmethod
    def compute_key_confirmation_tag(k_conf: bytes, transcript_hash: bytes) -> bytes:
        """Computes Tag_CONF = HMAC-SHA256(K_CONF, b"PaladinMedia-Confirm-Sender" || TH1)[:16]."""
        msg = CONFIRM_SENDER_PREFIX + transcript_hash
        return hmac.new(k_conf, msg, hashlib.sha256).digest()[:16]

    @staticmethod
    def compute_signature_binding(transcript_hash: bytes, key_confirmation_tag: bytes) -> bytes:
        """Computes M_sig = b"PaladinMedia-v1-Signature-Binding" || TH1 || Tag_CONF (81 bytes)."""
        msg = SIG_BINDING_PREFIX + transcript_hash + key_confirmation_tag
        assert len(msg) == 81, f"Expected 81-byte signature binding, got {len(msg)}"
        return msg

    def protect_payload(
        self,
        sender_id: str,
        sender_dsa_private_key: bytes,
        recipient_id: str,
        payload: bytes,
        option: str = "B",
        session_timestamp_ms: Optional[int] = None,
        custom_session_counter: Optional[int] = None,
        use_sidecar: bool = False,
    ) -> PQCEnvelope:
        """
        Secures payload using ML-KEM-768 encapsulation, ML-DSA-65 signature, and AES-256-GCM.

        Args:
            sender_id: Sender identity in trust store.
            sender_dsa_private_key: Sender's 1,952-byte ML-DSA-65 private key bytes.
            recipient_id: Recipient identity in trust store.
            payload: Plaintext payload bytes to encrypt.
            option: Delivery mode ("B" for inline 104B header, "A" for detached 16B header).
            session_timestamp_ms: Optional timestamp in ms (defaults to current time).
            custom_session_counter: Optional counter (defaults to monotonic counter from trust store).
            use_sidecar: When True with Option A, generates an authenticated PQCSidecar.

        Returns:
            Packed PQCEnvelope ready for carrier insertion.
        """
        sender = self.trust_store.get_identity(sender_id)
        if not sender:
            raise PQCIdentityError(f"Sender identity '{sender_id}' not found in trust store")
        sender.validate()

        recipient = self.trust_store.get_identity(recipient_id)
        if not recipient:
            raise PQCIdentityError(f"Recipient identity '{recipient_id}' not found in trust store")
        recipient.validate()

        now_ms = session_timestamp_ms if session_timestamp_ms is not None else int(time.time() * 1000)
        is_sidecar_mode = (option.upper() in ("A_SIDECAR", "SIDECAR")) or (option.upper() == "A" and use_sidecar)

        if option.upper() in ("A", "A_SIDECAR", "SIDECAR"):
            if is_sidecar_mode:
                ts_to_use = now_ms
                r_fp_to_use = recipient.kem_fingerprint
            else:
                ts_to_use = 0
                r_fp_to_use = hashlib.sha256(b"").digest()
        else:
            ts_to_use = now_ms
            r_fp_to_use = recipient.kem_fingerprint

        counter = (
            custom_session_counter
            if custom_session_counter is not None
            else self.trust_store.get_next_session_counter(sender_id)
        )

        # 1. Ephemeral ML-KEM-768 Encapsulation
        shared_secret, kem_ciphertext = self.provider.encapsulate(recipient.kem_public_key)

        # 2. Derive SID & build T_pre
        sid = self.derive_sid(counter, ts_to_use, kem_ciphertext)
        transcript = self.build_pre_confirmation_transcript(
            version=VERSION_PQC_V1,
            ciphersuite_id=CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256,
            session_counter=counter,
            session_timestamp_ms=ts_to_use,
            sid=sid,
            sender_dsa_fingerprint=sender.dsa_fingerprint,
            recipient_kem_fingerprint=r_fp_to_use,
            kem_ciphertext=kem_ciphertext,
        )

        # 3. Hash transcript & derive keys
        th1 = self.compute_transcript_hash(transcript)
        k_aes, k_conf = self.derive_keys(shared_secret, th1)

        # 4. Key confirmation tag
        tag_conf = self.compute_key_confirmation_tag(k_conf, th1)

        # 5. Sign transcript binding with sender's ML-DSA-65
        m_sig = self.compute_signature_binding(th1, tag_conf)
        signature = self.provider.sign(sender_dsa_private_key, m_sig)

        # 6. Assemble header and encrypt payload
        nonce = os.urandom(12)
        aesgcm = AESGCM(k_aes)

        if option.upper() == "B":
            header = PQCHeaderB1(
                session_counter=counter,
                session_timestamp_ms=now_ms,
                sender_dsa_fingerprint=sender.dsa_fingerprint,
                recipient_kem_fingerprint=recipient.kem_fingerprint,
                key_confirmation_tag=tag_conf,
            )
            aad = header.pack() + kem_ciphertext + signature
            ciphertext_with_tag = aesgcm.encrypt(nonce, payload, associated_data=aad)
            return PQCEnvelope(
                header=header,
                kem_ciphertext=kem_ciphertext,
                signature=signature,
                aes_gcm_payload=nonce + ciphertext_with_tag,
            )
        elif option.upper() == "A":
            if os.environ.get("PALADIN_ENABLE_PQC_LAYOUT_A", "false").lower() != "true":
                raise PQCEnvelopeError(
                    "Layout A detached signature mode is experimental and disabled by default. "
                    "Set PALADIN_ENABLE_PQC_LAYOUT_A=true to enable or use supported Layout B.1."
                )
            header = PQCHeaderA(session_counter=counter)
            aad = header.pack() + kem_ciphertext
            ciphertext_with_tag = aesgcm.encrypt(nonce, payload, associated_data=aad)
            sidecar = PQCSidecar(
                session_counter=counter,
                session_timestamp_ms=ts_to_use,
                sender_dsa_fingerprint=sender.dsa_fingerprint,
                recipient_kem_fingerprint=r_fp_to_use,
                key_confirmation_tag=tag_conf,
                kem_ciphertext_sha256=hashlib.sha256(kem_ciphertext).digest(),
                signature=signature,
            )
            envelope = PQCEnvelope(
                header=header,
                kem_ciphertext=kem_ciphertext,
                signature=signature,  # Passed for detached delivery
                aes_gcm_payload=nonce + ciphertext_with_tag,
            )
            envelope.sidecar = sidecar  # type: ignore[attr-defined]
            return envelope
        else:
            raise PQCEnvelopeError(f"Unsupported PQC envelope option: {option}")

    def unprotect_payload(
        self,
        recipient_kem_private_key: bytes,
        envelope: PQCEnvelope,
        detached_signature: Optional[bytes] = None,
        sender_id_override: Optional[str] = None,
        now_ms: Optional[int] = None,
        sidecar: Optional[PQCSidecar] = None,
    ) -> Tuple[str, bytes]:
        """
        Verifies and decrypts a PQCEnvelope.

        Args:
            recipient_kem_private_key: Recipient's ML-KEM-768 private key bytes.
            envelope: PQCEnvelope to unpack and verify.
            detached_signature: 3,309-byte ML-DSA-65 signature (required for Option A legacy).
            sender_id_override: Optional sender identity (used if Option A has no sender FP in header).
            now_ms: Optional reference timestamp in ms for testing clock skew.
            sidecar: Optional PQCSidecar instance carrying authenticated metadata for Option A.

        Returns:
            Tuple of (authenticated_sender_identity_id, decrypted_plaintext).
        """
        if isinstance(envelope.header, PQCHeaderB1):
            hdr = envelope.header
            # Lookup recipient
            recipient = self.trust_store.lookup_by_kem_fingerprint(hdr.recipient_kem_fingerprint)

            # Lookup sender
            sender = self.trust_store.lookup_by_dsa_fingerprint(hdr.sender_dsa_fingerprint)

            # Verify freshness window
            self.trust_store.verify_timestamp_freshness(hdr.session_timestamp_ms, now_ms=now_ms)

            # Check replay suppression WITHOUT advancing state yet
            self.trust_store.check_inbound_session_counter(sender.identity_id, hdr.session_counter)

            # Decapsulate shared secret (implicit rejection generates pseudo-secret if altered)
            shared_secret = self.provider.decapsulate(recipient_kem_private_key, envelope.kem_ciphertext)

            # Reconstruct SID & T_pre
            sid = self.derive_sid(hdr.session_counter, hdr.session_timestamp_ms, envelope.kem_ciphertext)
            transcript = self.build_pre_confirmation_transcript(
                version=hdr.version,
                ciphersuite_id=hdr.ciphersuite_id,
                session_counter=hdr.session_counter,
                session_timestamp_ms=hdr.session_timestamp_ms,
                sid=sid,
                sender_dsa_fingerprint=hdr.sender_dsa_fingerprint,
                recipient_kem_fingerprint=hdr.recipient_kem_fingerprint,
                kem_ciphertext=envelope.kem_ciphertext,
            )

            th1 = self.compute_transcript_hash(transcript)
            k_aes, k_conf = self.derive_keys(shared_secret, th1)

            # Verify key confirmation tag
            expected_tag = self.compute_key_confirmation_tag(k_conf, th1)
            if not hmac.compare_digest(expected_tag, hdr.key_confirmation_tag):
                raise PQCKeyConfirmationError("Key confirmation tag mismatch in AKE header")

            # Verify ML-DSA-65 signature
            if envelope.signature is None:
                raise PQCEnvelopeError("Missing signature in Option B envelope")
            m_sig = self.compute_signature_binding(th1, hdr.key_confirmation_tag)
            self.provider.verify(sender.dsa_public_key, m_sig, envelope.signature)

            # Decrypt AES-256-GCM
            if len(envelope.aes_gcm_payload) < 28:
                raise PQCEnvelopeError(f"AES-GCM payload too short: {len(envelope.aes_gcm_payload)} < 28")
            nonce = envelope.aes_gcm_payload[:12]
            ciphertext_with_tag = envelope.aes_gcm_payload[12:]
            aad = hdr.pack() + envelope.kem_ciphertext + envelope.signature

            aesgcm = AESGCM(k_aes)
            plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data=aad)

            # COMMIT counter only after AEAD tag and signature verification succeed
            self.trust_store.commit_inbound_session_counter(sender.identity_id, hdr.session_counter)
            return sender.identity_id, plaintext

        elif isinstance(envelope.header, PQCHeaderA):
            if os.environ.get("PALADIN_ENABLE_PQC_LAYOUT_A", "false").lower() != "true":
                raise PQCEnvelopeError(
                    "Layout A detached signature mode is experimental and disabled by default. "
                    "Set PALADIN_ENABLE_PQC_LAYOUT_A=true to enable or use supported Layout B.1."
                )

            if sidecar is not None:
                # Authenticated sidecar path
                expected_ct_hash = hashlib.sha256(envelope.kem_ciphertext).digest()
                if not hmac.compare_digest(sidecar.kem_ciphertext_sha256, expected_ct_hash):
                    raise PQCEnvelopeError("PQCSidecar carrier binding hash mismatch with envelope KEM ciphertext")
                if sidecar.session_counter != envelope.header.session_counter:
                    raise PQCEnvelopeError("PQCSidecar counter does not match envelope header counter")

                recipient = self.trust_store.lookup_by_kem_fingerprint(sidecar.recipient_kem_fingerprint)
                sender = self.trust_store.lookup_by_dsa_fingerprint(sidecar.sender_dsa_fingerprint)

                self.trust_store.verify_timestamp_freshness(sidecar.session_timestamp_ms, now_ms=now_ms)
                self.trust_store.check_inbound_session_counter(sender.identity_id, sidecar.session_counter)

                shared_secret = self.provider.decapsulate(recipient_kem_private_key, envelope.kem_ciphertext)
                sid = self.derive_sid(sidecar.session_counter, sidecar.session_timestamp_ms, envelope.kem_ciphertext)
                transcript = self.build_pre_confirmation_transcript(
                    version=sidecar.version,
                    ciphersuite_id=sidecar.ciphersuite_id,
                    session_counter=sidecar.session_counter,
                    session_timestamp_ms=sidecar.session_timestamp_ms,
                    sid=sid,
                    sender_dsa_fingerprint=sidecar.sender_dsa_fingerprint,
                    recipient_kem_fingerprint=sidecar.recipient_kem_fingerprint,
                    kem_ciphertext=envelope.kem_ciphertext,
                )
                th1 = self.compute_transcript_hash(transcript)
                k_aes, k_conf = self.derive_keys(shared_secret, th1)

                expected_tag = self.compute_key_confirmation_tag(k_conf, th1)
                if not hmac.compare_digest(expected_tag, sidecar.key_confirmation_tag):
                    raise PQCKeyConfirmationError("Key confirmation tag mismatch with sidecar")

                m_sig = self.compute_signature_binding(th1, sidecar.key_confirmation_tag)
                self.provider.verify(sender.dsa_public_key, m_sig, sidecar.signature)

                nonce = envelope.aes_gcm_payload[:12]
                ciphertext_with_tag = envelope.aes_gcm_payload[12:]
                aad = envelope.header.pack() + envelope.kem_ciphertext

                aesgcm = AESGCM(k_aes)
                plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data=aad)
                self.trust_store.commit_inbound_session_counter(sender.identity_id, sidecar.session_counter)
                return sender.identity_id, plaintext

            # Legacy detached signature fallback path
            sig = detached_signature or envelope.signature
            if not sig:
                raise PQCSignatureVerificationError("Detached signature required for Option A envelope verification")
            if not sender_id_override:
                raise PQCIdentityError("Sender identity required to verify Option A envelope")
            sender = self.trust_store.get_identity(sender_id_override)
            if not sender:
                raise PQCIdentityError(f"Sender identity '{sender_id_override}' not found")
            sender.validate()

            # Monotonic counter check
            self.trust_store.check_inbound_session_counter(sender.identity_id, envelope.header.session_counter)

            # Decapsulate
            shared_secret = self.provider.decapsulate(recipient_kem_private_key, envelope.kem_ciphertext)

            # Reconstruct with 0 timestamp for Option A legacy
            sid = self.derive_sid(envelope.header.session_counter, 0, envelope.kem_ciphertext)
            transcript = self.build_pre_confirmation_transcript(
                version=envelope.header.version,
                ciphersuite_id=envelope.header.ciphersuite_id,
                session_counter=envelope.header.session_counter,
                session_timestamp_ms=0,
                sid=sid,
                sender_dsa_fingerprint=sender.dsa_fingerprint,
                recipient_kem_fingerprint=hashlib.sha256(b"").digest(),  # placeholder for option A
                kem_ciphertext=envelope.kem_ciphertext,
            )
            th1 = self.compute_transcript_hash(transcript)
            k_aes, k_conf = self.derive_keys(shared_secret, th1)
            tag_conf = self.compute_key_confirmation_tag(k_conf, th1)

            m_sig = self.compute_signature_binding(th1, tag_conf)
            self.provider.verify(sender.dsa_public_key, m_sig, sig)

            nonce = envelope.aes_gcm_payload[:12]
            ciphertext_with_tag = envelope.aes_gcm_payload[12:]
            aad = envelope.header.pack() + envelope.kem_ciphertext

            aesgcm = AESGCM(k_aes)
            plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, associated_data=aad)
            self.trust_store.commit_inbound_session_counter(sender.identity_id, envelope.header.session_counter)
            return sender.identity_id, plaintext

        else:
            raise PQCEnvelopeError(f"Unsupported envelope header type: {type(envelope.header)}")
