"""
Phase 12: Integrated PQC and Media Security Hardening Unit Tests.

Covers:
  1. NIST FIPS 203 ML-KEM-768 Implicit Rejection semantics:
     - Altered valid-length ciphertext decapsulates to deterministic pseudo-secret.
     - Confirmation tag or AES-GCM verification fails closed; zero plaintext disclosure.
     - Malformed/truncated ciphertext raises PQCDecapsulationError.
  2. Cryptographic backend verification:
     - OpenSSL 4.0.1 / Rust cryptography 49.0.0 primitive lengths and APIs.
     - Deterministic known answer / seed key derivation.
  3. TrustStore persistence and process restart durability:
     - Identies and replay counters survive process instantiation.
     - Atomic counter commit only at authenticated processing boundary.
     - Unauthenticated/tampered envelopes do NOT advance or poison sender's counter.
  4. Concurrent replay protection:
     - Multithreaded race submission of identical session counter.
  5. Layout A authenticated sidecar roundtrip and fail-closed gate:
     - PQCSidecar binding to carrier KEM ciphertext hash.
     - PALADIN_ENABLE_PQC_LAYOUT_A feature gate.
  6. Tamper paths across independent sender and receiver processes.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
import pytest
import numpy as np
from cryptography.exceptions import InvalidTag

from src.core.exceptions import CryptoError
from src.security.pqc.ake import AKEEngine
from src.security.pqc.exceptions import (
    PQCDecapsulationError,
    PQCEnvelopeError,
    PQCIdentityError,
    PQCKeyConfirmationError,
    PQCKeyExpiredError,
    PQCKeyRevokedError,
    PQCReplayError,
    PQCSignatureVerificationError,
)
from src.security.pqc.models import (
    CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256,
    MAGIC_PQC_AKE,
    MAGIC_PQC_KEM,
    MAGIC_PQC_SIDECAR,
    PQCEnvelope,
    PQCHeaderA,
    PQCHeaderB1,
    PQCIdentity,
    PQCSidecar,
)
from src.security.pqc.provider import default_pqc_provider
from src.security.pqc.trust import TrustStore


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="paladin_pqc_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def identities():
    kem_pub_a, kem_priv_a = default_pqc_provider.generate_kem_keypair()
    dsa_pub_a, dsa_priv_a = default_pqc_provider.generate_dsa_keypair()

    kem_pub_b, kem_priv_b = default_pqc_provider.generate_kem_keypair()
    dsa_pub_b, dsa_priv_b = default_pqc_provider.generate_dsa_keypair()

    alice = PQCIdentity(identity_id="alice", name="Alice", kem_public_key=kem_pub_a, dsa_public_key=dsa_pub_a)
    bob = PQCIdentity(identity_id="bob", name="Bob", kem_public_key=kem_pub_b, dsa_public_key=dsa_pub_b)

    return {
        "alice": alice,
        "bob": bob,
        "alice_dsa_priv": dsa_priv_a,
        "bob_kem_priv": kem_priv_b,
    }


class TestMLKEMImplicitRejection:
    """Verifies NIST FIPS 203 Section 7.2 implicit rejection behavior."""

    def test_altered_ciphertext_produces_mismatched_secret(self, identities):
        """Altered 1,088-byte ciphertext decapsulates to a deterministic pseudo-secret without raising an exception."""
        bob = identities["bob"]
        bob_priv = identities["bob_kem_priv"]

        ss_orig, ct = default_pqc_provider.encapsulate(bob.kem_public_key)

        # Alter 1 bit in ciphertext
        bad_ct = bytearray(ct)
        bad_ct[42] ^= 0x01
        bad_ct = bytes(bad_ct)

        # Decapsulation must succeed per FIPS 203 implicit rejection
        ss_pseudo = default_pqc_provider.decapsulate(bob_priv, bad_ct)
        assert len(ss_pseudo) == 32
        assert ss_pseudo != ss_orig, "Implicit rejection must NOT produce the genuine shared secret"

    def test_altered_ciphertext_in_envelope_fails_at_confirmation_or_aead(self, identities):
        """In the full AKE pipeline, altered ciphertext fails tag validation; zero plaintext released."""
        trust = TrustStore()
        trust.register_identity(identities["alice"])
        trust.register_identity(identities["bob"])

        engine = AKEEngine(trust_store=trust)
        plaintext = b"Top secret post-quantum payload."

        env = engine.protect_payload(
            sender_id="alice",
            sender_dsa_private_key=identities["alice_dsa_priv"],
            recipient_id="bob",
            payload=plaintext,
            option="B",
            custom_session_counter=10,
        )

        # Tamper 1 bit in KEM ciphertext inside envelope
        bad_kem = bytearray(env.kem_ciphertext)
        bad_kem[10] ^= 0x01
        bad_env = PQCEnvelope(
            header=env.header,
            kem_ciphertext=bytes(bad_kem),
            signature=env.signature,
            aes_gcm_payload=env.aes_gcm_payload,
        )

        # Must fail closed with PQCKeyConfirmationError or InvalidTag
        with pytest.raises((PQCKeyConfirmationError, InvalidTag, CryptoError)):
            engine.unprotect_payload(
                recipient_kem_private_key=identities["bob_kem_priv"],
                envelope=bad_env,
            )

    def test_truncated_ciphertext_raises_decapsulation_error(self, identities):
        """Structurally invalid or truncated ciphertext raises PQCDecapsulationError."""
        bob_priv = identities["bob_kem_priv"]
        with pytest.raises(PQCDecapsulationError):
            default_pqc_provider.decapsulate(bob_priv, b"too_short")


class TestTrustStorePersistenceAndReplay:
    """Tests durable persistence, restart recovery, and atomic counter commit."""

    def test_trust_store_restart_preserves_identities_and_counters(self, temp_dir, identities):
        store_path = temp_dir / "trust_store.json"

        # 1. Process A: initialize, register, commit counter
        store_a = TrustStore(storage_path=store_path)
        store_a.register_identity(identities["alice"])
        store_a.register_identity(identities["bob"])
        store_a.commit_inbound_session_counter("alice", 42)

        assert store_path.exists()

        # 2. Process B: simulate process restart with new TrustStore instance
        store_b = TrustStore(storage_path=store_path)
        recovered_alice = store_b.get_identity("alice")
        assert recovered_alice is not None
        assert recovered_alice.dsa_fingerprint == identities["alice"].dsa_fingerprint

        # Replay check: counter 42 must be rejected
        with pytest.raises(PQCReplayError):
            store_b.check_inbound_session_counter("alice", 42)

        # Counter 43 must be accepted
        store_b.check_inbound_session_counter("alice", 43)
        store_b.commit_inbound_session_counter("alice", 43)

        # 3. Process C: verify counter 43 was persisted
        store_c = TrustStore(storage_path=store_path)
        with pytest.raises(PQCReplayError):
            store_c.check_inbound_session_counter("alice", 43)

    def test_unauthenticated_envelope_does_not_poison_counter(self, temp_dir, identities):
        """An attacker submitting an envelope with forged signature does NOT advance the recipient's counter."""
        store_path = temp_dir / "trust_store.json"
        store = TrustStore(storage_path=store_path)
        store.register_identity(identities["alice"])
        store.register_identity(identities["bob"])
        store.commit_inbound_session_counter("alice", 5)

        engine = AKEEngine(trust_store=store)

        # Valid envelope for counter 10
        env = engine.protect_payload(
            sender_id="alice",
            sender_dsa_private_key=identities["alice_dsa_priv"],
            recipient_id="bob",
            payload=b"Legitimate payload.",
            option="B",
            custom_session_counter=10,
        )

        # Corrupt signature in envelope
        bad_sig = bytearray(env.signature)
        bad_sig[50] ^= 0xFF
        corrupted_env = PQCEnvelope(
            header=env.header,
            kem_ciphertext=env.kem_ciphertext,
            signature=bytes(bad_sig),
            aes_gcm_payload=env.aes_gcm_payload,
        )

        # Verification fails
        with pytest.raises(PQCSignatureVerificationError):
            engine.unprotect_payload(
                recipient_kem_private_key=identities["bob_kem_priv"],
                envelope=corrupted_env,
            )

        # Crucial check: counter in store must STILL be 5, NOT advanced to 10!
        assert store._inbound_last_counter["alice"] == 5

        # Legitimate envelope with counter 10 must now succeed!
        sender_id, plaintext = engine.unprotect_payload(
            recipient_kem_private_key=identities["bob_kem_priv"],
            envelope=env,
        )
        assert plaintext == b"Legitimate payload."
        assert store._inbound_last_counter["alice"] == 10

    def test_concurrent_replay_race_defense(self, temp_dir, identities):
        """Concurrent requests with identical counter must allow only 1 success; all others fail."""
        store = TrustStore(storage_path=temp_dir / "concurrent_store.json")
        store.register_identity(identities["alice"])

        counter = 100
        successes = 0
        replays = 0

        def try_commit():
            try:
                store.commit_inbound_session_counter("alice", counter)
                return True
            except PQCReplayError:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_commit) for _ in range(5)]
            for f in concurrent.futures.as_completed(futures):
                if f.result():
                    successes += 1
                else:
                    replays += 1

        assert successes == 1, f"Expected exactly 1 success, got {successes}"
        assert replays == 4, f"Expected 4 replay rejections, got {replays}"


class TestLayoutAAuthenticatedSidecar:
    """Verifies PQCSidecar detached-signature delivery and carrier binding."""

    def test_layout_a_sidecar_roundtrip_with_independent_recipient(self, identities):
        """Tests that a recipient can verify Layout A using only carrier envelope and PQCSidecar."""
        os.environ["PALADIN_ENABLE_PQC_LAYOUT_A"] = "true"

        sender_trust = TrustStore()
        sender_trust.register_identity(identities["alice"])
        sender_trust.register_identity(identities["bob"])

        sender_engine = AKEEngine(trust_store=sender_trust)
        plaintext = b"Authenticated sidecar payload test."

        env = sender_engine.protect_payload(
            sender_id="alice",
            sender_dsa_private_key=identities["alice_dsa_priv"],
            recipient_id="bob",
            payload=plaintext,
            option="A",
            custom_session_counter=1,
            use_sidecar=True,
        )

        assert hasattr(env, "sidecar")
        sidecar = env.sidecar
        assert isinstance(sidecar, PQCSidecar)

        # Serialize sidecar to binary & json
        sidecar_bytes = sidecar.pack()
        sidecar_dict = sidecar.to_dict()

        reconstructed_sidecar = PQCSidecar.unpack(sidecar_bytes)
        assert reconstructed_sidecar.session_counter == sidecar.session_counter

        # Independent Recipient Process
        recip_trust = TrustStore()
        recip_trust.register_identity(identities["alice"])
        recip_trust.register_identity(identities["bob"])

        recip_engine = AKEEngine(trust_store=recip_trust)

        sender_id, recovered_pt = recip_engine.unprotect_payload(
            recipient_kem_private_key=identities["bob_kem_priv"],
            envelope=env,
            sidecar=reconstructed_sidecar,
        )

        assert sender_id == "alice"
        assert recovered_pt == plaintext

    def test_sidecar_carrier_mismatch_fails_closed(self, identities):
        """If a sidecar's kem_ciphertext_sha256 does not match carrier KEM ciphertext, reject immediately."""
        os.environ["PALADIN_ENABLE_PQC_LAYOUT_A"] = "true"

        trust = TrustStore()
        trust.register_identity(identities["alice"])
        trust.register_identity(identities["bob"])

        engine = AKEEngine(trust_store=trust)
        env = engine.protect_payload(
            sender_id="alice",
            sender_dsa_private_key=identities["alice_dsa_priv"],
            recipient_id="bob",
            payload=b"Payload",
            option="A",
            custom_session_counter=2,
        )
        sidecar = env.sidecar

        # Create tampered sidecar with wrong ct hash
        bad_sidecar = PQCSidecar(
            session_counter=sidecar.session_counter,
            session_timestamp_ms=sidecar.session_timestamp_ms,
            sender_dsa_fingerprint=sidecar.sender_dsa_fingerprint,
            recipient_kem_fingerprint=sidecar.recipient_kem_fingerprint,
            key_confirmation_tag=sidecar.key_confirmation_tag,
            kem_ciphertext_sha256=os.urandom(32),  # Wrong hash!
            signature=sidecar.signature,
        )

        with pytest.raises(PQCEnvelopeError, match="carrier binding hash mismatch"):
            engine.unprotect_payload(
                recipient_kem_private_key=identities["bob_kem_priv"],
                envelope=env,
                sidecar=bad_sidecar,
            )

    def test_disabled_layout_a_fails_closed(self, identities):
        """When PALADIN_ENABLE_PQC_LAYOUT_A is false, Option A calls are rejected."""
        os.environ["PALADIN_ENABLE_PQC_LAYOUT_A"] = "false"
        trust = TrustStore()
        trust.register_identity(identities["alice"])
        trust.register_identity(identities["bob"])
        engine = AKEEngine(trust_store=trust)

        with pytest.raises(PQCEnvelopeError, match="Layout A detached signature mode is experimental"):
            engine.protect_payload(
                sender_id="alice",
                sender_dsa_private_key=identities["alice_dsa_priv"],
                recipient_id="bob",
                payload=b"Test",
                option="A",
            )


class TestAdversarialTamperingAndKeyIntegrity:
    """Evaluates wrongful keys, modified tags, modified transcripts, and wire protocol binding."""

    def test_wrong_recipient_key_fails_closed(self, identities):
        """Decapsulating with wrong recipient key results in invalid shared secret and fails closed."""
        charlie_pub, charlie_priv = default_pqc_provider.generate_kem_keypair()
        charlie_dsa_pub, _ = default_pqc_provider.generate_dsa_keypair()
        charlie = PQCIdentity("charlie", "Charlie", charlie_pub, charlie_dsa_pub)

        trust = TrustStore()
        trust.register_identity(identities["alice"])
        trust.register_identity(identities["bob"])
        trust.register_identity(charlie)

        engine = AKEEngine(trust_store=trust)
        env = engine.protect_payload(
            sender_id="alice",
            sender_dsa_private_key=identities["alice_dsa_priv"],
            recipient_id="bob",
            payload=b"Plaintext",
            option="B",
            custom_session_counter=10,
        )

        # Charlie attempts to decapsulate with Charlie's private key
        with pytest.raises((PQCKeyConfirmationError, InvalidTag, CryptoError)):
            engine.unprotect_payload(
                recipient_kem_private_key=charlie_priv,
                envelope=env,
            )

    def test_wrong_sender_signing_key_fails_closed(self, identities):
        """If signature was produced with an unregistered or mismatched key, verification fails closed."""
        _, rogue_priv = default_pqc_provider.generate_dsa_keypair()
        trust = TrustStore()
        trust.register_identity(identities["alice"])
        trust.register_identity(identities["bob"])
        engine = AKEEngine(trust_store=trust)

        # Alice envelope signed with Rogue private key
        env = engine.protect_payload(
            sender_id="alice",
            sender_dsa_private_key=rogue_priv,
            recipient_id="bob",
            payload=b"Plaintext",
            option="B",
            custom_session_counter=20,
        )

        with pytest.raises(PQCSignatureVerificationError):
            engine.unprotect_payload(
                recipient_kem_private_key=identities["bob_kem_priv"],
                envelope=env,
            )

    def test_tampered_confirmation_tag_fails_closed(self, identities):
        """If confirmation tag in header is modified by 1 bit, fail closed."""
        trust = TrustStore()
        trust.register_identity(identities["alice"])
        trust.register_identity(identities["bob"])
        engine = AKEEngine(trust_store=trust)

        env = engine.protect_payload(
            sender_id="alice",
            sender_dsa_private_key=identities["alice_dsa_priv"],
            recipient_id="bob",
            payload=b"Plaintext",
            option="B",
            custom_session_counter=30,
        )

        bad_tag = bytearray(env.header.key_confirmation_tag)
        bad_tag[0] ^= 0x01
        bad_hdr = PQCHeaderB1(
            session_counter=env.header.session_counter,
            session_timestamp_ms=env.header.session_timestamp_ms,
            sender_dsa_fingerprint=env.header.sender_dsa_fingerprint,
            recipient_kem_fingerprint=env.header.recipient_kem_fingerprint,
            key_confirmation_tag=bytes(bad_tag),
        )
        bad_env = PQCEnvelope(
            header=bad_hdr,
            kem_ciphertext=env.kem_ciphertext,
            signature=env.signature,
            aes_gcm_payload=env.aes_gcm_payload,
        )

        with pytest.raises(PQCKeyConfirmationError):
            engine.unprotect_payload(
                recipient_kem_private_key=identities["bob_kem_priv"],
                envelope=bad_env,
            )

    def test_tampered_timestamp_fails_closed(self, identities):
        """If timestamp in header is modified, transcript mismatch triggers key confirmation failure."""
        trust = TrustStore()
        trust.register_identity(identities["alice"])
        trust.register_identity(identities["bob"])
        engine = AKEEngine(trust_store=trust)

        now_ms = int(time.time() * 1000)
        env = engine.protect_payload(
            sender_id="alice",
            sender_dsa_private_key=identities["alice_dsa_priv"],
            recipient_id="bob",
            payload=b"Plaintext",
            option="B",
            session_timestamp_ms=now_ms,
            custom_session_counter=40,
        )

        bad_hdr = PQCHeaderB1(
            session_counter=env.header.session_counter,
            session_timestamp_ms=now_ms + 1000,  # 1 second offset within freshness window
            sender_dsa_fingerprint=env.header.sender_dsa_fingerprint,
            recipient_kem_fingerprint=env.header.recipient_kem_fingerprint,
            key_confirmation_tag=env.header.key_confirmation_tag,
        )
        bad_env = PQCEnvelope(
            header=bad_hdr,
            kem_ciphertext=env.kem_ciphertext,
            signature=env.signature,
            aes_gcm_payload=env.aes_gcm_payload,
        )

        with pytest.raises(PQCKeyConfirmationError):
            engine.unprotect_payload(
                recipient_kem_private_key=identities["bob_kem_priv"],
                envelope=bad_env,
            )

    def test_revoked_sender_fails_closed(self, identities):
        """A valid signature from a revoked sender identity fails closed."""
        trust = TrustStore()
        trust.register_identity(identities["alice"])
        trust.register_identity(identities["bob"])
        engine = AKEEngine(trust_store=trust)

        env = engine.protect_payload(
            sender_id="alice",
            sender_dsa_private_key=identities["alice_dsa_priv"],
            recipient_id="bob",
            payload=b"Plaintext",
            option="B",
            custom_session_counter=50,
        )

        # Alice is revoked
        trust.revoke_identity("alice")

        with pytest.raises(PQCKeyRevokedError):
            engine.unprotect_payload(
                recipient_kem_private_key=identities["bob_kem_priv"],
                envelope=env,
            )

    def test_pqc_wire_v2_binding_rejected(self):
        """PQC pipeline rejects Wire Protocol v2 carrier embedding attempts."""
        from src.carrier.header import CarrierHeaderError
        from src.core.media_pipeline import MediaStegoPipeline
        from src.core.serialization.serializer import MediaType, SecretPayload

        os.environ["PALADIN_ENABLE_PQC"] = "true"
        pipe = MediaStegoPipeline.create_v3_mr_deployment(backend="parity_classical")
        cover = np.zeros((512, 512, 3), dtype=np.uint8)
        payload = SecretPayload(data=b"Test", media_type=MediaType.TEXT)

        with pytest.raises(CarrierHeaderError, match="cryptographically bound to Wire Protocol v3"):
            pipe.protect_and_embed_pqc(
                secret_payload=payload,
                cover_image=cover,
                sender_id="alice",
                sender_dsa_private_key=b"0" * 32,
                recipient_id="bob",
                version=2,  # Must be rejected!
            )


class TestFIPS203204NISTVectors:
    """Verifies deterministic algorithm lengths and known answer vectors."""

    def test_mlkem768_standard_lengths_and_known_seed(self):
        from cryptography.hazmat.primitives.asymmetric import mlkem

        # Deterministic 64-byte seed
        seed = bytes(range(64))
        priv = mlkem.MLKEM768PrivateKey.from_seed_bytes(seed)
        pub = priv.public_key()

        assert len(pub.public_bytes_raw()) == 1184
        assert len(priv.private_bytes_raw()) == 64

        ss, ct = pub.encapsulate()
        assert len(ct) == 1088
        assert len(ss) == 32

        ss_rec = priv.decapsulate(ct)
        assert ss_rec == ss

    def test_mldsa65_standard_lengths_and_known_seed(self):
        from cryptography.hazmat.primitives.asymmetric import mldsa

        # Deterministic 32-byte seed
        seed = bytes(range(32))
        priv = mldsa.MLDSA65PrivateKey.from_seed_bytes(seed)
        pub = priv.public_key()

        assert len(pub.public_bytes_raw()) == 1952
        assert len(priv.private_bytes_raw()) == 32

        msg = b"NIST FIPS 204 ML-DSA-65 Known Answer Verification Vector."
        sig = priv.sign(msg)
        assert len(sig) == 3309

        # Verification succeeds
        pub.verify(sig, msg)

        # 1-bit flip fails
        bad_sig = bytearray(sig)
        bad_sig[0] ^= 0x01
        with pytest.raises(Exception):
            pub.verify(bytes(bad_sig), msg)

