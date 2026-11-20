"""
Consolidated Unit and Integration Tests for Phase 11 PQC and AKE Subsystem.

Covers:
  1. NIST FIPS 203 (ML-KEM-768) and NIST FIPS 204 (ML-DSA-65) parameter conformity
  2. Canonical AKE transcript construction (T_pre 1,213 B, TH1, HKDF key schedule)
  3. Non-recursive Key Confirmation Tag and ML-DSA-65 signature binding (M_sig)
  4. TrustStore lifecycle (identity pinning, revocation, expiration, replay counter, clock skew)
  5. Tamper paths and fail-closed security (corrupt signature, corrupt tag, invalid recipient)
  6. Delivery options: Option B (104B inline header) and Option A (16B detached header)
  7. End-to-end carrier roundtrip under Wire Protocol v3-MR with ONNX neural modulation
  8. Feature-gate enforcement (PALADIN_ENABLE_PQC=false fails closed with HTTP 403 / PQCDisabledError)
  9. Backward compatibility: Wire Protocol v2 classical pipeline remains untouched
"""

import os
import time
import pytest
import numpy as np
from PIL import Image

os.environ["PALADIN_ENABLE_PQC_LAYOUT_A"] = "true"

from src.core.exceptions import CryptoError, AuthenticationError
from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization.serializer import MediaType, SecretPayload
from src.security.pqc import (
    AKEEngine,
    CryptographyPQCProvider,
    PQCDisabledError,
    PQCEnvelope,
    PQCEnvelopeError,
    PQCHeaderA,
    PQCHeaderB1,
    PQCIdentity,
    PQCIdentityError,
    PQCKeyConfirmationError,
    PQCKeyExpiredError,
    PQCKeyRevokedError,
    PQCReplayError,
    PQCSignatureVerificationError,
    TrustStore,
    default_pqc_provider,
    default_trust_store,
)


@pytest.fixture(autouse=True)
def clean_environment_and_trust_store():
    """Ensures clean trust store and isolated environment for each test."""
    orig_pqc = os.environ.get("PALADIN_ENABLE_PQC")
    orig_v3 = os.environ.get("PALADIN_ENABLE_V3_MR")
    os.environ["PALADIN_ENABLE_PQC"] = "true"
    os.environ["PALADIN_ENABLE_V3_MR"] = "true"
    default_trust_store.clear()
    yield
    default_trust_store.clear()
    if orig_pqc is not None:
        os.environ["PALADIN_ENABLE_PQC"] = orig_pqc
    else:
        os.environ.pop("PALADIN_ENABLE_PQC", None)
    if orig_v3 is not None:
        os.environ["PALADIN_ENABLE_V3_MR"] = orig_v3
    else:
        os.environ.pop("PALADIN_ENABLE_V3_MR", None)


@pytest.fixture
def alice_and_bob(clean_environment_and_trust_store):
    """Creates registered Alice (sender) and Bob (recipient) identities."""
    provider = default_pqc_provider

    a_kem_pub, a_kem_priv = provider.generate_kem_keypair()
    a_dsa_pub, a_dsa_priv = provider.generate_dsa_keypair()
    alice = PQCIdentity("alice", "Alice", a_kem_pub, a_dsa_pub)

    b_kem_pub, b_kem_priv = provider.generate_kem_keypair()
    b_dsa_pub, b_dsa_priv = provider.generate_dsa_keypair()
    bob = PQCIdentity("bob", "Bob", b_kem_pub, b_dsa_pub)

    default_trust_store.register_identity(alice)
    default_trust_store.register_identity(bob)

    return {
        "alice": alice,
        "alice_dsa_priv": a_dsa_priv,
        "bob": bob,
        "bob_kem_priv": b_kem_priv,
    }


# =============================================================================
# 1. NIST Parameter Conformity
# =============================================================================

class TestNISTParameterConformity:
    def test_ml_kem_768_parameters(self):
        provider = default_pqc_provider
        pub, priv = provider.generate_kem_keypair()
        assert len(pub) == 1184, f"ML-KEM-768 public key size mismatch: {len(pub)} != 1184"
        assert len(priv) == 64, f"ML-KEM-768 seed private key size mismatch: {len(priv)} != 64"

        ss, ct = provider.encapsulate(pub)
        assert len(ss) == 32, f"ML-KEM-768 shared secret size mismatch: {len(ss)} != 32"
        assert len(ct) == 1088, f"ML-KEM-768 ciphertext size mismatch: {len(ct)} != 1088"

        dec_ss = provider.decapsulate(priv, ct)
        assert dec_ss == ss, "Decapsulated shared secret does not match encapsulated shared secret"

    def test_ml_dsa_65_parameters(self):
        provider = default_pqc_provider
        pub, priv = provider.generate_dsa_keypair()
        assert len(pub) == 1952, f"ML-DSA-65 public key size mismatch: {len(pub)} != 1952"
        assert len(priv) == 32, f"ML-DSA-65 seed private key size mismatch: {len(priv)} != 32"

        msg = b"NIST FIPS 204 conformance test message"
        sig = provider.sign(priv, msg)
        assert len(sig) == 3309, f"ML-DSA-65 signature size mismatch: {len(sig)} != 3309"
        assert provider.verify(pub, msg, sig) is True


# =============================================================================
# 2. Canonical AKE Transcript & Key Derivation
# =============================================================================

class TestCanonicalAKETranscript:
    def test_pre_confirmation_transcript_exact_byte_size(self, alice_and_bob):
        engine = AKEEngine()
        ct_kem = b"\x01" * 1088
        sid = b"\x02" * 16
        s_fp = alice_and_bob["alice"].dsa_fingerprint
        r_fp = alice_and_bob["bob"].kem_fingerprint

        t_pre = engine.build_pre_confirmation_transcript(
            version=1,
            ciphersuite_id=1,
            session_counter=42,
            session_timestamp_ms=1700000000000,
            sid=sid,
            sender_dsa_fingerprint=s_fp,
            recipient_kem_fingerprint=r_fp,
            kem_ciphertext=ct_kem,
        )
        assert len(t_pre) == 1213, f"Canonical transcript must be 1,213 bytes, got {len(t_pre)}"

    def test_hkdf_key_schedule_and_confirmation_tag(self, alice_and_bob):
        engine = AKEEngine()
        ss = b"\xaa" * 32
        th1 = b"\xbb" * 32
        k_aes, k_conf = engine.derive_keys(ss, th1)
        assert len(k_aes) == 32
        assert len(k_conf) == 32
        assert k_aes != k_conf

        tag = engine.compute_key_confirmation_tag(k_conf, th1)
        assert len(tag) == 16

        m_sig = engine.compute_signature_binding(th1, tag)
        assert len(m_sig) == 81


# =============================================================================
# 3. Trust Store & Lifecycle Enforcement
# =============================================================================

class TestTrustStoreLifecycle:
    def test_unknown_identity_rejected(self):
        engine = AKEEngine()
        with pytest.raises(PQCIdentityError, match="not found in trust store"):
            engine.protect_payload(
                sender_id="unknown_sender",
                sender_dsa_private_key=b"\x00" * 32,
                recipient_id="bob",
                payload=b"test",
            )

    def test_revoked_identity_rejected(self, alice_and_bob):
        default_trust_store.revoke_identity("alice")
        engine = AKEEngine()
        with pytest.raises(PQCKeyRevokedError, match="has been revoked"):
            engine.protect_payload(
                sender_id="alice",
                sender_dsa_private_key=alice_and_bob["alice_dsa_priv"],
                recipient_id="bob",
                payload=b"test",
            )

    def test_expired_identity_rejected(self):
        provider = default_pqc_provider
        pub_kem, priv_kem = provider.generate_kem_keypair()
        pub_dsa, priv_dsa = provider.generate_dsa_keypair()
        
        # 1. Registration of already-expired key rejected immediately
        expired_ident = PQCIdentity(
            identity_id="eve",
            name="Eve",
            kem_public_key=pub_kem,
            dsa_public_key=pub_dsa,
            expires_at=time.time() - 100,  # Already expired
        )
        with pytest.raises(PQCKeyExpiredError, match="expired"):
            default_trust_store.register_identity(expired_ident)

        # 2. Key that expires post-registration rejected during protect
        soon_expiring = PQCIdentity(
            identity_id="eve2",
            name="Eve2",
            kem_public_key=pub_kem,
            dsa_public_key=pub_dsa,
            expires_at=time.time() + 0.05,
        )
        default_trust_store.register_identity(soon_expiring)
        time.sleep(0.06)
        engine = AKEEngine()
        with pytest.raises(PQCKeyExpiredError, match="expired"):
            engine.protect_payload(
                sender_id="eve2",
                sender_dsa_private_key=priv_dsa,
                recipient_id="bob",
                payload=b"test",
            )

    def test_replay_suppression_counter_monotony(self, alice_and_bob):
        engine = AKEEngine()
        payload = b"test payload"
        env1 = engine.protect_payload(
            "alice", alice_and_bob["alice_dsa_priv"], "bob", payload, custom_session_counter=5
        )
        engine.unprotect_payload(alice_and_bob["bob_kem_priv"], env1)

        # Replay same counter -> PQCReplayError
        with pytest.raises(PQCReplayError, match="counter replay or regression"):
            engine.unprotect_payload(alice_and_bob["bob_kem_priv"], env1)

        # Regressive counter -> PQCReplayError
        env_old = engine.protect_payload(
            "alice", alice_and_bob["alice_dsa_priv"], "bob", payload, custom_session_counter=4
        )
        with pytest.raises(PQCReplayError, match="counter replay or regression"):
            engine.unprotect_payload(alice_and_bob["bob_kem_priv"], env_old)

    def test_timestamp_freshness_window_clock_skew(self, alice_and_bob):
        engine = AKEEngine()
        payload = b"timestamp freshness test"
        now_ms = int(time.time() * 1000)

        # Stale timestamp: 10 minutes ago (> 5 min max skew)
        stale_ts = now_ms - (600 * 1000)
        env = engine.protect_payload(
            "alice", alice_and_bob["alice_dsa_priv"], "bob", payload, session_timestamp_ms=stale_ts
        )
        with pytest.raises(PQCReplayError, match="exceeds allowed maximum"):
            engine.unprotect_payload(alice_and_bob["bob_kem_priv"], env, now_ms=now_ms)


# =============================================================================
# 4. Tamper Paths and Fail-Closed Security
# =============================================================================

class TestFailClosedSecurity:
    def test_corrupted_signature_fails_closed(self, alice_and_bob):
        engine = AKEEngine()
        payload = b"tamper signature test"
        env = engine.protect_payload("alice", alice_and_bob["alice_dsa_priv"], "bob", payload, option="B")
        
        # Corrupt 1 byte in signature
        bad_sig = bytearray(env.signature)
        bad_sig[100] ^= 0xFF
        env_corrupt = PQCEnvelope(
            header=env.header,
            kem_ciphertext=env.kem_ciphertext,
            signature=bytes(bad_sig),
            aes_gcm_payload=env.aes_gcm_payload,
        )
        with pytest.raises(PQCSignatureVerificationError, match="signature verification failed"):
            engine.unprotect_payload(alice_and_bob["bob_kem_priv"], env_corrupt)

    def test_corrupted_ciphertext_tag_fails_closed(self, alice_and_bob):
        engine = AKEEngine()
        payload = b"tamper ciphertext test"
        env = engine.protect_payload("alice", alice_and_bob["alice_dsa_priv"], "bob", payload, option="B")
        
        # Corrupt 1 byte in ciphertext
        bad_payload = bytearray(env.aes_gcm_payload)
        bad_payload[-1] ^= 0xFF
        env_corrupt = PQCEnvelope(
            header=env.header,
            kem_ciphertext=env.kem_ciphertext,
            signature=env.signature,
            aes_gcm_payload=bytes(bad_payload),
        )
        with pytest.raises(Exception):  # cryptography InvalidTag
            engine.unprotect_payload(alice_and_bob["bob_kem_priv"], env_corrupt)

    def test_tampered_header_tag_fails_confirmation(self, alice_and_bob):
        engine = AKEEngine()
        payload = b"tamper confirmation tag"
        env = engine.protect_payload("alice", alice_and_bob["alice_dsa_priv"], "bob", payload, option="B")

        bad_tag = bytearray(env.header.key_confirmation_tag)
        bad_tag[0] ^= 0xFF
        hdr = env.header
        bad_hdr = PQCHeaderB1(
            session_counter=hdr.session_counter,
            session_timestamp_ms=hdr.session_timestamp_ms,
            sender_dsa_fingerprint=hdr.sender_dsa_fingerprint,
            recipient_kem_fingerprint=hdr.recipient_kem_fingerprint,
            key_confirmation_tag=bytes(bad_tag),
        )
        env_corrupt = PQCEnvelope(
            header=bad_hdr,
            kem_ciphertext=env.kem_ciphertext,
            signature=env.signature,
            aes_gcm_payload=env.aes_gcm_payload,
        )
        with pytest.raises(PQCKeyConfirmationError, match="confirmation tag mismatch"):
            engine.unprotect_payload(alice_and_bob["bob_kem_priv"], env_corrupt)


# =============================================================================
# 5. Delivery Options A and B Envelope Encoding
# =============================================================================

class TestDeliveryModes:
    def test_option_b_roundtrip_and_exact_overhead(self, alice_and_bob):
        engine = AKEEngine()
        plaintext = b"PQC Option B test payload data."
        env = engine.protect_payload("alice", alice_and_bob["alice_dsa_priv"], "bob", plaintext, option="B")
        packed = env.pack()
        
        # Exactly: 104 (hdr) + 1088 (kem) + 3309 (sig) + 12 (nonce) + 16 (tag) + len(plaintext)
        expected_len = 104 + 1088 + 3309 + 12 + 16 + len(plaintext)
        assert len(packed) == expected_len, f"Option B packed size mismatch: {len(packed)} != {expected_len}"

        unpacked = PQCEnvelope.unpack(packed)
        sender_id, recovered = engine.unprotect_payload(alice_and_bob["bob_kem_priv"], unpacked)
        assert sender_id == "alice"
        assert recovered == plaintext

    def test_option_a_roundtrip_and_exact_overhead(self, alice_and_bob):
        engine = AKEEngine()
        plaintext = b"PQC Option A detached signature test."
        env = engine.protect_payload("alice", alice_and_bob["alice_dsa_priv"], "bob", plaintext, option="A")
        packed = env.pack()
        
        # Exactly: 16 (hdr) + 1088 (kem) + 12 (nonce) + 16 (tag) + len(plaintext)
        expected_len = 16 + 1088 + 12 + 16 + len(plaintext)
        assert len(packed) == expected_len, f"Option A packed size mismatch: {len(packed)} != {expected_len}"

        unpacked = PQCEnvelope.unpack(packed)
        sender_id, recovered = engine.unprotect_payload(
            alice_and_bob["bob_kem_priv"],
            unpacked,
            detached_signature=env.signature,
            sender_id_override="alice",
        )
        assert sender_id == "alice"
        assert recovered == plaintext


# =============================================================================
# 6. End-to-End Carrier Roundtrip on DIV2K Image
# =============================================================================

class TestEndToEndCarrierRoundtrip:
    def test_pqc_v3_mr_end_to_end_option_b(self, alice_and_bob):
        img_path = "dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/0601.png"
        img = Image.open(img_path).crop((0, 0, 512, 512))

        secret_bytes = b"END-TO-END PQC V3-MR QUALIFIED PAYLOAD WITH RS(255,191) AND NEURAL ONNX" * 10
        payload = SecretPayload(data=secret_bytes, media_type=MediaType.RAW_BYTES, metadata={})

        pipe = MediaStegoPipeline.create_v3_mr_deployment()
        stego_res, rec_res = pipe.run_full_roundtrip_pqc(
            secret_payload=payload,
            cover_image=img,
            sender_id="alice",
            sender_dsa_private_key=alice_and_bob["alice_dsa_priv"],
            recipient_id="bob",
            recipient_kem_private_key=alice_and_bob["bob_kem_priv"],
            option="B",
            version=3,
        )

        assert rec_res.exact_match is True
        assert rec_res.authenticated_sender_id == "alice"
        assert stego_res.psnr_db >= 40.0, f"PSNR {stego_res.psnr_db} < 40 dB"
        assert stego_res.ssim >= 0.999, f"SSIM {stego_res.ssim} < 0.999"

    def test_pqc_v3_mr_end_to_end_option_a(self, alice_and_bob):
        img_path = "dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/0601.png"
        img = Image.open(img_path).crop((0, 0, 512, 512))

        secret_bytes = b"OPTION A EXTENDED PAYLOAD CAPACITY VERIFICATION TEST" * 40
        payload = SecretPayload(data=secret_bytes, media_type=MediaType.RAW_BYTES, metadata={})

        pipe = MediaStegoPipeline.create_v3_mr_deployment()
        stego_res, rec_res = pipe.run_full_roundtrip_pqc(
            secret_payload=payload,
            cover_image=img,
            sender_id="alice",
            sender_dsa_private_key=alice_and_bob["alice_dsa_priv"],
            recipient_id="bob",
            recipient_kem_private_key=alice_and_bob["bob_kem_priv"],
            option="A",
            version=3,
        )

        assert rec_res.exact_match is True
        assert rec_res.authenticated_sender_id == "alice"
        assert stego_res.psnr_db >= 40.0


# =============================================================================
# 7. Feature Gate Enforcement
# =============================================================================

class TestFeatureGateEnforcement:
    def test_pipeline_fails_closed_when_pqc_disabled(self, alice_and_bob):
        os.environ["PALADIN_ENABLE_PQC"] = "false"
        pipe = MediaStegoPipeline.create_v3_mr_deployment()
        payload = SecretPayload(data=b"test", media_type=MediaType.RAW_BYTES)
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))

        with pytest.raises(PQCDisabledError, match="Post-Quantum Cryptography is disabled"):
            pipe.protect_and_embed_pqc(
                secret_payload=payload,
                cover_image=img,
                sender_id="alice",
                sender_dsa_private_key=alice_and_bob["alice_dsa_priv"],
                recipient_id="bob",
            )

    def test_api_fails_closed_when_pqc_disabled(self):
        os.environ["PALADIN_ENABLE_PQC"] = "false"
        from fastapi.testclient import TestClient
        from src.api.server import create_app
        from src.api.security import get_api_settings

        client = TestClient(create_app(get_api_settings()))
        resp = client.get("/api/pqc/identities")
        assert resp.status_code == 403
        assert "Post-Quantum Cryptography is disabled" in resp.json()["detail"]


# =============================================================================
# 8. Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    def test_wire_protocol_v2_classical_default_undisturbed(self):
        pipe = MediaStegoPipeline.create_qualified_deployment()
        img = Image.open("dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/0601.png").crop((0, 0, 512, 512))
        payload = SecretPayload(data=b"Classical Wire Protocol v2 Payload", media_type=MediaType.RAW_BYTES)
        key = b"\x42" * 32

        stego_res, rec_res = pipe.run_full_roundtrip(payload, img, key, version=2)
        assert rec_res.exact_match is True
        assert rec_res.header.version == 2
        assert rec_res.header.ecc_parity == 48
