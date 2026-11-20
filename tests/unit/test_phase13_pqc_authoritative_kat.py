"""
Authoritative Known-Answer Tests (KAT) for NIST FIPS 203 (ML-KEM-768) and FIPS 204 (ML-DSA-65).

Validates:
1. ML-KEM-768 deterministic KeyGen from 64-byte seed (d || z) -> 1,184-byte encapsulation key (ek).
2. ML-KEM-768 decapsulation of valid ciphertexts to expected 32-byte shared secret (K).
3. ML-KEM-768 implicit rejection of corrupted/malleable ciphertexts to expected deterministic pseudorandom secret.
4. ML-DSA-65 deterministic KeyGen from 32-byte seed (xi) -> 1,952-byte public key (pk).
5. ML-DSA-65 signature verification across valid and invalid test vectors.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric import mlkem, mldsa
from cryptography.exceptions import InvalidSignature

VECTORS_FILE = Path(__file__).resolve().parents[1] / "data" / "pqc_kat_vectors.json"


@pytest.fixture(scope="module")
def kat_data():
    with open(VECTORS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_mlkem_768_authoritative_keygen_and_decaps(kat_data):
    """Verifies ML-KEM-768 KeyGen and Decapsulation against authoritative Wycheproof vectors."""
    for tc in kat_data["mlkem_768_vectors"]:
        tc_id = tc["tcId"]
        seed = bytes.fromhex(tc["seed_hex"])
        c = bytes.fromhex(tc["c_hex"])
        expected_k = tc["expected_k_hex"]
        expected_ek = tc["ek_hex"]

        # 1. Deterministic KeyGen
        priv = mlkem.MLKEM768PrivateKey.from_seed_bytes(seed)
        pub = priv.public_key()
        actual_ek = pub.public_bytes_raw().hex()
        assert actual_ek == expected_ek, f"KeyGen mismatch on tcId {tc_id}"

        # 2. Decapsulation (valid or implicit rejection)
        actual_k = priv.decapsulate(c).hex()
        assert actual_k == expected_k, f"Decapsulation mismatch on tcId {tc_id} ({tc['test_type']})"


def test_mldsa_65_authoritative_keygen(kat_data):
    """Verifies ML-DSA-65 deterministic KeyGen from 32-byte seed."""
    for i, tc in enumerate(kat_data["mldsa_65_keygen_vectors"]):
        seed = bytes.fromhex(tc["seed_hex"])
        expected_pk = tc["expected_pk_hex"]

        priv = mldsa.MLDSA65PrivateKey.from_seed_bytes(seed)
        actual_pk = priv.public_key().public_bytes_raw().hex()
        assert actual_pk == expected_pk, f"ML-DSA-65 KeyGen mismatch on case {i}"


def test_mldsa_65_authoritative_verification(kat_data):
    """Verifies ML-DSA-65 signature verification across valid and invalid vectors."""
    for tc in kat_data["mldsa_65_verify_vectors"]:
        tc_id = tc["tcId"]
        pk_bytes = bytes.fromhex(tc["pk_hex"])
        msg = bytes.fromhex(tc["msg_hex"])
        sig = bytes.fromhex(tc["sig_hex"])
        ctx = bytes.fromhex(tc["ctx_hex"]) if tc.get("ctx_hex") else None
        expected = tc["expected_result"]

        pub = mldsa.MLDSA65PublicKey.from_public_bytes(pk_bytes)
        if expected == "valid":
            pub.verify(sig, msg, context=ctx)
        else:
            with pytest.raises((InvalidSignature, Exception)):
                pub.verify(sig, msg, context=ctx)
