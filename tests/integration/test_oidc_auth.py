"""Integration tests for OAuth 2.0 / OpenID Connect (OIDC) BFF Authentication with PKCE.

Validates:
1. Generation of authorization URL with PKCE (S256 code_challenge, code_challenge_method, state, nonce).
2. Fail-closed rejection (HTTP 503) when OIDC is unconfigured in remote mode.
3. Successful authorization code exchange with mock IdP token endpoint using code_verifier.
4. Cryptographic validation of ID token claims:
   - Issuer (iss) validation
   - Audience (aud) validation
   - Nonce validation (preventing replay attacks)
   - Expiration (exp) enforcement
   - Subject (sub) extraction and session binding
5. Cookie-based BFF session establishment (HttpOnly paladin_session and readable paladin_csrf).
6. Replay protection: state parameter is strictly single-use.
7. Anti-spoofing: X-Forwarded-Proto is only honored from verified trusted proxy CIDRs/IPs.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import unittest.mock as mock
from typing import Any, Dict

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.security import (
    APISettings,
    default_auth_session_store,
    default_jwks_cache,
    default_oidc_state_store,
    generate_pkce_pair,
    is_request_https,
    is_trusted_proxy,
    reset_api_settings,
    reset_auth_session_store,
    reset_jwks_cache,
    reset_oidc_state_store,
    validate_id_token_claims,
)
from src.api.server import create_app

MOCK_ISSUER = "https://auth.paladinmedia.internal"
MOCK_CLIENT_ID = "paladinmedia-bff-client"
MOCK_CLIENT_SECRET = "super-secret-client-credential-key-12345"


def _make_mock_jwt(
    claims: Dict[str, Any],
    secret: str = MOCK_CLIENT_SECRET,
    alg: str = "HS256",
) -> str:
    header = {"alg": alg, "typ": "JWT"}
    h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    p_b64 = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    msg = f"{h_b64}.{p_b64}".encode("ascii")
    sig = (
        base64.urlsafe_b64encode(
            hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )
    return f"{h_b64}.{p_b64}.{sig}"


@pytest.fixture(autouse=True)
def clean_state():
    reset_auth_session_store()
    reset_oidc_state_store()
    reset_api_settings()
    reset_jwks_cache()
    yield
    reset_auth_session_store()
    reset_oidc_state_store()
    reset_api_settings()
    reset_jwks_cache()


def test_pkce_pair_generation():
    verifier, challenge = generate_pkce_pair()
    assert 43 <= len(verifier) <= 128
    # Recompute challenge to verify RFC 7636 math
    expected_digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected_challenge = base64.urlsafe_b64encode(expected_digest).decode("ascii").rstrip("=")
    assert challenge == expected_challenge


def test_oidc_login_redirect_url_generation():
    settings = APISettings(
        mode="remote",
        api_token="RemoteSecureSecretToken_1234567890abcdef!",
        oidc_issuer=MOCK_ISSUER,
        oidc_client_id=MOCK_CLIENT_ID,
        oidc_client_secret=MOCK_CLIENT_SECRET,
    )
    app = create_app(settings)
    client = TestClient(app)

    # 1. JSON inquiry for frontend SPA
    res_json = client.get("/api/auth/oidc/login?json=true")
    assert res_json.status_code == 200
    data = res_json.json()
    assert "authorization_url" in data
    auth_url = data["authorization_url"]
    assert MOCK_ISSUER in auth_url
    assert f"client_id={MOCK_CLIENT_ID}" in auth_url
    assert "code_challenge=" in auth_url
    assert "code_challenge_method=S256" in auth_url
    assert "state=" in auth_url
    assert "state" in data
    assert "code_challenge" in data

    # 2. Direct browser navigation 302 redirect
    res_redirect = client.get("/api/auth/oidc/login", follow_redirects=False)
    assert res_redirect.status_code == 302
    assert res_redirect.headers["location"].startswith(MOCK_ISSUER)


def test_oidc_login_unconfigured_remote_mode_fails_closed():
    settings = APISettings(
        mode="remote",
        api_token="RemoteSecureSecretToken_1234567890abcdef!",
        oidc_issuer=None,
        oidc_client_id=None,
    )
    app = create_app(settings)
    client = TestClient(app)

    res = client.get("/api/auth/oidc/login")
    assert res.status_code == 503
    assert "External IdP gate is OPEN" in res.json()["detail"]


def test_oidc_callback_complete_success_flow():
    settings = APISettings(
        mode="remote",
        api_token="RemoteSecureSecretToken_1234567890abcdef!",
        oidc_issuer=MOCK_ISSUER,
        oidc_client_id=MOCK_CLIENT_ID,
        oidc_client_secret=MOCK_CLIENT_SECRET,
    )
    app = create_app(settings)
    client = TestClient(app)

    # Step 1: Initiate login
    res_login = client.get("/api/auth/oidc/login?json=true")
    assert res_login.status_code == 200
    login_data = res_login.json()
    state = login_data["state"]

    # Verify state was stored with nonce
    stored_state = default_oidc_state_store._states.get(state)
    assert stored_state is not None
    nonce = stored_state.nonce

    # Step 2: Prepare valid ID token from IdP
    valid_claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "operator@paladinmedia.internal",
        "email": "operator@paladinmedia.internal",
        "roles": ["operator", "crypto-officer"],
        "nonce": nonce,
        "exp": time.time() + 3600,
        "iat": time.time(),
    }
    mock_id_token = _make_mock_jwt(valid_claims)

    # Step 3: Mock token exchange call
    mock_token_resp = {
        "access_token": "mock-access-token-12345",
        "token_type": "Bearer",
        "id_token": mock_id_token,
        "expires_in": 3600,
    }

    with mock.patch("src.api.server._exchange_code_at_idp", return_value=mock_token_resp) as mock_exchange:
        # Step 4: Callback arrives
        res_cb = client.get(
            f"/api/auth/oidc/callback?code=mock_authorization_code_abc&state={state}",
            follow_redirects=False,
        )
        assert res_cb.status_code == 302
        assert res_cb.headers["location"] == "/"

        # Check cookies
        cookies = res_cb.cookies
        assert "paladin_session" in cookies
        assert "paladin_csrf" in cookies

        # Verify token exchange parameters included code_verifier
        mock_exchange.assert_called_once()
        _, exchange_data = mock_exchange.call_args[0]
        assert exchange_data["grant_type"] == "authorization_code"
        assert exchange_data["code"] == "mock_authorization_code_abc"
        assert exchange_data["code_verifier"] == stored_state.code_verifier
        assert exchange_data["client_id"] == MOCK_CLIENT_ID
        assert exchange_data["client_secret"] == MOCK_CLIENT_SECRET

        # Step 5: Test session inspection using established cookie
        res_session = client.get("/api/auth/session", cookies=cookies)
        assert res_session.status_code == 200
        session_info = res_session.json()
        assert session_info["authenticated"] is True
        assert session_info["user"] == "operator@paladinmedia.internal"
        assert "operator" in session_info["roles"]
        assert session_info["csrf_token"] == cookies["paladin_csrf"]
        assert session_info["auth_method"] == "oidc"

        # Step 6: Test that replaying the same state immediately fails (single-use defense)
        res_replay = client.get(
            f"/api/auth/oidc/callback?code=mock_authorization_code_abc&state={state}",
            follow_redirects=False,
        )
        assert res_replay.status_code == 400
        assert "Invalid, expired, or replayed" in res_replay.json()["detail"]


def test_oidc_callback_state_tampering_rejected():
    settings = APISettings(
        mode="remote",
        api_token="RemoteSecureSecretToken_1234567890abcdef!",
        oidc_issuer=MOCK_ISSUER,
        oidc_client_id=MOCK_CLIENT_ID,
    )
    app = create_app(settings)
    client = TestClient(app)

    # Initiate login to generate a real state
    client.get("/api/auth/oidc/login?json=true")

    # Present forged state
    res_cb = client.get("/api/auth/oidc/callback?code=abc&state=tampered_or_unrecognized_state")
    assert res_cb.status_code == 400
    assert "Invalid, expired, or replayed" in res_cb.json()["detail"]


def test_oidc_callback_nonce_mismatch_rejected():
    settings = APISettings(
        mode="remote",
        api_token="RemoteSecureSecretToken_1234567890abcdef!",
        oidc_issuer=MOCK_ISSUER,
        oidc_client_id=MOCK_CLIENT_ID,
        oidc_client_secret=MOCK_CLIENT_SECRET,
    )
    app = create_app(settings)
    client = TestClient(app)

    res_login = client.get("/api/auth/oidc/login?json=true")
    state = res_login.json()["state"]

    # ID token with forged/wrong nonce
    tampered_claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "attacker@example.com",
        "nonce": "forged_nonce_12345",
        "exp": time.time() + 3600,
    }
    mock_id_token = _make_mock_jwt(tampered_claims)

    with mock.patch("src.api.server._exchange_code_at_idp", return_value={"id_token": mock_id_token}):
        res_cb = client.get(f"/api/auth/oidc/callback?code=abc&state={state}")
        assert res_cb.status_code == 401
        assert "nonce mismatch" in res_cb.json()["detail"]


def test_oidc_callback_issuer_mismatch_rejected():
    settings = APISettings(
        mode="remote",
        api_token="RemoteSecureSecretToken_1234567890abcdef!",
        oidc_issuer=MOCK_ISSUER,
        oidc_client_id=MOCK_CLIENT_ID,
        oidc_client_secret=MOCK_CLIENT_SECRET,
    )
    app = create_app(settings)
    client = TestClient(app)

    res_login = client.get("/api/auth/oidc/login?json=true")
    state = res_login.json()["state"]
    stored_state = default_oidc_state_store._states.get(state)

    tampered_claims = {
        "iss": "https://rogue-idp.attacker.com",
        "aud": MOCK_CLIENT_ID,
        "sub": "user@example.com",
        "nonce": stored_state.nonce,
        "exp": time.time() + 3600,
    }
    mock_id_token = _make_mock_jwt(tampered_claims)

    with mock.patch("src.api.server._exchange_code_at_idp", return_value={"id_token": mock_id_token}):
        res_cb = client.get(f"/api/auth/oidc/callback?code=abc&state={state}")
        assert res_cb.status_code == 401
        assert "issuer mismatch" in res_cb.json()["detail"]


def test_oidc_callback_audience_mismatch_rejected():
    settings = APISettings(
        mode="remote",
        api_token="RemoteSecureSecretToken_1234567890abcdef!",
        oidc_issuer=MOCK_ISSUER,
        oidc_client_id=MOCK_CLIENT_ID,
        oidc_client_secret=MOCK_CLIENT_SECRET,
    )
    app = create_app(settings)
    client = TestClient(app)

    res_login = client.get("/api/auth/oidc/login?json=true")
    state = res_login.json()["state"]
    stored_state = default_oidc_state_store._states.get(state)

    tampered_claims = {
        "iss": MOCK_ISSUER,
        "aud": "wrong-client-app-id",
        "sub": "user@example.com",
        "nonce": stored_state.nonce,
        "exp": time.time() + 3600,
    }
    mock_id_token = _make_mock_jwt(tampered_claims)

    with mock.patch("src.api.server._exchange_code_at_idp", return_value={"id_token": mock_id_token}):
        res_cb = client.get(f"/api/auth/oidc/callback?code=abc&state={state}")
        assert res_cb.status_code == 401
        assert "audience mismatch" in res_cb.json()["detail"]


def test_oidc_callback_expired_token_rejected():
    settings = APISettings(
        mode="remote",
        api_token="RemoteSecureSecretToken_1234567890abcdef!",
        oidc_issuer=MOCK_ISSUER,
        oidc_client_id=MOCK_CLIENT_ID,
        oidc_client_secret=MOCK_CLIENT_SECRET,
    )
    app = create_app(settings)
    client = TestClient(app)

    res_login = client.get("/api/auth/oidc/login?json=true")
    state = res_login.json()["state"]
    stored_state = default_oidc_state_store._states.get(state)

    expired_claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "user@example.com",
        "nonce": stored_state.nonce,
        "exp": time.time() - 60,  # 1 minute in the past
    }
    mock_id_token = _make_mock_jwt(expired_claims)

    with mock.patch("src.api.server._exchange_code_at_idp", return_value={"id_token": mock_id_token}):
        res_cb = client.get(f"/api/auth/oidc/callback?code=abc&state={state}")
        assert res_cb.status_code == 401
        assert "expired" in res_cb.json()["detail"]


def test_trusted_proxy_verification():
    trusted = ["127.0.0.1", "::1", "10.0.0.0/8", "172.16.0.0/12"]

    # Loopback is always trusted
    assert is_trusted_proxy("127.0.0.1", trusted) is True
    assert is_trusted_proxy("::1", trusted) is True

    # Internal subnets
    assert is_trusted_proxy("10.1.2.3", trusted) is True
    assert is_trusted_proxy("172.18.0.2", trusted) is True

    # Untrusted public IPs
    assert is_trusted_proxy("198.51.100.1", trusted) is False
    assert is_trusted_proxy("203.0.113.5", trusted) is False
    assert is_trusted_proxy(None, trusted) is False
    assert is_trusted_proxy("invalid-ip-string", trusted) is False


# ============================================================================
# Asymmetric (RS256 / ES256) & Advanced OIDC Security Test Suite
# ============================================================================

def _int_to_b64url(val: int) -> str:
    b = val.to_bytes((val.bit_length() + 7) // 8, byteorder="big")
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _make_rsa_key_and_jwk(kid: str = "rsa-key-1"):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": kid,
        "use": "sig",
        "alg": "RS256",
        "n": _int_to_b64url(pub_numbers.n),
        "e": _int_to_b64url(pub_numbers.e),
    }
    return private_key, jwk


def _make_ec_key_and_jwk(kid: str = "ec-key-1"):
    private_key = ec.generate_private_key(ec.SECP256R1())
    pub_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "kid": kid,
        "use": "sig",
        "alg": "ES256",
        "x": _int_to_b64url(pub_numbers.x),
        "y": _int_to_b64url(pub_numbers.y),
    }
    return private_key, jwk


def test_oidc_rs256_valid_token_verification():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "alice@paladinmedia.org",
        "nonce": "test-nonce-rs256-123",
        "exp": time.time() + 300,
        "iat": time.time(),
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "rsa-1"})

    validated = validate_id_token_claims(
        id_token=token,
        expected_issuer=MOCK_ISSUER,
        expected_audience=MOCK_CLIENT_ID,
        expected_nonce="test-nonce-rs256-123",
        jwks_data=jwks,
    )
    assert validated["sub"] == "alice@paladinmedia.org"
    assert validated["aud"] == MOCK_CLIENT_ID


def test_oidc_es256_valid_token_verification():
    priv_key, jwk = _make_ec_key_and_jwk("ec-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "bob@paladinmedia.org",
        "nonce": "test-nonce-es256-456",
        "exp": time.time() + 300,
        "iat": time.time(),
    }
    token = jwt.encode(claims, priv_key, algorithm="ES256", headers={"kid": "ec-1"})

    validated = validate_id_token_claims(
        id_token=token,
        expected_issuer=MOCK_ISSUER,
        expected_audience=MOCK_CLIENT_ID,
        expected_nonce="test-nonce-es256-456",
        jwks_data=jwks,
    )
    assert validated["sub"] == "bob@paladinmedia.org"
    assert validated["iss"] == MOCK_ISSUER


def test_oidc_invalid_signature_rejected():
    priv_key_signer, _ = _make_rsa_key_and_jwk("rsa-1")
    _, jwk_verifier = _make_rsa_key_and_jwk("rsa-1")  # Different key, same kid!
    jwks = {"keys": [jwk_verifier]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "charlie@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
    }
    token = jwt.encode(claims, priv_key_signer, algorithm="RS256", headers={"kid": "rsa-1"})

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "signature verification failed" in exc_info.value.detail.lower()


def test_oidc_expired_token_rejected_asymmetric():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "alice@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() - 120,  # Expired 2 minutes ago (beyond 60s leeway)
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "rsa-1"})

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_oidc_wrong_issuer_rejected_asymmetric():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": "https://attacker-idp.example.com",
        "aud": MOCK_CLIENT_ID,
        "sub": "alice@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "rsa-1"})

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "issuer mismatch" in exc_info.value.detail.lower()


def test_oidc_wrong_audience_rejected_asymmetric():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": "different-client-id",
        "sub": "alice@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "rsa-1"})

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "audience mismatch" in exc_info.value.detail.lower()


def test_oidc_missing_nonce_rejected_asymmetric():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "alice@paladinmedia.org",
        "exp": time.time() + 300,
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "rsa-1"})

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="expected-nonce-999",
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "nonce mismatch or missing" in exc_info.value.detail.lower()


def test_oidc_incorrect_nonce_rejected_asymmetric():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "alice@paladinmedia.org",
        "nonce": "actual-nonce-111",
        "exp": time.time() + 300,
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "rsa-1"})

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="expected-nonce-999",
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "nonce mismatch or missing" in exc_info.value.detail.lower()


def test_oidc_multiple_audiences_missing_azp_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": [MOCK_CLIENT_ID, "second-resource-server"],
        "sub": "alice@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
        # azp is omitted!
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "rsa-1"})

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "missing required 'azp'" in exc_info.value.detail.lower()


def test_oidc_multiple_audiences_incorrect_azp_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": [MOCK_CLIENT_ID, "second-resource-server"],
        "azp": "second-resource-server",  # Mismatched azp!
        "sub": "alice@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "rsa-1"})

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "'azp' mismatch" in exc_info.value.detail.lower()


def test_oidc_multiple_audiences_correct_azp_accepted():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": [MOCK_CLIENT_ID, "second-resource-server"],
        "azp": MOCK_CLIENT_ID,  # Correct azp
        "sub": "alice@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "rsa-1"})

    validated = validate_id_token_claims(
        id_token=token,
        expected_issuer=MOCK_ISSUER,
        expected_audience=MOCK_CLIENT_ID,
        expected_nonce="nonce-123",
        jwks_data=jwks,
    )
    assert validated["azp"] == MOCK_CLIENT_ID
    assert MOCK_CLIENT_ID in validated["aud"]


def test_oidc_unknown_kid_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("known-key-1")
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "alice@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
    }
    token = jwt.encode(claims, priv_key, algorithm="RS256", headers={"kid": "unknown-key-999"})

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "not found in jwks" in exc_info.value.detail.lower()


def test_oidc_jwks_key_rotation():
    priv_v1, jwk_v1 = _make_rsa_key_and_jwk("key-v1")
    priv_v2, jwk_v2 = _make_rsa_key_and_jwk("key-v2")

    jwks_endpoint = "https://auth.paladinmedia.internal/protocol/openid-connect/certs"

    # Initial state: only key-v1 in JWKS
    server_state = {"keys": [jwk_v1]}

    class MockResponse:
        def __init__(self, data):
            self.status = 200
            self._data = json.dumps(data).encode("utf-8")
        def read(self, amt=None):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def mock_urlopen(req, timeout=5.0):
        return MockResponse(server_state)

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        # 1. First token signed with v1 key: success and cached
        token_v1 = jwt.encode(
            {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u1", "nonce": "n1", "exp": time.time() + 300},
            priv_v1,
            algorithm="RS256",
            headers={"kid": "key-v1"}
        )
        res1 = validate_id_token_claims(
            id_token=token_v1,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="n1",
            jwks_uri=jwks_endpoint,
        )
        assert res1["sub"] == "u1"

        # 2. Key rotation occurs: IdP starts issuing tokens with key-v2
        server_state["keys"] = [jwk_v1, jwk_v2]

        token_v2 = jwt.encode(
            {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u2", "nonce": "n2", "exp": time.time() + 300},
            priv_v2,
            algorithm="RS256",
            headers={"kid": "key-v2"}
        )

        # Force rotation cooldown to allow immediate refresh in test
        cached_entry = default_jwks_cache._cache.get(jwks_endpoint)
        if cached_entry:
            cached_entry["last_rotation"] = time.time() - 20.0

        res2 = validate_id_token_claims(
            id_token=token_v2,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="n2",
            jwks_uri=jwks_endpoint,
        )
        assert res2["sub"] == "u2"


def test_oidc_jwks_endpoint_unavailable_fails_closed():
    priv_key, _ = _make_rsa_key_and_jwk("rsa-1")
    token = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u1", "nonce": "n1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"}
    )

    import urllib.error
    def mock_urlopen_fail(req, timeout=5.0):
        raise urllib.error.URLError("Connection refused by IdP")

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen_fail):
        with pytest.raises(HTTPException) as exc_info:
            validate_id_token_claims(
                id_token=token,
                expected_issuer=MOCK_ISSUER,
                expected_audience=MOCK_CLIENT_ID,
                expected_nonce="n1",
                jwks_uri="https://unavailable-idp.paladinmedia.internal/jwks",
            )
        assert exc_info.value.status_code == 401
        assert "unavailable or key retrieval failed" in exc_info.value.detail.lower()


def test_oidc_unsigned_token_alg_none_rejected():
    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "alice@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
    }
    # Hand-craft unsigned alg=none token
    h_b64 = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
    p_b64 = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    unsigned_token = f"{h_b64}.{p_b64}."

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=unsigned_token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
        )
    assert exc_info.value.status_code == 401
    assert "alg=none" in exc_info.value.detail.lower()


def test_oidc_unsupported_algorithm_rejected():
    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "alice@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
    }
    token = jwt.encode(claims, "some-secret", algorithm="HS512")

    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
        )
    assert exc_info.value.status_code == 401
    assert "disallowed id token algorithm" in exc_info.value.detail.lower()


def test_oidc_algorithm_confusion_attempt_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    from cryptography.hazmat.primitives import serialization
    pub_pem = priv_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    jwks = {"keys": [jwk]}

    claims = {
        "iss": MOCK_ISSUER,
        "aud": MOCK_CLIENT_ID,
        "sub": "attacker@paladinmedia.org",
        "nonce": "nonce-123",
        "exp": time.time() + 300,
    }
    # Attacker manually crafts an HS256 token signed with the RSA public key bytes
    h_b64 = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "rsa-1"}).encode()).decode().rstrip("=")
    p_b64 = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    msg = f"{h_b64}.{p_b64}".encode("ascii")
    sig = base64.urlsafe_b64encode(hmac.new(pub_pem, msg, hashlib.sha256).digest()).decode("ascii").rstrip("=")
    forged_token = f"{h_b64}.{p_b64}.{sig}"

    # 1. When client_secret is NOT configured (asymmetric only): HS256 is not allowed
    with pytest.raises(HTTPException) as exc_info:
        validate_id_token_claims(
            id_token=forged_token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
            client_secret=None,
            jwks_data=jwks,
        )
    assert exc_info.value.status_code == 401
    assert "disallowed id token algorithm" in exc_info.value.detail.lower()

    # 2. When client_secret IS configured: HS256 is strictly verified against client_secret, NOT pub_pem!
    with pytest.raises(HTTPException) as exc_info2:
        validate_id_token_claims(
            id_token=forged_token,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce="nonce-123",
            client_secret=MOCK_CLIENT_SECRET,
            jwks_data=jwks,
        )
    assert exc_info2.value.status_code == 401
    assert "signature verification failed" in exc_info2.value.detail.lower()


def test_oidc_state_replay_protection():
    default_oidc_state_store.store_state(
        state="single-use-state-999",
        nonce="nonce-999",
        code_verifier="verifier-999",
        redirect_uri="https://app.paladinmedia.internal/callback",
    )

    first_consumed = default_oidc_state_store.consume_state("single-use-state-999")
    assert first_consumed is not None
    assert first_consumed.nonce == "nonce-999"

    replayed = default_oidc_state_store.consume_state("single-use-state-999")
    assert replayed is None


def test_oidc_missing_and_malformed_iss_claim_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    # 1. Missing iss
    token_no_iss = jwt.encode(
        {"aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(
            id_token=token_no_iss,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc1.value.status_code == 401
    assert "missing or invalid 'iss'" in exc1.value.detail.lower()

    # 2. Malformed iss (numeric)
    token_num_iss = jwt.encode(
        {"iss": 12345, "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(
            id_token=token_num_iss,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc2.value.status_code == 401
    assert "missing or invalid 'iss'" in exc2.value.detail.lower()


def test_oidc_missing_and_malformed_aud_claim_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    # 1. Missing aud
    token_no_aud = jwt.encode(
        {"iss": MOCK_ISSUER, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(
            id_token=token_no_aud,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc1.value.status_code == 401
    assert "missing required 'aud'" in exc1.value.detail.lower()

    # 2. Malformed aud (numeric)
    token_num_aud = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": 99999, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(
            id_token=token_num_aud,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc2.value.status_code == 401
    assert "invalid 'aud'" in exc2.value.detail.lower()

    # 3. Malformed aud list (contains non-strings)
    token_bad_list_aud = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": [MOCK_CLIENT_ID, 12345], "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc3:
        validate_id_token_claims(
            id_token=token_bad_list_aud,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc3.value.status_code == 401
    assert "list elements must be strings" in exc3.value.detail.lower()


def test_oidc_missing_and_malformed_exp_claim_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    # 1. Missing exp
    token_no_exp = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u1"},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(
            id_token=token_no_exp,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc1.value.status_code == 401
    assert "missing required 'exp'" in exc1.value.detail.lower()

    # 2. Malformed exp (non-numeric string)
    token_bad_exp = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": "not-a-timestamp"},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(
            id_token=token_bad_exp,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc2.value.status_code == 401
    assert "invalid 'exp' timestamp" in exc2.value.detail.lower()


def test_oidc_malformed_and_future_iat_claim_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    # 1. Malformed iat (non-numeric)
    token_bad_iat = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300, "iat": "invalid-iat"},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(
            id_token=token_bad_iat,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc1.value.status_code == 401
    assert "invalid 'iat' timestamp" in exc1.value.detail.lower()

    # 2. Future-issued iat (outside 60s clock skew)
    token_future_iat = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 3600, "iat": time.time() + 600},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(
            id_token=token_future_iat,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc2.value.status_code == 401
    assert "'iat' is in the future" in exc2.value.detail.lower()


def test_oidc_missing_and_empty_sub_claim_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    # 1. Missing sub
    token_no_sub = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(
            id_token=token_no_sub,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc1.value.status_code == 401
    assert "missing required 'sub'" in exc1.value.detail.lower()

    # 2. Empty string sub
    token_empty_sub = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "   ", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(
            id_token=token_empty_sub,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc2.value.status_code == 401
    assert "missing required 'sub'" in exc2.value.detail.lower()


def test_oidc_single_audience_unexpected_mismatched_azp_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    # Single audience matches, but azp points to a different client!
    token_bad_azp = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "azp": "unauthorized-third-party", "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc:
        validate_id_token_claims(
            id_token=token_bad_azp,
            expected_issuer=MOCK_ISSUER,
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc.value.status_code == 401
    assert "'azp' mismatch" in exc.value.detail.lower()


def test_oidc_discovery_issuer_mismatch_rejected():
    # Discovery document returns issuer="https://rogue.example.com", which does not match expected_issuer
    discovery_response = {
        "issuer": "https://rogue.example.com",
        "jwks_uri": "https://auth.paladinmedia.internal/jwks",
    }

    class MockDiscoveryResp:
        def __init__(self):
            self.status = 200
            self._data = json.dumps(discovery_response).encode("utf-8")
        def read(self, n=-1):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    priv_key, _ = _make_rsa_key_and_jwk("rsa-1")
    token = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )

    with mock.patch("urllib.request.build_opener") as mock_opener_cls:
        mock_opener = mock.MagicMock()
        mock_opener.open.return_value = MockDiscoveryResp()
        mock_opener_cls.return_value = mock_opener

        with pytest.raises(HTTPException) as exc:
            validate_id_token_claims(
                id_token=token,
                expected_issuer=MOCK_ISSUER,
                expected_audience=MOCK_CLIENT_ID,
                expected_nonce=None,
                allow_local=True,
            )
        assert exc.value.status_code == 401
        assert "discovery document issuer mismatch" in exc.value.detail.lower()


def test_oidc_malformed_jwt_inputs_rejected():
    # 1. Non-string input
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(12345, expected_issuer=MOCK_ISSUER, expected_audience=MOCK_CLIENT_ID, expected_nonce=None)
    assert exc1.value.status_code == 400

    # 2. Wrong segment count (2 parts instead of 3)
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims("part1.part2", expected_issuer=MOCK_ISSUER, expected_audience=MOCK_CLIENT_ID, expected_nonce=None)
    assert exc2.value.status_code == 400
    assert "expected 3 dot-separated segments" in exc2.value.detail.lower()

    # 3. Malformed base64 header
    with pytest.raises(HTTPException) as exc3:
        validate_id_token_claims("!!!notbase64!!!.eyJzdWIiOiIxMjMifQ.sig", expected_issuer=MOCK_ISSUER, expected_audience=MOCK_CLIENT_ID, expected_nonce=None)
    assert exc3.value.status_code == 400
    assert "malformed id token header" in exc3.value.detail.lower()


def test_oidc_malformed_jwks_inputs_rejected():
    priv_key, _ = _make_rsa_key_and_jwk("rsa-1")
    token = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )

    # 1. Non-dict JWKS
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(token, MOCK_ISSUER, MOCK_CLIENT_ID, None, jwks_data="not-a-dict")
    assert exc1.value.status_code == 401
    assert "must be a json object" in exc1.value.detail.lower()

    # 2. Missing 'keys' member
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(token, MOCK_ISSUER, MOCK_CLIENT_ID, None, jwks_data={"other": []})
    assert exc2.value.status_code == 401
    assert "'keys' member must be a json array" in exc2.value.detail.lower()

    # 3. Empty 'keys' member
    with pytest.raises(HTTPException) as exc3:
        validate_id_token_claims(token, MOCK_ISSUER, MOCK_CLIENT_ID, None, jwks_data={"keys": []})
    assert exc3.value.status_code == 401
    assert "'keys' array is empty" in exc3.value.detail.lower()

    # 4. Duplicate 'kid' in JWKS
    _, jwk1 = _make_rsa_key_and_jwk("duplicate-kid")
    _, jwk2 = _make_rsa_key_and_jwk("duplicate-kid")
    with pytest.raises(HTTPException) as exc4:
        validate_id_token_claims(token, MOCK_ISSUER, MOCK_CLIENT_ID, None, jwks_data={"keys": [jwk1, jwk2]})
    assert exc4.value.status_code == 401
    assert "duplicate 'kid'" in exc4.value.detail.lower()


def test_oidc_jwks_oversized_response_rejected():
    # Simulate remote JWKS response exceeding MAX_JWKS_RESPONSE_BYTES (256 KB)
    class MockOversizedResp:
        def __init__(self):
            self.status = 200
        def read(self, n=-1):
            return b"A" * (256 * 1024 + 10)  # > 256 KB
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    priv_key, _ = _make_rsa_key_and_jwk("rsa-1")
    token = jwt.encode(
        {"iss": MOCK_ISSUER, "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )

    with mock.patch("urllib.request.build_opener") as mock_opener_cls:
        mock_opener = mock.MagicMock()
        mock_opener.open.return_value = MockOversizedResp()
        mock_opener_cls.return_value = mock_opener

        with pytest.raises(HTTPException) as exc:
            validate_id_token_claims(
                id_token=token,
                expected_issuer=MOCK_ISSUER,
                expected_audience=MOCK_CLIENT_ID,
                expected_nonce=None,
                jwks_uri="https://auth.paladinmedia.internal/jwks",
                allow_local=True,
            )
        assert exc.value.status_code == 401
        assert "exceeded maximum allowed size" in exc.value.detail.lower()


def test_oidc_ssrf_protection_remote_mode():
    priv_key, _ = _make_rsa_key_and_jwk("rsa-1")
    token = jwt.encode(
        {"iss": "https://127.0.0.1", "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )

    # 1. Loopback IP in remote mode (allow_local=False)
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(
            token,
            "https://127.0.0.1",
            MOCK_CLIENT_ID,
            None,
            jwks_uri="https://127.0.0.1/jwks",
            allow_local=False,
        )
    assert exc1.value.status_code == 401
    assert "ssrf protection" in exc1.value.detail.lower()

    # 2. Cloud metadata IP in remote mode
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(
            token,
            "https://169.254.169.254",
            MOCK_CLIENT_ID,
            None,
            jwks_uri="https://169.254.169.254/latest/meta-data/",
            allow_local=False,
        )
    assert exc2.value.status_code == 401
    assert "ssrf protection" in exc2.value.detail.lower()

    # 3. Plain HTTP in remote mode
    with pytest.raises(HTTPException) as exc3:
        validate_id_token_claims(
            token,
            "http://insecure-idp.com",
            MOCK_CLIENT_ID,
            None,
            jwks_uri="http://insecure-idp.com/jwks",
            allow_local=False,
        )
    assert exc3.value.status_code == 401
    assert "insecure url scheme" in exc3.value.detail.lower()


def test_oidc_strict_issuer_exact_match_accepted():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    # Exact matching issuer
    token = jwt.encode(
        {"iss": "https://auth.example.com/oauth2", "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    claims = validate_id_token_claims(
        id_token=token,
        expected_issuer="https://auth.example.com/oauth2",
        expected_audience=MOCK_CLIENT_ID,
        expected_nonce=None,
        jwks_data=jwks,
    )
    assert claims["iss"] == "https://auth.example.com/oauth2"


def test_oidc_strict_issuer_trailing_slash_mismatches_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    # Case A: Configured has trailing slash, token does NOT
    token_no_slash = jwt.encode(
        {"iss": "https://auth.example.com/oauth2", "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(
            id_token=token_no_slash,
            expected_issuer="https://auth.example.com/oauth2/",
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc1.value.status_code == 401
    assert "id token issuer mismatch" in exc1.value.detail.lower()
    assert "expected 'https://auth.example.com/oauth2/', got 'https://auth.example.com/oauth2'" in exc1.value.detail

    # Case B: Configured does NOT have trailing slash, token DOES
    token_with_slash = jwt.encode(
        {"iss": "https://auth.example.com/oauth2/", "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(
            id_token=token_with_slash,
            expected_issuer="https://auth.example.com/oauth2",
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc2.value.status_code == 401
    assert "id token issuer mismatch" in exc2.value.detail.lower()
    assert "expected 'https://auth.example.com/oauth2', got 'https://auth.example.com/oauth2/'" in exc2.value.detail


def test_oidc_strict_issuer_path_scheme_and_port_mismatches_rejected():
    priv_key, jwk = _make_rsa_key_and_jwk("rsa-1")
    jwks = {"keys": [jwk]}

    # 1. Different paths
    token_diff_path = jwt.encode(
        {"iss": "https://auth.example.com/oauth2/v2", "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(
            token_diff_path,
            expected_issuer="https://auth.example.com/oauth2/v1",
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc1.value.status_code == 401
    assert "id token issuer mismatch" in exc1.value.detail.lower()

    # 2. Different schemes
    token_diff_scheme = jwt.encode(
        {"iss": "http://auth.example.com/oauth2", "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(
            token_diff_scheme,
            expected_issuer="https://auth.example.com/oauth2",
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc2.value.status_code == 401
    assert "id token issuer mismatch" in exc2.value.detail.lower()

    # 3. Different ports
    token_diff_port = jwt.encode(
        {"iss": "https://auth.example.com:9443/oauth2", "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )
    with pytest.raises(HTTPException) as exc3:
        validate_id_token_claims(
            token_diff_port,
            expected_issuer="https://auth.example.com:8443/oauth2",
            expected_audience=MOCK_CLIENT_ID,
            expected_nonce=None,
            jwks_data=jwks,
        )
    assert exc3.value.status_code == 401
    assert "id token issuer mismatch" in exc3.value.detail.lower()


def test_oidc_strict_discovery_issuer_trailing_slash_mismatch_rejected():
    discovery_response = {
        "issuer": "https://auth.paladinmedia.internal/",  # Has trailing slash
        "jwks_uri": "https://auth.paladinmedia.internal/jwks",
    }

    class MockDiscoveryResp:
        def __init__(self):
            self.status = 200
            self._data = json.dumps(discovery_response).encode("utf-8")
        def read(self, n=-1):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    priv_key, _ = _make_rsa_key_and_jwk("rsa-1")
    token = jwt.encode(
        {"iss": "https://auth.paladinmedia.internal", "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )

    with mock.patch("urllib.request.build_opener") as mock_opener_cls:
        mock_opener = mock.MagicMock()
        mock_opener.open.return_value = MockDiscoveryResp()
        mock_opener_cls.return_value = mock_opener

        # Configured issuer does NOT have trailing slash
        with pytest.raises(HTTPException) as exc:
            validate_id_token_claims(
                id_token=token,
                expected_issuer="https://auth.paladinmedia.internal",  # Exact match without slash expected
                expected_audience=MOCK_CLIENT_ID,
                expected_nonce=None,
                allow_local=True,
            )
        assert exc.value.status_code == 401
        assert "discovery document issuer mismatch" in exc.value.detail.lower()
        assert "expected 'https://auth.paladinmedia.internal', got 'https://auth.paladinmedia.internal/'" in exc.value.detail


def test_oidc_ssrf_advanced_destinations_remote_mode():
    priv_key, _ = _make_rsa_key_and_jwk("rsa-1")
    token = jwt.encode(
        {"iss": "https://example.com", "aud": MOCK_CLIENT_ID, "sub": "u1", "exp": time.time() + 300},
        priv_key,
        algorithm="RS256",
        headers={"kid": "rsa-1"},
    )

    # 1. IPv4-mapped IPv6 loopback
    with pytest.raises(HTTPException) as exc1:
        validate_id_token_claims(token, "https://example.com", MOCK_CLIENT_ID, None, jwks_uri="https://[::ffff:127.0.0.1]/jwks", allow_local=False)
    assert exc1.value.status_code == 401
    assert "ssrf protection" in exc1.value.detail.lower()

    # 2. IPv4-mapped IPv6 cloud metadata
    with pytest.raises(HTTPException) as exc2:
        validate_id_token_claims(token, "https://example.com", MOCK_CLIENT_ID, None, jwks_uri="https://[::ffff:169.254.169.254]/jwks", allow_local=False)
    assert exc2.value.status_code == 401
    assert "ssrf protection" in exc2.value.detail.lower()

    # 3. CGNAT / Shared address space (100.64.0.0/10)
    with pytest.raises(HTTPException) as exc3:
        validate_id_token_claims(token, "https://example.com", MOCK_CLIENT_ID, None, jwks_uri="https://100.64.0.1/jwks", allow_local=False)
    assert exc3.value.status_code == 401
    assert "ssrf protection" in exc3.value.detail.lower()

    # 4. Embedded user credentials
    with pytest.raises(HTTPException) as exc4:
        validate_id_token_claims(token, "https://example.com", MOCK_CLIENT_ID, None, jwks_uri="https://user:pass@auth.example.com/jwks", allow_local=False)
    assert exc4.value.status_code == 401
    assert "credentials" in exc4.value.detail.lower()

    # 5. Cloud metadata hostnames (instance-data, Google metadata)
    with pytest.raises(HTTPException) as exc5:
        validate_id_token_claims(token, "https://example.com", MOCK_CLIENT_ID, None, jwks_uri="https://instance-data/latest/meta-data/", allow_local=False)
    assert exc5.value.status_code == 401
    assert "ssrf protection" in exc5.value.detail.lower()


def test_oidc_redirect_safety_and_depth_limit():
    import urllib.request
    from src.api.security import SafeRedirectHandler

    handler = SafeRedirectHandler(allow_local=False, max_redirects=2)
    dummy_req = urllib.request.Request("https://auth.example.com/start")

    # 1. SSRF target on redirect is rejected
    with pytest.raises(HTTPException) as exc1:
        handler.redirect_request(dummy_req, None, 302, "Found", {}, "http://127.0.0.1/internal-jwks")
    assert exc1.value.status_code == 401
    assert "disallowed redirect target url" in exc1.value.detail.lower()

    # 2. Exceeding max redirects is rejected
    handler_depth = SafeRedirectHandler(allow_local=True, max_redirects=2)
    handler_depth.redirect_request(dummy_req, None, 302, "Found", {}, "https://auth.example.com/hop1")
    handler_depth.redirect_request(dummy_req, None, 302, "Found", {}, "https://auth.example.com/hop2")
    with pytest.raises(HTTPException) as exc2:
        handler_depth.redirect_request(dummy_req, None, 302, "Found", {}, "https://auth.example.com/hop3")
    assert exc2.value.status_code == 401
    assert "too many redirects" in exc2.value.detail.lower()


def test_oidc_jwks_cache_concurrency_and_unknown_kid_bounding():
    import concurrent.futures
    from src.api.security import JWKSCache

    cache = JWKSCache(default_ttl_seconds=300.0, rotation_cooldown_seconds=10.0)

    priv_key, jwk = _make_rsa_key_and_jwk("valid-key")
    server_state = {"keys": [jwk]}

    fetch_count = 0
    def mock_fetch(url, timeout=5.0, allow_local=False):
        nonlocal fetch_count
        fetch_count += 1
        return server_state

    with mock.patch("src.api.security._safe_fetch_json", side_effect=mock_fetch):
        # Initial fetch
        cache.get_signing_key(kid="valid-key", alg="RS256", issuer="https://auth.example.com", jwks_uri="https://auth.example.com/jwks", allow_local=True)
        assert fetch_count == 1

        # Now simulate 20 concurrent requests for 20 DIFFERENT unknown kids
        def request_unknown_kid(i):
            try:
                cache.get_signing_key(kid=f"unknown-kid-{i}", alg="RS256", issuer="https://auth.example.com", jwks_uri="https://auth.example.com/jwks", allow_local=True)
                return "unexpected-success"
            except HTTPException as e:
                return e.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(request_unknown_kid, range(20)))

        # All 20 must fail closed with 401
        assert all(res == 401 for res in results)
        # Because rotation_cooldown is 10s and all lookups run under lock, fetch_count should be at most 2 (initial + 1 rotation refresh)
        assert fetch_count <= 2



