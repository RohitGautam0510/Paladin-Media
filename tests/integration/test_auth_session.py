"""Integration tests for Browser-for-Frontend (BFF) session authentication and CSRF protection.

Verifies:
1. Local development mode: unauthenticated access allowed, /api/auth/session reports requires_auth=False.
2. Remote production mode: endpoints reject unauthenticated requests with 401.
3. Login lifecycle: /api/auth/login with valid token creates session, returns csrf_token, and sets HttpOnly cookie.
4. Cookie-based authentication: GET requests with session cookie succeed.
5. CSRF enforcement: mutating requests (POST/PUT/DELETE) with session cookie fail with 403 without valid X-CSRF-Token.
6. CSRF success: mutating requests with session cookie and matching X-CSRF-Token succeed.
7. Dual-auth compatibility: headless clients with Bearer token header succeed on POST without CSRF header.
8. Logout lifecycle: /api/auth/logout invalidates session store and clears cookies.
9. Session eviction / TTL expiry: expired sessions fail closed with 401.
"""

from __future__ import annotations

import io
import time
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.security import (
    APISettings,
    default_auth_session_store,
    reset_api_settings,
    reset_auth_session_store,
)
from src.api.server import create_app

VALID_REMOTE_TOKEN = "a" * 32  # 32 characters minimum with diversity
VALID_KEY_HEX = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def _create_test_png(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clean_state():
    reset_auth_session_store()
    yield
    reset_auth_session_store()
    reset_api_settings()


def test_local_mode_auth_session():
    """In local mode without token, auth is not required and session check reflects this."""
    settings = APISettings(mode="local", api_token=None)
    app = create_app(settings)
    client = TestClient(app)

    res = client.get("/api/auth/session")
    assert res.status_code == 200
    data = res.json()
    assert data["authenticated"] is True
    assert data["requires_auth"] is False

    res_cap = client.get("/api/capabilities")
    assert res_cap.status_code == 200


def test_remote_mode_unauthenticated_rejection():
    """In remote mode, protected endpoints reject requests lacking auth with 401."""
    settings = APISettings(
        mode="remote",
        api_token="RemoteSecureSecretToken_1234567890abcdef!",
        allowed_origins=["http://localhost:5173"],
    )
    app = create_app(settings)
    client = TestClient(app)

    # Session check returns authenticated=False
    res_session = client.get("/api/auth/session")
    assert res_session.status_code == 200
    assert res_session.json()["authenticated"] is False
    assert res_session.json()["requires_auth"] is True

    # Capabilities rejected with 401
    res_cap = client.get("/api/capabilities")
    assert res_cap.status_code == 401

    # Upload rejected with 401
    png_bytes = _create_test_png()
    res_up = client.post("/api/upload", files={"file": ("test.png", png_bytes, "image/png")})
    assert res_up.status_code == 401


def test_login_and_cookie_session_lifecycle():
    """Verifies complete BFF login, CSRF enforcement, mutation, and logout."""
    token = "RemoteSecureSecretToken_1234567890abcdef!"
    settings = APISettings(
        mode="remote",
        api_token=token,
        allowed_origins=["http://localhost:5173"],
    )
    app = create_app(settings)
    client = TestClient(app)

    # 1. Invalid login attempt
    res_bad_login = client.post("/api/auth/login", json={"token": "wrong_token_value_here_123456"})
    assert res_bad_login.status_code == 401

    # 2. Valid login
    res_login = client.post("/api/auth/login", json={"token": token})
    assert res_login.status_code == 200
    login_data = res_login.json()
    assert login_data["status"] == "authenticated"
    csrf_token = login_data["csrf_token"]
    assert csrf_token is not None
    assert len(csrf_token) == 64

    # Verify cookies received
    assert "paladin_session" in client.cookies
    assert "paladin_csrf" in client.cookies

    # 3. GET /api/auth/session now returns authenticated
    res_session = client.get("/api/auth/session")
    assert res_session.status_code == 200
    assert res_session.json()["authenticated"] is True
    assert res_session.json()["csrf_token"] == csrf_token

    # 4. Safe GET endpoint (/api/capabilities) succeeds with session cookie (no CSRF header required)
    res_cap = client.get("/api/capabilities")
    assert res_cap.status_code == 200
    assert res_cap.json()["status"] == "OPERATIONAL"

    # 5. Mutating endpoint (/api/upload) without CSRF header fails with 403
    png_bytes = _create_test_png()
    res_up_nocsrf = client.post("/api/upload", files={"file": ("test.png", png_bytes, "image/png")})
    assert res_up_nocsrf.status_code == 403
    assert "CSRF validation failed" in res_up_nocsrf.json()["detail"]

    # 6. Mutating endpoint with wrong CSRF header fails with 403
    res_up_badcsrf = client.post(
        "/api/upload",
        files={"file": ("test.png", png_bytes, "image/png")},
        headers={"X-CSRF-Token": "invalid_csrf_token_value_00000000"},
    )
    assert res_up_badcsrf.status_code == 403

    # 7. Mutating endpoint with correct CSRF header succeeds!
    res_up_success = client.post(
        "/api/upload",
        files={"file": ("test.png", png_bytes, "image/png")},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert res_up_success.status_code == 200
    assert "session_id" in res_up_success.json()

    # 8. Logout invalidates session and clears cookies
    res_logout = client.post("/api/auth/logout")
    assert res_logout.status_code == 200
    assert res_logout.json()["status"] == "logged_out"

    # 9. Subsequent GET /api/capabilities now rejected with 401
    res_cap_after = client.get("/api/capabilities")
    assert res_cap_after.status_code == 401


def test_dual_auth_headless_client_bypass_csrf():
    """Headless API clients passing Bearer Authorization header do not require CSRF token."""
    token = "RemoteSecureSecretToken_1234567890abcdef!"
    settings = APISettings(
        mode="remote",
        api_token=token,
        allowed_origins=["http://localhost:5173"],
    )
    app = create_app(settings)
    client = TestClient(app)

    png_bytes = _create_test_png()
    res_up = client.post(
        "/api/upload",
        files={"file": ("test.png", png_bytes, "image/png")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_up.status_code == 200
    assert "session_id" in res_up.json()


def test_session_ttl_expiration():
    """An expired session cookie fails closed with 401."""
    token = "RemoteSecureSecretToken_1234567890abcdef!"
    settings = APISettings(
        mode="remote",
        api_token=token,
        allowed_origins=["http://localhost:5173"],
    )
    app = create_app(settings)
    client = TestClient(app)

    res_login = client.post("/api/auth/login", json={"token": token})
    assert res_login.status_code == 200
    session_id = client.cookies["paladin_session"]

    # Manually expire the session in the store
    session = default_auth_session_store.get_session(session_id)
    assert session is not None
    session.expires_at = time.time() - 10.0

    # Request with expired session cookie
    res_cap = client.get("/api/capabilities")
    assert res_cap.status_code == 401
    assert "Session expired" in res_cap.json()["detail"]


def test_https_secure_cookie_flag():
    """When connecting over HTTPS or behind TLS proxy, cookies include Secure flag."""
    token = "RemoteSecureSecretToken_1234567890abcdef!"
    settings = APISettings(
        mode="remote",
        api_token=token,
        allowed_origins=["https://paladinmedia.example.com"],
    )
    app = create_app(settings)
    client = TestClient(app, base_url="https://paladinmedia.example.com")

    res_login = client.post(
        "/api/auth/login",
        json={"token": token},
        headers={"X-Forwarded-Proto": "https"},
    )
    assert res_login.status_code == 200
    raw_cookie = res_login.headers.get("set-cookie", "")
    assert "secure" in raw_cookie.lower()
    assert "httponly" in raw_cookie.lower()
    assert "samesite=lax" in raw_cookie.lower()
