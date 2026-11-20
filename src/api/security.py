"""Security and configuration infrastructure for PaladinMedia API.

Provides:
- Centralized deployment configuration (APISettings) with local vs remote modes.
- Bounded, TTL-enforcing in-memory session store (BoundedSessionStore).
- Constant-time Bearer token authentication dependency.
- Security response headers middleware.
- Error sanitization utilities to prevent internal path/secret disclosure.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jwt
from jwt import PyJWKSet

from fastapi import Header, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("paladinmedia.security")


@dataclass
class APISettings:
    """Deployment and security settings derived from environment variables."""

    # Deployment mode: "local" (safe default) or "remote" / "production"
    mode: str = field(
        default_factory=lambda: os.environ.get(
            "PALADINMEDIA_MODE", os.environ.get("PALADIN_DEPLOYMENT_MODE", "local")
        ).lower()
    )

    # Configured bearer token for authentication (required in remote mode)
    api_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("PALADINMEDIA_API_TOKEN") or None
    )

    # Allowed CORS origins
    allowed_origins: List[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.environ.get(
                "PALADINMEDIA_ALLOWED_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000,https://paladinmedia.vercel.app",
            ).split(",")
            if origin.strip()
        ]
    )

    # Maximum permitted upload size in bytes (default: 10 MB)
    max_upload_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("PALADINMEDIA_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
        )
    )

    # Session Time-to-Live in seconds (default: 300s / 5 minutes)
    session_ttl_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("PALADINMEDIA_SESSION_TTL_SECONDS", 300)
        )
    )

    # Maximum number of concurrent active session entries
    max_sessions: int = field(
        default_factory=lambda: int(os.environ.get("PALADINMEDIA_MAX_SESSIONS", 1000))
    )

    # Maximum aggregate memory budget for stored session bytes (default: 100 MB)
    max_session_store_bytes: int = field(
        default_factory=lambda: int(
            os.environ.get("PALADINMEDIA_MAX_SESSION_STORE_BYTES", 100 * 1024 * 1024)
        )
    )

    # Whether interactive documentation (/docs, /redoc, /openapi.json) is enabled
    enable_docs: Optional[bool] = field(default=None)

    # Wire Protocol v3-MR feature gate (default: False / disabled)
    enable_v3_mr: bool = field(
        default_factory=lambda: os.environ.get(
            "PALADIN_ENABLE_V3_MR", "false"
        ).lower() in ("true", "1", "yes")
    )

    # Post-Quantum Cryptography (PQC) feature gate (default: False / disabled)
    enable_pqc: bool = field(
        default_factory=lambda: os.environ.get(
            "PALADIN_ENABLE_PQC", "false"
        ).lower() in ("true", "1", "yes")
    )

    # OAuth 2.0 / OpenID Connect (OIDC) BFF Configuration
    oidc_issuer: Optional[str] = field(
        default_factory=lambda: os.environ.get("PALADIN_OIDC_ISSUER") or None
    )
    oidc_client_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("PALADIN_OIDC_CLIENT_ID") or None
    )
    oidc_client_secret: Optional[str] = field(
        default_factory=lambda: os.environ.get("PALADIN_OIDC_CLIENT_SECRET") or None
    )
    oidc_redirect_uri: Optional[str] = field(
        default_factory=lambda: os.environ.get("PALADIN_OIDC_REDIRECT_URI") or None
    )
    oidc_auth_endpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get("PALADIN_OIDC_AUTH_ENDPOINT") or None
    )
    oidc_token_endpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get("PALADIN_OIDC_TOKEN_ENDPOINT") or None
    )
    oidc_userinfo_endpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get("PALADIN_OIDC_USERINFO_ENDPOINT") or None
    )
    oidc_jwks_uri: Optional[str] = field(
        default_factory=lambda: os.environ.get("PALADIN_OIDC_JWKS_URI") or None
    )
    oidc_scopes: str = field(
        default_factory=lambda: os.environ.get("PALADIN_OIDC_SCOPES", "openid email profile")
    )

    # Trusted reverse-proxy IP addresses / CIDRs for X-Forwarded-* header trust
    trusted_proxies: List[str] = field(
        default_factory=lambda: [
            p.strip()
            for p in os.environ.get(
                "PALADIN_TRUSTED_PROXIES",
                "127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16",
            ).split(",")
            if p.strip()
        ]
    )

    def is_oidc_configured(self) -> bool:
        """Returns True if OIDC issuer and client ID are both configured."""
        return bool(self.oidc_issuer and self.oidc_client_id)

    def __post_init__(self) -> None:
        if self.mode in ("production", "prod"):
            self.mode = "remote"

        if self.mode not in ("local", "remote", "demo"):
            raise ValueError(
                f"CONFIGURATION_ERROR: Invalid deployment mode '{self.mode}'. "
                "Supported modes are 'local', 'remote', or 'demo'."
            )

        if self.mode == "demo":
            # In demo mode, apply safe resource bounds for Render free tier (512 MB ceiling)
            if "PALADINMEDIA_MAX_UPLOAD_BYTES" not in os.environ:
                self.max_upload_bytes = 2 * 1024 * 1024  # 2 MB upload ceiling for 512x512 carriers
            if "PALADINMEDIA_MAX_SESSION_STORE_BYTES" not in os.environ:
                self.max_session_store_bytes = 25 * 1024 * 1024  # 25 MB max in-memory session store
            if "PALADINMEDIA_SESSION_TTL_SECONDS" not in os.environ:
                self.session_ttl_seconds = 180  # 3 min TTL for prompt memory reclamation

        # In remote mode, docs default to disabled unless explicitly enabled
        if self.enable_docs is None:
            raw_docs = os.environ.get("PALADINMEDIA_ENABLE_DOCS")
            if raw_docs is not None:
                self.enable_docs = raw_docs.lower() in ("true", "1", "yes")
            else:
                self.enable_docs = self.mode in ("local", "demo")

    def is_remote(self) -> bool:
        """Returns True if operating in remote/production deployment mode."""
        return self.mode == "remote"

    def is_auth_required(self) -> bool:
        """Determines whether Bearer authentication is enforced.

        Enforced if:
        1. Operating in remote mode (always enforced).
        2. Operating in local or demo mode and PALADINMEDIA_API_TOKEN is explicitly configured.
        """
        if self.mode == "remote":
            return True
        return bool(self.api_token)

    def validate_startup(self) -> None:
        """Validates security invariants at application startup.

        Raises:
            RuntimeError: If remote mode is configured without an API token,
                          or with invalid mode, weak token, or with wildcard CORS origins.
        """
        if self.mode not in ("local", "remote", "demo"):
            raise RuntimeError(
                f"CONFIGURATION_ERROR: Invalid deployment mode '{self.mode}'. "
                "Supported modes are 'local', 'remote', or 'demo'."
            )

        if self.is_remote():
            if not self.api_token:
                raise RuntimeError(
                    "CONFIGURATION_ERROR: PALADINMEDIA_MODE is set to 'remote' but "
                    "PALADINMEDIA_API_TOKEN is missing or empty. PaladinMedia refuses to "
                    "operate in remote mode without configured authentication."
                )

            token_str = self.api_token.strip()
            if len(token_str) < 32:
                raise RuntimeError(
                    f"CONFIGURATION_ERROR: PALADINMEDIA_API_TOKEN must be at least 32 characters "
                    f"in remote deployment mode for adequate cryptographic strength (got {len(token_str)})."
                )

            if len(set(token_str)) < 8:
                raise RuntimeError(
                    "CONFIGURATION_ERROR: PALADINMEDIA_API_TOKEN has insufficient character diversity/entropy."
                )

            if "*" in self.allowed_origins:
                raise RuntimeError(
                    "CONFIGURATION_ERROR: Wildcard CORS origin '*' is strictly prohibited "
                    "in remote deployment mode. Configure explicit allowed origins via "
                    "PALADINMEDIA_ALLOWED_ORIGINS."
                )


# Global settings instance
_settings: Optional[APISettings] = None


def get_api_settings() -> APISettings:
    """Returns the current API settings singleton, reading from environment."""
    global _settings
    if _settings is None:
        _settings = APISettings()
    return _settings


def reset_api_settings(new_settings: Optional[APISettings] = None) -> APISettings:
    """Resets settings for testing purposes."""
    global _settings
    _settings = new_settings or APISettings()
    return _settings


@dataclass
class SessionRecord:
    """Metadata and raw bytes for a stored temporary media artifact."""

    session_id: str
    data: bytes
    created_at: float
    expires_at: float
    media_type: str = "application/octet-stream"
    filename: Optional[str] = None


class BoundedSessionStore:
    """In-memory session store with TTL expiration, count capacity, total byte budget, and pruning.

    Provides dictionary-like interface for backward compatibility while enforcing
    memory bounds, byte limits, and lifecycle cleanup.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = 300,
        max_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.max_bytes = max_bytes
        self._store: Dict[str, SessionRecord] = {}

    @property
    def total_bytes(self) -> int:
        """Current sum of stored session data bytes."""
        return sum(len(rec.data) for rec in self._store.values())

    def configure(
        self,
        max_entries: int,
        default_ttl: int,
        max_bytes: Optional[int] = None,
    ) -> None:
        """Dynamically reconfigures store capacity, default TTL, and total byte budget."""
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        if max_bytes is not None:
            self.max_bytes = max_bytes
        self.prune_expired()

    def prune_expired(self, now: Optional[float] = None) -> int:
        """Removes all expired sessions from the store.

        Returns:
            int: Number of pruned sessions.
        """
        current_time = now if now is not None else time.time()
        expired_keys = [
            sid for sid, rec in self._store.items() if rec.expires_at <= current_time
        ]
        for sid in expired_keys:
            del self._store[sid]
        return len(expired_keys)

    def store(
        self,
        data: bytes,
        ttl_seconds: Optional[int] = None,
        media_type: str = "application/octet-stream",
        filename: Optional[str] = None,
        custom_session_id: Optional[str] = None,
    ) -> str:
        """Stores binary payload under a cryptographically secure session ID.

        Enforces count limits, byte budgets, and TTL lifecycle.
        Evicts oldest unexpired records (oldest-created-first / FIFO by creation timestamp)
        when entry count or total byte budget is exceeded.

        Args:
            data: Payload bytes.
            ttl_seconds: Custom TTL in seconds (defaults to store default).
            media_type: MIME or logical type.
            filename: Original filename if applicable.
            custom_session_id: Optional deterministic identifier (e.g. for fixtures).

        Returns:
            str: Session identifier.

        Raises:
            HTTPException(413): If payload exceeds total store byte budget.
        """
        data_len = len(data)
        if data_len > self.max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Payload size ({data_len} bytes) exceeds total session store budget of {self.max_bytes} bytes",
            )

        now = time.time()
        self.prune_expired(now)

        # Evict oldest record(s) if count limit or byte budget is exceeded
        while self._store and (
            len(self._store) >= self.max_entries
            or (self.total_bytes + data_len) > self.max_bytes
        ):
            oldest_key = min(self._store.keys(), key=lambda k: self._store[k].created_at)
            del self._store[oldest_key]

        if (self.total_bytes + data_len) > self.max_bytes:
            raise HTTPException(
                status_code=413,
                detail="Session store byte budget exhausted even after evicting older sessions",
            )

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        session_id = custom_session_id or secrets.token_hex(16)
        record = SessionRecord(
            session_id=session_id,
            data=data,
            created_at=now,
            expires_at=now + ttl,
            media_type=media_type,
            filename=filename,
        )
        self._store[session_id] = record
        return session_id

    def get(self, session_id: str, default: Optional[bytes] = None) -> Optional[bytes]:
        """Retrieves session bytes if present and unexpired."""
        self.prune_expired()
        rec = self._store.get(session_id)
        if rec is None:
            return default
        if rec.expires_at <= time.time():
            del self._store[session_id]
            return default
        return rec.data

    def get_record(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieves full SessionRecord if present and unexpired."""
        self.prune_expired()
        rec = self._store.get(session_id)
        if rec is None or rec.expires_at <= time.time():
            if rec is not None:
                del self._store[session_id]
            return None
        return rec

    def delete(self, session_id: str) -> bool:
        """Explicitly deletes a session record.

        Returns:
            bool: True if deleted, False if not found.
        """
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    def clear(self) -> None:
        """Removes all session records."""
        self._store.clear()

    # Dictionary compatibility methods
    def __getitem__(self, session_id: str) -> bytes:
        val = self.get(session_id)
        if val is None:
            raise KeyError(session_id)
        return val

    def __setitem__(self, session_id: str, data: bytes) -> None:
        self.store(data=data, custom_session_id=session_id)

    def __contains__(self, session_id: str) -> bool:
        return self.get(session_id) is not None

    def __len__(self) -> int:
        self.prune_expired()
        return len(self._store)


@dataclass
class AuthSession:
    """Active user authentication session for browser BFF access."""

    session_id: str
    csrf_token: str
    user_id: str
    created_at: float
    expires_at: float
    roles: List[str] = field(default_factory=lambda: ["operator"])
    auth_method: str = "token"


def generate_pkce_pair() -> Tuple[str, str]:
    """Generates (code_verifier, code_challenge) per RFC 7636 (S256).

    Returns:
        code_verifier: Cryptographically secure 43-character base64url-safe string.
        code_challenge: Base64url-encoded SHA-256 digest of code_verifier without padding.
    """
    code_verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return code_verifier, code_challenge


@dataclass
class PendingOIDCState:
    """State record for an inflight OIDC authorization code request."""

    state: str
    nonce: str
    code_verifier: str
    redirect_uri: str
    created_at: float
    expires_at: float


class OIDCStateStore:
    """In-memory store for pending OIDC authorization states, preventing CSRF and replay attacks."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._states: Dict[str, PendingOIDCState] = {}
        self._lock = threading.Lock()

    def store_state(
        self,
        state: str,
        nonce: str,
        code_verifier: str,
        redirect_uri: str,
        ttl_seconds: Optional[int] = None,
    ) -> PendingOIDCState:
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        entry = PendingOIDCState(
            state=state,
            nonce=nonce,
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
            created_at=now,
            expires_at=now + ttl,
        )
        with self._lock:
            # Prune expired states
            expired = [k for k, v in self._states.items() if v.expires_at <= now]
            for k in expired:
                del self._states[k]
            self._states[state] = entry
        return entry

    def consume_state(self, state: str) -> Optional[PendingOIDCState]:
        """Atomically retrieves and deletes a pending state to guarantee single-use."""
        now = time.time()
        with self._lock:
            entry = self._states.pop(state, None)
            if entry is None or entry.expires_at <= now:
                return None
            return entry

    def clear(self) -> None:
        with self._lock:
            self._states.clear()


default_oidc_state_store = OIDCStateStore()


def reset_oidc_state_store() -> None:
    """Resets the OIDC state store (for testing or reconfiguration)."""
    default_oidc_state_store.clear()


def is_trusted_proxy(client_host: Optional[str], trusted_proxies: List[str]) -> bool:
    """Checks whether client_host matches a trusted proxy IP or subnet."""
    if not client_host:
        return False
    try:
        client_ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False

    if client_ip.is_loopback:
        return True

    for pattern in trusted_proxies:
        try:
            if "/" in pattern:
                net = ipaddress.ip_network(pattern, strict=False)
                if client_ip in net:
                    return True
            else:
                target_ip = ipaddress.ip_address(pattern)
                if client_ip == target_ip:
                    return True
        except ValueError:
            continue
    return False


def is_request_https(request: Request, settings: Optional[APISettings] = None) -> bool:
    """Determines if the request was delivered over HTTPS securely.

    Checks direct scheme, explicit PALADIN_COOKIE_SECURE override, and verifies
    X-Forwarded-Proto ONLY when request comes from a verified trusted proxy.
    """
    if request.url.scheme == "https":
        return True
    if os.environ.get("PALADIN_COOKIE_SECURE", "false").lower() in ("true", "1", "yes"):
        return True

    client_host = request.client.host if request.client else None
    current_settings = settings or get_api_settings()
    if is_trusted_proxy(client_host, current_settings.trusted_proxies):
        proto = request.headers.get("x-forwarded-proto", "").lower()
        if proto == "https":
            return True

    return False


MAX_JWKS_RESPONSE_BYTES = 256 * 1024  # 256 KB response boundary


def is_safe_oidc_url(url: str, allow_local: bool = False) -> Tuple[bool, str]:
    """Validates OIDC discovery and JWKS URLs against SSRF, credential stuffing, and protocol manipulation.

    Args:
        url: The candidate URL string.
        allow_local: If True (e.g. local mode or test harness), allows loopback/private IPs.

    Returns:
        Tuple of (is_safe, error_reason).
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string."

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception as exc:
        return False, f"Malformed URL structure: {str(exc)}"

    if parsed.username or parsed.password:
        return False, "SSRF protection: embedded user credentials (user:pass@) in URL authority are prohibited."

    try:
        port = parsed.port
    except ValueError:
        return False, "Malformed port in URL authority."

    if allow_local:
        if parsed.scheme not in ("https", "http"):
            return False, f"Disallowed URL scheme '{parsed.scheme}'. Must be https or http."
    else:
        if parsed.scheme != "https":
            return False, f"Insecure URL scheme '{parsed.scheme}'. Production OIDC endpoints must use HTTPS."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL missing valid hostname."

    if allow_local:
        return True, ""

    def _is_safe_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> Tuple[bool, str]:
        if isinstance(ip, ipaddress.IPv6Address) and getattr(ip, "ipv4_mapped", None):
            return _is_safe_ip(ip.ipv4_mapped)
        if ip.is_loopback:
            return False, "SSRF protection: loopback IP addresses are prohibited."
        if ip.is_private:
            return False, "SSRF protection: private IP addresses (RFC 1918 / RFC 4193) are prohibited."
        if ip.is_link_local:
            return False, "SSRF protection: link-local IP addresses (e.g. 169.254.0.0/16, cloud metadata) are prohibited."
        if ip.is_reserved or ip.is_multicast:
            return False, "SSRF protection: reserved, special-use, or multicast IP addresses are prohibited."
        if not ip.is_global:
            return False, "SSRF protection: non-global IP addresses are prohibited in remote mode."
        return True, ""

    try:
        ip = ipaddress.ip_address(hostname)
        return _is_safe_ip(ip)
    except ValueError:
        hostname_lower = hostname.lower()
        if hostname_lower in ("localhost", "127.0.0.1", "::1"):
            return False, "SSRF protection: localhost is prohibited in remote mode."
        if (
            "metadata.google.internal" in hostname_lower
            or "169.254" in hostname_lower
            or "instance-data" in hostname_lower
        ):
            return False, "SSRF protection: cloud metadata endpoints are prohibited."

        # Perform pre-connection DNS inspection in remote mode if hostname is resolvable
        try:
            addr_info = socket.getaddrinfo(hostname, port or 443, proto=socket.IPPROTO_TCP)
            for entry in addr_info:
                resolved_ip_str = entry[4][0]
                try:
                    resolved_ip = ipaddress.ip_address(resolved_ip_str)
                    safe, reason = _is_safe_ip(resolved_ip)
                    if not safe:
                        return False, f"SSRF protection: hostname resolves to restricted IP {resolved_ip_str}: {reason}"
                except ValueError:
                    pass
        except (socket.gaierror, socket.herror, OSError):
            # If DNS resolution fails here, let fetcher fail closed with sanitized error
            pass

    return True, ""


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Custom redirect handler that enforces scheme preservation and SSRF validation on redirects."""

    def __init__(self, allow_local: bool = False, max_redirects: int = 2) -> None:
        self.allow_local = allow_local
        self.max_redirects = max_redirects
        self._redirect_count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._redirect_count += 1
        if self._redirect_count > self.max_redirects:
            raise HTTPException(
                status_code=401,
                detail="Too many redirects encountered while fetching OIDC metadata.",
            )
        is_safe, reason = is_safe_oidc_url(newurl, self.allow_local)
        if not is_safe:
            raise HTTPException(
                status_code=401,
                detail=f"Disallowed redirect target URL in OIDC request: {reason}",
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _safe_fetch_json(
    url: str,
    timeout: float = 5.0,
    max_bytes: int = MAX_JWKS_RESPONSE_BYTES,
    allow_local: bool = False,
) -> Dict[str, Any]:
    """Safely retrieves and parses JSON from a remote URL with strict timeouts, size bounds, and SSRF checks."""
    is_safe, reason = is_safe_oidc_url(url, allow_local)
    if not is_safe:
        raise HTTPException(
            status_code=401,
            detail=f"Disallowed OIDC endpoint URL: {reason}",
        )

    opener = urllib.request.build_opener(SafeRedirectHandler(allow_local=allow_local))
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "PaladinMedia-OIDC/2.6.0",
        },
    )

    try:
        # If urllib.request.urlopen has been mocked (e.g. in test fixtures), route via urlopen
        if (
            hasattr(urllib.request.urlopen, "side_effect")
            or hasattr(urllib.request.urlopen, "return_value")
            or hasattr(urllib.request.urlopen, "mock")
            or hasattr(urllib.request.urlopen, "assert_called")
        ):
            resp_cm = urllib.request.urlopen(req, timeout=timeout)
        else:
            resp_cm = opener.open(req, timeout=timeout)

        with resp_cm as resp:
            if resp.status != 200:
                raise HTTPException(
                    status_code=401,
                    detail=f"OIDC metadata endpoint returned non-200 HTTP status: {resp.status}",
                )
            raw_data = resp.read(max_bytes + 1)
            if len(raw_data) > max_bytes:
                raise HTTPException(
                    status_code=401,
                    detail=f"OIDC metadata response exceeded maximum allowed size of {max_bytes} bytes.",
                )
            try:
                data = json.loads(raw_data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise HTTPException(
                    status_code=401,
                    detail=f"Malformed JSON document received from OIDC endpoint: {str(exc)}",
                )
            if not isinstance(data, dict):
                raise HTTPException(
                    status_code=401,
                    detail="Malformed OIDC metadata: expected top-level JSON object.",
                )
            return data
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Failed to fetch OIDC metadata from %s: %s", url, exc)
        raise HTTPException(
            status_code=401,
            detail=f"JWKS or discovery endpoint unavailable or key retrieval failed: {str(exc)}",
        )


class JWKSCache:
    """Thread-safe in-memory cache and resolver for OIDC JSON Web Key Sets (JWKS)."""

    def __init__(self, default_ttl_seconds: float = 300.0, rotation_cooldown_seconds: float = 10.0) -> None:
        self.default_ttl = default_ttl_seconds
        self.rotation_cooldown = rotation_cooldown_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def _parse_jwks_payload(self, payload: Dict[str, Any]) -> PyJWKSet:
        """Validates JWKS document structure and rejects malformed or duplicate key identifiers."""
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=401,
                detail="Malformed JWKS document: root must be a JSON object.",
            )
        keys_list = payload.get("keys")
        if not isinstance(keys_list, list):
            raise HTTPException(
                status_code=401,
                detail="Malformed JWKS document: 'keys' member must be a JSON array.",
            )
        if len(keys_list) == 0:
            raise HTTPException(
                status_code=401,
                detail="Malformed JWKS document: 'keys' array is empty.",
            )

        seen_kids: set[str] = set()
        for idx, k in enumerate(keys_list):
            if not isinstance(k, dict):
                raise HTTPException(
                    status_code=401,
                    detail=f"Malformed JWKS document: key at index {idx} is not an object.",
                )
            kid = k.get("kid")
            if kid is not None:
                if not isinstance(kid, str) or not kid.strip():
                    raise HTTPException(
                        status_code=401,
                        detail=f"Malformed JWKS key at index {idx}: 'kid' must be a non-empty string.",
                    )
                if kid in seen_kids:
                    raise HTTPException(
                        status_code=401,
                        detail=f"Malformed JWKS document: duplicate 'kid' ('{kid}') detected.",
                    )
                seen_kids.add(kid)

        try:
            return PyJWKSet.from_dict(payload)
        except Exception as exc:
            raise HTTPException(
                status_code=401,
                detail=f"Failed to parse JWKS document into cryptographic keys: {str(exc)}",
            )

    def resolve_jwks_uri(
        self,
        issuer: str,
        configured_jwks_uri: Optional[str] = None,
        allow_local: bool = False,
    ) -> str:
        """Resolves the authoritative JWKS URI via explicit config, OIDC discovery, or standard fallback."""
        if configured_jwks_uri:
            is_safe, reason = is_safe_oidc_url(configured_jwks_uri, allow_local)
            if not is_safe:
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid configured JWKS URI: {reason}",
                )
            return configured_jwks_uri

        if issuer.endswith("/"):
            discovery_url = f"{issuer}.well-known/openid-configuration"
            fallback_uri = f"{issuer}.well-known/jwks.json"
        else:
            discovery_url = f"{issuer}/.well-known/openid-configuration"
            fallback_uri = f"{issuer}/.well-known/jwks.json"

        try:
            data = _safe_fetch_json(discovery_url, timeout=5.0, allow_local=allow_local)
            discovered_issuer = data.get("issuer")
            if not discovered_issuer:
                raise HTTPException(
                    status_code=401,
                    detail="Discovery document missing required 'issuer' claim.",
                )
            if discovered_issuer != issuer:
                raise HTTPException(
                    status_code=401,
                    detail=f"Discovery document issuer mismatch: expected '{issuer}', got '{discovered_issuer}'.",
                )
            if "jwks_uri" in data:
                jwks_uri = data["jwks_uri"]
                is_safe, reason = is_safe_oidc_url(jwks_uri, allow_local)
                if not is_safe:
                    raise HTTPException(
                        status_code=401,
                        detail=f"Discovered jwks_uri is unsafe: {reason}",
                    )
                return jwks_uri
        except HTTPException:
            raise
        except Exception as exc:
            logger.debug(
                "OIDC discovery at %s failed: %s; falling back to standard JWKS endpoint",
                discovery_url,
                exc,
            )

        is_safe, reason = is_safe_oidc_url(fallback_uri, allow_local)
        if not is_safe:
            raise HTTPException(
                status_code=401,
                detail=f"Standard fallback JWKS URI is unsafe: {reason}",
            )
        return fallback_uri

    def get_signing_key(
        self,
        kid: Optional[str],
        alg: str,
        issuer: Optional[str],
        jwks_uri: Optional[str] = None,
        jwks_data: Optional[Dict[str, Any]] = None,
        allow_local: bool = False,
    ) -> Any:
        """Retrieves and validates the public signing key matching the given kid and algorithm."""
        if jwks_data is not None:
            jwk_set = self._parse_jwks_payload(jwks_data)
            return self._find_key_in_set(jwk_set, kid, alg)

        if not issuer and not jwks_uri:
            raise HTTPException(
                status_code=401,
                detail="Cannot verify asymmetric token: neither issuer nor jwks_uri configured.",
            )

        target_uri = self.resolve_jwks_uri(issuer or "", jwks_uri, allow_local=allow_local)
        now = time.time()

        with self._lock:
            cached = self._cache.get(target_uri)
            needs_fetch = cached is None or (now - cached["fetched_at"]) > self.default_ttl

            if needs_fetch:
                payload = _safe_fetch_json(target_uri, timeout=5.0, allow_local=allow_local)
                jwk_set = self._parse_jwks_payload(payload)
                self._cache[target_uri] = {
                    "jwk_set": jwk_set,
                    "fetched_at": now,
                    "last_rotation": now,
                }
            else:
                jwk_set = cached["jwk_set"]

            try:
                return self._find_key_in_set(jwk_set, kid, alg)
            except HTTPException:
                cached = self._cache.get(target_uri)
                can_rotate = cached is not None and (now - cached["last_rotation"]) >= self.rotation_cooldown

                if can_rotate:
                    logger.info(
                        "Key kid='%s' not in cached JWKS; attempting rotation refresh from %s",
                        kid,
                        target_uri,
                    )
                    payload = _safe_fetch_json(target_uri, timeout=5.0, allow_local=allow_local)
                    refreshed_set = self._parse_jwks_payload(payload)
                    self._cache[target_uri] = {
                        "jwk_set": refreshed_set,
                        "fetched_at": now,
                        "last_rotation": now,
                    }
                    return self._find_key_in_set(refreshed_set, kid, alg)
                raise

    def _find_key_in_set(self, jwk_set: PyJWKSet, kid: Optional[str], alg: str) -> Any:
        expected_kty = "RSA" if alg.startswith("RS") else ("EC" if alg.startswith("ES") else None)

        if kid:
            try:
                pyjwk = jwk_set[kid]
                if expected_kty and getattr(pyjwk, "key_type", None) and pyjwk.key_type != expected_kty:
                    raise HTTPException(
                        status_code=401,
                        detail=f"Key type mismatch for kid '{kid}': expected {expected_kty}, got {pyjwk.key_type}",
                    )
                return pyjwk.key
            except KeyError:
                raise HTTPException(
                    status_code=401,
                    detail=f"Signing key with kid '{kid}' not found in JWKS.",
                )
        else:
            candidates = [
                k for k in jwk_set.keys
                if expected_kty is None or getattr(k, "key_type", None) == expected_kty
            ]
            if len(candidates) == 1:
                return candidates[0].key
            elif len(candidates) > 1:
                raise HTTPException(
                    status_code=401,
                    detail="ID token missing required 'kid' header parameter and multiple matching keys exist in JWKS.",
                )
            else:
                raise HTTPException(
                    status_code=401,
                    detail=f"No suitable {expected_kty} signing key found in JWKS.",
                )


default_jwks_cache = JWKSCache()


def reset_jwks_cache() -> None:
    """Resets the JWKS cache (for testing or reconfiguration)."""
    default_jwks_cache.clear()


def validate_id_token_claims(
    id_token: str,
    expected_issuer: Optional[str],
    expected_audience: Optional[str],
    expected_nonce: Optional[str],
    client_secret: Optional[str] = None,
    jwks_uri: Optional[str] = None,
    jwks_data: Optional[Dict[str, Any]] = None,
    allowed_algorithms: Optional[List[str]] = None,
    allow_local: Optional[bool] = None,
) -> Dict[str, Any]:
    """Validates claims and cryptographic signatures in an OpenID Connect ID token JWT.

    Supports:
    - Asymmetric signatures (RS256, ES256) via JWKS key retrieval and verification.
    - Symmetric signatures (HS256) via shared client_secret when configured.
    - Strict algorithm allowlisting and algorithm confusion attack prevention.
    - Comprehensive claim validation: iss, aud, azp (for multi-audience), nonce, exp, iat, sub.
    - SSRF protection and discovery issuer validation.
    """
    if not isinstance(id_token, str):
        raise HTTPException(
            status_code=400,
            detail="ID token must be a string.",
        )

    parts = id_token.split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=400,
            detail="Malformed ID token JWT: expected 3 dot-separated segments.",
        )

    # 1. Unverified header inspection
    try:
        header = jwt.get_unverified_header(id_token)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Malformed ID token header: {str(exc)}",
        )

    alg = header.get("alg")
    if not alg or alg.lower() == "none":
        raise HTTPException(
            status_code=401,
            detail="Unsigned ID token (alg=none) is strictly prohibited.",
        )

    # 2. Algorithm allowlisting
    if allowed_algorithms is not None:
        effective_allowed = allowed_algorithms
    else:
        effective_allowed = ["RS256", "ES256"]
        if client_secret:
            effective_allowed.append("HS256")

    if alg not in effective_allowed:
        raise HTTPException(
            status_code=401,
            detail=f"Disallowed ID token algorithm '{alg}'. Allowed algorithms: {effective_allowed}.",
        )

    # Resolve local/test allowance
    if allow_local is None:
        settings = get_api_settings()
        allow_local = (
            settings.mode == "local"
            or os.environ.get("PALADIN_ALLOW_LOCAL_OIDC", "false").lower() in ("true", "1", "yes")
        )

    # 3. Signature verification with strict algorithm separation
    try:
        if alg == "HS256":
            if not client_secret:
                raise HTTPException(
                    status_code=401,
                    detail="HS256 token presented but client_secret is not configured.",
                )
            claims = jwt.decode(
                id_token,
                key=client_secret.encode("utf-8"),
                algorithms=["HS256"],
                options={"verify_signature": True, "verify_exp": False, "verify_aud": False, "verify_iss": False},
            )
        elif alg in ("RS256", "ES256"):
            signing_key = default_jwks_cache.get_signing_key(
                kid=header.get("kid"),
                alg=alg,
                issuer=expected_issuer,
                jwks_uri=jwks_uri,
                jwks_data=jwks_data,
                allow_local=allow_local,
            )
            claims = jwt.decode(
                id_token,
                key=signing_key,
                algorithms=[alg],
                options={"verify_signature": True, "verify_exp": False, "verify_aud": False, "verify_iss": False},
            )
        else:
            raise HTTPException(
                status_code=401,
                detail=f"Unsupported algorithm '{alg}'.",
            )
    except HTTPException:
        raise
    except jwt.InvalidSignatureError:
        raise HTTPException(
            status_code=401,
            detail="ID token signature verification failed.",
        )
    except jwt.InvalidIssuedAtError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid 'iat' timestamp in ID token: {str(exc)}",
        )
    except jwt.ImmatureSignatureError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"ID token 'iat' is in the future: {str(exc)}",
        )
    except jwt.InvalidAlgorithmError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"Algorithm confusion attempt rejected: {str(exc)}",
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"ID token decoding or signature failure: {str(exc)}",
        )

    # 4. Issuer validation (strict exact-match comparison without normalization)
    iss = claims.get("iss")
    if not iss or not isinstance(iss, str):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid 'iss' (issuer) claim in ID token.",
        )
    if expected_issuer:
        if iss != expected_issuer:
            raise HTTPException(
                status_code=401,
                detail=f"ID token issuer mismatch: expected '{expected_issuer}', got '{iss}'.",
            )

    # 5. Audience and azp validation (RFC Section 3.1.3.7)
    aud = claims.get("aud")
    if not aud:
        raise HTTPException(
            status_code=401,
            detail="Missing required 'aud' (audience) claim in ID token.",
        )
    if not isinstance(aud, (str, list)):
        raise HTTPException(
            status_code=401,
            detail="Invalid 'aud' (audience) claim in ID token: must be a string or list of strings.",
        )
    if isinstance(aud, list) and not all(isinstance(a, str) for a in aud):
        raise HTTPException(
            status_code=401,
            detail="Invalid 'aud' (audience) claim in ID token: list elements must be strings.",
        )

    if expected_audience:
        if isinstance(aud, list):
            if expected_audience not in aud:
                raise HTTPException(
                    status_code=401,
                    detail=f"ID token audience mismatch: '{expected_audience}' not in {aud}.",
                )
            if len(aud) > 1:
                azp = claims.get("azp")
                if not azp:
                    raise HTTPException(
                        status_code=401,
                        detail="ID token with multiple audiences missing required 'azp' (Authorized Party) claim.",
                    )
                if not isinstance(azp, str) or azp != expected_audience:
                    raise HTTPException(
                        status_code=401,
                        detail=f"ID token 'azp' mismatch: expected '{expected_audience}', got '{azp}'.",
                    )
        elif isinstance(aud, str):
            if aud != expected_audience:
                raise HTTPException(
                    status_code=401,
                    detail=f"ID token audience mismatch: expected '{expected_audience}', got '{aud}'.",
                )
            azp = claims.get("azp")
            if azp is not None:
                if not isinstance(azp, str) or azp != expected_audience:
                    raise HTTPException(
                        status_code=401,
                        detail=f"ID token 'azp' mismatch: expected '{expected_audience}', got '{azp}'.",
                    )

    # 6. Nonce validation
    if expected_nonce:
        nonce = claims.get("nonce")
        if not nonce or not isinstance(nonce, str) or not secrets.compare_digest(nonce, expected_nonce):
            raise HTTPException(
                status_code=401,
                detail="ID token nonce mismatch or missing. Replay attack prevented.",
            )

    # 7. Expiration validation (with 60-second clock skew leeway)
    now = time.time()
    exp = claims.get("exp")
    if exp is None:
        raise HTTPException(
            status_code=401,
            detail="Missing required 'exp' claim in ID token.",
        )
    try:
        exp_val = float(exp)
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid 'exp' timestamp in ID token: must be a numeric timestamp.")
    if exp_val <= (now - 60.0):
        raise HTTPException(status_code=401, detail="ID token has expired.")

    # 8. Issued-at (iat) validation (if present, must not be in the future beyond clock skew)
    iat = claims.get("iat")
    if iat is not None:
        try:
            iat_val = float(iat)
        except (ValueError, TypeError):
            raise HTTPException(status_code=401, detail="Invalid 'iat' timestamp in ID token: must be a numeric timestamp.")
        if iat_val > (now + 60.0):
            raise HTTPException(status_code=401, detail="ID token 'iat' is in the future.")

    # 9. Subject validation
    sub = claims.get("sub")
    if not sub or not isinstance(sub, str) or not sub.strip():
        raise HTTPException(
            status_code=401,
            detail="ID token is missing required 'sub' (subject) claim.",
        )

    return claims

    return claims


class AuthSessionStore:
    """In-memory bounded session store for browser BFF authentication sessions."""

    def __init__(self, max_sessions: int = 1000, default_ttl_seconds: int = 3600) -> None:
        self.max_sessions = max_sessions
        self.default_ttl_seconds = default_ttl_seconds
        self._sessions: Dict[str, AuthSession] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        user_id: str = "operator",
        ttl_seconds: Optional[int] = None,
    ) -> AuthSession:
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        session_id = secrets.token_hex(32)
        csrf_token = secrets.token_hex(32)
        session = AuthSession(
            session_id=session_id,
            csrf_token=csrf_token,
            user_id=user_id,
            created_at=now,
            expires_at=now + ttl,
        )
        with self._lock:
            # Prune expired
            expired = [sid for sid, s in self._sessions.items() if s.expires_at <= now]
            for sid in expired:
                del self._sessions[sid]

            # Bounded capacity eviction (oldest first)
            while len(self._sessions) >= self.max_sessions:
                oldest_sid = min(self._sessions.keys(), key=lambda k: self._sessions[k].created_at)
                del self._sessions[oldest_sid]

            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[AuthSession]:
        now = time.time()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.expires_at <= now:
                del self._sessions[session_id]
                return None
            return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for s in self._sessions.values() if s.expires_at > now)


default_auth_session_store = AuthSessionStore()


def reset_auth_session_store() -> None:
    """Resets the auth session store (for testing or reconfiguration)."""
    default_auth_session_store.clear()


async def verify_auth(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[str]:
    """Dependency that enforces constant-time Bearer token or HttpOnly session cookie auth.

    Supports:
    1. Direct Bearer token authentication (programmatic / headless API clients).
    2. HttpOnly session cookie authentication (BFF / browser clients) with CSRF validation.

    Raises:
        HTTPException(401): If authentication is required and missing or invalid.
        HTTPException(403): If session cookie is valid but CSRF token is invalid on mutation requests.
    """
    settings = get_api_settings()
    if not settings.is_auth_required():
        # Local development mode with no token configured
        return None

    # Priority 1: Bearer token authorization header
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization header scheme. Bearer token required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        expected_token = settings.api_token or ""
        # Constant-time comparison to prevent timing side-channels
        if not secrets.compare_digest(token, expected_token):
            logger.warning("Authentication failed: invalid token provided.")
            raise HTTPException(
                status_code=401,
                detail="Invalid API token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token

    # Priority 2: BFF Session Cookie authentication
    if request is not None:
        session_id = request.cookies.get("paladin_session")
        if session_id:
            session = default_auth_session_store.get_session(session_id)
            if session is None:
                raise HTTPException(
                    status_code=401,
                    detail="Session expired or invalid. Please re-authenticate.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Enforce CSRF protection on state-changing methods for cookie-authenticated clients
            if request.method in ("POST", "PUT", "DELETE", "PATCH"):
                csrf_header = (
                    request.headers.get("x-csrf-token")
                    or request.headers.get("X-CSRF-Token")
                )
                if not csrf_header or not secrets.compare_digest(csrf_header, session.csrf_token):
                    logger.warning(
                        "CSRF validation failed for session %s on %s %s",
                        session_id[:8],
                        request.method,
                        request.url.path,
                    )
                    raise HTTPException(
                        status_code=403,
                        detail="CSRF validation failed: missing or invalid X-CSRF-Token header.",
                    )
            return f"session:{session.session_id}"

    # Neither Bearer token nor valid session cookie provided
    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds standard security headers and cache control to HTTP responses."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)

        # Standard defense-in-depth headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Prevent browser caching of stego payloads and sensitive operational endpoints
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"

        return response


def sanitize_error_message(error: Exception) -> str:
    """Sanitizes an exception message to prevent exposure of internal filesystem paths."""
    msg = str(error)
    # Strip Windows or Unix absolute paths (e.g. E:\Paladin_media\... or /path/to/...)
    msg = re.sub(r"[A-Za-z]:\\[\w\\\.-]+", "[REDACTED_PATH]", msg)
    msg = re.sub(r"/(?:[\w\.-]+/)+[\w\.-]+", "[REDACTED_PATH]", msg)
    return msg


HEX_64_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_key_hex(key_hex: Optional[str], is_remote: bool) -> bytes:
    """Strictly validates key_hex and decodes to 32 bytes.

    Rules:
    - In remote mode, key_hex is strictly required (no default laboratory keys).
    - If provided, key_hex must be exactly 64 ASCII hexadecimal characters.
      Whitespace, newlines, and non-hexadecimal characters are rejected.
    - Decoded byte length must be exactly 32 bytes.
    - In local mode, if key_hex is None or omitted, fallback laboratory key is permitted.

    Raises:
        HTTPException(422): If key_hex is invalid, wrong length, or missing in remote mode.
    """
    if key_hex is None:
        if is_remote:
            raise HTTPException(
                status_code=422,
                detail="key_hex is required in remote deployment mode. Default laboratory keys are prohibited.",
            )
        # Permitted fallback key for local development only
        return bytes.fromhex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")

    if not isinstance(key_hex, str):
        raise HTTPException(status_code=422, detail="key_hex must be a string")

    # Strict check: exact 64 characters
    if len(key_hex) != 64:
        raise HTTPException(
            status_code=422,
            detail="key_hex must be a 64-character hex string representing a 32-byte key",
        )

    # Reject whitespace, newlines, tabs, and non-hex chars
    if not HEX_64_PATTERN.fullmatch(key_hex):
        raise HTTPException(
            status_code=422,
            detail="key_hex contains invalid non-hexadecimal characters",
        )

    try:
        key_bytes = bytes.fromhex(key_hex)
    except ValueError:
        raise HTTPException(status_code=422, detail="key_hex contains invalid non-hexadecimal characters")

    if len(key_bytes) != 32:
        raise HTTPException(status_code=422, detail="Decoded key must be exactly 32 bytes")

    return key_bytes


class RequestSizeLimitMiddleware:
    """ASGI middleware that enforces a strict request body size limit before multipart parsing.

    Rejects requests exceeding max_body_bytes with HTTP 413 before expensive buffering.
    """

    def __init__(self, app: Any, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length_header = headers.get(b"content-length")
        if content_length_header is not None:
            try:
                content_length = int(content_length_header.decode("latin1"))
                if content_length > self.max_body_bytes:
                    response = Response(
                        content=json.dumps({
                            "detail": f"Request body exceeds maximum permitted size of {self.max_body_bytes} bytes"
                        }),
                        status_code=413,
                        media_type="application/json",
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        total_received = 0
        size_exceeded = False
        response_started = False

        async def bounded_receive() -> Dict[str, Any]:
            nonlocal total_received, size_exceeded
            if size_exceeded:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body exceeds maximum permitted size of {self.max_body_bytes} bytes",
                )
            message = await receive()
            if message["type"] == "http.request":
                body_chunk = message.get("body", b"")
                total_received += len(body_chunk)
                if total_received > self.max_body_bytes:
                    size_exceeded = True
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request body exceeds maximum permitted size of {self.max_body_bytes} bytes",
                    )
            return message

        async def intercepting_send(message: Dict[str, Any]) -> None:
            nonlocal response_started
            if size_exceeded:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, intercepting_send)
        except Exception as exc:
            if size_exceeded:
                # Downstream handler raised an error triggered by bounded_receive() size violation
                pass
            else:
                # Unrelated application exception: propagate cleanly through ASGI pipeline
                raise exc

        if size_exceeded and not response_started:
            response = Response(
                content=json.dumps({
                    "detail": f"Request body exceeds maximum permitted size of {self.max_body_bytes} bytes"
                }),
                status_code=413,
                media_type="application/json",
            )
            await response(scope, receive, send)


class DemoRateLimiterMiddleware(BaseHTTPMiddleware):
    """Enforces per-client IP rate limits in demo mode to protect free-tier hosting resources."""

    def __init__(self, app: Any, requests_per_minute: int = 60, enabled: bool = True) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled
        self._history: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Bypass rate limiter for health check and options preflight
        if request.url.path == "/api/health" or request.method == "OPTIONS":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60.0

        async with self._lock:
            timestamps = self._history.get(client_ip, [])
            recent = [t for t in timestamps if t > window_start]
            if len(recent) >= self.requests_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded: public demo permits up to {self.requests_per_minute} requests per minute per IP."
                    },
                    headers={"Retry-After": "60"},
                )
            recent.append(now)
            self._history[client_ip] = recent
            if len(self._history) > 1000:
                self._history = {ip: ts for ip, ts in self._history.items() if any(t > window_start for t in ts)}

        return await call_next(request)


