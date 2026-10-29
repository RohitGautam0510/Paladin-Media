"""Security and configuration infrastructure for PaladinMedia API.

Provides:
- Centralized deployment configuration (APISettings) with local vs remote modes.
- Bounded, TTL-enforcing in-memory session store (BoundedSessionStore).
- Constant-time Bearer token authentication dependency.
- Security response headers middleware.
- Error sanitization utilities to prevent internal path/secret disclosure.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("paladinmedia.security")


@dataclass
class APISettings:
    """Deployment and security settings derived from environment variables."""

    # Deployment mode: "local" (safe default) or "remote" / "production"
    mode: str = field(
        default_factory=lambda: os.environ.get("PALADINMEDIA_MODE", "local").lower()
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
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
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

    def __post_init__(self) -> None:
        if self.mode in ("production", "prod"):
            self.mode = "remote"

        if self.mode not in ("local", "remote"):
            raise ValueError(
                f"CONFIGURATION_ERROR: Invalid deployment mode '{self.mode}'. "
                "Supported modes are 'local' or 'remote'."
            )

        # In remote mode, docs default to disabled unless explicitly enabled
        if self.enable_docs is None:
            raw_docs = os.environ.get("PALADINMEDIA_ENABLE_DOCS")
            if raw_docs is not None:
                self.enable_docs = raw_docs.lower() in ("true", "1", "yes")
            else:
                self.enable_docs = self.mode == "local"

    def is_remote(self) -> bool:
        """Returns True if operating in remote/production deployment mode."""
        return self.mode == "remote"

    def is_auth_required(self) -> bool:
        """Determines whether Bearer authentication is enforced.

        Enforced if:
        1. Operating in remote mode (always enforced).
        2. Operating in local mode and PALADINMEDIA_API_TOKEN is explicitly configured.
        """
        if self.is_remote():
            return True
        return bool(self.api_token)

    def validate_startup(self) -> None:
        """Validates security invariants at application startup.

        Raises:
            RuntimeError: If remote mode is configured without an API token,
                          or with invalid mode, weak token, or with wildcard CORS origins.
        """
        if self.mode not in ("local", "remote"):
            raise RuntimeError(
                f"CONFIGURATION_ERROR: Invalid deployment mode '{self.mode}'. "
                "Supported modes are 'local' or 'remote'."
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


async def verify_auth(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[str]:
    """Dependency that enforces constant-time Bearer token authentication.

    Raises:
        HTTPException(401): If authentication is required and missing or invalid.
    """
    settings = get_api_settings()
    if not settings.is_auth_required():
        # Local development mode with no token configured
        return None

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
