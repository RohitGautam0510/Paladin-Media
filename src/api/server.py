"""FastAPI Backend Server for PaladinMedia Deployment.

Provides:
- Qualified deployment configuration endpoint (/api/config).
- Public health probe (/api/health).
- Streaming media pipeline execution (/api/stream) via MediaEventAdapter.
- Direct operational embedding and extraction endpoints (/api/embed, /api/extract).
- Upload and session lifecycle management (/api/upload, /api/session/{session_id}).
- Enforces lossless PNG carrier policy, dimension constraints, upload bounds, and authentication.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import base64
import hashlib
import io
import json
import logging
import os
import secrets
import struct
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from src.api.diagnostics import diagnostics_store
from src.api.event_adapter import (
    DEFAULT_MAX_SSE_EVENT_BYTES,
    MediaEventAdapter,
    pil_to_base64_data_url,
    serialize_sse_event,
)
from src.api.security import (
    APISettings,
    AuthSession,
    BoundedSessionStore,
    DemoRateLimiterMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    default_auth_session_store,
    default_oidc_state_store,
    generate_pkce_pair,
    get_api_settings,
    is_request_https,
    reset_api_settings,
    reset_auth_session_store,
    sanitize_error_message,
    validate_id_token_claims,
    validate_key_hex,
    verify_auth,
)
from src.api.telemetry import SustainabilityTelemetryTracker, telemetry_tracker
from src.carrier.capacity import CapacityCalculator, PayloadCapacityExceededError
from src.carrier.header import CarrierHeaderError
from src.carrier.modulation import create_modulation_engine
from src.core.exceptions import (
    AuthenticationError,
    ECCDecodeError,
    UnsupportedCarrierFormatError,
)
from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization.serializer import MediaType, SecretPayload

logger = logging.getLogger("paladinmedia.api")
logging.basicConfig(level=logging.INFO)

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SECRET_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "secret_small.png"
COVER_FIXTURE_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "carrier_512.png"
    if (REPO_ROOT / "tests" / "fixtures" / "carrier_512.png").exists()
    else (
        REPO_ROOT / "dataset" / "MVP 0 Dataset" / "0001x2.png"
        if (REPO_ROOT / "dataset" / "MVP 0 Dataset" / "0001x2.png").exists()
        else REPO_ROOT / "MVP 0 Dataset" / "0001x2.png"
    )
)
_checkpoints_env = os.environ.get("PALADIN_CHECKPOINTS_DIR") or os.environ.get("PALADIN_MODEL_DIR")
if _checkpoints_env:
    CHECKPOINTS_DIR = Path(_checkpoints_env).resolve()
    ONNX_DIR = CHECKPOINTS_DIR if (CHECKPOINTS_DIR / "acs_sub_inn_embedding_opset14.onnx").exists() else (CHECKPOINTS_DIR / "phase5a_onnx")
else:
    CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
    ONNX_DIR = CHECKPOINTS_DIR / "phase5a_onnx"

EXPECTED_ONNX_HASHES = {
    "acs_sub_inn_embedding_opset14.onnx": "3cd22ba35e55d4addfb89e59cdc695f1086383bef5d507f39a3c91dd4917c173",
    "acs_sub_inn_extraction_opset14.onnx": "ab696814635b39dec34b432b78e80bb949c5fcd68517a3cec5890102152ff8ba",
    "neural_spatial_embedding_opset14.onnx": "7cda5a3becb8283c6b84572e7adb6d64ea29ecbf9f1e667cf34dcd0324e2d784",
    "neural_spatial_extraction_opset14.onnx": "bf85d8404469d90c08aa50282db582e3c82b2c6ea2159293423ae6e846b82be2",
}


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_onnx_integrity(onnx_dir: Path = ONNX_DIR) -> Dict[str, Any]:
    """Verifies existence and SHA-256 integrity of the four qualified ONNX model artifacts."""
    files_present = {}
    hashes_verified = {}
    actual_hashes = {}

    for fname, exp_hash in EXPECTED_ONNX_HASHES.items():
        fpath = onnx_dir / fname
        present = fpath.exists()
        files_present[fname] = present
        if present:
            act_hash = compute_file_sha256(fpath)
            actual_hashes[fname] = act_hash
            hashes_verified[fname] = (act_hash == exp_hash)
        else:
            actual_hashes[fname] = ""
            hashes_verified[fname] = False

    all_present = all(files_present.values())
    all_verified = all(hashes_verified.values())

    return {
        "files_present": files_present,
        "hashes_verified": hashes_verified,
        "actual_hashes": actual_hashes,
        "all_present": all_present,
        "all_verified": all_verified,
        "integrity_passed": all_present and all_verified,
    }


# Operational qualified deployment pipeline
try:
    qualified_pipeline: Optional[MediaStegoPipeline] = MediaStegoPipeline.create_qualified_deployment(
        backend="onnx",
        onnx_dir=ONNX_DIR,
    )
    _sessions_initialized = True
except Exception as e:
    logger.warning("Failed to pre-initialize qualified ONNX pipeline: %s", sanitize_error_message(e))
    qualified_pipeline = None
    _sessions_initialized = False

# Global bounded session store with count, TTL, and total byte lifecycle
_initial_settings = get_api_settings()
session_store = BoundedSessionStore(
    max_entries=_initial_settings.max_sessions,
    default_ttl=_initial_settings.session_ttl_seconds,
    max_bytes=_initial_settings.max_session_store_bytes,
)


def check_runtime_operational() -> Tuple[bool, str, Dict[str, Any]]:
    """Determines whether the steganography runtime is operational.

    Returns:
        (is_operational, reason, integrity_details)
    """
    integrity = verify_onnx_integrity()
    if not integrity["all_present"]:
        missing = [fn for fn, p in integrity["files_present"].items() if not p]
        return False, f"Missing qualified ONNX model files: {missing}", integrity
    if not integrity["all_verified"]:
        mismatched = [fn for fn, v in integrity["hashes_verified"].items() if not v]
        return False, f"ONNX model SHA-256 hash verification failed for: {mismatched}", integrity
    if qualified_pipeline is None:
        return False, "Qualified steganography pipeline is not initialized", integrity
    if not _sessions_initialized:
        return False, "ONNX Runtime inference sessions are not initialized", integrity
    return True, "Operational", integrity



# Concurrency & Memory Admission Control
MAX_CONCURRENT_ENGINE_JOBS = int(os.environ.get("PALADIN_MAX_CONCURRENT_ENGINE_JOBS", "1"))
ENGINE_ADMISSION_TIMEOUT_SECONDS = float(os.environ.get("PALADIN_ENGINE_ADMISSION_TIMEOUT", "10.0"))

_engine_semaphore: Optional[asyncio.Semaphore] = None


def get_engine_semaphore() -> asyncio.Semaphore:
    """Returns the singleton semaphore bounding concurrent neural engine executions."""
    global _engine_semaphore
    if _engine_semaphore is None:
        _engine_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ENGINE_JOBS)
    return _engine_semaphore


def reset_engine_semaphore(max_jobs: Optional[int] = None) -> None:
    """Resets the concurrency admission semaphore (used for testing or dynamic reconfiguration)."""
    global _engine_semaphore, MAX_CONCURRENT_ENGINE_JOBS
    if max_jobs is not None:
        MAX_CONCURRENT_ENGINE_JOBS = max_jobs
    else:
        MAX_CONCURRENT_ENGINE_JOBS = int(os.environ.get("PALADIN_MAX_CONCURRENT_ENGINE_JOBS", "1"))
    _engine_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ENGINE_JOBS)


@asynccontextmanager
async def engine_admission_gate(timeout: float = ENGINE_ADMISSION_TIMEOUT_SECONDS):
    """Guarantees memory budget invariants by bounding concurrent neural steganography engine tasks.

    Under the 512 MB container memory limit, a single execution of FP32 steganography consumes
    <= 255 MB peak RSS. By limiting concurrent executions (default: 1), total peak RSS remains
    strictly within the 512 MB budget. Concurrent requests wait up to timeout seconds before
    returning HTTP 503 (Capacity/Queue full).
    """
    sem = get_engine_semaphore()
    acquired = False
    try:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=timeout)
            acquired = True
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail=f"Engine concurrency limit reached ({MAX_CONCURRENT_ENGINE_JOBS} concurrent job(s)). Admission queue timeout ({timeout}s) exceeded. Upgrade server memory allocation (768MB-1024MB) or increase PALADIN_MAX_CONCURRENT_ENGINE_JOBS.",
            )
        yield
    finally:
        if acquired:
            sem.release()


def get_deployment_configuration(subband_dim: Union[int, Tuple[int, int]] = 256) -> Dict[str, Any]:
    """Exposes non-sensitive operational deployment configuration facts."""
    settings = get_api_settings()
    is_operational, reason, integrity = check_runtime_operational()

    if is_operational and qualified_pipeline is not None:
        cap_info = qualified_pipeline.get_capacity_limits(subband_dim=subband_dim)
    else:
        cap_info = {
            "ecc_profile": "rs_2t_48",
            "ecc_parity_symbols": 48,
            "carrier_format_policy": "Lossless 8-bit RGB PNG only. Lossy transcoded carriers (e.g. JPEG) are strictly unsupported.",
            "raw_secret_capacity_bytes": 6578,
            "reference_metadata_capacity_bytes": 100,
            "max_protected_bytes": 6578,
            "is_qualified_deployment": False,
            "operational_qualification_note": f"Qualified ONNX runtime is degraded: {reason}",
        }

    return {
        "status": "OPERATIONAL" if is_operational else "DEGRADED",
        "inference_backend": "onnx",
        "ecc_profile": cap_info["ecc_profile"],
        "ecc_parity_symbols": cap_info["ecc_parity_symbols"],
        "carrier_format_policy": cap_info["carrier_format_policy"],
        "carrier_dimensions": "dynamic",
        "baseline_carrier_dimensions": "512x512",
        "carrier_dimensions_policy": "Lossless 8-bit RGB PNG with even dimensions (>= 32x32, <= 2048x2048).",
        "raw_secret_capacity_bytes": cap_info["raw_secret_capacity_bytes"],
        "reference_metadata_capacity_bytes": cap_info["reference_metadata_capacity_bytes"],
        "max_protected_payload_bytes": cap_info["max_protected_bytes"],
        "is_qualified_deployment": cap_info["is_qualified_deployment"],
        "is_production_deployed": False,
        "runtime_integrity_verified": is_operational,
        "qualification_state": "pre_deployment_qualification",
        "pqc_enabled": settings.enable_pqc,
        "pqc_supported": True,
        "pqc_ciphersuite": "ML-KEM-768 + ML-DSA-65 + AES-256-GCM + HKDF-SHA256",
        "onnx_models_verified": integrity["all_verified"],
        "onnx_models_present": integrity["all_present"],
        "onnx_files_present": integrity["files_present"],
        "onnx_hashes_verified": integrity["hashes_verified"],
        "sessions_initialized": _sessions_initialized,
        "is_operational": is_operational,
        "operational_qualification_note": cap_info["operational_qualification_note"],
        "deployment_mode": settings.mode,
        "authentication_enforced": settings.is_auth_required(),
        "max_upload_size_bytes": settings.max_upload_bytes,
        "session_ttl_seconds": settings.session_ttl_seconds,
        "max_session_store_bytes": settings.max_session_store_bytes,
        "v3_mr_enabled": settings.enable_v3_mr,
        "default_protocol_version": 2,
        "supported_protocol_versions": [2, 3] if settings.enable_v3_mr else [2],
    }


# Router holding all operational endpoints
api_router = APIRouter()


@api_router.get("/api/health")
async def health_check() -> JSONResponse:
    """Public health and readiness probe (non-sensitive operational status)."""
    settings = get_api_settings()
    is_ready, reason, _ = check_runtime_operational()
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "healthy" if is_ready else "degraded",
            "live": True,
            "ready": is_ready,
            "deployment_mode": settings.mode,
            "is_qualified_deployment": is_ready,
            "is_production_deployed": False,
            "runtime_integrity_verified": is_ready,
            "qualification_state": "pre_deployment_qualification",
            "pqc_enabled": settings.enable_pqc,
            "reason": reason,
        },
    )


@api_router.get("/api/health/live")
async def liveness_probe() -> Dict[str, Any]:
    """Process liveness probe (independent of pipeline readiness)."""
    return {"status": "live", "live": True}


@api_router.get("/api/health/ready")
async def readiness_probe() -> JSONResponse:
    """Application readiness probe (verifies qualified pipeline usability)."""
    is_ready, reason, _ = check_runtime_operational()
    if not is_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "ready": False, "detail": reason},
        )
    return JSONResponse(status_code=200, content={"status": "ready", "ready": True})


def _exchange_code_at_idp(token_endpoint: str, data: Dict[str, str], timeout: float = 10.0) -> Dict[str, Any]:
    """Exchanges an authorization code with the IdP token endpoint per RFC 6749 / OIDC Core."""
    encoded_data = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        token_endpoint,
        data=encoded_data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "PaladinMedia-BFF/2.6.0-rc2",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


class LoginRequest(BaseModel):
    api_token: Optional[str] = None
    token: Optional[str] = None


@api_router.get("/api/auth/oidc/login")
async def auth_oidc_login(
    request: Request,
    redirect_to: Optional[str] = Query(None),
    json_response: bool = Query(False, alias="json"),
) -> Response:
    """Initiates OAuth 2.0 / OpenID Connect authorization code flow with PKCE."""
    settings = get_api_settings()

    if not settings.is_oidc_configured():
        if not settings.is_auth_required():
            # In local unauthenticated mode, redirect directly to requested target
            target = redirect_to or "/"
            return RedirectResponse(url=target, status_code=302)
        raise HTTPException(
            status_code=503,
            detail="OIDC Identity Provider is not configured (PALADIN_OIDC_ISSUER / PALADIN_OIDC_CLIENT_ID required). External IdP gate is OPEN.",
        )

    code_verifier, code_challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)

    redirect_uri = settings.oidc_redirect_uri
    if not redirect_uri:
        # Auto-derive from request base_url
        redirect_uri = str(request.base_url).rstrip("/") + "/api/auth/oidc/callback"

    default_oidc_state_store.store_state(
        state=state,
        nonce=nonce,
        code_verifier=code_verifier,
        redirect_uri=redirect_uri,
        ttl_seconds=300,
    )

    auth_endpoint = settings.oidc_auth_endpoint
    if not auth_endpoint:
        base = settings.oidc_issuer or ""
        auth_endpoint = f"{base}protocol/openid-connect/auth" if base.endswith("/") else f"{base}/protocol/openid-connect/auth"

    params = {
        "client_id": settings.oidc_client_id,
        "response_type": "code",
        "scope": settings.oidc_scopes,
        "redirect_uri": redirect_uri,
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    authorization_url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"

    accept = request.headers.get("accept", "")
    if json_response or "application/json" in accept:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "authorization_url": authorization_url,
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            },
        )

    return RedirectResponse(url=authorization_url, status_code=302)


@api_router.get("/api/auth/oidc/callback")
async def auth_oidc_callback(
    request: Request,
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
) -> Response:
    """Handles authorization-code callback from IdP, validates PKCE/state/nonce, sets session."""
    settings = get_api_settings()

    if error:
        raise HTTPException(
            status_code=400,
            detail=f"OIDC IdP authorization error: {error} - {error_description or ''}".strip(),
        )

    if not code or not state:
        raise HTTPException(
            status_code=400,
            detail="Missing required 'code' or 'state' parameters in OIDC callback.",
        )

    stored = default_oidc_state_store.consume_state(state)
    if stored is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid, expired, or replayed OAuth state parameter.",
        )

    if not settings.is_oidc_configured():
        raise HTTPException(
            status_code=503,
            detail="OIDC Identity Provider configuration missing at callback time.",
        )

    token_endpoint = settings.oidc_token_endpoint
    if not token_endpoint:
        base = settings.oidc_issuer or ""
        token_endpoint = f"{base}protocol/openid-connect/token" if base.endswith("/") else f"{base}/protocol/openid-connect/token"

    token_form: Dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": stored.redirect_uri,
        "client_id": settings.oidc_client_id or "",
        "code_verifier": stored.code_verifier,
    }
    if settings.oidc_client_secret:
        token_form["client_secret"] = settings.oidc_client_secret

    try:
        token_data = await asyncio.to_thread(_exchange_code_at_idp, token_endpoint, token_form)
    except Exception as exc:
        logger.error("Failed to exchange code at IdP token endpoint %s: %s", token_endpoint, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to exchange authorization code with IdP token endpoint: {str(exc)}",
        )

    id_token = token_data.get("id_token")
    if not id_token:
        raise HTTPException(
            status_code=502,
            detail="IdP token response did not include required 'id_token'.",
        )

    claims = validate_id_token_claims(
        id_token=id_token,
        expected_issuer=settings.oidc_issuer,
        expected_audience=settings.oidc_client_id,
        expected_nonce=stored.nonce,
        client_secret=settings.oidc_client_secret,
        jwks_uri=settings.oidc_jwks_uri,
    )

    user_id = (
        claims.get("email")
        or claims.get("preferred_username")
        or claims.get("sub")
        or "authenticated-user"
    )
    roles = claims.get("roles") or ["operator"]
    if isinstance(roles, str):
        roles = [roles]

    ttl = getattr(settings, "session_ttl_seconds", 3600) or 3600
    session = default_auth_session_store.create_session(user_id=user_id, ttl_seconds=ttl)
    session.roles = roles
    session.auth_method = "oidc"

    is_secure = is_request_https(request, settings)

    redirect_url = "/"
    response = RedirectResponse(url=redirect_url, status_code=302)
    response.set_cookie(
        key="paladin_session",
        value=session.session_id,
        httponly=True,
        secure=is_secure,
        samesite="lax",
        path="/",
        max_age=ttl,
    )
    response.set_cookie(
        key="paladin_csrf",
        value=session.csrf_token,
        httponly=False,
        secure=is_secure,
        samesite="lax",
        path="/",
        max_age=ttl,
    )
    return response


@api_router.post("/api/auth/login")
async def auth_login(req: LoginRequest, request: Request) -> Response:
    """Authenticates operator via API token and establishes a secure HttpOnly session cookie."""
    settings = get_api_settings()
    provided_token = (req.api_token or req.token or "").strip()

    if settings.is_auth_required():
        expected_token = (settings.api_token or "").strip()
        if not provided_token or not secrets.compare_digest(provided_token, expected_token):
            raise HTTPException(
                status_code=401,
                detail="Invalid API token",
            )

    ttl = getattr(settings, "session_ttl_seconds", 3600) or 3600
    session = default_auth_session_store.create_session(user_id="operator", ttl_seconds=ttl)

    is_secure = is_request_https(request, settings)
    samesite_val = "lax"

    response = JSONResponse(
        status_code=200,
        content={
            "status": "authenticated",
            "user": session.user_id,
            "csrf_token": session.csrf_token,
            "expires_at": session.expires_at,
        },
    )
    # Set HttpOnly session cookie (inaccessible to browser JavaScript/XSS)
    response.set_cookie(
        key="paladin_session",
        value=session.session_id,
        httponly=True,
        secure=is_secure,
        samesite=samesite_val,
        path="/",
        max_age=ttl,
    )
    # Set CSRF cookie (readable by frontend JS so it can submit X-CSRF-Token header)
    response.set_cookie(
        key="paladin_csrf",
        value=session.csrf_token,
        httponly=False,
        secure=is_secure,
        samesite=samesite_val,
        path="/",
        max_age=ttl,
    )
    return response


@api_router.get("/api/auth/session")
async def get_auth_session(request: Request) -> Dict[str, Any]:
    """Inspects the current browser authentication state."""
    settings = get_api_settings()
    is_oidc = settings.is_oidc_configured()
    idp_verified = os.environ.get("PALADIN_IDP_VERIFIED", "false").lower() in ("true", "1")
    idp_status = "CLOSED_VALIDATED" if (is_oidc and idp_verified) else "OPEN_PENDING_EXTERNAL_IDP"

    if not settings.is_auth_required():
        return {
            "authenticated": True,
            "mode": settings.mode,
            "requires_auth": False,
            "user": "local-operator",
            "roles": ["operator", "admin"],
            "csrf_token": "local-csrf-token",
            "oidc_configured": is_oidc,
            "oidc_issuer": settings.oidc_issuer,
            "oidc_login_url": "/api/auth/oidc/login" if is_oidc else None,
            "idp_gate_status": idp_status,
        }

    session_id = request.cookies.get("paladin_session")
    if session_id:
        session = default_auth_session_store.get_session(session_id)
        if session:
            return {
                "authenticated": True,
                "mode": settings.mode,
                "requires_auth": True,
                "user": session.user_id,
                "roles": getattr(session, "roles", ["operator"]),
                "csrf_token": session.csrf_token,
                "expires_at": session.expires_at,
                "auth_method": getattr(session, "auth_method", "token"),
                "oidc_configured": is_oidc,
                "oidc_issuer": settings.oidc_issuer,
                "oidc_login_url": "/api/auth/oidc/login" if is_oidc else None,
                "idp_gate_status": idp_status,
            }

    return {
        "authenticated": False,
        "mode": settings.mode,
        "requires_auth": True,
        "user": None,
        "roles": [],
        "csrf_token": None,
        "oidc_configured": is_oidc,
        "oidc_issuer": settings.oidc_issuer,
        "oidc_login_url": "/api/auth/oidc/login" if is_oidc else None,
        "idp_gate_status": idp_status,
    }


@api_router.post("/api/auth/logout")
async def auth_logout(request: Request) -> Response:
    """Invalidates the active session in the session store and clears cookies."""
    session_id = request.cookies.get("paladin_session")
    if session_id:
        default_auth_session_store.delete_session(session_id)

    response = JSONResponse(
        status_code=200,
        content={"status": "logged_out"},
    )
    response.delete_cookie(key="paladin_session", path="/")
    response.delete_cookie(key="paladin_csrf", path="/")
    return response


@api_router.get("/api/auth/csrf")
async def get_csrf_token(request: Request) -> Dict[str, Any]:
    """Returns the CSRF token for the active session."""
    session_id = request.cookies.get("paladin_session")
    if session_id:
        session = default_auth_session_store.get_session(session_id)
        if session:
            return {"csrf_token": session.csrf_token}

    settings = get_api_settings()
    if not settings.is_auth_required():
        return {"csrf_token": "local-csrf-token"}

    raise HTTPException(status_code=401, detail="Authentication required to obtain CSRF token.")


@api_router.get("/api/capabilities")
async def get_capabilities(
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Exposes real runtime capabilities, available modulation engines, protocol versions, and telemetry support."""
    settings = get_api_settings()
    is_operational, reason, integrity = check_runtime_operational()

    return {
        "status": "OPERATIONAL" if is_operational else "DEGRADED",
        "system_title": "PaladinMedia",
        "subtitle": "Post-Quantum Media Steganography & Wavelet Core",
        "active_engine": "neural_spatial_onnx",
        "available_engines": [
            {
                "id": "neural_spatial_onnx",
                "name": "Neural Spatial ONNX (Qualified)",
                "description": "Production opset14 spatial ONNX model with internal Haar DWT/IDWT",
                "status": "QUALIFIED",
                "is_default": True,
            },
            {
                "id": "acs_sub_inn_onnx",
                "name": "ACS-Sub-INN Subband ONNX",
                "description": "Transform-domain subband INN model with explicit Python 2D Haar DWT/IDWT",
                "status": "QUALIFIED",
                "is_default": False,
            },
            {
                "id": "acs_sub_inn_pytorch",
                "name": "ACS-Sub-INN PyTorch Reference",
                "description": "Native PyTorch ACSSubINN + NeuralSpatialBridge reference implementation",
                "status": "REFERENCE",
                "is_default": False,
            },
            {
                "id": "parity_classical",
                "name": "Deterministic Parity Modulation",
                "description": "Classical Phase 2 integer parity transform-domain modulation",
                "status": "LEGACY_FALLBACK",
                "is_default": False,
            },
        ],
        "default_protocol_version": 2,
        "supported_protocol_versions": [2, 3] if settings.enable_v3_mr else [2],
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "int8"],
        "concurrency": {
            "max_concurrent_engine_jobs": MAX_CONCURRENT_ENGINE_JOBS,
            "engine_admission_timeout_seconds": ENGINE_ADMISSION_TIMEOUT_SECONDS,
            "memory_budget_target_mb": 512,
            "single_job_peak_rss_mb": 255.0,
            "concurrent_memory_guideline": "1 worker <= 255 MB (fits 512 MB cgroups); N >= 2 workers require 768 MB - 1024 MB allocation.",
        },
        "v3_mr_enabled": settings.enable_v3_mr,
        "pqc_enabled": settings.enable_pqc,
        "pqc_supported": True,
        "pqc_ciphersuite": "ML-KEM-768 + ML-DSA-65 + AES-256-GCM + HKDF-SHA256",
        "is_production_deployed": False,
        "runtime_integrity_verified": is_operational,
        "qualification_state": "pre_deployment_qualification",
        "ecc_profile": "rs_2t_48",
        "supported_ecc_profiles": ["rs_2t_48", "rs_2t_64", "rs_2t_32"] if settings.enable_v3_mr else ["rs_2t_48", "rs_2t_32"],
        "telemetry": {
            "available": True,
            "source": "codecarbon" if SustainabilityTelemetryTracker.is_codecarbon_available() else "estimated_hardware_model",
            "is_estimated": not SustainabilityTelemetryTracker.is_codecarbon_available(),
            "metrics_tracked": [
                "execution_time_ms",
                "cpu_utilization_percent",
                "memory_rss_mb",
                "estimated_power_watts",
                "energy_kwh",
                "co2_emissions_grams",
            ],
        },
        "pipeline_stages": [
            {"id": "LOAD", "name": "Load Media", "domain": "io"},
            {"id": "SERIALIZE", "name": "Payload Serialization (CRC-32)", "domain": "framing"},
            {"id": "ENCRYPT", "name": "AES-256-GCM Encryption", "domain": "crypto"},
            {"id": "RS_ENCODE", "name": "Reed-Solomon ECC Encoding", "domain": "ecc"},
            {"id": "CAPACITY_CHECK", "name": "Carrier Capacity Hard Gate", "domain": "gate"},
            {"id": "DWT", "name": "1-Level 2D Haar DWT", "domain": "wavelet"},
            {"id": "EMBED", "name": "Transform-Domain Modulation", "domain": "modulation"},
            {"id": "IDWT", "name": "Inverse DWT & PNG Synthesis", "domain": "wavelet"},
            {"id": "RECEIVER_DWT", "name": "Receiver Stego DWT", "domain": "wavelet"},
            {"id": "EXTRACT", "name": "Header & Payload Extraction", "domain": "demodulation"},
            {"id": "RS_DECODE", "name": "Reed-Solomon Error Correction", "domain": "ecc"},
            {"id": "AES_VERIFY", "name": "AES-GCM Tag Verification", "domain": "crypto"},
            {"id": "SECRET_RECOVER", "name": "Secret Media Reconstruction", "domain": "framing"},
            {"id": "EXACT_MATCH_VERIFY", "name": "Exact Byte Equality Verification", "domain": "verification"},
        ],
        "onnx_models_verified": integrity["all_verified"],
        "max_upload_size_bytes": settings.max_upload_bytes,
        "max_carrier_dimension": 512 if settings.mode == "demo" else 2048,
        "max_carrier_pixels": (512 * 512) if settings.mode == "demo" else (4 * 1024 * 1024),
    }


@api_router.get("/api/telemetry/{run_id}")
async def get_run_telemetry(
    run_id: str,
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Retrieves green computing telemetry metrics for a specific completed pipeline execution."""
    metrics = telemetry_tracker.get_metrics(run_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No telemetry found for run_id '{run_id}'")
    return metrics.to_dict()


@api_router.get("/api/config")
async def get_config(
    width: Optional[int] = Query(None, description="Optional carrier width to compute dynamic capacity"),
    height: Optional[int] = Query(None, description="Optional carrier height to compute dynamic capacity"),
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Exposes non-sensitive operational facts for client verification."""
    subband_dim: Union[int, Tuple[int, int]] = 256
    if width is not None and height is not None:
        subband_dim = (height // 2, width // 2)
    elif width is not None:
        subband_dim = width // 2
    return get_deployment_configuration(subband_dim=subband_dim)


@api_router.get("/api/carrier/capacity")
async def get_carrier_capacity(
    width: int = Query(512, description="Carrier width in pixels"),
    height: int = Query(512, description="Carrier height in pixels"),
    protocol_version: Optional[int] = Query(None, description="Wire protocol version (2 or 3)"),
    protocol: Optional[str] = Query(None, description="Wire protocol selector ('v2' or 'v3_mr')"),
    version: Optional[int] = Query(None, description="Wire protocol version alias"),
    ecc_profile: Optional[str] = Query(None, description="Reed-Solomon ECC profile ('rs_2t_48' or 'rs_2t_64')"),
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Computes dynamic secret capacity limits for a given carrier width, height, and protocol."""
    settings = get_api_settings()
    if width < 32 or height < 32:
        raise HTTPException(
            status_code=422,
            detail=f"Carrier dimensions must be at least 32x32 pixels, got {width}x{height}",
        )
    if width % 2 != 0 or height % 2 != 0:
        raise HTTPException(
            status_code=422,
            detail=f"Carrier dimensions must be even for DWT decomposition, got {width}x{height}",
        )
    if width > 2048 or height > 2048 or (width * height) > 4 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Carrier dimensions {width}x{height} exceed maximum resource limit (2048x2048 / 4 Mpx)",
        )

    # Resolve target wire protocol version
    target_version = 2
    if protocol is not None:
        p_clean = protocol.strip().lower()
        if p_clean in ("v3_mr", "v3-mr", "v3_maximum_resilience"):
            target_version = 3
        elif p_clean in ("v2", "wire_protocol_v2", "2"):
            target_version = 2
        elif p_clean in ("v1", "wire_protocol_v1", "1"):
            target_version = 1
        elif p_clean in ("v3_standard", "v3", "3"):
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported wire protocol '{protocol}'. Wire Protocol v3 Standard is an unpromoted experimental protocol. Only v3-MR is supported.",
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported wire protocol '{protocol}'. Supported options are 'v2' or 'v3_mr'.",
            )
    elif ecc_profile is not None:
        e_clean = ecc_profile.strip().lower()
        if e_clean in ("rs_2t_64", "rs(255,191)", "64"):
            target_version = 3
        elif e_clean in ("rs_2t_48", "rs(255,207)", "48"):
            target_version = 2
        elif e_clean in ("rs_2t_32", "rs(255,223)", "32"):
            target_version = 1
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported ECC profile '{ecc_profile}'. Supported profiles are 'rs_2t_48' (v2 default) or 'rs_2t_64' (v3-MR).",
            )
    elif protocol_version is not None or version is not None:
        ver = protocol_version if protocol_version is not None else version
        if ver not in (1, 2, 3):
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported wire protocol version: {ver}. Supported versions are 2 (default) or 3 (v3-MR).",
            )
        target_version = ver

    if target_version == 3 and not settings.enable_v3_mr:
        raise HTTPException(
            status_code=403,
            detail="Wire Protocol v3-MR is disabled on this server. Contact administrator or set PALADIN_ENABLE_V3_MR=true.",
        )

    subband_dim = (height // 2, width // 2)
    ecc_parity = 64 if target_version == 3 else (48 if target_version == 2 else 32)
    raw_cap = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=0, subband_dim=subband_dim, rs_parity=ecc_parity, version=target_version
    )
    ref_cap = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=2, subband_dim=subband_dim, rs_parity=ecc_parity, version=target_version
    )
    hdr_bytes = CapacityCalculator.get_header_bytes(version=target_version)
    max_prot = (subband_dim[0] * subband_dim[1] // 8) - hdr_bytes

    return {
        "carrier_width": width,
        "carrier_height": height,
        "subband_height": subband_dim[0],
        "subband_width": subband_dim[1],
        "wire_protocol_version": target_version,
        "protocol_version": target_version,
        "header_wire_bytes": hdr_bytes,
        "ecc_profile": f"rs_2t_{ecc_parity}",
        "ecc_parity_symbols": ecc_parity,
        "raw_secret_capacity_bytes": raw_cap,
        "reference_metadata_capacity_bytes": ref_cap,
        "max_protected_bytes": max_prot,
        "is_qualified_deployment": (target_version == 2),
        "carrier_format_policy": "Lossless 8-bit RGB PNG only. Lossy transcoded carriers (e.g. JPEG) are strictly unsupported.",
        "operational_qualification_note": (
            "Wire Protocol v3 Maximum Resilience (RS(255,191), 2t=64, Row 32 anchor, symbol-interleaved)."
            if target_version == 3
            else (
                "Primary qualified deployment profile rs_2t_48 with native ONNX Runtime backend and Wire Protocol v2 Header ECC."
                if target_version == 2
                else "Legacy Wire Protocol v1 (unprotected header)."
            )
        ),
    }


@api_router.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Uploads secret or cover media, storing bytes under a session id with TTL."""
    settings = get_api_settings()

    # Read file stream in chunks to strictly enforce size ceiling before buffering
    chunk_size = 64 * 1024
    total_size = 0
    buffer = io.BytesIO()

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds maximum permitted size of {settings.max_upload_bytes} bytes",
            )
        buffer.write(chunk)

    data = buffer.getvalue()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    media_type = file.content_type or "application/octet-stream"
    MAX_PIXEL_COUNT = (512 * 512) if settings.mode == "demo" else (4 * 1024 * 1024)
    MAX_DIMENSION = 512 if settings.mode == "demo" else 2048

    try:
        Image.MAX_IMAGE_PIXELS = MAX_PIXEL_COUNT
        with Image.open(io.BytesIO(data)) as probed:
            probed.verify()
            if probed.format:
                media_type = f"image/{probed.format.lower()}"
    except Exception:
        if media_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Malformed or corrupted image content")

    width = None
    height = None
    img_format = None
    img_mode = None
    if media_type.startswith("image/"):
        try:
            with Image.open(io.BytesIO(data)) as probed:
                w, h = probed.size
                if w * h > MAX_PIXEL_COUNT or w > MAX_DIMENSION or h > MAX_DIMENSION:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image dimensions ({w}x{h}={w*h} pixels) exceed maximum allowed pixel budget of {MAX_PIXEL_COUNT} pixels ({MAX_DIMENSION}x{MAX_DIMENSION}).",
                    )
                width = w
                height = h
                img_format = probed.format
                img_mode = probed.mode
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="Malformed or corrupted image content")

    safe_filename = os.path.basename(file.filename) if file.filename else None
    session_id = session_store.store(
        data=data,
        ttl_seconds=settings.session_ttl_seconds,
        media_type=media_type,
        filename=safe_filename,
    )
    resp: Dict[str, Any] = {"session_id": session_id, "size": len(data), "filename": safe_filename}
    if width is not None and height is not None:
        resp["width"] = width
        resp["height"] = height
        resp["format"] = img_format
        resp["mode"] = img_mode
    return resp


@api_router.delete("/api/session/{session_id}")
async def delete_session(
    session_id: str,
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Explicitly deletes a session record and its stored binary data."""
    deleted = session_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found or already expired")
    return {"status": "deleted", "session_id": session_id}


@api_router.get("/api/fixtures/defaults")
async def get_default_fixtures(
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Provides base64 sample fixtures for instant laboratory evaluation."""
    settings = get_api_settings()
    secret_b64 = None
    secret_session_id = None
    if SECRET_FIXTURE_PATH.exists():
        with open(SECRET_FIXTURE_PATH, "rb") as f:
            sec_bytes = f.read()
            secret_b64 = f"data:image/png;base64,{base64.b64encode(sec_bytes).decode('ascii')}"
            secret_session_id = "default_secret"
            session_store.store(
                data=sec_bytes,
                ttl_seconds=settings.session_ttl_seconds,
                media_type="image/png",
                custom_session_id=secret_session_id,
            )

    cover_b64 = None
    cover_session_id = None
    if COVER_FIXTURE_PATH.exists():
        with open(COVER_FIXTURE_PATH, "rb") as f:
            cov_bytes = f.read()
            cover_b64 = f"data:image/png;base64,{base64.b64encode(cov_bytes).decode('ascii')}"
            cover_session_id = "default_cover"
            session_store.store(
                data=cov_bytes,
                ttl_seconds=settings.session_ttl_seconds,
                media_type="image/png",
                custom_session_id=cover_session_id,
            )

    secret_meta: Dict[str, Any] = {}
    if secret_b64 and SECRET_FIXTURE_PATH.exists():
        try:
            with Image.open(SECRET_FIXTURE_PATH) as s_img:
                secret_meta = {"width": s_img.size[0], "height": s_img.size[1], "format": s_img.format, "mode": s_img.mode}
        except Exception:
            pass

    cover_meta: Dict[str, Any] = {}
    if cover_b64 and COVER_FIXTURE_PATH.exists():
        try:
            with Image.open(COVER_FIXTURE_PATH) as c_img:
                cover_meta = {"width": c_img.size[0], "height": c_img.size[1], "format": c_img.format, "mode": c_img.mode}
        except Exception:
            pass

    return {
        "secret": {
            "session_id": secret_session_id,
            "data_url": secret_b64,
            "name": "secret_small.png (24x24)",
            "size": len(session_store.get(secret_session_id, b"")),
            **secret_meta,
        },
        "cover": {
            "session_id": cover_session_id,
            "data_url": cover_b64,
            "name": "0001x2.png (512x512)",
            "size": len(session_store.get(cover_session_id, b"")),
            **cover_meta,
        },
    }


def validate_carrier_image(cover_bytes: bytes) -> Image.Image:
    """Strictly validates a carrier image against the qualified deployment contract:
    Lossless 8-bit RGB PNG with even dimensions (>= 32x32, <= 2048x2048).

    Directly inspects the raw PNG signature and IHDR chunk before Pillow decoding:
    - Verifies 8-byte PNG signature
    - Verifies 13-byte IHDR chunk type and CRC32 checksum
    - Verifies even dimensions >= 32x32 and <= 2048x2048 (resource limit)
    - Verifies bit depth of exactly 8 bits per sample (rejects 16-bit PNGs)
    - Verifies color type 2 (Truecolor RGB without alpha)
    - Rejects JPEG, RGBA, grayscale, palette, odd dimensions, < 32x32, and malformed files with HTTP 422.
    - Does NOT perform silent resizing, cropping, padding, or color conversion.
    """
    MAX_PIXEL_COUNT = 4 * 1024 * 1024
    MAX_DIMENSION = 2048
    PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

    # 1. Minimum header length check (8 signature + 4 len + 4 type + 13 data + 4 crc = 33 bytes)
    if len(cover_bytes) < 33:
        raise HTTPException(
            status_code=422,
            detail="Malformed or corrupted cover carrier image: File too small for PNG header",
        )

    # 2. Check PNG signature
    if not cover_bytes.startswith(PNG_SIGNATURE):
        detected = "unknown"
        if cover_bytes.startswith(b"\xff\xd8\xff"):
            detected = "JPEG"
        elif cover_bytes.startswith(b"RIFF") and len(cover_bytes) >= 12 and cover_bytes[8:12] == b"WEBP":
            detected = "WEBP"
        elif cover_bytes.startswith(b"GIF87a") or cover_bytes.startswith(b"GIF89a"):
            detected = "GIF"
        elif cover_bytes.startswith(b"BM"):
            detected = "BMP"
        raise HTTPException(
            status_code=422,
            detail=f"Carrier format must be lossless PNG, got {detected}",
        )

    # 3. Parse and validate IHDR chunk directly from raw bytes
    ihdr_len = struct.unpack(">I", cover_bytes[8:12])[0]
    ihdr_type = cover_bytes[12:16]
    if ihdr_len != 13 or ihdr_type != b"IHDR":
        raise HTTPException(
            status_code=422,
            detail="Malformed or corrupted cover carrier image: First chunk must be 13-byte IHDR",
        )

    ihdr_crc = struct.unpack(">I", cover_bytes[29:33])[0]
    calc_crc = zlib.crc32(cover_bytes[12:29])
    if calc_crc != ihdr_crc:
        raise HTTPException(
            status_code=422,
            detail="Malformed or corrupted cover carrier image: IHDR CRC32 checksum mismatch",
        )

    width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", cover_bytes[16:29])

    # Enforce minimum dimension requirement (>= 32x32)
    if width < 32 or height < 32:
        raise HTTPException(
            status_code=422,
            detail=f"Cover carrier dimensions must be at least 32x32 pixels, got {width}x{height}",
        )

    # Enforce even dimensions requirement for Haar DWT decomposition
    if width % 2 != 0 or height % 2 != 0:
        raise HTTPException(
            status_code=422,
            detail=f"Cover carrier dimensions must be even for DWT decomposition, got {width}x{height}",
        )

    # Enforce resource boundaries (<= 2048x2048 and <= 4 Mpx)
    if width > MAX_DIMENSION or height > MAX_DIMENSION or (width * height) > MAX_PIXEL_COUNT:
        raise HTTPException(
            status_code=413,
            detail=f"Cover carrier dimensions {width}x{height} exceed maximum resource limit ({MAX_DIMENSION}x{MAX_DIMENSION} / {MAX_PIXEL_COUNT} px)",
        )

    # Enforce strict 8-bit sample depth (reject 16-bit PNGs before Pillow can downsample)
    if bit_depth != 8:
        raise HTTPException(
            status_code=422,
            detail=f"Carrier bit depth must be 8 bits per channel, got {bit_depth}",
        )

    # Enforce Truecolor RGB color type (type 2, 24bpp without alpha)
    if color_type != 2:
        if color_type == 6:
            err_detail = "Carrier color mode must be 8-bit RGB without alpha, got RGBA"
        elif color_type == 0:
            err_detail = "Carrier color mode must be 8-bit RGB without alpha, got L"
        elif color_type == 3:
            err_detail = "Carrier color mode must be 8-bit RGB without alpha, got P"
        else:
            err_detail = f"Carrier color mode must be 8-bit RGB without alpha, got type {color_type}"
        raise HTTPException(status_code=422, detail=err_detail)

    # 4. Verify decoded structure, decompression, and integrity via Pillow
    try:
        with Image.open(io.BytesIO(cover_bytes)) as img:
            w, h = img.size
            if w < 32 or h < 32:
                raise HTTPException(
                    status_code=422,
                    detail=f"Cover carrier dimensions must be at least 32x32 pixels, got {w}x{h}",
                )
            if w % 2 != 0 or h % 2 != 0:
                raise HTTPException(
                    status_code=422,
                    detail=f"Cover carrier dimensions must be even for DWT decomposition, got {w}x{h}",
                )
            if w > MAX_DIMENSION or h > MAX_DIMENSION or (w * h) > MAX_PIXEL_COUNT:
                raise HTTPException(
                    status_code=413,
                    detail=f"Cover carrier dimensions {w}x{h} exceed maximum resource limit ({MAX_DIMENSION}x{MAX_DIMENSION} / {MAX_PIXEL_COUNT} px)",
                )
            if img.mode != "RGB":
                raise HTTPException(
                    status_code=422,
                    detail=f"Carrier color mode must be 8-bit RGB without alpha, got {img.mode}",
                )
            img.load()
            return img.copy()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Malformed or corrupted cover carrier image: {sanitize_error_message(e)}",
        )


def resolve_cover_image(
    cover_session_id: Optional[str],
    is_remote: bool,
) -> Image.Image:
    """Resolves and validates the cover carrier image.

    In remote mode:
    - cover_session_id is required. Missing or expired session raises 404/422.
    - Fixture fallback is strictly prohibited.
    - Carrier must be lossless 8-bit RGB PNG with even dimensions (>= 32x32, <= 2048x2048).

    In local mode:
    - If cover_session_id is provided, it must exist (missing/expired raises 404).
    - If omitted, falls back to bundled laboratory fixture or neutral canvas.
    - All carriers must strictly satisfy the lossless 8-bit RGB PNG contract.
    """
    if is_remote:
        if not cover_session_id:
            raise HTTPException(
                status_code=422,
                detail="cover_session_id is required in remote deployment mode. Bundled fixture fallback is prohibited.",
            )
        cover_bytes = session_store.get(cover_session_id)
        if not cover_bytes:
            raise HTTPException(status_code=404, detail="Cover carrier session not found or expired")
        return validate_carrier_image(cover_bytes)

    # Local mode
    if cover_session_id:
        cover_bytes = session_store.get(cover_session_id)
        if not cover_bytes:
            raise HTTPException(status_code=404, detail="Cover carrier session not found or expired")
        return validate_carrier_image(cover_bytes)

    # Local default fallback
    if COVER_FIXTURE_PATH.exists():
        with open(COVER_FIXTURE_PATH, "rb") as f:
            fixture_bytes = f.read()
        return validate_carrier_image(fixture_bytes)

    # Emergency fallback neutral canvas (synthesize 512x512 PNG bytes)
    neutral_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    neutral_buf = io.BytesIO()
    neutral_img.save(neutral_buf, format="PNG")
    return validate_carrier_image(neutral_buf.getvalue())


class DirectEmbedRequest(BaseModel):
    secret_session_id: str
    cover_session_id: Optional[str] = None
    key_hex: Optional[str] = Field(None, description="64-character hex string representing a 32-byte key")
    session_counter: int = 1
    model_name: Optional[str] = Field("neural_spatial_onnx", description="Modulation engine selector")
    precision: Optional[str] = Field("fp32", description="Model precision ('fp32' or 'int8')")
    protocol: Optional[str] = Field(None, description="Wire protocol selector ('v2' or 'v3_mr')")
    version: Optional[int] = Field(None, description="Wire protocol version (2 or 3)")
    pqc_enabled: Optional[bool] = Field(False, description="Enable Post-Quantum Cryptography (ML-KEM-768 + ML-DSA-65)")
    pqc_sender_id: Optional[str] = Field(None, description="Sender identity ID in trust store")
    pqc_sender_dsa_private_key_hex: Optional[str] = Field(None, description="Sender ML-DSA-65 private key hex")
    pqc_recipient_id: Optional[str] = Field(None, description="Recipient identity ID in trust store")
    pqc_option: Optional[str] = Field("B", description="PQC delivery option ('B' or 'A')")


@api_router.post("/api/embed")
async def embed_media(
    req: DirectEmbedRequest,
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Direct operational embedding endpoint using the qualified deployment pipeline."""
    settings = get_api_settings()
    is_ready, reason, _ = check_runtime_operational()
    if not is_ready:
        raise HTTPException(status_code=503, detail=f"Steganography runtime is unavailable: {reason}")

    target_version = 2
    if req.protocol is not None:
        p_clean = req.protocol.strip().lower()
        if p_clean in ("v3_mr", "v3-mr", "v3_maximum_resilience"):
            target_version = 3
        elif p_clean in ("v2", "wire_protocol_v2", "2"):
            target_version = 2
        elif p_clean in ("v1", "wire_protocol_v1", "1"):
            target_version = 1
        elif p_clean in ("v3_standard", "v3", "3"):
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported wire protocol '{req.protocol}'. Wire Protocol v3 Standard is an unpromoted experimental protocol. Only v3-MR is supported.",
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported wire protocol '{req.protocol}'. Supported options are 'v2' or 'v3_mr'.",
            )
    elif req.version is not None:
        if req.version not in (1, 2, 3):
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported wire protocol version: {req.version}. Supported versions are 2 (default) or 3 (v3-MR).",
            )
        target_version = req.version

    if target_version == 3 and not settings.enable_v3_mr:
        raise HTTPException(
            status_code=403,
            detail="Wire Protocol v3-MR is disabled on this server. Contact administrator or set PALADIN_ENABLE_V3_MR=true.",
        )

    key = validate_key_hex(req.key_hex, is_remote=settings.is_remote())

    secret_bytes = session_store.get(req.secret_session_id)
    if not secret_bytes:
        raise HTTPException(status_code=404, detail="Secret payload session not found or expired")

    cover_img = resolve_cover_image(req.cover_session_id, is_remote=settings.is_remote())

    # Preflight capacity check before DWT or neural modulation
    cover_w, cover_h = cover_img.size
    sub_dim = (cover_h // 2, cover_w // 2)
    ecc_parity = 64 if target_version == 3 else (48 if target_version == 2 else 32)
    meta_dict = {"filename": "secret_upload.png", "source": "api_embed"}
    import json as _json
    meta_len = len(_json.dumps(meta_dict, separators=(",", ":")).encode("utf-8"))
    usable_cap = CapacityCalculator.calculate_usable_secret_capacity(
        metadata_len=meta_len, subband_dim=sub_dim, rs_parity=ecc_parity, version=target_version
    )
    if len(secret_bytes) > usable_cap:
        raise HTTPException(
            status_code=413,
            detail=f"PAYLOAD_CAPACITY_EXCEEDED: Secret payload data ({len(secret_bytes)} B) "
                   f"exceeds maximum usable capacity ({usable_cap} B with {meta_len}B metadata) "
                   f"for ECC parity {ecc_parity} under Wire Protocol v{target_version}. Embedding aborted.",
        )

    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.IMAGE,
        metadata=meta_dict,
    )

    if req.pqc_enabled:
        if not settings.enable_pqc:
            raise HTTPException(
                status_code=403,
                detail="Post-Quantum Cryptography is disabled on this server. Contact administrator or set PALADIN_ENABLE_PQC=true.",
            )
        if not req.pqc_sender_id or not req.pqc_sender_dsa_private_key_hex or not req.pqc_recipient_id:
            raise HTTPException(
                status_code=422,
                detail="pqc_sender_id, pqc_sender_dsa_private_key_hex, and pqc_recipient_id are required when pqc_enabled=true.",
            )
        try:
            sender_dsa_priv = bytes.fromhex(req.pqc_sender_dsa_private_key_hex)
        except ValueError:
            raise HTTPException(status_code=422, detail="pqc_sender_dsa_private_key_hex must be valid hex.")

        pqc_opt = (req.pqc_option or "B").upper()
        pqc_overhead = 4547 if pqc_opt == "B" else 1150
        pre_rs_cap = (32 * 191) if target_version == 3 else (32 * 207)
        max_pqc_payload = pre_rs_cap - pqc_overhead
        if len(secret_bytes) > max_pqc_payload:
            raise HTTPException(
                status_code=413,
                detail=f"Secret payload data ({len(secret_bytes)} B) exceeds PQC Option {pqc_opt} usable capacity ({max_pqc_payload} B).",
            )

        precision_val = (req.precision or "fp32").lower()
        try:
            assert qualified_pipeline is not None
            if target_version == 3:
                pipe = MediaStegoPipeline.create_v3_mr_deployment(
                    backend=req.model_name or "neural_spatial_onnx",
                    precision=precision_val,
                )
            elif (req.model_name and req.model_name != "neural_spatial_onnx") or precision_val != "fp32":
                pipe = MediaStegoPipeline(
                    crypto_provider=qualified_pipeline.security_pipeline.crypto_provider,
                    ecc_encoder=qualified_pipeline.security_pipeline.ecc_encoder,
                    backend=req.model_name or "neural_spatial_onnx",
                    precision=precision_val,
                )
            else:
                pipe = qualified_pipeline

            async with engine_admission_gate():
                res = pipe.protect_and_embed_pqc(
                    secret_payload=secret_payload,
                    cover_image=cover_img,
                    sender_id=req.pqc_sender_id,
                    sender_dsa_private_key=sender_dsa_priv,
                    recipient_id=req.pqc_recipient_id,
                    option=pqc_opt,
                    session_counter=req.session_counter,
                    version=target_version,
                )
        except PayloadCapacityExceededError as e:
            raise HTTPException(status_code=413, detail=sanitize_error_message(e))
        except Exception as e:
            logger.error("PQC embedding execution failure: %s", sanitize_error_message(e))
            raise HTTPException(status_code=500, detail=f"PQC embedding failed: {sanitize_error_message(e)}")
    else:
        precision_val = (req.precision or "fp32").lower()
        try:
            assert qualified_pipeline is not None
            if target_version == 3:
                pipe = MediaStegoPipeline.create_v3_mr_deployment(
                    backend=req.model_name or "neural_spatial_onnx",
                    precision=precision_val,
                )
            elif (req.model_name and req.model_name != "neural_spatial_onnx") or precision_val != "fp32":
                pipe = MediaStegoPipeline(
                    crypto_provider=qualified_pipeline.security_pipeline.crypto_provider,
                    ecc_encoder=qualified_pipeline.security_pipeline.ecc_encoder,
                    backend=req.model_name or "neural_spatial_onnx",
                    precision=precision_val,
                )
            else:
                pipe = qualified_pipeline

            async with engine_admission_gate():
                res = pipe.protect_and_embed(
                    secret_payload=secret_payload,
                    cover_image=cover_img,
                    key=key,
                    session_counter=req.session_counter,
                    version=target_version,
                )
        except PayloadCapacityExceededError as e:
            raise HTTPException(status_code=413, detail=sanitize_error_message(e))
        except Exception as e:
            logger.error("Embedding execution failure: %s", sanitize_error_message(e))
            raise HTTPException(status_code=500, detail=f"Embedding failed: {sanitize_error_message(e)}")

    stego_session_id = session_store.store(
        data=res.stego_png_bytes,
        ttl_seconds=settings.session_ttl_seconds,
        media_type="image/png",
    )

    return {
        "status": "success",
        "stego_session_id": stego_session_id,
        "stego_size_bytes": len(res.stego_png_bytes),
        "carrier_format": res.carrier_format,
        "output_format_policy": res.output_format_policy,
        "psnr_db": round(res.psnr_db, 2),
        "ssim": round(res.ssim, 5),
        "ecc_profile": f"rs_2t_{ecc_parity}",
        "wire_protocol_version": target_version,
    }


class DirectExtractRequest(BaseModel):
    stego_session_id: str
    key_hex: Optional[str] = Field(None, description="64-character hex string representing a 32-byte key")
    model_name: Optional[str] = Field("neural_spatial_onnx", description="Modulation engine selector")
    precision: Optional[str] = Field("fp32", description="Model precision ('fp32' or 'int8')")
    pqc_enabled: Optional[bool] = Field(False, description="Expect Post-Quantum Cryptography envelope")
    pqc_recipient_kem_private_key_hex: Optional[str] = Field(None, description="Recipient ML-KEM-768 private key hex")
    pqc_detached_signature_hex: Optional[str] = Field(None, description="Detached ML-DSA-65 signature hex (for Option A)")
    pqc_sender_id_override: Optional[str] = Field(None, description="Sender ID override (for Option A)")


@api_router.post("/api/extract")
async def extract_media(
    req: DirectExtractRequest,
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Direct operational extraction endpoint using the qualified deployment pipeline."""
    settings = get_api_settings()
    is_ready, reason, _ = check_runtime_operational()
    if not is_ready:
        raise HTTPException(status_code=503, detail=f"Steganography runtime is unavailable: {reason}")

    key = validate_key_hex(req.key_hex, is_remote=settings.is_remote())

    stego_bytes = session_store.get(req.stego_session_id)
    if not stego_bytes:
        raise HTTPException(status_code=404, detail="Stego carrier session not found or expired")

    # Validate stego carrier dimensions if decodeable as an image
    try:
        with Image.open(io.BytesIO(stego_bytes)) as img:
            w, h = img.size
            if w < 32 or h < 32:
                raise HTTPException(
                    status_code=422,
                    detail=f"Stego carrier dimensions must be at least 32x32 pixels, got {w}x{h}",
                )
            if w % 2 != 0 or h % 2 != 0:
                raise HTTPException(
                    status_code=422,
                    detail=f"Stego carrier dimensions must be even for DWT decomposition, got {w}x{h}",
                )
            if w > 2048 or h > 2048 or (w * h) > 4 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"Stego carrier dimensions {w}x{h} exceed maximum resource limit (2048x2048 / 4 Mpx)",
                )
    except HTTPException:
        raise
    except Exception:
        # Non-image or damaged headers handled by pipeline fail-closed
        pass

    if req.pqc_enabled:
        if not settings.enable_pqc:
            raise HTTPException(
                status_code=403,
                detail="Post-Quantum Cryptography is disabled on this server. Contact administrator or set PALADIN_ENABLE_PQC=true.",
            )
        if not req.pqc_recipient_kem_private_key_hex:
            raise HTTPException(
                status_code=422,
                detail="pqc_recipient_kem_private_key_hex is required when pqc_enabled=true.",
            )
        try:
            recipient_kem_priv = bytes.fromhex(req.pqc_recipient_kem_private_key_hex)
        except ValueError:
            raise HTTPException(status_code=422, detail="pqc_recipient_kem_private_key_hex must be valid hex.")

        detached_sig = None
        if req.pqc_detached_signature_hex:
            try:
                detached_sig = bytes.fromhex(req.pqc_detached_signature_hex)
            except ValueError:
                raise HTTPException(status_code=422, detail="pqc_detached_signature_hex must be valid hex.")

        precision_val = (req.precision or "fp32").lower()
        try:
            assert qualified_pipeline is not None
            if (req.model_name and req.model_name != "neural_spatial_onnx") or precision_val != "fp32":
                pipe = MediaStegoPipeline(
                    crypto_provider=qualified_pipeline.security_pipeline.crypto_provider,
                    ecc_encoder=qualified_pipeline.security_pipeline.ecc_encoder,
                    backend=req.model_name or "neural_spatial_onnx",
                    precision=precision_val,
                )
            else:
                pipe = qualified_pipeline

            async with engine_admission_gate():
                recovery = pipe.extract_and_unprotect_pqc(
                    stego_carrier=stego_bytes,
                    recipient_kem_private_key=recipient_kem_priv,
                    detached_signature=detached_sig,
                    sender_id_override=req.pqc_sender_id_override,
                )
        except UnsupportedCarrierFormatError as e:
            raise HTTPException(status_code=400, detail=sanitize_error_message(e))
        except CarrierHeaderError as e:
            err_msg = sanitize_error_message(e)
            if "Wire Protocol v3 is disabled" in err_msg or "v3-MR is not authorized" in err_msg:
                raise HTTPException(status_code=403, detail=err_msg)
            raise HTTPException(
                status_code=422,
                detail=f"Extraction failed closed: {err_msg}. Please ensure the carrier is the original lossless PNG.",
            )
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"PQC extraction failed closed: {sanitize_error_message(e)}. Please ensure the carrier is the original lossless PNG.",
            )
    else:
        precision_val = (req.precision or "fp32").lower()
        try:
            assert qualified_pipeline is not None
            if (req.model_name and req.model_name != "neural_spatial_onnx") or precision_val != "fp32":
                pipe = MediaStegoPipeline(
                    crypto_provider=qualified_pipeline.security_pipeline.crypto_provider,
                    ecc_encoder=qualified_pipeline.security_pipeline.ecc_encoder,
                    backend=req.model_name or "neural_spatial_onnx",
                    precision=precision_val,
                )
            else:
                pipe = qualified_pipeline
            async with engine_admission_gate():
                recovery = pipe.extract_and_unprotect(stego_carrier=stego_bytes, key=key)
        except UnsupportedCarrierFormatError as e:
            raise HTTPException(status_code=400, detail=sanitize_error_message(e))
        except CarrierHeaderError as e:
            err_msg = sanitize_error_message(e)
            if "Wire Protocol v3 is disabled" in err_msg or "v3-MR is not authorized" in err_msg:
                raise HTTPException(status_code=403, detail=err_msg)
            raise HTTPException(
                status_code=422,
                detail=f"Extraction failed closed: {err_msg}. Please ensure the carrier is the original lossless PNG.",
            )
        except (ECCDecodeError, AuthenticationError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Extraction failed closed: {sanitize_error_message(e)}. Please ensure the carrier is the original lossless PNG.",
            )
        except Exception as e:
            logger.error("Extraction execution failure: %s", sanitize_error_message(e))
            raise HTTPException(status_code=500, detail=f"Extraction failure: {sanitize_error_message(e)}")

    rec_session_id = session_store.store(
        data=recovery.recovered_payload.data,
        ttl_seconds=settings.session_ttl_seconds,
        media_type="application/octet-stream",
    )

    resp: Dict[str, Any] = {
        "status": "success",
        "recovered_session_id": rec_session_id,
        "recovered_bytes": len(recovery.recovered_payload.data),
        "media_type": recovery.recovered_payload.media_type.name,
        "session_counter": recovery.header.session_counter,
        "ecc_parity": recovery.header.ecc_parity,
        "protocol_version": recovery.header.version,
        "wire_protocol_version": recovery.header.version,
    }
    if recovery.authenticated_sender_id:
        resp["pqc_authenticated_sender_id"] = recovery.authenticated_sender_id
    return resp


class StreamPipelineRequest(BaseModel):
    secret_session_id: str
    cover_session_id: Optional[str] = None
    key_hex: Optional[str] = Field(None, description="64-character hex string representing a 32-byte key")
    model_name: Optional[str] = Field("neural_spatial_onnx", description="Modulation engine selector")
    precision: Optional[str] = Field("fp32", description="Model precision ('fp32' or 'int8')")
    session_counter: Optional[int] = Field(101, description="Session counter for mask PRNG")
    protocol: Optional[str] = Field(None, description="Wire protocol selector ('v2' or 'v3_mr')")
    version: Optional[int] = Field(None, description="Wire protocol version (2 or 3)")


@api_router.post("/api/stream")
async def stream_media_pipeline(
    req: StreamPipelineRequest,
    token: Optional[str] = Depends(verify_auth),
) -> StreamingResponse:
    """Streams the real media steganography pipeline execution via SSE using the qualified deployment pipeline."""
    settings = get_api_settings()
    is_ready, reason, _ = check_runtime_operational()
    if not is_ready:
        raise HTTPException(status_code=503, detail=f"Steganography runtime is unavailable: {reason}")

    target_version = 2
    if req.protocol is not None:
        p_clean = req.protocol.strip().lower()
        if p_clean in ("v3_mr", "v3-mr", "v3_maximum_resilience"):
            target_version = 3
        elif p_clean in ("v2", "wire_protocol_v2", "2"):
            target_version = 2
        elif p_clean in ("v1", "wire_protocol_v1", "1"):
            target_version = 1
        elif p_clean in ("v3_standard", "v3", "3"):
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported wire protocol '{req.protocol}'. Wire Protocol v3 Standard is an unpromoted experimental protocol. Only v3-MR is supported.",
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported wire protocol '{req.protocol}'. Supported options are 'v2' or 'v3_mr'.",
            )
    elif req.version is not None:
        if req.version not in (1, 2, 3):
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported wire protocol version: {req.version}. Supported versions are 2 (default) or 3 (v3-MR).",
            )
        target_version = req.version

    if target_version == 3 and not settings.enable_v3_mr:
        raise HTTPException(
            status_code=403,
            detail="Wire Protocol v3-MR is disabled on this server. Contact administrator or set PALADIN_ENABLE_V3_MR=true.",
        )

    key = validate_key_hex(req.key_hex, is_remote=settings.is_remote())

    secret_bytes = session_store.get(req.secret_session_id)
    if not secret_bytes:
        raise HTTPException(status_code=404, detail="Secret payload session not found or expired")

    cover_img = resolve_cover_image(req.cover_session_id, is_remote=settings.is_remote())

    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.IMAGE,
        metadata={"filename": "secret_upload.png", "source": "ui_upload"},
    )

    precision_val = (req.precision or "fp32").lower()
    assert qualified_pipeline is not None
    adapter = MediaEventAdapter(pipeline=qualified_pipeline, session_store=session_store)

    async def event_generator():
        last_run_id = None
        try:
            async with engine_admission_gate():
                async for event in adapter.run_media_pipeline(
                    secret_payload=secret_payload,
                    cover_image=cover_img,
                    key=key,
                    session_counter=req.session_counter or 101,
                    model_name=req.model_name or "neural_spatial_onnx",
                    version=target_version,
                    precision=precision_val,
                ):
                    if isinstance(event, dict) and event.get("metadata", {}).get("run_id"):
                        last_run_id = event["metadata"]["run_id"]
                    yield serialize_sse_event(event)
        except (asyncio.CancelledError, GeneratorExit):
            logger.info("Streaming client disconnected or stream cancelled for run_id=%s", last_run_id)
            if last_run_id:
                curr = diagnostics_store.get(last_run_id)
                if curr and curr.verification_status == "PENDING":
                    diagnostics_store.update_verification(
                        run_id=last_run_id,
                        verification_status="ABORTED",
                        exact_match=False,
                        error_message="Stream aborted by client cancellation",
                    )
            raise
        except Exception as e:
            logger.error("Streaming pipeline failure: %s", sanitize_error_message(e))
            if last_run_id:
                curr = diagnostics_store.get(last_run_id)
                if curr and curr.verification_status == "PENDING":
                    diagnostics_store.update_verification(
                        run_id=last_run_id,
                        verification_status="FAILED",
                        exact_match=False,
                        error_message=sanitize_error_message(e),
                    )
            yield serialize_sse_event({"stage": "ERROR", "status": "failed", "error": sanitize_error_message(e)})
        finally:
            if last_run_id:
                curr = diagnostics_store.get(last_run_id)
                if curr and curr.verification_status == "PENDING":
                    diagnostics_store.update_verification(
                        run_id=last_run_id,
                        verification_status="ABORTED",
                        exact_match=False,
                        error_message="Stream terminated prematurely",
                    )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@api_router.get("/api/diagnostics/latest")
async def get_latest_diagnostics(
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Retrieves execution diagnostics for the most recent pipeline run."""
    diag = diagnostics_store.get_latest()
    if diag is None:
        raise HTTPException(status_code=404, detail="No diagnostic data available. Execute a pipeline run first.")
    return diag.to_dict()


@api_router.get("/api/diagnostics/{run_id}")
async def get_run_diagnostics(
    run_id: str,
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Retrieves execution diagnostics for a specific pipeline run."""
    diag = diagnostics_store.get(run_id)
    if diag is None:
        raise HTTPException(status_code=404, detail=f"Diagnostic record for run '{run_id}' not found or expired.")
    return diag.to_dict()


@api_router.get("/api/session/{session_id}")
async def get_session_media(
    session_id: str,
    token: Optional[str] = Depends(verify_auth),
) -> Response:
    """Retrieves stored binary media for a session record."""
    rec = session_store.get_record(session_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Session record not found or expired")
    return Response(
        content=rec.data,
        media_type=rec.media_type or "application/octet-stream",
        headers={
            "Cache-Control": "private, no-cache, no-store, must-revalidate",
            "Content-Disposition": f'inline; filename="{rec.filename or session_id}.png"',
        },
    )


@api_router.get("/api/runs/{run_id}/subbands")
async def get_run_subbands(
    run_id: str,
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Retrieves DWT subband visualizations for a specific run."""
    raw = session_store.get(f"subbands_{run_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Subbands for run '{run_id}' not found or expired.")
    try:
        subbands = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="Corrupted subband metadata record")
    return {
        "run_id": run_id,
        "subbands": subbands,
    }


class PQCIdentityRegistrationRequest(BaseModel):
    identity_id: str
    name: str
    kem_public_key_hex: str
    dsa_public_key_hex: str


@api_router.get("/api/pqc/identities")
async def list_pqc_identities(
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Lists registered PQC identities (without private keys)."""
    settings = get_api_settings()
    if not settings.enable_pqc:
        raise HTTPException(
            status_code=403,
            detail="Post-Quantum Cryptography is disabled on this server. Contact administrator or set PALADIN_ENABLE_PQC=true.",
        )
    from src.security.pqc.trust import default_trust_store

    with default_trust_store._lock:
        identities = [
            {
                "identity_id": ident.identity_id,
                "name": ident.name,
                "kem_fingerprint_hex": ident.kem_fingerprint.hex(),
                "dsa_fingerprint_hex": ident.dsa_fingerprint.hex(),
                "created_at": ident.created_at,
                "expires_at": ident.expires_at,
                "is_revoked": ident.is_revoked,
            }
            for ident in default_trust_store._identities_by_id.values()
        ]
    return {"identities": identities, "count": len(identities)}


@api_router.post("/api/pqc/identities")
async def register_pqc_identity(
    req: PQCIdentityRegistrationRequest,
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Registers a new validated PQC identity into the trust store."""
    settings = get_api_settings()
    if not settings.enable_pqc:
        raise HTTPException(
            status_code=403,
            detail="Post-Quantum Cryptography is disabled on this server. Contact administrator or set PALADIN_ENABLE_PQC=true.",
        )
    from src.security.pqc.models import PQCIdentity
    from src.security.pqc.trust import default_trust_store

    try:
        kem_pk = bytes.fromhex(req.kem_public_key_hex)
        dsa_pk = bytes.fromhex(req.dsa_public_key_hex)
        ident = PQCIdentity(
            identity_id=req.identity_id,
            name=req.name,
            kem_public_key=kem_pk,
            dsa_public_key=dsa_pk,
        )
        default_trust_store.register_identity(ident)
        return {
            "status": "registered",
            "identity_id": ident.identity_id,
            "name": ident.name,
            "kem_fingerprint_hex": ident.kem_fingerprint.hex(),
            "dsa_fingerprint_hex": ident.dsa_fingerprint.hex(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Identity registration failed: {sanitize_error_message(e)}",
        )


def create_app(settings: Optional[APISettings] = None) -> FastAPI:
    """Factory creating and configuring the FastAPI application.

    Note:
        Direct `uvicorn src.api.server:app` startup is unsupported for production
        or remote deployments because Uvicorn CLI bypasses launcher host binding,
        worker count validation, and configuration preflight checks.
        Use `src.api.launcher` (`python -m src.api.launcher`).
    """
    active_settings = settings or get_api_settings()
    active_settings.validate_startup()

    # In remote mode, verify qualified pipeline and ONNX integrity
    if active_settings.is_remote():
        is_ready, reason, _ = check_runtime_operational()
        if not is_ready:
            raise RuntimeError(
                f"REMOTE_STARTUP_PREFLIGHT_FAILED: Remote mode requires operational runtime: {reason}"
            )

    # Synchronize global settings and session store configuration
    reset_api_settings(active_settings)
    session_store.configure(
        max_entries=active_settings.max_sessions,
        default_ttl=active_settings.session_ttl_seconds,
        max_bytes=active_settings.max_session_store_bytes,
    )

    application = FastAPI(
        title="PaladinMedia Qualified Deployment API",
        version="1.0.0",
        description="Operational API for ACS-Sub-INN v3 transform-domain steganography with rs_2t_48 ECC.",
        docs_url="/docs" if active_settings.enable_docs else None,
        redoc_url="/redoc" if active_settings.enable_docs else None,
        openapi_url="/openapi.json" if active_settings.enable_docs else None,
    )

    # CORS configuration
    allow_creds = "*" not in active_settings.allowed_origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.allowed_origins,
        allow_origin_regex=r"^https://paladinmedia(-[a-z0-9]+)?\.vercel\.app$" if active_settings.mode == "demo" else None,
        allow_credentials=allow_creds,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "X-Requested-With",
            "X-CSRF-Token",
            "x-csrf-token",
        ],
    )

    # Security response headers & cache prevention
    application.add_middleware(SecurityHeadersMiddleware)

    # Demo rate limiter (demo mode only: 60 req/min per IP)
    if active_settings.mode == "demo":
        application.add_middleware(DemoRateLimiterMiddleware, requests_per_minute=60, enabled=True)

    # Enforce request body size limit at ASGI boundary before multipart buffering
    # (max_upload_bytes + 64KB for HTTP multipart framing/boundary overhead)
    max_request_body_bytes = active_settings.max_upload_bytes + (64 * 1024)
    application.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=max_request_body_bytes)

    # Register operational routes
    application.include_router(api_router)

    return application


app = create_app(_initial_settings)


# Log observable configuration facts at startup (non-sensitive only)
_cfg = get_deployment_configuration()
logger.info(
    "PaladinMedia Qualified Deployment: mode=%s, auth=%s, backend=%s, ecc=%s (%d symbols), format='%s', onnx_ready=%s",
    _cfg["deployment_mode"],
    _cfg["authentication_enforced"],
    _cfg["inference_backend"],
    _cfg["ecc_profile"],
    _cfg["ecc_parity_symbols"],
    _cfg["carrier_format_policy"],
    _cfg["onnx_models_verified"],
)


if __name__ == "__main__":
    from src.api.launcher import run_launcher
    run_launcher()
