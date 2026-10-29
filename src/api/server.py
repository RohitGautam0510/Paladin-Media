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
import base64
import hashlib
import io
import json
import logging
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
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
    BoundedSessionStore,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    get_api_settings,
    reset_api_settings,
    sanitize_error_message,
    validate_key_hex,
    verify_auth,
)
from src.carrier.capacity import PayloadCapacityExceededError
from src.carrier.header import CarrierHeaderError
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
ONNX_DIR = REPO_ROOT / "checkpoints" / "phase5a_onnx"

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
    width: int = Query(..., description="Carrier width in pixels"),
    height: int = Query(..., description="Carrier height in pixels"),
    token: Optional[str] = Depends(verify_auth),
) -> Dict[str, Any]:
    """Computes dynamic secret capacity limits for a given carrier width and height."""
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
    subband_dim = (height // 2, width // 2)
    if qualified_pipeline is not None:
        limits = qualified_pipeline.get_capacity_limits(subband_dim=subband_dim)
    else:
        from src.carrier.capacity import CapacityCalculator
        raw_cap = CapacityCalculator.calculate_usable_secret_capacity(subband_dim=subband_dim, rs_parity=48)
        limits = {
            "ecc_profile": "rs_2t_48",
            "ecc_parity_symbols": 48,
            "carrier_format_policy": "Lossless 8-bit RGB PNG only. Lossy transcoded carriers (e.g. JPEG) are strictly unsupported.",
            "raw_secret_capacity_bytes": raw_cap,
            "reference_metadata_capacity_bytes": 100,
            "max_protected_bytes": raw_cap,
            "is_qualified_deployment": False,
            "operational_qualification_note": "Qualified ONNX runtime is degraded",
        }
    return {
        "carrier_width": width,
        "carrier_height": height,
        "subband_height": subband_dim[0],
        "subband_width": subband_dim[1],
        **limits,
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
    MAX_PIXEL_COUNT = 4 * 1024 * 1024  # 4 megapixels max
    MAX_DIMENSION = 2048

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

    key = validate_key_hex(req.key_hex, is_remote=settings.is_remote())

    secret_bytes = session_store.get(req.secret_session_id)
    if not secret_bytes:
        raise HTTPException(status_code=404, detail="Secret payload session not found or expired")

    cover_img = resolve_cover_image(req.cover_session_id, is_remote=settings.is_remote())

    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.IMAGE,
        metadata={"filename": "secret_upload.png", "source": "api_embed"},
    )

    try:
        assert qualified_pipeline is not None
        res = qualified_pipeline.protect_and_embed(
            secret_payload=secret_payload,
            cover_image=cover_img,
            key=key,
            session_counter=req.session_counter,
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
        "ecc_profile": "rs_2t_48",
    }


class DirectExtractRequest(BaseModel):
    stego_session_id: str
    key_hex: Optional[str] = Field(None, description="64-character hex string representing a 32-byte key")


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

    try:
        assert qualified_pipeline is not None
        recovery = qualified_pipeline.extract_and_unprotect(stego_carrier=stego_bytes, key=key)
    except UnsupportedCarrierFormatError as e:
        raise HTTPException(status_code=400, detail=sanitize_error_message(e))
    except (CarrierHeaderError, ECCDecodeError, AuthenticationError) as e:
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

    return {
        "status": "success",
        "recovered_session_id": rec_session_id,
        "recovered_bytes": len(recovery.recovered_payload.data),
        "media_type": recovery.recovered_payload.media_type.name,
        "session_counter": recovery.header.session_counter,
        "ecc_parity": recovery.header.ecc_parity,
    }


class StreamPipelineRequest(BaseModel):
    secret_session_id: str
    cover_session_id: Optional[str] = None
    key_hex: Optional[str] = Field(None, description="64-character hex string representing a 32-byte key")


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

    assert qualified_pipeline is not None
    adapter = MediaEventAdapter(pipeline=qualified_pipeline, session_store=session_store)

    async def event_generator():
        last_run_id = None
        try:
            async for event in adapter.run_media_pipeline(
                secret_payload=secret_payload,
                cover_image=cover_img,
                key=key,
                session_counter=101,
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
        allow_credentials=allow_creds,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
    )

    # Security response headers & cache prevention
    application.add_middleware(SecurityHeadersMiddleware)

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
