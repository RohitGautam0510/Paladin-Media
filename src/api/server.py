"""FastAPI Backend Server for PaladinMedia Phase 2 Observability Layer."""

import base64
import io
import json
import os
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from src.api.event_adapter import MediaEventAdapter, pil_to_base64_data_url
from src.core.serialization.serializer import MediaType, SecretPayload

app = FastAPI(title="PaladinMedia Phase 2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for media uploads
session_store: Dict[str, bytes] = {}

# Preload default sample fixtures
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SECRET_FIXTURE_PATH = os.path.join(REPO_ROOT, "tests", "fixtures", "secret_small.png")
COVER_FIXTURE_PATH = os.path.join(REPO_ROOT, "MVP 0 Dataset", "0001x2.png")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads secret or cover media, storing bytes under a session id."""
    data = await file.read()
    session_id = os.urandom(8).hex()
    session_store[session_id] = data
    return {"session_id": session_id, "size": len(data), "filename": file.filename}


@app.get("/api/fixtures/defaults")
async def get_default_fixtures():
    """Provides base64 sample fixtures for instant laboratory evaluation."""
    secret_b64 = None
    secret_session_id = None
    if os.path.exists(SECRET_FIXTURE_PATH):
        with open(SECRET_FIXTURE_PATH, "rb") as f:
            sec_bytes = f.read()
            secret_b64 = f"data:image/png;base64,{base64.b64encode(sec_bytes).decode('ascii')}"
            secret_session_id = "default_secret"
            session_store[secret_session_id] = sec_bytes

    cover_b64 = None
    cover_session_id = None
    if os.path.exists(COVER_FIXTURE_PATH):
        with open(COVER_FIXTURE_PATH, "rb") as f:
            cov_bytes = f.read()
            cover_b64 = f"data:image/png;base64,{base64.b64encode(cov_bytes).decode('ascii')}"
            cover_session_id = "default_cover"
            session_store[cover_session_id] = cov_bytes

    return {
        "secret": {
            "session_id": secret_session_id,
            "data_url": secret_b64,
            "name": "secret_small.png (24x24)",
            "size": len(session_store.get(secret_session_id, b"")),
        },
        "cover": {
            "session_id": cover_session_id,
            "data_url": cover_b64,
            "name": "0001x2.png (512x512)",
            "size": len(session_store.get(cover_session_id, b"")),
        },
    }


@app.get("/api/stream")
async def stream_media_pipeline(
    secret_session_id: str,
    cover_session_id: Optional[str] = None,
    key_hex: Optional[str] = None,
):
    """Streams the real media steganography pipeline execution via SSE."""
    secret_bytes = session_store.get(secret_session_id)
    if not secret_bytes:
        raise HTTPException(status_code=400, detail="Secret payload session not found")

    cover_bytes = session_store.get(cover_session_id) if cover_session_id else None
    if cover_bytes:
        cover_img = Image.open(io.BytesIO(cover_bytes)).convert("RGB")
    elif os.path.exists(COVER_FIXTURE_PATH):
        cover_img = Image.open(COVER_FIXTURE_PATH).convert("RGB")
    else:
        cover_img = Image.new("RGB", (512, 512), color=(128, 128, 128))

    # Ensure cover image is resized/cropped to 512x512 RGB if needed
    if cover_img.size != (512, 512):
        cover_img = cover_img.resize((512, 512), Image.Resampling.LANCZOS)

    key = bytes.fromhex(key_hex) if key_hex else bytes.fromhex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")

    # Construct SecretPayload
    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.IMAGE,
        metadata={"filename": "secret_upload.png", "source": "ui_upload"},
    )

    adapter = MediaEventAdapter()

    async def event_generator():
        try:
            async for event in adapter.run_media_pipeline(
                secret_payload=secret_payload,
                cover_image=cover_img,
                key=key,
                session_counter=101,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'ERROR', 'status': 'failed', 'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
