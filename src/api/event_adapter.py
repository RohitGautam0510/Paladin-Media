"""Event Adapter for PaladinMedia Media-to-Media Steganography.

Streams granular real backend execution events to the frontend:
LOAD -> SERIALIZE -> ENCRYPT -> RS_ENCODE -> CAPACITY_CHECK -> DWT -> EMBED -> IDWT ->
STEGO_WRITE -> RECEIVER_DWT -> EXTRACT -> RS_DECODE -> AES_VERIFY -> SECRET_RECOVER -> EXACT_MATCH_VERIFY.

Wired to the qualified deployment pipeline (rs_2t_48 + ONNX Runtime) by default,
while preserving backward-compatibility for legacy pipeline instances.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from src.api.diagnostics import RunDiagnostics, diagnostics_store
from src.carrier.capacity import CapacityCalculator, PayloadCapacityExceededError
from src.carrier.dwt import HaarDWT2D, subband_to_image, visualize_subbands
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator
from src.core.exceptions import UnsupportedCarrierFormatError
from src.core.media_pipeline import MediaStegoPipeline, calculate_psnr, calculate_ssim
from src.core.pipeline import SecurityPipeline
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.ecc.encoder import ReedSolomonEncoder

# Documented boundary limits for diagnostics and SSE events
DEFAULT_MAX_DIAGNOSTIC_SUBOBJECT_BYTES: int = 512 * 1024  # 512 KB canonical UTF-8 bytes
DEFAULT_MAX_SSE_EVENT_BYTES: int = 1024 * 1024           # 1 MB complete UTF-8 SSE event wire limit


def serialize_sse_event(
    event: Dict[str, Any],
    max_event_bytes: int = DEFAULT_MAX_SSE_EVENT_BYTES,
) -> str:
    """Serializes a pipeline event dictionary into standard SSE wire framing: `data: <json>\n\n`.

    Guarantees that `len(serialized.encode('utf-8')) <= max_event_bytes`.
    Framing includes the leading `data: ` prefix and trailing `\n\n`.

    Bounding policy when event exceeds max_event_bytes:
    1. If `metadata.diagnostics` contains coordinate arrays, truncate coordinates
       until the event fits, setting `coordinates_truncated=True` and `completeness_status="PARTIAL"`.
    2. If still exceeding, omit image payload fields (e.g., `metadata.modified_hh`,
       `metadata.stego_image`, `metadata.recovered_image`) with explicit `*_omitted=True` flags
       rather than corrupting valid base64 data URLs.
    3. If still exceeding, set `metadata.diagnostics` to minimal representation:
       `{"completeness_status": "UNAVAILABLE", "error": "EVENT_SIZE_LIMIT_EXCEEDED"}`.
    4. If base event still exceeds max_event_bytes, trim user-facing message strings.
    """
    raw_json = json.dumps(event)
    line = f"data: {raw_json}\n\n"
    if len(line.encode("utf-8")) <= max_event_bytes:
        return line

    ev = copy.deepcopy(event)
    meta = ev.get("metadata")
    if isinstance(meta, dict):
        # Step 1: Truncate coordinate lists in metadata.diagnostics
        diag = meta.get("diagnostics")
        if isinstance(diag, dict):
            diag["coordinates_truncated"] = True
            diag["completeness_status"] = "PARTIAL"
            emb = list(diag.get("embedded_positions", []))
            chg = list(diag.get("changed_positions", []))

            while emb or chg:
                test_line = f"data: {json.dumps(ev)}\n\n"
                if len(test_line.encode("utf-8")) <= max_event_bytes:
                    return test_line
                emb = emb[:len(emb) // 2]
                chg = chg[:len(chg) // 2]
                diag["embedded_positions"] = emb
                diag["changed_positions"] = chg

            diag["embedded_positions"] = []
            diag["changed_positions"] = []
            test_line = f"data: {json.dumps(ev)}\n\n"
            if len(test_line.encode("utf-8")) <= max_event_bytes:
                return test_line

        # Step 2: Omit image and visual payload fields if still exceeding
        for img_key, flag_key in [
            ("subbands", "subbands_omitted"),
            ("modified_hh", "modified_hh_omitted"),
            ("stego_image", "stego_image_omitted"),
            ("recovered_image", "recovered_image_omitted"),
        ]:
            if img_key in meta and meta[img_key] is not None:
                meta[img_key] = None
                meta[flag_key] = True
                test_line = f"data: {json.dumps(ev)}\n\n"
                if len(test_line.encode("utf-8")) <= max_event_bytes:
                    return test_line

        # Step 3: Minimal diagnostics fallback
        if "diagnostics" in meta:
            meta["diagnostics"] = {
                "run_id": diag.get("run_id") if isinstance(diag, dict) else meta.get("run_id"),
                "completeness_status": "UNAVAILABLE",
                "error": "EVENT_SIZE_LIMIT_EXCEEDED",
            }
            test_line = f"data: {json.dumps(ev)}\n\n"
            if len(test_line.encode("utf-8")) <= max_event_bytes:
                return test_line

    # Step 4: Trim message and error strings
    test_line = f"data: {json.dumps(ev)}\n\n"
    excess = len(test_line.encode("utf-8")) - max_event_bytes
    if excess > 0:
        if "message" in ev and ev["message"]:
            msg = str(ev["message"])
            ev["message"] = msg[:max(10, len(msg) - excess - 25)] + "... [truncated]"
            test_line = f"data: {json.dumps(ev)}\n\n"
            if len(test_line.encode("utf-8")) <= max_event_bytes:
                return test_line

        excess = len(test_line.encode("utf-8")) - max_event_bytes
        if excess > 0 and "error" in ev and ev["error"]:
            err = str(ev["error"])
            ev["error"] = err[:max(10, len(err) - excess - 25)] + "... [truncated]"
            test_line = f"data: {json.dumps(ev)}\n\n"
            if len(test_line.encode("utf-8")) <= max_event_bytes:
                return test_line

    # Step 5: Irreducible event minimal bounded fallback
    test_line = f"data: {json.dumps(ev)}\n\n"
    if len(test_line.encode("utf-8")) <= max_event_bytes:
        return test_line

    stage = str(ev.get("stage", "ERROR"))
    status = str(ev.get("status", "failed"))
    run_id = meta.get("run_id") if isinstance(meta, dict) else None
    exact_match = meta.get("exact_match") if isinstance(meta, dict) and "exact_match" in meta else None

    minimal_meta = {"event_size_exceeded": True}
    if run_id is not None:
        minimal_meta["run_id"] = str(run_id)
    if exact_match is not None:
        minimal_meta["exact_match"] = exact_match

    minimal_ev = {
        "stage": stage,
        "status": status,
        "message": f"Event exceeded maximum permitted wire size ({max_event_bytes} B).",
        "metadata": minimal_meta,
    }
    if status == "failed" or "error" in ev:
        minimal_ev["error"] = "EVENT_SIZE_LIMIT_EXCEEDED"

    final_line = f"data: {json.dumps(minimal_ev)}\n\n"
    if len(final_line.encode("utf-8")) <= max_event_bytes:
        return final_line

    # Step 6: Ultra-minimal wire envelope
    ultra_ev = {"stage": stage, "status": status}
    ultra_line = f"data: {json.dumps(ultra_ev)}\n\n"
    if len(ultra_line.encode("utf-8")) <= max_event_bytes:
        return ultra_line

    raise ValueError(f"Cannot serialize SSE event within {max_event_bytes} bytes")


def pil_to_base64_data_url(img: Image.Image) -> str:
    """Converts a PIL Image into a base64 PNG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def create_bounded_thumbnail(img: Image.Image, max_dim: int = 256) -> Image.Image:
    """Creates a proportional thumbnail if image dimensions exceed max_dim."""
    if max(img.width, img.height) <= max_dim:
        return img
    scale = max_dim / max(img.width, img.height)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


class MediaEventAdapter:
    """Observability adapter executing the real transform-domain pipeline while yielding events."""

    def __init__(
        self,
        pipeline: Optional[MediaStegoPipeline] = None,
        session_store: Optional[Any] = None,
    ) -> None:
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = MediaStegoPipeline.create_qualified_deployment()
        self.security_pipeline = self.pipeline.security_pipeline
        self.session_store = session_store
        if self.session_store is None:
            try:
                from src.api.server import session_store as global_session_store
                self.session_store = global_session_store
            except Exception:
                self.session_store = None

    async def run_media_pipeline(
        self,
        secret_payload: SecretPayload,
        cover_image: Image.Image,
        key: bytes,
        aad: Optional[bytes] = None,
        session_counter: int = 1,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Runs the complete media-to-media pipeline, yielding real events and artifacts."""
        t_start = time.perf_counter()
        run_id = f"run_{session_counter}_{int(time.time() * 1000)}_{uuid.uuid4().hex}"

        # Register initial bounded diagnostic lifecycle record from run start
        diagnostics_store.store(
            RunDiagnostics(
                run_id=run_id,
                timestamp=time.time(),
                session_counter=session_counter,
                carrier_dims=(cover_image.size[0], cover_image.size[1]),
                stego_dims=(0, 0),
                subband_dim=(cover_image.size[1] // 2, cover_image.size[0] // 2),
                header_bits_count=0,
                payload_bits_count=0,
                total_embedded_bits=0,
                changed_coefficients_count=0,
                max_hh_diff=0.0,
                embedded_positions=(),
                changed_positions=(),
                pixel_diff_metrics={},
                verification_status="PENDING",
                exact_match=None,
                coordinates_truncated=False,
                total_embedded_positions_count=0,
                total_changed_positions_count=0,
                completeness_status="UNAVAILABLE",
            )
        )

        # Stage 1: Load Media
        yield {
            "stage": "LOAD",
            "status": "processing",
            "message": f"Loading media: Secret ({len(secret_payload.data)} B), Cover ({cover_image.size[0]}x{cover_image.size[1]} RGB)",
            "metadata": {
                "run_id": run_id,
                "secret_bytes": len(secret_payload.data),
                "cover_dims": f"{cover_image.size[0]}x{cover_image.size[1]}",
            },
        }
        await asyncio.sleep(0.3)

        # Validate cover carrier contract (Lossless 8-bit RGB PNG with even dimensions >= 32x32, <= 2048x2048)
        cov_w, cov_h = cover_image.size
        cov_fmt = getattr(cover_image, "format", None)
        if (
            cov_w < 32
            or cov_h < 32
            or cov_w % 2 != 0
            or cov_h % 2 != 0
            or cov_w > 2048
            or cov_h > 2048
            or (cov_w * cov_h) > 4 * 1024 * 1024
            or cover_image.mode != "RGB"
            or (cov_fmt and cov_fmt.upper() != "PNG")
        ):
            msg = f"Cover carrier must be lossless 8-bit RGB PNG with even dimensions (>= 32x32, <= 2048x2048), got {cov_w}x{cov_h} mode={cover_image.mode}"
            diagnostics_store.update_verification(
                run_id=run_id,
                verification_status="PRE_EMBED_FAILED",
                exact_match=False,
                error_message=msg,
            )
            yield {
                "stage": "LOAD",
                "status": "failed",
                "error": msg,
                "metadata": {"run_id": run_id},
            }
            return

        yield {
            "stage": "LOAD",
            "status": "complete",
            "message": f"Loaded secret payload ({len(secret_payload.data)} bytes) and cover carrier.",
        }

        # Stage 2: Serialize Secret Payload
        yield {
            "stage": "SERIALIZE",
            "status": "processing",
            "message": "Serializing secret payload with canonical binary framing and CRC-32...",
        }
        await asyncio.sleep(0.3)
        try:
            serialized = SecretSerializer.serialize(secret_payload)
            yield {
                "stage": "SERIALIZE",
                "status": "complete",
                "message": f"Payload serialized ({len(serialized)} bytes). CRC-32 verified.",
                "metadata": {"serialized_size": len(serialized)},
            }
        except Exception as e:
            diagnostics_store.update_verification(
                run_id=run_id,
                verification_status="PRE_EMBED_FAILED",
                exact_match=False,
                error_message=str(e),
            )
            yield {"stage": "SERIALIZE", "status": "failed", "error": str(e)}
            return

        # Stage 3: Authenticated Encryption (AES-256-GCM)
        yield {
            "stage": "ENCRYPT",
            "status": "processing",
            "message": "Applying AES-256-GCM authenticated encryption (96-bit nonce + 128-bit GHASH tag)...",
        }
        await asyncio.sleep(0.3)
        try:
            encrypted = self.security_pipeline.crypto_provider.encrypt(serialized, key, aad=aad)
            yield {
                "stage": "ENCRYPT",
                "status": "complete",
                "message": f"Encryption complete ({len(encrypted)} bytes ciphertext + tag).",
                "metadata": {"encrypted_size": len(encrypted)},
            }
        except Exception as e:
            diagnostics_store.update_verification(
                run_id=run_id,
                verification_status="PRE_EMBED_FAILED",
                exact_match=False,
                error_message=str(e),
            )
            yield {"stage": "ENCRYPT", "status": "failed", "error": str(e)}
            return

        # Stage 4: Error Correction Coding (Reed-Solomon GF(2^8))
        ecc_symbols = getattr(self.security_pipeline.ecc_encoder, "parity_symbols", 48)
        yield {
            "stage": "RS_ENCODE",
            "status": "processing",
            "message": f"Applying systematic Reed-Solomon encoding over GF(2^8) (2t={ecc_symbols} parity/block)...",
        }
        await asyncio.sleep(0.3)
        try:
            protected_payload = self.security_pipeline.ecc_encoder.encode(encrypted)
            yield {
                "stage": "RS_ENCODE",
                "status": "complete",
                "message": f"Reed-Solomon encoding complete ({len(protected_payload)} bytes protected payload, profile=rs_2t_{ecc_symbols}).",
                "metadata": {"rs_size": len(protected_payload), "ecc_parity": ecc_symbols},
            }
        except Exception as e:
            diagnostics_store.update_verification(
                run_id=run_id,
                verification_status="PRE_EMBED_FAILED",
                exact_match=False,
                error_message=str(e),
            )
            yield {"stage": "RS_ENCODE", "status": "failed", "error": str(e)}
            return

        # Stage 5: Capacity Analysis & Hard Gate
        subband_dim = (cover_image.size[1] // 2, cover_image.size[0] // 2)
        yield {
            "stage": "CAPACITY_CHECK",
            "status": "processing",
            "message": "Analyzing carrier transform capacity and evaluating hard gate...",
        }
        await asyncio.sleep(0.3)
        try:
            cap_report = CapacityCalculator.check_capacity(
                len(protected_payload), subband_dim=subband_dim, rs_parity=ecc_symbols
            )
            if not cap_report.fits:
                msg = (
                    f"PAYLOAD_CAPACITY_EXCEEDED: Protected payload ({len(protected_payload)} B) "
                    f"exceeds maximum carrier capacity ({cap_report.max_protected_bytes} B). "
                    f"Usable capacity is {cap_report.usable_capacity_bytes} B. Embedding aborted."
                )
                diagnostics_store.update_verification(
                    run_id=run_id,
                    verification_status="PRE_EMBED_FAILED",
                    exact_match=False,
                    error_message=msg,
                )
                yield {
                    "stage": "CAPACITY_CHECK",
                    "status": "failed",
                    "message": msg,
                    "error": "PAYLOAD_CAPACITY_EXCEEDED",
                    "metadata": cap_report.to_dict(),
                }
                return
        except PayloadCapacityExceededError as e:
            sub_h, sub_w = CapacityCalculator.resolve_subband_dims(subband_dim)
            max_protected = (sub_h * sub_w // 8) - CapacityCalculator.CARRIER_HEADER_BYTES
            diagnostics_store.update_verification(
                run_id=run_id,
                verification_status="PRE_EMBED_FAILED",
                exact_match=False,
                error_message=str(e),
            )
            yield {
                "stage": "CAPACITY_CHECK",
                "status": "failed",
                "message": str(e),
                "error": "PAYLOAD_CAPACITY_EXCEEDED",
                "metadata": {
                    "fits": False,
                    "protected_payload_bytes": len(protected_payload),
                    "max_protected_bytes": max_protected,
                    "status": "PAYLOAD_CAPACITY_EXCEEDED",
                },
            }
            return

        yield {
            "stage": "CAPACITY_CHECK",
            "status": "complete",
            "message": (
                f"Capacity verified: {len(protected_payload)} B fits within {cap_report.max_protected_bytes} B allocation. "
                f"Remaining: {cap_report.remaining_capacity_bytes} B."
            ),
            "metadata": cap_report.to_dict(),
        }

        # Stage 6: Carrier 2D Haar DWT
        sub_h, sub_w = cover_image.size[1] // 2, cover_image.size[0] // 2
        yield {
            "stage": "DWT",
            "status": "processing",
            "message": f"Computing one-level 2D Haar Discrete Wavelet Transform on {cover_image.size[0]}x{cover_image.size[1]} RGB carrier...",
        }
        await asyncio.sleep(0.4)
        try:
            cover_arr = np.array(cover_image.convert("RGB"), dtype=np.uint8)
            subbands = HaarDWT2D.forward(cover_arr)
            subband_visuals = visualize_subbands(subbands)
            dwt_data_urls = {
                k: pil_to_base64_data_url(create_bounded_thumbnail(v, max_dim=256))
                for k, v in subband_visuals.items()
            }

            if self.session_store is not None:
                try:
                    self.session_store.store(
                        data=json.dumps(dwt_data_urls).encode("utf-8"),
                        ttl_seconds=600,
                        media_type="application/json",
                        custom_session_id=f"subbands_{run_id}",
                    )
                except Exception as store_err:
                    logger.debug("Could not cache subbands in session store: %s", store_err)

            yield {
                "stage": "DWT",
                "status": "complete",
                "message": f"Haar DWT complete: LL, LH, HL, HH subbands generated (3, {sub_h}, {sub_w}).",
                "metadata": {
                    "subbands": dwt_data_urls,
                    "subband_width": sub_w,
                    "subband_height": sub_h,
                    "run_id": run_id,
                },
            }
        except Exception as e:
            diagnostics_store.update_verification(
                run_id=run_id,
                verification_status="PRE_EMBED_FAILED",
                exact_match=False,
                error_message=str(e),
            )
            yield {"stage": "DWT", "status": "failed", "error": str(e)}
            return

        # Stage 7: Deterministic Mask & Transform-Domain Embedding
        engine_name = self.pipeline.modulation_engine.__class__.__name__
        yield {
            "stage": "EMBED",
            "status": "processing",
            "message": f"Generating deterministic position mask and modulating HH channel via {engine_name}...",
        }
        await asyncio.sleep(0.4)
        try:
            header = CarrierHeader.create(
                session_counter=session_counter,
                payload_length=len(protected_payload),
                ecc_parity=ecc_symbols,
            )
            header_bytes = header.pack()
            mask = DeterministicMaskGenerator.generate_mask(
                session_counter=session_counter,
                payload_length_bytes=len(protected_payload),
                subband_dim=subband_dim,
            )
            header_bits = BaselineEmbedder.bytes_to_bits(header_bytes)
            payload_bits = BaselineEmbedder.bytes_to_bits(protected_payload)

            stego_arr = self.pipeline.modulation_engine.embed_bits(
                cover_array=cover_arr,
                header_bits=header_bits,
                payload_bits=payload_bits,
                mask=mask,
            )
            stego_pil = Image.fromarray(stego_arr, mode="RGB")
            mod_subbands = HaarDWT2D.forward(stego_arr)
            mod_hh_img = subband_to_image(mod_subbands.hh, is_ll=False)
            mod_hh_url = pil_to_base64_data_url(create_bounded_thumbnail(mod_hh_img, max_dim=384))

            # Extract genuine transform diagnostic coordinates
            header_positions = [(int(p[0]), int(p[1])) for p in mask.header_positions]
            payload_positions = [(int(p[0]), int(p[1])) for p in mask.payload_positions]
            embedded_positions = header_positions + payload_positions

            hh_orig = subbands.hh[0]
            hh_mod = mod_subbands.hh[0]
            hh_diff = np.abs(hh_mod - hh_orig)
            changed_indices = np.argwhere(hh_diff > 1e-4)
            changed_positions = [(int(c[0]), int(c[1])) for c in changed_indices]
            changed_count = int(len(changed_positions))
            max_hh_diff = float(np.max(hh_diff)) if hh_diff.size > 0 else 0.0

            # Apply single unified diagnostic budgeting policy before SSE emission and store
            bounded_diag = diagnostics_store.create_bounded_record(
                run_id=run_id,
                timestamp=time.time(),
                session_counter=session_counter,
                carrier_dims=(int(cover_arr.shape[1]), int(cover_arr.shape[0])),
                stego_dims=(int(stego_arr.shape[1]), int(stego_arr.shape[0])),
                subband_dim=subband_dim,
                header_bits_count=int(len(header_bits)),
                payload_bits_count=int(len(payload_bits)),
                total_embedded_bits=int(len(header_bits) + len(payload_bits)),
                changed_coefficients_count=changed_count,
                max_hh_diff=round(max_hh_diff, 4),
                embedded_positions=embedded_positions,
                changed_positions=changed_positions,
                pixel_diff_metrics={},
                verification_status="PENDING",
                exact_match=None,
                total_embedded_positions_count=int(len(header_bits) + len(payload_bits)),
                total_changed_positions_count=changed_count,
            )

            # Store bounded record immediately
            diagnostics_store.store(bounded_diag)

            # Emit the EXACT same bounded dictionary in SSE
            diagnostics_summary = bounded_diag.to_dict()

            yield {
                "stage": "EMBED",
                "status": "complete",
                "message": f"Carrier header (16B, flags={header.flags:#04x}) and {len(protected_payload)}B payload modulated into HH subband.",
                "metadata": {
                    "modified_hh": mod_hh_url,
                    "session_counter": session_counter,
                    "header_flags": header.flags,
                    "ecc_profile": f"rs_2t_{ecc_symbols}",
                    "run_id": run_id,
                    "diagnostics": diagnostics_summary,
                },
            }
        except Exception as e:
            diagnostics_store.update_verification(
                run_id=run_id,
                verification_status="FAILED",
                exact_match=False,
                error_message=str(e),
            )
            yield {"stage": "EMBED", "status": "failed", "error": str(e)}
            return

        # Stage 8: IDWT & Stego PNG Synthesis
        yield {
            "stage": "IDWT",
            "status": "processing",
            "message": "Performing Inverse DWT and encoding 8-bit RGB Stego PNG...",
        }
        await asyncio.sleep(0.4)
        try:
            png_buf = io.BytesIO()
            stego_pil.save(png_buf, format="PNG")
            stego_png_bytes = png_buf.getvalue()

            stego_session_id: Optional[str] = None
            if self.session_store is not None:
                try:
                    stego_session_id = self.session_store.store(
                        data=stego_png_bytes,
                        ttl_seconds=600,
                        media_type="image/png",
                        filename="stego.png",
                        custom_session_id=f"stego_{run_id}",
                    )
                except Exception as store_err:
                    logger.debug("Could not cache stego image in session store: %s", store_err)

            # Generate bounded visual preview for streaming if stego PNG is large
            if len(stego_png_bytes) > 400_000:
                stego_preview_pil = create_bounded_thumbnail(stego_pil, max_dim=512)
                stego_url = pil_to_base64_data_url(stego_preview_pil)
                is_preview = True
            else:
                stego_url = pil_to_base64_data_url(stego_pil)
                is_preview = False

            psnr_val = calculate_psnr(cover_arr, stego_arr)
            ssim_val = calculate_ssim(cover_arr, stego_arr)

            pixel_diff = np.abs(stego_arr.astype(int) - cover_arr.astype(int))
            changed_pixel_mask = np.any(pixel_diff > 0, axis=-1)
            changed_pixels_count = int(np.sum(changed_pixel_mask))
            max_channel_diff = int(np.max(pixel_diff))
            pixel_diff_metrics = {
                "changed_pixels_count": changed_pixels_count,
                "max_channel_diff": max_channel_diff,
                "total_pixels": int(cover_arr.shape[0] * cover_arr.shape[1]),
                "changed_percentage": round((changed_pixels_count / (cover_arr.shape[0] * cover_arr.shape[1])) * 100, 2),
            }

            # Update existing RunDiagnostics with genuine pixel diff metrics
            diagnostics_store.update_pixel_diff(run_id, pixel_diff_metrics)

            yield {
                "stage": "IDWT",
                "status": "complete",
                "message": f"Stego PNG generated ({len(stego_png_bytes)} B). Fidelity: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.5f}.",
                "metadata": {
                    "stego_image": stego_url,
                    "stego_session_id": stego_session_id,
                    "stego_download_url": f"/api/session/{stego_session_id}" if stego_session_id else None,
                    "is_preview": is_preview,
                    "psnr_db": round(psnr_val, 2),
                    "ssim": round(ssim_val, 5),
                    "stego_size_bytes": len(stego_png_bytes),
                    "carrier_format": "PNG",
                    "output_format_policy": "Lossless 8-bit RGB PNG only. Do not transcode or compress to JPEG.",
                    "run_id": run_id,
                    "pixel_diff_metrics": pixel_diff_metrics,
                },
            }
        except Exception as e:
            diagnostics_store.update_verification(
                run_id=run_id,
                verification_status="FAILED",
                exact_match=False,
                error_message=str(e),
            )
            yield {"stage": "IDWT", "status": "failed", "error": str(e)}
            return

        # Stage 9: Receiver Side: Stego PNG Reload & Format Verification
        yield {
            "stage": "RECEIVER_DWT",
            "status": "processing",
            "message": "Receiver reading stego container and verifying carrier format...",
        }
        await asyncio.sleep(0.3)
        try:
            with Image.open(io.BytesIO(stego_png_bytes)) as probed:
                carrier_format = probed.format
            if carrier_format and carrier_format.upper() in ("JPEG", "JPG"):
                raise UnsupportedCarrierFormatError(
                    "JPEG-transcoded carriers are not qualified for the current deployment. "
                    "PaladinMedia requires the original lossless 8-bit RGB PNG carrier."
                )
            reloaded_img = Image.open(io.BytesIO(stego_png_bytes)).convert("RGB")
            reloaded_arr = np.array(reloaded_img, dtype=np.uint8)
            yield {
                "stage": "RECEIVER_DWT",
                "status": "complete",
                "message": f"Receiver carrier container verified (format={carrier_format}).",
            }
        except Exception as e:
            diagnostics_store.update_verification(run_id=run_id, verification_status="FAILED", exact_match=False, error_message=str(e))
            yield {"stage": "RECEIVER_DWT", "status": "failed", "error": str(e)}
            return

        # Stage 10: Extraction from Transform Domain
        yield {
            "stage": "EXTRACT",
            "status": "processing",
            "message": "Extracting carrier header, reconstructing mask, and extracting payload bits...",
        }
        await asyncio.sleep(0.3)
        try:
            bit_extractor = self.pipeline.modulation_engine.begin_extraction(reloaded_arr, subband_dim=subband_dim)
            hdr_positions = DeterministicMaskGenerator.generate_header_positions(subband_dim=subband_dim)
            hdr_bits = bit_extractor.extract_bits(hdr_positions)
            hdr_bytes = Extractor.bits_to_bytes(hdr_bits)
            carrier_hdr = CarrierHeader.unpack(hdr_bytes)

            payload_positions = DeterministicMaskGenerator.generate_payload_positions(
                session_counter=carrier_hdr.session_counter,
                payload_length_bytes=carrier_hdr.payload_length,
                subband_dim=subband_dim,
            )
            extracted_bits = bit_extractor.extract_bits(payload_positions)
            extracted_protected = Extractor.bits_to_bytes(extracted_bits)

            yield {
                "stage": "EXTRACT",
                "status": "complete",
                "message": (
                    f"Carrier header verified (session={carrier_hdr.session_counter}, "
                    f"length={carrier_hdr.payload_length}B, profile=rs_2t_{carrier_hdr.ecc_parity}). "
                    f"Payload extracted."
                ),
                "metadata": {
                    "session_counter": carrier_hdr.session_counter,
                    "payload_length": carrier_hdr.payload_length,
                    "ecc_parity": carrier_hdr.ecc_parity,
                },
            }
        except Exception as e:
            diagnostics_store.update_verification(run_id=run_id, verification_status="FAILED", exact_match=False, error_message=str(e))
            yield {"stage": "EXTRACT", "status": "failed", "error": str(e)}
            return

        # Stage 11: Reed-Solomon Decoding
        yield {
            "stage": "RS_DECODE",
            "status": "processing",
            "message": f"Executing Reed-Solomon error correction decoding with profile rs_2t_{carrier_hdr.ecc_parity}...",
        }
        await asyncio.sleep(0.3)
        try:
            decoder = ReedSolomonEncoder(parity_symbols=carrier_hdr.ecc_parity)
            ecc_decoded = decoder.decode(extracted_protected)
            yield {
                "stage": "RS_DECODE",
                "status": "complete",
                "message": f"Reed-Solomon decoding successful ({len(ecc_decoded)} bytes ciphertext restored).",
            }
        except Exception as e:
            diagnostics_store.update_verification(run_id=run_id, verification_status="FAILED", exact_match=False, error_message=str(e))
            yield {"stage": "RS_DECODE", "status": "failed", "error": str(e)}
            return

        # Stage 12: AES-GCM Authentication & Decryption
        yield {
            "stage": "AES_VERIFY",
            "status": "processing",
            "message": "Verifying AES-256-GCM authentication tag and decrypting...",
        }
        await asyncio.sleep(0.3)
        try:
            decrypted = self.security_pipeline.crypto_provider.decrypt(ecc_decoded, key, aad=aad)
            yield {
                "stage": "AES_VERIFY",
                "status": "complete",
                "message": "AES-256-GCM authentication tag VERIFIED. Decryption successful.",
            }
        except Exception as e:
            diagnostics_store.update_verification(run_id=run_id, verification_status="FAILED", exact_match=False, error_message=str(e))
            yield {"stage": "AES_VERIFY", "status": "failed", "error": str(e)}
            return

        # Stage 13: Deserialization & Secret Recovery
        yield {
            "stage": "SECRET_RECOVER",
            "status": "processing",
            "message": "Deserializing payload structure and validating CRC-32 integrity...",
        }
        await asyncio.sleep(0.3)
        try:
            recovered_payload = SecretSerializer.deserialize(decrypted)
            recovered_img_url: Optional[str] = None
            rec_dims: Optional[List[int]] = None
            if recovered_payload.media_type == MediaType.IMAGE:
                try:
                    rec_pil = Image.open(io.BytesIO(recovered_payload.data))
                    rec_dims = [rec_pil.size[0], rec_pil.size[1]]
                    recovered_img_url = pil_to_base64_data_url(rec_pil)
                except Exception:
                    pass

            orig_dims: Optional[List[int]] = None
            if secret_payload.media_type == MediaType.IMAGE:
                try:
                    with Image.open(io.BytesIO(secret_payload.data)) as o_img:
                        orig_dims = [o_img.size[0], o_img.size[1]]
                except Exception:
                    pass

            yield {
                "stage": "SECRET_RECOVER",
                "status": "complete",
                "message": f"Secret recovered ({len(recovered_payload.data)} bytes, type={recovered_payload.media_type.name}).",
                "metadata": {
                    "recovered_image": recovered_img_url,
                    "media_type": recovered_payload.media_type.name,
                    "recovered_bytes": len(recovered_payload.data),
                    "original_bytes": len(secret_payload.data),
                    "recovered_dims": rec_dims,
                    "original_dims": orig_dims,
                    "run_id": run_id,
                },
            }
        except Exception as e:
            diagnostics_store.update_verification(run_id=run_id, verification_status="FAILED", exact_match=False, error_message=str(e))
            yield {"stage": "SECRET_RECOVER", "status": "failed", "error": str(e)}
            return

        # Stage 14: Verbatim Exact-Match Byte Equality Check
        exact_match = (
            recovered_payload.data == secret_payload.data
            and recovered_payload.media_type == secret_payload.media_type
            and recovered_payload.metadata == secret_payload.metadata
        )

        # Update diagnostic store with verified terminal outcome
        diagnostics_store.update_verification(
            run_id=run_id,
            verification_status="VERIFIED" if exact_match else "FAILED",
            exact_match=exact_match,
        )

        elapsed = time.perf_counter() - t_start
        yield {
            "stage": "EXACT_MATCH_VERIFY",
            "status": "complete" if exact_match else "failed",
            "message": (
                "EXACT RECOVERY VERIFIED: 100% byte-for-byte match."
                if exact_match
                else "RECOVERY FAILED: Decoded payload bytes mismatch."
            ),
            "metadata": {
                "exact_match": exact_match,
                "elapsed_seconds": round(elapsed, 2),
                "psnr_db": round(psnr_val, 2),
                "ssim": round(ssim_val, 5),
                "bytes_verified": len(secret_payload.data),
                "run_id": run_id,
            },
        }
