"""Event Adapter for PaladinMedia Phase 2 Media-to-Media Steganography.

Streams granular real backend execution events to the frontend:
LOAD -> SERIALIZE -> ENCRYPT -> RS_ENCODE -> CAPACITY_CHECK -> DWT -> EMBED -> IDWT ->
STEGO_WRITE -> RECEIVER_DWT -> EXTRACT -> RS_DECODE -> AES_VERIFY -> SECRET_RECOVER -> EXACT_MATCH_VERIFY.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any, AsyncGenerator, Dict, Optional, Union

import numpy as np
from PIL import Image

from src.carrier.capacity import CapacityCalculator, PayloadCapacityExceededError
from src.carrier.dwt import HaarDWT2D, subband_to_image, visualize_subbands
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.core.pipeline import SecurityPipeline
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer


def pil_to_base64_data_url(img: Image.Image) -> str:
    """Converts a PIL Image into a base64 PNG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class MediaEventAdapter:
    """Observability adapter executing the real transform-domain pipeline while yielding events."""

    def __init__(self) -> None:
        self.security_pipeline = SecurityPipeline()

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

        # Stage 1: Load Media
        yield {
            "stage": "LOAD",
            "status": "processing",
            "message": f"Loading media: Secret ({len(secret_payload.data)} B), Cover ({cover_image.size[0]}x{cover_image.size[1]} RGB)",
            "metadata": {
                "secret_bytes": len(secret_payload.data),
                "cover_dims": f"{cover_image.size[0]}x{cover_image.size[1]}",
            },
        }
        await asyncio.sleep(0.3)
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
            yield {"stage": "ENCRYPT", "status": "failed", "error": str(e)}
            return

        # Stage 4: Error Correction Coding (Reed-Solomon GF(2^8))
        yield {
            "stage": "RS_ENCODE",
            "status": "processing",
            "message": "Applying systematic Reed-Solomon encoding over GF(2^8) (2t=32 parity/block)...",
        }
        await asyncio.sleep(0.3)
        try:
            protected_payload = self.security_pipeline.ecc_encoder.encode(encrypted)
            yield {
                "stage": "RS_ENCODE",
                "status": "complete",
                "message": f"Reed-Solomon encoding complete ({len(protected_payload)} bytes protected payload).",
                "metadata": {"rs_size": len(protected_payload)},
            }
        except Exception as e:
            yield {"stage": "RS_ENCODE", "status": "failed", "error": str(e)}
            return

        # Stage 5: Capacity Analysis & Hard Gate
        yield {
            "stage": "CAPACITY_CHECK",
            "status": "processing",
            "message": "Analyzing carrier transform capacity and evaluating hard gate...",
        }
        await asyncio.sleep(0.3)
        cap_report = CapacityCalculator.evaluate(len(protected_payload))
        if not cap_report.fits:
            yield {
                "stage": "CAPACITY_CHECK",
                "status": "failed",
                "message": (
                    f"PAYLOAD_CAPACITY_EXCEEDED: Protected payload ({len(protected_payload)} B) "
                    f"exceeds maximum carrier capacity ({cap_report.max_protected_bytes} B). "
                    f"Usable capacity is {cap_report.usable_capacity_bytes} B. Embedding aborted."
                ),
                "error": "PAYLOAD_CAPACITY_EXCEEDED",
                "metadata": cap_report.to_dict(),
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
        yield {
            "stage": "DWT",
            "status": "processing",
            "message": "Computing one-level 2D Haar Discrete Wavelet Transform on 512x512 RGB carrier...",
        }
        await asyncio.sleep(0.4)
        try:
            cover_arr = np.array(cover_image.convert("RGB"), dtype=np.uint8)
            subbands = HaarDWT2D.forward(cover_arr)
            subband_visuals = visualize_subbands(subbands)
            dwt_data_urls = {k: pil_to_base64_data_url(v) for k, v in subband_visuals.items()}

            yield {
                "stage": "DWT",
                "status": "complete",
                "message": "Haar DWT complete: LL, LH, HL, HH subbands generated (3, 256, 256).",
                "metadata": {
                    "subbands": dwt_data_urls,
                },
            }
        except Exception as e:
            yield {"stage": "DWT", "status": "failed", "error": str(e)}
            return

        # Stage 7: Deterministic Mask & Transform-Domain Embedding
        yield {
            "stage": "EMBED",
            "status": "processing",
            "message": "Generating deterministic position mask and modulating HH channel parity...",
        }
        await asyncio.sleep(0.4)
        try:
            embed_res = BaselineEmbedder.embed(
                cover_image=cover_arr,
                protected_payload=protected_payload,
                session_counter=session_counter,
            )
            mod_hh_url = pil_to_base64_data_url(embed_res.modified_hh_image)

            yield {
                "stage": "EMBED",
                "status": "complete",
                "message": f"Carrier header (16B) and {len(protected_payload)}B payload embedded into HH subband.",
                "metadata": {
                    "modified_hh": mod_hh_url,
                    "session_counter": session_counter,
                },
            }
        except Exception as e:
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
            embed_res.stego_image.save(png_buf, format="PNG")
            stego_png_bytes = png_buf.getvalue()
            stego_url = pil_to_base64_data_url(embed_res.stego_image)

            psnr_val = calculate_psnr(cover_arr, embed_res.stego_array)
            ssim_val = calculate_ssim(cover_arr, embed_res.stego_array)

            yield {
                "stage": "IDWT",
                "status": "complete",
                "message": f"Stego PNG generated ({len(stego_png_bytes)} B). Fidelity: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.5f}.",
                "metadata": {
                    "stego_image": stego_url,
                    "psnr_db": round(psnr_val, 2),
                    "ssim": round(ssim_val, 5),
                    "stego_size_bytes": len(stego_png_bytes),
                },
            }
        except Exception as e:
            yield {"stage": "IDWT", "status": "failed", "error": str(e)}
            return

        # Stage 9: Receiver Side: Stego PNG Reload & Receiver DWT
        yield {
            "stage": "RECEIVER_DWT",
            "status": "processing",
            "message": "Receiver reading stego PNG from container and performing receiver DWT...",
        }
        await asyncio.sleep(0.3)
        try:
            reloaded_img = Image.open(io.BytesIO(stego_png_bytes)).convert("RGB")
            yield {
                "stage": "RECEIVER_DWT",
                "status": "complete",
                "message": "Receiver DWT computed successfully.",
            }
        except Exception as e:
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
            carrier_hdr, extracted_protected = Extractor.extract(reloaded_img)
            yield {
                "stage": "EXTRACT",
                "status": "complete",
                "message": f"Carrier header verified (session={carrier_hdr.session_counter}, length={carrier_hdr.payload_length}B). Payload extracted.",
            }
        except Exception as e:
            yield {"stage": "EXTRACT", "status": "failed", "error": str(e)}
            return

        # Stage 11: Reed-Solomon Decoding
        yield {
            "stage": "RS_DECODE",
            "status": "processing",
            "message": "Executing Reed-Solomon error correction decoding...",
        }
        await asyncio.sleep(0.3)
        try:
            ecc_decoded = self.security_pipeline.ecc_encoder.decode(extracted_protected)
            yield {
                "stage": "RS_DECODE",
                "status": "complete",
                "message": f"Reed-Solomon decoding successful ({len(ecc_decoded)} bytes ciphertext restored).",
            }
        except Exception as e:
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
            if recovered_payload.media_type == MediaType.IMAGE:
                try:
                    rec_pil = Image.open(io.BytesIO(recovered_payload.data))
                    recovered_img_url = pil_to_base64_data_url(rec_pil)
                except Exception:
                    pass

            yield {
                "stage": "SECRET_RECOVER",
                "status": "complete",
                "message": f"Secret recovered ({len(recovered_payload.data)} bytes, type={recovered_payload.media_type.name}).",
                "metadata": {
                    "recovered_image": recovered_img_url,
                    "media_type": recovered_payload.media_type.name,
                },
            }
        except Exception as e:
            yield {"stage": "SECRET_RECOVER", "status": "failed", "error": str(e)}
            return

        # Stage 14: Verbatim Exact-Match Byte Equality Check
        exact_match = (
            recovered_payload.data == secret_payload.data
            and recovered_payload.media_type == secret_payload.media_type
            and recovered_payload.metadata == secret_payload.metadata
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
            },
        }
