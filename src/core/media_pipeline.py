"""Media-to-Media Steganography Pipeline for PaladinMedia MVP 0 / Phase 2.

Integrates:
1. Secret Media Serialization (SecretSerializer, CRC-32)
2. Authenticated Encryption (AES-256-GCM, fail-closed)
3. Error Correction Coding (Reed-Solomon GF(2^8))
4. Capacity Hard Gate (CapacityCalculator)
5. Carrier Steganographic Framing (CarrierHeader)
6. Discrete Wavelet Transform (1-level 2D Haar DWT / IDWT)
7. Deterministic Mask Placement (DeterministicMaskGenerator)
8. Baseline Transform-Domain Embedding & Extraction (BaselineEmbedder, Extractor)
9. Lossless 8-bit RGB PNG Carrier Serialization & Reload
10. Exact-Match Cryptographic and Media Verification

Architectural Invariants:
- INV-01: Strict unidirectional sequence: Serialize -> Encrypt -> ECC -> Embed.
- INV-02: DWT and embedding operate on protected ciphertext (never plaintext).
- INV-03: Hard-gate capacity check aborts with PayloadCapacityExceededError if payload exceeds allocation.
- INV-04: Fail-closed security on tampering.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.carrier.capacity import CapacityCalculator, CapacityReport, PayloadCapacityExceededError
from src.carrier.dwt import DWTSubbands, HaarDWT2D, visualize_subbands
from src.carrier.embedding import BaselineEmbedder, EmbeddingResult
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.core.pipeline import SecurityPipeline
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.crypto.provider import AESGCMProvider, CryptoProvider
from src.security.ecc.encoder import ECCEncoder, ReedSolomonEncoder

__all__ = [
    "MediaStegoPipeline",
    "MediaStegoResult",
    "MediaRecoveryResult",
    "calculate_psnr",
    "calculate_ssim",
]


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB between two RGB images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM) between two RGB images."""
    arr1 = img1.astype(np.float64)
    arr2 = img2.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    sigma1_sq = np.var(arr1)
    sigma2_sq = np.var(arr2)
    sigma12 = np.mean((arr1 - mu1) * (arr2 - mu2))

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(numerator / denominator)


@dataclass(frozen=True)
class MediaStegoResult:
    """Result emitted by the sender media embedding pipeline."""

    stego_image: Image.Image
    stego_png_bytes: bytes
    capacity_report: CapacityReport
    dwt_visualizations: Dict[str, Image.Image]
    modified_hh_image: Image.Image
    psnr_db: float
    ssim: float
    session_counter: int
    header: CarrierHeader


@dataclass(frozen=True)
class MediaRecoveryResult:
    """Result emitted by the receiver extraction and recovery pipeline."""

    recovered_payload: SecretPayload
    header: CarrierHeader
    exact_match: bool
    recovered_image: Optional[Image.Image] = None


class MediaStegoPipeline:
    """End-to-end media-to-media steganography pipeline."""

    def __init__(
        self,
        crypto_provider: Optional[CryptoProvider] = None,
        ecc_encoder: Optional[ECCEncoder] = None,
    ) -> None:
        self.security_pipeline = SecurityPipeline(
            crypto_provider=crypto_provider,
            ecc_encoder=ecc_encoder,
        )

    def protect_and_embed(
        self,
        secret_payload: SecretPayload,
        cover_image: Union[np.ndarray, Image.Image],
        key: bytes,
        aad: Optional[bytes] = None,
        session_counter: int = 1,
    ) -> MediaStegoResult:
        """Executes sender-side protection, transform, and carrier embedding.

        Sequence:
            1. SecurityPipeline.protect: Serialize -> AES-256-GCM -> Reed-Solomon.
            2. Capacity hard-gate: Abort if protected payload > 8,176 bytes.
            3. DWT on 512x512 RGB cover.
            4. Embed 16B header + protected payload into HH channel 0.
            5. IDWT & serialize to lossless 8-bit RGB PNG.
            6. Compute fidelity metrics (PSNR, SSIM).
        """
        # Ensure cover image is PIL Image or valid array
        if isinstance(cover_image, np.ndarray):
            cover_arr = cover_image
            cover_pil = Image.fromarray(cover_image, mode="RGB")
        else:
            cover_pil = cover_image.convert("RGB")
            cover_arr = np.array(cover_pil, dtype=np.uint8)

        subband_dim = cover_arr.shape[0] // 2

        # Step 1: Cryptographic protection & ECC
        protected_payload = self.security_pipeline.protect(secret_payload, key, aad=aad)

        # Step 2: Capacity check (Hard gate)
        capacity_report = CapacityCalculator.check_capacity(
            len(protected_payload), subband_dim=subband_dim
        )

        # Step 3 & 4: DWT and transform-domain embedding
        embed_result = BaselineEmbedder.embed(
            cover_image=cover_arr,
            protected_payload=protected_payload,
            session_counter=session_counter,
        )

        # Step 5: Serialize to PNG bytes
        png_buffer = io.BytesIO()
        embed_result.stego_image.save(png_buffer, format="PNG")
        stego_png_bytes = png_buffer.getvalue()

        # Step 6: Generate visual artifacts and metrics
        dwt_visuals = visualize_subbands(embed_result.original_subbands)
        psnr_val = calculate_psnr(cover_arr, embed_result.stego_array)
        ssim_val = calculate_ssim(cover_arr, embed_result.stego_array)

        return MediaStegoResult(
            stego_image=embed_result.stego_image,
            stego_png_bytes=stego_png_bytes,
            capacity_report=capacity_report,
            dwt_visualizations=dwt_visuals,
            modified_hh_image=embed_result.modified_hh_image,
            psnr_db=psnr_val,
            ssim=ssim_val,
            session_counter=session_counter,
            header=embed_result.header,
        )

    def extract_and_unprotect(
        self,
        stego_carrier: Union[np.ndarray, Image.Image, bytes],
        key: bytes,
        aad: Optional[bytes] = None,
        reference_payload: Optional[SecretPayload] = None,
    ) -> MediaRecoveryResult:
        """Executes receiver-side extraction, error correction, decryption, and recovery.

        Sequence:
            1. Decode PNG carrier into 8-bit RGB array.
            2. Compute DWT and extract 16B header from anchor positions.
            3. Reconstruct deterministic mask and extract protected payload bits.
            4. SecurityPipeline.unprotect: Reed-Solomon decode -> AES-GCM decrypt -> Deserialize.
            5. Verify exact byte equality against reference if provided.
        """
        if isinstance(stego_carrier, bytes):
            carrier_img = Image.open(io.BytesIO(stego_carrier)).convert("RGB")
        elif isinstance(stego_carrier, Image.Image):
            carrier_img = stego_carrier.convert("RGB")
        elif isinstance(stego_carrier, np.ndarray):
            carrier_img = Image.fromarray(stego_carrier, mode="RGB")
        else:
            raise TypeError(f"Unsupported stego carrier type: {type(stego_carrier).__name__}")

        # Step 1 & 2: Extract header and protected bytes
        header, extracted_protected = Extractor.extract(carrier_img)

        # Step 3: Unprotect payload (RS decode -> AES decrypt -> Deserialize)
        recovered_payload = self.security_pipeline.unprotect(
            extracted_protected, key, aad=aad
        )

        # Reconstruct PIL Image if payload is MediaType.IMAGE
        recovered_img: Optional[Image.Image] = None
        if recovered_payload.media_type == MediaType.IMAGE:
            try:
                recovered_img = Image.open(io.BytesIO(recovered_payload.data))
            except Exception:
                recovered_img = None

        # Step 4: Verify exact match
        exact_match = False
        if reference_payload is not None:
            exact_match = (
                recovered_payload.data == reference_payload.data
                and recovered_payload.media_type == reference_payload.media_type
                and recovered_payload.metadata == reference_payload.metadata
            )

        return MediaRecoveryResult(
            recovered_payload=recovered_payload,
            header=header,
            exact_match=exact_match,
            recovered_image=recovered_img,
        )

    def run_full_roundtrip(
        self,
        secret_payload: SecretPayload,
        cover_image: Union[np.ndarray, Image.Image],
        key: bytes,
        aad: Optional[bytes] = None,
        session_counter: int = 1,
    ) -> Tuple[MediaStegoResult, MediaRecoveryResult]:
        """Convenience method running the complete sender-to-receiver media pipeline."""
        stego_result = self.protect_and_embed(
            secret_payload=secret_payload,
            cover_image=cover_image,
            key=key,
            aad=aad,
            session_counter=session_counter,
        )

        recovery_result = self.extract_and_unprotect(
            stego_carrier=stego_result.stego_png_bytes,
            key=key,
            aad=aad,
            reference_payload=secret_payload,
        )

        return stego_result, recovery_result
