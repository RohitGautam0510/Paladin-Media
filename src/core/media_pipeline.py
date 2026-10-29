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
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.carrier.capacity import CapacityCalculator, CapacityReport, PayloadCapacityExceededError
from src.carrier.dwt import DWTSubbands, HaarDWT2D, subband_to_image, visualize_subbands
from src.carrier.embedding import BaselineEmbedder, EmbeddingResult
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator
from src.carrier.modulation import CarrierModulationEngine, ParityModulationEngine
from src.core.exceptions import UnsupportedCarrierFormatError
from src.core.pipeline import SecurityPipeline
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.crypto.provider import AESGCMProvider, CryptoProvider
from src.security.ecc.encoder import ECCEncoder, ReedSolomonEncoder

__all__ = [
    "MediaStegoPipeline",
    "MediaStegoResult",
    "MediaRecoveryResult",
    "UnsupportedCarrierFormatError",
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
    carrier_format: str = "PNG"
    output_format_policy: str = "Lossless 8-bit RGB PNG only. Do not transcode or compress to JPEG."


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
        modulation_engine: Optional[CarrierModulationEngine] = None,
        backend: Optional[str] = None,
        onnx_dir: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        spatial_bridge: Optional[Any] = None,
    ) -> None:
        self.security_pipeline = SecurityPipeline(
            crypto_provider=crypto_provider,
            ecc_encoder=ecc_encoder,
        )
        if modulation_engine is not None:
            self.modulation_engine = modulation_engine
        elif backend is not None:
            from src.carrier.modulation import NeuralModulationEngine

            self.modulation_engine = NeuralModulationEngine(
                spatial_bridge=spatial_bridge,
                backend=backend,
                onnx_dir=onnx_dir,
                checkpoint_path=checkpoint_path,
            )
        else:
            self.modulation_engine = ParityModulationEngine()

    @classmethod
    def create_qualified_deployment(
        cls,
        backend: str = "onnx",
        onnx_dir: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        spatial_bridge: Optional[Any] = None,
        crypto_provider: Optional[CryptoProvider] = None,
    ) -> MediaStegoPipeline:
        """Factory creating the primary qualified Phase 5B deployment pipeline.

        Explicitly configures the primary qualified Error Correction profile:
        Reed-Solomon RS(255, 207) with 2t=48 parity symbols, backed by native
        ONNX Runtime session pooling for spatial carrier modulation.

        Preserves:
            - Standard __init__ retains the legacy default (RS(255, 223), 2t=32).
            - Wire-level CarrierHeader.flags (0x01) encodes the 2t=48 profile.
            - Receiver instances dynamically adapt via header flags regardless of constructor defaults.
        """
        from src.security.ecc.encoder import ReedSolomonEncoder

        ecc = ReedSolomonEncoder(parity_symbols=48)
        return cls(
            crypto_provider=crypto_provider,
            ecc_encoder=ecc,
            backend=backend,
            onnx_dir=onnx_dir,
            checkpoint_path=checkpoint_path,
            spatial_bridge=spatial_bridge,
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
            3. Wire framing & deterministic mask generation.
            4. Embed 16B header + protected payload into HH channel 0 via selected engine.
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

        subband_dim = (cover_arr.shape[0] // 2, cover_arr.shape[1] // 2)
        ecc_parity = getattr(self.security_pipeline.ecc_encoder, "parity_symbols", 32)

        # Preflight usable capacity check
        meta_len = 0
        if secret_payload.metadata:
            try:
                import json
                meta_json = json.dumps(
                    secret_payload.metadata, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                meta_len = len(meta_json)
            except Exception:
                meta_len = 0

        usable_cap = CapacityCalculator.calculate_usable_secret_capacity(
            metadata_len=meta_len, subband_dim=subband_dim, rs_parity=ecc_parity
        )
        if len(secret_payload.data) > usable_cap:
            raise PayloadCapacityExceededError(
                f"PAYLOAD_CAPACITY_EXCEEDED: Secret payload data ({len(secret_payload.data)} B) "
                f"exceeds maximum usable capacity ({usable_cap} B with {meta_len}B metadata) "
                f"for ECC parity {ecc_parity}."
            )

        # Step 1: Cryptographic protection & ECC
        protected_payload = self.security_pipeline.protect(secret_payload, key, aad=aad)

        # Step 2: Capacity check (Hard gate)
        capacity_report = CapacityCalculator.check_capacity(
            len(protected_payload), subband_dim=subband_dim, rs_parity=ecc_parity
        )

        # Step 3: Wire framing & deterministic mask generation
        header = CarrierHeader.create(
            session_counter=session_counter,
            payload_length=len(protected_payload),
            ecc_parity=ecc_parity,
        )
        header_bytes = header.pack()
        mask = DeterministicMaskGenerator.generate_mask(
            session_counter=session_counter,
            payload_length_bytes=len(protected_payload),
            subband_dim=subband_dim,
        )

        header_bits = BaselineEmbedder.bytes_to_bits(header_bytes)
        payload_bits = BaselineEmbedder.bytes_to_bits(protected_payload)

        # Step 4: Modulate via selected engine
        stego_arr = self.modulation_engine.embed_bits(
            cover_array=cover_arr,
            header_bits=header_bits,
            payload_bits=payload_bits,
            mask=mask,
        )
        stego_pil = Image.fromarray(stego_arr, mode="RGB")

        # Step 5: Serialize to PNG bytes
        png_buffer = io.BytesIO()
        stego_pil.save(png_buffer, format="PNG")
        stego_png_bytes = png_buffer.getvalue()

        # Step 6: Generate visual artifacts and metrics
        original_subbands = HaarDWT2D.forward(cover_arr)
        modified_subbands = HaarDWT2D.forward(stego_arr)
        modified_hh_img = subband_to_image(modified_subbands.hh, is_ll=False)
        dwt_visuals = visualize_subbands(original_subbands)
        psnr_val = calculate_psnr(cover_arr, stego_arr)
        ssim_val = calculate_ssim(cover_arr, stego_arr)

        return MediaStegoResult(
            stego_image=stego_pil,
            stego_png_bytes=stego_png_bytes,
            capacity_report=capacity_report,
            dwt_visualizations=dwt_visuals,
            modified_hh_image=modified_hh_img,
            psnr_db=psnr_val,
            ssim=ssim_val,
            session_counter=session_counter,
            header=header,
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
            2. Prepare extraction via selected modulation engine.
            3. Extract 16B header from fixed anchor positions.
            4. Reconstruct deterministic mask and extract protected payload bits.
            5. SecurityPipeline.unprotect: Reed-Solomon decode -> AES-GCM decrypt -> Deserialize.
            6. Verify exact byte equality against reference if provided.
        """
        format_detected: Optional[str] = None
        if isinstance(stego_carrier, bytes):
            if stego_carrier.startswith(b"\xff\xd8\xff"):
                format_detected = "JPEG"
            else:
                try:
                    with Image.open(io.BytesIO(stego_carrier)) as probed:
                        format_detected = probed.format
                except Exception:
                    pass
            carrier_img = Image.open(io.BytesIO(stego_carrier)).convert("RGB")
        elif isinstance(stego_carrier, Image.Image):
            format_detected = getattr(stego_carrier, "format", None)
            carrier_img = stego_carrier.convert("RGB")
        elif isinstance(stego_carrier, (str, Path)):
            p = Path(stego_carrier)
            if p.suffix.lower() in (".jpg", ".jpeg"):
                format_detected = "JPEG"
            with open(p, "rb") as f:
                raw_bytes = f.read()
            return self.extract_and_unprotect(
                stego_carrier=raw_bytes,
                key=key,
                aad=aad,
                reference_payload=reference_payload,
            )
        elif isinstance(stego_carrier, np.ndarray):
            carrier_img = Image.fromarray(stego_carrier, mode="RGB")
        else:
            raise TypeError(f"Unsupported stego carrier type: {type(stego_carrier).__name__}")

        if format_detected and format_detected.upper() in ("JPEG", "JPG"):
            raise UnsupportedCarrierFormatError(
                "JPEG-transcoded carriers are not qualified for the current deployment. "
                "PaladinMedia requires the original lossless 8-bit RGB PNG carrier."
            )

        arr = np.array(carrier_img, dtype=np.uint8)
        subband_dim = (arr.shape[0] // 2, arr.shape[1] // 2)

        # Step 2: Initialize extraction via selected modulation engine
        bit_extractor = self.modulation_engine.begin_extraction(arr, subband_dim=subband_dim)

        # Step 3: Extract 128 header bits from fixed anchor positions
        header_positions = DeterministicMaskGenerator.generate_header_positions(subband_dim=subband_dim)
        extracted_hdr_bits = bit_extractor.extract_bits(header_positions)
        extracted_hdr_bytes = Extractor.bits_to_bytes(extracted_hdr_bits)
        header = CarrierHeader.unpack(extracted_hdr_bytes)

        # Step 4: Extract protected payload bits using regenerated deterministic mask
        if header.payload_length == 0:
            extracted_protected = b""
        else:
            payload_positions = DeterministicMaskGenerator.generate_payload_positions(
                session_counter=header.session_counter,
                payload_length_bytes=header.payload_length,
                subband_dim=subband_dim,
            )
            extracted_payload_bits = bit_extractor.extract_bits(payload_positions)
            extracted_protected = Extractor.bits_to_bytes(extracted_payload_bits)

        # Step 5: Unprotect payload (RS decode -> AES decrypt -> Deserialize)
        # Adapt decoder to matching ECC profile from carrier header flags
        header_parity = header.ecc_parity
        if (
            isinstance(self.security_pipeline.ecc_encoder, ReedSolomonEncoder)
            and self.security_pipeline.ecc_encoder.parity_symbols == header_parity
        ):
            decoder = self.security_pipeline.ecc_encoder
        else:
            decoder = ReedSolomonEncoder(parity_symbols=header_parity)

        recovered_payload = self.security_pipeline.unprotect(
            extracted_protected, key, aad=aad, ecc_encoder=decoder
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
            exact_match = (recovered_payload == reference_payload)

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

    def get_capacity_limits(self, subband_dim: Union[int, Tuple[int, int], Sequence[int]] = 256) -> Dict[str, Any]:
        """Returns non-sensitive operational capacity limits, ceilings, and format policies."""
        ecc_parity = getattr(self.security_pipeline.ecc_encoder, "parity_symbols", 32)
        raw_cap = CapacityCalculator.calculate_usable_secret_capacity(
            metadata_len=0, subband_dim=subband_dim, rs_parity=ecc_parity
        )
        ref_cap = CapacityCalculator.calculate_usable_secret_capacity(
            metadata_len=2, subband_dim=subband_dim, rs_parity=ecc_parity
        )
        sub_h, sub_w = (subband_dim[0], subband_dim[1]) if isinstance(subband_dim, (tuple, list)) else (subband_dim, subband_dim)
        max_prot = (sub_h * sub_w // 8) - CapacityCalculator.CARRIER_HEADER_BYTES
        engine_backend = getattr(self.modulation_engine, "backend", "")
        is_qualified = (ecc_parity == 48 and engine_backend == "onnx")
        return {
            "ecc_profile": f"rs_2t_{ecc_parity}",
            "ecc_parity_symbols": ecc_parity,
            "raw_secret_capacity_bytes": raw_cap,
            "reference_metadata_capacity_bytes": ref_cap,
            "max_protected_bytes": max_prot,
            "is_qualified_deployment": is_qualified,
            "carrier_format_policy": "Lossless 8-bit RGB PNG only. Lossy transcoded carriers (e.g. JPEG) are strictly unsupported.",
            "operational_qualification_note": (
                "Primary qualified deployment profile rs_2t_48 with native ONNX Runtime backend. "
                "Reliable 100% exact recovery across 100B-5000B lossless payloads and robust up to Gaussian noise sigma<=0.5."
                if is_qualified
                else f"Legacy/unqualified profile (rs_2t_{ecc_parity}). Use create_qualified_deployment() for production."
            ),
        }


def create_qualified_deployment_pipeline(
    backend: str = "onnx",
    onnx_dir: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    spatial_bridge: Optional[Any] = None,
    crypto_provider: Optional[CryptoProvider] = None,
) -> MediaStegoPipeline:
    """Convenience factory creating the Phase 5B qualified deployment pipeline (rs_2t_48 + ONNX)."""
    return MediaStegoPipeline.create_qualified_deployment(
        backend=backend,
        onnx_dir=onnx_dir,
        checkpoint_path=checkpoint_path,
        spatial_bridge=spatial_bridge,
        crypto_provider=crypto_provider,
    )
