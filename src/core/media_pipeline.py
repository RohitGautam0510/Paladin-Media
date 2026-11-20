"""Media-to-Media Steganography Pipeline for PaladinMedia MVP 0 / Phase 2.

Integrates:
1. Secret Media Serialization (SecretSerializer, CRC-32)
2. Authenticated Encryption (AES-256-GCM, fail-closed)
3. Error Correction Coding (Reed-Solomon GF(2^8))
4. Capacity Hard Gate (CapacityCalculator)
5. Carrier Steganographic Framing (CarrierHeader v2 / v3)
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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from src.carrier.capacity import CapacityCalculator, CapacityReport, PayloadCapacityExceededError
from src.carrier.dwt import DWTSubbands, HaarDWT2D, subband_to_image, visualize_subbands
from src.carrier.embedding import BaselineEmbedder, EmbeddingResult
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
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
    pqc_authenticated_sender: Optional[str] = None
    pqc_recipient_id: Optional[str] = None
    pqc_option: Optional[str] = None
    detached_signature: Optional[bytes] = None


@dataclass(frozen=True)
class MediaRecoveryResult:
    """Result emitted by the receiver extraction and recovery pipeline."""

    recovered_payload: SecretPayload
    header: CarrierHeader
    exact_match: bool
    recovered_image: Optional[Image.Image] = None
    authenticated_sender_id: Optional[str] = None


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
        enable_v3_mr: Optional[bool] = None,
        precision: str = "fp32",
    ) -> None:
        self.security_pipeline = SecurityPipeline(
            crypto_provider=crypto_provider,
            ecc_encoder=ecc_encoder,
        )
        if enable_v3_mr is None:
            enable_v3_mr = os.environ.get("PALADIN_ENABLE_V3_MR", "false").lower() in ("true", "1", "yes")
        self.enable_v3_mr: bool = enable_v3_mr
        self.precision = precision.lower()

        if modulation_engine is not None:
            self.modulation_engine = modulation_engine
        elif backend is not None:
            from src.carrier.modulation import create_modulation_engine

            self.modulation_engine = create_modulation_engine(
                engine_name=backend,
                spatial_bridge=spatial_bridge,
                onnx_dir=onnx_dir,
                checkpoint_path=checkpoint_path,
                precision=self.precision,
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
        enable_v3_mr: Optional[bool] = None,
        precision: str = "fp32",
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
            enable_v3_mr=enable_v3_mr,
            precision=precision,
        )

    @classmethod
    def create_v3_mr_deployment(
        cls,
        backend: str = "onnx",
        onnx_dir: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        spatial_bridge: Optional[Any] = None,
        crypto_provider: Optional[CryptoProvider] = None,
        precision: str = "fp32",
    ) -> MediaStegoPipeline:
        """Factory creating the qualified Wire Protocol v3-MR deployment pipeline.

        Configures the Maximum Resilience (MR) Error Correction profile:
        Reed-Solomon RS(255, 191) with 2t=64 parity symbols, backed by native
        ONNX Runtime session pooling for spatial carrier modulation.
        """
        from src.security.ecc.encoder import ReedSolomonEncoder

        ecc = ReedSolomonEncoder(parity_symbols=64)
        return cls(
            crypto_provider=crypto_provider,
            ecc_encoder=ecc,
            backend=backend,
            onnx_dir=onnx_dir,
            checkpoint_path=checkpoint_path,
            spatial_bridge=spatial_bridge,
            enable_v3_mr=True,
            precision=precision,
        )

    def protect_and_embed(
        self,
        secret_payload: SecretPayload,
        cover_image: Union[np.ndarray, Image.Image],
        key: bytes,
        aad: Optional[bytes] = None,
        session_counter: int = 1,
        version: int = CarrierHeader.VERSION_2,
    ) -> MediaStegoResult:
        """Executes sender-side protection, transform, and carrier embedding.

        Sequence:
            1. SecurityPipeline.protect: Serialize -> AES-256-GCM -> Reed-Solomon.
            2. Capacity hard-gate: Abort if protected payload exceeds usable capacity.
            3. Wire framing & deterministic mask generation (Wire Protocol v2 default or v3).
            4. Embed header + protected payload into HH channel 0.
            5. IDWT & serialize to lossless 8-bit RGB PNG.
            6. Compute fidelity metrics (PSNR, SSIM).
        """
        if version == CarrierHeader.VERSION_3 and not self.enable_v3_mr:
            raise CarrierHeaderError(
                "Wire Protocol v3-MR is disabled on this pipeline instance. "
                "Opt-in and server-side feature gate PALADIN_ENABLE_V3_MR are required."
            )

        # Ensure cover image is PIL Image or valid array
        if isinstance(cover_image, np.ndarray):
            cover_arr = cover_image
            cover_pil = Image.fromarray(cover_image, mode="RGB")
        else:
            cover_pil = cover_image.convert("RGB")
            cover_arr = np.array(cover_pil, dtype=np.uint8)

        subband_dim = (cover_arr.shape[0] // 2, cover_arr.shape[1] // 2)

        # Determine ECC profile and encoder based on version
        if version == CarrierHeader.VERSION_3:
            ecc_parity = getattr(self.security_pipeline.ecc_encoder, "parity_symbols", 64)
            if ecc_parity not in (32, 48, 64) or ecc_parity < 64:
                ecc_parity = 64
            encoder = ReedSolomonEncoder(parity_symbols=ecc_parity)
        else:
            ecc_parity = getattr(self.security_pipeline.ecc_encoder, "parity_symbols", 48 if version == 2 else 32)
            encoder = self.security_pipeline.ecc_encoder

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
            metadata_len=meta_len, subband_dim=subband_dim, rs_parity=ecc_parity, version=version
        )
        if len(secret_payload.data) > usable_cap:
            raise PayloadCapacityExceededError(
                f"PAYLOAD_CAPACITY_EXCEEDED: Secret payload data ({len(secret_payload.data)} B) "
                f"exceeds maximum usable capacity ({usable_cap} B with {meta_len}B metadata) "
                f"for ECC parity {ecc_parity} under Wire Protocol v{version}."
            )

        # Step 1: Cryptographic protection & ECC
        protected_payload = self.security_pipeline.protect(secret_payload, key, aad=aad, ecc_encoder=encoder)

        # Step 2: Capacity check (Hard gate)
        capacity_report = CapacityCalculator.check_capacity(
            len(protected_payload), subband_dim=subband_dim, rs_parity=ecc_parity, version=version
        )

        # Step 3: Optional Symbol Interleaving (for Wire Protocol v3)
        if version == CarrierHeader.VERSION_3:
            from src.carrier.interleaver import DeterministicSymbolInterleaver
            payload_to_embed = DeterministicSymbolInterleaver.interleave(
                protected_payload, scheme="coprime", session_counter=session_counter
            )
        else:
            payload_to_embed = protected_payload

        # Step 4: Wire framing & deterministic mask generation
        header = CarrierHeader.create(
            session_counter=session_counter,
            payload_length=len(protected_payload),
            ecc_parity=ecc_parity,
            version=version,
        )
        header_bytes = header.pack()
        mask = DeterministicMaskGenerator.generate_mask(
            session_counter=session_counter,
            payload_length_bytes=len(protected_payload),
            subband_dim=subband_dim,
            version=version,
        )

        header_bits = BaselineEmbedder.bytes_to_bits(header_bytes)
        payload_bits = BaselineEmbedder.bytes_to_bits(payload_to_embed)

        # Step 5: Modulate via selected engine
        stego_arr = self.modulation_engine.embed_bits(
            cover_array=cover_arr,
            header_bits=header_bits,
            payload_bits=payload_bits,
            mask=mask,
        )
        stego_pil = Image.fromarray(stego_arr, mode="RGB")

        # Step 6: Serialize to PNG bytes
        png_buffer = io.BytesIO()
        stego_pil.save(png_buffer, format="PNG")
        stego_png_bytes = png_buffer.getvalue()

        # Step 7: Generate visual artifacts and metrics
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
            3. Dual-anchor header probing (Row 32 v3 primary, Row 0 v2/v1 fallback).
            4. Reconstruct deterministic mask and extract protected payload bits.
            5. Deinterleave symbols if FLAG_SYMBOL_INTERLEAVED is set.
            6. SecurityPipeline.unprotect: Reed-Solomon decode -> AES-GCM decrypt -> Deserialize.
            7. Verify exact byte equality against reference if provided.
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

        # Step 3: Dual-Anchor Header Probing (Row 32 only if enable_v3_mr is True)
        header_v3: Optional[CarrierHeader] = None
        if self.enable_v3_mr and subband_dim[0] > CarrierHeader.HEADER_ROW_OFFSET_V3:
            pos_v3 = DeterministicMaskGenerator.generate_header_positions(
                subband_dim=subband_dim, version=CarrierHeader.VERSION_3
            )
            hdr_bits_v3 = bit_extractor.extract_bits(pos_v3)
            raw_v3 = Extractor.bits_to_bytes(hdr_bits_v3)
            try:
                cand_v3 = CarrierHeader.unpack(raw_v3, allow_v3=True)
                if cand_v3.version == CarrierHeader.VERSION_3:
                    header_v3 = cand_v3
            except CarrierHeaderError:
                pass

        header_v2: Optional[CarrierHeader] = None
        last_v2_err: Optional[Exception] = None
        pos_v2 = DeterministicMaskGenerator.generate_header_positions(
            subband_dim=subband_dim, version=CarrierHeader.VERSION_2
        )
        hdr_bits_v2 = bit_extractor.extract_bits(pos_v2)
        raw_v2 = Extractor.bits_to_bytes(hdr_bits_v2)
        try:
            cand_v2 = CarrierHeader.unpack(raw_v2, allow_v3=self.enable_v3_mr)
            if cand_v2.version in (CarrierHeader.VERSION_1, CarrierHeader.VERSION_2):
                header_v2 = cand_v2
        except CarrierHeaderError as ex:
            last_v2_err = ex

        # Ambiguity resolution & fail-closed arbitrament
        if header_v3 is not None and header_v2 is not None:
            if (
                header_v3.version == header_v2.version
                and header_v3.flags == header_v2.flags
                and header_v3.session_counter == header_v2.session_counter
                and header_v3.payload_length == header_v2.payload_length
            ):
                header = header_v3 if header_v3.version == CarrierHeader.VERSION_3 else header_v2
            else:
                raise CarrierHeaderError(
                    "Ambiguous carrier header detected: multiple valid headers with conflicting metadata "
                    "at distinct anchor positions (Row 32 and Row 0)."
                )
        elif header_v3 is not None:
            header = header_v3
        elif header_v2 is not None:
            header = header_v2
        else:
            if not self.enable_v3_mr and last_v2_err is not None:
                raise last_v2_err
            detail = f" Details: {last_v2_err}" if last_v2_err is not None else ""
            if self.enable_v3_mr:
                raise CarrierHeaderError(
                    f"Uncorrectable carrier header: both Row 32 (v3) and Row 0 (v2/v1) anchor probes failed.{detail}"
                )
            raise CarrierHeaderError(
                f"Uncorrectable carrier header at Row 0 anchor (Wire Protocol v2/v1).{detail}"
            )

        # Hard-gate capacity check before allocation or mask generation
        CapacityCalculator.check_capacity(
            protected_payload_bytes=header.payload_length,
            subband_dim=subband_dim,
            rs_parity=header.ecc_parity,
            version=header.version,
        )

        # Step 4: Extract protected payload bits using regenerated deterministic mask
        if header.payload_length == 0:
            extracted_protected = b""
        else:
            payload_positions = DeterministicMaskGenerator.generate_payload_positions(
                session_counter=header.session_counter,
                payload_length_bytes=header.payload_length,
                subband_dim=subband_dim,
                version=header.version,
            )
            extracted_payload_bits = bit_extractor.extract_bits(payload_positions)
            extracted_protected = Extractor.bits_to_bytes(extracted_payload_bits)

            # Conditional deinterleaving
            if header.symbol_interleaved:
                from src.carrier.interleaver import DeterministicSymbolInterleaver
                extracted_protected = DeterministicSymbolInterleaver.deinterleave(
                    extracted_protected, scheme="coprime", session_counter=header.session_counter
                )

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

        # Step 6: Verify exact match
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
        version: int = CarrierHeader.VERSION_2,
    ) -> Tuple[MediaStegoResult, MediaRecoveryResult]:
        """Convenience method running the complete sender-to-receiver media pipeline."""
        stego_result = self.protect_and_embed(
            secret_payload=secret_payload,
            cover_image=cover_image,
            key=key,
            aad=aad,
            session_counter=session_counter,
            version=version,
        )
        recovery_result = self.extract_and_unprotect(
            stego_carrier=stego_result.stego_png_bytes,
            key=key,
            aad=aad,
            reference_payload=secret_payload,
        )
        return stego_result, recovery_result

    def protect_and_embed_pqc(
        self,
        secret_payload: Union[SecretPayload, bytes],
        cover_image: Union[np.ndarray, Image.Image],
        sender_id: str,
        sender_dsa_private_key: bytes,
        recipient_id: str,
        option: str = "B",
        session_counter: int = 1,
        session_timestamp_ms: Optional[int] = None,
        version: int = CarrierHeader.VERSION_3,
        trust_store: Optional[Any] = None,
        pqc_provider: Optional[Any] = None,
    ) -> MediaStegoResult:
        """Executes sender-side Post-Quantum Authenticated protection and embedding.

        Pipeline:
            1. Validate PALADIN_ENABLE_PQC feature gate (fail-closed).
            2. Canonical serialization: SecretPayload -> bytes (CRC-32).
            3. AKE Protection: Ephemeral ML-KEM-768 + ML-DSA-65 + HKDF + AES-256-GCM -> PQCEnvelope.
            4. Reed-Solomon GF(2^8) ECC encoding over entire PQCEnvelope.
            5. Usable carrier capacity verification (hard gate).
            6. Wire Protocol v3-MR framing with Row 32 anchor & symbol interleaving.
            7. Transform-domain carrier modulation & IDWT PNG synthesis.
        """
        if not os.environ.get("PALADIN_ENABLE_PQC", "false").lower() in ("true", "1", "yes"):
            from src.security.pqc.exceptions import PQCDisabledError
            raise PQCDisabledError(
                "Post-Quantum Cryptography is disabled on this system. "
                "Set PALADIN_ENABLE_PQC=true to enable."
            )

        if version != CarrierHeader.VERSION_3:
            raise CarrierHeaderError(
                f"PQC steganography envelope is cryptographically bound to Wire Protocol v3 (v3-MR). "
                f"Carrier version {version} is incompatible with canonical transcript 'PaladinMedia-WireV3-PQC-v1'."
            )

        if not self.enable_v3_mr:
            raise CarrierHeaderError(
                "Wire Protocol v3-MR is disabled on this pipeline instance. "
                "Opt-in and server-side feature gate PALADIN_ENABLE_V3_MR are required."
            )

        from src.security.pqc.ake import AKEEngine
        from src.security.pqc.provider import default_pqc_provider
        from src.security.pqc.trust import default_trust_store

        engine = AKEEngine(provider=pqc_provider or default_pqc_provider, trust_store=trust_store or default_trust_store)

        # Step 1: Canonical payload serialization
        if isinstance(secret_payload, (bytes, bytearray)):
            serialized_payload = bytes(secret_payload)
        else:
            serialized_payload = SecretSerializer.serialize(secret_payload)

        # Step 2: AKE Envelope protection
        envelope = engine.protect_payload(
            sender_id=sender_id,
            sender_dsa_private_key=sender_dsa_private_key,
            recipient_id=recipient_id,
            payload=serialized_payload,
            option=option,
            session_timestamp_ms=session_timestamp_ms,
            custom_session_counter=session_counter,
        )
        packed_envelope = envelope.pack()

        # Step 3: Reed-Solomon ECC Encoding
        ecc_encoder = self.security_pipeline.ecc_encoder
        protected_payload = ecc_encoder.encode(packed_envelope)

        # Step 4: Carrier setup & capacity hard gate
        if isinstance(cover_image, np.ndarray):
            cover_arr = cover_image
            cover_pil = Image.fromarray(cover_image, mode="RGB")
        else:
            cover_pil = cover_image.convert("RGB")
            cover_arr = np.array(cover_pil, dtype=np.uint8)

        subband_dim = (cover_arr.shape[0] // 2, cover_arr.shape[1] // 2)
        parity = ecc_encoder.parity_symbols if isinstance(ecc_encoder, ReedSolomonEncoder) else 64

        cap_report = CapacityCalculator.check_capacity(
            protected_payload_bytes=len(protected_payload),
            subband_dim=subband_dim,
            rs_parity=parity,
            version=version,
        )

        # Step 5: Symbol interleaving if v3
        to_embed = protected_payload
        if version == CarrierHeader.VERSION_3:
            from src.carrier.interleaver import DeterministicSymbolInterleaver
            to_embed = DeterministicSymbolInterleaver.interleave(
                protected_payload, scheme="coprime", session_counter=session_counter
            )

        header = CarrierHeader.create(
            session_counter=session_counter,
            payload_length=len(protected_payload),
            ecc_parity=parity,
            version=version,
        )
        header_bytes = header.pack()

        mask = DeterministicMaskGenerator.generate_mask(
            session_counter=session_counter,
            payload_length_bytes=len(protected_payload),
            subband_dim=subband_dim,
            version=version,
        )

        header_bits = BaselineEmbedder.bytes_to_bits(header_bytes)
        payload_bits = BaselineEmbedder.bytes_to_bits(to_embed)

        # Step 6: Carrier modulation
        stego_arr = self.modulation_engine.embed_bits(
            cover_array=cover_arr,
            header_bits=header_bits,
            payload_bits=payload_bits,
            mask=mask,
        )
        stego_pil = Image.fromarray(stego_arr, mode="RGB")

        # Step 7: Serialize to PNG
        buf = io.BytesIO()
        stego_pil.save(buf, format="PNG")
        stego_png_bytes = buf.getvalue()

        # Step 8: Visualizations & metrics
        original_subbands = HaarDWT2D.forward(cover_arr)
        modified_subbands = HaarDWT2D.forward(stego_arr)
        mod_hh_img = subband_to_image(modified_subbands.hh, is_ll=False)
        vis = visualize_subbands(original_subbands)
        psnr_val = calculate_psnr(cover_arr, stego_arr)
        ssim_val = calculate_ssim(cover_arr, stego_arr)

        return MediaStegoResult(
            stego_image=stego_pil,
            stego_png_bytes=stego_png_bytes,
            capacity_report=cap_report,
            dwt_visualizations=vis,
            modified_hh_image=mod_hh_img,
            psnr_db=psnr_val,
            ssim=ssim_val,
            session_counter=session_counter,
            header=header,
            pqc_authenticated_sender=sender_id,
            pqc_recipient_id=recipient_id,
            pqc_option=option,
            detached_signature=envelope.signature if option.upper() == "A" else None,
        )

    def extract_and_unprotect_pqc(
        self,
        stego_carrier: Union[np.ndarray, Image.Image, bytes],
        recipient_kem_private_key: bytes,
        detached_signature: Optional[bytes] = None,
        sender_id_override: Optional[str] = None,
        reference_payload: Optional[SecretPayload] = None,
        trust_store: Optional[Any] = None,
        pqc_provider: Optional[Any] = None,
        now_ms: Optional[int] = None,
        sidecar: Optional[Any] = None,
    ) -> MediaRecoveryResult:
        """Executes receiver-side extraction, Reed-Solomon error correction, and PQC-AKE unprotection.

        Sequence:
            1. Validate PALADIN_ENABLE_PQC feature gate.
            2. Dual-anchor header probing and carrier bit extraction.
            3. Deterministic deinterleaving.
            4. Reed-Solomon decoding -> PQCEnvelope bytes.
            5. PQC-AKE decapsulation, transcript reconstruction, key confirmation, and ML-DSA-65 verification.
            6. AES-256-GCM payload decryption -> serialized bytes.
            7. SecretSerializer deserialization -> SecretPayload.
            8. Exact match verification.
        """
        if not os.environ.get("PALADIN_ENABLE_PQC", "false").lower() in ("true", "1", "yes"):
            from src.security.pqc.exceptions import PQCDisabledError
            raise PQCDisabledError(
                "Post-Quantum Cryptography is disabled on this system. "
                "Set PALADIN_ENABLE_PQC=true to enable."
            )

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
            return self.extract_and_unprotect_pqc(
                stego_carrier=raw_bytes,
                recipient_kem_private_key=recipient_kem_private_key,
                detached_signature=detached_signature,
                sender_id_override=sender_id_override,
                reference_payload=reference_payload,
                trust_store=trust_store,
                pqc_provider=pqc_provider,
                now_ms=now_ms,
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

        bit_extractor = self.modulation_engine.begin_extraction(arr, subband_dim=subband_dim)

        # Dual-Anchor Header Probing
        header_v3: Optional[CarrierHeader] = None
        if self.enable_v3_mr and subband_dim[0] > CarrierHeader.HEADER_ROW_OFFSET_V3:
            pos_v3 = DeterministicMaskGenerator.generate_header_positions(
                subband_dim=subband_dim, version=CarrierHeader.VERSION_3
            )
            hdr_bits_v3 = bit_extractor.extract_bits(pos_v3)
            raw_v3 = Extractor.bits_to_bytes(hdr_bits_v3)
            try:
                cand_v3 = CarrierHeader.unpack(raw_v3, allow_v3=True)
                if cand_v3.version == CarrierHeader.VERSION_3:
                    header_v3 = cand_v3
            except CarrierHeaderError:
                pass

        header_v2: Optional[CarrierHeader] = None
        pos_v2 = DeterministicMaskGenerator.generate_header_positions(
            subband_dim=subband_dim, version=CarrierHeader.VERSION_2
        )
        hdr_bits_v2 = bit_extractor.extract_bits(pos_v2)
        raw_v2 = Extractor.bits_to_bytes(hdr_bits_v2)
        try:
            cand_v2 = CarrierHeader.unpack(raw_v2, allow_v3=self.enable_v3_mr)
            if cand_v2.version in (CarrierHeader.VERSION_1, CarrierHeader.VERSION_2):
                header_v2 = cand_v2
        except CarrierHeaderError:
            pass

        if header_v3 is not None and header_v2 is not None:
            if (
                header_v3.version == header_v2.version
                and header_v3.flags == header_v2.flags
                and header_v3.session_counter == header_v2.session_counter
                and header_v3.payload_length == header_v2.payload_length
            ):
                header = header_v3 if header_v3.version == CarrierHeader.VERSION_3 else header_v2
            else:
                raise CarrierHeaderError(
                    "Ambiguous carrier header detected: multiple valid headers with conflicting metadata "
                    "at distinct anchor positions (Row 32 and Row 0)."
                )
        elif header_v3 is not None:
            header = header_v3
        elif header_v2 is not None:
            header = header_v2
        else:
            raise CarrierHeaderError("Uncorrectable carrier header during PQC extraction.")

        if header.version != CarrierHeader.VERSION_3:
            raise CarrierHeaderError(
                f"PQC steganography envelope requires Wire Protocol v3 carrier header. "
                f"Carrier version {header.version} is incompatible with canonical transcript 'PaladinMedia-WireV3-PQC-v1'."
            )

        # Capacity check
        CapacityCalculator.check_capacity(
            protected_payload_bytes=header.payload_length,
            subband_dim=subband_dim,
            rs_parity=header.ecc_parity,
            version=header.version,
        )

        # Extract bits
        payload_positions = DeterministicMaskGenerator.generate_payload_positions(
            session_counter=header.session_counter,
            payload_length_bytes=header.payload_length,
            subband_dim=subband_dim,
            version=header.version,
        )
        extracted_payload_bits = bit_extractor.extract_bits(payload_positions)
        extracted_protected = Extractor.bits_to_bytes(extracted_payload_bits)

        if header.symbol_interleaved:
            from src.carrier.interleaver import DeterministicSymbolInterleaver
            extracted_protected = DeterministicSymbolInterleaver.deinterleave(
                extracted_protected, scheme="coprime", session_counter=header.session_counter
            )

        # Reed-Solomon decode
        decoder = ReedSolomonEncoder(parity_symbols=header.ecc_parity)
        envelope_bytes = decoder.decode(extracted_protected)

        # PQC Unprotect
        from src.security.pqc.models import PQCEnvelope
        from src.security.pqc.ake import AKEEngine
        from src.security.pqc.provider import default_pqc_provider
        from src.security.pqc.trust import default_trust_store

        envelope = PQCEnvelope.unpack(envelope_bytes)
        engine = AKEEngine(provider=pqc_provider or default_pqc_provider, trust_store=trust_store or default_trust_store)
        sender_id, serialized_plaintext = engine.unprotect_payload(
            recipient_kem_private_key=recipient_kem_private_key,
            envelope=envelope,
            detached_signature=detached_signature,
            sender_id_override=sender_id_override,
            now_ms=now_ms,
            sidecar=sidecar,
        )

        try:
            recovered_payload = SecretSerializer.deserialize(serialized_plaintext)
        except Exception:
            recovered_payload = SecretPayload(data=serialized_plaintext, media_type=MediaType.RAW_BYTES)

        recovered_img: Optional[Image.Image] = None
        if recovered_payload.media_type == MediaType.IMAGE:
            try:
                recovered_img = Image.open(io.BytesIO(recovered_payload.data))
            except Exception:
                recovered_img = None

        exact_match = False
        if reference_payload is not None:
            if isinstance(reference_payload, (bytes, bytearray)):
                exact_match = (recovered_payload.data == bytes(reference_payload))
            else:
                exact_match = (recovered_payload == reference_payload)

        return MediaRecoveryResult(
            recovered_payload=recovered_payload,
            header=header,
            exact_match=exact_match,
            recovered_image=recovered_img,
            authenticated_sender_id=sender_id,
        )

    def run_full_roundtrip_pqc(
        self,
        secret_payload: Union[SecretPayload, bytes],
        cover_image: Union[np.ndarray, Image.Image],
        sender_id: str,
        sender_dsa_private_key: bytes,
        recipient_id: str,
        recipient_kem_private_key: bytes,
        option: str = "B",
        session_counter: int = 1,
        session_timestamp_ms: Optional[int] = None,
        version: int = CarrierHeader.VERSION_3,
        trust_store: Optional[Any] = None,
        pqc_provider: Optional[Any] = None,
    ) -> Tuple[MediaStegoResult, MediaRecoveryResult]:
        """Convenience method running the complete PQC sender-to-receiver media pipeline."""
        stego_result = self.protect_and_embed_pqc(
            secret_payload=secret_payload,
            cover_image=cover_image,
            sender_id=sender_id,
            sender_dsa_private_key=sender_dsa_private_key,
            recipient_id=recipient_id,
            option=option,
            session_counter=session_counter,
            session_timestamp_ms=session_timestamp_ms,
            version=version,
            trust_store=trust_store,
            pqc_provider=pqc_provider,
        )
        recovery_result = self.extract_and_unprotect_pqc(
            stego_carrier=stego_result.stego_png_bytes,
            recipient_kem_private_key=recipient_kem_private_key,
            detached_signature=stego_result.detached_signature,
            sender_id_override=sender_id if option.upper() == "A" else None,
            reference_payload=secret_payload,
            trust_store=trust_store,
            pqc_provider=pqc_provider,
        )
        return stego_result, recovery_result

    def get_capacity_limits(
        self,
        subband_dim: Union[int, Tuple[int, int], Sequence[int]] = 256,
        version: int = CarrierHeader.VERSION_2,
    ) -> Dict[str, Any]:
        """Returns non-sensitive operational capacity limits, ceilings, and format policies."""
        if version == CarrierHeader.VERSION_3:
            ecc_parity = getattr(self.security_pipeline.ecc_encoder, "parity_symbols", 64)
            if ecc_parity not in (32, 48, 64) or ecc_parity < 64:
                ecc_parity = 64
        else:
            ecc_parity = getattr(self.security_pipeline.ecc_encoder, "parity_symbols", 48 if version == 2 else 32)

        raw_cap = CapacityCalculator.calculate_usable_secret_capacity(
            metadata_len=0, subband_dim=subband_dim, rs_parity=ecc_parity, version=version
        )
        ref_cap = CapacityCalculator.calculate_usable_secret_capacity(
            metadata_len=2, subband_dim=subband_dim, rs_parity=ecc_parity, version=version
        )
        sub_h, sub_w = (subband_dim[0], subband_dim[1]) if isinstance(subband_dim, (tuple, list)) else (subband_dim, subband_dim)
        hdr_bytes = CapacityCalculator.get_header_bytes(version=version)
        max_prot = (sub_h * sub_w // 8) - hdr_bytes
        engine_backend = getattr(self.modulation_engine, "backend", "")
        is_qualified = (ecc_parity in (48, 64) and engine_backend in ("onnx", "neural_spatial_onnx"))
        return {
            "wire_protocol_version": version,
            "header_wire_bytes": hdr_bytes,
            "ecc_profile": f"rs_2t_{ecc_parity}",
            "ecc_parity_symbols": ecc_parity,
            "raw_secret_capacity_bytes": raw_cap,
            "reference_metadata_capacity_bytes": ref_cap,
            "max_protected_bytes": max_prot,
            "is_qualified_deployment": is_qualified,
            "carrier_format_policy": "Lossless 8-bit RGB PNG only. Lossy transcoded carriers (e.g. JPEG) are strictly unsupported.",
            "operational_qualification_note": (
                f"Qualified deployment profile rs_2t_{ecc_parity} with native ONNX Runtime backend and Wire Protocol v{version} Header ECC. "
                "Reliable exact recovery across lossless payloads and robust against transform-domain quantization."
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
    enable_v3_mr: Optional[bool] = None,
) -> MediaStegoPipeline:
    """Convenience factory creating the Phase 5B qualified deployment pipeline (rs_2t_48 + ONNX)."""
    return MediaStegoPipeline.create_qualified_deployment(
        backend=backend,
        onnx_dir=onnx_dir,
        checkpoint_path=checkpoint_path,
        spatial_bridge=spatial_bridge,
        crypto_provider=crypto_provider,
        enable_v3_mr=enable_v3_mr,
    )
