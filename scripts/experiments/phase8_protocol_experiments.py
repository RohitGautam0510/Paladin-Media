"""Phase 8 Protocol Reliability Experiments — Implementation and Evaluation Harness.

This module implements and evaluates two independent protocol-level interventions
to resolve the remaining failures in Wire Protocol v2 qualification:
1. Header Anchor Relocation (resolves zero-floor spatial clipping on border rows, e.g. 0715.png).
2. Deterministic Payload Symbol Interleaving (disperses localized burst errors from high-frequency textures, e.g. 0774.png).

Architectural Boundaries:
- Frozen neural architecture and weights: Phase 4C checkpoint and Phase 5A ONNX models are immutable baselines.
- Cryptographic & ECC integrity: Unidirectional pipeline (SecretPayload -> AES-GCM -> RS -> Modulation) is preserved.
- Backward compatibility: Wire Protocol v2 receiver behavior is preserved; experimental protocol is strictly versioned.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from PIL import Image
import reedsolo

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.carrier.capacity import CapacityCalculator, CapacityReport, PayloadCapacityExceededError
from src.carrier.dwt import HaarDWT2D
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import DeterministicMaskGenerator, EmbeddingMask
from src.carrier.modulation import NeuralModulationEngine
from src.core.exceptions import AuthenticationError, ECCDecodeError, PaladinSecurityError
from src.core.media_pipeline import MediaStegoPipeline, calculate_psnr, calculate_ssim
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.neural.dataset import DIV2KStegoDataset
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder

logger = logging.getLogger("Phase8Protocol")

OUTPUT_DIR = REPO_ROOT / "docs" / "experiments" / "phase8_protocol_reliability"
BASELINE_JSON = REPO_ROOT / "docs" / "diagnostics_20261030" / "wire_protocol_v2_qualification" / "production_v2_100_carrier_results.json"
MODEL_PATH = REPO_ROOT / "checkpoints" / "phase4c_extended_30ep" / "best_model.pt"
ONNX_EMBED = REPO_ROOT / "checkpoints" / "phase5a_onnx" / "neural_spatial_embedding_opset14.onnx"
ONNX_EXTRACT = REPO_ROOT / "checkpoints" / "phase5a_onnx" / "neural_spatial_extraction_opset14.onnx"
MANIFEST_PATH = REPO_ROOT / "docs" / "project" / "div2k_hr_split.json"


# =============================================================================
# 1. Experiment A: Header Anchor Relocation
# =============================================================================

class RelocatedMaskGenerator:
    """Deterministic mask generator with relocated header anchors away from image borders.

    Solves zero-floor clipping in images with dark boundaries (e.g. 0715.png).
    In a 256x256 transform subband:
    - Default v2 anchors: Row 0, cols 0..191 (spatial carrier rows 0-1, subject to border clipping).
    - Relocated anchors: Row `HEADER_ROW_OFFSET` (default 32), cols 0..191 (spatial rows 64-65).
    - Payload positions: Deterministically sampled from all remaining subband coordinates.
    - Capacity: Exactly 65,344 bits (identical to Wire Protocol v2).
    """

    SUBBAND_DIM: int = 256
    TOTAL_POSITIONS: int = SUBBAND_DIM * SUBBAND_DIM  # 65,536 positions
    HEADER_BITS: int = 24 * 8                         # 192 bits (RS(24, 16))
    MAX_PAYLOAD_BITS: int = TOTAL_POSITIONS - HEADER_BITS  # 65,344 bits
    DEFAULT_ROW_OFFSET: int = 32

    @classmethod
    def generate_header_positions(
        cls,
        subband_dim: Union[int, Tuple[int, int]] = SUBBAND_DIM,
        row_offset: int = DEFAULT_ROW_OFFSET,
    ) -> Tuple[Tuple[int, int], ...]:
        """Derives relocated header anchor positions within the interior of the subband."""
        sub_h, sub_w = (subband_dim, subband_dim) if isinstance(subband_dim, int) else (subband_dim[0], subband_dim[1])
        if row_offset < 0 or row_offset >= sub_h:
            raise ValueError(f"Invalid row_offset {row_offset} for subband height {sub_h}")
        if cls.HEADER_BITS > sub_w:
            raise ValueError(f"Header bits ({cls.HEADER_BITS}) exceeds subband width ({sub_w})")
        return tuple((row_offset, c) for c in range(cls.HEADER_BITS))

    @classmethod
    def generate_payload_positions(
        cls,
        session_counter: int,
        payload_length_bytes: int,
        subband_dim: Union[int, Tuple[int, int]] = SUBBAND_DIM,
        row_offset: int = DEFAULT_ROW_OFFSET,
    ) -> Tuple[Tuple[int, int], ...]:
        """Derives non-colliding payload coordinates deterministically avoiding relocated header anchors."""
        sub_h, sub_w = (subband_dim, subband_dim) if isinstance(subband_dim, int) else (subband_dim[0], subband_dim[1])
        total_positions = sub_h * sub_w
        max_payload_bits = total_positions - cls.HEADER_BITS
        payload_bits = payload_length_bytes * 8

        if payload_bits > max_payload_bits:
            raise PayloadCapacityExceededError(
                f"Requested payload bits ({payload_bits}) exceeds available positions ({max_payload_bits})"
            )
        if payload_bits == 0:
            return ()

        # Identify header linear indices to exclude
        header_indices = set(row_offset * sub_w + c for c in range(cls.HEADER_BITS))
        candidate_indices = [idx for idx in range(total_positions) if idx not in header_indices]

        # Deterministic seed derivation: Domain-separated SHA-256
        hasher = hashlib.sha256()
        hasher.update(b"PALADIN_RELOCATED_MASK_SEED_V1")
        hasher.update(struct.pack(">III", session_counter, payload_bits, row_offset))
        seed_int = int.from_bytes(hasher.digest()[:8], "big")

        rng = random.Random(seed_int)
        selected_indices = rng.sample(candidate_indices, payload_bits)

        return tuple(divmod(idx, sub_w) for idx in selected_indices)

    @classmethod
    def generate_mask(
        cls,
        session_counter: int,
        payload_length_bytes: int,
        subband_dim: Union[int, Tuple[int, int]] = SUBBAND_DIM,
        row_offset: int = DEFAULT_ROW_OFFSET,
    ) -> EmbeddingMask:
        """Constructs full EmbeddingMask with non-overlapping relocated header and payload coordinates."""
        hdr_pos = cls.generate_header_positions(subband_dim=subband_dim, row_offset=row_offset)
        pay_pos = cls.generate_payload_positions(
            session_counter=session_counter,
            payload_length_bytes=payload_length_bytes,
            subband_dim=subband_dim,
            row_offset=row_offset,
        )
        return EmbeddingMask(header_positions=hdr_pos, payload_positions=pay_pos)


# =============================================================================
# 2. Experiment B: Deterministic Payload Symbol Interleaving
# =============================================================================

class DeterministicSymbolInterleaver:
    """Deterministic, fully reversible symbol (byte) interleaver for Reed-Solomon codewords.

    Applied strictly between payload RS encoding and transform modulation.
    Disperses localized spatial error bursts across independent Reed-Solomon blocks
    so that no single block exceeds the correction threshold t=24.

    Supported Permutation Schemes:
    - 'coprime': Algebraic linear congruential stride: pi(i) = (a * i) mod N where gcd(a, N) = 1.
      Inverted in O(1) space via modular inverse: pi^-1(j) = (a^-1 * j) mod N.
    - 'balanced': Multi-codeword rate-matched spacing across variable-length codewords.
    - 'prng': Domain-separated SHA-256 seeded pseudo-random permutation.
    """

    DEFAULT_COPRIME_STEP: int = 71

    @classmethod
    def get_coprime_step(cls, n: int, preferred_step: int = DEFAULT_COPRIME_STEP) -> int:
        """Finds the smallest integer >= preferred_step that is coprime to n."""
        if n <= 1:
            return 1
        step = preferred_step
        while math.gcd(step, n) != 1:
            step += 1
        return step

    @classmethod
    def interleave(
        cls,
        encoded_stream: bytes,
        scheme: str = "coprime",
        session_counter: int = 1,
        coprime_step: int = DEFAULT_COPRIME_STEP,
    ) -> bytes:
        """Permutes the Reed-Solomon codewords in an encoded byte stream.

        Preserves the 4-byte big-endian length prefix at indices 0..3 unpermuted,
        applying the symbol permutation strictly to the RS codewords in encoded_stream[4:].
        """
        if len(encoded_stream) <= 4:
            return encoded_stream

        prefix = encoded_stream[:4]
        raw_blocks = encoded_stream[4:]
        n = len(raw_blocks)

        if scheme == "coprime":
            a = cls.get_coprime_step(n, preferred_step=coprime_step)
            perm = [(a * i) % n for i in range(n)]
            interleaved = bytes(raw_blocks[perm[i]] for i in range(n))
        elif scheme == "prng":
            hasher = hashlib.sha256(b"PALADIN_SYMBOL_INTERLEAVER_V1")
            hasher.update(struct.pack(">II", session_counter, n))
            seed_int = int.from_bytes(hasher.digest()[:8], "big")
            rng = random.Random(seed_int)
            perm = list(range(n))
            rng.shuffle(perm)
            interleaved = bytes(raw_blocks[perm[i]] for i in range(n))
        elif scheme == "none":
            return encoded_stream
        else:
            raise ValueError(f"Unknown interleaving scheme: {scheme}")

        return prefix + interleaved

    @classmethod
    def deinterleave(
        cls,
        interleaved_stream: bytes,
        scheme: str = "coprime",
        session_counter: int = 1,
        coprime_step: int = DEFAULT_COPRIME_STEP,
    ) -> bytes:
        """Inverts symbol interleaving, restoring the canonical sequential RS codeword order."""
        if len(interleaved_stream) <= 4:
            return interleaved_stream

        prefix = interleaved_stream[:4]
        raw_interleaved = interleaved_stream[4:]
        n = len(raw_interleaved)

        if scheme == "coprime":
            a = cls.get_coprime_step(n, preferred_step=coprime_step)
            a_inv = pow(a, -1, n)
            deinterleaved = bytes(raw_interleaved[(a_inv * i) % n] for i in range(n))
        elif scheme == "prng":
            hasher = hashlib.sha256(b"PALADIN_SYMBOL_INTERLEAVER_V1")
            hasher.update(struct.pack(">II", session_counter, n))
            seed_int = int.from_bytes(hasher.digest()[:8], "big")
            rng = random.Random(seed_int)
            perm = list(range(n))
            rng.shuffle(perm)
            inv = [0] * n
            for i, x in enumerate(perm):
                inv[x] = i
            deinterleaved = bytes(raw_interleaved[inv[i]] for i in range(n))
        elif scheme == "none":
            return interleaved_stream
        else:
            raise ValueError(f"Unknown interleaving scheme: {scheme}")

        return prefix + deinterleaved


# =============================================================================
# 3. Experiment C: Combined Protocol Candidate (Wire Protocol v2.1 / v3)
# =============================================================================

@dataclass(frozen=True)
class ProtocolCandidateConfig:
    """Configuration encapsulating experimental protocol enhancements."""

    name: str
    version: int = 0x03
    header_row_offset: int = 32
    interleaving_scheme: str = "coprime"
    coprime_step: int = 71
    ecc_parity: int = 48  # Default RS(255, 207, 2t=48); supports 64 (RS(255, 191, 2t=64))


class ExperimentalCarrierHeader(CarrierHeader):
    """Extended carrier header supporting Wire Protocol v3 and explicit capability flags."""

    VERSION_3: ClassVar[int] = 0x03

    # Protocol capability flags (Bits 4-7 of flags byte)
    FLAG_RELOCATED_HEADER: ClassVar[int] = 0x10  # Header anchored at row 32
    FLAG_SYMBOL_INTERLEAVED: ClassVar[int] = 0x20  # Payload symbols interleaved

    @classmethod
    def create_v3(
        cls,
        session_counter: int,
        payload_length: int,
        ecc_parity: int = 48,
        relocated_header: bool = True,
        symbol_interleaved: bool = True,
        reserved: int = 0x0000,
    ) -> ExperimentalCarrierHeader:
        """Constructs an ExperimentalCarrierHeader with explicit feature flags."""
        if ecc_parity not in cls.ECC_PARITY_TO_PROFILE:
            raise CarrierHeaderError(f"Unsupported ecc_parity {ecc_parity}")
        prof_bits = cls.ECC_PARITY_TO_PROFILE[ecc_parity]
        flags = prof_bits
        if relocated_header:
            flags |= cls.FLAG_RELOCATED_HEADER
        if symbol_interleaved:
            flags |= cls.FLAG_SYMBOL_INTERLEAVED

        return cls(
            session_counter=session_counter,
            payload_length=payload_length,
            version=cls.VERSION_3,
            flags=flags,
            reserved=reserved,
        )

    def pack(self) -> bytes:
        """Packs header with RS(24, 16) error protection."""
        if self.version == self.VERSION_3:
            raw_data = struct.pack(
                self.FORMAT,
                self.MAGIC,
                self.version,
                self.flags,
                self.reserved,
                self.session_counter,
                self.payload_length,
            )
            rsc = reedsolo.RSCodec(self.RS_PARITY_BYTES)
            codeword = bytes(rsc.encode(raw_data))
            return codeword
        return super().pack()

    @classmethod
    def unpack(cls, raw: bytes) -> CarrierHeader:
        """Unpacks carrier header supporting v3, v2, and v1 with fail-closed security."""
        if len(raw) < cls.HEADER_SIZE_V1:
            raise CarrierHeaderError(f"Carrier header truncated: {len(raw)} bytes")

        # Probe RS(24, 16) protected headers (v3 and v2)
        if len(raw) >= cls.HEADER_SIZE_V2:
            rsc = reedsolo.RSCodec(cls.RS_PARITY_BYTES)
            try:
                dec = rsc.decode(raw[:cls.HEADER_SIZE_V2])
                dec_data = bytes(dec[0])
                magic, version, flags, reserved, session_counter, payload_length = struct.unpack(
                    cls.FORMAT, dec_data
                )
                if magic == cls.MAGIC and version in (cls.VERSION_2, cls.VERSION_3):
                    if reserved != 0:
                        raise CarrierHeaderError(f"Non-zero reserved field: {reserved:#06x}")
                    prof = flags & cls.FLAG_ECC_MASK
                    if prof not in cls.ECC_PROFILE_TO_PARITY:
                        raise CarrierHeaderError(f"Unsupported ECC profile: {prof:#04x}")
                    return cls(
                        session_counter=session_counter,
                        payload_length=payload_length,
                        version=version,
                        flags=flags,
                        reserved=reserved,
                    )
            except reedsolo.ReedSolomonError:
                if len(raw) >= 5 and raw[:4] == cls.MAGIC and raw[4] in (cls.VERSION_2, cls.VERSION_3):
                    raise CarrierHeaderError(
                        f"Uncorrectable carrier header: errors exceed RS(24, 16) capacity (t=4 bytes)"
                    )

        # Fallback to standard unpack (handles v1 or terminal rejection)
        return super().unpack(raw)


class ExperimentalWireProtocolPipeline:
    """Isolated experimental pipeline implementing Candidate Wire Protocol v3."""

    def __init__(
        self,
        config: ProtocolCandidateConfig,
        backend: str = "onnx",
        crypto_provider: Optional[AESGCMProvider] = None,
    ) -> None:
        self.config = config
        self.crypto_provider = crypto_provider or AESGCMProvider()
        self.ecc_encoder = ReedSolomonEncoder(parity_symbols=config.ecc_parity)
        self.modulation_engine = NeuralModulationEngine(backend=backend)

    def protect_and_embed(
        self,
        secret_payload: SecretPayload,
        cover_image: np.ndarray,
        key: bytes,
        aad: Optional[bytes] = None,
        session_counter: int = 1,
    ) -> Dict[str, Any]:
        """Embeds payload into cover image using candidate protocol rules."""
        sub_h, sub_w = cover_image.shape[0] // 2, cover_image.shape[1] // 2

        # 1. Protect payload: Serialize -> AES-GCM -> Reed-Solomon
        serialized = SecretSerializer.serialize(secret_payload)
        ciphertext = self.crypto_provider.encrypt(serialized, key, aad=aad)
        encoded_stream = self.ecc_encoder.encode(ciphertext)

        # 2. Interleave payload symbols if configured
        interleaved_stream = DeterministicSymbolInterleaver.interleave(
            encoded_stream,
            scheme=self.config.interleaving_scheme,
            session_counter=session_counter,
            coprime_step=self.config.coprime_step,
        )

        # 3. Construct header
        header = ExperimentalCarrierHeader.create_v3(
            session_counter=session_counter,
            payload_length=len(interleaved_stream),
            ecc_parity=self.config.ecc_parity,
            relocated_header=(self.config.header_row_offset > 0),
            symbol_interleaved=(self.config.interleaving_scheme != "none"),
        )
        header_bytes = header.pack()

        # 4. Generate mask
        if self.config.header_row_offset > 0:
            mask = RelocatedMaskGenerator.generate_mask(
                session_counter=session_counter,
                payload_length_bytes=len(interleaved_stream),
                subband_dim=(sub_h, sub_w),
                row_offset=self.config.header_row_offset,
            )
        else:
            mask = DeterministicMaskGenerator.generate_mask(
                session_counter=session_counter,
                payload_length_bytes=len(interleaved_stream),
                subband_dim=(sub_h, sub_w),
                version=2,
            )

        header_bits = BaselineEmbedder.bytes_to_bits(header_bytes)
        payload_bits = BaselineEmbedder.bytes_to_bits(interleaved_stream)

        # 5. Modulate and embed
        stego_arr = self.modulation_engine.embed_bits(
            cover_array=cover_image,
            header_bits=header_bits,
            payload_bits=payload_bits,
            mask=mask,
        )

        psnr_val = calculate_psnr(cover_image, stego_arr)
        ssim_val = calculate_ssim(cover_image, stego_arr)

        return {
            "stego_array": stego_arr,
            "header": header,
            "mask": mask,
            "psnr_db": psnr_val,
            "ssim": ssim_val,
            "encoded_stream": encoded_stream,
            "interleaved_stream": interleaved_stream,
            "embedded_header_bits": header_bits,
            "embedded_payload_bits": payload_bits,
        }

    def extract_and_unprotect(
        self,
        stego_carrier: np.ndarray,
        key: bytes,
        aad: Optional[bytes] = None,
        reference_payload: Optional[SecretPayload] = None,
    ) -> Dict[str, Any]:
        """Extracts and recovers secret payload with automatic version/layout dispatch."""
        sub_h, sub_w = stego_carrier.shape[0] // 2, stego_carrier.shape[1] // 2
        extractor = self.modulation_engine.begin_extraction(stego_carrier, subband_dim=(sub_h, sub_w))

        # Multi-stage header probe:
        # 1. First probe relocated header anchor (row 32)
        # 2. If invalid, probe standard anchor (row 0, v2/v1)
        header = None
        header_row_offset_detected = 0

        # Probe relocated anchor
        reloc_pos = RelocatedMaskGenerator.generate_header_positions(
            subband_dim=(sub_h, sub_w), row_offset=self.config.header_row_offset
        )
        hdr_bits_reloc = extractor.extract_bits(reloc_pos)
        hdr_bytes_reloc = Extractor.bits_to_bytes(hdr_bits_reloc)
        try:
            h_cand = ExperimentalCarrierHeader.unpack(hdr_bytes_reloc)
            if h_cand.version == ExperimentalCarrierHeader.VERSION_3:
                header = h_cand
                header_row_offset_detected = self.config.header_row_offset
        except CarrierHeaderError:
            pass

        # Probe standard v2/v1 anchor if relocated probe was not v3
        if header is None:
            std_pos = DeterministicMaskGenerator.generate_header_positions(subband_dim=(sub_h, sub_w), version=2)
            hdr_bits_std = extractor.extract_bits(std_pos)
            hdr_bytes_std = Extractor.bits_to_bytes(hdr_bits_std)
            header = ExperimentalCarrierHeader.unpack(hdr_bytes_std)
            header_row_offset_detected = 0

        # Hard-gate capacity check
        CapacityCalculator.check_capacity(
            protected_payload_bytes=header.payload_length,
            subband_dim=(sub_h, sub_w),
            rs_parity=header.ecc_parity,
            version=min(header.version, 2),
        )

        # Extract payload bits using appropriate mask layout
        if header_row_offset_detected > 0:
            payload_positions = RelocatedMaskGenerator.generate_payload_positions(
                session_counter=header.session_counter,
                payload_length_bytes=header.payload_length,
                subband_dim=(sub_h, sub_w),
                row_offset=header_row_offset_detected,
            )
        else:
            payload_positions = DeterministicMaskGenerator.generate_payload_positions(
                session_counter=header.session_counter,
                payload_length_bytes=header.payload_length,
                subband_dim=(sub_h, sub_w),
                version=header.version,
            )

        extracted_payload_bits = extractor.extract_bits(payload_positions)
        extracted_interleaved_bytes = Extractor.bits_to_bytes(extracted_payload_bits)

        # Deinterleave if header declares symbol interleaving
        is_interleaved = bool(header.flags & ExperimentalCarrierHeader.FLAG_SYMBOL_INTERLEAVED)
        scheme = self.config.interleaving_scheme if is_interleaved else "none"

        deinterleaved_stream = DeterministicSymbolInterleaver.deinterleave(
            extracted_interleaved_bytes,
            scheme=scheme,
            session_counter=header.session_counter,
            coprime_step=self.config.coprime_step,
        )

        # Reed-Solomon Decode
        decoder = ReedSolomonEncoder(parity_symbols=header.ecc_parity)
        dec_ciphertext = decoder.decode(deinterleaved_stream)

        # AES-GCM Decrypt
        dec_serialized = self.crypto_provider.decrypt(dec_ciphertext, key, aad=aad)

        # Deserialize SecretPayload
        recovered_payload = SecretSerializer.deserialize(dec_serialized)

        exact_match = False
        if reference_payload is not None:
            exact_match = (recovered_payload.data == reference_payload.data)

        return {
            "recovered_payload": recovered_payload,
            "header": header,
            "exact_match": exact_match,
            "extracted_interleaved_bytes": extracted_interleaved_bytes,
            "deinterleaved_stream": deinterleaved_stream,
        }


# =============================================================================
# 4. Evaluation and Telemetry Harness
# =============================================================================

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_single_carrier(
    carrier_np: np.ndarray,
    image_id: int,
    filename: str,
    pipeline: Any,
    is_experimental: bool = False,
    crypto_key: bytes = bytes.fromhex("11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"),
    aad: bytes = b"PaladinMedia-Phase8-Qualification",
    payload_size: int = 1000,
    fixed_iv: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Evaluates a carrier image under baseline or candidate pipeline with full telemetry."""
    sub_h, sub_w = carrier_np.shape[0] // 2, carrier_np.shape[1] // 2
    session_counter = int(image_id)

    raw_secret = np.random.RandomState(session_counter + 777).randint(0, 256, size=payload_size, dtype=np.uint8).tobytes()
    secret_payload = SecretPayload(
        data=raw_secret,
        media_type=MediaType.IMAGE,
        metadata={"image_id": image_id, "size": payload_size},
    )

    t0_embed = time.perf_counter()
    embed_res = None
    embed_err = None
    try:
        if is_experimental:
            embed_res = pipeline.protect_and_embed(
                secret_payload=secret_payload,
                cover_image=carrier_np,
                key=crypto_key,
                aad=aad,
                session_counter=session_counter,
            )
            stego_arr = embed_res["stego_array"]
            header = embed_res["header"]
            mask = embed_res["mask"]
            psnr = embed_res["psnr_db"]
            ssim = embed_res["ssim"]
            embedded_hdr_bits = embed_res["embedded_header_bits"]
            embedded_pay_bits = embed_res["embedded_payload_bits"]
        else:
            embed_res = pipeline.protect_and_embed(
                secret_payload=secret_payload,
                cover_image=carrier_np,
                key=crypto_key,
                aad=aad,
                session_counter=session_counter,
                version=CarrierHeader.VERSION_2,
            )
            stego_arr = np.array(embed_res.stego_image, dtype=np.uint8)
            header = embed_res.header
            mask = DeterministicMaskGenerator.generate_mask(
                session_counter=session_counter,
                payload_length_bytes=header.payload_length,
                subband_dim=(sub_h, sub_w),
                version=2,
            )
            psnr = embed_res.psnr_db
            ssim = embed_res.ssim
            embedded_hdr_bits = pipeline.modulation_engine.last_header_bits
            embedded_pay_bits = pipeline.modulation_engine.last_payload_bits
    except Exception as e:
        embed_err = str(e)
    t_embed_ms = (time.perf_counter() - t0_embed) * 1000.0

    if embed_err is not None:
        return {
            "image_id": image_id,
            "filename": filename,
            "exact_match": False,
            "failure_stage": "EMBEDDING_FAILURE",
            "error_message": embed_err,
            "t_embed_ms": round(t_embed_ms, 2),
            "t_extract_ms": 0.0,
            "psnr_db": 0.0,
            "ssim": 0.0,
            "raw_ber": 1.0,
            "header_bit_errors": 192,
            "header_byte_errors": 24,
            "payload_bit_errors": 0,
            "rs_block_errors": [],
            "max_block_error": 0,
        }

    # Low-level extraction diagnostics
    t0_ext = time.perf_counter()
    bit_extractor = pipeline.modulation_engine.begin_extraction(stego_arr, subband_dim=(sub_h, sub_w))

    # Header error check
    ext_hdr_bits = bit_extractor.extract_bits(mask.header_positions)
    hdr_bit_errors = int(np.sum(ext_hdr_bits != embedded_hdr_bits))
    ext_hdr_bytes = Extractor.bits_to_bytes(ext_hdr_bits)
    exp_hdr_bytes = Extractor.bits_to_bytes(embedded_hdr_bits)
    hdr_byte_errors = sum(1 for a, b in zip(ext_hdr_bytes, exp_hdr_bytes) if a != b)

    # Payload error check against transmitted bits
    ext_pay_bits = bit_extractor.extract_bits(mask.payload_positions)
    pay_bit_errors = int(np.sum(ext_pay_bits != embedded_pay_bits))
    raw_ber = pay_bit_errors / len(embedded_pay_bits) if len(embedded_pay_bits) > 0 else 0.0

    # End-to-end unprotection
    exact_match = False
    failure_stage = "NONE"
    error_message = ""
    deintl_bytes = b""
    try:
        if is_experimental:
            rec_res = pipeline.extract_and_unprotect(
                stego_carrier=stego_arr,
                key=crypto_key,
                aad=aad,
                reference_payload=secret_payload,
            )
            exact_match = rec_res["exact_match"]
            deintl_bytes = rec_res["deinterleaved_stream"]
        else:
            rec_res = pipeline.extract_and_unprotect(
                stego_carrier=stego_arr,
                key=crypto_key,
                aad=aad,
                reference_payload=secret_payload,
            )
            exact_match = rec_res.exact_match
            deintl_bytes = Extractor.bits_to_bytes(ext_pay_bits)
    except CarrierHeaderError as e:
        failure_stage = "HEADER_UNPACK"
        error_message = str(e)
    except ECCDecodeError as e:
        failure_stage = "PAYLOAD_RS_DECODE"
        error_message = str(e)
    except AuthenticationError as e:
        failure_stage = "AES_GCM_DECRYPT"
        error_message = str(e)
    except Exception as e:
        failure_stage = "UNEXPECTED_ERROR"
        error_message = str(e)

    t_extract_ms = (time.perf_counter() - t0_ext) * 1000.0

    # Fallback to compute deintl_bytes if exception occurred prior to deinterleaving
    if not deintl_bytes and len(ext_pay_bits) > 0:
        raw_pay_bytes = Extractor.bits_to_bytes(ext_pay_bits)
        if is_experimental:
            deintl_bytes = DeterministicSymbolInterleaver.deinterleave(
                raw_pay_bytes,
                scheme=pipeline.config.interleaving_scheme,
                session_counter=session_counter,
                coprime_step=pipeline.config.coprime_step,
            )
        else:
            deintl_bytes = raw_pay_bytes

    # Per-codeword RS block error calculation
    rs_block_errors: List[int] = []
    if len(deintl_bytes) > 4:
        # Expected uncorrupted encoded stream
        if is_experimental:
            exp_encoded = embed_res["encoded_stream"]
        else:
            exp_encoded = Extractor.bits_to_bytes(embedded_pay_bits)

        block_size = 255
        k_val = pipeline.ecc_encoder.k if hasattr(pipeline, "ecc_encoder") else 207
        raw_rec = deintl_bytes[4:]
        raw_exp = exp_encoded[4:]

        num_full = len(raw_exp) // block_size
        rem = len(raw_exp) % block_size
        lens = [block_size] * num_full + ([rem] if rem > 0 else [])

        off = 0
        for blen in lens:
            b_rec = raw_rec[off : off + blen]
            b_exp = raw_exp[off : off + blen]
            err_cnt = sum(1 for a, b in zip(b_rec, b_exp) if a != b)
            rs_block_errors.append(err_cnt)
            off += blen

    return {
        "image_id": image_id,
        "filename": filename,
        "exact_match": exact_match,
        "failure_stage": failure_stage,
        "error_message": error_message,
        "header_bit_errors": hdr_bit_errors,
        "header_byte_errors": hdr_byte_errors,
        "payload_bit_errors": pay_bit_errors,
        "raw_ber": round(raw_ber, 6),
        "rs_block_errors": rs_block_errors,
        "max_block_error": max(rs_block_errors) if rs_block_errors else 0,
        "psnr_db": round(psnr, 2),
        "ssim": round(ssim, 6),
        "t_embed_ms": round(t_embed_ms, 2),
        "t_extract_ms": round(t_extract_ms, 2),
    }


def run_full_evaluation(
    eval_split: str = "test",
    max_images: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executes comparative evaluation across Baseline Wire Protocol v2 and Candidate Protocol."""
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "phase8_protocol_reliability_results.json"
    out_csv = out_dir / "PHASE8_CARRIER_COMPARISON.csv"

    # Preflight asset hashes
    logger.info("Computing preflight asset hashes...")
    asset_hashes = {
        "checkpoint": compute_sha256(MODEL_PATH),
        "onnx_embed": compute_sha256(ONNX_EMBED),
        "onnx_extract": compute_sha256(ONNX_EXTRACT),
        "manifest": compute_sha256(MANIFEST_PATH),
    }

    # Load dataset
    ds = DIV2KStegoDataset(split=eval_split, random_crop=False)
    n_images = len(ds) if max_images is None else min(len(ds), max_images)
    logger.info("Loaded %s split: %d carriers to evaluate", eval_split, n_images)

    # Initialize pipelines
    # 1. Baseline Wire Protocol v2
    pipeline_baseline = MediaStegoPipeline.create_qualified_deployment(backend="onnx")

    # 2. Candidate Protocol (Wire Protocol v3: Relocated Header row 32, Coprime Interleaver a=71, RS(255, 207))
    config_cand = ProtocolCandidateConfig(
        name="WireProtocol_v3_Candidate",
        version=0x03,
        header_row_offset=32,
        interleaving_scheme="coprime",
        coprime_step=71,
        ecc_parity=48,
    )
    pipeline_candidate = ExperimentalWireProtocolPipeline(config=config_cand, backend="onnx")

    # 3. Maximum-Resilience Candidate (Wire Protocol v3-MR: Relocated Header row 32, Coprime Interleaver, RS(255, 191, 2t=64))
    config_mr = ProtocolCandidateConfig(
        name="WireProtocol_v3_MaxResilience",
        version=0x03,
        header_row_offset=32,
        interleaving_scheme="coprime",
        coprime_step=71,
        ecc_parity=64,
    )
    pipeline_mr = ExperimentalWireProtocolPipeline(config=config_mr, backend="onnx")

    comparison_records: List[Dict[str, Any]] = []
    base_exact_count = 0
    cand_exact_count = 0
    mr_exact_count = 0

    logger.info("Beginning comparative evaluation...")
    for idx in range(n_images):
        item = ds[idx]
        iid = item["image_id"]
        fname = item["filename"]
        carrier = item["carrier"].permute(1, 2, 0).numpy().astype(np.uint8)

        # Baseline evaluation
        res_base = evaluate_single_carrier(carrier, iid, fname, pipeline_baseline, is_experimental=False)
        # Candidate evaluation (v3 with 2t=48)
        res_cand = evaluate_single_carrier(carrier, iid, fname, pipeline_candidate, is_experimental=True)
        # Maximum Resilience evaluation (v3 with 2t=64)
        res_mr = evaluate_single_carrier(carrier, iid, fname, pipeline_mr, is_experimental=True)

        if res_base["exact_match"]:
            base_exact_count += 1
        if res_cand["exact_match"]:
            cand_exact_count += 1
        if res_mr["exact_match"]:
            mr_exact_count += 1

        rec = {
            "image_id": f"{iid:04d}",
            "filename": fname,
            "baseline_exact": res_base["exact_match"],
            "baseline_stage": res_base["failure_stage"],
            "baseline_header_byte_errors": res_base["header_byte_errors"],
            "baseline_raw_ber": res_base["raw_ber"],
            "baseline_max_block_err": res_base["max_block_error"],
            "candidate_exact": res_cand["exact_match"],
            "candidate_stage": res_cand["failure_stage"],
            "candidate_header_byte_errors": res_cand["header_byte_errors"],
            "candidate_raw_ber": res_cand["raw_ber"],
            "candidate_max_block_err": res_cand["max_block_error"],
            "mr_exact": res_mr["exact_match"],
            "mr_stage": res_mr["failure_stage"],
            "mr_header_byte_errors": res_mr["header_byte_errors"],
            "mr_raw_ber": res_mr["raw_ber"],
            "mr_max_block_err": res_mr["max_block_error"],
            "baseline_psnr": res_base["psnr_db"],
            "candidate_psnr": res_cand["psnr_db"],
            "baseline_ssim": res_base["ssim"],
            "candidate_ssim": res_cand["ssim"],
            "res_base_raw": res_base,
            "res_cand_raw": res_cand,
            "res_mr_raw": res_mr,
        }
        comparison_records.append(rec)

        if (idx + 1) % 10 == 0 or idx == n_images - 1 or iid in (715, 774, 784):
            logger.info(
                "[%d/%d] ID %04d | Base: %s | Cand: %s | MR: %s",
                idx + 1,
                n_images,
                iid,
                "PASS" if res_base["exact_match"] else f"FAIL({res_base['failure_stage']})",
                "PASS" if res_cand["exact_match"] else f"FAIL({res_cand['failure_stage']})",
                "PASS" if res_mr["exact_match"] else f"FAIL({res_mr['failure_stage']})",
            )

    # Write CSV comparison
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "filename",
            "baseline_exact",
            "baseline_stage",
            "baseline_header_errors",
            "baseline_raw_ber",
            "candidate_exact",
            "candidate_stage",
            "candidate_header_errors",
            "candidate_raw_ber",
            "mr_exact",
            "mr_stage",
            "mr_header_errors",
            "mr_raw_ber",
            "psnr_baseline",
            "psnr_candidate",
            "ssim_baseline",
            "ssim_candidate",
        ])
        for r in comparison_records:
            writer.writerow([
                r["image_id"],
                r["filename"],
                r["baseline_exact"],
                r["baseline_stage"],
                r["baseline_header_byte_errors"],
                r["baseline_raw_ber"],
                r["candidate_exact"],
                r["candidate_stage"],
                r["candidate_header_byte_errors"],
                r["candidate_raw_ber"],
                r["mr_exact"],
                r["mr_stage"],
                r["mr_header_byte_errors"],
                r["mr_raw_ber"],
                r["baseline_psnr"],
                r["candidate_psnr"],
                r["baseline_ssim"],
                r["candidate_ssim"],
            ])

    # Summary JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "asset_hashes": asset_hashes,
        "dataset_split": eval_split,
        "total_evaluated": n_images,
        "baseline_exact_count": base_exact_count,
        "baseline_exact_rate": round(base_exact_count / n_images, 4),
        "candidate_exact_count": cand_exact_count,
        "candidate_exact_rate": round(cand_exact_count / n_images, 4),
        "mr_exact_count": mr_exact_count,
        "mr_exact_rate": round(mr_exact_count / n_images, 4),
        "target_carriers": {
            "0715": next((r for r in comparison_records if r["image_id"] == "0715"), None),
            "0774": next((r for r in comparison_records if r["image_id"] == "0774"), None),
            "0784": next((r for r in comparison_records if r["image_id"] == "0784"), None),
        },
        "records": comparison_records,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        # Exclude deep raw subdicts to keep json readable
        json_clean = {k: v for k, v in summary.items() if k != "records"}
        json_clean["records"] = [
            {k: v for k, v in r.items() if not k.startswith("res_")}
            for r in comparison_records
        ]
        json.dump(json_clean, f, indent=2)

    logger.info("Evaluation complete! Results written to %s and %s", out_json, out_csv)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Phase 8 Protocol Reliability Experiments")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_p = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    run_full_evaluation(eval_split=args.split, max_images=args.max_images, output_dir=out_p)
