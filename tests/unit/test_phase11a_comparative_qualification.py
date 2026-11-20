"""Phase 11A — Wire Protocol v3 Comparative Qualification Test Suite.

Validates Candidate Wire Protocol v3 against Production Wire Protocol v2
across the six mandated qualification workstreams:
- B1: Functional Correctness & Exact Payload Recovery
- B2: Header Reliability, Error Correction & Detection Modes
- B3: Error & Corruption Handling (Burst Dispersion & Anchor Relocation)
- B4: Capacity Enforcement & Boundary Condition Hard-Gates
- B5: Regression, Fail-Closed Security & Cross-Protocol Compatibility
- B6: Determinism, Reproducibility & Mathematical Invertibility
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import struct
from typing import Optional
import numpy as np
from PIL import Image
import pytest
import reedsolo

from scripts.experiments.phase8_protocol_experiments import (
    DeterministicSymbolInterleaver,
    ExperimentalCarrierHeader,
    ExperimentalWireProtocolPipeline,
    ProtocolCandidateConfig,
    RelocatedMaskGenerator,
)
from src.carrier.capacity import CapacityCalculator, PayloadCapacityExceededError
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import DeterministicMaskGenerator, MaskGenerationError
from src.core.exceptions import ECCDecodeError, PaladinSecurityError
from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


@pytest.fixture
def standard_cover_image() -> np.ndarray:
    """Returns the standard 512x512 carrier image fixture."""
    fixture_path = Path("tests/fixtures/carrier_512.png")
    if fixture_path.exists():
        return np.array(Image.open(fixture_path).convert("RGB"), dtype=np.uint8)
    # Fallback to smooth gradient test carrier if fixture missing
    y, x = np.mgrid[0:512, 0:512]
    r = ((x / 512.0) * 180 + 40).astype(np.uint8)
    g = ((y / 512.0) * 180 + 40).astype(np.uint8)
    b = (((x + y) / 1024.0) * 180 + 40).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


@pytest.fixture
def shared_crypto_key() -> bytes:
    return bytes.fromhex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")


@pytest.fixture
def v3_candidate_pipeline() -> ExperimentalWireProtocolPipeline:
    config = ProtocolCandidateConfig(
        name="v3_Candidate_Qualified",
        version=0x03,
        header_row_offset=32,
        interleaving_scheme="coprime",
        coprime_step=71,
        ecc_parity=48,
    )
    return ExperimentalWireProtocolPipeline(config=config, backend="onnx")


@pytest.fixture
def v2_production_pipeline() -> MediaStegoPipeline:
    return MediaStegoPipeline.create_qualified_deployment(backend="onnx")


# =============================================================================
# Workstream B1: Functional Correctness & Payload Recovery
# =============================================================================

class TestB1FunctionalCorrectnessPayloadRecovery:
    """Validates exact payload recovery across boundary and representative sizes."""

    @pytest.mark.parametrize("payload_size", [0, 1, 100, 1500, 3500, 6578])
    def test_v3_exact_recovery_across_payload_sizes(
        self,
        v3_candidate_pipeline: ExperimentalWireProtocolPipeline,
        standard_cover_image: np.ndarray,
        shared_crypto_key: bytes,
        payload_size: int,
    ) -> None:
        """Verifies candidate v3 recovers payloads of varying sizes bit-for-bit."""
        rng = np.random.RandomState(42 + payload_size)
        raw_data = rng.randint(0, 256, size=payload_size, dtype=np.uint8).tobytes()
        metadata = None if payload_size == 6578 else {"size": payload_size, "test": "B1_Functional"}
        secret = SecretPayload(data=raw_data, media_type=MediaType.TEXT, metadata=metadata)

        embed_res = v3_candidate_pipeline.protect_and_embed(
            secret_payload=secret,
            cover_image=standard_cover_image,
            key=shared_crypto_key,
            aad=b"PaladinMedia-Phase11A-B1",
            session_counter=100 + payload_size,
        )

        min_psnr = 40.0 if payload_size == 6578 else 45.0
        assert embed_res["psnr_db"] > min_psnr
        assert embed_res["ssim"] > 0.99

        rec_res = v3_candidate_pipeline.extract_and_unprotect(
            stego_carrier=embed_res["stego_array"],
            key=shared_crypto_key,
            aad=b"PaladinMedia-Phase11A-B1",
            reference_payload=secret,
        )

        assert rec_res["exact_match"] is True
        assert rec_res["recovered_payload"].data == raw_data
        assert rec_res["recovered_payload"].media_type == MediaType.TEXT
        assert rec_res["recovered_payload"].metadata == metadata
        assert rec_res["header"].version == ExperimentalCarrierHeader.VERSION_3

    @pytest.mark.parametrize("payload_size", [0, 1, 100, 1500, 3500, 6578])
    def test_v2_exact_recovery_across_payload_sizes(
        self,
        v2_production_pipeline: MediaStegoPipeline,
        standard_cover_image: np.ndarray,
        shared_crypto_key: bytes,
        payload_size: int,
    ) -> None:
        """Verifies production v2 baseline recovers payloads across boundary sizes."""
        rng = np.random.RandomState(84 + payload_size)
        raw_data = rng.randint(0, 256, size=payload_size, dtype=np.uint8).tobytes()
        metadata = None if payload_size == 6578 else {"size": payload_size, "protocol": "v2"}
        secret = SecretPayload(data=raw_data, media_type=MediaType.RAW_BYTES, metadata=metadata)

        embed_res = v2_production_pipeline.protect_and_embed(
            secret_payload=secret,
            cover_image=standard_cover_image,
            key=shared_crypto_key,
            aad=b"PaladinMedia-Phase11A-B1",
            session_counter=200 + payload_size,
            version=CarrierHeader.VERSION_2,
        )

        rec_res = v2_production_pipeline.extract_and_unprotect(
            stego_carrier=embed_res.stego_image,
            key=shared_crypto_key,
            aad=b"PaladinMedia-Phase11A-B1",
            reference_payload=secret,
        )

        assert rec_res.exact_match is True
        assert rec_res.recovered_payload.data == raw_data
        assert rec_res.recovered_payload.media_type == MediaType.RAW_BYTES
        assert rec_res.recovered_payload.metadata == metadata
        assert rec_res.header.version == CarrierHeader.VERSION_2

    def test_media_type_and_metadata_fidelity(
        self,
        v3_candidate_pipeline: ExperimentalWireProtocolPipeline,
        standard_cover_image: np.ndarray,
        shared_crypto_key: bytes,
    ) -> None:
        """Verifies all supported media types and complex metadata serialize and deserialize faithfully."""
        for media_type in (MediaType.TEXT, MediaType.IMAGE, MediaType.AUDIO, MediaType.RAW_BYTES, MediaType.JSON):
            meta = {
                "author": "PaladinAgent",
                "media_type": media_type.name,
                "nested": {"level": 2, "tags": ["post-quantum", "stego"]},
            }
            secret = SecretPayload(data=b"Fidelity payload: " + media_type.name.encode(), media_type=media_type, metadata=meta)
            embed_res = v3_candidate_pipeline.protect_and_embed(
                secret_payload=secret,
                cover_image=standard_cover_image,
                key=shared_crypto_key,
                aad=b"AAD-Fidelity",
                session_counter=55,
            )
            rec_res = v3_candidate_pipeline.extract_and_unprotect(
                stego_carrier=embed_res["stego_array"],
                key=shared_crypto_key,
                aad=b"AAD-Fidelity",
                reference_payload=secret,
            )
            assert rec_res["exact_match"] is True
            assert rec_res["recovered_payload"].media_type == media_type
            assert rec_res["recovered_payload"].metadata == meta


# =============================================================================
# Workstream B2: Header Reliability & Detection
# =============================================================================

class TestB2HeaderReliabilityAndDetection:
    """Validates header parsing, error correction bounds, and fail-closed rejections."""

    @pytest.mark.parametrize("corrupt_count", [1, 2, 3, 4])
    def test_v3_header_error_correction_within_rs_bound(self, corrupt_count: int) -> None:
        """Verifies v3 header RS(24, 16) corrects up to 4 corrupted bytes."""
        hdr = ExperimentalCarrierHeader.create_v3(
            session_counter=9999,
            payload_length=2048,
            ecc_parity=48,
            relocated_header=True,
            symbol_interleaved=True,
        )
        packed = bytearray(hdr.pack())
        assert len(packed) == 24

        rng = random.Random(1337 + corrupt_count)
        corrupted_indices = rng.sample(range(24), corrupt_count)
        for idx in corrupted_indices:
            packed[idx] ^= 0xAA

        unpacked = ExperimentalCarrierHeader.unpack(bytes(packed))
        assert unpacked.version == ExperimentalCarrierHeader.VERSION_3
        assert unpacked.session_counter == 9999
        assert unpacked.payload_length == 2048
        assert unpacked.ecc_parity == 48
        assert unpacked.flags & ExperimentalCarrierHeader.FLAG_RELOCATED_HEADER
        assert unpacked.flags & ExperimentalCarrierHeader.FLAG_SYMBOL_INTERLEAVED

    @pytest.mark.parametrize("corrupt_count", [5, 6, 8, 12])
    def test_v3_header_fail_closed_on_excessive_corruption(self, corrupt_count: int) -> None:
        """Verifies uncorrectable header noise (> 4 bytes) fails closed with CarrierHeaderError."""
        hdr = ExperimentalCarrierHeader.create_v3(
            session_counter=101,
            payload_length=512,
            ecc_parity=48,
        )
        packed = bytearray(hdr.pack())

        rng = random.Random(2026 + corrupt_count)
        corrupted_indices = rng.sample(range(24), corrupt_count)
        for idx in corrupted_indices:
            packed[idx] ^= 0x55

        with pytest.raises(CarrierHeaderError):
            ExperimentalCarrierHeader.unpack(bytes(packed))

    def test_header_detection_failure_modes(self) -> None:
        """Verifies fail-closed rejection on truncated, malformed, or uncorrectable headers."""
        # Truncated
        with pytest.raises(CarrierHeaderError):
            ExperimentalCarrierHeader.unpack(b"")
        with pytest.raises(CarrierHeaderError):
            ExperimentalCarrierHeader.unpack(b"PLDH\x03\x00")
        with pytest.raises(CarrierHeaderError):
            ExperimentalCarrierHeader.unpack(b"\x00" * 15)

        # Uncorrectable Magic (> 4 byte errors)
        hdr = ExperimentalCarrierHeader.create_v3(1, 100)
        bad_magic = bytearray(hdr.pack())
        bad_magic[0:4] = b"NOPE"
        bad_magic[16:18] = b"\x00\x00"  # 6 bytes corrupted > t=4
        with pytest.raises(CarrierHeaderError):
            ExperimentalCarrierHeader.unpack(bytes(bad_magic))

        # Unsupported Version
        raw_v4 = struct.pack(">4sBBHII", b"PLDH", 0x04, 0x00, 0x0000, 1, 100)
        rsc = reedsolo.RSCodec(8)
        codeword_v4 = bytes(rsc.encode(raw_v4))
        with pytest.raises(CarrierHeaderError):
            ExperimentalCarrierHeader.unpack(codeword_v4)

        # Non-zero Reserved Field
        raw_bad_res = struct.pack(">4sBBHII", b"PLDH", 0x03, 0x31, 0x0001, 1, 100)
        codeword_bad_res = bytes(rsc.encode(raw_bad_res))
        with pytest.raises(CarrierHeaderError):
            ExperimentalCarrierHeader.unpack(codeword_bad_res)

        # Unsupported ECC profile
        raw_bad_ecc = struct.pack(">4sBBHII", b"PLDH", 0x03, 0x3F, 0x0000, 1, 100)
        codeword_bad_ecc = bytes(rsc.encode(raw_bad_ecc))
        with pytest.raises(CarrierHeaderError):
            ExperimentalCarrierHeader.unpack(codeword_bad_ecc)


# =============================================================================
# Workstream B3: Error and Corruption Handling
# =============================================================================

class TestB3ErrorAndCorruptionHandling:
    """Validates burst error dispersion and anchor relocation resilience."""

    def test_burst_error_dispersion_v2_failure_vs_v3_recovery(self) -> None:
        """Demonstrates that a 30-symbol burst destroys v2 (sequential) but is corrected by v3 (coprime interleaved)."""
        # Multi-codeword payload: 6 blocks of RS(255, 207)
        n_blocks = 6
        block_len = 255
        k_len = 207
        raw_data = bytes([0x55] * (n_blocks * k_len))

        # 1. Sequential RS encoding (v2 wire format)
        encoder = ReedSolomonEncoder(parity_symbols=48)
        seq_encoded = encoder.encode(raw_data)
        assert len(seq_encoded) == 4 + n_blocks * block_len

        # 2. Interleaved RS encoding (v3 wire format)
        v3_encoded = DeterministicSymbolInterleaver.interleave(
            seq_encoded, scheme="coprime", session_counter=774, coprime_step=71
        )

        # 3. Inject a localized 30-byte contiguous burst error
        burst_offset = 50
        burst_len = 30  # Exceeds t=24 for a single block

        # Corrupt v2 sequential stream
        corrupt_v2 = bytearray(seq_encoded)
        for i in range(burst_offset, burst_offset + burst_len):
            corrupt_v2[i] ^= 0xFF

        # Corrupt v3 interleaved stream at exact same channel position
        corrupt_v3 = bytearray(v3_encoded)
        for i in range(burst_offset, burst_offset + burst_len):
            corrupt_v3[i] ^= 0xFF

        # --- v2 Sequential Evaluation ---
        # The 30 errors fall entirely within block 0 (indices 4..258).
        # Reed-Solomon t=24 cannot correct 30 errors -> must fail
        with pytest.raises(ECCDecodeError):
            encoder.decode(bytes(corrupt_v2))

        # --- v3 Interleaved Evaluation ---
        # Receiver deinterleaves the stream first
        deinterleaved_v3 = DeterministicSymbolInterleaver.deinterleave(
            bytes(corrupt_v3), scheme="coprime", session_counter=774, coprime_step=71
        )

        # Count errors per block after deinterleaving
        errors_per_block = []
        for b in range(n_blocks):
            blk_orig = seq_encoded[4 + b * block_len : 4 + (b + 1) * block_len]
            blk_deintl = deinterleaved_v3[4 + b * block_len : 4 + (b + 1) * block_len]
            errs = sum(1 for x, y in zip(blk_orig, blk_deintl) if x != y)
            errors_per_block.append(errs)

        # Assert no single block exceeds the correction threshold t=24
        assert max(errors_per_block) <= 8
        assert sum(errors_per_block) == burst_len

        # Decoder decodes all blocks cleanly!
        recovered_v3 = encoder.decode(deinterleaved_v3)
        assert recovered_v3 == raw_data

    def test_anchor_corruption_resilience_row0_vs_row32(
        self,
        standard_cover_image: np.ndarray,
        shared_crypto_key: bytes,
    ) -> None:
        """Demonstrates that boundary zero-clipping on Row 0 destroys v2 but leaves v3 completely unaffected."""
        sub_h, sub_w = 256, 256

        # 1. Embed identical message under v2 and v3
        secret = SecretPayload(data=b"Boundary clipping resilience test.", media_type=MediaType.TEXT, metadata={})

        v2_pipe = MediaStegoPipeline.create_qualified_deployment(backend="onnx")
        stego_v2 = v2_pipe.protect_and_embed(
            secret_payload=secret,
            cover_image=standard_cover_image,
            key=shared_crypto_key,
            session_counter=715,
            version=CarrierHeader.VERSION_2,
        )

        config_v3 = ProtocolCandidateConfig(name="v3_Resilience", version=0x03, header_row_offset=32)
        v3_pipe = ExperimentalWireProtocolPipeline(config=config_v3, backend="onnx")
        stego_v3 = v3_pipe.protect_and_embed(
            secret_payload=secret,
            cover_image=standard_cover_image,
            key=shared_crypto_key,
            session_counter=715,
        )

        # 2. Simulate border zero-clipping: zero out top spatial row of both carriers
        carrier_v2_clipped = np.array(stego_v2.stego_image, dtype=np.uint8)
        carrier_v2_clipped[0:2, :] = 0  # Zero out spatial rows 0-1 (which form Haar subband row 0)

        carrier_v3_clipped = np.copy(stego_v3["stego_array"])
        carrier_v3_clipped[0:2, :] = 0  # Same spatial zero-out

        # 3. v2 header is located on subband row 0 -> severely corrupted (> 4 byte errors) -> fails closed
        with pytest.raises(CarrierHeaderError):
            v2_pipe.extract_and_unprotect(carrier_v2_clipped, key=shared_crypto_key)

        # 4. v3 header is located on subband row 32 (spatial rows 64-65) -> unaffected -> unpacks cleanly
        ext_v3 = v3_pipe.modulation_engine.begin_extraction(carrier_v3_clipped, subband_dim=(sub_h, sub_w))
        v3_hdr_pos = RelocatedMaskGenerator.generate_header_positions((sub_h, sub_w), row_offset=32)
        v3_hdr_bits = ext_v3.extract_bits(v3_hdr_pos)
        v3_hdr_bytes = Extractor.bits_to_bytes(v3_hdr_bits)
        unpacked_v3 = ExperimentalCarrierHeader.unpack(v3_hdr_bytes)
        assert unpacked_v3.version == ExperimentalCarrierHeader.VERSION_3
        assert unpacked_v3.session_counter == 715


# =============================================================================
# Workstream B4: Capacity & Boundary Validation
# =============================================================================

class TestB4CapacityAndBoundaryValidation:
    """Validates hard-gate capacity limits and boundary conditions."""

    def test_subband_capacity_constants(self) -> None:
        """Verifies exact capacity numbers for 256x256 transform subband."""
        total_positions = 256 * 256  # 65,536 positions
        header_bits = 192            # 24 bytes (RS(24, 16))
        max_payload_bits = 65344     # 8,168 bytes

        assert RelocatedMaskGenerator.TOTAL_POSITIONS == total_positions
        assert RelocatedMaskGenerator.HEADER_BITS == header_bits
        assert RelocatedMaskGenerator.MAX_PAYLOAD_BITS == max_payload_bits

        assert DeterministicMaskGenerator.TOTAL_POSITIONS == total_positions
        assert DeterministicMaskGenerator.HEADER_BITS_V2 == header_bits
        assert DeterministicMaskGenerator.MAX_PAYLOAD_BITS_V2 == max_payload_bits

    def test_capacity_check_hard_gate(self) -> None:
        """Verifies CapacityCalculator.check_capacity enforces the 8,168 byte limit fail-closed."""
        # Exact limit succeeds
        rep = CapacityCalculator.check_capacity(protected_payload_bytes=8168, subband_dim=(256, 256), rs_parity=48)
        assert rep.max_protected_bytes == 8168
        assert rep.protected_payload_bytes == 8168
        assert rep.usable_capacity_bytes == 6578
        assert rep.fits is True

        # 1 byte overflow fails closed
        with pytest.raises(PayloadCapacityExceededError):
            CapacityCalculator.check_capacity(protected_payload_bytes=8169, subband_dim=(256, 256))

    def test_mask_generation_overflow_rejected(self) -> None:
        """Verifies mask generators fail closed on payload exceeding 8,168 bytes."""
        with pytest.raises(PayloadCapacityExceededError):
            RelocatedMaskGenerator.generate_mask(session_counter=1, payload_length_bytes=8169)

        with pytest.raises(MaskGenerationError):
            DeterministicMaskGenerator.generate_mask(session_counter=1, payload_length_bytes=8169, version=2)

    def test_boundary_payload_sizes(self) -> None:
        """Verifies 0-byte and 1-byte payloads generate valid non-overlapping coordinates."""
        for gen in (RelocatedMaskGenerator, DeterministicMaskGenerator):
            # 0 bytes
            mask_0 = gen.generate_mask(session_counter=1, payload_length_bytes=0)
            assert len(mask_0.header_positions) == 192
            assert len(mask_0.payload_positions) == 0

            # 1 byte
            mask_1 = gen.generate_mask(session_counter=1, payload_length_bytes=1)
            assert len(mask_1.header_positions) == 192
            assert len(mask_1.payload_positions) == 8
            assert set(mask_1.header_positions).isdisjoint(set(mask_1.payload_positions))


# =============================================================================
# Workstream B5: Regression, Fail-Closed Security & Compatibility
# =============================================================================

class TestB5RegressionAndCompatibility:
    """Validates backward compatibility of v3 receiver and fail-closed rejection by v2 receiver."""

    def test_candidate_v3_receiver_decodes_production_v2_carrier(
        self,
        v2_production_pipeline: MediaStegoPipeline,
        v3_candidate_pipeline: ExperimentalWireProtocolPipeline,
        standard_cover_image: np.ndarray,
        shared_crypto_key: bytes,
    ) -> None:
        """Verifies candidate v3 receiver cleanly detects, decodes, and recovers production v2 carriers."""
        secret = SecretPayload(data=b"Backward compatibility validation payload.", media_type=MediaType.TEXT, metadata={"legacy": True})

        # Embed with production v2
        embed_v2 = v2_production_pipeline.protect_and_embed(
            secret_payload=secret,
            cover_image=standard_cover_image,
            key=shared_crypto_key,
            aad=b"Paladin-Compat",
            session_counter=888,
            version=CarrierHeader.VERSION_2,
        )

        # Extract with candidate v3
        rec_v3 = v3_candidate_pipeline.extract_and_unprotect(
            stego_carrier=np.array(embed_v2.stego_image, dtype=np.uint8),
            key=shared_crypto_key,
            aad=b"Paladin-Compat",
            reference_payload=secret,
        )

        assert rec_v3["exact_match"] is True
        assert rec_v3["recovered_payload"].data == secret.data
        assert rec_v3["header"].version == CarrierHeader.VERSION_2

    def test_production_v2_receiver_rejects_v3_carrier_fail_closed(
        self,
        v2_production_pipeline: MediaStegoPipeline,
        v3_candidate_pipeline: ExperimentalWireProtocolPipeline,
        standard_cover_image: np.ndarray,
        shared_crypto_key: bytes,
    ) -> None:
        """Verifies production v2 receiver rejects candidate v3 carriers fail-closed without corrupt recovery."""
        secret = SecretPayload(data=b"Candidate v3 carrier content.", media_type=MediaType.TEXT, metadata={})

        # Embed with candidate v3 (relocated header + symbol interleaving)
        embed_v3 = v3_candidate_pipeline.protect_and_embed(
            secret_payload=secret,
            cover_image=standard_cover_image,
            key=shared_crypto_key,
            session_counter=999,
        )

        # Production v2 attempts extraction from row 0: does not find v2 header -> fails closed
        with pytest.raises(CarrierHeaderError):
            v2_production_pipeline.extract_and_unprotect(
                stego_carrier=embed_v3["stego_array"],
                key=shared_crypto_key,
            )

    def test_production_v2_header_unpack_rejects_v3_flags_fail_closed(self) -> None:
        """Verifies CarrierHeader.unpack rejects v3 headers with invalid flags (bits 4-7) fail-closed."""
        v3_hdr = ExperimentalCarrierHeader.create_v3(
            session_counter=123,
            payload_length=500,
            ecc_parity=48,
            relocated_header=True,
            symbol_interleaved=True,
        )
        packed_v3 = v3_hdr.pack()

        # Production CarrierHeader unpack must reject
        with pytest.raises(CarrierHeaderError) as exc_info:
            CarrierHeader.unpack(packed_v3)
        assert "Invalid flag bits" in str(exc_info.value) or "Unsupported carrier header version" in str(exc_info.value) or "Uncorrectable" in str(exc_info.value)


# =============================================================================
# Workstream B6: Determinism & Reproducibility
# =============================================================================

class TestB6DeterminismAndReproducibility:
    """Validates strict mathematical determinism, invertibility, and reproducibility."""

    def test_deterministic_mask_reproducibility(self) -> None:
        """Verifies masks are identical across multiple invocations with identical parameters."""
        for counter in (1, 42, 1000, 99999):
            for p_len in (0, 10, 500, 4000):
                m1 = RelocatedMaskGenerator.generate_mask(counter, p_len, subband_dim=256, row_offset=32)
                m2 = RelocatedMaskGenerator.generate_mask(counter, p_len, subband_dim=256, row_offset=32)
                assert m1.header_positions == m2.header_positions
                assert m1.payload_positions == m2.payload_positions

    @pytest.mark.parametrize("length", [5, 17, 64, 128, 255, 510, 1024, 2048])
    def test_coprime_interleaver_invertibility_exhaustive(self, length: int) -> None:
        """Verifies deinterleave(interleave(x)) == x for coprime permutation scheme."""
        rng = np.random.RandomState(length * 7)
        stream = b"\x00\x00\x01\x00" + bytes(rng.randint(0, 256, size=length - 4, dtype=np.uint8))
        interleaved = DeterministicSymbolInterleaver.interleave(stream, scheme="coprime", coprime_step=71)
        deinterleaved = DeterministicSymbolInterleaver.deinterleave(interleaved, scheme="coprime", coprime_step=71)
        assert deinterleaved == stream
        assert interleaved[:4] == stream[:4]  # Prefix preserved

    def test_end_to_end_deterministic_embedding(
        self,
        v3_candidate_pipeline: ExperimentalWireProtocolPipeline,
        standard_cover_image: np.ndarray,
        shared_crypto_key: bytes,
    ) -> None:
        """Verifies identical inputs and deterministic IV yield bit-identical stego images."""
        secret = SecretPayload(data=b"Deterministic embedding verification test.", media_type=MediaType.TEXT, metadata={})

        class DeterministicCryptoProvider(AESGCMProvider):
            def encrypt(self, plaintext: bytes, key: bytes, aad: Optional[bytes] = None, nonce: Optional[bytes] = None) -> bytes:
                return super().encrypt(plaintext, key, aad=aad, nonce=b"\x01" * 12)

        det_pipeline = ExperimentalWireProtocolPipeline(
            config=v3_candidate_pipeline.config,
            backend="onnx",
            crypto_provider=DeterministicCryptoProvider(),
        )

        res1 = det_pipeline.protect_and_embed(secret, standard_cover_image, shared_crypto_key, session_counter=77)
        res2 = det_pipeline.protect_and_embed(secret, standard_cover_image, shared_crypto_key, session_counter=77)

        assert np.array_equal(res1["stego_array"], res2["stego_array"])
        assert res1["psnr_db"] == res2["psnr_db"]
        assert res1["ssim"] == res2["ssim"]


# =============================================================================
# Workstream B: Capacity Gate Reconciliation Tests
# =============================================================================

class TestWorkstreamBCapacityReconciliation:
    """Validates exact capacity limits, boundary overflows, and metadata accounting across profiles."""

    def test_v2_and_v3_standard_capacity_boundary(self) -> None:
        """Verifies exact 6,578 byte maximum for v2 and v3 standard (2t=48), and failure at 6,579."""
        crypto = AESGCMProvider()
        encoder = ReedSolomonEncoder(parity_symbols=48)

        # 1. Exact maximum 6,578 bytes with 0 metadata bytes
        payload_6578 = SecretPayload(data=b"M" * 6578, media_type=MediaType.RAW_BYTES, metadata=None)
        serialized_6578 = SecretSerializer.serialize(payload_6578)
        assert len(serialized_6578) == 6578 + 18  # 6596 bytes
        ciphertext_6578 = crypto.encrypt(serialized_6578, b"0" * 32)
        assert len(ciphertext_6578) == 6596 + 28  # 6624 bytes (32 blocks of 207 bytes)
        encoded_6578 = encoder.encode(ciphertext_6578)
        assert len(encoded_6578) == 4 + 32 * 255  # 8164 bytes <= 8168 bytes
        # Hard gate check
        rep = CapacityCalculator.check_capacity(len(encoded_6578), subband_dim=256, rs_parity=48, version=2)
        assert rep.fits is True
        assert rep.usable_capacity_bytes == 6578

        # 2. 1 byte over maximum: 6,579 bytes
        payload_6579 = SecretPayload(data=b"M" * 6579, media_type=MediaType.RAW_BYTES, metadata=None)
        serialized_6579 = SecretSerializer.serialize(payload_6579)
        assert len(serialized_6579) == 6579 + 18  # 6597 bytes
        ciphertext_6579 = crypto.encrypt(serialized_6579, b"0" * 32)
        assert len(ciphertext_6579) == 6597 + 28  # 6625 bytes (requires 33rd shortened block)
        encoded_6579 = encoder.encode(ciphertext_6579)
        assert len(encoded_6579) == 4 + 32 * 255 + (1 + 48)  # 8213 bytes > 8168 bytes
        with pytest.raises(PayloadCapacityExceededError):
            CapacityCalculator.check_capacity(len(encoded_6579), subband_dim=256, rs_parity=48, version=2)

    def test_v3_maximum_resilience_capacity_boundary(self) -> None:
        """Verifies exact 6,066 byte maximum for v3 maximum-resilience (2t=64), and failure at 6,067."""
        crypto = AESGCMProvider()
        encoder_64 = ReedSolomonEncoder(parity_symbols=64)

        # 1. Exact maximum 6,066 bytes with 0 metadata bytes
        payload_6066 = SecretPayload(data=b"R" * 6066, media_type=MediaType.RAW_BYTES, metadata=None)
        serialized_6066 = SecretSerializer.serialize(payload_6066)
        assert len(serialized_6066) == 6066 + 18  # 6084 bytes
        ciphertext_6066 = crypto.encrypt(serialized_6066, b"0" * 32)
        assert len(ciphertext_6066) == 6084 + 28  # 6112 bytes (32 blocks of 191 bytes)
        encoded_6066 = encoder_64.encode(ciphertext_6066)
        assert len(encoded_6066) == 4 + 32 * 255  # 8164 bytes <= 8168 bytes
        # Hard gate check
        rep = CapacityCalculator.check_capacity(len(encoded_6066), subband_dim=256, rs_parity=64, version=2)
        assert rep.fits is True
        assert rep.usable_capacity_bytes == 6066

        # 2. 1 byte over maximum: 6,067 bytes
        payload_6067 = SecretPayload(data=b"R" * 6067, media_type=MediaType.RAW_BYTES, metadata=None)
        serialized_6067 = SecretSerializer.serialize(payload_6067)
        ciphertext_6067 = crypto.encrypt(serialized_6067, b"0" * 32)
        encoded_6067 = encoder_64.encode(ciphertext_6067)
        assert len(encoded_6067) == 4 + 32 * 255 + (1 + 64)  # 8229 bytes > 8168 bytes
        with pytest.raises(PayloadCapacityExceededError):
            CapacityCalculator.check_capacity(len(encoded_6067), subband_dim=256, rs_parity=64, version=2)

    def test_metadata_bearing_payload_accounting(self) -> None:
        """Verifies metadata bytes consume capacity and are accounted for accurately."""
        metadata = {"sender": "Alice", "classification": "TOP_SECRET"}
        meta_json_len = len(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        assert meta_json_len > 0

        # With 6,578 bytes data AND metadata, it must exceed capacity
        payload_with_meta = SecretPayload(data=b"A" * 6578, media_type=MediaType.TEXT, metadata=metadata)
        ser = SecretSerializer.serialize(payload_with_meta)
        assert len(ser) == 6578 + 18 + meta_json_len
        crypto = AESGCMProvider()
        c = crypto.encrypt(ser, b"0" * 32)
        enc = ReedSolomonEncoder(48).encode(c)
        assert len(enc) > 8168
        with pytest.raises(PayloadCapacityExceededError):
            CapacityCalculator.check_capacity(len(enc), subband_dim=256, rs_parity=48, version=2)


# =============================================================================
# Workstream C: Interleaver Mathematical & Implementation Audit Tests
# =============================================================================

class TestWorkstreamCInterleaverAudit:
    """Validates mathematical properties, coprimality, and bijectivity of the symbol interleaver."""

    @pytest.mark.parametrize(
        "n",
        [0, 1, 2, 70, 71, 72, 140, 141, 142, 213, 255, 256, 510, 1020, 1530, 8164],
    )
    def test_interleaver_edge_cases_and_bijectivity(self, n: int) -> None:
        """Verifies coprime derivation, bijectivity, and exact roundtrip across all requested edge-case lengths."""
        prefix = b"\x00\x00\x01\x00" if n > 0 else b""
        raw_blocks = bytes((i * 13) % 256 for i in range(n))
        stream = prefix + raw_blocks if n > 0 else b""

        # 1. Coprimality check
        if n > 1:
            step = DeterministicSymbolInterleaver.get_coprime_step(n, preferred_step=71)
            assert step >= 71
            assert math.gcd(step, n) == 1

            # 2. Explicit bijectivity check: pi must be a permutation of range(n)
            perm = [(step * i) % n for i in range(n)]
            assert len(perm) == n
            assert len(set(perm)) == n, f"Permutation collision detected for n={n}, step={step}!"

        # 3. Interleave and Deinterleave
        interleaved = DeterministicSymbolInterleaver.interleave(stream, scheme="coprime", coprime_step=71)
        assert len(interleaved) == len(stream)
        if len(stream) >= 4:
            assert interleaved[:4] == prefix  # Prefix preserved

        deinterleaved = DeterministicSymbolInterleaver.deinterleave(interleaved, scheme="coprime", coprime_step=71)
        assert deinterleaved == stream, f"Roundtrip mismatch for n={n}!"

    def test_interleaver_short_stream_handling(self) -> None:
        """Verifies boundary streams of length 0, 1, 2, 3, 4 bytes pass safely without index errors."""
        for length in (0, 1, 2, 3, 4):
            data = bytes(range(length))
            assert DeterministicSymbolInterleaver.interleave(data, scheme="coprime") == data
            assert DeterministicSymbolInterleaver.deinterleave(data, scheme="coprime") == data

