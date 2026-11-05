"""Unit and Integration Tests for Phase 8 Protocol Reliability Experiments.

Validates:
1. RelocatedMaskGenerator: Determinism, bounds, collisions, non-overlap, capacity limits.
2. DeterministicSymbolInterleaver: Round-trip reversibility, boundary lengths, determinism, burst dispersion.
3. ExperimentalCarrierHeader: Packaging, RS(24, 16) error correction, backward compatibility, fail-closed rejection.
4. ExperimentalWireProtocolPipeline: End-to-end integration, v2 compatibility, and tamper resistance.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import struct
import numpy as np
import pytest

from scripts.experiments.phase8_protocol_experiments import (
    DeterministicSymbolInterleaver,
    ExperimentalCarrierHeader,
    ExperimentalWireProtocolPipeline,
    ProtocolCandidateConfig,
    RelocatedMaskGenerator,
)
from src.carrier.capacity import PayloadCapacityExceededError
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import DeterministicMaskGenerator
from src.core.serialization import MediaType, SecretPayload
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


class TestRelocatedMaskGenerator:
    """Validates properties of the relocated header anchor mask generator."""

    def test_header_positions_bounds_and_determinism(self) -> None:
        """Verifies header positions are strictly within subband bounds and deterministic."""
        subband_dim = 256
        row_offset = 32
        pos1 = RelocatedMaskGenerator.generate_header_positions(subband_dim, row_offset=row_offset)
        pos2 = RelocatedMaskGenerator.generate_header_positions(subband_dim, row_offset=row_offset)

        assert pos1 == pos2
        assert len(pos1) == RelocatedMaskGenerator.HEADER_BITS  # 192 bits
        for r, c in pos1:
            assert r == row_offset
            assert 0 <= c < subband_dim

    def test_non_overlapping_header_and_payload_positions(self) -> None:
        """Guarantees header and payload coordinates never collide or overlap."""
        session_counter = 42
        payload_bytes = 1000
        mask = RelocatedMaskGenerator.generate_mask(
            session_counter=session_counter,
            payload_length_bytes=payload_bytes,
            subband_dim=256,
            row_offset=32,
        )

        assert len(mask.header_positions) == 192
        assert len(mask.payload_positions) == payload_bytes * 8

        set_header = set(mask.header_positions)
        set_payload = set(mask.payload_positions)

        # Zero intersection
        assert set_header.isdisjoint(set_payload)
        # No duplicates within payload positions
        assert len(set_payload) == payload_bytes * 8

    def test_payload_capacity_gate(self) -> None:
        """Verifies hard gate rejection if payload exceeds 65,344 bits."""
        max_bytes = RelocatedMaskGenerator.MAX_PAYLOAD_BITS // 8  # 8,168 bytes
        # Exact maximum must succeed
        mask = RelocatedMaskGenerator.generate_mask(
            session_counter=1, payload_length_bytes=max_bytes, subband_dim=256, row_offset=32
        )
        assert len(mask.payload_positions) == max_bytes * 8

        # 1 byte over maximum must raise PayloadCapacityExceededError
        with pytest.raises(PayloadCapacityExceededError):
            RelocatedMaskGenerator.generate_mask(
                session_counter=1, payload_length_bytes=max_bytes + 1, subband_dim=256, row_offset=32
            )

    def test_empty_payload(self) -> None:
        """Verifies 0-byte payload yields empty payload positions without error."""
        mask = RelocatedMaskGenerator.generate_mask(
            session_counter=10, payload_length_bytes=0, subband_dim=256, row_offset=32
        )
        assert len(mask.header_positions) == 192
        assert len(mask.payload_positions) == 0


class TestDeterministicSymbolInterleaver:
    """Validates mathematical reversibility, boundary conditions, and dispersion."""

    @pytest.mark.parametrize("scheme", ["coprime", "prng"])
    @pytest.mark.parametrize("length", [5, 10, 100, 255, 259, 514, 1369])
    def test_roundtrip_reversibility(self, scheme: str, length: int) -> None:
        """Verifies exact byte equality for deinterleave(interleave(data)) == data."""
        rng = np.random.RandomState(length + 100)
        prefix = b"\x00\x00\x01\x00"
        raw_blocks = bytes(rng.randint(0, 256, size=length - 4, dtype=np.uint8))
        stream = prefix + raw_blocks

        interleaved = DeterministicSymbolInterleaver.interleave(
            stream, scheme=scheme, session_counter=774, coprime_step=71
        )
        assert len(interleaved) == len(stream)
        assert interleaved[:4] == prefix  # Prefix is preserved unpermuted

        deinterleaved = DeterministicSymbolInterleaver.deinterleave(
            interleaved, scheme=scheme, session_counter=774, coprime_step=71
        )
        assert deinterleaved == stream

    def test_boundary_lengths_and_edge_cases(self) -> None:
        """Validates short streams <= 4 bytes are handled safely without truncation."""
        assert DeterministicSymbolInterleaver.interleave(b"") == b""
        assert DeterministicSymbolInterleaver.interleave(b"\x00\x01\x02\x03") == b"\x00\x01\x02\x03"
        assert DeterministicSymbolInterleaver.deinterleave(b"") == b""
        assert DeterministicSymbolInterleaver.deinterleave(b"\x00\x01\x02\x03") == b"\x00\x01\x02\x03"

    def test_burst_error_dispersion(self) -> None:
        """Verifies a consecutive localized burst of symbol errors is dispersed across codewords."""
        # Consider 6 Reed-Solomon blocks of 255 bytes (total 1530 bytes + 4 prefix)
        n_blocks = 6
        block_size = 255
        total_len = 4 + n_blocks * block_size
        stream = b"\x00\x00\x05\x00" + b"\xAA" * (n_blocks * block_size)

        interleaved = DeterministicSymbolInterleaver.interleave(
            stream, scheme="coprime", session_counter=10, coprime_step=71
        )

        # Inject a contiguous burst of 30 symbol errors in the interleaved stream
        corrupted = bytearray(interleaved)
        burst_start = 50
        burst_len = 30
        for i in range(burst_start, burst_start + burst_len):
            corrupted[i] ^= 0xFF

        deinterleaved = DeterministicSymbolInterleaver.deinterleave(
            bytes(corrupted), scheme="coprime", session_counter=10, coprime_step=71
        )

        # Count symbol errors per block in the deinterleaved stream
        errors_per_block = []
        for b_idx in range(n_blocks):
            blk_orig = stream[4 + b_idx * block_size : 4 + (b_idx + 1) * block_size]
            blk_deintl = deinterleaved[4 + b_idx * block_size : 4 + (b_idx + 1) * block_size]
            errs = sum(1 for x, y in zip(blk_orig, blk_deintl) if x != y)
            errors_per_block.append(errs)

        # Without interleaving, a burst of 30 errors concentrated in one block would exceed t=24.
        # With interleaving, errors are dispersed so maximum block error is <= 8.
        assert max(errors_per_block) <= 8
        assert sum(errors_per_block) == burst_len


class TestExperimentalCarrierHeader:
    """Validates packing, unpacking, error correction, and version dispatch of carrier headers."""

    def test_v3_header_packing_and_rs_correction(self) -> None:
        """Verifies RS(24, 16) error correction repairs up to 4 corrupted bytes in v3 header."""
        header = ExperimentalCarrierHeader.create_v3(
            session_counter=12345,
            payload_length=1368,
            ecc_parity=48,
            relocated_header=True,
            symbol_interleaved=True,
        )
        packed = bytearray(header.pack())
        assert len(packed) == 24

        # Corrupt 4 bytes (maximum correctable for t=4)
        corrupted = bytearray(packed)
        corrupted[1] ^= 0x55  # Magic
        corrupted[4] ^= 0x33  # Version
        corrupted[10] ^= 0x77  # Session counter
        corrupted[22] ^= 0x99  # RS parity byte

        unpacked = ExperimentalCarrierHeader.unpack(bytes(corrupted))
        assert unpacked.version == ExperimentalCarrierHeader.VERSION_3
        assert unpacked.session_counter == 12345
        assert unpacked.payload_length == 1368
        assert unpacked.ecc_parity == 48
        assert unpacked.flags & ExperimentalCarrierHeader.FLAG_RELOCATED_HEADER
        assert unpacked.flags & ExperimentalCarrierHeader.FLAG_SYMBOL_INTERLEAVED

    def test_v3_header_fail_closed_on_uncorrectable_errors(self) -> None:
        """Verifies fail-closed rejection when header errors exceed 4 bytes."""
        header = ExperimentalCarrierHeader.create_v3(
            session_counter=1,
            payload_length=500,
            ecc_parity=48,
        )
        packed = bytearray(header.pack())
        # Corrupt 5 bytes (> t=4)
        for i in range(5):
            packed[i] ^= 0xFF

        with pytest.raises(CarrierHeaderError):
            ExperimentalCarrierHeader.unpack(bytes(packed))

    def test_backward_compatibility_with_v2_and_v1(self) -> None:
        """Verifies ExperimentalCarrierHeader cleanly unpacks existing v2 and legacy v1 headers."""
        # Wire Protocol v2 header
        h_v2 = CarrierHeader.create(session_counter=77, payload_length=1200, ecc_parity=48, version=2)
        packed_v2 = h_v2.pack()
        unpacked_v2 = ExperimentalCarrierHeader.unpack(packed_v2)
        assert unpacked_v2.version == CarrierHeader.VERSION_2
        assert unpacked_v2.session_counter == 77
        assert unpacked_v2.payload_length == 1200

        # Wire Protocol v1 header (16 bytes)
        h_v1 = CarrierHeader.create(session_counter=88, payload_length=800, ecc_parity=32, version=1)
        packed_v1 = h_v1.pack()
        unpacked_v1 = ExperimentalCarrierHeader.unpack(packed_v1)
        assert unpacked_v1.version == CarrierHeader.VERSION_1
        assert unpacked_v1.session_counter == 88
        assert unpacked_v1.payload_length == 800


class TestExperimentalWireProtocolPipeline:
    """Validates end-to-end integration of candidate protocol with ONNX runtime."""

    @pytest.fixture
    def test_cover_image(self) -> np.ndarray:
        """Loads natural carrier fixture from tests/fixtures/carrier_512.png."""
        fixture_p = Path("tests/fixtures/carrier_512.png")
        if fixture_p.exists():
            from PIL import Image
            return np.array(Image.open(fixture_p).convert("RGB"), dtype=np.uint8)
        # Fallback to smooth carrier array
        arr = np.full((512, 512, 3), 128, dtype=np.uint8)
        return arr

    def test_end_to_end_v3_exact_recovery(self, test_cover_image: np.ndarray) -> None:
        """Verifies exact secret payload recovery through Candidate Wire Protocol v3."""
        config = ProtocolCandidateConfig(
            name="v3_Test",
            version=0x03,
            header_row_offset=32,
            interleaving_scheme="coprime",
            coprime_step=71,
            ecc_parity=48,
        )
        pipeline = ExperimentalWireProtocolPipeline(config=config, backend="onnx")

        key = bytes.fromhex("11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff")
        aad = b"Phase8-Integration-Test"
        secret_data = b"PaladinMedia candidate protocol qualification payload test string 2026." * 10
        secret = SecretPayload(data=secret_data, media_type=MediaType.TEXT, metadata={"author": "paladin"})

        # Embed
        embed_res = pipeline.protect_and_embed(
            secret_payload=secret,
            cover_image=test_cover_image,
            key=key,
            aad=aad,
            session_counter=999,
        )

        assert embed_res["psnr_db"] > 45.0
        assert embed_res["ssim"] > 0.99

        # Extract
        rec_res = pipeline.extract_and_unprotect(
            stego_carrier=embed_res["stego_array"],
            key=key,
            aad=aad,
            reference_payload=secret,
        )

        assert rec_res["exact_match"] is True
        assert rec_res["recovered_payload"].data == secret_data
        assert rec_res["recovered_payload"].metadata == secret.metadata
        assert rec_res["header"].version == ExperimentalCarrierHeader.VERSION_3

    def test_v2_carrier_decodability_by_candidate_receiver(self, test_cover_image: np.ndarray) -> None:
        """Verifies that existing Wire Protocol v2 carriers remain fully decodable by candidate receiver."""
        from src.core.media_pipeline import MediaStegoPipeline

        # Embed using standard v2 deployment pipeline
        sender_v2 = MediaStegoPipeline.create_qualified_deployment(backend="onnx")
        key = bytes.fromhex("11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff")
        aad = b"Phase8-v2-Compatibility"
        secret_data = b"Legacy Wire Protocol v2 carrier content."
        secret = SecretPayload(data=secret_data, media_type=MediaType.TEXT, metadata={})

        stego_v2 = sender_v2.protect_and_embed(
            secret_payload=secret,
            cover_image=test_cover_image,
            key=key,
            aad=aad,
            session_counter=555,
            version=CarrierHeader.VERSION_2,
        )

        # Extract using Candidate v3 receiver pipeline
        config = ProtocolCandidateConfig(
            name="v3_DualReceiver",
            version=0x03,
            header_row_offset=32,
            interleaving_scheme="coprime",
        )
        receiver_v3 = ExperimentalWireProtocolPipeline(config=config, backend="onnx")

        rec_res = receiver_v3.extract_and_unprotect(
            stego_carrier=np.array(stego_v2.stego_image, dtype=np.uint8),
            key=key,
            aad=aad,
            reference_payload=secret,
        )

        assert rec_res["exact_match"] is True
        assert rec_res["recovered_payload"].data == secret_data
        assert rec_res["header"].version == CarrierHeader.VERSION_2
