"""Baseline Transform-Domain Embedding Engine for PaladinMedia Phase 2.

Implements the MVP 0 baseline embedding mechanism per Architecture §10/§12 and
Phase-2 instructions:
- Target: Red channel (channel 0) of the HH subband.
- Bit representation: Exact integer-parity modulation on the 2D Haar 2*HH coefficient:
    2*HH = a - b - c + d  (where a, b, c, d are the 2x2 spatial carrier pixels)
- Bit embedding rule:
    Enforces (2*HH mod 2) == bit by adjusting carrier pixel 'a' by +/-1.
- Guarantees 0 clipping, minimal distortion (PSNR > 57 dB), and exact bit preservation
  through 8-bit RGB PNG serialization and reload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from PIL import Image

from src.carrier.capacity import CapacityCalculator, CapacityReport
from src.carrier.dwt import DWTSubbands, HaarDWT2D, subband_to_image
from src.carrier.header import CarrierHeader
from src.carrier.mask import DeterministicMaskGenerator, EmbeddingMask

__all__ = [
    "BaselineEmbedder",
    "EmbeddingResult",
]


@dataclass(frozen=True)
class EmbeddingResult:
    """Contains artifacts resulting from carrier embedding.

    Attributes:
        stego_array: Stego RGB image array (512, 512, 3) uint8.
        stego_image: Stego PIL Image in RGB format.
        original_subbands: DWTSubbands before embedding.
        modified_subbands: DWTSubbands after embedding.
        modified_hh_image: Normalized visualization PIL Image of modified HH'.
        capacity_report: CapacityReport detailing payload sizes and capacity.
        header: CarrierHeader packed and embedded into anchor positions.
        mask: EmbeddingMask used for placement.
    """
    stego_array: np.ndarray
    stego_image: Image.Image
    original_subbands: DWTSubbands
    modified_subbands: DWTSubbands
    modified_hh_image: Image.Image
    capacity_report: CapacityReport
    header: CarrierHeader
    mask: EmbeddingMask


class BaselineEmbedder:
    """Embeds header and protected payload into carrier transform coefficients."""

    EMBEDDING_CHANNEL: int = 0  # Red channel of HH

    @classmethod
    def bytes_to_bits(cls, data: bytes) -> np.ndarray:
        """Converts bytes to a 1D numpy array of bits (uint8 0 or 1), MSB-first."""
        if not data:
            return np.empty(0, dtype=np.uint8)
        byte_arr = np.frombuffer(data, dtype=np.uint8)
        # Unpack bits: 8 bits per byte, MSB-first
        bits = np.unpackbits(byte_arr)
        return bits

    @classmethod
    def embed(
        cls,
        cover_image: Union[np.ndarray, Image.Image],
        protected_payload: bytes,
        session_counter: int = 1,
    ) -> EmbeddingResult:
        """Executes the complete baseline embedding workflow.

        Sequence:
            1. Capacity check: Verify protected_payload <= 8,176 bytes (hard gate).
            2. Header packaging: Create and pack 16-byte carrier header.
            3. DWT computation: Forward 2D 1-level Haar DWT on 512x512 cover image.
            4. Deterministic mask: Derive fixed header anchors + payload coordinates.
            5. Transform embedding: Modulate parity of 2*HH on channel 0.
            6. IDWT & Stego generation: Synthesize 8-bit RGB stego image.

        Args:
            cover_image: 512x512 RGB carrier (numpy array or PIL Image).
            protected_payload: Reed-Solomon protected byte stream.
            session_counter: 32-bit session/file counter for context and mask seed.

        Returns:
            EmbeddingResult with stego image, subband arrays, and visual artifacts.

        Raises:
            PayloadCapacityExceededError: If protected payload exceeds usable capacity.
        """
        # Obtain underlying spatial carrier array uint8
        if isinstance(cover_image, Image.Image):
            arr = np.array(cover_image.convert("RGB"), dtype=np.uint8)
        else:
            arr = np.array(cover_image, dtype=np.uint8)

        if len(arr.shape) != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB carrier, got shape {arr.shape}")
        if arr.shape[0] % 2 != 0 or arr.shape[1] % 2 != 0:
            raise ValueError(f"Carrier dimensions must be even, got {arr.shape[:2]}")
        if arr.shape[0] < 32 or arr.shape[1] < 32:
            raise ValueError(f"Carrier dimensions too small: must be at least 32x32, got {arr.shape[:2]}")

        subband_dim = (arr.shape[0] // 2, arr.shape[1] // 2)

        # Step 1: Hard-gate capacity check
        capacity_report = CapacityCalculator.check_capacity(
            len(protected_payload), subband_dim=subband_dim
        )

        # Step 2: Construct 16-byte carrier header
        header = CarrierHeader(
            session_counter=session_counter,
            payload_length=len(protected_payload),
        )
        header_bytes = header.pack()

        # Step 3: Compute Haar DWT
        original_subbands = HaarDWT2D.forward(arr)

        # Step 4: Generate deterministic mask
        mask = DeterministicMaskGenerator.generate_mask(
            session_counter=session_counter,
            payload_length_bytes=len(protected_payload),
            subband_dim=subband_dim,
        )

        # Prepare bit streams
        header_bits = cls.bytes_to_bits(header_bytes)
        payload_bits = cls.bytes_to_bits(protected_payload)

        stego_arr = arr.copy()
        ch = cls.EMBEDDING_CHANNEL

        # a: [0::2, 0::2], b: [0::2, 1::2], c: [1::2, 0::2], d: [1::2, 1::2]
        # Modulate Header bits at header anchor positions
        for bit, (r, c_idx) in zip(header_bits, mask.header_positions):
            a_val = int(stego_arr[2 * r, 2 * c_idx, ch])
            b_val = int(stego_arr[2 * r, 2 * c_idx + 1, ch])
            c_val = int(stego_arr[2 * r + 1, 2 * c_idx, ch])
            d_val = int(stego_arr[2 * r + 1, 2 * c_idx + 1, ch])

            two_hh = a_val - b_val - c_val + d_val
            if ((two_hh % 2 + 2) % 2) != int(bit):
                # Adjust pixel 'a' by +1 (or -1 if saturated)
                new_a = a_val + 1 if a_val < 255 else a_val - 1
                stego_arr[2 * r, 2 * c_idx, ch] = new_a

        # Modulate Payload bits at deterministic payload positions
        for bit, (r, c_idx) in zip(payload_bits, mask.payload_positions):
            a_val = int(stego_arr[2 * r, 2 * c_idx, ch])
            b_val = int(stego_arr[2 * r, 2 * c_idx + 1, ch])
            c_val = int(stego_arr[2 * r + 1, 2 * c_idx, ch])
            d_val = int(stego_arr[2 * r + 1, 2 * c_idx + 1, ch])

            two_hh = a_val - b_val - c_val + d_val
            if ((two_hh % 2 + 2) % 2) != int(bit):
                new_a = a_val + 1 if a_val < 255 else a_val - 1
                stego_arr[2 * r, 2 * c_idx, ch] = new_a

        # Compute modified DWT subbands from stego carrier
        modified_subbands = HaarDWT2D.forward(stego_arr)
        modified_hh_img = subband_to_image(modified_subbands.hh, is_ll=False)
        stego_pil = Image.fromarray(stego_arr, mode="RGB")

        return EmbeddingResult(
            stego_array=stego_arr,
            stego_image=stego_pil,
            original_subbands=original_subbands,
            modified_subbands=modified_subbands,
            modified_hh_image=modified_hh_img,
            capacity_report=capacity_report,
            header=header,
            mask=mask,
        )
