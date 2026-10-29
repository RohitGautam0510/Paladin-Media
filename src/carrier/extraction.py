"""Steganographic Carrier Extractor for PaladinMedia Phase 2.

Implements the receiver extraction procedure per Architecture §8.2/§19.2:
- Reads the 16-byte fixed carrier header from fixed anchor positions in HH.
- Validates carrier framing and extracts session counter + payload length.
- Reconstructs the deterministic position mask.
- Extracts bits and packs them into the Reed-Solomon protected byte stream.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
from PIL import Image

from src.carrier.dwt import HaarDWT2D
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import DeterministicMaskGenerator

__all__ = [
    "Extractor",
]


class Extractor:
    """Extracts carrier header and protected payload from stego carrier."""

    EMBEDDING_CHANNEL: int = 0  # Red channel of HH

    @classmethod
    def bits_to_bytes(cls, bits: Sequence[int]) -> bytes:
        """Packs a sequence of bits (0 or 1, length multiple of 8) into bytes MSB-first."""
        if len(bits) % 8 != 0:
            raise ValueError(f"Bit length must be a multiple of 8, got {len(bits)}")
        bit_arr = np.array(bits, dtype=np.uint8)
        packed = np.packbits(bit_arr)
        return packed.tobytes()

    @classmethod
    def extract(
        cls, stego_image: Union[np.ndarray, Image.Image]
    ) -> Tuple[CarrierHeader, bytes]:
        """Extracts the carrier header and protected payload from a stego carrier image.

        Args:
            stego_image: Square even-dimension RGB carrier (numpy array or PIL Image).

        Returns:
            Tuple of (CarrierHeader, protected_payload_bytes).

        Raises:
            CarrierHeaderError: If carrier header magic or framing is invalid.
        """
        if isinstance(stego_image, Image.Image):
            arr = np.array(stego_image.convert("RGB"), dtype=np.uint8)
        else:
            arr = np.array(stego_image, dtype=np.uint8)

        if len(arr.shape) != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB carrier, got shape {arr.shape}")
        if arr.shape[0] % 2 != 0 or arr.shape[1] % 2 != 0:
            raise ValueError(f"Carrier dimensions must be even, got {arr.shape[:2]}")
        if arr.shape[0] < 32 or arr.shape[1] < 32:
            raise ValueError(f"Carrier dimensions too small: must be at least 32x32, got {arr.shape[:2]}")

        subband_dim = (arr.shape[0] // 2, arr.shape[1] // 2)
        ch = cls.EMBEDDING_CHANNEL

        # 1. Extract 128 header bits from fixed anchor positions
        header_positions = DeterministicMaskGenerator.generate_header_positions(
            subband_dim=subband_dim
        )
        header_bits = []
        for r, c_idx in header_positions:
            a_val = int(arr[2 * r, 2 * c_idx, ch])
            b_val = int(arr[2 * r, 2 * c_idx + 1, ch])
            c_val = int(arr[2 * r + 1, 2 * c_idx, ch])
            d_val = int(arr[2 * r + 1, 2 * c_idx + 1, ch])

            two_hh = a_val - b_val - c_val + d_val
            bit = (two_hh % 2 + 2) % 2
            header_bits.append(bit)

        header_bytes = cls.bits_to_bytes(header_bits)
        header = CarrierHeader.unpack(header_bytes)

        # 2. Reconstruct deterministic mask for payload bits
        payload_length = header.payload_length
        if payload_length == 0:
            return header, b""

        payload_positions = DeterministicMaskGenerator.generate_payload_positions(
            session_counter=header.session_counter,
            payload_length_bytes=payload_length,
            subband_dim=subband_dim,
        )

        # 3. Extract payload bits
        payload_bits = []
        for r, c_idx in payload_positions:
            a_val = int(arr[2 * r, 2 * c_idx, ch])
            b_val = int(arr[2 * r, 2 * c_idx + 1, ch])
            c_val = int(arr[2 * r + 1, 2 * c_idx, ch])
            d_val = int(arr[2 * r + 1, 2 * c_idx + 1, ch])

            two_hh = a_val - b_val - c_val + d_val
            bit = (two_hh % 2 + 2) % 2
            payload_bits.append(bit)

        protected_payload = cls.bits_to_bytes(payload_bits)

        return header, protected_payload
