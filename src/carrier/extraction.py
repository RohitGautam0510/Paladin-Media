"""Steganographic Carrier Extractor for PaladinMedia.

Implements the receiver extraction procedure per Architecture §8.2/§19.2 and Wire Protocol v2/v3:
- Dual-Anchor Probing: Probes Row 32 (v3 relocated anchor) and Row 0 (v2/v1 boundary anchor).
- Dual-Valid Collision Detection: Fails closed with CarrierHeaderError if conflicting valid
  headers are detected at both Row 32 and Row 0.
- Corrects up to 4 byte errors in Wire Protocol v2/v3 headers via shortened RS(24, 16) decoding.
- Validates carrier framing, protocol version, and capacity hard-gate before allocation.
- Reconstructs the deterministic position mask using the header's wire version layout.
- Deinterleaves payload symbols when FLAG_SYMBOL_INTERLEAVED is asserted.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from src.carrier.capacity import CapacityCalculator
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
    def extract_bits_from_positions(
        cls, arr: np.ndarray, positions: Sequence[Tuple[int, int]], ch: int = 0
    ) -> List[int]:
        """Extracts demodulated parity bits from exact (row, col) subband coordinates."""
        bits: List[int] = []
        for r, c_idx in positions:
            a_val = int(arr[2 * r, 2 * c_idx, ch])
            b_val = int(arr[2 * r, 2 * c_idx + 1, ch])
            c_val = int(arr[2 * r + 1, 2 * c_idx, ch])
            d_val = int(arr[2 * r + 1, 2 * c_idx + 1, ch])

            two_hh = a_val - b_val - c_val + d_val
            bit = (two_hh % 2 + 2) % 2
            bits.append(bit)
        return bits

    @classmethod
    def extract(
        cls, stego_image: Union[np.ndarray, Image.Image], allow_v3: Optional[bool] = None
    ) -> Tuple[CarrierHeader, bytes]:
        """Extracts the carrier header and protected payload from a stego carrier image.

        Args:
            stego_image: Even-dimension RGB carrier (numpy array or PIL Image).
            allow_v3: If True, enables dual-anchor probing (Row 32 and Row 0).
                      If False, restricts probing exclusively to Row 0 (v2/v1).
                      If None, defaults to PALADIN_ENABLE_V3_MR environment setting (default False).

        Returns:
            Tuple of (CarrierHeader, protected_payload_bytes).

        Raises:
            CarrierHeaderError: If carrier header framing is invalid or ambiguous.
            PayloadCapacityExceededError: If payload length exceeds carrier capacity.
        """
        if allow_v3 is None:
            allow_v3 = os.environ.get("PALADIN_ENABLE_V3_MR", "false").lower() in ("true", "1", "yes")

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

        # -----------------------------------------------------------------
        # 1. Dual-Anchor Probing (Row 32 v3 primary, Row 0 v2/v1 fallback)
        # -----------------------------------------------------------------
        header_v3: Optional[CarrierHeader] = None
        if allow_v3 and subband_dim[0] > CarrierHeader.HEADER_ROW_OFFSET_V3:
            pos_v3 = DeterministicMaskGenerator.generate_header_positions(
                subband_dim=subband_dim, version=CarrierHeader.VERSION_3
            )
            raw_v3 = cls.bits_to_bytes(cls.extract_bits_from_positions(arr, pos_v3, ch))
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
        raw_v2 = cls.bits_to_bytes(cls.extract_bits_from_positions(arr, pos_v2, ch))
        try:
            cand_v2 = CarrierHeader.unpack(raw_v2, allow_v3=allow_v3)
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
            if not allow_v3 and last_v2_err is not None:
                raise last_v2_err
            detail = f" Details: {last_v2_err}" if last_v2_err is not None else ""
            if allow_v3:
                raise CarrierHeaderError(
                    f"Uncorrectable carrier header: both Row 32 (v3) and Row 0 (v2/v1) anchor probes failed.{detail}"
                )
            raise CarrierHeaderError(
                f"Uncorrectable carrier header at Row 0 anchor (Wire Protocol v2/v1).{detail}"
            )

        # -----------------------------------------------------------------
        # 2. Hard-gate capacity check before allocation or mask generation
        # -----------------------------------------------------------------
        CapacityCalculator.check_capacity(
            protected_payload_bytes=header.payload_length,
            subband_dim=subband_dim,
            rs_parity=header.ecc_parity,
            version=header.version,
        )

        # -----------------------------------------------------------------
        # 3. Reconstruct deterministic mask for payload bits
        # -----------------------------------------------------------------
        payload_length = header.payload_length
        if payload_length == 0:
            return header, b""

        payload_positions = DeterministicMaskGenerator.generate_payload_positions(
            session_counter=header.session_counter,
            payload_length_bytes=payload_length,
            subband_dim=subband_dim,
            version=header.version,
        )

        # -----------------------------------------------------------------
        # 4. Extract payload bits
        # -----------------------------------------------------------------
        payload_bits = cls.extract_bits_from_positions(arr, payload_positions, ch)
        protected_payload = cls.bits_to_bytes(payload_bits)

        # -----------------------------------------------------------------
        # 5. Conditional Deinterleaving
        # -----------------------------------------------------------------
        if header.symbol_interleaved:
            from src.carrier.interleaver import DeterministicSymbolInterleaver
            protected_payload = DeterministicSymbolInterleaver.deinterleave(
                protected_payload, scheme="coprime", session_counter=header.session_counter
            )

        return header, protected_payload
