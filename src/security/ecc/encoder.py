"""Error correction encoder implementation for PaladinMedia.

This module implements the abstract base class ECCEncoder and the concrete
ReedSolomonEncoder using Reed-Solomon block codes over GF(2^8) via `reedsolo`.
It provides systematic chunking, length-framed wire packaging, symbol-level
error correction, and fail-fast uncorrectable error detection.
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import reedsolo

try:
    from src.core.exceptions import ECCDecodeError, ECCError
except ImportError:
    from core.exceptions import ECCDecodeError, ECCError  # type: ignore[no-redef]

__all__ = [
    "ECCEncoder",
    "ReedSolomonEncoder",
    "ECCError",
    "ECCDecodeError",
    "ECCParameterError",
    "ECCResult",
]


class ECCParameterError(ECCError, ValueError):
    """Raised on invalid parameters or arguments for error correction operations."""
    pass


@dataclass(frozen=True)
class ECCResult:
    """Encapsulates decoded data along with error correction statistics.

    Attributes:
        data: The exact recovered payload bytes.
        corrected_symbols: Total count of symbol errors corrected across all blocks.
        blocks_count: Total number of RS blocks processed during decoding.
        prefix_recovered: True if the 4-byte length prefix was corrupted and recovered
                          via mathematical length derivation from valid stream framing.
    """
    data: bytes
    corrected_symbols: int = 0
    blocks_count: int = 0
    prefix_recovered: bool = False


class ECCEncoder(ABC):
    """Abstract Base Class for Error Correction Coding (ECC) encoders."""

    @abstractmethod
    def encode(self, data: bytes) -> bytes:
        """Encodes arbitrary input data with error correction redundancy.

        Args:
            data: Payload bytes to protect.

        Returns:
            Encoded bytes containing length framing and redundant ECC blocks.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, encoded_data: bytes) -> bytes:
        """Decodes error-correction encoded stream, repairing correctable errors.

        Args:
            encoded_data: Encoded byte stream potentially corrupted by transmission noise.

        Returns:
            Exact recovered original payload bytes.

        Raises:
            ECCDecodeError: If uncorrectable errors exceed correction capability or framing is invalid.
        """
        raise NotImplementedError


class ReedSolomonEncoder(ECCEncoder):
    """Reed-Solomon Error Correction Codec operating over GF(2^8).

    Implements systematic block chunking with 4-byte big-endian length prefix framing.
    Each block adds `ecc_symbols` ($2t$) parity bytes, allowing up to
    $t = \\lfloor \\text{ecc\\_symbols} / 2 \\rfloor$ symbol (byte) errors per block to be corrected.

    Attributes:
        ecc_symbols: Number of parity symbols ($2t$) per block (default 32).
        parity_symbols: Alias for ecc_symbols.
        block_size: Maximum total codeword length $n \\le 255$ in GF(2^8) (default 255).
        k: Maximum data chunk size per block ($k = n - 2t$, default 223).
        t: Maximum correctable symbol errors per block ($t = \\lfloor 2t / 2 \\rfloor$, default 16).
        max_correctable_symbols: Alias for t.
    """

    DEFAULT_ECC_SYMBOLS: int = 32
    DEFAULT_BLOCK_SIZE: int = 255

    def __init__(
        self,
        ecc_symbols: int = DEFAULT_ECC_SYMBOLS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        *,
        parity_symbols: Optional[int] = None,
    ) -> None:
        """Initializes the Reed-Solomon encoder with specified parameters.

        Args:
            ecc_symbols: Number of parity symbols per block (2t). Defaults to 32 (t=16).
            block_size: Maximum block length in symbols (bytes). Defaults to 255.
            parity_symbols: Optional alias for ecc_symbols. If provided, overrides ecc_symbols.

        Raises:
            ECCParameterError: If parameters are invalid (non-positive, ecc_symbols >= block_size,
                               or block_size > 255).
        """
        if parity_symbols is not None:
            ecc_symbols = parity_symbols

        if not isinstance(ecc_symbols, int) or ecc_symbols <= 0:
            raise ECCParameterError(
                f"ecc_symbols must be a positive integer, got {ecc_symbols!r}"
            )
        if not isinstance(block_size, int) or block_size <= 0:
            raise ECCParameterError(
                f"block_size must be a positive integer, got {block_size!r}"
            )
        if block_size > 255:
            raise ECCParameterError(
                f"block_size cannot exceed 255 for GF(2^8), got {block_size}"
            )
        if ecc_symbols >= block_size:
            raise ECCParameterError(
                f"ecc_symbols ({ecc_symbols}) must be strictly less than block_size ({block_size})"
            )

        self.ecc_symbols: int = ecc_symbols
        self.parity_symbols: int = ecc_symbols
        self.block_size: int = block_size
        self.k: int = block_size - ecc_symbols
        self.t: int = ecc_symbols // 2
        self.max_correctable_symbols: int = self.t

        # Initialize underlying reedsolo RSCodec
        # nsym = ecc_symbols, nsize = block_size, GF(2^8) primitive polynomial 0x11D (285)
        self._rsc = reedsolo.RSCodec(nsym=self.ecc_symbols, nsize=self.block_size)

    def encode(self, data: bytes) -> bytes:
        """Encodes arbitrary-length byte payload with chunked Reed-Solomon error correction.

        Wire Framing:
            `[4-byte Length L] + [Block 1] + [Block 2] + ...`
            Where each Block i is `rsc.encode(chunk_i)` of size `len(chunk_i) + ecc_symbols`.
            If data is empty ($L=0$), the body contains 0 blocks.

        Args:
            data: Arbitrary payload bytes to encode.

        Returns:
            4-byte length prefix concatenated with systematic RS-encoded blocks.

        Raises:
            TypeError: If input data is not bytes-like.
            ECCParameterError: If payload length exceeds 32-bit unsigned integer maximum.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}")

        raw = bytes(data)
        length = len(raw)

        if length > 0xFFFFFFFF:
            raise ECCParameterError("Payload length exceeds 32-bit unsigned integer maximum")

        header = struct.pack(">I", length)

        if length == 0:
            return header

        blocks: list[bytes] = []
        k = self.k

        for offset in range(0, length, k):
            chunk = raw[offset : offset + k]
            encoded_block = bytes(self._rsc.encode(chunk))
            blocks.append(encoded_block)

        return header + b"".join(blocks)

    @classmethod
    def derive_payload_length(
        cls,
        encoded_stream_len: int,
        k: int = 223,
        ecc_symbols: int = 32,
        block_size: int = 255,
    ) -> int:
        """Mathematically derives the unique decoded payload length from a valid encoded stream length.

        In systematic chunked RS(block_size, k) with ecc_symbols parity:
        Every full block has size block_size = k + ecc_symbols.
        Any shortened final block with R data bytes (1 <= R < k) has size R + ecc_symbols.
        Because 1 <= R < k, the final block size is strictly bounded by
        (ecc_symbols + 1) <= size < block_size.
        Consequently, M = encoded_stream_len - 4 uniquely determines:
        - full_blocks = M // block_size
        - remainder = M % block_size
        and is bijectively invertible to payload data length L.

        Args:
            encoded_stream_len: Total length of the RS-encoded stream including 4-byte prefix.
            k: Data symbols per full block (default 223).
            ecc_symbols: Parity symbols per block (default 32).
            block_size: Total symbols per full block (default 255).

        Returns:
            The exact decoded data length in bytes.

        Raises:
            ECCDecodeError: If encoded_stream_len cannot represent any valid set of RS codewords.
        """
        if encoded_stream_len < 4:
            raise ECCDecodeError(
                f"Encoded data length ({encoded_stream_len} bytes) is too short to contain 4-byte header"
            )
        M = encoded_stream_len - 4
        if M == 0:
            return 0
        rem = M % block_size
        full_blocks = M // block_size
        if rem == 0:
            return full_blocks * k
        elif rem > ecc_symbols:
            return full_blocks * k + (rem - ecc_symbols)
        else:
            raise ECCDecodeError(
                f"Invalid encoded stream length ({encoded_stream_len} bytes): final block of {rem} bytes "
                f"is too short to contain {ecc_symbols} parity symbols plus at least 1 payload byte"
            )

    def decode(self, encoded_data: bytes, strict_prefix: bool = False) -> bytes:
        """Decodes Reed-Solomon encoded data, correcting up to t symbol errors per block.

        Args:
            encoded_data: Length-prefixed RS-encoded bitstream.
            strict_prefix: If True, raises ECCDecodeError on length-prefix mismatch rather
                           than falling back to mathematical length derivation.

        Returns:
            Exact original payload bytes.

        Raises:
            TypeError: If encoded_data is not bytes-like.
            ECCDecodeError: If uncorrectable errors exceed correction capacity t,
                            or wire framing/length is corrupted.
        """
        return self.decode_with_result(encoded_data, strict_prefix=strict_prefix).data

    def decode_with_result(self, encoded_data: bytes, strict_prefix: bool = False) -> ECCResult:
        """Decodes Reed-Solomon encoded data and returns detailed result with error statistics.

        Sprint 3.3.3: Supports backward-compatible mathematical length derivation fallback.
        When the 4-byte prefix suffers bit corruption in transmission but the stream length
        is structurally valid, decoding succeeds using the derived length.

        Args:
            encoded_data: Length-prefixed RS-encoded bitstream.
            strict_prefix: If True, strictly enforces that raw_enc[:4] matches the derived length.
                           Defaults to False (safe prefix-recovery enabled).

        Returns:
            ECCResult containing recovered payload bytes, total corrected symbol count,
            total blocks count, and prefix_recovered flag.

        Raises:
            TypeError: If encoded_data is not bytes-like.
            ECCDecodeError: If uncorrectable errors exceed correction capacity t,
                            or wire framing/length is corrupted.
        """
        if not isinstance(encoded_data, (bytes, bytearray, memoryview)):
            raise TypeError(f"encoded_data must be bytes-like, got {type(encoded_data).__name__}")

        raw_enc = bytes(encoded_data)
        if len(raw_enc) < 4:
            raise ECCDecodeError(
                f"Encoded data length ({len(raw_enc)} bytes) is too short to contain 4-byte header"
            )

        (declared_length,) = struct.unpack(">I", raw_enc[:4])
        derived_length = self.derive_payload_length(
            len(raw_enc), k=self.k, ecc_symbols=self.ecc_symbols, block_size=self.block_size
        )

        if declared_length == derived_length:
            total_length = declared_length
            prefix_recovered = False
        else:
            if strict_prefix:
                raise ECCDecodeError(
                    f"Encoded data length mismatch: declared length {declared_length} requires "
                    f"different stream length than actual {len(raw_enc)} bytes"
                )
            total_length = derived_length
            prefix_recovered = True

        # Handle 0-byte payload edge case
        if total_length == 0:
            if len(raw_enc) == 4:
                return ECCResult(data=b"", corrected_symbols=0, blocks_count=0, prefix_recovered=prefix_recovered)
            if len(raw_enc) == 4 + self.ecc_symbols:
                # Tolerates an encoded empty block with parity
                try:
                    res = self._rsc.decode(raw_enc[4:])
                    corrected = len(res[2]) if len(res) > 2 and res[2] is not None else 0
                    return ECCResult(data=bytes(res[0]), corrected_symbols=corrected, blocks_count=1, prefix_recovered=prefix_recovered)
                except reedsolo.ReedSolomonError as exc:
                    raise ECCDecodeError(
                        "Reed-Solomon decoding failed: uncorrectable symbol errors exceed capacity"
                    ) from exc
                except Exception as exc:
                    raise ECCDecodeError(f"Reed-Solomon decoding failed: {exc}") from exc
            raise ECCDecodeError(
                f"Invalid encoded data length ({len(raw_enc)} bytes) for 0-byte payload"
            )

        k = self.k
        num_full_blocks = total_length // k
        remainder = total_length % k
        total_blocks = num_full_blocks + (1 if remainder > 0 else 0)

        offset = 4
        decoded_chunks: list[bytes] = []
        total_corrected = 0

        # Decode full blocks
        full_block_len = k + self.ecc_symbols
        for block_idx in range(num_full_blocks):
            block = raw_enc[offset : offset + full_block_len]
            offset += full_block_len
            try:
                res = self._rsc.decode(block)
                chunk = bytes(res[0])
                corrected = len(res[2]) if len(res) > 2 and res[2] is not None else 0
                total_corrected += corrected
            except reedsolo.ReedSolomonError as exc:
                raise ECCDecodeError(
                    f"Reed-Solomon decoding failed in block {block_idx + 1}/{total_blocks}: "
                    "uncorrectable symbol errors exceed capacity"
                ) from exc
            except Exception as exc:
                raise ECCDecodeError(
                    f"Reed-Solomon decoding failed in block {block_idx + 1}/{total_blocks}: {exc}"
                ) from exc

            if len(chunk) != k:
                raise ECCDecodeError(
                    f"Decoded block {block_idx + 1} data length mismatch: expected {k}, got {len(chunk)}"
                )
            decoded_chunks.append(chunk)

        # Decode last shortened block if remainder > 0
        if remainder > 0:
            last_block_len = remainder + self.ecc_symbols
            block = raw_enc[offset : offset + last_block_len]
            offset += last_block_len
            try:
                res = self._rsc.decode(block)
                chunk = bytes(res[0])
                corrected = len(res[2]) if len(res) > 2 and res[2] is not None else 0
                total_corrected += corrected
            except reedsolo.ReedSolomonError as exc:
                raise ECCDecodeError(
                    f"Reed-Solomon decoding failed in final block {total_blocks}/{total_blocks}: "
                    "uncorrectable symbol errors exceed capacity"
                ) from exc
            except Exception as exc:
                raise ECCDecodeError(
                    f"Reed-Solomon decoding failed in final block {total_blocks}/{total_blocks}: {exc}"
                ) from exc

            if len(chunk) != remainder:
                raise ECCDecodeError(
                    f"Decoded final block data length mismatch: expected {remainder}, got {len(chunk)}"
                )
            decoded_chunks.append(chunk)

        reassembled = b"".join(decoded_chunks)
        if len(reassembled) != total_length:
            raise ECCDecodeError(
                f"Reassembled payload length {len(reassembled)} does not match header length {total_length}"
            )

        return ECCResult(
            data=reassembled,
            corrected_symbols=total_corrected,
            blocks_count=total_blocks,
            prefix_recovered=prefix_recovered,
        )

    def encode_block(self, chunk: bytes) -> bytes:
        """Encodes a single chunk of data (up to k bytes) with RS parity without framing.

        Args:
            chunk: Input data chunk of at most k bytes.

        Returns:
            Raw RS codeword of length len(chunk) + ecc_symbols.

        Raises:
            TypeError: If chunk is not bytes-like.
            ECCParameterError: If chunk length exceeds k.
        """
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError(f"chunk must be bytes-like, got {type(chunk).__name__}")
        chunk_bytes = bytes(chunk)
        if len(chunk_bytes) > self.k:
            raise ECCParameterError(
                f"chunk size ({len(chunk_bytes)} bytes) exceeds maximum block data capacity k ({self.k})"
            )
        if len(chunk_bytes) == 0:
            return b""
        return bytes(self._rsc.encode(chunk_bytes))

    def decode_block(
        self,
        block: bytes,
        erase_pos: Optional[Sequence[int]] = None,
    ) -> bytes:
        """Decodes a single RS block directly without framing, correcting symbol errors.

        Args:
            block: Raw RS codeword of length len(chunk) + ecc_symbols.
            erase_pos: Optional sequence of known erasure positions.

        Returns:
            Exact repaired chunk bytes.

        Raises:
            TypeError: If block is not bytes-like.
            ECCDecodeError: If uncorrectable errors exceed capacity or block size is invalid.
        """
        if not isinstance(block, (bytes, bytearray, memoryview)):
            raise TypeError(f"block must be bytes-like, got {type(block).__name__}")
        block_bytes = bytes(block)
        if len(block_bytes) == 0:
            return b""
        if len(block_bytes) < self.ecc_symbols:
            raise ECCDecodeError(
                f"Block size ({len(block_bytes)} bytes) is smaller than parity symbols ({self.ecc_symbols})"
            )
        if len(block_bytes) > self.block_size:
            raise ECCDecodeError(
                f"Block size ({len(block_bytes)} bytes) exceeds maximum block size ({self.block_size})"
            )
        try:
            if erase_pos is not None:
                res = self._rsc.decode(block_bytes, erase_pos=list(erase_pos))
            else:
                res = self._rsc.decode(block_bytes)
            return bytes(res[0])
        except reedsolo.ReedSolomonError as exc:
            raise ECCDecodeError(
                "Reed-Solomon decoding failed: uncorrectable symbol errors exceed capacity"
            ) from exc
        except Exception as exc:
            raise ECCDecodeError(f"Reed-Solomon decoding failed: {exc}") from exc
