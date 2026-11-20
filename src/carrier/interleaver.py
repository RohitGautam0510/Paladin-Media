"""Deterministic Symbol Interleaver for PaladinMedia Reed-Solomon Codewords.

Implements byte-level symbol interleaving per Architecture §8.4 / Wire Protocol v3:
- Disperses localized spatial error bursts across independent Reed-Solomon blocks
  so that no single block exceeds the correction threshold (t=32 for v3-MR).
- Preserves the 4-byte big-endian length prefix at indices 0..3 unpermuted.
- Strictly operates between payload Reed-Solomon encoding and transform-domain modulation.
- Fully reversible with O(1) space overhead via modular arithmetic inversion.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from typing import ClassVar


class DeterministicSymbolInterleaver:
    """Deterministic, fully reversible symbol (byte) interleaver for Reed-Solomon codewords.

    Applied strictly between payload RS encoding and transform modulation.
    Disperses localized spatial error bursts across independent Reed-Solomon blocks.

    Supported Permutation Schemes:
    - 'coprime': Algebraic linear congruential stride: pi(i) = (a * i) mod N where gcd(a, N) = 1.
      Inverted in O(1) space via modular inverse: pi^-1(j) = (a^-1 * j) mod N.
    - 'prng': Domain-separated SHA-256 seeded pseudo-random permutation.
    - 'none': Identity passthrough (no interleaving).
    """

    DEFAULT_COPRIME_STEP: ClassVar[int] = 71

    def __init__(
        self,
        session_counter: int = 1,
        scheme: str = "coprime",
        coprime_step: int = DEFAULT_COPRIME_STEP,
    ) -> None:
        self.session_counter = session_counter
        self.scheme = scheme
        self.coprime_step = coprime_step

    def interleave_instance(self, encoded_stream: bytes) -> bytes:
        return self.interleave(
            encoded_stream=encoded_stream,
            scheme=self.scheme,
            session_counter=self.session_counter,
            coprime_step=self.coprime_step,
        )

    def deinterleave_instance(self, interleaved_stream: bytes) -> bytes:
        return self.deinterleave(
            interleaved_stream=interleaved_stream,
            scheme=self.scheme,
            session_counter=self.session_counter,
            coprime_step=self.coprime_step,
        )

    @classmethod
    def _compute_stride(cls, n: int, seed: int = 0, preferred_step: int = DEFAULT_COPRIME_STEP) -> int:
        """Alias for computing coprime step >= 71."""
        return cls.get_coprime_step(n, preferred_step=preferred_step)

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
        If len(encoded_stream) <= 4, returns the stream unmodified.
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
        """Inverts symbol interleaving, restoring the canonical sequential RS codeword order.

        Preserves the 4-byte big-endian length prefix at indices 0..3 unpermuted.
        If len(interleaved_stream) <= 4, returns the stream unmodified.
        """
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
