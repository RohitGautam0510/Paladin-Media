"""Error correction coding (ECC) module for PaladinMedia.

Provides abstract base class ECCEncoder and concrete ReedSolomonEncoder implementing
systematic chunked Reed-Solomon over GF(2^8) with symbol-level error correction.
"""

from src.security.ecc.encoder import (
    ECCEncoder,
    ECCParameterError,
    ECCResult,
    ReedSolomonEncoder,
)
from src.core.exceptions import ECCDecodeError, ECCError

__all__ = [
    "ECCEncoder",
    "ReedSolomonEncoder",
    "ECCError",
    "ECCDecodeError",
    "ECCParameterError",
    "ECCResult",
]
