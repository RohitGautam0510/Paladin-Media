"""PaladinMedia: Energy-Efficient Post-Quantum Steganography and Media Protection Runtime.

Phase 13.1 Package Root providing clean top-level imports and version metadata.
"""

from __future__ import annotations

__version__ = "2.6.0rc1"

from src.core.media_pipeline import MediaStegoPipeline
from src.core.serialization.serializer import MediaType, SecretPayload

__all__ = [
    "MediaStegoPipeline",
    "SecretPayload",
    "MediaType",
    "__version__",
]
