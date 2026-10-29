"""Carrier and signal processing modules for PaladinMedia Phase 2 and Phase 3."""

from src.carrier.capacity import CapacityCalculator, CapacityReport, PayloadCapacityExceededError, resolve_subband_dims
from src.carrier.dwt import DWTSubbands, HaarDWT2D
from src.carrier.embedding import BaselineEmbedder, EmbeddingResult
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader, CarrierHeaderError
from src.carrier.mask import DeterministicMaskGenerator, EmbeddingMask, MaskGenerationError
from src.carrier.modulation import (
    CarrierBitExtractor,
    CarrierModulationEngine,
    NeuralBitExtractor,
    NeuralModulationEngine,
    ParityBitExtractor,
    ParityModulationEngine,
)

__all__ = [
    "CapacityCalculator",
    "CapacityReport",
    "PayloadCapacityExceededError",
    "HaarDWT2D",
    "DWTSubbands",
    "BaselineEmbedder",
    "EmbeddingResult",
    "Extractor",
    "CarrierHeader",
    "CarrierHeaderError",
    "DeterministicMaskGenerator",
    "EmbeddingMask",
    "MaskGenerationError",
    "CarrierBitExtractor",
    "CarrierModulationEngine",
    "ParityBitExtractor",
    "ParityModulationEngine",
    "NeuralBitExtractor",
    "NeuralModulationEngine",
]
