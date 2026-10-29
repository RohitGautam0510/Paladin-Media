"""Carrier Modulation Engine Abstraction and Implementations for PaladinMedia.

Sprint 3.3.3: Unifies transform-domain carrier modulation across classical spatial
parity (Phase 2 MVP 0) and neural wavelet steganography (Phase 3 ACS-Sub-INN v3).

Architecture & Integration Invariants:
- INV-01: CarrierModulationEngine defines a narrow, engine-agnostic contract for
  carrier embedding and extraction.
- INV-02: ParityModulationEngine preserves exact Phase 2 integer-parity behavior:
  (2*HH mod 2) == bit by adjusting pixel 'a' by +/-1.
- INV-03: NeuralModulationEngine maps bits to 2D tensors, calls NeuralSpatialBridge
  differentiable DWT/IDWT and ACSSubINN v3, and handles uint8 PNG quantization.
- INV-04: MediaStegoPipeline defaults to ParityModulationEngine for 100% backward
  compatibility with existing Phase 2 callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from src.carrier.mask import EmbeddingMask


class CarrierBitExtractor(ABC):
    """Abstract interface for extracting bits from a modulated carrier container."""

    @abstractmethod
    def extract_bits(self, positions: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Extracts bits from specified (row, col) coordinates within the subband grid.

        Args:
            positions: Sequence of (row, col) coordinate pairs in the subband.

        Returns:
            1D uint8 numpy array of extracted binary values (0 or 1).
        """
        raise NotImplementedError


class CarrierModulationEngine(ABC):
    """Abstract interface for transform-domain carrier modulation and demodulation."""

    @abstractmethod
    def embed_bits(
        self,
        cover_array: np.ndarray,
        header_bits: np.ndarray,
        payload_bits: np.ndarray,
        mask: EmbeddingMask,
    ) -> np.ndarray:
        """Embeds header and payload bits into the carrier image.

        Args:
            cover_array: (H, W, 3) uint8 RGB carrier numpy array.
            header_bits: 1D uint8 array of 128 header bits.
            payload_bits: 1D uint8 array of payload bits.
            mask: EmbeddingMask containing header and payload coordinates.

        Returns:
            Stego RGB numpy array of shape (H, W, 3), dtype uint8.
        """
        raise NotImplementedError

    @abstractmethod
    def begin_extraction(
        self,
        stego_array: np.ndarray,
        subband_dim: int,
    ) -> CarrierBitExtractor:
        """Prepares extraction on a stego carrier image and returns a bit extractor.

        Args:
            stego_array: (H, W, 3) uint8 RGB stego carrier array.
            subband_dim: Dimension of square subband (e.g. 256 for 512x512).

        Returns:
            CarrierBitExtractor instance capable of extracting bits at arbitrary coordinates.
        """
        raise NotImplementedError


# =============================================================================
# Classical Spatial Integer-Parity Engine (Phase 2 Default)
# =============================================================================

class ParityBitExtractor(CarrierBitExtractor):
    """Extracts bits by evaluating spatial integer parity: (a - b - c + d) mod 2."""

    def __init__(self, carrier_array: np.ndarray, channel: int = 0) -> None:
        self.arr = carrier_array
        self.ch = channel

    def extract_bits(self, positions: Sequence[Tuple[int, int]]) -> np.ndarray:
        bits = []
        ch = self.ch
        arr = self.arr
        for r, c_idx in positions:
            a_val = int(arr[2 * r, 2 * c_idx, ch])
            b_val = int(arr[2 * r, 2 * c_idx + 1, ch])
            c_val = int(arr[2 * r + 1, 2 * c_idx, ch])
            d_val = int(arr[2 * r + 1, 2 * c_idx + 1, ch])

            two_hh = a_val - b_val - c_val + d_val
            bit = (two_hh % 2 + 2) % 2
            bits.append(bit)

        return np.array(bits, dtype=np.uint8)


class ParityModulationEngine(CarrierModulationEngine):
    """Classical Phase 2 transform-domain integer parity modulation engine.

    Modulates the Haar 2*HH coefficient on Red channel (channel 0):
        2*HH = a - b - c + d  (where a, b, c, d are 2x2 spatial pixels)
    Adjusts carrier pixel 'a' by +/-1 to enforce (2*HH mod 2) == bit.
    """

    EMBEDDING_CHANNEL: int = 0

    def embed_bits(
        self,
        cover_array: np.ndarray,
        header_bits: np.ndarray,
        payload_bits: np.ndarray,
        mask: EmbeddingMask,
    ) -> np.ndarray:
        stego_arr = cover_array.copy()
        ch = self.EMBEDDING_CHANNEL

        # 1. Modulate header bits at fixed anchor positions
        for bit, (r, c_idx) in zip(header_bits, mask.header_positions):
            a_val = int(stego_arr[2 * r, 2 * c_idx, ch])
            b_val = int(stego_arr[2 * r, 2 * c_idx + 1, ch])
            c_val = int(stego_arr[2 * r + 1, 2 * c_idx, ch])
            d_val = int(stego_arr[2 * r + 1, 2 * c_idx + 1, ch])

            two_hh = a_val - b_val - c_val + d_val
            if ((two_hh % 2 + 2) % 2) != int(bit):
                new_a = a_val + 1 if a_val < 255 else a_val - 1
                stego_arr[2 * r, 2 * c_idx, ch] = new_a

        # 2. Modulate payload bits at deterministic payload positions
        for bit, (r, c_idx) in zip(payload_bits, mask.payload_positions):
            a_val = int(stego_arr[2 * r, 2 * c_idx, ch])
            b_val = int(stego_arr[2 * r, 2 * c_idx + 1, ch])
            c_val = int(stego_arr[2 * r + 1, 2 * c_idx, ch])
            d_val = int(stego_arr[2 * r + 1, 2 * c_idx + 1, ch])

            two_hh = a_val - b_val - c_val + d_val
            if ((two_hh % 2 + 2) % 2) != int(bit):
                new_a = a_val + 1 if a_val < 255 else a_val - 1
                stego_arr[2 * r, 2 * c_idx, ch] = new_a

        return stego_arr

    def begin_extraction(
        self,
        stego_array: np.ndarray,
        subband_dim: int,
    ) -> CarrierBitExtractor:
        return ParityBitExtractor(carrier_array=stego_array, channel=self.EMBEDDING_CHANNEL)


# =============================================================================
# Neural Wavelet Modulation Engine (Phase 3 ACS-Sub-INN v3)
# =============================================================================

class NeuralBitExtractor(CarrierBitExtractor):
    """Extracts bits from pre-extracted 2D binary subband tensor emitted by ACSSubINN."""

    def __init__(self, extracted_binary_2d: np.ndarray) -> None:
        """Args:
            extracted_binary_2d: (H/2, W/2) uint8 or float binary array from neural bridge.
        """
        self.binary_grid = extracted_binary_2d

    def extract_bits(self, positions: Sequence[Tuple[int, int]]) -> np.ndarray:
        grid = self.binary_grid
        bits = [int(grid[r, c]) for r, c in positions]
        return np.array(bits, dtype=np.uint8)


class NeuralModulationEngine(CarrierModulationEngine):
    """Phase 3/5 Neural Wavelet Modulation Engine backed by PyTorch or ONNX Runtime.

    Modulates continuous Haar DWT coefficients on Red-HH subband subject to
    gamma_min = 1.0 context scaling and invertible affine coupling blocks.
    Supports both PyTorch reference execution and high-performance ONNX Runtime inference.
    """

    def __init__(
        self,
        spatial_bridge: Optional[Any] = None,
        backend: str = "pytorch",
        onnx_dir: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        session_options: Optional[Any] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes neural modulation engine with PyTorch or ONNX Runtime backend.

        Args:
            spatial_bridge: Optional NeuralSpatialBridge instance for PyTorch backend.
            backend: Inference backend selection: 'pytorch' (default) or 'onnx'.
            onnx_dir: Directory containing exported ONNX models (checkpoints/phase5a_onnx).
            checkpoint_path: Optional path to PyTorch checkpoint (default: checkpoints/phase4c_extended_30ep/best_model.pt).
            session_options: Optional onnxruntime.SessionOptions instance.
            providers: Optional sequence of ONNX execution providers (default: CPUExecutionProvider).
        """
        self.backend = backend.strip().lower()
        if self.backend not in ("pytorch", "onnx"):
            raise ValueError(f"Unsupported backend '{backend}'. Must be 'pytorch' or 'onnx'.")

        self.bridge = None
        self.onnx_sess_embed = None
        self.onnx_sess_extract = None
        self.last_clip_stats: Optional[Dict[str, Any]] = None
        self.last_embedding_result: Optional[Any] = None
        self.last_header_bits: Optional[np.ndarray] = None
        self.last_payload_bits: Optional[np.ndarray] = None

        if self.backend == "pytorch":
            if spatial_bridge is not None:
                self.bridge = spatial_bridge
            else:
                from src.neural.acs_sub_inn import ACSSubINN
                from src.neural.spatial_bridge import NeuralSpatialBridge
                from src.neural.training_engine import load_training_checkpoint

                if checkpoint_path is None:
                    ckpt_p = (
                        Path(__file__).resolve().parent.parent.parent
                        / "checkpoints"
                        / "phase4c_extended_30ep"
                        / "best_model.pt"
                    )
                else:
                    ckpt_p = Path(checkpoint_path)

                model = ACSSubINN(gamma_min=1.0)
                self.bridge = NeuralSpatialBridge(acs_model=model)
                if ckpt_p.exists():
                    load_training_checkpoint(ckpt_p, bridge=self.bridge)
                self.bridge.eval()
        else:
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise ImportError(
                    "ONNX Runtime backend requested, but 'onnxruntime' is not installed."
                ) from e

            if onnx_dir is None:
                onnx_dir = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "phase5a_onnx"
            else:
                onnx_dir = Path(onnx_dir)

            embed_path = onnx_dir / "neural_spatial_embedding_opset14.onnx"
            extract_path = onnx_dir / "neural_spatial_extraction_opset14.onnx"

            if not embed_path.exists():
                raise FileNotFoundError(f"ONNX embedding model missing at {embed_path}")
            if not extract_path.exists():
                raise FileNotFoundError(f"ONNX extraction model missing at {extract_path}")

            provs = list(providers) if providers is not None else ["CPUExecutionProvider"]
            opts = session_options if session_options is not None else ort.SessionOptions()

            self.onnx_sess_embed = ort.InferenceSession(str(embed_path), opts, providers=provs)
            self.onnx_sess_extract = ort.InferenceSession(str(extract_path), opts, providers=provs)
            self.onnx_dir = onnx_dir

    def embed_bits(
        self,
        cover_array: np.ndarray,
        header_bits: np.ndarray,
        payload_bits: np.ndarray,
        mask: EmbeddingMask,
    ) -> np.ndarray:
        if self.backend == "onnx":
            h, w, c = cover_array.shape
            sub_h, sub_w = h // 2, w // 2

            carrier_np = np.ascontiguousarray(
                cover_array.transpose(2, 0, 1)[None, ...], dtype=np.float32
            )
            mask_arr = np.zeros((1, 1, sub_h, sub_w), dtype=np.float32)
            payload_arr = np.zeros((1, 1, sub_h, sub_w), dtype=np.float32)

            for bit, (r, c_idx) in zip(header_bits, mask.header_positions):
                mask_arr[0, 0, r, c_idx] = 1.0
                payload_arr[0, 0, r, c_idx] = 1.0 if int(bit) == 1 else -1.0

            for bit, (r, c_idx) in zip(payload_bits, mask.payload_positions):
                mask_arr[0, 0, r, c_idx] = 1.0
                payload_arr[0, 0, r, c_idx] = 1.0 if int(bit) == 1 else -1.0

            stego_img, ctx, gamma = self.onnx_sess_embed.run(
                None,
                {
                    "carrier": carrier_np,
                    "mask": mask_arr,
                    "payload_bipolar": payload_arr,
                },
            )

            stego_uint8 = np.clip(np.round(stego_img), 0.0, 255.0).astype(np.uint8)
            self.last_header_bits = header_bits.copy()
            self.last_payload_bits = payload_bits.copy()
            self.last_clip_stats = {
                "min_val_raw": float(np.min(stego_img)),
                "max_val_raw": float(np.max(stego_img)),
                "num_clipped_low": int(np.sum(stego_img < 0.0)),
                "num_clipped_high": int(np.sum(stego_img > 255.0)),
                "num_clipped_total": int(np.sum(stego_img < 0.0) + np.sum(stego_img > 255.0)),
                "clip_fraction": float((np.sum(stego_img < 0.0) + np.sum(stego_img > 255.0)) / stego_img.size),
            }
            return stego_uint8[0].transpose(1, 2, 0)

        # PyTorch reference execution path
        import torch

        h, w, c = cover_array.shape
        sub_h, sub_w = h // 2, w // 2

        cover_tensor = torch.from_numpy(cover_array).permute(2, 0, 1).unsqueeze(0).float()
        mask_tensor = torch.zeros(1, 1, sub_h, sub_w, dtype=torch.float32)
        payload_tensor = torch.zeros(1, 1, sub_h, sub_w, dtype=torch.float32)

        for bit, (r, c_idx) in zip(header_bits, mask.header_positions):
            mask_tensor[0, 0, r, c_idx] = 1.0
            payload_tensor[0, 0, r, c_idx] = 1.0 if int(bit) == 1 else -1.0

        for bit, (r, c_idx) in zip(payload_bits, mask.payload_positions):
            mask_tensor[0, 0, r, c_idx] = 1.0
            payload_tensor[0, 0, r, c_idx] = 1.0 if int(bit) == 1 else -1.0

        emb_res = self.bridge.embed_spatial(
            carrier=cover_tensor,
            payload_bipolar=payload_tensor,
            mask=mask_tensor,
            differentiable=False,
            clip_pixels=True,
        )

        self.last_clip_stats = emb_res.clip_stats
        self.last_embedding_result = emb_res
        self.last_header_bits = header_bits.copy()
        self.last_payload_bits = payload_bits.copy()
        stego_arr = emb_res.stego_image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return stego_arr

    def begin_extraction(
        self,
        stego_array: np.ndarray,
        subband_dim: int,
    ) -> CarrierBitExtractor:
        if self.backend == "onnx":
            carrier_np = np.ascontiguousarray(
                stego_array.transpose(2, 0, 1)[None, ...], dtype=np.float32
            )
            ext_bin, delta = self.onnx_sess_extract.run(
                None, {"stego_carrier": carrier_np}
            )
            binary_np = ext_bin[0, 0].astype(np.uint8)
            return NeuralBitExtractor(binary_np)

        # PyTorch reference execution path
        import torch

        stego_tensor = torch.from_numpy(stego_array).permute(2, 0, 1).unsqueeze(0).float()
        ext_res = self.bridge.extract_spatial(stego_tensor)
        binary_np = ext_res.extracted_binary.squeeze().cpu().numpy().astype(np.uint8)
        return NeuralBitExtractor(binary_np)
