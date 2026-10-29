"""Unit tests for Baseline Embedding and PNG Round-Trip (T-017, T-018)."""

import io
import numpy as np
import pytest
from PIL import Image

from src.carrier.embedding import BaselineEmbedder, EmbeddingResult
from src.carrier.extraction import Extractor


def test_t017_baseline_embed_extract():
    """T-017: Verify baseline transform-domain embedding and matching extraction."""
    # Synthetic 512x512 RGB carrier
    rng = np.random.RandomState(42)
    cover_arr = rng.randint(10, 245, (512, 512, 3), dtype=np.uint8)

    # 1,500 byte payload
    payload = bytes(rng.randint(0, 256, 1500, dtype=np.uint8))
    session = 1001

    res = BaselineEmbedder.embed(cover_arr, payload, session_counter=session)
    assert isinstance(res, EmbeddingResult)
    assert res.stego_array.shape == (512, 512, 3)

    # Extract directly from stego array
    header, extracted_payload = Extractor.extract(res.stego_array)
    assert header.session_counter == session
    assert header.payload_length == 1500
    assert extracted_payload == payload


def test_t018_png_save_reload_carrier():
    """T-018: Verify carrier survives 8-bit RGB PNG encoding/decoding with zero bit error."""
    # Load actual image from MVP 0 Dataset if present, else synthetic carrier
    cover = Image.new("RGB", (512, 512), color=(140, 180, 210))

    payload = b"PALADIN_MEDIA_PNG_ROUNDTRIP_TEST_DATA_" * 40  # 1520 bytes
    session = 2002

    res = BaselineEmbedder.embed(cover, payload, session_counter=session)

    # PNG round-trip: save to in-memory bytes and re-open
    png_buf = io.BytesIO()
    res.stego_image.save(png_buf, format="PNG")
    png_buf.seek(0)
    reloaded_stego = Image.open(png_buf)

    # Extract from reloaded PNG
    header, extracted_payload = Extractor.extract(reloaded_stego)
    assert header.session_counter == session
    assert header.payload_length == len(payload)
    assert extracted_payload == payload, "Extracted payload must match verbatim through PNG reload"


def test_1024_embed_extract_png_roundtrip():
    """Verify 1024x1024 carrier embedding and extraction through PNG roundtrip."""
    cover = Image.new("RGB", (1024, 1024), color=(100, 150, 200))
    # 10 KB payload (well above 512x512 8,176 B limit, but within 1024x1024 32,752 B limit)
    rng = np.random.RandomState(1024)
    payload = bytes(rng.randint(0, 256, 10000, dtype=np.uint8))
    session = 4004

    res = BaselineEmbedder.embed(cover, payload, session_counter=session)
    assert res.stego_array.shape == (1024, 1024, 3)

    png_buf = io.BytesIO()
    res.stego_image.save(png_buf, format="PNG")
    png_buf.seek(0)
    reloaded_stego = Image.open(png_buf)

    header, extracted_payload = Extractor.extract(reloaded_stego)
    assert header.session_counter == session
    assert header.payload_length == 10000
    assert extracted_payload == payload
