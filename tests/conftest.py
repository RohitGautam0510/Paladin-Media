"""Global pytest fixtures and configuration for PaladinMedia test suite."""

import os
import sys
import pytest
from typing import Any, Dict

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.fixtures.vectors import (
    PAYLOAD_EMPTY,
    PAYLOAD_SINGLE_BYTE,
    PAYLOAD_TEXT,
    PAYLOAD_JSON,
    PAYLOAD_IMAGE_SNIPPET,
    PAYLOAD_CAPACITY,
    PAYLOAD_MULTI_BLOCK,
)


@pytest.fixture(scope="session")
def sample_key() -> bytes:
    """Provides a fixed 32-byte (256-bit) AES key for deterministic tests."""
    return bytes.fromhex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")


@pytest.fixture(scope="session")
def alternative_key() -> bytes:
    """Provides an alternative 32-byte key to test key mismatch / negative tests."""
    return bytes.fromhex("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210")


@pytest.fixture(scope="session")
def sample_nonce() -> bytes:
    """Provides a fixed 12-byte (96-bit) nonce for deterministic tests."""
    return bytes.fromhex("4e6f6e636531323334353637")  # b"Nonce1234567"


@pytest.fixture(scope="session")
def sample_aad() -> bytes:
    """Provides sample Associated Authenticated Data (AAD)."""
    return b"PaladinMedia-Session-001:HeaderContext"


@pytest.fixture(scope="session")
def sample_payloads() -> Dict[str, bytes]:
    """Provides a dictionary of representative secret payloads."""
    return {
        "empty": PAYLOAD_EMPTY,
        "single_byte": PAYLOAD_SINGLE_BYTE,
        "text": PAYLOAD_TEXT,
        "json": PAYLOAD_JSON,
        "image_snippet": PAYLOAD_IMAGE_SNIPPET,
        "capacity": PAYLOAD_CAPACITY,
        "multi_block": PAYLOAD_MULTI_BLOCK,
    }


def get_serialization_components():
    """Resolves SecretPayload, MediaType, SecretSerializer."""
    from src.core.serialization import SecretPayload, MediaType, SecretSerializer
    return SecretPayload, MediaType, SecretSerializer


def get_crypto_components():
    """Resolves CryptoProvider, AESGCMProvider cleanly from src.security.crypto."""
    from src.security.crypto import CryptoProvider, AESGCMProvider
    return CryptoProvider, AESGCMProvider


def get_ecc_components():
    """Resolves ECCEncoder, ReedSolomonEncoder cleanly from src.security.ecc."""
    from src.security.ecc import ECCEncoder, ReedSolomonEncoder
    return ECCEncoder, ReedSolomonEncoder


def get_pipeline_component():
    """Resolves SecurityPipeline cleanly from src.core.pipeline."""
    from src.core.pipeline import SecurityPipeline
    return SecurityPipeline


@pytest.fixture
def crypto_provider():
    """Provides an initialized instance of AESGCMProvider."""
    _, provider_cls = get_crypto_components()
    return provider_cls()


@pytest.fixture
def ecc_encoder():
    """Provides an initialized instance of ReedSolomonEncoder."""
    _, encoder_cls = get_ecc_components()
    return encoder_cls()
