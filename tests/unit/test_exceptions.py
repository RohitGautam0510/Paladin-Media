"""Unit tests for PaladinMedia core exception hierarchy."""

import pytest
from src.core.exceptions import (
    AuthenticationError,
    CryptoError,
    ECCDecodeError,
    ECCError,
    PaladinSecurityError,
    SerializationError,
)


def test_exception_inheritance():
    """Verify that all custom exceptions conform to the architectural inheritance hierarchy."""
    # Base exception
    assert issubclass(PaladinSecurityError, Exception)

    # Level 1 specializations
    assert issubclass(SerializationError, PaladinSecurityError)
    assert issubclass(CryptoError, PaladinSecurityError)
    assert issubclass(ECCError, PaladinSecurityError)

    # Level 2 specializations
    assert issubclass(AuthenticationError, CryptoError)
    assert issubclass(AuthenticationError, PaladinSecurityError)

    assert issubclass(ECCDecodeError, ECCError)
    assert issubclass(ECCDecodeError, PaladinSecurityError)


def test_exception_instantiation_and_messages():
    """Verify that exceptions retain provided messages and can be raised/caught properly."""
    msg = "Serialization failed: invalid magic byte sequence"
    with pytest.raises(SerializationError) as exc_info:
        raise SerializationError(msg)
    assert msg in str(exc_info.value)

    auth_msg = "AES-GCM authentication failed: tag mismatch"
    with pytest.raises(CryptoError) as exc_info:
        raise AuthenticationError(auth_msg)
    assert auth_msg in str(exc_info.value)

    ecc_msg = "Uncorrectable symbol errors: count 17 > t 16"
    with pytest.raises(ECCError) as exc_info:
        raise ECCDecodeError(ecc_msg)
    assert ecc_msg in str(exc_info.value)
