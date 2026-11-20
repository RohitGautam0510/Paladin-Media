"""
Post-Quantum Cryptography (PQC) Subsystem for PaladinMedia.

Provides NIST FIPS 203 (ML-KEM-768), NIST FIPS 204 (ML-DSA-65),
canonical AKE transcript generation, trust management, and fail-closed security.
"""

from src.security.pqc.ake import AKEEngine
from src.security.pqc.exceptions import (
    PQCDisabledError,
    PQCEnvelopeError,
    PQCError,
    PQCIdentityError,
    PQCKeyConfirmationError,
    PQCKeyExpiredError,
    PQCKeyRevokedError,
    PQCProviderError,
    PQCReplayError,
    PQCSignatureVerificationError,
    PQCDecapsulationError,
)
from src.security.pqc.models import (
    CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256,
    FLAG_DETACHED_SIGNATURE,
    FLAG_EPHEMERAL_KEM,
    MAGIC_PQC_AKE,
    MAGIC_PQC_KEM,
    MAGIC_PQC_SIDECAR,
    VERSION_PQC_V1,
    PQCEnvelope,
    PQCHeaderA,
    PQCHeaderB1,
    PQCIdentity,
    PQCSidecar,
)
from src.security.pqc.provider import CryptographyPQCProvider, PQCProvider, default_pqc_provider
from src.security.pqc.trust import TrustStore, default_trust_store

__all__ = [
    "AKEEngine",
    "CryptographyPQCProvider",
    "PQCProvider",
    "default_pqc_provider",
    "TrustStore",
    "default_trust_store",
    "PQCIdentity",
    "PQCHeaderB1",
    "PQCHeaderA",
    "PQCEnvelope",
    "PQCSidecar",
    "MAGIC_PQC_SIDECAR",
    "PQCError",
    "PQCProviderError",
    "PQCDisabledError",
    "PQCEnvelopeError",
    "PQCIdentityError",
    "PQCKeyExpiredError",
    "PQCKeyRevokedError",
    "PQCSignatureVerificationError",
    "PQCKeyConfirmationError",
    "PQCReplayError",
    "PQCDecapsulationError",
    "CIPHERSUITE_MLKEM768_MLDSA65_AES256GCM_HKDF_SHA256",
    "MAGIC_PQC_AKE",
    "MAGIC_PQC_KEM",
    "VERSION_PQC_V1",
    "FLAG_DETACHED_SIGNATURE",
    "FLAG_EPHEMERAL_KEM",
]
