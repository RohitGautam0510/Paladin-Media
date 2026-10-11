#!/usr/bin/env python3
"""PaladinMedia MVP 0 Sprint 1: End-to-End Programmatic Verification Script.

This standalone verification script comprehensively validates Sprint 1 acceptance criteria:
"A programmatic integration script successfully passes a secret through Serialization ->
 Encryption -> ECC and fully reverses the process to recover the exact original bytes."

Mandatory Acceptance Demonstrations:
1. Full Reversibility across diverse media types (Text, JSON, Binary, Image, Audio).
2. Boundary Condition Handling (0-byte, 1-byte, ~7KB single sub-band capacity, 12KB multi-block).
3. Exact Symbol-Level Error Correction ($GF(2^8)$, up to $t=16$ errors per block, 128-bit burst errors).
4. Strict Fail-Closed Security (tampered ciphertext, tag, nonce, AAD, key, uncorrectable noise, 0 plaintext).

Returns exit code 0 if all checks pass; non-zero on failure.
"""

from __future__ import annotations

import os
import random
import struct
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.exceptions import (
    AuthenticationError,
    CryptoError,
    ECCDecodeError,
    ECCError,
    PaladinSecurityError,
    SerializationError,
)
from src.core.pipeline import SecurityPipeline
from src.core.serialization import MediaType, SecretPayload, SecretSerializer
from src.security.crypto import AESGCMProvider, CryptoProvider
from src.security.ecc import ECCEncoder, ReedSolomonEncoder


# =============================================================================
# ANSI Color & Formatting Utilities
# =============================================================================
class Colors:
    """Terminal styling codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    GRAY = "\033[90m"


def supports_color() -> bool:
    """Detects whether terminal supports ANSI color formatting."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = supports_color()


def c(text: str, color_code: str) -> str:
    """Applies color code if terminal supports it."""
    if USE_COLOR:
        return f"{color_code}{text}{Colors.RESET}"
    return text


# =============================================================================
# Error Injection & Tampering Utilities
# =============================================================================
def flip_bit(data: bytes, byte_pos: int, bit_pos: int = 0) -> bytes:
    """Flips a single bit at the specified byte offset and bit position."""
    if not (0 <= byte_pos < len(data)):
        raise ValueError(f"byte_pos {byte_pos} out of range for data of length {len(data)}")
    if not (0 <= bit_pos <= 7):
        raise ValueError(f"bit_pos {bit_pos} must be between 0 and 7")
    byte_arr = bytearray(data)
    byte_arr[byte_pos] ^= 1 << bit_pos
    return bytes(byte_arr)


def inject_symbol_errors(
    data: bytes,
    count: int,
    block_offset: int,
    block_length: int,
    seed: Optional[int] = None,
) -> Tuple[bytes, List[int]]:
    """Injects exact count of byte (symbol) substitutions within a specific block slice."""
    if count > block_length:
        raise ValueError(f"Error count {count} cannot exceed block length {block_length}")
    rng = random.Random(seed)
    indices = rng.sample(range(block_offset, block_offset + block_length), count)
    byte_arr = bytearray(data)
    for idx in indices:
        original = byte_arr[idx]
        delta = rng.randint(1, 255)
        byte_arr[idx] = (original ^ delta) if (original ^ delta) != original else (original ^ 0xFF)
    return bytes(byte_arr), indices


def inject_burst_error(
    data: bytes,
    start_offset: int,
    length: int,
    pattern: int = 0xFF,
) -> bytes:
    """Injects contiguous byte burst corruption across a slice of bytes."""
    byte_arr = bytearray(data)
    for i in range(start_offset, min(len(data), start_offset + length)):
        byte_arr[i] ^= pattern
    return bytes(byte_arr)


# =============================================================================
# Verification Harness Framework
# =============================================================================
class VerificationSuite:
    """Manages check execution, timing, assertions, and reporting."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.total: int = 0
        self.current_section: str = ""
        self.start_time: float = time.perf_counter()

    def section(self, name: str) -> None:
        """Starts a new named section in the verification output."""
        self.current_section = name
        print()
        print(c(f"=== {name} ===", Colors.BOLD + Colors.CYAN))

    def check(self, name: str, fn: Callable[[], bool], description: str = "") -> None:
        """Executes a verification check and records the result."""
        self.total += 1
        desc_str = f" ({description})" if description else ""
        sys.stdout.write(f"  [{self.total:02d}] {name}{desc_str} ... ")
        sys.stdout.flush()

        t0 = time.perf_counter()
        try:
            success = fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if success:
                self.passed += 1
                status = c("PASS", Colors.BOLD + Colors.GREEN)
                print(f"{status} {c(f'({elapsed_ms:.1f}ms)', Colors.GRAY)}")
            else:
                self.failed += 1
                status = c("FAIL", Colors.BOLD + Colors.RED)
                print(f"{status} {c(f'({elapsed_ms:.1f}ms)', Colors.GRAY)}")
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.failed += 1
            status = c("ERROR", Colors.BOLD + Colors.RED)
            print(f"{status} {c(f'({elapsed_ms:.1f}ms)', Colors.GRAY)}")
            print(c(f"       Unexpected Exception: {type(exc).__name__}: {exc}", Colors.RED))

    def summarize(self) -> int:
        """Prints the final summary report and returns process exit code."""
        duration = time.perf_counter() - self.start_time
        print()
        print(c("==================================================================", Colors.BOLD))
        print(c("                 PALADINMEDIA SPRINT 1 VERIFICATION REPORT        ", Colors.BOLD))
        print(c("==================================================================", Colors.BOLD))
        print(f"Total Checks Executed : {self.total}")
        print(f"Passed Checks         : {c(str(self.passed), Colors.GREEN if self.passed == self.total else Colors.YELLOW)}")
        print(f"Failed Checks         : {c(str(self.failed), Colors.GREEN if self.failed == 0 else Colors.RED)}")
        print(f"Elapsed Time          : {duration:.3f} seconds")
        print("------------------------------------------------------------------")

        if self.failed == 0:
            print(c("RESULT: ALL SPRINT 1 ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY (PASS)", Colors.BOLD + Colors.GREEN))
            return 0
        else:
            print(c(f"RESULT: {self.failed} CHECK(S) FAILED. SPRINT 1 ACCEPTANCE CRITERIA NOT MET.", Colors.BOLD + Colors.RED))
            return 1


# =============================================================================
# Main Verification Routine
# =============================================================================
def main() -> int:
    """Executes the complete Sprint 1 verification suite."""
    print(c("==================================================================", Colors.BOLD))
    print(c("  PaladinMedia MVP 0 Sprint 1: Security Pipeline Verification     ", Colors.BOLD + Colors.BLUE))
    print(c("  Architectural Mandate: Serialization -> AES-GCM -> Reed-Solomon ", Colors.GRAY))
    print(c("==================================================================", Colors.BOLD))

    suite = VerificationSuite()
    pipeline = SecurityPipeline()

    test_key = bytes.fromhex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
    alt_key = bytes.fromhex("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210")
    test_aad = b"PaladinMedia-VerificationSession-2026-ContextAAD"

    # =========================================================================
    # Section 1: Architecture & Sequence Invariants
    # =========================================================================
    suite.section("1. Architecture & Sequence Invariants (INV-01, INV-02)")

    def check_pipeline_initialization() -> bool:
        p1 = SecurityPipeline()
        assert isinstance(p1.crypto_provider, CryptoProvider)
        assert isinstance(p1.ecc_encoder, ECCEncoder)
        custom_crypto = AESGCMProvider()
        custom_ecc = ReedSolomonEncoder(ecc_symbols=32)
        p2 = SecurityPipeline(crypto_provider=custom_crypto, ecc_encoder=custom_ecc)
        assert p2.crypto_provider is custom_crypto
        assert p2.ecc_encoder is custom_ecc
        return True

    suite.check("Pipeline Initialization & Default Providers", check_pipeline_initialization, "Crypto & ECC injection")

    def check_strict_wire_framing_and_sequence() -> bool:
        # Step-by-step trace matching pipeline.protect
        payload = SecretPayload(data=b"Test sequence invariant", media_type=MediaType.TEXT)
        serialized = SecretSerializer.serialize(payload)
        # 1. Check Magic b"PLDS" and Version 0x01
        assert serialized[:4] == b"PLDS"
        assert serialized[4] == 0x01
        # 2. Encrypt
        encrypted = pipeline.crypto_provider.encrypt(serialized, test_key, aad=test_aad)
        assert len(encrypted) == len(serialized) + 28  # 12B nonce + 16B tag
        # 3. ECC Encode
        encoded = pipeline.ecc_encoder.encode(encrypted)
        # Check length prefix
        orig_len = struct.unpack(">I", encoded[:4])[0]
        assert orig_len == len(encrypted)
        return True

    suite.check("Strict Unidirectional Transformation Sequence", check_strict_wire_framing_and_sequence, "Serialize -> Encrypt -> ECC")

    # =========================================================================
    # Section 2: Full Reversibility Across Multiple Media Types
    # =========================================================================
    suite.section("2. Full Reversibility Across Media Types (Acceptance Criteria)")

    def check_reversibility_text() -> bool:
        text_data = "Classified Operation Directive: Deploy stealth watermark at carrier coordinate (41.8902, 12.4922).".encode("utf-8")
        payload = SecretPayload(
            data=text_data,
            media_type=MediaType.TEXT,
            metadata={"classification": "TOP SECRET", "author": "HQ", "ttl": 3600},
        )
        protected = pipeline.protect(payload, test_key, aad=test_aad)
        recovered = pipeline.unprotect(protected, test_key, aad=test_aad)
        return (
            recovered.data == payload.data
            and recovered.media_type == MediaType.TEXT
            and recovered.metadata == payload.metadata
        )

    suite.check("Media Type TEXT Roundtrip", check_reversibility_text, "UTF-8 text with JSON metadata")

    def check_reversibility_json() -> bool:
        json_data = b'{"status": "active", "nodes": [101, 102, 103], "threshold": 0.995, "flags": {"stealth": true}}'
        payload = SecretPayload(
            data=json_data,
            media_type=MediaType.JSON,
            metadata={"schema": "v1.2", "compressed": False},
        )
        protected = pipeline.protect(payload, test_key, aad=test_aad)
        recovered = pipeline.unprotect(protected, test_key, aad=test_aad)
        return (
            recovered.data == payload.data
            and recovered.media_type == MediaType.JSON
            and recovered.metadata == payload.metadata
        )

    suite.check("Media Type JSON Roundtrip", check_reversibility_json, "Structured JSON payload & metadata")

    def check_reversibility_binary() -> bool:
        # High-entropy random binary key/seed material
        binary_data = bytes(range(256)) + os.urandom(128)
        payload = SecretPayload(
            data=binary_data,
            media_type=MediaType.RAW_BYTES,
            metadata={"entropy_bits": 3072, "format": "raw_entropy"},
        )
        protected = pipeline.protect(payload, test_key, aad=test_aad)
        recovered = pipeline.unprotect(protected, test_key, aad=test_aad)
        return (
            recovered.data == payload.data
            and recovered.media_type == MediaType.RAW_BYTES
            and recovered.metadata == payload.metadata
        )

    suite.check("Media Type RAW_BYTES Roundtrip", check_reversibility_binary, "High-entropy binary entropy stream")

    def check_reversibility_image() -> bool:
        # 16x16 8-bit grayscale pixel raster snippet (256 bytes)
        raster_pixels = bytes([((x * 17) ^ (y * 31)) & 0xFF for x in range(16) for y in range(16)])
        payload = SecretPayload(
            data=raster_pixels,
            media_type=MediaType.IMAGE,
            metadata={"width": 16, "height": 16, "channels": 1, "format": "GRAYSCALE_RAW"},
        )
        protected = pipeline.protect(payload, test_key, aad=test_aad)
        recovered = pipeline.unprotect(protected, test_key, aad=test_aad)
        return (
            recovered.data == payload.data
            and recovered.media_type == MediaType.IMAGE
            and recovered.metadata == payload.metadata
        )

    suite.check("Media Type IMAGE Roundtrip", check_reversibility_image, "16x16 pixel raster image snippet")

    def check_reversibility_audio() -> bool:
        # Synthetic 16-bit PCM waveform chunk
        audio_samples = bytes([int(128 + 120 * (i % 20) / 20) for i in range(500)])
        payload = SecretPayload(
            data=audio_samples,
            media_type=MediaType.AUDIO,
            metadata={"sample_rate": 44100, "channels": 1, "bit_depth": 8},
        )
        protected = pipeline.protect(payload, test_key, aad=test_aad)
        recovered = pipeline.unprotect(protected, test_key, aad=test_aad)
        return (
            recovered.data == payload.data
            and recovered.media_type == MediaType.AUDIO
            and recovered.metadata == payload.metadata
        )

    suite.check("Media Type AUDIO Roundtrip", check_reversibility_audio, "Synthetic acoustic waveform samples")

    # =========================================================================
    # Section 3: Boundary Conditions & Capacity Validation
    # =========================================================================
    suite.section("3. Boundary Conditions & Capacity Validation (T-001, Capacity Rules)")

    def check_boundary_zero_bytes() -> bool:
        payload = SecretPayload(data=b"", media_type=MediaType.RAW_BYTES, metadata=None)
        protected = pipeline.protect(payload, test_key)
        recovered = pipeline.unprotect(protected, test_key)
        return recovered.data == b"" and recovered.media_type == MediaType.RAW_BYTES

    suite.check("Boundary: 0-Byte Payload (Empty)", check_boundary_zero_bytes, "Minimum valid payload size")

    def check_boundary_single_byte() -> bool:
        payload = SecretPayload(data=b"\x5a", media_type=MediaType.RAW_BYTES, metadata={"tag": "single"})
        protected = pipeline.protect(payload, test_key)
        recovered = pipeline.unprotect(protected, test_key)
        return recovered.data == b"\x5a" and recovered.metadata == {"tag": "single"}

    suite.check("Boundary: 1-Byte Payload", check_boundary_single_byte, "Single byte content")

    def check_boundary_dwt_capacity_7088_bytes() -> bool:
        # PaladinMedia single DWT sub-band capacity payload: 7,088 bytes
        # Generated deterministically
        capacity_data = bytes((i * 37 + 13) & 0xFF for i in range(7088))
        payload = SecretPayload(
            data=capacity_data,
            media_type=MediaType.RAW_BYTES,
            metadata={"subband": "HH", "capacity_bytes": 7088, "dwt_level": 2},
        )
        protected = pipeline.protect(payload, test_key, aad=test_aad)
        recovered = pipeline.unprotect(protected, test_key, aad=test_aad)
        return (
            len(recovered.data) == 7088
            and recovered.data == capacity_data
            and recovered.metadata["capacity_bytes"] == 7088
        )

    suite.check("Boundary: ~7KB DWT Sub-Band Capacity Payload", check_boundary_dwt_capacity_7088_bytes, "7,088-byte full subband limit")

    def check_boundary_multi_block_12kb() -> bool:
        # Multi-block stream: 12,000 bytes spanning ~54 RS blocks
        multi_data = bytes((i * 59) & 0xFF for i in range(12000))
        payload = SecretPayload(data=multi_data, media_type=MediaType.RAW_BYTES)
        protected = pipeline.protect(payload, test_key)
        recovered = pipeline.unprotect(protected, test_key)
        return recovered.data == multi_data and len(recovered.data) == 12000

    suite.check("Boundary: 12KB Large Multi-Block Stream", check_boundary_multi_block_12kb, "Spanning ~54 RS blocks")

    # =========================================================================
    # Section 4: Exact Symbol-Level Error Correction
    # =========================================================================
    suite.section("4. Symbol-Level Error Correction (GF(2^8), t=16 Symbol Budget)")

    def check_ecc_single_symbol_error() -> bool:
        payload = SecretPayload(data=b"Single symbol error correction test", media_type=MediaType.TEXT)
        protected = pipeline.protect(payload, test_key, aad=test_aad)

        # Corrupt 1 byte in the RS codeword (offset 10, past 4B length prefix)
        corrupted, _ = inject_symbol_errors(protected, count=1, block_offset=4, block_length=100, seed=123)
        recovered = pipeline.unprotect(corrupted, test_key, aad=test_aad)
        return recovered.data == payload.data

    suite.check("ECC: 1 Symbol Error Repaired", check_ecc_single_symbol_error, "1 corrupted byte corrected")

    def check_ecc_half_bound_8_errors() -> bool:
        payload = SecretPayload(data=b"Testing error correction at half bound (8 symbols per block)", media_type=MediaType.TEXT)
        protected = pipeline.protect(payload, test_key, aad=test_aad)

        # Inject 8 symbol errors in first block
        corrupted, indices = inject_symbol_errors(protected, count=8, block_offset=4, block_length=min(255, len(protected) - 4), seed=456)
        assert len(indices) == 8
        recovered = pipeline.unprotect(corrupted, test_key, aad=test_aad)
        return recovered.data == payload.data

    suite.check("ECC: 8 Symbol Errors Repaired (Half Bound)", check_ecc_half_bound_8_errors, "t/2 = 8 corrupted symbols")

    def check_ecc_exact_bound_t16_errors() -> bool:
        # Payload large enough to produce a full 255-byte codeword
        raw_text = b"P" * 200
        payload = SecretPayload(data=raw_text, media_type=MediaType.RAW_BYTES)
        protected = pipeline.protect(payload, test_key, aad=test_aad)

        # Inject EXACTLY t=16 symbol errors into the 255-byte block
        corrupted, indices = inject_symbol_errors(protected, count=16, block_offset=4, block_length=255, seed=789)
        assert len(indices) == 16
        recovered = pipeline.unprotect(corrupted, test_key, aad=test_aad)
        return recovered.data == payload.data

    suite.check("ECC: Exactly t=16 Symbol Errors Repaired (Maximum Bound)", check_ecc_exact_bound_t16_errors, "Exact theoretical limit t=16")

    def check_ecc_contiguous_burst_error_16_bytes() -> bool:
        # A contiguous burst of 16 corrupted bytes = 128 consecutive bit errors
        # In RS GF(2^8), this consumes exactly 16 symbols of the budget
        raw_text = b"Burst error resilience: 16 contiguous bytes corrupted completely!" * 4
        payload = SecretPayload(data=raw_text, media_type=MediaType.TEXT)
        protected = pipeline.protect(payload, test_key, aad=test_aad)

        # Inject 16-byte burst in first block (offset 20 to 36)
        corrupted = inject_burst_error(protected, start_offset=20, length=16, pattern=0xAA)
        recovered = pipeline.unprotect(corrupted, test_key, aad=test_aad)
        return recovered.data == payload.data

    suite.check("ECC: 16-Byte Contiguous Burst Error (128 Bits)", check_ecc_contiguous_burst_error_16_bytes, "128 consecutive flipped bits")

    def check_ecc_multi_block_distributed_noise() -> bool:
        # Capacity payload ~7KB across ~32 RS blocks; inject 10 errors into blocks 1, 2, 3
        capacity_data = bytes((i * 41) & 0xFF for i in range(5000))
        payload = SecretPayload(data=capacity_data, media_type=MediaType.RAW_BYTES)
        protected = pipeline.protect(payload, test_key)

        corrupted = protected
        # Block 1 (offset 4, len 255)
        corrupted, _ = inject_symbol_errors(corrupted, count=10, block_offset=4, block_length=255, seed=101)
        # Block 2 (offset 259, len 255)
        corrupted, _ = inject_symbol_errors(corrupted, count=10, block_offset=259, block_length=255, seed=202)
        # Block 3 (offset 514, len 255)
        corrupted, _ = inject_symbol_errors(corrupted, count=10, block_offset=514, block_length=255, seed=303)

        recovered = pipeline.unprotect(corrupted, test_key)
        return recovered.data == capacity_data

    suite.check("ECC: Multi-Block Distributed Noise Correction", check_ecc_multi_block_distributed_noise, "10 errors/block across 3 blocks")

    # =========================================================================
    # Section 5: Fail-Closed Security & Zero Plaintext Leakage
    # =========================================================================
    suite.section("5. Fail-Closed Security & Zero Plaintext Leakage (INV-03)")

    def check_fail_closed_tampered_ciphertext() -> bool:
        payload = SecretPayload(data=b"Confidential payload for tamper testing", media_type=MediaType.TEXT)
        # We manually perform steps to tamper the post-ECC decoded stream or re-encoded tampered ciphertext
        serialized = SecretSerializer.serialize(payload)
        crypto = pipeline.crypto_provider
        encrypted = crypto.encrypt(serialized, test_key, aad=test_aad)

        # Adversary flips 1 bit in ciphertext (offset 15) and validly RS-encodes it
        tampered_encrypted = flip_bit(encrypted, byte_pos=15, bit_pos=0)
        tampered_protected = pipeline.ecc_encoder.encode(tampered_encrypted)

        caught = False
        result = "LEAKED"
        try:
            result = pipeline.unprotect(tampered_protected, test_key, aad=test_aad)
        except AuthenticationError:
            caught = True
        return caught and result == "LEAKED"

    suite.check("Fail-Closed: Tampered Ciphertext Bit", check_fail_closed_tampered_ciphertext, "AuthenticationError, 0 plaintext")

    def check_fail_closed_tampered_tag() -> bool:
        payload = SecretPayload(data=b"Confidential payload for tag tamper testing", media_type=MediaType.TEXT)
        serialized = SecretSerializer.serialize(payload)
        crypto = pipeline.crypto_provider
        encrypted = crypto.encrypt(serialized, test_key, aad=test_aad)

        # Flip bit in authentication tag (last byte) and validly RS-encode it
        tampered_encrypted = flip_bit(encrypted, byte_pos=len(encrypted) - 1, bit_pos=0)
        tampered_protected = pipeline.ecc_encoder.encode(tampered_encrypted)

        caught = False
        result = "LEAKED"
        try:
            result = pipeline.unprotect(tampered_protected, test_key, aad=test_aad)
        except AuthenticationError:
            caught = True
        return caught and result == "LEAKED"

    suite.check("Fail-Closed: Tampered GHASH Tag Bit", check_fail_closed_tampered_tag, "AuthenticationError, 0 plaintext")

    def check_fail_closed_tampered_nonce() -> bool:
        payload = SecretPayload(data=b"Confidential payload for nonce tamper testing", media_type=MediaType.TEXT)
        serialized = SecretSerializer.serialize(payload)
        crypto = pipeline.crypto_provider
        encrypted = crypto.encrypt(serialized, test_key, aad=test_aad)

        # Flip bit in nonce (first byte) and validly RS-encode it
        tampered_encrypted = flip_bit(encrypted, byte_pos=0, bit_pos=0)
        tampered_protected = pipeline.ecc_encoder.encode(tampered_encrypted)

        caught = False
        result = "LEAKED"
        try:
            result = pipeline.unprotect(tampered_protected, test_key, aad=test_aad)
        except AuthenticationError:
            caught = True
        return caught and result == "LEAKED"

    suite.check("Fail-Closed: Tampered Nonce Bit", check_fail_closed_tampered_nonce, "AuthenticationError, 0 plaintext")

    def check_fail_closed_mismatched_aad() -> bool:
        payload = SecretPayload(data=b"Confidential payload for AAD testing", media_type=MediaType.TEXT)
        protected = pipeline.protect(payload, test_key, aad=test_aad)

        caught = False
        result = "LEAKED"
        wrong_aad = b"Altered-AAD-Context-Attack"
        try:
            result = pipeline.unprotect(protected, test_key, aad=wrong_aad)
        except AuthenticationError:
            caught = True
        return caught and result == "LEAKED"

    suite.check("Fail-Closed: Mismatched Context AAD", check_fail_closed_mismatched_aad, "AuthenticationError, 0 plaintext")

    def check_fail_closed_missing_expected_aad() -> bool:
        payload = SecretPayload(data=b"Confidential payload for missing AAD testing", media_type=MediaType.TEXT)
        protected = pipeline.protect(payload, test_key, aad=test_aad)

        caught = False
        result = "LEAKED"
        try:
            # Encrypted with test_aad, unprotect attempted with aad=None
            result = pipeline.unprotect(protected, test_key, aad=None)
        except AuthenticationError:
            caught = True
        return caught and result == "LEAKED"

    suite.check("Fail-Closed: Missing Expected AAD (aad=None)", check_fail_closed_missing_expected_aad, "AuthenticationError, 0 plaintext")

    def check_fail_closed_wrong_key() -> bool:
        payload = SecretPayload(data=b"Confidential payload for key mismatch testing", media_type=MediaType.TEXT)
        protected = pipeline.protect(payload, test_key, aad=test_aad)

        caught = False
        result = "LEAKED"
        try:
            result = pipeline.unprotect(protected, alt_key, aad=test_aad)
        except AuthenticationError:
            caught = True
        return caught and result == "LEAKED"

    suite.check("Fail-Closed: Wrong 256-Bit Decryption Key", check_fail_closed_wrong_key, "AuthenticationError, 0 plaintext")

    def check_fail_closed_uncorrectable_noise_exceeding_t() -> bool:
        payload = SecretPayload(data=b"Payload that encounters uncorrectable transmission noise", media_type=MediaType.TEXT)
        protected = pipeline.protect(payload, test_key, aad=test_aad)

        # Inject 18 symbol errors in first block (exceeds t=16)
        corrupted, _ = inject_symbol_errors(protected, count=18, block_offset=4, block_length=min(255, len(protected) - 4), seed=999)

        caught = False
        result = "LEAKED"
        try:
            # Must abort at ECC layer without executing crypto decryption
            result = pipeline.unprotect(corrupted, test_key, aad=test_aad)
        except ECCDecodeError:
            caught = True
        return caught and result == "LEAKED"

    suite.check("Fail-Closed: Uncorrectable Noise (> t Errors)", check_fail_closed_uncorrectable_noise_exceeding_t, "Halts at ECC with ECCDecodeError")

    def check_fail_closed_random_garbage_stream() -> bool:
        # Completely random byte stream presented to pipeline unprotect
        garbage = os.urandom(300)
        caught = False
        result = "LEAKED"
        try:
            result = pipeline.unprotect(garbage, test_key)
        except (ECCDecodeError, AuthenticationError, SerializationError):
            caught = True
        return caught and result == "LEAKED"

    suite.check("Fail-Closed: Completely Corrupted Bitstream", check_fail_closed_random_garbage_stream, "Halts fail-closed with 0 plaintext")

    # =========================================================================
    # Final Summary and Exit Code
    # =========================================================================
    return suite.summarize()


if __name__ == "__main__":
    sys.exit(main())
