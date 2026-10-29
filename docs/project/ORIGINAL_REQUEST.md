# Original User Request

## Initial Request — 2026-09-22T17:25:37Z

Implement Sprint 1 (Cryptographic Foundation & Payload) for PaladinMedia MVP 0, including Payload Serialization, AES-256-GCM protection, and Reed-Solomon error correction in Python, strictly adhering to the Architecture and System & Research specifications.

Working directory: e:/Paladin_media/
Integrity mode: demo

## Requirements

### R1. Core Implementations
Implement `SecretSerializer` (PDM-S1-005) in `src/core/serialization`, `CryptoProvider` using AES-256-GCM (PDM-S1-002, 003) in `src/security/crypto`, and `ECCEncoder` using Reed-Solomon (PDM-S1-004) in `src/security/ecc`. Use standard external Python libraries (e.g., `cryptography`, `reedsolo`).

### R2. Architectural Invariants
The data flow must strictly follow: Serialize -> Encrypt (AES-GCM) -> ECC (Reed-Solomon). Error correction operates on the protected payload (ciphertext).

### R3. Security Handling
AES-GCM authentication failures must be fail-closed. If tampering is detected, the process must reject the payload entirely and leak no plaintext.

## Verification Resources
The team should create a comprehensive `pytest` suite in `tests/unit` and `tests/integration` based on the T-001 through T-004 scenarios outlined in the Sprint 0 Test Strategy.

## Acceptance Criteria

### Unit and Integration Tests
- [ ] Pytest suite executes successfully and covers serialization, encryption, and ECC separately.
- [ ] Negative security tests prove that modified AES-GCM ciphertext or tags result in an exception and no plaintext return.
- [ ] Reed-Solomon tests evaluate success based on exact symbol-level correction, not just BER.

### End-to-End Pipeline
- [ ] A programmatic integration script successfully passes a secret through Serialization -> Encryption -> ECC and fully reverses the process to recover the exact original bytes.
