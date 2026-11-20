"""
Cryptographic Trust Store and Identity Lifecycle Management for Post-Quantum Operations.

Enforces:
  - Identity registration and public key pinning
  - Preimage-resistant lookup by 32-byte SHA-256 fingerprints
  - Fail-closed expiration and revocation verification
  - Monotonic session counter tracking for replay suppression
  - Clock skew / timestamp freshness window validation
  - Durable persistence across process restarts
  - Two-phase counter verification: check before decryption, commit only after authenticated verification
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Iterator, Optional, Tuple, Union

from src.security.pqc.exceptions import (
    PQCIdentityError,
    PQCKeyExpiredError,
    PQCKeyRevokedError,
    PQCReplayError,
)
from src.security.pqc.models import PQCIdentity


@contextmanager
def _process_file_lock(lock_path: Path) -> Iterator[None]:
    """Cross-platform advisory process file lock for multi-worker synchronization."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import msvcrt
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT | os.O_BINARY)
        try:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(fd)
    else:
        import fcntl
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


class TrustStore:
    """Thread-safe and process-safe cryptographic trust store for PaladinMedia PQC identities."""

    DEFAULT_MAX_CLOCK_SKEW_MS: int = 300_000  # 5 minutes in milliseconds

    def __init__(
        self,
        max_clock_skew_ms: int = DEFAULT_MAX_CLOCK_SKEW_MS,
        storage_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._max_clock_skew_ms = max_clock_skew_ms
        if storage_path is None:
            storage_path = os.environ.get("PALADIN_TRUST_STORE_PATH")
        self._storage_path: Optional[Path] = Path(storage_path) if storage_path else None
        self._lock_path: Optional[Path] = self._storage_path.with_suffix(".lock") if self._storage_path else None
        self._identities_by_id: Dict[str, PQCIdentity] = {}
        self._kem_fp_index: Dict[bytes, str] = {}
        self._dsa_fp_index: Dict[bytes, str] = {}
        self._inbound_last_counter: Dict[str, int] = {}
        self._outbound_last_counter: Dict[str, int] = {}

        if self._storage_path and self._storage_path.exists():
            self._load_from_disk()

    @contextmanager
    def _file_sync_lock(self) -> Iterator[None]:
        """Acquires the advisory process file lock if a storage path is configured."""
        if self._lock_path is not None:
            with _process_file_lock(self._lock_path):
                yield
        else:
            yield

    def _save_to_disk_unlocked(self) -> None:
        """Saves state to disk assuming advisory process lock is already held."""
        if not self._storage_path:
            return
        data: Dict[str, Any] = {
            "inbound_counters": self._inbound_last_counter,
            "outbound_counters": self._outbound_last_counter,
            "identities": {},
        }
        for identity_id, id_obj in self._identities_by_id.items():
            data["identities"][identity_id] = {
                "name": id_obj.name,
                "kem_public_key_hex": id_obj.kem_public_key.hex(),
                "dsa_public_key_hex": id_obj.dsa_public_key.hex(),
                "created_at": id_obj.created_at,
                "expires_at": id_obj.expires_at,
                "is_revoked": id_obj.is_revoked,
                "metadata": id_obj.metadata,
            }

        tmp_name = f"{self._storage_path.stem}_{os.getpid()}_{threading.get_ident()}_{time.time_ns()}.tmp"
        tmp_path = self._storage_path.parent / tmp_name
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self._storage_path)

    def _save_to_disk(self) -> None:
        """Atomically persists trust store state to disk under advisory lock."""
        with self._file_sync_lock():
            self._save_to_disk_unlocked()

    def _load_from_disk_unlocked(self) -> None:
        """Loads trust store state assuming advisory process lock is already held."""
        if not self._storage_path or not self._storage_path.exists():
            return
        with open(self._storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._inbound_last_counter = data.get("inbound_counters", {})
        self._outbound_last_counter = data.get("outbound_counters", {})
        identities_dict = data.get("identities", {})
        self._identities_by_id.clear()
        self._kem_fp_index.clear()
        self._dsa_fp_index.clear()
        for identity_id, info in identities_dict.items():
            identity = PQCIdentity(
                identity_id=identity_id,
                name=info["name"],
                kem_public_key=bytes.fromhex(info["kem_public_key_hex"]),
                dsa_public_key=bytes.fromhex(info["dsa_public_key_hex"]),
                created_at=info.get("created_at", time.time()),
                expires_at=info.get("expires_at", time.time() + 86400 * 365),
                is_revoked=info.get("is_revoked", False),
                metadata=info.get("metadata", {}),
            )
            self._identities_by_id[identity_id] = identity
            self._kem_fp_index[identity.kem_fingerprint] = identity_id
            self._dsa_fp_index[identity.dsa_fingerprint] = identity_id

    def _load_from_disk(self) -> None:
        """Loads trust store state from disk under advisory file lock."""
        with self._file_sync_lock():
            self._load_from_disk_unlocked()

    def register_identity(self, identity: PQCIdentity) -> None:
        """Registers a verified PQCIdentity in the trust store under atomic lock."""
        with self._lock:
            identity.validate()
            with self._file_sync_lock():
                if self._storage_path and self._storage_path.exists():
                    self._load_from_disk_unlocked()
                self._identities_by_id[identity.identity_id] = identity
                self._kem_fp_index[identity.kem_fingerprint] = identity.identity_id
                self._dsa_fp_index[identity.dsa_fingerprint] = identity.identity_id
                if identity.identity_id not in self._inbound_last_counter:
                    self._inbound_last_counter[identity.identity_id] = 0
                if identity.identity_id not in self._outbound_last_counter:
                    self._outbound_last_counter[identity.identity_id] = 0
                self._save_to_disk_unlocked()

    def revoke_identity(self, identity_id: str) -> None:
        """Marks an identity as revoked under atomic lock."""
        with self._lock:
            with self._file_sync_lock():
                if self._storage_path and self._storage_path.exists():
                    self._load_from_disk_unlocked()
                if identity_id in self._identities_by_id:
                    identity = self._identities_by_id[identity_id]
                    updated = PQCIdentity(
                        identity_id=identity.identity_id,
                        name=identity.name,
                        kem_public_key=identity.kem_public_key,
                        dsa_public_key=identity.dsa_public_key,
                        created_at=identity.created_at,
                        expires_at=identity.expires_at,
                        is_revoked=True,
                        metadata=identity.metadata,
                    )
                    self._identities_by_id[identity_id] = updated
                    self._save_to_disk_unlocked()

    def get_identity(self, identity_id: str) -> Optional[PQCIdentity]:
        """Retrieves an identity by its identifier."""
        with self._lock:
            return self._identities_by_id.get(identity_id)

    def lookup_by_kem_fingerprint(self, kem_fp: bytes, check_validity: bool = True) -> PQCIdentity:
        """Resolves recipient identity by 32-byte ML-KEM-768 public key fingerprint."""
        with self._lock:
            identity_id = self._kem_fp_index.get(kem_fp)
            if not identity_id:
                raise PQCIdentityError(f"No identity found matching KEM fingerprint {kem_fp.hex()[:16]}...")
            identity = self._identities_by_id[identity_id]
            if check_validity:
                identity.validate()
            return identity

    def lookup_by_dsa_fingerprint(self, dsa_fp: bytes, check_validity: bool = True) -> PQCIdentity:
        """Resolves sender identity by 32-byte ML-DSA-65 public key fingerprint."""
        with self._lock:
            identity_id = self._dsa_fp_index.get(dsa_fp)
            if not identity_id:
                raise PQCIdentityError(f"No identity found matching DSA fingerprint {dsa_fp.hex()[:16]}...")
            identity = self._identities_by_id[identity_id]
            if check_validity:
                identity.validate()
            return identity

    def check_inbound_session_counter(self, sender_id: str, session_counter: int) -> None:
        """Verifies session counter is strictly greater than last processed counter without mutating state."""
        with self._lock:
            with self._file_sync_lock():
                if self._storage_path and self._storage_path.exists():
                    self._load_from_disk_unlocked()
                last = self._inbound_last_counter.get(sender_id, 0)
                if session_counter <= last:
                    raise PQCReplayError(
                        f"Session counter replay or regression detected for sender '{sender_id}': received {session_counter} <= last {last}"
                    )

    def commit_inbound_session_counter(self, sender_id: str, session_counter: int) -> None:
        """
        Atomically updates the sender's inbound counter under process file lock.
        MUST only be called after full cryptographic authentication and AEAD tag verification succeed!
        """
        with self._lock:
            with self._file_sync_lock():
                if self._storage_path and self._storage_path.exists():
                    self._load_from_disk_unlocked()
                last = self._inbound_last_counter.get(sender_id, 0)
                if session_counter <= last:
                    raise PQCReplayError(
                        f"Session counter replay or regression detected for sender '{sender_id}': received {session_counter} <= last {last}"
                    )
                self._inbound_last_counter[sender_id] = session_counter
                self._save_to_disk_unlocked()

    def verify_and_update_session_counter(self, sender_id: str, session_counter: int) -> None:
        """Legacy helper: checks and immediately commits counter (for trusted/atomic pipelines)."""
        with self._lock:
            self.check_inbound_session_counter(sender_id, session_counter)
            self.commit_inbound_session_counter(sender_id, session_counter)

    def get_next_session_counter(self, sender_id: str) -> int:
        """Allocates and increments the next monotonic session counter for an outbound message under process lock."""
        with self._lock:
            with self._file_sync_lock():
                if self._storage_path and self._storage_path.exists():
                    self._load_from_disk_unlocked()
                current = self._outbound_last_counter.get(sender_id, 0)
                nxt = current + 1
                self._outbound_last_counter[sender_id] = nxt
                self._save_to_disk_unlocked()
                return nxt

    def verify_timestamp_freshness(self, session_timestamp_ms: int, now_ms: Optional[int] = None) -> None:
        """Enforces that session timestamp falls within the allowed clock-skew window."""
        current = now_ms if now_ms is not None else int(time.time() * 1000)
        skew = abs(current - session_timestamp_ms)
        if skew > self._max_clock_skew_ms:
            raise PQCReplayError(
                f"Session timestamp skew {skew} ms exceeds allowed maximum {self._max_clock_skew_ms} ms"
            )

    def clear(self) -> None:
        """Clears all registered identities and state (used for testing)."""
        with self._lock:
            self._identities_by_id.clear()
            self._kem_fp_index.clear()
            self._dsa_fp_index.clear()
            self._inbound_last_counter.clear()
            self._outbound_last_counter.clear()
            if self._storage_path and self._storage_path.exists():
                try:
                    self._storage_path.unlink()
                except OSError:
                    pass
            if self._lock_path and self._lock_path.exists():
                try:
                    self._lock_path.unlink()
                except OSError:
                    pass


# Default process-level trust store
default_trust_store = TrustStore()
