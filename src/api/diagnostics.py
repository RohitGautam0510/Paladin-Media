"""Diagnostic Data Management for PaladinMedia Steganography Inspection.

Provides structured, bounded diagnostic representations of genuine transform-domain
embedding positions, modified coefficients, carrier-level pixel differences, and
verifiable execution lifecycle states.
Exposes real pipeline telemetry without leaking encryption keys or plaintext.
"""

from __future__ import annotations

import collections
import copy
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("paladinmedia.diagnostics")

# Memory and serialization bounding constants
DEFAULT_MAX_ENTRIES: int = 20
DEFAULT_MAX_RECORD_BYTES: int = 512 * 1024       # 512 KB per record maximum serialized UTF-8 bytes
DEFAULT_MAX_TOTAL_BYTES: int = 4 * 1024 * 1024   # 4 MB aggregate store budget
MAX_ERROR_MESSAGE_LEN: int = 512                 # Maximum character length for error messages


class FrozenDict(dict):
    """Immutable dictionary subclass that prevents item assignment and in-place mutation.

    Provides JSON serialization parity with standard Python dicts while preventing
    external callers from mutating internal diagnostic data structures.
    """

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError(f"'{self.__class__.__name__}' object does not support item assignment")

    def __delitem__(self, key: Any) -> None:
        raise TypeError(f"'{self.__class__.__name__}' object does not support item deletion")

    def clear(self) -> None:
        raise TypeError(f"'{self.__class__.__name__}' object does not support mutation")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f"'{self.__class__.__name__}' object does not support mutation")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"'{self.__class__.__name__}' object does not support mutation")

    def popitem(self) -> Any:
        raise TypeError(f"'{self.__class__.__name__}' object does not support mutation")

    def setdefault(self, key: Any, default: Any = None) -> Any:
        raise TypeError(f"'{self.__class__.__name__}' object does not support mutation")

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: Any) -> FrozenDict:
        return FrozenDict({copy.deepcopy(k, memo): copy.deepcopy(v, memo) for k, v in self.items()})


def freeze_value(val: Any) -> Any:
    """Recursively converts mappings to FrozenDict, sequences to immutable tuples, and sets to frozenset."""
    if isinstance(val, (dict, FrozenDict)):
        return FrozenDict({k: freeze_value(v) for k, v in val.items()})
    elif isinstance(val, (list, tuple)):
        return tuple(freeze_value(x) for x in val)
    elif isinstance(val, (set, frozenset)):
        return frozenset(freeze_value(x) for x in val)
    return val


def unfreeze_value(val: Any) -> Any:
    """Recursively converts FrozenDict to standard dicts and tuples to lists for API consumers."""
    if isinstance(val, (dict, FrozenDict)):
        return {k: unfreeze_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [unfreeze_value(x) for x in val]
    elif isinstance(val, (set, frozenset)):
        return [unfreeze_value(x) for x in val]
    return val


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical UTF-8 JSON serializer using compact separators and sorted keys.

    Serves as the strict source of truth for all byte measurements across storage and streaming.
    """
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


@dataclass(frozen=True)
class RunDiagnostics:
    """Immutable diagnostic data model for an individual steganography pipeline execution.

    All coordinate collections are stored as immutable tuples of (row, col) pairs.
    External callers cannot mutate the internal stored structures.
    """

    run_id: str
    timestamp: float
    session_counter: int
    carrier_dims: Tuple[int, int]  # (width, height)
    stego_dims: Tuple[int, int]    # (width, height)
    subband_dim: Union[int, Tuple[int, int]]  # e.g. 256 or (sub_h, sub_w)
    header_bits_count: int        # 128 bits
    payload_bits_count: int       # protected payload bits
    total_embedded_bits: int      # header + payload
    changed_coefficients_count: int
    max_hh_diff: float
    embedded_positions: Tuple[Tuple[int, int], ...]
    changed_positions: Tuple[Tuple[int, int], ...]
    pixel_diff_metrics: Dict[str, Any]
    verification_status: str = "PENDING"  # PENDING | VERIFIED | FAILED | ABORTED | PRE_EMBED_FAILED
    exact_match: Optional[bool] = None     # None while pending, True if verified, False if failed
    coordinates_truncated: bool = False
    total_embedded_positions_count: Optional[int] = None
    total_changed_positions_count: Optional[int] = None
    completeness_status: str = "COMPLETE" # COMPLETE | PARTIAL | UNAVAILABLE | FAILED
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        # Enforce immutable tuples of coordinates
        object.__setattr__(
            self,
            "embedded_positions",
            tuple((int(p[0]), int(p[1])) for p in self.embedded_positions),
        )
        object.__setattr__(
            self,
            "changed_positions",
            tuple((int(p[0]), int(p[1])) for p in self.changed_positions),
        )

        # Deeply freeze pixel_diff_metrics and all nested dictionaries/lists/sets
        if self.pixel_diff_metrics:
            object.__setattr__(
                self,
                "pixel_diff_metrics",
                freeze_value(copy.deepcopy(dict(self.pixel_diff_metrics))),
            )
        else:
            object.__setattr__(self, "pixel_diff_metrics", FrozenDict())

        # Ensure carrier and stego dimensions are tuples of integers
        object.__setattr__(
            self,
            "carrier_dims",
            (int(self.carrier_dims[0]), int(self.carrier_dims[1])),
        )
        object.__setattr__(
            self,
            "stego_dims",
            (int(self.stego_dims[0]), int(self.stego_dims[1])),
        )
        if isinstance(self.subband_dim, (tuple, list)):
            object.__setattr__(
                self,
                "subband_dim",
                (int(self.subband_dim[0]), int(self.subband_dim[1])),
            )
        else:
            object.__setattr__(self, "subband_dim", int(self.subband_dim))

        # Bound error message length
        if self.error_message and len(self.error_message) > MAX_ERROR_MESSAGE_LEN:
            truncated_msg = self.error_message[:MAX_ERROR_MESSAGE_LEN - 15] + "... [truncated]"
            object.__setattr__(self, "error_message", truncated_msg)

        # Infer completeness_status if default
        if self.verification_status in ("FAILED", "PRE_EMBED_FAILED"):
            object.__setattr__(self, "completeness_status", "FAILED")
        elif self.coordinates_truncated:
            object.__setattr__(self, "completeness_status", "PARTIAL")
        elif not self.embedded_positions and not self.changed_positions and self.total_embedded_bits == 0:
            object.__setattr__(self, "completeness_status", "UNAVAILABLE")

    def to_dict(self) -> Dict[str, Any]:
        """Converts to a fresh, JSON-serializable dictionary matching the external API contract.

        Always produces fresh list and dict copies to prevent external mutation of internal state.
        """
        if isinstance(self.subband_dim, (tuple, list)):
            if self.subband_dim[0] == self.subband_dim[1]:
                sub_val: Any = int(self.subband_dim[0])
            else:
                sub_val = [int(self.subband_dim[0]), int(self.subband_dim[1])]
        else:
            sub_val = int(self.subband_dim)

        return {
            "run_id": str(self.run_id),
            "timestamp": float(self.timestamp),
            "session_counter": int(self.session_counter),
            "carrier_dims": [int(self.carrier_dims[0]), int(self.carrier_dims[1])],
            "stego_dims": [int(self.stego_dims[0]), int(self.stego_dims[1])],
            "subband_dim": sub_val,
            "header_bits_count": int(self.header_bits_count),
            "payload_bits_count": int(self.payload_bits_count),
            "total_embedded_bits": int(self.total_embedded_bits),
            "changed_coefficients_count": int(self.changed_coefficients_count),
            "max_hh_diff": round(float(self.max_hh_diff), 4),
            "embedded_positions": [[int(pos[0]), int(pos[1])] for pos in self.embedded_positions],
            "changed_positions": [[int(pos[0]), int(pos[1])] for pos in self.changed_positions],
            "pixel_diff_metrics": unfreeze_value(self.pixel_diff_metrics),
            "verification_status": str(self.verification_status),
            "exact_match": self.exact_match,
            "coordinates_truncated": bool(self.coordinates_truncated),
            "total_embedded_positions_count": int(
                self.total_embedded_positions_count
                if self.total_embedded_positions_count is not None
                else len(self.embedded_positions)
            ),
            "total_changed_positions_count": int(
                self.total_changed_positions_count
                if self.total_changed_positions_count is not None
                else len(self.changed_positions)
            ),
            "completeness_status": str(self.completeness_status),
            "error_message": self.error_message,
        }

    def canonical_bytes(self) -> bytes:
        """Returns the canonical UTF-8 serialized byte representation of this record."""
        return canonical_json_bytes(self.to_dict())

    def serialized_size(self) -> int:
        """Returns the exact UTF-8 serialized size in bytes."""
        return len(self.canonical_bytes())


def bound_diagnostic_record(
    run_id: str,
    timestamp: float,
    session_counter: int,
    carrier_dims: Tuple[int, int],
    stego_dims: Tuple[int, int],
    subband_dim: Union[int, Tuple[int, int]],
    header_bits_count: int,
    payload_bits_count: int,
    total_embedded_bits: int,
    changed_coefficients_count: int,
    max_hh_diff: float,
    embedded_positions: Sequence[Tuple[int, int]],
    changed_positions: Sequence[Tuple[int, int]],
    pixel_diff_metrics: Dict[str, Any],
    verification_status: str = "PENDING",
    exact_match: Optional[bool] = None,
    error_message: Optional[str] = None,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
    total_embedded_positions_count: Optional[int] = None,
    total_changed_positions_count: Optional[int] = None,
) -> RunDiagnostics:
    """Constructs a RunDiagnostics record guaranteed not to exceed max_record_bytes when serialized.

    If coordinate collections exceed the byte budget, they are truncated proportionally while
    preserving the truthful total counts and setting coordinates_truncated=True.
    """
    total_emb = (
        total_embedded_positions_count
        if total_embedded_positions_count is not None
        else (total_embedded_bits if total_embedded_bits > 0 else len(embedded_positions))
    )
    total_chg = (
        total_changed_positions_count
        if total_changed_positions_count is not None
        else (changed_coefficients_count if changed_coefficients_count > 0 else len(changed_positions))
    )

    # First, construct candidate with full coordinates
    candidate = RunDiagnostics(
        run_id=run_id,
        timestamp=timestamp,
        session_counter=session_counter,
        carrier_dims=carrier_dims,
        stego_dims=stego_dims,
        subband_dim=subband_dim,
        header_bits_count=header_bits_count,
        payload_bits_count=payload_bits_count,
        total_embedded_bits=total_embedded_bits,
        changed_coefficients_count=changed_coefficients_count,
        max_hh_diff=max_hh_diff,
        embedded_positions=tuple((int(p[0]), int(p[1])) for p in embedded_positions),
        changed_positions=tuple((int(p[0]), int(p[1])) for p in changed_positions),
        pixel_diff_metrics=pixel_diff_metrics,
        verification_status=verification_status,
        exact_match=exact_match,
        coordinates_truncated=False,
        total_embedded_positions_count=total_emb,
        total_changed_positions_count=total_chg,
        completeness_status="COMPLETE",
        error_message=error_message,
    )

    if candidate.serialized_size() <= max_record_bytes:
        return candidate

    # Truncate coordinates to fit max_record_bytes
    # Base dictionary without coordinates to measure fixed metadata overhead
    base_dict = candidate.to_dict()
    base_dict["embedded_positions"] = []
    base_dict["changed_positions"] = []
    base_dict["coordinates_truncated"] = True
    base_dict["completeness_status"] = "PARTIAL"
    base_size = len(canonical_json_bytes(base_dict))

    current_error = candidate.error_message
    current_metrics = candidate.pixel_diff_metrics

    # If base metadata alone consumes or exceeds the budget, trim error message and metrics
    if base_size >= max_record_bytes:
        if current_error:
            excess = base_size - max_record_bytes + 25
            allowed_len = max(5, len(current_error) - excess)
            current_error = current_error[:allowed_len] + "... [truncated]"
            base_dict["error_message"] = current_error
            base_size = len(canonical_json_bytes(base_dict))

        if base_size > max_record_bytes:
            current_metrics = {}
            base_dict["pixel_diff_metrics"] = {}
            base_size = len(canonical_json_bytes(base_dict))

        while base_size > max_record_bytes and current_error and len(current_error) > 16:
            current_error = current_error[:max(0, len(current_error) - 20)] + "... [truncated]"
            base_dict["error_message"] = current_error
            base_size = len(canonical_json_bytes(base_dict))

        if base_size > max_record_bytes:
            current_error = None
            base_dict["error_message"] = None
            base_size = len(canonical_json_bytes(base_dict))

        return RunDiagnostics(
            run_id=run_id,
            timestamp=timestamp,
            session_counter=session_counter,
            carrier_dims=carrier_dims,
            stego_dims=stego_dims,
            subband_dim=subband_dim,
            header_bits_count=header_bits_count,
            payload_bits_count=payload_bits_count,
            total_embedded_bits=total_embedded_bits,
            changed_coefficients_count=changed_coefficients_count,
            max_hh_diff=max_hh_diff,
            embedded_positions=(),
            changed_positions=(),
            pixel_diff_metrics=current_metrics,
            verification_status=verification_status,
            exact_match=exact_match,
            coordinates_truncated=True,
            total_embedded_positions_count=total_emb,
            total_changed_positions_count=total_chg,
            completeness_status="PARTIAL",
            error_message=current_error,
        )

    available_bytes = max_record_bytes - base_size
    # Each coordinate pair `[r,c],` takes between 7 and 11 bytes in JSON
    max_coords = max(0, available_bytes // 11)

    total_coords = len(embedded_positions) + len(changed_positions)
    if total_coords > 0:
        allow_emb = min(len(embedded_positions), int(max_coords * (len(embedded_positions) / total_coords)))
        allow_chg = min(len(changed_positions), max_coords - allow_emb)
    else:
        allow_emb, allow_chg = 0, 0

    curr_emb = tuple((int(p[0]), int(p[1])) for p in embedded_positions[:allow_emb])
    curr_chg = tuple((int(p[0]), int(p[1])) for p in changed_positions[:allow_chg])

    # Iterative refinement if necessary to guarantee exact budget ceiling
    while (allow_emb > 0 or allow_chg > 0):
        test_rec = RunDiagnostics(
            run_id=run_id,
            timestamp=timestamp,
            session_counter=session_counter,
            carrier_dims=carrier_dims,
            stego_dims=stego_dims,
            subband_dim=subband_dim,
            header_bits_count=header_bits_count,
            payload_bits_count=payload_bits_count,
            total_embedded_bits=total_embedded_bits,
            changed_coefficients_count=changed_coefficients_count,
            max_hh_diff=max_hh_diff,
            embedded_positions=curr_emb,
            changed_positions=curr_chg,
            pixel_diff_metrics=pixel_diff_metrics,
            verification_status=verification_status,
            exact_match=exact_match,
            coordinates_truncated=True,
            total_embedded_positions_count=total_emb,
            total_changed_positions_count=total_chg,
            completeness_status="PARTIAL",
            error_message=error_message,
        )
        if test_rec.serialized_size() <= max_record_bytes:
            return test_rec

        # Trim slightly
        if allow_emb > allow_chg and allow_emb > 0:
            allow_emb = max(0, allow_emb - max(1, allow_emb // 10))
        elif allow_chg > 0:
            allow_chg = max(0, allow_chg - max(1, allow_chg // 10))
        else:
            allow_emb = 0
            allow_chg = 0

        curr_emb = tuple((int(p[0]), int(p[1])) for p in embedded_positions[:allow_emb])
        curr_chg = tuple((int(p[0]), int(p[1])) for p in changed_positions[:allow_chg])

    return RunDiagnostics(
        run_id=run_id,
        timestamp=timestamp,
        session_counter=session_counter,
        carrier_dims=carrier_dims,
        stego_dims=stego_dims,
        subband_dim=subband_dim,
        header_bits_count=header_bits_count,
        payload_bits_count=payload_bits_count,
        total_embedded_bits=total_embedded_bits,
        changed_coefficients_count=changed_coefficients_count,
        max_hh_diff=max_hh_diff,
        embedded_positions=(),
        changed_positions=(),
        pixel_diff_metrics=pixel_diff_metrics,
        verification_status=verification_status,
        exact_match=exact_match,
        coordinates_truncated=True,
        total_embedded_positions_count=total_emb,
        total_changed_positions_count=total_chg,
        completeness_status="PARTIAL",
        error_message=error_message,
    )


class DiagnosticsStore:
    """Thread-safe bounded in-memory store for execution run diagnostics.

    Guarantees:
    1. Maximum entry count bounded by `max_entries` (FIFO eviction).
    2. Maximum serialized UTF-8 bytes per record bounded by `max_record_bytes`.
    3. Maximum aggregate serialized UTF-8 bytes retained bounded by `max_total_bytes`.
    4. Lifecycle updates re-evaluate budgets and never exceed the configured ceiling.
    5. Immutability: Stored records cannot be modified externally.
    6. Thread-safety: Synchronized via threading.Lock.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self.max_entries = max_entries
        self.max_record_bytes = max_record_bytes
        self.max_total_bytes = max_total_bytes
        self._lock = threading.Lock()
        self._store: collections.OrderedDict[str, RunDiagnostics] = collections.OrderedDict()
        self._entry_sizes: Dict[str, int] = {}
        self._current_total_bytes: int = 0
        self._latest_run_id: Optional[str] = None

    def create_bounded_record(
        self,
        run_id: str,
        timestamp: float,
        session_counter: int,
        carrier_dims: Tuple[int, int],
        stego_dims: Tuple[int, int],
        subband_dim: Union[int, Tuple[int, int]],
        header_bits_count: int,
        payload_bits_count: int,
        total_embedded_bits: int,
        changed_coefficients_count: int,
        max_hh_diff: float,
        embedded_positions: Sequence[Tuple[int, int]],
        changed_positions: Sequence[Tuple[int, int]],
        pixel_diff_metrics: Dict[str, Any],
        verification_status: str = "PENDING",
        exact_match: Optional[bool] = None,
        error_message: Optional[str] = None,
        total_embedded_positions_count: Optional[int] = None,
        total_changed_positions_count: Optional[int] = None,
    ) -> RunDiagnostics:
        """Helper to construct a bounded RunDiagnostics record matching this store's limits."""
        return bound_diagnostic_record(
            run_id=run_id,
            timestamp=timestamp,
            session_counter=session_counter,
            carrier_dims=carrier_dims,
            stego_dims=stego_dims,
            subband_dim=subband_dim,
            header_bits_count=header_bits_count,
            payload_bits_count=payload_bits_count,
            total_embedded_bits=total_embedded_bits,
            changed_coefficients_count=changed_coefficients_count,
            max_hh_diff=max_hh_diff,
            embedded_positions=embedded_positions,
            changed_positions=changed_positions,
            pixel_diff_metrics=pixel_diff_metrics,
            verification_status=verification_status,
            exact_match=exact_match,
            error_message=error_message,
            max_record_bytes=self.max_record_bytes,
            total_embedded_positions_count=total_embedded_positions_count,
            total_changed_positions_count=total_changed_positions_count,
        )

    def _bound_record_to_size(self, diag: RunDiagnostics, size_limit: int) -> RunDiagnostics:
        """Enforces a specific byte budget ceiling on an existing record."""
        return bound_diagnostic_record(
            run_id=diag.run_id,
            timestamp=diag.timestamp,
            session_counter=diag.session_counter,
            carrier_dims=diag.carrier_dims,
            stego_dims=diag.stego_dims,
            subband_dim=diag.subband_dim,
            header_bits_count=diag.header_bits_count,
            payload_bits_count=diag.payload_bits_count,
            total_embedded_bits=diag.total_embedded_bits,
            changed_coefficients_count=diag.changed_coefficients_count,
            max_hh_diff=diag.max_hh_diff,
            embedded_positions=diag.embedded_positions,
            changed_positions=diag.changed_positions,
            pixel_diff_metrics=diag.pixel_diff_metrics,
            verification_status=diag.verification_status,
            exact_match=diag.exact_match,
            error_message=diag.error_message,
            max_record_bytes=size_limit,
            total_embedded_positions_count=diag.total_embedded_positions_count,
            total_changed_positions_count=diag.total_changed_positions_count,
        )

    def _bound_record(self, diag: RunDiagnostics) -> RunDiagnostics:
        """Enforces per-record byte budget on an existing record."""
        if diag.serialized_size() <= self.max_record_bytes:
            return diag
        return self._bound_record_to_size(diag, self.max_record_bytes)

    def _reconcile_total_bytes(self) -> None:
        """Strictly synchronizes the internal counter with the actual sum of retained serialized record sizes."""
        self._entry_sizes = {rid: rec.serialized_size() for rid, rec in self._store.items()}
        self._current_total_bytes = sum(self._entry_sizes.values())

    def _evict_to_fit(self, incoming_id: str, incoming_size: int) -> None:
        """Evicts oldest entries (excluding incoming_id) until entry count and aggregate byte bounds are met."""
        while self._store:
            other_keys = [k for k in self._store if k != incoming_id]
            if not other_keys:
                break
            other_bytes = sum(self._entry_sizes[k] for k in other_keys)
            if (len(other_keys) + 1 <= self.max_entries and (other_bytes + incoming_size <= self.max_total_bytes)):
                break
            evict_key = other_keys[0]
            del self._store[evict_key]
            self._entry_sizes.pop(evict_key, 0)
            if self._latest_run_id == evict_key:
                self._latest_run_id = next(reversed(self._store)) if self._store else None

    def store(self, diagnostics: RunDiagnostics) -> None:
        """Stores a run diagnostic record, enforcing per-record and aggregate byte bounds.

        Guarantees defensive copying on insertion: constructs an independent internal
        representation with recursively frozen nested structures, retaining zero
        caller-owned mutable objects.
        """
        # Construct completely independent internal representation
        clean_record = RunDiagnostics(
            run_id=str(diagnostics.run_id),
            timestamp=float(diagnostics.timestamp),
            session_counter=int(diagnostics.session_counter),
            carrier_dims=(int(diagnostics.carrier_dims[0]), int(diagnostics.carrier_dims[1])),
            stego_dims=(int(diagnostics.stego_dims[0]), int(diagnostics.stego_dims[1])),
            subband_dim=(
                (int(diagnostics.subband_dim[0]), int(diagnostics.subband_dim[1]))
                if isinstance(diagnostics.subband_dim, (tuple, list))
                else int(diagnostics.subband_dim)
            ),
            header_bits_count=int(diagnostics.header_bits_count),
            payload_bits_count=int(diagnostics.payload_bits_count),
            total_embedded_bits=int(diagnostics.total_embedded_bits),
            changed_coefficients_count=int(diagnostics.changed_coefficients_count),
            max_hh_diff=float(diagnostics.max_hh_diff),
            embedded_positions=copy.deepcopy(diagnostics.embedded_positions),
            changed_positions=copy.deepcopy(diagnostics.changed_positions),
            pixel_diff_metrics=copy.deepcopy(diagnostics.pixel_diff_metrics),
            verification_status=str(diagnostics.verification_status),
            exact_match=diagnostics.exact_match,
            coordinates_truncated=bool(diagnostics.coordinates_truncated),
            total_embedded_positions_count=diagnostics.total_embedded_positions_count,
            total_changed_positions_count=diagnostics.total_changed_positions_count,
            completeness_status=str(diagnostics.completeness_status),
            error_message=str(diagnostics.error_message) if diagnostics.error_message is not None else None,
        )

        bounded = self._bound_record(clean_record)
        if bounded.serialized_size() > self.max_total_bytes:
            bounded = self._bound_record_to_size(bounded, self.max_total_bytes)
        size = bounded.serialized_size()

        with self._lock:
            self._evict_to_fit(bounded.run_id, size)
            if size > self.max_total_bytes:
                bounded = self._bound_record_to_size(bounded, self.max_total_bytes)
                size = bounded.serialized_size()

            self._store[bounded.run_id] = bounded
            self._entry_sizes[bounded.run_id] = size
            self._latest_run_id = bounded.run_id
            self._reconcile_total_bytes()

    def update_verification(
        self,
        run_id: str,
        verification_status: str,
        exact_match: Optional[bool] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Thread-safe update of verification status enforcing strict per-record and total budgets."""
        with self._lock:
            existing = self._store.get(run_id)
            if existing is None:
                return False

            # Bound error message length
            safe_error = error_message if error_message is not None else existing.error_message
            if safe_error and len(safe_error) > MAX_ERROR_MESSAGE_LEN:
                safe_error = safe_error[:MAX_ERROR_MESSAGE_LEN - 15] + "... [truncated]"

            candidate = RunDiagnostics(
                run_id=existing.run_id,
                timestamp=existing.timestamp,
                session_counter=existing.session_counter,
                carrier_dims=existing.carrier_dims,
                stego_dims=existing.stego_dims,
                subband_dim=existing.subband_dim,
                header_bits_count=existing.header_bits_count,
                payload_bits_count=existing.payload_bits_count,
                total_embedded_bits=existing.total_embedded_bits,
                changed_coefficients_count=existing.changed_coefficients_count,
                max_hh_diff=existing.max_hh_diff,
                embedded_positions=existing.embedded_positions,
                changed_positions=existing.changed_positions,
                pixel_diff_metrics=existing.pixel_diff_metrics,
                verification_status=verification_status,
                exact_match=exact_match if exact_match is not None else existing.exact_match,
                coordinates_truncated=existing.coordinates_truncated,
                total_embedded_positions_count=existing.total_embedded_positions_count,
                total_changed_positions_count=existing.total_changed_positions_count,
                completeness_status=existing.completeness_status,
                error_message=safe_error,
            )

            bounded = self._bound_record(candidate)
            if bounded.serialized_size() > self.max_total_bytes:
                bounded = self._bound_record_to_size(bounded, self.max_total_bytes)
            new_size = bounded.serialized_size()

            # Evict other older records if needed to satisfy max_total_bytes
            self._evict_to_fit(run_id, new_size)
            if new_size > self.max_total_bytes:
                bounded = self._bound_record_to_size(bounded, self.max_total_bytes)
                new_size = bounded.serialized_size()

            self._store[run_id] = bounded
            self._entry_sizes[run_id] = new_size
            self._reconcile_total_bytes()
            return True

    def update_pixel_diff(self, run_id: str, pixel_diff_metrics: Dict[str, Any]) -> bool:
        """Thread-safe update of pixel difference metrics for an existing run."""
        with self._lock:
            existing = self._store.get(run_id)
            if existing is None:
                return False

            candidate = RunDiagnostics(
                run_id=existing.run_id,
                timestamp=existing.timestamp,
                session_counter=existing.session_counter,
                carrier_dims=existing.carrier_dims,
                stego_dims=existing.stego_dims,
                subband_dim=existing.subband_dim,
                header_bits_count=existing.header_bits_count,
                payload_bits_count=existing.payload_bits_count,
                total_embedded_bits=existing.total_embedded_bits,
                changed_coefficients_count=existing.changed_coefficients_count,
                max_hh_diff=existing.max_hh_diff,
                embedded_positions=existing.embedded_positions,
                changed_positions=existing.changed_positions,
                pixel_diff_metrics=pixel_diff_metrics,
                verification_status=existing.verification_status,
                exact_match=existing.exact_match,
                coordinates_truncated=existing.coordinates_truncated,
                total_embedded_positions_count=existing.total_embedded_positions_count,
                total_changed_positions_count=existing.total_changed_positions_count,
                completeness_status=existing.completeness_status,
                error_message=existing.error_message,
            )

            bounded = self._bound_record(candidate)
            if bounded.serialized_size() > self.max_total_bytes:
                bounded = self._bound_record_to_size(bounded, self.max_total_bytes)
            new_size = bounded.serialized_size()

            self._evict_to_fit(run_id, new_size)
            if new_size > self.max_total_bytes:
                bounded = self._bound_record_to_size(bounded, self.max_total_bytes)
                new_size = bounded.serialized_size()

            self._store[run_id] = bounded
            self._entry_sizes[run_id] = new_size
            self._reconcile_total_bytes()
            return True

    def get(self, run_id: str) -> Optional[RunDiagnostics]:
        """Returns the requested run diagnostics record (defensive deep copy)."""
        with self._lock:
            diag = self._store.get(run_id)
            if diag is None:
                return None
            return copy.deepcopy(diag)

    def get_latest(self) -> Optional[RunDiagnostics]:
        """Returns the most recently registered run diagnostics record (defensive deep copy)."""
        with self._lock:
            if self._latest_run_id and self._latest_run_id in self._store:
                return copy.deepcopy(self._store[self._latest_run_id])
            if self._store:
                return copy.deepcopy(next(reversed(self._store.values())))
            return None

    def clear(self) -> None:
        """Clears all stored diagnostic records."""
        with self._lock:
            self._store.clear()
            self._entry_sizes.clear()
            self._current_total_bytes = 0
            self._latest_run_id = None

    def get_total_bytes(self) -> int:
        """Returns the exact current total serialized bytes of all records in the store."""
        with self._lock:
            return self._current_total_bytes

    def get_entry_count(self) -> int:
        """Returns the number of records in the store."""
        with self._lock:
            return len(self._store)


# Global singleton instance
diagnostics_store = DiagnosticsStore(
    max_entries=DEFAULT_MAX_ENTRIES,
    max_record_bytes=DEFAULT_MAX_RECORD_BYTES,
    max_total_bytes=DEFAULT_MAX_TOTAL_BYTES,
)
