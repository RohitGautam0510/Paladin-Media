#!/usr/bin/env python3
"""Lightweight consistency check ensuring Phase 5B parity element counts, percentages, and gate reasons

agree exactly between docs/experiments/phase5b_results.json and the audit report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JSON_PATH = REPO_ROOT / "docs" / "experiments" / "phase5b_results.json"
REPORT_PATH = REPO_ROOT / "docs" / "audits" / "PHASE5B_ONNX_INTEGRATION_AND_ROBUSTNESS_REPORT.md"


def check_consistency() -> bool:
    print(f"[CHECK] Loading JSON source of truth: {JSON_PATH}")
    if not JSON_PATH.exists():
        print(f"ERROR: JSON file not found at {JSON_PATH}")
        return False

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[CHECK] Loading Audit Report: {REPORT_PATH}")
    if not REPORT_PATH.exists():
        print(f"ERROR: Report file not found at {REPORT_PATH}")
        return False

    report_text = REPORT_PATH.read_text(encoding="utf-8")

    ws_a = data.get("workstream_a_parity", {})
    summary = ws_a.get("validation_scan_summary", {})
    elem_dist = summary.get("element_level_distributions", {})

    # 1. Element-level distribution verification
    total = elem_dist.get("total_elements_evaluated")
    c_1e5 = elem_dist.get("elements_within_strict_1e5_count")
    c_2e5 = elem_dist.get("elements_within_adjusted_2e5_count")
    c_5e5 = elem_dist.get("elements_within_5e5_count")
    c_1e4 = elem_dist.get("elements_within_1e4_count")

    expected_counts = {
        "total": 78643200,
        "1e-5": 77827472,
        "2e-5": 78640358,
        "5e-5": 78643193,
        "1e-4": 78643200,
    }

    assert total == expected_counts["total"], f"Expected total {expected_counts['total']}, got {total}"
    assert c_1e5 == expected_counts["1e-5"], f"Expected 1e-5 {expected_counts['1e-5']}, got {c_1e5}"
    assert c_2e5 == expected_counts["2e-5"], f"Expected 2e-5 {expected_counts['2e-5']}, got {c_2e5}"
    assert c_5e5 == expected_counts["5e-5"], f"Expected 5e-5 {expected_counts['5e-5']}, got {c_5e5}"
    assert c_1e4 == expected_counts["1e-4"], f"Expected 1e-4 {expected_counts['1e-4']}, got {c_1e4}"

    pct_1e5_calc = f"{c_1e5 / total * 100.0:.4f}"
    pct_2e5_calc = f"{c_2e5 / total * 100.0:.4f}"
    pct_5e5_calc = f"{c_5e5 / total * 100.0:.6f}"
    pct_1e4_calc = f"{c_1e4 / total * 100.0:.1f}"

    print(f"  - Total elements: {total:,}")
    print(f"  - Absolute error <= 1e-5: {c_1e5:,} / {total:,} ({pct_1e5_calc}%)")
    print(f"  - Absolute error <= 2e-5: {c_2e5:,} / {total:,} ({pct_2e5_calc}%)")
    print(f"  - Absolute error <= 5e-5: {c_5e5:,} / {total:,} ({pct_5e5_calc}%)")
    print(f"  - Absolute error <= 1e-4: {c_1e4:,} / {total:,} ({pct_1e4_calc}%)")

    # 2. Verify JSON percentages match calculated values
    assert abs(elem_dist.get("elements_within_strict_1e5_pct") - 98.9627) < 1e-3
    assert abs(elem_dist.get("elements_within_adjusted_2e5_pct") - 99.9964) < 1e-3
    assert abs(elem_dist.get("elements_within_5e5_pct") - 99.999991) < 1e-5
    assert abs(elem_dist.get("elements_within_1e4_pct") - 100.0) < 1e-3

    # 3. Verify Report contains exact counts and percentages
    required_report_snippets = [
        "77,827,472",
        "98.9627%",
        "78,640,358",
        "99.9964%",
        "78,643,193",
        "99.999991%",
        "78,643,200",
        "100.0%",
        "2.9211",
    ]

    for snippet in required_report_snippets:
        if snippet not in report_text:
            print(f"ERROR: Report missing required snippet: '{snippet}'")
            return False
        print(f"  [OK] Report contains verified snippet: '{snippet}'")

    # 4. Verify stale rounded figures are eliminated from the report
    stale_snippets = [
        "78,643,184",  # old stale count
        "99.99998%",   # old stale percentage
        "77,823,803",  # old count
        "98.9581%",    # old percentage
        "2.9190",      # old mean
    ]
    for stale in stale_snippets:
        if stale in report_text:
            print(f"ERROR: Report still contains stale snippet: '{stale}'")
            return False
        print(f"  [OK] Confirmed stale snippet absent: '{stale}'")

    # 5. Verify JSON gate_reason accurately distinguishes continuous errors, direct same-input, and separate roundtrip
    gate_reason = ws_a.get("gate_reason", "")
    print(f"[CHECK] Verifying gate_reason text:\n  '{gate_reason}'")

    direct_extract = summary.get("direct_extraction_same_input_parity", {})
    roundtrip = summary.get("separate_roundtrip_parity", {})

    assert direct_extract.get("total_bit_errors") == 5
    assert direct_extract.get("total_bits_evaluated") == 6553600
    assert roundtrip.get("total_bit_errors") == 0
    assert roundtrip.get("total_bits_evaluated") == 6553600

    assert "5 bit errors across 6,553,600 bits" in gate_reason, "gate_reason must state 5 bit errors for direct extraction"
    assert "0 bit errors across 6,553,600 bits" in gate_reason, "gate_reason must state 0 bit errors for roundtrip extraction"
    assert "direct" in gate_reason.lower(), "gate_reason must mention direct extraction"
    assert "roundtrip" in gate_reason.lower(), "gate_reason must mention separate roundtrip extraction"

    print("[SUCCESS] All consistency checks passed without error.")
    return True


if __name__ == "__main__":
    success = check_consistency()
    sys.exit(0 if success else 1)
