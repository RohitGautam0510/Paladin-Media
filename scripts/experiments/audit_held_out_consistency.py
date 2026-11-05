"""Audit existing held_out_test_results_20261029.json and reconcile the 5 failures.

Output: docs/diagnostics_20261029/task2_audit_consistency.json
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

INPUT_JSON = REPO_ROOT / "docs" / "experiments" / "held_out_test_results_20261029.json"
DIAG_DIR = REPO_ROOT / "docs" / "diagnostics_20261029"
OUTPUT_JSON = DIAG_DIR / "task2_audit_consistency.json"

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["per_image_results"]
    summary_q = data["primary_qualified_production_rs48"]
    summary_b = data["baseline_deployment_rs32"]
    summary_py = data["pytorch_reference_rs48"]

    # 1. Verify Agreement of Counts
    n_records = len(records)
    
    # Exact recovery
    exact_q_list = [r["image_id"] for r in records if r["onnx_qualified_rs48"]["receiver_exact_byte_match"]]
    exact_b_list = [r["image_id"] for r in records if r["onnx_baseline_rs32"]["receiver_exact_byte_match"]]
    exact_py_list = [r["image_id"] for r in records if r["pytorch_reference_rs48"]["receiver_exact_byte_match"]]

    # AES auth
    aes_q_list = [r["image_id"] for r in records if r["onnx_qualified_rs48"]["receiver_aes_authenticated"]]

    # RS decode success
    rs_q_list = [r["image_id"] for r in records if r["onnx_qualified_rs48"]["payload_rs_decode_success"]]
    rs_b_list = [r["image_id"] for r in records if r["onnx_baseline_rs32"]["payload_rs_decode_success"]]
    rs_py_list = [r["image_id"] for r in records if r["pytorch_reference_rs48"]["payload_rs_decode_success"]]

    # Header intact
    hdr_q_list = [r["image_id"] for r in records if r["onnx_qualified_rs48"]["header_intact"]]

    count_checks = {
        "total_records": n_records,
        "onnx_q_exact_matches": {
            "sum_from_records": len(exact_q_list),
            "summary_reported": summary_q["exact_recovery_count"],
            "summary_pct": summary_q["exact_recovery_rate_pct"],
            "consistent": len(exact_q_list) == summary_q["exact_recovery_count"]
        },
        "onnx_q_aes_authenticated": {
            "sum_from_records": len(aes_q_list),
            "summary_reported_pct": summary_q["aes_gcm_auth_rate_pct"],
            "consistent": (len(aes_q_list) / n_records * 100.0) == summary_q["aes_gcm_auth_rate_pct"]
        },
        "onnx_q_payload_rs_success": {
            "sum_from_records": len(rs_q_list),
            "summary_reported_pct": summary_q["payload_rs_success_rate_pct"],
            "consistent": (len(rs_q_list) / n_records * 100.0) == summary_q["payload_rs_success_rate_pct"]
        },
        "onnx_q_header_intact": {
            "sum_from_records": len(hdr_q_list),
            "summary_reported_pct": summary_q["header_intact_rate_pct"],
            "consistent": (len(hdr_q_list) / n_records * 100.0) == summary_q["header_intact_rate_pct"]
        },
        "baseline_exact_matches": {
            "sum_from_records": len(exact_b_list),
            "summary_reported": summary_b["exact_recovery_count"],
            "consistent": len(exact_b_list) == summary_b["exact_recovery_count"]
        },
        "baseline_rs_success": {
            "sum_from_records": len(rs_b_list),
            "summary_reported_pct": summary_b["payload_rs_success_rate_pct"],
            "consistent": (len(rs_b_list) / n_records * 100.0) == summary_b["payload_rs_success_rate_pct"]
        },
        "pytorch_exact_matches": {
            "sum_from_records": len(exact_py_list),
            "summary_reported_pct": summary_py["exact_recovery_rate_pct"],
            "consistent": (len(exact_py_list) / n_records * 100.0) == summary_py["exact_recovery_rate_pct"]
        },
    }

    # 2. Detailed Per-Image Failure Analysis for 0715, 0774, 0777, 0785, 0795
    target_failed_ids = [715, 774, 777, 785, 795]
    failed_reconciliation = {}

    for r in records:
        img_id = r["image_id"]
        if img_id in target_failed_ids:
            fn = r["filename"]
            q = r["onnx_qualified_rs48"]
            b = r["onnx_baseline_rs32"]
            py = r["pytorch_reference_rs48"]

            # Detailed classification
            header_failed = not q["header_intact"]
            payload_rs_failed_q = not q["payload_rs_decode_success"]
            payload_rs_failed_b = not b["payload_rs_decode_success"]
            aes_failed = not q["receiver_aes_authenticated"]
            exact_failed = not q["receiver_exact_byte_match"]

            max_err_q = q["rs_max_codeword_error"]
            exceeded_t24 = max_err_q > 24
            exceeded_t16 = max_err_q > 16

            # Determine earliest failure point
            if header_failed:
                earliest_failure_point = "HEADER_CORRUPTION"
            elif payload_rs_failed_q:
                earliest_failure_point = "PAYLOAD_RS_EXCEEDED"
            elif aes_failed:
                earliest_failure_point = "AES_GCM_AUTH_FAIL"
            elif exact_failed:
                earliest_failure_point = "BYTE_MISMATCH"
            else:
                earliest_failure_point = "NONE"

            # Categorization
            if header_failed and exceeded_t24:
                compound_category = "DUAL_FAILURE (Header Corrupted AND Payload Exceeded t=24)"
            elif header_failed and not exceeded_t24:
                compound_category = "HEADER_ONLY_FAILURE (Header Corrupted; Payload Within t=24 Budget)"
            elif not header_failed and exceeded_t24:
                compound_category = "PAYLOAD_ONLY_FAILURE (Header Intact; Payload Exceeded t=24 Budget)"
            else:
                compound_category = "OTHER_FAILURE"

            failed_reconciliation[fn] = {
                "image_id": img_id,
                "filename": fn,
                "psnr_db": q["psnr_db"],
                "ssim": q["ssim"],
                "payload_ber": q["payload_ber"],
                "header_bit_errors": q["header_bit_errors"],
                "header_intact": q["header_intact"],
                "header_error_msg": q["header_error_msg"],
                "rs_max_codeword_error": max_err_q,
                "rs_per_codeword_errors": q["rs_per_codeword_errors"],
                "exceeded_t24_budget": exceeded_t24,
                "exceeded_t16_budget": exceeded_t16,
                "payload_rs_decode_success_rs48": q["payload_rs_decode_success"],
                "payload_rs_decode_success_rs32": b["payload_rs_decode_success"],
                "receiver_error": q["receiver_error"],
                "receiver_exact_byte_match": q["receiver_exact_byte_match"],
                "receiver_aes_authenticated": q["receiver_aes_authenticated"],
                "earliest_failure_point": earliest_failure_point,
                "compound_category": compound_category,
            }

    output = {
        "count_checks": count_checks,
        "failed_reconciliation": failed_reconciliation,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Task 2 Audit complete. Output written to:", OUTPUT_JSON)
    print("\nCount verification:")
    for k, v in count_checks.items():
        print(f"  {k}: {v}")

    print("\nFailed images summary:")
    for fn, d in failed_reconciliation.items():
        print(f"  {fn}: Earliest='{d['earliest_failure_point']}', Category='{d['compound_category']}', MaxErr={d['rs_max_codeword_error']}/24")

if __name__ == "__main__":
    main()
