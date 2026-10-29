"""Experiment E07: Subband Allocation Evaluation.

Evaluates embedding across different DWT subbands:
1. HH only (Diagonal detail - Baseline)
2. HL only (Vertical detail / horizontal edges)
3. LH only (Horizontal detail / vertical edges)
4. HH + HL (Diagonal + Vertical detail)
5. HH + HL + LH (All high-frequency detail subbands)
6. LL control: Analysis of low-frequency approximation degradation

Measures:
- Usable capacity
- PSNR, SSIM, MAE, Max pixel error
- Bit Error Rate (BER)
- Exact byte recovery
"""

import io
import json
import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.carrier.header import CarrierHeader
from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


def embed_subbands(cover_arr: np.ndarray, protected_payload: bytes, subband_modes: list, ch: int = 0):
    """Embeds bits into selected subbands using orthogonal spatial perturbations."""
    total_positions = len(subband_modes) * 256 * 256
    raw_capacity_bytes = total_positions // 8
    max_protected = raw_capacity_bytes - 16

    if len(protected_payload) > max_protected:
        raise ValueError(f"Payload ({len(protected_payload)} B) exceeds capacity ({max_protected} B)")

    header = CarrierHeader(session_counter=70, payload_length=len(protected_payload))
    header_bytes = header.pack()

    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    payload_bits = np.unpackbits(np.frombuffer(protected_payload, dtype=np.uint8))
    all_bits = np.concatenate([header_bits, payload_bits])

    stego_arr = cover_arr.copy()

    # Subband formulas and target pixel:
    # 2*HH = a - b - c + d -> adjust 'a' (+1 / -1)
    # 2*HL = a + b - c - d -> adjust 'b' (+1 / -1)
    # 2*LH = a - b + c - d -> adjust 'c' (+1 / -1)
    bit_idx = 0
    total_bits_to_embed = len(all_bits)

    for mode in subband_modes:
        for r in range(256):
            for c in range(256):
                if bit_idx >= total_bits_to_embed:
                    break
                bit = int(all_bits[bit_idx])

                a = int(stego_arr[2 * r, 2 * c, ch])
                b = int(stego_arr[2 * r, 2 * c + 1, ch])
                c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
                d = int(stego_arr[2 * r + 1, 2 * c + 1, ch])

                if mode == "HH":
                    two_coeff = a - b - c_val + d
                    if ((two_coeff % 2 + 2) % 2) != bit:
                        stego_arr[2 * r, 2 * c, ch] = a + 1 if a < 255 else a - 1
                elif mode == "HL":
                    two_coeff = a + b - c_val - d
                    if ((two_coeff % 2 + 2) % 2) != bit:
                        stego_arr[2 * r, 2 * c + 1, ch] = b + 1 if b < 255 else b - 1
                elif mode == "LH":
                    two_coeff = a - b + c_val - d
                    if ((two_coeff % 2 + 2) % 2) != bit:
                        stego_arr[2 * r + 1, 2 * c, ch] = c_val + 1 if c_val < 255 else c_val - 1
                elif mode == "LL":
                    # Modulating LL directly modifies average block luminance
                    two_coeff = a + b + c_val + d
                    if ((two_coeff % 2 + 2) % 2) != bit:
                        # Disrupts low-frequency approximation
                        stego_arr[2 * r, 2 * c, ch] = a + 1 if a < 255 else a - 1

                bit_idx += 1
            if bit_idx >= total_bits_to_embed:
                break

    return stego_arr, header


def extract_subbands(stego_arr: np.ndarray, subband_modes: list, ch: int = 0):
    extracted_bits = []

    # First extract 128 bits from the first subband mode
    bit_idx = 0
    for mode in subband_modes:
        for r in range(256):
            for c in range(256):
                a = int(stego_arr[2 * r, 2 * c, ch])
                b = int(stego_arr[2 * r, 2 * c + 1, ch])
                c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
                d = int(stego_arr[2 * r + 1, 2 * c + 1, ch])

                if mode == "HH":
                    two_coeff = a - b - c_val + d
                elif mode == "HL":
                    two_coeff = a + b - c_val - d
                elif mode == "LH":
                    two_coeff = a - b + c_val - d
                elif mode == "LL":
                    two_coeff = a + b + c_val + d

                bit = (two_coeff % 2 + 2) % 2
                extracted_bits.append(bit)
                bit_idx += 1
                if bit_idx == 128:
                    break
            if bit_idx == 128:
                break
        if bit_idx == 128:
            break

    header_bytes = np.packbits(np.array(extracted_bits[:128], dtype=np.uint8)).tobytes()
    header = CarrierHeader.unpack(header_bytes)

    total_needed_bits = 128 + header.payload_length * 8
    extracted_all_bits = []
    bit_idx = 0

    for mode in subband_modes:
        for r in range(256):
            for c in range(256):
                if bit_idx >= total_needed_bits:
                    break
                a = int(stego_arr[2 * r, 2 * c, ch])
                b = int(stego_arr[2 * r, 2 * c + 1, ch])
                c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
                d = int(stego_arr[2 * r + 1, 2 * c + 1, ch])

                if mode == "HH":
                    two_coeff = a - b - c_val + d
                elif mode == "HL":
                    two_coeff = a + b - c_val - d
                elif mode == "LH":
                    two_coeff = a - b + c_val - d
                elif mode == "LL":
                    two_coeff = a + b + c_val + d

                extracted_all_bits.append((two_coeff % 2 + 2) % 2)
                bit_idx += 1
            if bit_idx >= total_needed_bits:
                break
        if bit_idx >= total_needed_bits:
            break

    payload_bits = extracted_all_bits[128:total_needed_bits]
    protected_payload = np.packbits(np.array(payload_bits, dtype=np.uint8)).tobytes()

    return header, protected_payload


def run_e07():
    cover_path = os.path.join("MVP 0 Dataset", "0001x2.png")
    secret_path = os.path.join("tests", "fixtures", "secret_small.png")
    key = b"01234567890123456789012345678901"
    aad = b"PALADIN_E07_SUBBANDS"

    with Image.open(cover_path) as img:
        cover_arr = np.array(img.convert("RGB"), dtype=np.uint8)

    with open(secret_path, "rb") as f:
        secret_bytes = f.read()

    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.IMAGE,
        metadata={"filename": "secret_small.png", "experiment": "E07"},
    )
    serialized = SecretSerializer.serialize(secret_payload)
    crypto = AESGCMProvider()
    encrypted = crypto.encrypt(serialized, key, aad=aad)
    ecc = ReedSolomonEncoder()
    protected = ecc.encode(encrypted)

    test_modes = [
        {"name": "HH only (Baseline)", "subbands": ["HH"]},
        {"name": "HL only", "subbands": ["HL"]},
        {"name": "LH only", "subbands": ["LH"]},
        {"name": "HH + HL", "subbands": ["HH", "HL"]},
        {"name": "HH + HL + LH (All Details)", "subbands": ["HH", "HL", "LH"]},
        {"name": "LL only (Low-Freq Control)", "subbands": ["LL"]},
    ]

    results = []
    for tm in test_modes:
        s_list = tm["subbands"]
        raw_positions = len(s_list) * 256 * 256
        raw_bytes = raw_positions // 8
        max_protected = raw_bytes - 16
        rs_blocks = (max_protected - 4) // 255
        usable_secret = max(0, rs_blocks * 223 - 28 - 18)

        stego_arr, header = embed_subbands(cover_arr, protected, s_list)

        # PNG save / reload
        png_buf = io.BytesIO()
        Image.fromarray(stego_arr).save(png_buf, format="PNG")
        reloaded_img = Image.open(io.BytesIO(png_buf.getvalue())).convert("RGB")
        reloaded_arr = np.array(reloaded_img, dtype=np.uint8)

        # Extraction
        ex_header, ex_protected = extract_subbands(reloaded_arr, s_list)
        decrypted_enc = ecc.decode(ex_protected)
        decrypted_serialized = crypto.decrypt(decrypted_enc, key, aad=aad)
        recovered = SecretSerializer.deserialize(decrypted_serialized)

        psnr_val = calculate_psnr(cover_arr, reloaded_arr)
        ssim_val = calculate_ssim(cover_arr, reloaded_arr)
        diff_arr = np.abs(cover_arr.astype(int) - reloaded_arr.astype(int))
        mae_val = float(np.mean(diff_arr))
        max_err = int(np.max(diff_arr))

        bit_errors = int(np.sum(np.unpackbits(np.frombuffer(protected, dtype=np.uint8)) !=
                                np.unpackbits(np.frombuffer(ex_protected, dtype=np.uint8))))
        ber = float(bit_errors / (len(protected) * 8))
        exact_match = (recovered.data == secret_bytes)

        res_entry = {
            "configuration": tm["name"],
            "subbands_used": s_list,
            "raw_capacity_bytes": raw_bytes,
            "max_protected_bytes": max_protected,
            "usable_secret_capacity_bytes": usable_secret,
            "psnr_db": round(psnr_val, 4),
            "ssim": round(ssim_val, 6),
            "mae": round(mae_val, 6),
            "max_pixel_error": max_err,
            "ber": ber,
            "exact_match": exact_match,
        }
        results.append(res_entry)

    out_file = os.path.join("docs", "experiments", "e07_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== E07 SUBBAND ALLOCATION RESULTS ===")
    for r in results:
        print(f"[{r['configuration']}] Raw: {r['raw_capacity_bytes']} B | Usable: {r['usable_secret_capacity_bytes']} B | "
              f"PSNR: {r['psnr_db']} dB | SSIM: {r['ssim']} | BER: {r['ber']} | ExactMatch: {r['exact_match']}")
    print(f"Saved E07 results to {out_file}")

    return results


if __name__ == "__main__":
    run_e07()
