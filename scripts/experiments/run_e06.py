"""Experiment E06: Supported Embedding Densities Evaluation.

Evaluates embedding density (bits per coefficient position):
1. 1 bit/coeff (Baseline): Modulo 2 parity on 2*HH, delta in {-1, 0, +1}
2. 2 bits/coeff: Modulo 4 modulation on 2*HH, delta in {-1, 0, +1, +2}
3. 3 bits/coeff: Modulo 8 modulation on 2*HH, delta in {-3, -2, -1, 0, +1, +2, +3, +4}

Measures:
- Raw and usable capacity
- PSNR, SSIM, MAE, Max pixel error
- Bit Error Rate (BER) through PNG save/reload
- Exact byte recovery
- Runtime and memory
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


def embed_density(cover_arr: np.ndarray, protected_payload: bytes, bits_per_coeff: int, ch: int = 0):
    total_coeffs = 256 * 256
    raw_bits = total_coeffs * bits_per_coeff
    raw_bytes = raw_bits // 8
    max_protected = raw_bytes - 16

    if len(protected_payload) > max_protected:
        raise ValueError(f"Payload ({len(protected_payload)} B) exceeds capacity ({max_protected} B)")

    header = CarrierHeader(session_counter=60, payload_length=len(protected_payload))
    header_bytes = header.pack()

    # Header is always embedded at 1 bit/coeff in first 128 positions for bootstrap reversibility
    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    payload_bits = np.unpackbits(np.frombuffer(protected_payload, dtype=np.uint8))

    stego_arr = cover_arr.copy()

    # 1. Embed header at 1 bit/coeff in positions 0..127
    for idx, bit in enumerate(header_bits):
        r, c = divmod(idx, 256)
        a = int(stego_arr[2 * r, 2 * c, ch])
        b = int(stego_arr[2 * r, 2 * c + 1, ch])
        c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
        d = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
        two_hh = a - b - c_val + d
        if ((two_hh % 2 + 2) % 2) != int(bit):
            stego_arr[2 * r, 2 * c, ch] = a + 1 if a < 255 else a - 1

    # 2. Embed payload at bits_per_coeff
    modulus = 1 << bits_per_coeff
    # Group payload bits into symbols of bits_per_coeff
    pad_len = (bits_per_coeff - (len(payload_bits) % bits_per_coeff)) % bits_per_coeff
    if pad_len > 0:
        payload_bits_padded = np.pad(payload_bits, (0, pad_len))
    else:
        payload_bits_padded = payload_bits

    symbols = []
    for i in range(0, len(payload_bits_padded), bits_per_coeff):
        sym = 0
        for b_idx in range(bits_per_coeff):
            sym = (sym << 1) | int(payload_bits_padded[i + b_idx])
        symbols.append(sym)

    for idx, target_sym in enumerate(symbols):
        coeff_idx = 128 + idx
        r, c = divmod(coeff_idx, 256)
        a = int(stego_arr[2 * r, 2 * c, ch])
        b = int(stego_arr[2 * r, 2 * c + 1, ch])
        c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
        d = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
        two_hh = a - b - c_val + d
        curr_sym = ((two_hh % modulus) + modulus) % modulus
        diff = (target_sym - curr_sym) % modulus

        # Map diff to optimal delta in range [-modulus//2, modulus//2]
        if diff > modulus // 2:
            delta = diff - modulus
        else:
            delta = diff

        # Apply delta to pixel a with saturation protection
        new_a = a + delta
        if new_a < 0:
            new_a = a + (delta + modulus)
        elif new_a > 255:
            new_a = a + (delta - modulus)

        stego_arr[2 * r, 2 * c, ch] = np.clip(new_a, 0, 255)

    return stego_arr, header


def extract_density(stego_arr: np.ndarray, bits_per_coeff: int, ch: int = 0):
    # Extract 128 header bits
    header_bits = []
    for idx in range(128):
        r, c = divmod(idx, 256)
        a = int(stego_arr[2 * r, 2 * c, ch])
        b = int(stego_arr[2 * r, 2 * c + 1, ch])
        c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
        d = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
        two_hh = a - b - c_val + d
        header_bits.append(((two_hh % 2) + 2) % 2)

    header_bytes = np.packbits(np.array(header_bits, dtype=np.uint8)).tobytes()
    header = CarrierHeader.unpack(header_bytes)

    total_payload_bits = header.payload_length * 8
    modulus = 1 << bits_per_coeff
    num_symbols = (total_payload_bits + bits_per_coeff - 1) // bits_per_coeff

    extracted_bits = []
    for idx in range(num_symbols):
        coeff_idx = 128 + idx
        r, c = divmod(coeff_idx, 256)
        a = int(stego_arr[2 * r, 2 * c, ch])
        b = int(stego_arr[2 * r, 2 * c + 1, ch])
        c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
        d = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
        two_hh = a - b - c_val + d
        sym = ((two_hh % modulus) + modulus) % modulus
        # Unpack symbol bits
        for b_idx in range(bits_per_coeff - 1, -1, -1):
            extracted_bits.append((sym >> b_idx) & 1)

    extracted_payload_bits = extracted_bits[:total_payload_bits]
    protected_payload = np.packbits(np.array(extracted_payload_bits, dtype=np.uint8)).tobytes()

    return header, protected_payload


def run_e06():
    cover_path = os.path.join("MVP 0 Dataset", "0001x2.png")
    secret_path = os.path.join("tests", "fixtures", "secret_small.png")
    key = b"01234567890123456789012345678901"
    aad = b"PALADIN_E06_DENSITY"

    with Image.open(cover_path) as img:
        cover_arr = np.array(img.convert("RGB"), dtype=np.uint8)

    with open(secret_path, "rb") as f:
        secret_bytes = f.read()

    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.IMAGE,
        metadata={"filename": "secret_small.png", "experiment": "E06"},
    )
    serialized = SecretSerializer.serialize(secret_payload)
    crypto = AESGCMProvider()
    encrypted = crypto.encrypt(serialized, key, aad=aad)
    ecc = ReedSolomonEncoder()
    protected = ecc.encode(encrypted)

    densities = [1, 2, 3]
    results = []

    for bpc in densities:
        raw_bits = 256 * 256 * bpc
        raw_bytes = raw_bits // 8
        max_protected = raw_bytes - 16
        rs_blocks = (max_protected - 4) // 255
        usable_secret = max(0, rs_blocks * 223 - 28 - 18)

        t0 = time.perf_counter()
        stego_arr, header = embed_density(cover_arr, protected, bits_per_coeff=bpc)
        t_embed = time.perf_counter() - t0

        # PNG save / reload
        png_buf = io.BytesIO()
        Image.fromarray(stego_arr).save(png_buf, format="PNG")
        reloaded_img = Image.open(io.BytesIO(png_buf.getvalue())).convert("RGB")
        reloaded_arr = np.array(reloaded_img, dtype=np.uint8)

        # Extraction
        t0 = time.perf_counter()
        ex_header, ex_protected = extract_density(reloaded_arr, bits_per_coeff=bpc)
        decrypted_enc = ecc.decode(ex_protected)
        decrypted_serialized = crypto.decrypt(decrypted_enc, key, aad=aad)
        recovered = SecretSerializer.deserialize(decrypted_serialized)
        t_extract = time.perf_counter() - t0

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
            "bits_per_coefficient": bpc,
            "modulus": 1 << bpc,
            "raw_capacity_bytes": raw_bytes,
            "max_protected_bytes": max_protected,
            "usable_secret_capacity_bytes": usable_secret,
            "psnr_db": round(psnr_val, 4),
            "ssim": round(ssim_val, 6),
            "mae": round(mae_val, 6),
            "max_pixel_error": max_err,
            "ber": ber,
            "exact_match": exact_match,
            "embed_time_ms": round(t_embed * 1000, 2),
            "extract_time_ms": round(t_extract * 1000, 2),
        }
        results.append(res_entry)

    out_file = os.path.join("docs", "experiments", "e06_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== E06 EMBEDDING DENSITY RESULTS ===")
    for r in results:
        print(f"[{r['bits_per_coefficient']} bit(s)/coeff] Raw: {r['raw_capacity_bytes']} B | Usable: {r['usable_secret_capacity_bytes']} B | "
              f"PSNR: {r['psnr_db']} dB | SSIM: {r['ssim']} | MaxErr: {r['max_pixel_error']} | BER: {r['ber']} | ExactMatch: {r['exact_match']}")
    print(f"Saved E06 results to {out_file}")

    return results


if __name__ == "__main__":
    run_e06()
