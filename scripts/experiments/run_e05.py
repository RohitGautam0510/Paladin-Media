"""Experiment E05: HH Channel Allocation Evaluation.

Evaluates embedding across:
1. 1 Channel: Red (Channel 0)
2. 2 Channels: Red + Green (Channels 0, 1)
3. 3 Channels: Red + Green + Blue (Channels 0, 1, 2)

Measures:
- Usable capacity
- Per-channel distortion (PSNR_R, PSNR_G, PSNR_B)
- Overall PSNR, SSIM, MAE
- Bit Error Rate (BER)
- Exact byte equality
- Runtime and memory
"""

import io
import json
import os
import sys
import time
import hashlib
import random
import struct
import numpy as np
from PIL import Image
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.carrier.header import CarrierHeader
from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


def embed_multi_channel(cover_arr: np.ndarray, protected_payload: bytes, channels: list, session_counter: int = 50):
    total_positions = len(channels) * 256 * 256
    raw_capacity_bytes = total_positions // 8
    max_protected = raw_capacity_bytes - 16

    if len(protected_payload) > max_protected:
        raise ValueError(f"Payload ({len(protected_payload)} B) exceeds capacity ({max_protected} B)")

    header = CarrierHeader(session_counter=session_counter, payload_length=len(protected_payload))
    header_bytes = header.pack()

    # Convert bytes to bits
    header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
    payload_bits = np.unpackbits(np.frombuffer(protected_payload, dtype=np.uint8))

    # All eligible positions: (ch, r, c)
    # Channel ordering: sequentially channels[0], channels[1], ...
    all_coords = []
    for ch in channels:
        for r in range(256):
            for c in range(256):
                all_coords.append((ch, r, c))

    # Fixed header positions: first 128 positions
    header_coords = all_coords[:128]

    # Deterministic payload positions from remaining
    hasher = hashlib.sha256()
    hasher.update(b"PALADIN_MASK_SEED_E05")
    hasher.update(struct.pack(">II", session_counter, len(payload_bits)))
    seed_int = int.from_bytes(hasher.digest()[:8], "big")
    rng = random.Random(seed_int)

    candidate_indices = range(128, len(all_coords))
    selected_indices = rng.sample(candidate_indices, len(payload_bits))
    payload_coords = [all_coords[idx] for idx in selected_indices]

    stego_arr = cover_arr.copy()

    # Embed header bits
    for bit, (ch, r, c) in zip(header_bits, header_coords):
        a_val = int(stego_arr[2 * r, 2 * c, ch])
        b_val = int(stego_arr[2 * r, 2 * c + 1, ch])
        c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
        d_val = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
        two_hh = a_val - b_val - c_val + d_val
        if ((two_hh % 2 + 2) % 2) != int(bit):
            stego_arr[2 * r, 2 * c, ch] = a_val + 1 if a_val < 255 else a_val - 1

    # Embed payload bits
    for bit, (ch, r, c) in zip(payload_bits, payload_coords):
        a_val = int(stego_arr[2 * r, 2 * c, ch])
        b_val = int(stego_arr[2 * r, 2 * c + 1, ch])
        c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
        d_val = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
        two_hh = a_val - b_val - c_val + d_val
        if ((two_hh % 2 + 2) % 2) != int(bit):
            stego_arr[2 * r, 2 * c, ch] = a_val + 1 if a_val < 255 else a_val - 1

    return stego_arr, header


def extract_multi_channel(stego_arr: np.ndarray, channels: list):
    all_coords = []
    for ch in channels:
        for r in range(256):
            for c in range(256):
                all_coords.append((ch, r, c))

    header_coords = all_coords[:128]
    header_bits = []
    for ch, r, c in header_coords:
        a_val = int(stego_arr[2 * r, 2 * c, ch])
        b_val = int(stego_arr[2 * r, 2 * c + 1, ch])
        c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
        d_val = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
        two_hh = a_val - b_val - c_val + d_val
        header_bits.append((two_hh % 2 + 2) % 2)

    header_bytes = np.packbits(np.array(header_bits, dtype=np.uint8)).tobytes()
    header = CarrierHeader.unpack(header_bytes)

    payload_bits_len = header.payload_length * 8
    hasher = hashlib.sha256()
    hasher.update(b"PALADIN_MASK_SEED_E05")
    hasher.update(struct.pack(">II", header.session_counter, payload_bits_len))
    seed_int = int.from_bytes(hasher.digest()[:8], "big")
    rng = random.Random(seed_int)

    candidate_indices = range(128, len(all_coords))
    selected_indices = rng.sample(candidate_indices, payload_bits_len)
    payload_coords = [all_coords[idx] for idx in selected_indices]

    payload_bits = []
    for ch, r, c in payload_coords:
        a_val = int(stego_arr[2 * r, 2 * c, ch])
        b_val = int(stego_arr[2 * r, 2 * c + 1, ch])
        c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
        d_val = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
        two_hh = a_val - b_val - c_val + d_val
        payload_bits.append((two_hh % 2 + 2) % 2)

    protected_bytes = np.packbits(np.array(payload_bits, dtype=np.uint8)).tobytes()
    return header, protected_bytes


def run_e05():
    cover_path = os.path.join("MVP 0 Dataset", "0001x2.png")
    secret_path = os.path.join("tests", "fixtures", "secret_small.png")
    key = b"01234567890123456789012345678901"
    aad = b"PALADIN_E05_CHANNELS"

    with Image.open(cover_path) as img:
        cover_arr = np.array(img.convert("RGB"), dtype=np.uint8)

    with open(secret_path, "rb") as f:
        secret_bytes = f.read()

    secret_payload = SecretPayload(
        data=secret_bytes,
        media_type=MediaType.IMAGE,
        metadata={"filename": "secret_small.png", "experiment": "E05"},
    )
    serialized = SecretSerializer.serialize(secret_payload)
    crypto = AESGCMProvider()
    encrypted = crypto.encrypt(serialized, key, aad=aad)
    ecc = ReedSolomonEncoder()
    protected = ecc.encode(encrypted)

    channel_configs = [
        {"name": "1 Channel (R)", "channels": [0], "desc": "Channel 0 only"},
        {"name": "2 Channels (R+G)", "channels": [0, 1], "desc": "Channels 0 and 1"},
        {"name": "3 Channels (RGB)", "channels": [0, 1, 2], "desc": "Channels 0, 1, and 2"},
    ]

    results = []
    for cfg in channel_configs:
        chs = cfg["channels"]
        num_ch = len(chs)
        raw_bits = num_ch * 256 * 256
        raw_bytes = raw_bits // 8
        max_protected = raw_bytes - 16
        rs_blocks = (max_protected - 4) // 255
        usable_secret = max(0, rs_blocks * 223 - 28 - 18)

        t0 = time.perf_counter()
        stego_arr, header = embed_multi_channel(cover_arr, protected, chs)
        t_embed = time.perf_counter() - t0

        # PNG save / reload
        png_buf = io.BytesIO()
        Image.fromarray(stego_arr).save(png_buf, format="PNG")
        reloaded_img = Image.open(io.BytesIO(png_buf.getvalue())).convert("RGB")
        reloaded_arr = np.array(reloaded_img, dtype=np.uint8)

        # Extraction
        t0 = time.perf_counter()
        ex_header, ex_protected = extract_multi_channel(reloaded_arr, chs)
        decrypted_enc = ecc.decode(ex_protected)
        decrypted_serialized = crypto.decrypt(decrypted_enc, key, aad=aad)
        recovered = SecretSerializer.deserialize(decrypted_serialized)
        t_extract = time.perf_counter() - t0

        # Quality metrics
        psnr_overall = calculate_psnr(cover_arr, reloaded_arr)
        ssim_val = calculate_ssim(cover_arr, reloaded_arr)

        # Per-channel PSNR
        per_ch_psnr = {}
        ch_names = ["R", "G", "B"]
        for idx in range(3):
            c_orig = cover_arr[:, :, idx]
            c_stego = reloaded_arr[:, :, idx]
            mse = np.mean((c_orig.astype(float) - c_stego.astype(float)) ** 2)
            per_ch_psnr[ch_names[idx]] = float("inf") if mse == 0 else round(20 * np.log10(255.0 / np.sqrt(mse)), 4)

        bit_errors = int(np.sum(np.unpackbits(np.frombuffer(protected, dtype=np.uint8)) !=
                                np.unpackbits(np.frombuffer(ex_protected, dtype=np.uint8))))
        ber = float(bit_errors / (len(protected) * 8))
        exact_match = (recovered.data == secret_bytes)

        res_entry = {
            "configuration": cfg["name"],
            "channels_used": chs,
            "raw_capacity_bytes": raw_bytes,
            "max_protected_bytes": max_protected,
            "usable_secret_capacity_bytes": usable_secret,
            "overall_psnr_db": round(psnr_overall, 4),
            "per_channel_psnr_db": per_ch_psnr,
            "ssim": round(ssim_val, 6),
            "ber": ber,
            "exact_match": exact_match,
            "embed_time_ms": round(t_embed * 1000, 2),
            "extract_time_ms": round(t_extract * 1000, 2),
        }
        results.append(res_entry)

    out_file = os.path.join("docs", "experiments", "e05_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== E05 HH CHANNEL ALLOCATION RESULTS ===")
    for r in results:
        print(f"[{r['configuration']}] Raw: {r['raw_capacity_bytes']} B | Usable: {r['usable_secret_capacity_bytes']} B | "
              f"Overall PSNR: {r['overall_psnr_db']} dB | Per-ch: {r['per_channel_psnr_db']} | ExactMatch: {r['exact_match']}")
    print(f"Saved E05 results to {out_file}")

    return results


if __name__ == "__main__":
    run_e05()
