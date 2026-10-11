"""Experiment E09: Multi-Carrier Partitioning Protocol (MCPP) Evaluation.

Evaluates dividing protected payloads across multiple carrier images:
- Partitioning a 15,000 B secret payload (which exceeds single 8,176 B carrier limit) across 3 carriers.
- Deterministic framing using CarrierHeader flags and partition index.
- Out-of-order reassembly handling.
- Missing carrier detection (fail-closed).
- Tampered partition detection (fail-closed).
"""

import io
import json
import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.carrier.capacity import CapacityCalculator
from src.carrier.dwt import HaarDWT2D
from src.carrier.embedding import BaselineEmbedder
from src.carrier.extraction import Extractor
from src.carrier.header import CarrierHeader
from src.core.media_pipeline import calculate_psnr, calculate_ssim
from src.core.serialization.serializer import MediaType, SecretPayload, SecretSerializer
from src.security.crypto.provider import AESGCMProvider
from src.security.ecc.encoder import ReedSolomonEncoder


class MultiCarrierProtocol:
    FLAG_PARTITIONED = 0x01

    @classmethod
    def partition_and_embed(cls, protected_payload: bytes, carriers: list, session_id: int):
        num_carriers = len(carriers)
        chunk_size = (len(protected_payload) + num_carriers - 1) // num_carriers
        stego_carriers = []

        for idx, carrier_arr in enumerate(carriers):
            start = idx * chunk_size
            end = min(len(protected_payload), (idx + 1) * chunk_size)
            chunk = protected_payload[start:end]

            # In CarrierHeader, reserved field encodes: (partition_index << 8) | total_partitions
            reserved_field = (idx << 8) | num_carriers
            header = CarrierHeader(
                session_counter=session_id,
                payload_length=len(chunk),
                flags=cls.FLAG_PARTITIONED,
                reserved=reserved_field,
            )

            # Modulate header and chunk into carrier
            stego_arr, _ = cls._embed_single_carrier(carrier_arr, header, chunk)
            stego_carriers.append((idx, stego_arr))

        return stego_carriers

    @classmethod
    def _embed_single_carrier(cls, carrier_arr: np.ndarray, header: CarrierHeader, chunk: bytes, ch: int = 0):
        stego_arr = carrier_arr.copy()
        header_bytes = header.pack()
        header_bits = np.unpackbits(np.frombuffer(header_bytes, dtype=np.uint8))
        chunk_bits = np.unpackbits(np.frombuffer(chunk, dtype=np.uint8))

        # Embed 128 header bits in first 128 positions
        for i, bit in enumerate(header_bits):
            r, c = divmod(i, 256)
            a = int(stego_arr[2 * r, 2 * c, ch])
            b = int(stego_arr[2 * r, 2 * c + 1, ch])
            c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
            d = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
            two_hh = a - b - c_val + d
            if ((two_hh % 2 + 2) % 2) != int(bit):
                stego_arr[2 * r, 2 * c, ch] = a + 1 if a < 255 else a - 1

        # Embed chunk bits sequentially in positions 128..
        for i, bit in enumerate(chunk_bits):
            pos = 128 + i
            r, c = divmod(pos, 256)
            a = int(stego_arr[2 * r, 2 * c, ch])
            b = int(stego_arr[2 * r, 2 * c + 1, ch])
            c_val = int(stego_arr[2 * r + 1, 2 * c, ch])
            d = int(stego_arr[2 * r + 1, 2 * c + 1, ch])
            two_hh = a - b - c_val + d
            if ((two_hh % 2 + 2) % 2) != int(bit):
                stego_arr[2 * r, 2 * c, ch] = a + 1 if a < 255 else a - 1

        return stego_arr, header

    @classmethod
    def extract_and_reassemble(cls, stego_carrier_images: list, ch: int = 0):
        partitions = {}
        total_partitions = None
        session_id = None

        for img_arr in stego_carrier_images:
            # Extract header
            header_bits = []
            for i in range(128):
                r, c = divmod(i, 256)
                a = int(img_arr[2 * r, 2 * c, ch])
                b = int(img_arr[2 * r, 2 * c + 1, ch])
                c_val = int(img_arr[2 * r + 1, 2 * c, ch])
                d = int(img_arr[2 * r + 1, 2 * c + 1, ch])
                two_hh = a - b - c_val + d
                header_bits.append((two_hh % 2 + 2) % 2)

            header_bytes = np.packbits(np.array(header_bits, dtype=np.uint8)).tobytes()
            header = CarrierHeader.unpack(header_bytes)

            if not (header.flags & cls.FLAG_PARTITIONED):
                raise ValueError("Carrier does not have PARTITIONED flag set")

            part_idx = header.reserved >> 8
            num_parts = header.reserved & 0xFF

            if total_partitions is None:
                total_partitions = num_parts
                session_id = header.session_counter
            else:
                if num_parts != total_partitions:
                    raise ValueError(f"Partition count mismatch: {num_parts} vs {total_partitions}")
                if header.session_counter != session_id:
                    raise ValueError(f"Session counter mismatch: {header.session_counter} vs {session_id}")

            # Extract chunk bits
            chunk_bits = []
            for i in range(header.payload_length * 8):
                pos = 128 + i
                r, c = divmod(pos, 256)
                a = int(img_arr[2 * r, 2 * c, ch])
                b = int(img_arr[2 * r, 2 * c + 1, ch])
                c_val = int(img_arr[2 * r + 1, 2 * c, ch])
                d = int(img_arr[2 * r + 1, 2 * c + 1, ch])
                two_hh = a - b - c_val + d
                chunk_bits.append((two_hh % 2 + 2) % 2)

            chunk_bytes = np.packbits(np.array(chunk_bits, dtype=np.uint8)).tobytes()
            partitions[part_idx] = chunk_bytes

        # Validate that all partitions [0..total_partitions-1] are present
        if len(partitions) != total_partitions:
            missing = set(range(total_partitions)) - set(partitions.keys())
            raise ValueError(f"Incomplete carrier set: missing partition indices {missing}")

        # Assemble in strict sequential order
        reassembled = b"".join(partitions[i] for i in range(total_partitions))
        return session_id, reassembled


def run_e09():
    carrier_paths = [
        os.path.join("MVP 0 Dataset", "0001x2.png"),
        os.path.join("MVP 0 Dataset", "0002x2.png"),
        os.path.join("MVP 0 Dataset", "0003x2.png"),
    ]
    carriers = [np.array(Image.open(p).convert("RGB"), dtype=np.uint8) for p in carrier_paths]

    # Create a 15,000-byte secret payload (large payload requiring 3 carriers)
    rng = np.random.RandomState(42)
    large_secret_bytes = bytes(rng.randint(0, 256, size=15000, dtype=np.uint8))
    key = b"01234567890123456789012345678901"
    aad = b"PALADIN_E09_MCPP"
    session_id = 9001

    secret_payload = SecretPayload(
        data=large_secret_bytes,
        media_type=MediaType.IMAGE,
        metadata={"filename": "large_secret_15k.bin", "experiment": "E09"},
    )
    serialized = SecretSerializer.serialize(secret_payload)
    crypto = AESGCMProvider()
    encrypted = crypto.encrypt(serialized, key, aad=aad)
    ecc = ReedSolomonEncoder()
    protected = ecc.encode(encrypted)

    print(f"Total protected payload size: {len(protected)} bytes")
    print(f"Max single-carrier capacity: {CapacityCalculator.MAX_PROTECTED_BYTES} bytes")
    print(f"Payload fits single carrier: {len(protected) <= CapacityCalculator.MAX_PROTECTED_BYTES}")

    # Embed across 3 carriers
    stego_pairs = MultiCarrierProtocol.partition_and_embed(protected, carriers, session_id=session_id)
    stego_images = [pair[1] for pair in stego_pairs]

    # PNG save / reload simulation for all carriers
    reloaded_carriers = []
    for s_arr in stego_images:
        buf = io.BytesIO()
        Image.fromarray(s_arr).save(buf, format="PNG")
        reloaded = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8)
        reloaded_carriers.append(reloaded)

    # 1. Normal in-order reassembly test
    _, recovered_protected_normal = MultiCarrierProtocol.extract_and_reassemble(reloaded_carriers)
    dec_enc_normal = ecc.decode(recovered_protected_normal)
    dec_ser_normal = crypto.decrypt(dec_enc_normal, key, aad=aad)
    recovered_normal = SecretSerializer.deserialize(dec_ser_normal)
    normal_match = (recovered_normal.data == large_secret_bytes)

    # 2. Shuffled order test (carriers presented out of order: [2, 0, 1])
    shuffled_carriers = [reloaded_carriers[2], reloaded_carriers[0], reloaded_carriers[1]]
    _, recovered_protected_shuffled = MultiCarrierProtocol.extract_and_reassemble(shuffled_carriers)
    dec_enc_shuffled = ecc.decode(recovered_protected_shuffled)
    dec_ser_shuffled = crypto.decrypt(dec_enc_shuffled, key, aad=aad)
    recovered_shuffled = SecretSerializer.deserialize(dec_ser_shuffled)
    shuffled_match = (recovered_shuffled.data == large_secret_bytes)

    # 3. Missing partition test (only carriers [0, 2] provided, missing 1)
    missing_partition_caught = False
    try:
        MultiCarrierProtocol.extract_and_reassemble([reloaded_carriers[0], reloaded_carriers[2]])
    except ValueError as e:
        if "Incomplete carrier set" in str(e):
            missing_partition_caught = True

    # 4. Tampered partition test: uncorrectable payload corruption in carrier 1
    tampered_caught = False
    try:
        corrupted_payload_carrier = reloaded_carriers[1].copy()
        for i in range(128, 3000):
            r, c = divmod(i, 256)
            corrupted_payload_carrier[2 * r, 2 * c, 0] ^= 0x01

        _, rec = MultiCarrierProtocol.extract_and_reassemble(
            [reloaded_carriers[0], corrupted_payload_carrier, reloaded_carriers[2]]
        )
        dec = ecc.decode(rec)
        crypto.decrypt(dec, key, aad=aad)
    except Exception:
        tampered_caught = True

    # Distortion across all 3 carriers
    psnr_list = [calculate_psnr(c_orig, s_arr) for c_orig, s_arr in zip(carriers, reloaded_carriers)]
    ssim_list = [calculate_ssim(c_orig, s_arr) for c_orig, s_arr in zip(carriers, reloaded_carriers)]

    results = {
        "experiment_id": "E09",
        "protocol": "Multi-Carrier Partitioning Protocol (MCPP)",
        "secret_plaintext_bytes": len(large_secret_bytes),
        "total_protected_payload_bytes": len(protected),
        "carriers_utilized": len(carriers),
        "bytes_per_carrier": len(protected) // len(carriers),
        "single_carrier_capacity_limit": CapacityCalculator.MAX_PROTECTED_BYTES,
        "normal_reassembly_match": normal_match,
        "shuffled_reassembly_match": shuffled_match,
        "missing_carrier_fail_closed": missing_partition_caught,
        "tampered_carrier_fail_closed": tampered_caught,
        "per_carrier_psnr_db": [round(p, 4) for p in psnr_list],
        "per_carrier_ssim": [round(s, 6) for s in ssim_list],
    }

    out_file = os.path.join("docs", "experiments", "e09_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== E09 MULTI-CARRIER PARTITIONING RESULTS ===")
    print(f"Secret: {len(large_secret_bytes)} B | Protected: {len(protected)} B across {len(carriers)} carriers")
    print(f"In-Order Exact Match: {normal_match}")
    print(f"Out-of-Order Reassembly Match: {shuffled_match}")
    print(f"Missing Carrier Detection (Fail-Closed): {missing_partition_caught}")
    print(f"Tampered Carrier Detection (Fail-Closed): {tampered_caught}")
    print(f"Per-carrier PSNR: {results['per_carrier_psnr_db']} dB")
    print(f"Saved E09 results to {out_file}")

    return results


if __name__ == "__main__":
    run_e09()
