import hashlib
import concurrent.futures
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from .common import digest_file


def perceptual_hashes(image):
    gray = image.convert("L")
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coefficients = dctn(small, type=2, norm="ortho")[:8, :8].reshape(-1)[1:]
    bits = coefficients > np.median(coefficients)
    perceptual = int.from_bytes(np.packbits(bits).tobytes(), "big")
    difference = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    gradient = int.from_bytes(np.packbits(difference[:, 1:] > difference[:, :-1]).tobytes(), "big")
    return perceptual, gradient


def perceptual_hash_variants(image):
    gray = image.convert("L")
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.uint8)
    variants = []
    for rotations in range(4):
        rotated = np.rot90(small, rotations)
        for transformed in (rotated, np.fliplr(rotated)):
            coefficients = dctn(transformed.astype(np.float32), type=2, norm="ortho")[:8, :8].reshape(-1)[1:]
            bits = coefficients > np.median(coefficients)
            perceptual = int.from_bytes(np.packbits(bits).tobytes(), "big")
            difference = np.asarray(Image.fromarray(transformed).resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
            gradient = int.from_bytes(np.packbits(difference[:, 1:] > difference[:, :-1]).tobytes(), "big")
            variants.append((perceptual, gradient))
    return variants


class HashTree:
    def __init__(self):
        self.root = None

    def add(self, value, item):
        if self.root is None:
            self.root = [value, [item], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(item)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [item], {}]
                return
            node = node[2][distance]

    def nearby(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield from node[1]
            pending.extend(child for edge, child in node[2].items() if distance - radius <= edge <= distance + radius)


def image_fingerprints(row):
    with Image.open(row["image_path"]) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        decoded_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        hashes = perceptual_hash_variants(image)
    record = {"image_path": row["image_path"], "sha256": digest_file(row["image_path"]), "decoded_sha256": decoded_hash}
    return record, hashes


def similarity_scan(rows, progress=None):
    tree = HashTree()
    decoded = {}
    pairs = {}
    records = []
    executor_class = concurrent.futures.ProcessPoolExecutor if len(rows) >= 1000 else concurrent.futures.ThreadPoolExecutor
    worker_count = 8 if len(rows) >= 1000 else min(16, max(1, len(rows)))
    with executor_class(max_workers=worker_count) as executor:
        fingerprints = executor.map(image_fingerprints, rows)
        for index, (record, hashes) in enumerate(fingerprints):
            decoded_hash = record["decoded_sha256"]
            records.append(record)
            if decoded_hash in decoded:
                pairs[(decoded[decoded_hash], index)] = "identical_decoded_pixels"
            decoded.setdefault(decoded_hash, index)
            for perceptual, gradient in hashes:
                for other, other_gradient in tree.nearby(perceptual, 4):
                    if (gradient ^ other_gradient).bit_count() <= 6:
                        pairs.setdefault((other, index), "perceptual_candidate")
            tree.add(hashes[0][0], (index, hashes[0][1]))
            if progress and (index + 1) % 500 == 0:
                progress(index + 1, len(rows))
    return {"images": len(rows), "records": records, "pairs": [{"left": left, "right": right, "reason": reason} for (left, right), reason in sorted(pairs.items())], "method": "Decoded RGB hashes plus 63-bit DCT hash distance <=4 AND 64-bit gradient hash distance <=6; rotations/mirrors checked. Similarity is a conservative candidate screen, not proof of patient identity or complete duplicate detection."}


def connected_groups(rows, pairs):
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for field in ("patient_id", "lesion_id"):
            if row.get(field):
                key = (field, row["source"], row[field])
                if key in seen:
                    parents[find(index)] = find(seen[key])
                else:
                    seen[key] = index
    for pair in pairs:
        parents[find(pair["left"])] = find(pair["right"])
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    return list(groups.values())
