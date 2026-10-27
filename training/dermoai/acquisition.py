import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import write_json
from .dataset import write_manifest

CANONICAL_LABELS = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL", "Acne", "Alopecia", "Nail", "Fungal", "Vascular", "Healthy"]


def kaggle_download(handle, destination):
    import kagglehub
    path = kagglehub.dataset_download(handle, output_dir=str(destination))
    return {"path": path, "handle": handle}


def prepare_pad(folder, manifest, seed=42):
    folder = Path(folder).resolve()
    metadata = list(folder.rglob("metadata.csv"))
    if len(metadata) != 1:
        raise ValueError("Expected exactly one PAD-UFES-20 metadata.csv")
    with metadata[0].open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    images = {}
    for path in folder.rglob("*.png"):
        if path.name in images:
            raise ValueError(f"Ambiguous repeated image filename: {path.name}")
        images[path.name] = path
    rows = []
    for record in records:
        label = record["diagnostic"]
        if label not in CANONICAL_LABELS[:6]:
            raise ValueError(f"Unexpected PAD diagnosis: {label}")
        if not record["patient_id"]:
            raise ValueError("Patient ID missing")
        if record["img_id"] not in images:
            raise ValueError(f"Missing image: {record['img_id']}")
        bucket = int(hashlib.sha256(f"{seed}:PAD-UFES-20:{record['patient_id']}".encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        rows.append({"image_path": str(images[record["img_id"]]), "label": label, "source": "PAD-UFES-20", "patient_id": record["patient_id"], "lesion_id": record["lesion_id"], "split": split, "approved": "false", "original_validation": "false", "smartphone": "true"})
    write_manifest(manifest, rows)
    provenance = {"source": "PAD-UFES-20", "kaggle": "https://www.kaggle.com/datasets/mahdavi1202/skin-cancer", "original": "https://doi.org/10.17632/zr7vgbcyr2.1", "license": "CC BY 4.0", "patient_ids_preserved": True, "split_policy": f"New patient-hash 70/15/15 split; seed={seed}; not the unavailable historical validation set", "canonical_class_order": CANONICAL_LABELS, "class_counts": dict(Counter(row["label"] for row in rows)), "split_counts": dict(Counter(row["split"] for row in rows)), "labels_approved_for_training": False}
    write_json(Path(manifest).with_suffix(".provenance.json"), provenance)
    return provenance


def curate_import(audit_path, destination, seed=42):
    from .common import read_json
    audit = read_json(audit_path)
    rows = audit["snapshot"]
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        keys = [("image", row["sha256"])]
        for field in ("patient_id", "lesion_id"):
            if row.get(field):
                keys.append((field, row["source"], row[field]))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    group_patients = {}
    for index, row in enumerate(rows):
        if row["source"] == "PAD-UFES-20":
            group_patients.setdefault(find(index), []).append(row["patient_id"])
    conflicting = {group["sha256"] for group in audit["exact_duplicates"] if len({rows[index]["label"] for index in group["rows"]}) > 1}
    selected, quarantine, aliases = [], [], []
    kept = set()
    for index, original in enumerate(rows):
        row = dict(original)
        if row["sha256"] in conflicting:
            quarantine.append(row)
            continue
        if row["sha256"] in kept:
            aliases.append(row)
            continue
        kept.add(row["sha256"])
        if row["source"] == "PAD-UFES-20":
            group = min(group_patients[find(index)])
            bucket = int(hashlib.sha256(f"{seed}:PAD-connected:{group}".encode()).hexdigest()[:8], 16) % 100
            row["split"] = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        selected.append(row)
    destination = Path(destination)
    write_manifest(destination, selected)
    write_json(destination.with_suffix(".quarantine.json"), quarantine)
    write_json(destination.with_suffix(".aliases.json"), aliases)
    summary = {"original_images": audit["images"], "selected_unique_images": len(selected), "conflicting_unique_images_quarantined": len(conflicting), "quarantined_records": len(quarantine), "same_label_duplicate_records": len(aliases), "class_counts": dict(Counter(row["label"] for row in selected)), "patient_groups": len(group_patients), "policy": "PAD new splits use connected patient/lesion/exact-image groups. Healthy source fold preserved. DermNet remains unassigned. Original files and raw manifests retained."}
    write_json(destination.with_suffix(".curation.json"), summary)
    return summary
