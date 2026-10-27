import csv
import concurrent.futures
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from .common import digest_file, write_json

FIELDS = ["image_path", "label", "source", "patient_id", "lesion_id", "split", "approved", "original_validation", "smartphone", "task_id", "source_split", "label_status", "similarity_group"]


def truth(value):
    return str(value).lower() in ("true", "1", "yes")


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS[:7]).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {FIELDS[:7]}")
        rows = list(reader)
    for row in rows:
        row["image_path"] = str((path.parent / row["image_path"]).resolve())
    return rows


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inspect_image(row):
    try:
        with Image.open(row["image_path"]) as image:
            image.load()
        return digest_file(row["image_path"]), None
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return None, type(error).__name__


def inspect_manifest(path, labels=None, require_patient_ids=True, training=False, source_labels_authorized=False, missing_patient_sources=()):
    rows = read_manifest(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        image_results = list(executor.map(inspect_image, rows))
    errors, corrupt, duplicates = [], [], []
    patients, lesions, hashes = defaultdict(set), defaultdict(set), defaultdict(list)
    snapshots = []
    counts = Counter()
    similarity_groups = defaultdict(set)
    unknown_patients = Counter()
    source_labels = 0
    for index, row in enumerate(rows):
        split = row["split"]
        active = split in ("train", "validation", "test")
        if split not in ("train", "validation", "test", "unassigned"):
            errors.append(f"Row {index}: unknown split")
        if truth(row.get("original_validation")) and split != "validation":
            errors.append(f"Row {index}: original validation image moved")
        source_split = row.get("source_split", "").lower()
        if source_split == "test" and split not in ("test", "unassigned"):
            errors.append(f"Row {index}: source test image cannot enter training or validation")
        if source_split in ("validation", "valid", "val") and split not in ("validation", "unassigned"):
            errors.append(f"Row {index}: source validation image cannot enter training or test")
        if not row["source"]:
            errors.append(f"Row {index}: provenance missing")
        if labels is not None and row["label"] not in labels:
            errors.append(f"Row {index}: unknown label {row['label']}")
        if require_patient_ids and active and not row["patient_id"]:
            if row["source"] in missing_patient_sources:
                unknown_patients[row["source"]] += 1
            else:
                errors.append(f"Row {index}: patient ID missing; patient leakage cannot be excluded")
        source_label = source_labels_authorized and row.get("label_status") == "dataset_source_unreviewed"
        if active and source_label and not truth(row["approved"]):
            source_labels += 1
        if training and active and not truth(row["approved"]) and not source_label:
            errors.append(f"Row {index}: labeled data not approved")
        if row.get("similarity_group"):
            similarity_groups[row["similarity_group"]].add(split)
        if row["patient_id"]:
            patients[(row["source"], row["patient_id"])].add(split)
        if row.get("lesion_id"):
            lesions[(row["source"], row["lesion_id"])].add(split)
        digest, image_error = image_results[index]
        if image_error is None:
            hashes[digest].append((index, split))
            snapshots.append({**row, "sha256": digest})
        else:
            corrupt.append(index)
        counts[(split, row["label"])] += 1
    for mapping, kind in ((patients, "patient"), (lesions, "lesion"), (similarity_groups, "similarity group")):
        for identity, splits in mapping.items():
            if len(splits - {"unassigned"}) > 1:
                errors.append(f"Cross-split {kind}: {identity}")
    for digest, entries in hashes.items():
        if len(entries) > 1:
            duplicates.append({"sha256": digest, "rows": [entry[0] for entry in entries]})
            if len({rows[entry[0]]["label"] for entry in entries}) > 1:
                errors.append(f"Conflicting labels for identical image: {digest}")
            if len({entry[1] for entry in entries} - {"unassigned"}) > 1:
                errors.append(f"Exact image duplicate crosses splits: {digest}")
    if corrupt:
        errors.append(f"{len(corrupt)} unreadable/corrupt images")
    if training:
        for split in ("train", "validation"):
            if not any(row["split"] == split for row in rows):
                errors.append(f"No {split} data")
        train_labels = {row["label"] for row in rows if row["split"] == "train"}
        if labels and set(labels) != train_labels:
            errors.append("Every configured class must have training examples")
    version = hashlib.sha256(json.dumps(snapshots, sort_keys=True).encode()).hexdigest()
    split_ids = {split: hashlib.sha256(json.dumps(sorted((row["sha256"], row["label"], row["source"], row["patient_id"]) for row in snapshots if row["split"] == split)).encode()).hexdigest() for split in ("train", "validation", "test")}
    warnings = []
    if unknown_patients:
        warnings.append("Exploratory only: patient separation cannot be verified for " + str(dict(unknown_patients)))
    if source_labels:
        warnings.append(f"{source_labels} dataset-source labels authorized for exploratory use; not human-reviewed annotations")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "unknown_patient_counts": dict(unknown_patients), "source_labeled_images": source_labels, "images": len(rows), "corrupt_rows": corrupt, "exact_duplicates": duplicates, "class_distribution": [{"split": split, "label": label, "count": count} for (split, label), count in sorted(counts.items())], "dataset_version": version, "split_ids": split_ids, "snapshot": snapshots}


def audit_config(config, training=False):
    audit = inspect_manifest(config["manifest"], config["labels"], config["require_patient_ids"], training=training, source_labels_authorized=config.get("source_label_training_authorized", False), missing_patient_sources=config.get("missing_patient_sources", []))
    if config.get("expected_dataset_version") and config["expected_dataset_version"] != audit["dataset_version"]:
        audit["errors"].append("Prepared dataset fingerprint changed; create and audit an explicit new dataset version")
    if config.get("require_all_split_classes"):
        for split in ("train", "validation", "test"):
            present = {row["label"] for row in audit["snapshot"] if row["split"] == split}
            if present != set(config["labels"]):
                audit["errors"].append(f"Every configured class must occur in {split}")
    audit["valid"] = not audit["errors"]
    return audit


def scan_folder(folder, destination, source):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    rows = [{"image_path": str(path.resolve()), "label": "", "source": source, "patient_id": "", "lesion_id": "", "split": "unassigned", "approved": "false"} for path in sorted(Path(folder).rglob("*")) if path.suffix.lower() in extensions]
    write_manifest(destination, rows)
    return {"images": len(rows), "manifest": str(destination), "note": "No labels or patient identities inferred from folder names"}


class ImageDataset:
    def __init__(self, rows, labels, transform, second_transform=None):
        self.rows, self.labels, self.transform, self.second_transform = rows, labels, transform, second_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
            second_tensor = self.second_transform(image.copy()) if self.second_transform is not None else None
        target = self.labels.index(row["label"])
        return (tensor, second_tensor, target) if second_tensor is not None else (tensor, target)
