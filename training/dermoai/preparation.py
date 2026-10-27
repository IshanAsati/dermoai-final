import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from .common import digest_file, read_json, validate_config, write_json
from .dataset import audit_config, read_manifest, truth, write_manifest
from .similarity import connected_groups


def prepare_rows(rows, scan, seed=42):
    if len(rows) != len(scan["records"]):
        raise ValueError("Similarity scan does not match manifest")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        current_hashes = list(executor.map(digest_file, [row["image_path"] for row in rows]))
    for row, record, current_hash in zip(rows, scan["records"], current_hashes):
        if row["image_path"] != record["image_path"] or current_hash != record["sha256"]:
            raise ValueError("An image changed since similarity scanning")
    conflicting = set()
    for pair in scan["pairs"]:
        if rows[pair["left"]]["label"] != rows[pair["right"]]["label"]:
            conflicting.update((pair["left"], pair["right"]))
    selected, excluded, flexible = [], [], []
    for members in connected_groups(rows, scan["pairs"]):
        group_id = hashlib.sha256("|".join(sorted(scan["records"][index]["sha256"] for index in members)).encode()).hexdigest()
        group_rows = []
        for index in members:
            original = rows[index]
            row = {**original, "similarity_group": group_id, "label_status": "human_reviewed" if truth(original["approved"]) else "dataset_source_unreviewed"}
            if original["source"] == "DermNet-Kaggle":
                if original["source_split"] not in ("train", "test"):
                    raise ValueError("Unexpected DermNet source split")
                row["split"] = "test" if original["source_split"] == "test" else "flexible"
            elif original["split"] not in ("train", "validation", "test"):
                raise ValueError("Unexpected source or missing protected split")
            group_rows.append(row)
        if conflicting.intersection(members):
            excluded.extend({**row, "exclusion_reason": "Similarity candidate with conflicting labels in connected group"} for row in group_rows)
            continue
        fixed = {row["split"] for row in group_rows} - {"flexible"}
        if not fixed:
            flexible.extend(group_rows)
            continue
        chosen = "test" if "test" in fixed else "validation" if "validation" in fixed else "train"
        for row in group_rows:
            if row["split"] == chosen or chosen == "train" and row["split"] == "flexible":
                row["split"] = chosen
                selected.append(row)
            else:
                excluded.append({**row, "exclusion_reason": "Connected to protected held-out data; excluded rather than reassigned"})
    if flexible:
        stratifier = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_indices, validation_indices = next(stratifier.split(flexible, [row["label"] for row in flexible], [row["similarity_group"] for row in flexible]))
        for split, indices in (("train", train_indices), ("validation", validation_indices)):
            selected.extend({**flexible[index], "split": split} for index in indices)
    return sorted(selected, key=lambda row: row["image_path"]), excluded


def prepare_project(root, authorize_source_labels=False, accept_missing_patients=False):
    if not authorize_source_labels or not accept_missing_patients:
        raise PermissionError("Explicit authorization for source labels and unknown DermNet patient identities is required")
    root = Path(root).resolve()
    destination = root / "datasets/training-v1.csv"
    config_path = root / "training-config.json"
    if destination.exists() or config_path.exists():
        raise FileExistsError("Prepared dataset/configuration already exists; do not overwrite a frozen split")
    original = read_manifest(root / "datasets/import-manifest.csv")
    scan = read_json(root / "datasets/similarity-audit.json")
    rows, excluded = prepare_rows(original, scan)
    config = read_json(root / "config.json")
    config["media_manifest"] = "datasets/import-manifest.csv"
    config.update(manifest="datasets/training-v1.csv", exploratory=True, source_label_training_authorized=True, missing_patient_sources=["DermNet-Kaggle"], require_patient_ids=True, architecture="convnext_tiny", mode="partial", unfreeze_prefixes=["stages.3.", "head."], epochs=20, early_stopping_patience=5, learning_rate=0.0001, weight_decay=0.01, class_balance="sqrt_inverse", label_smoothing=0.05, augmentation="conservative_skin", schedule="cosine", schedule_epochs=20, warmup_epochs=2, batch_size=2, accumulation_steps=8, image_size=224, seed=42, pretrained=True, checkpoint=None)
    validate_config(config)
    for split in ("train", "validation", "test"):
        if {row["label"] for row in rows if row["split"] == split} != set(config["labels"]):
            raise ValueError(f"Not every class has eligible {split} images after exclusions")
    write_manifest(destination, rows)
    audit = audit_config({**config, "manifest": str(destination)}, training=True)
    write_json(root / "datasets/training-v1-audit.json", audit)
    write_json(root / "datasets/training-v1-exclusions.json", excluded)
    if not audit["valid"]:
        raise ValueError("Prepared dataset failed audit: " + "; ".join(audit["errors"][:8]))
    config.update(expected_dataset_version=audit["dataset_version"], require_all_split_classes=True)
    summary = {"ready_for_exploratory_training": True, "ready_for_clinical_validation": False, "images": len(rows), "excluded_images": len(excluded), "similarity_candidate_pairs": len(scan["pairs"]), "splits": dict(Counter(row["split"] for row in rows)), "class_distribution": audit["class_distribution"], "warnings": audit["warnings"], "dataset_version": audit["dataset_version"], "split_ids": audit["split_ids"], "training_started": False, "annotations_modified": 0, "method": "Preserve PAD/Healthy split membership and DermNet source test membership. Conservatively exclude ambiguous or held-out-related training/validation records. Split remaining DermNet source-training groups using the first prespecified stratified-group fold (five folds, seed 42): approximately 80% training and 20% validation. Original files and import manifest are unchanged.", "evaluation_limitations": ["Unknown DermNet patient identities: similarity grouping cannot guarantee patient separation.", "Source labels are unreviewed and broad categories may contain label noise.", "Classes are strongly confounded with dataset source; internal results do not establish real-world smartphone generalization.", "Healthy and melanoma test sets are small; estimates will be unstable.", "Source license/consent constraints still apply; research-only outputs."]}
    write_json(config_path, config)
    write_json(root / "training-readiness.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--authorize-source-labels", action="store_true")
    parser.add_argument("--accept-missing-patients", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(prepare_project(arguments.root, arguments.authorize_source_labels, arguments.accept_missing_patients), indent=2))
