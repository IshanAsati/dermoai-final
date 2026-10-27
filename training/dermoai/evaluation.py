import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .common import write_json
from .dataset import ImageDataset, audit_config, read_manifest, truth
from .models import build_model, device_for, transforms_for

HISTORICAL_BASELINE = {"status": "historical user-reported results; not measured by this pipeline", "accuracy": 0.8279, "cancer_sensitivity": 0.8730, "cancer_specificity": 0.9227, "validation_images_reported": 4555, "comparability": "Original dataset, class mapping and split unavailable; direct comparison is not valid"}


def calculate_metrics(targets, predicted, labels, high_risk, smartphone):
    indices = list(range(len(labels)))
    matrix = confusion_matrix(targets, predicted, labels=indices)
    report = classification_report(targets, predicted, labels=indices, target_names=labels, output_dict=True, zero_division=0)
    result = {"research_only": True, "n": len(targets), "accuracy": float(accuracy_score(targets, predicted)), "macro_f1": float(f1_score(targets, predicted, labels=indices, average="macro", zero_division=0)), "per_class": {label: report[label] for label in labels}, "confusion_matrix": matrix.tolist(), "class_order": labels, "cancer_sensitivity": None, "cancer_specificity": None, "false_positives": None, "false_negatives": None, "smartphone_accuracy": None}
    if high_risk:
        positive = {labels.index(label) for label in high_risk}
        actual = np.isin(targets, list(positive))
        inferred = np.isin(predicted, list(positive))
        true_positive = int(np.sum(actual & inferred))
        true_negative = int(np.sum(~actual & ~inferred))
        false_positive = int(np.sum(~actual & inferred))
        false_negative = int(np.sum(actual & ~inferred))
        result.update(cancer_sensitivity=true_positive / (true_positive + false_negative) if true_positive + false_negative else None, cancer_specificity=true_negative / (true_negative + false_positive) if true_negative + false_positive else None, false_positives=false_positive, false_negatives=false_negative, high_risk_classes=high_risk)
    selected = [index for index, value in enumerate(smartphone) if value]
    if selected:
        result["smartphone_accuracy"] = float(accuracy_score(np.asarray(targets)[selected], np.asarray(predicted)[selected]))
    result["smartphone_n"] = len(selected)
    return result


def infer_rows(model, config, rows, device):
    dataset = ImageDataset(rows, config["labels"], transforms_for(model, config))
    loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["workers"], pin_memory=device.type == "cuda")
    targets, predictions = [], []
    model.eval()
    with torch.inference_mode():
        for images, target in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                scores = model(images.to(device, non_blocking=True))
            targets.extend(target.tolist())
            predictions.extend(scores.argmax(1).cpu().tolist())
    return targets, predictions


def save_reports(destination, metrics, rows=None, targets=None, predicted=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "metrics.json", metrics)
    write_json(destination / "historical_baseline.json", HISTORICAL_BASELINE)
    with (destination / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        for label, values in metrics["per_class"].items():
            writer.writerow({"class": label, **values})
    if rows is not None:
        with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_path", "source", "patient_id", "actual", "prediction"])
            for row, actual, prediction in zip(rows, targets, predicted):
                writer.writerow([row["image_path"], row["source"], row["patient_id"], metrics["class_order"][actual], metrics["class_order"][prediction]])
    figure, axes = plt.subplots(figsize=(9, 8))
    axes.imshow(metrics["confusion_matrix"], cmap="Blues")
    axes.set(xticks=range(len(metrics["class_order"])), yticks=range(len(metrics["class_order"])), xticklabels=metrics["class_order"], yticklabels=metrics["class_order"], xlabel="Predicted", ylabel="Actual", title="Research-only confusion matrix")
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
    for row_index, row in enumerate(metrics["confusion_matrix"]):
        for column_index, count in enumerate(row):
            axes.text(column_index, row_index, str(count), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def evaluate_checkpoint(checkpoint, manifest, split, destination):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation requires validation or test")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    audit = audit_config({**config, "manifest": manifest}, training=True)
    if not audit["valid"]:
        raise ValueError("Dataset validation failed: " + "; ".join(audit["errors"][:8]))
    if audit["dataset_version"] != state["dataset_version"]:
        raise ValueError("Dataset differs from the immutable training snapshot")
    rows = [row for row in read_manifest(manifest) if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split} examples")
    device = device_for(config)
    model = build_model(config, load_pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device)
    targets, predicted = infer_rows(model, config, rows, device)
    metrics = calculate_metrics(targets, predicted, config["labels"], config["high_risk_classes"], [truth(row.get("smartphone")) for row in rows])
    metrics.update(split=split, split_id=audit["split_ids"][split], dataset_version=audit["dataset_version"], checkpoint=str(checkpoint), synthetic=config.get("synthetic", False), exploratory=config.get("exploratory", False), data_warnings=audit["warnings"])
    save_reports(destination, metrics, rows, targets, predicted)
    return metrics
