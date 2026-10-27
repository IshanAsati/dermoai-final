import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        Path(temporary).unlink(missing_ok=True)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_from(path):
    path = Path(path).resolve()
    config = read_json(path)
    for key in ("manifest", "media_manifest", "local_files_root", "runtime", "checkpoint", "warm_start"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    config["config_path"] = str(path)
    return config


def validate_config(config):
    from .acquisition import CANONICAL_LABELS
    labels = config.get("labels", [])
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Configure at least two unique, verified class names")
    if not config.get("synthetic") and labels != CANONICAL_LABELS:
        raise ValueError("DermoAI requires the exact canonical 12-class order")
    if not set(config.get("high_risk_classes", [])).issubset(labels):
        raise ValueError("High-risk classes must be an explicit subset of labels")
    for key in ("epochs", "batch_size", "accumulation_steps", "image_size"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("device") != "cuda" and not config.get("synthetic"):
        raise ValueError("Real training requires CUDA; CPU is only allowed for explicit synthetic tests")
    if config.get("mode") not in ("frozen", "partial", "full"):
        raise ValueError("mode must be frozen, partial or full")
    if config.get("learning_rate", 0) <= 0 or config.get("workers", 0) < 0:
        raise ValueError("Invalid learning rate or worker count")
    if (config.get("source_label_training_authorized") or config.get("missing_patient_sources")) and not config.get("exploratory"):
        raise ValueError("Source-label or missing-patient exceptions require explicit exploratory mode")
    if not isinstance(config.get("missing_patient_sources", []), list):
        raise ValueError("missing_patient_sources must be an explicit list")
    if config.get("class_balance", "none") not in ("none", "sqrt_inverse"):
        raise ValueError("Unsupported class balance strategy")
    if config.get("sampling", "none") not in ("none", "sqrt_inverse", "source_sqrt_inverse", "source_class_sqrt_inverse"):
        raise ValueError("Unsupported sampling strategy")
    if config.get("loss", "cross_entropy") not in ("cross_entropy", "balanced_softmax"):
        raise ValueError("Unsupported classification loss")
    if config.get("augmentation") not in (None, "conservative_skin", "aggressive_smartphone_skin"):
        raise ValueError("Unsupported augmentation strategy")
    if config.get("selection_metric", "macro_f1") not in ("macro_f1", "balanced_cancer_safety"):
        raise ValueError("Unsupported checkpoint selection strategy")
    if not 0 <= config.get("label_smoothing", 0.0) < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if config.get("early_stopping_patience", 0) < 0:
        raise ValueError("early_stopping_patience must be nonnegative")
    minimum_epochs = config.get("minimum_epochs", 0)
    if not isinstance(minimum_epochs, int) or not 0 <= minimum_epochs <= config["epochs"]:
        raise ValueError("minimum_epochs must be between zero and epochs")
    if config.get("schedule", "constant") not in ("constant", "cosine"):
        raise ValueError("Unsupported learning-rate schedule")
    if config.get("schedule") == "cosine" and config.get("schedule_epochs", 0) < config["epochs"]:
        raise ValueError("Cosine schedule_epochs must cover all configured epochs")
    if not 0 <= config.get("mixstyle_probability", 0.0) <= 1:
        raise ValueError("mixstyle_probability must be in [0, 1]")
    if config.get("mixstyle_probability", 0.0) and config.get("mixstyle_alpha", 0.0) <= 0:
        raise ValueError("mixstyle_alpha must be positive when MixStyle is enabled")
    for key in ("cancer_loss_weight", "family_loss_weight", "hard_negative_weight", "consistency_weight"):
        if config.get(key, 0.0) < 0:
            raise ValueError(f"{key} must be nonnegative")
    if config.get("consistency_weight", 0.0) and config.get("consistency_interval", 0) < 1:
        raise ValueError("consistency_interval must be positive when consistency training is enabled")
    phases = config.get("progressive_unfreeze", [])
    if phases:
        starts = [phase.get("start_epoch") for phase in phases]
        if starts != sorted(starts) or starts[0] != 1 or any(not isinstance(start, int) for start in starts):
            raise ValueError("progressive_unfreeze phases must start at epoch 1 in ascending order")
        if any(phase.get("mode") not in ("frozen", "partial", "full") for phase in phases):
            raise ValueError("Progressive phases require frozen, partial or full mode")


def safe_error(error):
    if hasattr(error, "status_code"):
        body = getattr(error, "body", None)
        detail = body.get("detail", "API request failed") if isinstance(body, dict) else "API request failed"
        message = f"HTTP {error.status_code}: {detail}"
    else:
        message = str(error)
    for name in ("LABEL_STUDIO_API_KEY", "DERMOAI_CONTROL_TOKEN", "KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "<REDACTED>")
    return re.sub(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "<REDACTED>", message)
