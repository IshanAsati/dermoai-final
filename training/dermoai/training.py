import gc
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, WeightedRandomSampler

from .common import digest_file, read_json, safe_error, validate_config, write_json
from .dataset import ImageDataset, audit_config, read_manifest, truth
from .evaluation import calculate_metrics, infer_rows, save_reports
from .models import REGISTRY, auxiliary_modules, build_model, device_for, training_mode, transforms_for


RESUME_CONFIG_KEYS = (
    "architecture", "labels", "mode", "image_size", "unfreeze_prefixes", "seed", "learning_rate",
    "backbone_lr_scale", "weight_decay", "batch_size", "accumulation_steps", "class_balance", "sampling",
    "loss", "label_smoothing", "augmentation", "selection_metric", "schedule", "schedule_epochs",
    "warmup_epochs", "early_stopping_patience", "minimum_epochs", "warm_start", "channels_last",
    "exploratory", "source_label_training_authorized", "missing_patient_sources", "auxiliary_heads",
    "cancer_loss_weight", "family_loss_weight", "hard_negative_weight", "hard_negative_margin",
    "hard_negative_pairs", "mixstyle_probability", "mixstyle_alpha", "consistency_weight",
    "consistency_interval", "consistency_temperature", "progressive_unfreeze",
)

FAMILY_BY_LABEL = {
    "BCC": 0, "SCC": 0, "MEL": 0,
    "ACK": 1, "NEV": 1, "SEK": 1, "Vascular": 1,
    "Acne": 2, "Alopecia": 2, "Nail": 2, "Fungal": 2,
    "Healthy": 3,
}


def save_checkpoint(path, state):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Checkpoint replacement is prohibited")
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.rename(temporary, path)


def class_weights(rows, labels, strategy):
    if strategy == "none":
        return None
    if strategy != "sqrt_inverse":
        raise ValueError("Unsupported class balance strategy")
    counts = Counter(row["label"] for row in rows)
    if any(counts[label] == 0 for label in labels):
        raise ValueError("Every class needs training examples for weighting")
    weights = [1 / math.sqrt(counts[label]) for label in labels]
    normalization = sum(counts[label] * weight for label, weight in zip(labels, weights)) / len(rows)
    return torch.tensor([weight / normalization for weight in weights], dtype=torch.float32)


def sample_weights(rows, strategy):
    if strategy == "none":
        return None
    if strategy == "sqrt_inverse":
        keys = [row["label"] for row in rows]
    elif strategy == "source_sqrt_inverse":
        keys = [row["source"] for row in rows]
    elif strategy == "source_class_sqrt_inverse":
        keys = [(row["source"], row["label"]) for row in rows]
    else:
        raise ValueError("Unsupported sampling strategy")
    counts = Counter(keys)
    return torch.tensor([1 / math.sqrt(counts[key]) for key in keys], dtype=torch.double)


def balanced_softmax_loss(logits, targets, class_counts, label_smoothing=0.0, reduction="sum"):
    adjusted_logits = logits + class_counts.log().unsqueeze(0)
    return functional.cross_entropy(adjusted_logits, targets, label_smoothing=label_smoothing, reduction=reduction)


def hard_negative_margin_loss(logits, targets, labels, pairs, margin=0.5):
    total = logits.sum() * 0
    for target_label, negative_labels in pairs.items():
        if target_label not in labels:
            continue
        target_index = labels.index(target_label)
        negative_indices = [labels.index(label) for label in negative_labels if label in labels and label != target_label]
        selected = targets == target_index
        if not negative_indices or not selected.any():
            continue
        correct = logits[selected, target_index]
        hardest_negative = logits[selected][:, negative_indices].max(dim=1).values
        total = total + functional.relu(margin - correct + hardest_negative).sum()
    return total


def phase_for_epoch(config, epoch):
    phases = config.get("progressive_unfreeze", [])
    if not phases:
        return {"mode": config["mode"], "prefixes": config.get("unfreeze_prefixes", [])}
    return next(phase for phase in reversed(phases) if phase["start_epoch"] <= epoch)


def apply_training_phase(model, config, epoch):
    phase = phase_for_epoch(config, epoch)
    mode = phase["mode"]
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    classifier_modules = [model.get_classifier(), *auxiliary_modules(model)]
    for module in classifier_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    if mode == "partial":
        prefixes = phase.get("prefixes", config.get("unfreeze_prefixes", []))
        for prefix in prefixes:
            matched = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix)]
            if not matched:
                raise ValueError(f"No parameters match progressive prefix: {prefix}")
            for parameter in matched:
                parameter.requires_grad = True
    return {"mode": mode, "start_epoch": phase.get("start_epoch", 1), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}


def training_outputs(model, images):
    features = model.forward_features(images)
    embedding = model.forward_head(features, pre_logits=True)
    logits = model.get_classifier()(embedding)
    cancer_logits = model.aux_cancer_head(embedding) if hasattr(model, "aux_cancer_head") else None
    family_logits = model.aux_family_head(embedding) if hasattr(model, "aux_family_head") else None
    return logits, cancer_logits, family_logits


def epoch_learning_rate(config, epoch):
    base = config["learning_rate"]
    if config.get("schedule", "constant") == "constant":
        return base
    warmup = config.get("warmup_epochs", 0)
    if epoch <= warmup:
        return base * epoch / warmup
    progress = min(1.0, max(0.0, (epoch - warmup - 1) / max(1, config["schedule_epochs"] - warmup - 1)))
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


def selection_score(metrics, strategy):
    if strategy == "macro_f1":
        return metrics["macro_f1"]
    if strategy != "balanced_cancer_safety":
        raise ValueError("Unsupported checkpoint selection strategy")
    sensitivity = metrics.get("cancer_sensitivity")
    specificity = metrics.get("cancer_specificity")
    if sensitivity is None or specificity is None or sensitivity + specificity == 0:
        return metrics["macro_f1"]
    safety_f1 = 2 * sensitivity * specificity / (sensitivity + specificity)
    return 0.5 * metrics["macro_f1"] + 0.5 * safety_f1


def load_warm_start(model, config, dataset_version):
    checkpoint = Path(config["warm_start"]).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("architecture", "labels"):
        if state["config"][key] != config[key]:
            raise ValueError(f"Warm-start configuration mismatch: {key}")
    if state["dataset_version"] != dataset_version:
        raise ValueError("Warm-start dataset differs from the immutable training snapshot")
    if config.get("auxiliary_heads"):
        result = model.load_state_dict(state["model"], strict=False)
        allowed_missing = {
            "aux_cancer_head.weight", "aux_cancer_head.bias",
            "aux_family_head.weight", "aux_family_head.bias",
        }
        if set(result.missing_keys) - allowed_missing or result.unexpected_keys:
            raise RuntimeError(f"Warm-start weights are incompatible: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    else:
        model.load_state_dict(state["model"], strict=True)
    return {"checkpoint": str(checkpoint), "sha256": digest_file(checkpoint), "epoch": state["epoch"]}


def should_stop(config, epoch, best_epoch):
    patience = config.get("early_stopping_patience", 0)
    return bool(patience and epoch >= config.get("minimum_epochs", 0) and epoch - best_epoch >= patience)


def train(run_dir, resume=None):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    validate_config(config)
    device = device_for(config)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    audit = audit_config(config, training=True)
    if not audit["valid"]:
        raise ValueError("Dataset validation failed: " + "; ".join(audit["errors"][:8]))
    write_json(run_dir / "dataset_snapshot.json", audit)
    model = build_model(config, load_pretrained=False if resume or config.get("warm_start") else None)
    warm_start = load_warm_start(model, config, audit["dataset_version"]) if config.get("warm_start") and not resume else None
    model.to(device)
    if device.type == "cuda" and config.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    classifier_parameters = {id(parameter) for module in [model.get_classifier(), *auxiliary_modules(model)] for parameter in module.parameters()}
    optimizable = list(model.parameters()) if config.get("progressive_unfreeze") else [parameter for parameter in model.parameters() if parameter.requires_grad]
    classifier = [parameter for parameter in optimizable if id(parameter) in classifier_parameters]
    backbone = [parameter for parameter in optimizable if id(parameter) not in classifier_parameters]
    parameter_groups = []
    if backbone:
        parameter_groups.append({"params": backbone, "lr_scale": config.get("backbone_lr_scale", 1.0)})
    if classifier:
        parameter_groups.append({"params": classifier, "lr_scale": 1.0})
    optimizer = torch.optim.AdamW(parameter_groups, lr=config["learning_rate"], weight_decay=config["weight_decay"], fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_score, best_epoch = 0, -1.0, 0
    best_checkpoint = None
    batch_size = config["batch_size"]
    effective_batch = batch_size * config["accumulation_steps"]
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        for key in RESUME_CONFIG_KEYS:
            if state["config"].get(key) != config.get(key):
                raise ValueError(f"Resume configuration mismatch: {key}")
        if state["dataset_version"] != audit["dataset_version"]:
            raise ValueError("Dataset changed since checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, best_score, best_epoch = state["epoch"], state["best_score"], state["best_epoch"]
        best_checkpoint = state.get("best_checkpoint")
        if not best_checkpoint and best_epoch:
            candidate = Path(resume).parent / f"epoch-{best_epoch:04d}.pt"
            if candidate.exists():
                best_checkpoint = str(candidate.resolve())
        batch_size = state.get("batch_size", batch_size)
    rows = read_manifest(config["manifest"])
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    weights = class_weights(train_rows, config["labels"], config.get("class_balance", "none"))
    sampling_weights = sample_weights(train_rows, config.get("sampling", "none"))
    counts = Counter(row["label"] for row in train_rows)
    class_counts = torch.tensor([counts[label] for label in config["labels"]], dtype=torch.float32, device=device)
    cancer_lookup = torch.tensor([int(label in config["high_risk_classes"]) for label in config["labels"]], dtype=torch.long, device=device)
    family_lookup = torch.tensor([FAMILY_BY_LABEL[label] for label in config["labels"]], dtype=torch.long, device=device)
    if weights is not None:
        weights = weights.to(device)
    primary_transform = transforms_for(model, config, training=True)
    second_transform = transforms_for(model, config, training=True) if config.get("consistency_weight", 0.0) else None
    dataset = ImageDataset(train_rows, config["labels"], primary_transform, second_transform)
    started = time.time()
    status = {"state": "running", "pid": os.getpid(), "architecture": config["architecture"], "pretrained_checkpoint": REGISTRY[config["architecture"]], "pretrained_used": config["pretrained"], "dataset_version": audit["dataset_version"], "split_ids": audit["split_ids"], "epoch": start_epoch, "epochs": config["epochs"], "best_epoch": best_epoch, "synthetic": config.get("synthetic", False), "checkpoints": [], "resumed_from": str(resume) if resume else None}
    status.update(exploratory=config.get("exploratory", False), data_warnings=audit["warnings"], class_weights=weights.tolist() if weights is not None else None, sampling=config.get("sampling", "none"))
    status.update(warm_start=warm_start, train_images=len(train_rows), validation_images=len(validation_rows), trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), loss=config.get("loss", "cross_entropy"), auxiliary_heads=config.get("auxiliary_heads", False))
    write_json(run_dir / "status.json", status)

    def snapshot(epoch):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "best_checkpoint": best_checkpoint, "config": config, "dataset_version": audit["dataset_version"], "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None, "batch_size": batch_size}

    rollback = run_dir / "initial.pt"
    if warm_start:
        status.update(phase="baseline_validation", batch=0, batches=0)
        write_json(run_dir / "status.json", status)
        targets_list, predicted = infer_rows(model, config, validation_rows, device)
        metrics = calculate_metrics(targets_list, predicted, config["labels"], config["high_risk_classes"], [truth(row.get("smartphone")) for row in validation_rows])
        best_score = selection_score(metrics, config.get("selection_metric", "macro_f1"))
        best_checkpoint = str(rollback.resolve())
        metrics.update(epoch=0, split="validation", split_id=audit["split_ids"]["validation"], dataset_version=audit["dataset_version"], selection_score=best_score, selection_metric=config.get("selection_metric", "macro_f1"))
        save_reports(run_dir / "validation-0000", metrics)
        status.update(best_selection_score=best_score, validation_metrics=metrics, best_validation_metrics=metrics)
    save_checkpoint(rollback, snapshot(start_epoch))
    status["last_checkpoint"] = str(rollback.resolve())
    status["best_checkpoint"] = best_checkpoint
    stopped = False
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        while True:
            loader = None
            try:
                phase = apply_training_phase(model, config, epoch)
                status.update(phase="training", training_phase=phase, epoch_in_progress=epoch, batch=0, trainable_parameters=phase["trainable_parameters"])
                learning_rate = epoch_learning_rate(config, epoch)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * group.get("lr_scale", 1.0)
                status["learning_rate"] = learning_rate
                status["parameter_group_learning_rates"] = [group["lr"] for group in optimizer.param_groups]
                generator = torch.Generator().manual_seed(config["seed"] + epoch)
                sampler = WeightedRandomSampler(sampling_weights, len(sampling_weights), replacement=True, generator=generator) if sampling_weights is not None else None
                loader_options = {"batch_size": batch_size, "shuffle": sampler is None, "sampler": sampler, "num_workers": config["workers"], "pin_memory": device.type == "cuda", "generator": generator}
                if config["workers"] > 0:
                    loader_options.update(persistent_workers=True, prefetch_factor=2)
                loader = DataLoader(dataset, **loader_options)
                training_mode(model, phase["mode"])
                optimizer.zero_grad(set_to_none=True)
                accumulated, total_loss, seen = 0, 0.0, 0
                for step, batch in enumerate(loader):
                    if (run_dir / "STOP").exists():
                        stopped = True
                        break
                    if len(batch) == 3:
                        images, second_images, targets = batch
                    else:
                        images, targets = batch
                        second_images = None
                    images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    if device.type == "cuda" and config.get("channels_last", False):
                        images = images.contiguous(memory_format=torch.channels_last)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits, cancer_logits, family_logits = training_outputs(model, images)
                        if config.get("loss", "cross_entropy") == "balanced_softmax":
                            loss = balanced_softmax_loss(logits, targets, class_counts, config.get("label_smoothing", 0.0))
                        else:
                            loss = functional.cross_entropy(logits, targets, weight=weights, label_smoothing=config.get("label_smoothing", 0.0), reduction="sum")
                        if cancer_logits is not None and config.get("cancer_loss_weight", 0.0):
                            loss = loss + config["cancer_loss_weight"] * functional.cross_entropy(cancer_logits, cancer_lookup[targets], reduction="sum")
                        if family_logits is not None and config.get("family_loss_weight", 0.0):
                            loss = loss + config["family_loss_weight"] * functional.cross_entropy(family_logits, family_lookup[targets], reduction="sum")
                        if config.get("hard_negative_weight", 0.0):
                            hard_loss = hard_negative_margin_loss(logits, targets, config["labels"], config.get("hard_negative_pairs", {}), config.get("hard_negative_margin", 0.5))
                            loss = loss + config["hard_negative_weight"] * hard_loss
                        consistency_interval = config.get("consistency_interval", 1)
                        if second_images is not None and config.get("consistency_weight", 0.0) and (step + 1) % consistency_interval == 0:
                            second_images = second_images.to(device, non_blocking=True)
                            if device.type == "cuda" and config.get("channels_last", False):
                                second_images = second_images.contiguous(memory_format=torch.channels_last)
                            second_logits = model(second_images)
                            temperature = config.get("consistency_temperature", 1.0)
                            teacher = functional.softmax(logits.detach() / temperature, dim=1)
                            consistency = functional.kl_div(functional.log_softmax(second_logits / temperature, dim=1), teacher, reduction="batchmean") * len(targets) * temperature ** 2
                            loss = loss + config["consistency_weight"] * consistency
                    scaler.scale(loss).backward()
                    accumulated += len(targets)
                    seen += len(targets)
                    total_loss += loss.detach().item()
                    if accumulated >= effective_batch or step == len(loader) - 1:
                        scaler.unscale_(optimizer)
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(accumulated)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        accumulated = 0
                    if step % 10 == 0:
                        status.update(epoch_in_progress=epoch, batch=step + 1, batches=len(loader), batch_size=batch_size, elapsed_seconds=time.time() - started, gpu_allocated_mb=torch.cuda.memory_allocated() / 2**20 if device.type == "cuda" else 0)
                        write_json(run_dir / "status.json", status)
                if stopped:
                    break
                evaluation_config = {**config, "batch_size": batch_size}
                status.update(phase="validation", batch=len(loader), batches=len(loader))
                write_json(run_dir / "status.json", status)
                targets_list, predicted = infer_rows(model, evaluation_config, validation_rows, device)
                metrics = calculate_metrics(targets_list, predicted, config["labels"], config["high_risk_classes"], [truth(row.get("smartphone")) for row in validation_rows])
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                images = targets = logits = loss = loader = None
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise RuntimeError("CUDA OOM at batch size 1; use a smaller architecture, frozen mode or image size. Last complete checkpoint retained") from None
                batch_size = max(1, batch_size // 2)
                state = torch.load(rollback, map_location="cpu", weights_only=True)
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                scaler.load_state_dict(state["scaler"])
                torch.set_rng_state(state["torch_rng"])
                random.setstate(state["python_rng"])
                torch.cuda.set_rng_state_all(state["cuda_rng"])
                status.update(oom_retries=status.get("oom_retries", 0) + 1, batch_size=batch_size)
                write_json(run_dir / "status.json", status)
        if stopped:
            break
        score = selection_score(metrics, config.get("selection_metric", "macro_f1"))
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
        checkpoint = run_dir / f"epoch-{epoch:04d}.pt"
        if improved:
            best_checkpoint = str(checkpoint.resolve())
        save_checkpoint(checkpoint, snapshot(epoch))
        rollback = checkpoint
        metrics.update(epoch=epoch, train_loss=total_loss / seen, split="validation", split_id=audit["split_ids"]["validation"], dataset_version=audit["dataset_version"], synthetic=config.get("synthetic", False), selection_score=score, selection_metric=config.get("selection_metric", "macro_f1"))
        metrics.update(exploratory=config.get("exploratory", False), data_warnings=audit["warnings"], learning_rate=learning_rate)
        save_reports(run_dir / f"validation-{epoch:04d}", metrics)
        status["checkpoints"].append(str(checkpoint.resolve()))
        status.update(epoch=epoch, best_epoch=best_epoch, best_selection_score=best_score, last_checkpoint=str(checkpoint.resolve()), validation_metrics=metrics, elapsed_seconds=time.time() - started, gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0)
        if improved:
            status["best_checkpoint"] = str(checkpoint.resolve())
            status["best_validation_metrics"] = metrics
        write_json(run_dir / "status.json", status)
        if should_stop(config, epoch, best_epoch):
            status["stop_reason"] = "validation_selection_score_early_stopping"
            break
    status.update(state="stopped" if stopped else "completed", elapsed_seconds=time.time() - started)
    write_json(run_dir / "status.json", status)
    return status


def worker(run_dir, resume=None):
    try:
        return train(run_dir, resume)
    except Exception as error:
        status_path = Path(run_dir) / "status.json"
        status = read_json(status_path) if status_path.exists() else {}
        status.update(state="failed", error=safe_error(error), error_type=type(error).__name__)
        write_json(status_path, status)
        raise
