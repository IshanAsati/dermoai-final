import hashlib
from pathlib import Path

import timm
from timm.data import create_transform, resolve_model_data_config


MODEL_NAMES = {
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(config, load_pretrained=False):
    architecture = config.get("architecture")
    if architecture not in MODEL_NAMES:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    return timm.create_model(
        MODEL_NAMES[architecture],
        pretrained=load_pretrained,
        num_classes=len(config["labels"]),
    )


def transforms_for(model, config, training=False):
    if training:
        raise ValueError("The packaged model tester only supports inference transforms")
    data = resolve_model_data_config(model)
    data["input_size"] = (3, config["image_size"], config["image_size"])
    if config.get("augmentation") in ("conservative_skin", "aggressive_smartphone_skin"):
        data.update(crop_pct=1.0, crop_mode="border")
    return create_transform(**data, is_training=False)
