from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image, ImageOps

from .common import digest_file
from .dataset import read_manifest
from .models import build_model, device_for, transforms_for


class Predictor:
    def __init__(self, config):
        self.config = config
        self.lock = Lock()
        self.model = None
        self.loaded_checkpoint = None
        self.media = {}
        media_manifest = config.get("media_manifest", config["manifest"])
        if Path(media_manifest).exists():
            root = Path(config["local_files_root"]).resolve()
            for row in read_manifest(media_manifest):
                path = Path(row["image_path"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Image is outside local_files_root")
                self.media[digest_file(path)] = path

    def image_path(self, value):
        parsed = urlparse(value)
        if parsed.path.startswith("/media/"):
            digest = parsed.path.rsplit("/", 1)[-1]
            if digest in self.media:
                return self.media[digest]
        if parsed.path == "/data/local-files/":
            relative = parse_qs(parsed.query).get("d", [""])[0]
            candidate = (Path(self.config["local_files_root"]) / relative).resolve()
        elif not parsed.scheme or (len(parsed.scheme) == 1 and Path(value).is_absolute()):
            candidate = Path(value).resolve()
        else:
            raise ValueError("Only registered local images are accepted; remote fetching is disabled")
        if candidate not in self.media.values():
            raise ValueError("Image is not registered in the configured manifest")
        return candidate

    def predict(self, value):
        with self.lock:
            checkpoint = self.config.get("checkpoint")
            if not checkpoint:
                raise ValueError("No trained checkpoint selected; train and explicitly select one first")
            if checkpoint != self.loaded_checkpoint:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if state["config"]["labels"] != self.config["labels"]:
                    raise ValueError("Checkpoint class order differs from the project's canonical order")
                if state["config"].get("synthetic") and not self.config.get("synthetic"):
                    raise ValueError("Synthetic checkpoints cannot be used for real image predictions")
                self.model_config = state["config"]
                self.device = device_for(self.model_config)
                self.model = build_model(self.model_config, load_pretrained=False)
                self.model.load_state_dict(state["model"])
                self.model.to(self.device).eval()
                self.transform = transforms_for(self.model, self.model_config)
                self.loaded_checkpoint = checkpoint
                self.version = "research-" + digest_file(checkpoint)[:16]
            with Image.open(self.image_path(value)) as image:
                tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                probabilities = self.model(tensor).float().softmax(1)[0].cpu().tolist()
            index = max(range(len(probabilities)), key=probabilities.__getitem__)
            return {"label": self.config["labels"][index], "class_index": index, "score": probabilities[index], "probabilities": dict(zip(self.config["labels"], probabilities)), "model_version": self.version, "research_only": True}

    def task_prediction(self, task):
        prediction = self.predict(task["data"][self.config["image_key"]])
        return {"model_version": prediction["model_version"], "score": prediction["score"], "result": [{"from_name": self.config["from_name"], "to_name": self.config["to_name"], "type": "choices", "value": {"choices": [prediction["label"]]}}]}
