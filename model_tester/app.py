import argparse
import io
import os
import time
from pathlib import Path
from threading import Lock

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


APP_ROOT = Path(__file__).resolve().parent
CHECKPOINT_CANDIDATES = (
    APP_ROOT.parent / "model_weights" / "dermoai_v2_convnextv2_base.pth",
    APP_ROOT.parent / "runtime" / "20261016-152650-e30bec65" / "epoch-0002.pt",
)
DEFAULT_CHECKPOINT = next((path for path in CHECKPOINT_CANDIDATES if path.is_file()), CHECKPOINT_CANDIDATES[0])
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

from model_runtime import build_model, digest_file, transforms_for


CLASS_DETAILS = (
    {"code": "BCC", "name": "Basal Cell Carcinoma", "group": "cancer-associated"},
    {"code": "ACK", "name": "Actinic Keratosis", "group": "precancer-associated"},
    {"code": "NEV", "name": "Melanocytic Nevus", "group": "other modeled class"},
    {"code": "SEK", "name": "Seborrheic Keratosis", "group": "other modeled class"},
    {"code": "SCC", "name": "Squamous Cell Carcinoma", "group": "cancer-associated"},
    {"code": "MEL", "name": "Melanoma", "group": "cancer-associated"},
    {"code": "Acne", "name": "Acne and Rosacea", "group": "other modeled class"},
    {"code": "Alopecia", "name": "Alopecia and Hair Loss", "group": "other modeled class"},
    {"code": "Nail", "name": "Nail Fungus and Disease", "group": "other modeled class"},
    {"code": "Fungal", "name": "Fungal Infections", "group": "other modeled class"},
    {"code": "Vascular", "name": "Vascular Tumors", "group": "other modeled class"},
    {"code": "Healthy", "name": "Healthy Skin", "group": "other modeled class"},
)
CANONICAL_LABELS = [entry["code"] for entry in CLASS_DETAILS]


class InferenceEngine:
    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint).resolve()
        self.lock = Lock()
        self.model = None
        self.transform = None
        self.device = None
        self.version = None
        self.image_size = None

    def load(self):
        with self.lock:
            self._load_unlocked()

    def _load_unlocked(self):
        if self.model is not None:
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"V2 checkpoint not found: {self.checkpoint}")

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        config = state.get("config", {})
        if config.get("labels") != CANONICAL_LABELS:
            raise ValueError("Checkpoint class order does not match DermoAI's canonical 12 classes")
        if config.get("synthetic"):
            raise ValueError("Synthetic checkpoints cannot be used for image testing")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(config, load_pretrained=False)
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        self.model = model
        self.transform = transforms_for(model, config, training=False)
        self.device = device
        self.image_size = config["image_size"]
        self.version = f"v2-{digest_file(self.checkpoint)[:12]}"

    def status(self):
        return {
            "available": self.checkpoint.is_file(),
            "loaded": self.model is not None,
            "device": self.device.type.upper() if self.device else None,
            "model_version": self.version,
            "architecture": "ConvNeXt V2 Base",
            "class_count": len(CANONICAL_LABELS),
        }

    def predict_bytes(self, data):
        if not data:
            raise ValueError("The uploaded image is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Image exceeds the 16 MB upload limit")

        try:
            with Image.open(io.BytesIO(data)) as opened:
                image_format = opened.format or "Unknown"
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("That file is not a readable image") from error

        if width < 32 or height < 32:
            raise ValueError("Image must be at least 32 by 32 pixels")
        if width * height > 80_000_000:
            raise ValueError("Image dimensions are too large")

        with self.lock:
            self._load_unlocked()
            tensor = self.transform(image).unsqueeze(0).to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                probabilities = self.model(tensor).float().softmax(dim=1)[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            inference_ms = round((time.perf_counter() - started) * 1000, 1)
            values = probabilities.cpu().tolist()

        ranked = []
        for class_index, (details, probability) in enumerate(zip(CLASS_DETAILS, values)):
            ranked.append(
                {
                    "class_index": class_index,
                    "code": details["code"],
                    "name": details["name"],
                    "group": details["group"],
                    "probability": probability,
                    "percentage": round(probability * 100, 2),
                }
            )
        ranked.sort(key=lambda entry: entry["probability"], reverse=True)
        for rank, entry in enumerate(ranked, start=1):
            entry["rank"] = rank

        top = ranked[0]
        return {
            "prediction": top,
            "predictions": ranked,
            "image": {"width": width, "height": height, "format": image_format},
            "model": {
                "version": self.version,
                "architecture": "ConvNeXt V2 Base",
                "device": self.device.type.upper(),
                "input_size": self.image_size,
                "inference_ms": inference_ms,
            },
            "research_only": True,
            "notice": "Scores are uncalibrated model outputs, not medical probabilities or a diagnosis.",
        }


def create_app(engine=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES, JSON_SORT_KEYS=False)
    prediction_engine = engine or InferenceEngine(os.environ.get("DERMOAI_CHECKPOINT", DEFAULT_CHECKPOINT))
    app.extensions["dermoai_engine"] = prediction_engine

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        status = prediction_engine.status()
        return jsonify({"ok": status["available"], "model": status}), 200 if status["available"] else 503

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify({"error": "Choose an image before running the model."}), 400
        try:
            result = prediction_engine.predict_bytes(upload.stream.read(MAX_UPLOAD_BYTES + 1))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify({"error": "The model could not analyze this image. Check the local server and try again."}), 500
        return jsonify(result)

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return jsonify({"error": "Image exceeds the 16 MB upload limit."}), 413

    @app.after_request
    def security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local DermoAI V2 image tester")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-preload", action="store_true", help="Load the model on the first prediction")
    return parser.parse_args()


def main():
    from waitress import serve

    args = parse_args()
    app = create_app()
    if not args.no_preload:
        app.extensions["dermoai_engine"].load()
    pid_path = APP_ROOT / "server.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        print(f"DermoAI V2 tester: http://{args.host}:{args.port}", flush=True)
        serve(app, host=args.host, port=args.port, threads=4)
    finally:
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink()


if __name__ == "__main__":
    main()
