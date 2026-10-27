import hmac
import logging
import os
import uuid
from pathlib import Path
from xml.etree import ElementTree

from flask import jsonify, request, send_file

from .common import safe_error
from .evaluation import evaluate_checkpoint
from .experiments import Experiments
from .inference import Predictor


def create_backend(config):
    model_dir = Path(config["runtime"]) / "backend-cache"
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MODEL_DIR"] = str(model_dir)
    os.environ["CACHE_TYPE"] = "sqlite"
    from label_studio_ml.api import init_app
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.response import ModelResponse

    logging.getLogger("label_studio_ml").setLevel(logging.WARNING)
    predictor = Predictor(config)
    experiments = Experiments(config["runtime"])

    class DermoAIBackend(LabelStudioMLBase):
        def setup(self):
            if config.get("project_id") and str(self.project_id) != str(config["project_id"]):
                raise ValueError("Backend is bound to a different project")
            if self.label_config:
                root = ElementTree.fromstring(self.label_config)
                choices = root.find(f".//Choices[@name='{config['from_name']}']")
                if choices is None or choices.get("toName") != config["to_name"] or [choice.get("value") for choice in choices.findall("Choice")] != config["labels"]:
                    raise ValueError("Label Studio class order or tag names differ from configuration")
            self.set("model_version", "research-untrained" if not config.get("checkpoint") else "research-" + Path(config["checkpoint"]).stem)

        def predict(self, tasks, context=None, **kwargs):
            if any(run["state"] in ("starting", "running") for run in experiments.list()):
                raise RuntimeError("Training is active; request predictions after it completes")
            return ModelResponse(predictions=[predictor.task_prediction(task) for task in tasks])

        def fit(self, event, data, **kwargs):
            if event == "START_TRAINING":
                return {"state": "approval_required", "instruction": "Use the local start_training API with explicit approval, or CLI train --approve"}
            return {"state": "annotation_event_recorded", "training_started": False}

    app = init_app(DermoAIBackend)
    app.before_request_funcs[None] = [callback for callback in app.before_request_funcs.get(None, []) if callback.__name__ != "log_request_info"]
    app.after_request_funcs[None] = [callback for callback in app.after_request_funcs.get(None, []) if callback.__name__ != "log_response_info"]
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    control_token = os.environ.get("DERMOAI_CONTROL_TOKEN")
    if not control_token:
        raise RuntimeError("Set DERMOAI_CONTROL_TOKEN before starting the local backend")

    @app.before_request
    def local_controls():
        if request.host.split(":", 1)[0] not in ("127.0.0.1", "localhost"):
            return jsonify(error="Invalid local host"), 403
        if request.path.startswith("/control/"):
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, "Bearer " + control_token):
                return jsonify(error="Control token required"), 401

    @app.errorhandler(ValueError)
    @app.errorhandler(PermissionError)
    @app.errorhandler(RuntimeError)
    def controlled_error(error):
        return jsonify(error=safe_error(error)), 409 if isinstance(error, RuntimeError) else 400

    @app.get("/media/<digest>")
    def media(digest):
        path = predictor.media.get(digest)
        if path is None:
            return jsonify(error="Unknown registered image"), 404
        return send_file(path, conditional=True)

    @app.post("/control/start_training")
    def start_training():
        body = request.get_json()
        overrides = body.get("overrides", {})
        allowed = {"architecture", "epochs", "learning_rate", "batch_size", "accumulation_steps", "mode", "unfreeze_prefixes"}
        if set(overrides) - allowed:
            raise ValueError("Unsupported override; edit the local configuration explicitly")
        return jsonify(experiments.start({**config, **overrides}, approved=body.get("approved") is True, resume=body.get("resume"))), 202

    @app.post("/control/stop_training")
    def stop_training():
        return jsonify(experiments.stop(request.get_json()["run_id"]))

    @app.get("/control/get_training_status")
    def get_training_status():
        return jsonify(experiments.status(request.args["run_id"]))

    @app.get("/control/list_experiments")
    def list_experiments():
        return jsonify(experiments.list())

    @app.post("/control/evaluate_model")
    def evaluate_model():
        if any(run["state"] in ("starting", "running") for run in experiments.list()):
            raise RuntimeError("Wait for training to finish before GPU evaluation")
        body = request.get_json()
        destination = Path(config["runtime"]) / ("evaluation-" + uuid.uuid4().hex[:12])
        return jsonify(evaluate_checkpoint(body["checkpoint"], config["manifest"], body.get("split", "validation"), destination))

    @app.post("/control/predict_image")
    def predict_image():
        if any(run["state"] in ("starting", "running") for run in experiments.list()):
            raise RuntimeError("Wait for training to finish before GPU prediction")
        return jsonify(predictor.predict(request.get_json()["image"]))

    return app


def serve(config, port=9090):
    from waitress import serve as waitress_serve
    waitress_serve(create_backend(config), host="127.0.0.1", port=port, threads=4)
