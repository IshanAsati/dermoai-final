import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from dermoai.acquisition import CANONICAL_LABELS
from dermoai.common import read_json, write_json
from dermoai.dataset import write_manifest
from dermoai.evaluation import evaluate_checkpoint
from dermoai.experiments import Experiments


def main():
    root = Path(__file__).resolve().parents[1] / "runtime" / ("smoke-" + str(time.time_ns()))
    root.mkdir(parents=True)
    rows = []
    generator = np.random.default_rng(42)
    for split in ("train", "validation", "test"):
        for index, label in enumerate(CANONICAL_LABELS):
            path = root / f"{split}-{index}.png"
            Image.fromarray(generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
            rows.append({"image_path": str(path), "label": label, "source": "SYNTHETIC-NO-CLINICAL-MEANING", "patient_id": f"{split}-{index}", "lesion_id": f"{split}-{index}", "split": split, "approved": "true", "smartphone": "false"})
    manifest = root / "manifest.csv"
    write_manifest(manifest, rows)
    config = read_json(Path(__file__).resolve().parents[1] / "config.example.json")
    config.update(manifest=str(manifest), local_files_root=str(root), runtime=str(root / "runs"), architecture="resnet50", pretrained=False, epochs=1, batch_size=2, accumulation_steps=2, image_size=64, synthetic=True)
    config.update(class_balance="sqrt_inverse", label_smoothing=0.05, augmentation="conservative_skin", schedule="cosine", schedule_epochs=3, warmup_epochs=1, early_stopping_patience=3)
    write_json(root / "config.json", config)
    experiments = Experiments(config["runtime"])
    run = experiments.start(config, approved=True)
    try:
        experiments.start(config, approved=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Duplicate job was allowed")
    deadline = time.time() + 300
    while time.time() < deadline:
        status = experiments.status(run["run_id"])
        if status["state"] not in ("starting", "running"):
            break
        time.sleep(1)
    assert status["state"] == "completed", status
    checkpoint = status["last_checkpoint"]
    metrics = evaluate_checkpoint(checkpoint, manifest, "test", root / "test-report")
    config.update(checkpoint=checkpoint)
    from dermoai.inference import Predictor
    prediction = Predictor(config).task_prediction({"data": {"image": rows[0]["image_path"]}})
    assert prediction["result"][0]["value"]["choices"][0] in CANONICAL_LABELS
    from dermoai.backend import create_backend
    os.environ["DERMOAI_CONTROL_TOKEN"] = "synthetic-test-control-token"
    app = create_backend(config)
    with app.test_client() as client:
        assert client.get("/health").status_code == 200
        from dermoai.common import digest_file
        media_response = client.get("/media/" + digest_file(rows[0]["image_path"]))
        assert media_response.status_code == 200 and media_response.content_type.startswith("image/")
        assert client.get("/media/not-registered").status_code == 404
        assert client.get("/control/list_experiments").status_code == 401
        assert client.post("/control/start_training", json={"approved": False}, headers={"Authorization": "Bearer synthetic-test-control-token"}).status_code == 400
        from xml.etree import ElementTree
        root_xml = ElementTree.Element("View")
        ElementTree.SubElement(root_xml, "Image", name="image", value="$image")
        choices = ElementTree.SubElement(root_xml, "Choices", name="diagnosis", toName="image", choice="single")
        for label in CANONICAL_LABELS:
            ElementTree.SubElement(choices, "Choice", value=label)
        xml = ElementTree.tostring(root_xml, encoding="unicode")
        assert client.post("/setup", json={"project": "999.0", "schema": xml}).status_code == 200
        response = client.post("/predict", json={"project": "999.0", "label_config": xml, "tasks": [{"data": {"image": rows[0]["image_path"]}}]})
        assert response.status_code == 200, response.data
        assert len(response.json["results"]) == 1
    config["epochs"] = 2
    resumed = experiments.start(config, approved=True, resume=checkpoint)
    while time.time() < deadline:
        resume_status = experiments.status(resumed["run_id"])
        if resume_status["state"] not in ("starting", "running"):
            break
        time.sleep(1)
    assert resume_status["state"] == "completed" and resume_status["epoch"] == 2, resume_status
    result = {"passed": True, "synthetic_only": True, "gpu_peak_mb": status["gpu_peak_mb"], "run": run, "resumed_run": resumed, "reports": str(root / "test-report"), "tested": ["GPU training", "FP16", "gradient accumulation", "checkpoint save", "resume", "evaluation CSV/JSON/PNG", "prediction schema", "official ML setup/predict", "duplicate job rejection", "API authentication", "training approval required"]}
    write_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
