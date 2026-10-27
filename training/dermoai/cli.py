import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from .common import config_from, read_json, safe_error, write_json


def main():
    parser = argparse.ArgumentParser(description="DermoAI local research workflow")
    parser.add_argument("--config", default="config.json")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "health", "models", "experiments", "compare"):
        commands.add_parser(name)
    audit = commands.add_parser("audit")
    audit.add_argument("--output", default=None)
    audit.add_argument("--training", action="store_true")
    scan = commands.add_parser("scan")
    scan.add_argument("folder")
    scan.add_argument("destination")
    scan.add_argument("--source", required=True)
    download = commands.add_parser("download")
    download.add_argument("handle")
    download.add_argument("destination")
    prepare = commands.add_parser("prepare-pad")
    prepare.add_argument("folder")
    prepare.add_argument("manifest")
    curate = commands.add_parser("curate-import")
    curate.add_argument("audit")
    curate.add_argument("destination")
    create = commands.add_parser("create-project")
    create.add_argument("title")
    create.add_argument("--approve", action="store_true")
    connect = commands.add_parser("connect-backend")
    connect.add_argument("--approve", action="store_true")
    import_command = commands.add_parser("import")
    import_command.add_argument("--media-url", default="http://127.0.0.1:9090")
    import_command.add_argument("--approve", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("destination")
    sync = commands.add_parser("sync-annotations")
    sync.add_argument("destination")
    sync.add_argument("--approved-task-ids", required=True)
    train = commands.add_parser("train")
    train.add_argument("--approve", action="store_true")
    train.add_argument("--resume")
    train.add_argument("--publish-predictions", action="store_true")
    for command in ("status", "stop"):
        commands.add_parser(command).add_argument("run_id")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluate.add_argument("--output", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("image")
    push = commands.add_parser("push-predictions")
    push.add_argument("--approve", action="store_true")
    backend = commands.add_parser("serve")
    backend.add_argument("--port", type=int, default=9090)
    worker = commands.add_parser("worker")
    worker.add_argument("run_dir")
    worker.add_argument("--resume")
    args = parser.parse_args()
    try:
        if args.command == "worker":
            from .training import worker
            worker(args.run_dir, args.resume)
            return
        if args.command == "health":
            import requests
            import torch
            result = {"python": sys.version, "torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "api_key_configured": bool(os.environ.get("LABEL_STUDIO_API_KEY"))}
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                result.update(gpu_free_mb=free / 2**20, gpu_total_mb=total / 2**20)
            try:
                response = requests.get("http://127.0.0.1:8080/api/version", timeout=5)
                result["label_studio"] = response.json().get("release")
            except requests.RequestException:
                result["label_studio"] = "unreachable"
        elif args.command == "models":
            import timm
            from .models import REGISTRY
            result = {key: {"checkpoint": value, "registered_pretrained": value in timm.list_models(pretrained=True, include_tags=True)} for key, value in REGISTRY.items()}
        elif args.command == "download":
            from .acquisition import kaggle_download
            result = kaggle_download(args.handle, args.destination)
        elif args.command == "prepare-pad":
            from .acquisition import prepare_pad
            result = prepare_pad(args.folder, args.manifest)
        elif args.command == "curate-import":
            from .acquisition import curate_import
            result = curate_import(args.audit, args.destination)
        elif args.command == "scan":
            from .dataset import scan_folder
            result = scan_folder(args.folder, args.destination, args.source)
        else:
            config = config_from(args.config)
            if args.command == "audit":
                from .dataset import audit_config
                result = audit_config(config, training=args.training)
                if args.output:
                    write_json(args.output, result)
                result = {key: value for key, value in result.items() if key != "snapshot"}
            elif args.command in ("inspect", "create-project", "connect-backend", "import", "export", "sync-annotations", "push-predictions"):
                from .client import Studio
                studio = Studio(config)
                if args.command == "inspect":
                    result = studio.inspect()
                elif args.command == "create-project":
                    result = studio.create_project(args.title, args.approve)
                elif args.command == "connect-backend":
                    result = studio.connect_backend(args.approve)
                elif args.command == "import":
                    result = studio.import_manifest(config["manifest"], args.media_url, args.approve)
                elif args.command == "export":
                    result = studio.export(args.destination)
                elif args.command == "sync-annotations":
                    result = studio.annotations_to_manifest(config["manifest"], args.destination, set(read_json(args.approved_task_ids)))
                else:
                    from .inference import Predictor
                    predictor = Predictor(config)
                    predictions = ({"task": task.id, **predictor.task_prediction(task.model_dump())} for task in studio.tasks())
                    result = studio.push_predictions(predictions, args.approve)
            elif args.command in ("train", "status", "stop", "experiments", "compare"):
                from .experiments import Experiments
                experiments = Experiments(config["runtime"])
                if args.command == "train":
                    if args.publish_predictions:
                        from .client import Studio
                        studio = Studio(config)
                    result = experiments.start(config, args.approve, args.resume)
                    if args.publish_predictions:
                        import time
                        from .client import Studio
                        from .inference import Predictor
                        while True:
                            status = experiments.status(result["run_id"])
                            if status["state"] not in ("starting", "running"):
                                break
                            time.sleep(2)
                        if status["state"] != "completed":
                            raise RuntimeError("Training did not complete; no predictions published")
                        predictor = Predictor({**config, "checkpoint": status["best_checkpoint"]})
                        predictions = ({"task": task.id, **predictor.task_prediction(task.model_dump())} for task in studio.tasks())
                        result["publication"] = studio.push_predictions(predictions, approved=True)
                elif args.command == "status":
                    result = experiments.status(args.run_id)
                elif args.command == "stop":
                    result = experiments.stop(args.run_id)
                elif args.command == "compare":
                    groups = {}
                    for run in experiments.list():
                        if run.get("validation_metrics"):
                            key = run["validation_metrics"]["split_id"]
                            groups.setdefault(key, []).append({"run_id": run["run_id"], "architecture": run["architecture"], "macro_f1": run["best_macro_f1"], "synthetic": run.get("synthetic", False)})
                    result = {"comparison_groups_by_identical_validation_split": groups}
                else:
                    result = experiments.list()
            elif args.command == "evaluate":
                from .evaluation import evaluate_checkpoint
                result = evaluate_checkpoint(args.checkpoint, config["manifest"], args.split, args.output)
            elif args.command == "predict":
                from .inference import Predictor
                result = Predictor(config).predict(args.image)
            elif args.command == "serve":
                from .backend import serve
                serve(config, args.port)
                return
        print(json.dumps(result, indent=2, default=str))
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__, "error": safe_error(error)}), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
