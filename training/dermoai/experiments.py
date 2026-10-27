import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
from filelock import FileLock

from .common import read_json, validate_config, write_json


class Experiments:
    def __init__(self, runtime):
        self.root = Path(runtime).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.root / "jobs.lock"), timeout=10)

    def status(self, run_id):
        if Path(run_id).name != run_id or not run_id:
            raise ValueError("Invalid run ID")
        path = self.root / run_id / "status.json"
        status = read_json(path)
        process_path = path.parent / "process.json"
        if process_path.exists():
            process_info = read_json(process_path)
            status.update(pid=process_info["pid"], process_created=process_info["created"])
        if status["state"] in ("starting", "running") and status.get("pid"):
            try:
                process = psutil.Process(status["pid"])
                alive = process.is_running() and abs(process.create_time() - status.get("process_created", process.create_time())) < 2
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                status["state"] = "interrupted"
        return {"run_id": run_id, **status}

    def list(self):
        return [self.status(path.parent.name) for path in sorted(self.root.glob("*/status.json"))]

    def start(self, config, approved=False, resume=None):
        if not approved:
            raise PermissionError("Explicit approval is required to start training")
        validate_config(config)
        with self.lock:
            if any(item["state"] in ("starting", "running") for item in self.list()):
                raise RuntimeError("A training job is already active")
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            run_dir = self.root / run_id
            run_dir.mkdir()
            write_json(run_dir / "config.json", config)
            write_json(run_dir / "status.json", {"state": "starting", "pid": None, "synthetic": config.get("synthetic", False)})
            command = [sys.executable, "-m", "dermoai.cli", "worker", str(run_dir)]
            if resume:
                command.extend(["--resume", str(Path(resume).resolve())])
            environment = os.environ.copy()
            for name in ("LABEL_STUDIO_API_KEY", "DERMOAI_CONTROL_TOKEN", "KAGGLE_API_TOKEN", "KAGGLE_KEY"):
                environment.pop(name, None)
            try:
                with (run_dir / "worker.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=log, env=environment, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                write_json(run_dir / "process.json", {"pid": process.pid, "created": psutil.Process(process.pid).create_time()})
            except Exception:
                write_json(run_dir / "status.json", {"state": "failed", "error": "Unable to launch worker"})
                raise
            return {"run_id": run_id, "run_dir": str(run_dir)}

    def stop(self, run_id):
        status = self.status(run_id)
        if status["state"] not in ("starting", "running"):
            return status
        (self.root / run_id / "STOP").touch()
        return {"run_id": run_id, "state": "stop_requested", "note": "Stops at a batch boundary; resume from last completed epoch"}
