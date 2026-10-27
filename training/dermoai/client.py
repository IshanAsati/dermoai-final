import json
import os
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

from label_studio_sdk import LabelStudio

from .common import digest_file, safe_error
from .dataset import read_manifest, write_manifest


def local_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1") or parsed.username or parsed.password:
        raise ValueError("Only local HTTP endpoints are supported")
    return url.rstrip("/")


class Studio:
    def __init__(self, config):
        key = os.environ.get("LABEL_STUDIO_API_KEY")
        if not key:
            raise RuntimeError("Set LABEL_STUDIO_API_KEY locally; never put it in config.json")
        self.config = config
        self.client = LabelStudio(base_url=local_url(config["label_studio_url"]), api_key=key)

    def inspect(self):
        projects = []
        for project in self.client.projects.list():
            data = project.model_dump()
            projects.append({key: data.get(key) for key in ("id", "title", "label_config", "task_number", "num_tasks_with_annotations")})
        return {"projects": projects, "dermoai_candidates": [project["id"] for project in projects if "dermo" in project.get("title", "").lower()]}

    def create_project(self, title, approved=False):
        if not approved:
            raise PermissionError("Project creation requires approval")
        config = self.config
        root = ElementTree.Element("View")
        ElementTree.SubElement(root, "Header", value="DermoAI research only — not a clinical diagnosis")
        ElementTree.SubElement(root, "Image", name=config["to_name"], value="$" + config["image_key"])
        choices = ElementTree.SubElement(root, "Choices", name=config["from_name"], toName=config["to_name"], choice="single", showInLine="true")
        for label in config["labels"]:
            ElementTree.SubElement(choices, "Choice", value=label)
        return self.client.projects.create(title=title, label_config=ElementTree.tostring(root, encoding="unicode")).model_dump()

    def tasks(self):
        if not self.config.get("project_id"):
            raise ValueError("Set project_id")
        return self.client.tasks.list(project=self.config["project_id"])

    def connect_backend(self, approved=False):
        if not approved:
            raise PermissionError("Connecting a backend requires approval")
        project = self.config["project_id"]
        for backend in self.client.ml.list(project=project):
            if backend.url.rstrip("/") == "http://127.0.0.1:9090":
                return {"id": backend.id, "already_registered": True, "connected": backend.state == "CO", "state": backend.state}
        try:
            backend = self.client.ml.create(project=project, url="http://127.0.0.1:9090", title="DermoAI local PyTorch - research only", auto_update=False, is_interactive=False, timeout=60)
        except Exception as error:
            for backend in self.client.ml.list(project=project):
                if backend.url.rstrip("/") == "http://127.0.0.1:9090":
                    return {"id": backend.id, "connected": backend.state == "CO", "state": backend.state, "registration_warning": safe_error(error)}
            raise
        return {"id": backend.id, "connected": backend.state == "CO", "state": backend.state}

    def export(self, destination):
        with Path(destination).open("x", encoding="utf-8") as stream:
            stream.write("[")
            for index, task in enumerate(self.tasks()):
                if index:
                    stream.write(",\n")
                stream.write(json.dumps(task.model_dump(mode="json")))
            stream.write("]")
        return {"export": str(destination)}

    def import_manifest(self, manifest, media_url, approved=False):
        if not approved:
            raise PermissionError("Import requires approval")
        media_url = local_url(media_url)
        rows = read_manifest(manifest)
        existing = {task.data.get("dermoai_sha256") for task in self.tasks()}
        pending, imported = [], 0
        for row in rows:
            if row["label"] not in self.config["labels"]:
                raise ValueError("Unknown source label; map it explicitly before import")
            digest = digest_file(row["image_path"])
            if digest in existing:
                continue
            pending.append({"data": {self.config["image_key"]: f"{media_url}/media/{digest}", "dermoai_sha256": digest, "reference_label": row["label"], "source": row["source"], "patient_id": row["patient_id"], "lesion_id": row.get("lesion_id", ""), "split": row["split"], "source_split": row.get("source_split", ""), "original_validation": row.get("original_validation", "false"), "research_only": True}})
            existing.add(digest)
            if len(pending) == 100:
                self.client.projects.import_tasks(id=self.config["project_id"], request=pending)
                imported += len(pending)
                pending = []
        if pending:
            self.client.projects.import_tasks(id=self.config["project_id"], request=pending)
            imported += len(pending)
        return {"imported": imported, "annotations_modified": 0}

    def annotations_to_manifest(self, manifest, destination, approved_task_ids):
        rows = read_manifest(manifest)
        by_digest = {digest_file(row["image_path"]): row for row in rows}
        updated = 0
        for task in self.tasks():
            if task.id not in approved_task_ids:
                continue
            full_task = self.client.tasks.get(id=task.id).model_dump()
            row = by_digest.get(task.data.get("dermoai_sha256"))
            if row is None:
                continue
            labels = set()
            for annotation in full_task.get("annotations", []):
                if annotation.get("was_cancelled"):
                    continue
                for result in annotation.get("result", []):
                    if result.get("from_name") == self.config["from_name"] and result.get("type") == "choices":
                        labels.update(result.get("value", {}).get("choices", []))
            if len(labels) != 1 or not labels.issubset(self.config["labels"]):
                raise ValueError(f"Task {task.id}: missing or conflicting annotations; adjudication required")
            row.update(label=labels.pop(), approved="true", task_id=str(task.id))
            updated += 1
        write_manifest(destination, rows)
        return {"approved_rows": updated, "manifest": str(destination), "annotations_modified": 0}

    def push_predictions(self, predictions, approved=False):
        if not approved:
            raise PermissionError("Publishing predictions requires approval")
        count = 0
        for prediction in predictions:
            self.client.predictions.create(**prediction)
            count += 1
        return {"predictions_created": count, "annotations_modified": 0}
