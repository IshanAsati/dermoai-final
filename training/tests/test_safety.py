import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from dermoai.common import validate_config
from dermoai.dataset import inspect_manifest, write_manifest
from dermoai.evaluation import calculate_metrics
from dermoai.experiments import Experiments


class DataSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = []
        for index, split in enumerate(("train", "validation", "test")):
            image = self.root / f"{index}.png"
            Image.new("RGB", (16, 16), (index * 40, 20, 50)).save(image)
            self.rows.append({"image_path": str(image), "label": "BCC", "source": "fixture", "patient_id": str(index), "lesion_id": str(index), "split": split, "approved": "true"})

    def tearDown(self):
        self.temporary.cleanup()

    def audit(self):
        manifest = self.root / "manifest.csv"
        write_manifest(manifest, self.rows)
        return inspect_manifest(manifest, ["BCC"])

    def test_patient_crosses_splits(self):
        self.rows[1]["patient_id"] = self.rows[0]["patient_id"]
        self.assertTrue(any("Cross-split patient" in error for error in self.audit()["errors"]))

    def test_exact_duplicate_crosses_sources(self):
        self.rows[1].update(image_path=self.rows[0]["image_path"], source="other-source")
        self.assertTrue(any("Exact image duplicate crosses splits" in error for error in self.audit()["errors"]))

    def test_original_validation_cannot_be_training(self):
        self.rows[0]["original_validation"] = "true"
        self.assertFalse(self.audit()["valid"])

    def test_unknown_patients_fail_closed(self):
        self.rows[0]["patient_id"] = ""
        self.assertFalse(self.audit()["valid"])

    def test_source_test_split_cannot_become_training(self):
        self.rows[0]["source_split"] = "test"
        self.assertFalse(self.audit()["valid"])

    def test_sdk_errors_do_not_log_response_cookies(self):
        from dermoai.common import safe_error
        from label_studio_sdk.core.api_error import ApiError
        error = ApiError(status_code=403, headers={"set-cookie": "sessionid=secret-fixture"}, body={"detail": "Forbidden"})
        self.assertEqual(safe_error(error), "HTTP 403: Forbidden")

    def test_atomic_json_write_retries_transient_windows_lock(self):
        from dermoai.common import write_json
        import dermoai.common as common
        destination = self.root / "status.json"
        real_replace = common.os.replace
        attempts = 0

        def locked_twice(source, target):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise PermissionError("transient reader lock")
            return real_replace(source, target)

        with patch.object(common.os, "replace", side_effect=locked_twice), patch.object(common.time, "sleep"):
            write_json(destination, {"state": "running"})
        self.assertEqual(destination.read_text(encoding="utf-8"), '{\n  "state": "running"\n}')
        self.assertEqual(attempts, 3)

    def test_corrupt_images_fail(self):
        Path(self.rows[0]["image_path"]).write_bytes(b"invalid image")
        self.assertEqual(self.audit()["corrupt_rows"], [0])

    def test_same_image_conflicting_labels_fails_even_within_split(self):
        self.rows[1].update(image_path=self.rows[0]["image_path"], label="MEL", split="train")
        self.assertTrue(any("Conflicting labels" in error for error in self.audit()["errors"]))

    def test_split_signature_changes_when_label_changes(self):
        first = self.audit()
        (self.root / "manifest.csv").unlink()
        self.rows[1]["label"] = "MEL"
        second = self.audit()
        self.assertNotEqual(first["split_ids"]["validation"], second["split_ids"]["validation"])

    def test_training_requires_approval(self):
        with self.assertRaises(PermissionError):
            Experiments(self.root / "runs").start({})

    def test_cancer_metrics_and_undefined_denominator(self):
        metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 0, 1], ["BCC", "Healthy"], ["BCC"], [True] * 4)
        self.assertEqual(metrics["cancer_sensitivity"], 0.5)
        self.assertEqual(metrics["cancer_specificity"], 0.5)
        self.assertEqual(metrics["false_negatives"], 1)
        undefined = calculate_metrics([0], [0], ["BCC", "Healthy"], ["BCC"], [False])
        self.assertIsNone(undefined["cancer_specificity"])
        self.assertIsNone(undefined["smartphone_accuracy"])

    def test_canonical_class_order_enforced(self):
        from dermoai.acquisition import CANONICAL_LABELS
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_config({"labels": list(reversed(CANONICAL_LABELS))})


if __name__ == "__main__":
    unittest.main()
