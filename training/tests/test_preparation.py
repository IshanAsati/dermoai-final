import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from dermoai.common import digest_file
from dermoai.dataset import audit_config, inspect_manifest, write_manifest
from dermoai.preparation import prepare_rows
from dermoai.similarity import similarity_scan
from dermoai.training import balanced_softmax_loss, class_weights, epoch_learning_rate, hard_negative_margin_loss, phase_for_epoch, sample_weights, selection_score


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def row(self, name, source="PAD-UFES-20", split="train", label="BCC", patient="patient"):
        path = self.root / name
        generator = np.random.default_rng(sum(name.encode()))
        Image.fromarray(generator.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
        return {"image_path": str(path), "source": source, "split": split, "source_split": "", "label": label, "patient_id": patient, "approved": "false", "label_status": "dataset_source_unreviewed"}

    def scan(self, rows, pairs):
        return {"records": [{"image_path": row["image_path"], "sha256": digest_file(row["image_path"])} for row in rows], "pairs": pairs}

    def test_source_labels_need_explicit_authorization(self):
        rows = [self.row("train.png"), self.row("validation.png", split="validation", patient="other")]
        path = self.root / "manifest.csv"
        write_manifest(path, rows)
        self.assertFalse(inspect_manifest(path, ["BCC"], training=True)["valid"])
        audit = inspect_manifest(path, ["BCC"], training=True, source_labels_authorized=True)
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["warnings"])
        self.assertFalse(rows[0]["approved"] == "true")

    def test_missing_patient_exception_is_source_specific(self):
        rows = [self.row("missing.png", patient="")]
        path = self.root / "manifest.csv"
        write_manifest(path, rows)
        self.assertFalse(inspect_manifest(path, missing_patient_sources=["DermNet-Kaggle"])["valid"])

    def test_unknown_patient_warning_is_retained(self):
        rows = [self.row("dermnet.png", source="DermNet-Kaggle", patient="")]
        path = self.root / "manifest.csv"
        write_manifest(path, rows)
        audit = inspect_manifest(path, missing_patient_sources=["DermNet-Kaggle"])
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["unknown_patient_counts"], {"DermNet-Kaggle": 1})

    def test_heldout_similarity_excludes_training_without_moving_test(self):
        rows = [self.row("train.png"), self.row("test.png", split="test", patient="other")]
        selected, excluded = prepare_rows(rows, self.scan(rows, [{"left": 0, "right": 1}]))
        self.assertEqual([row["split"] for row in selected], ["test"])
        self.assertEqual(len(excluded), 1)

    def test_source_test_never_becomes_training(self):
        rows = [self.row("dermnet.png", source="DermNet-Kaggle", split="unassigned", patient="")]
        rows[0]["source_split"] = "test"
        selected, excluded = prepare_rows(rows, self.scan(rows, []))
        self.assertEqual(selected[0]["split"], "test")
        self.assertFalse(excluded)

    def test_conflicting_similarity_is_quarantined_not_relabelled(self):
        rows = [self.row("first.png"), self.row("second.png", label="MEL", patient="other")]
        selected, excluded = prepare_rows(rows, self.scan(rows, [{"left": 0, "right": 1}]))
        self.assertFalse(selected)
        self.assertEqual({row["label"] for row in excluded}, {"BCC", "MEL"})

    def test_changed_image_rejects_scan(self):
        rows = [self.row("changed.png")]
        scan = self.scan(rows, [])
        Path(rows[0]["image_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_rows(rows, scan)

    def test_mirrored_similar_image_is_detected(self):
        rows = [self.row("original.png"), self.row("mirror.png", patient="other")]
        with Image.open(rows[0]["image_path"]) as image:
            ImageOps.mirror(image).save(rows[1]["image_path"])
        scan = similarity_scan(rows)
        self.assertTrue(any(pair["left"] == 0 and pair["right"] == 1 for pair in scan["pairs"]))

    def test_class_weights_are_tempered_and_normalized(self):
        rows = [{"label": "BCC"}] * 9 + [{"label": "MEL"}]
        weights = class_weights(rows, ["BCC", "MEL"], "sqrt_inverse")
        self.assertAlmostEqual(float(weights[1] / weights[0]), 3)
        self.assertAlmostEqual(float((9 * weights[0] + weights[1]) / 10), 1)
        self.assertTrue(torch.isfinite(weights).all())

    def test_minority_sampling_is_tempered(self):
        rows = [{"label": "BCC"}] * 9 + [{"label": "MEL"}]
        weights = sample_weights(rows, "sqrt_inverse")
        self.assertAlmostEqual(float(weights[-1] / weights[0]), 3)
        self.assertTrue(torch.isfinite(weights).all())

    def test_source_class_sampling_balances_dataset_groups(self):
        rows = [{"label": "BCC", "source": "large"}] * 9 + [{"label": "BCC", "source": "small"}]
        weights = sample_weights(rows, "source_class_sqrt_inverse")
        self.assertAlmostEqual(float(weights[-1] / weights[0]), 3)

    def test_balanced_softmax_penalizes_rare_class_errors(self):
        logits = torch.zeros((2, 2))
        counts = torch.tensor([9.0, 1.0])
        common = balanced_softmax_loss(logits[:1], torch.tensor([0]), counts)
        rare = balanced_softmax_loss(logits[1:], torch.tensor([1]), counts)
        self.assertGreater(float(rare), float(common))

    def test_hard_negative_margin_targets_configured_confusion(self):
        labels = ["BCC", "SCC", "MEL"]
        logits = torch.tensor([[1.0, 1.2, -2.0], [3.0, 0.0, -2.0]])
        loss = hard_negative_margin_loss(logits, torch.tensor([0, 0]), labels, {"BCC": ["SCC"]}, margin=0.5)
        self.assertAlmostEqual(float(loss), 0.7, places=5)

    def test_progressive_phase_uses_latest_started_stage(self):
        config = {"progressive_unfreeze": [{"start_epoch": 1, "mode": "frozen"}, {"start_epoch": 3, "mode": "partial"}, {"start_epoch": 7, "mode": "full"}]}
        self.assertEqual(phase_for_epoch(config, 1)["mode"], "frozen")
        self.assertEqual(phase_for_epoch(config, 5)["mode"], "partial")
        self.assertEqual(phase_for_epoch(config, 8)["mode"], "full")

    def test_schedule_is_deterministic_with_warmup(self):
        config = {"learning_rate": 0.001, "schedule": "cosine", "warmup_epochs": 2, "schedule_epochs": 20}
        self.assertAlmostEqual(epoch_learning_rate(config, 1), 0.0005)
        self.assertAlmostEqual(epoch_learning_rate(config, 2), 0.001)
        self.assertAlmostEqual(epoch_learning_rate(config, 20), 0.00005)

    def test_checkpoint_selection_balances_cancer_safety(self):
        metrics = {"macro_f1": 0.8, "cancer_sensitivity": 0.9, "cancer_specificity": 0.9}
        self.assertAlmostEqual(selection_score(metrics, "balanced_cancer_safety"), 0.85)

    def test_frozen_fingerprint_changes_fail_preflight(self):
        rows = [self.row("frozen.png")]
        path = self.root / "manifest.csv"
        write_manifest(path, rows)
        config = {"manifest": str(path), "labels": ["BCC"], "require_patient_ids": True, "expected_dataset_version": "wrong"}
        self.assertFalse(audit_config(config)["valid"])

    def test_every_split_class_guard(self):
        rows = [self.row("one-class.png")]
        path = self.root / "manifest.csv"
        write_manifest(path, rows)
        config = {"manifest": str(path), "labels": ["BCC", "MEL"], "require_patient_ids": True, "require_all_split_classes": True}
        self.assertFalse(audit_config(config)["valid"])

    def test_grouped_dermnet_split_is_deterministic(self):
        rows = [self.row(f"sample-{index}.png", source="DermNet-Kaggle", split="unassigned", patient="", label="Acne" if index < 10 else "Nail") for index in range(20)]
        for row in rows:
            row["source_split"] = "train"
        scan = self.scan(rows, [{"left": 0, "right": 1}])
        first, excluded = prepare_rows(rows, scan)
        second, unused = prepare_rows(rows, scan)
        self.assertEqual(first, second)
        self.assertFalse(excluded or unused)
        splits = {row["image_path"]: row["split"] for row in first}
        self.assertEqual(splits[rows[0]["image_path"]], splits[rows[1]["image_path"]])


if __name__ == "__main__":
    unittest.main()
