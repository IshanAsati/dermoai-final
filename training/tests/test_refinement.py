import tempfile
import unittest
from pathlib import Path

import torch

from dermoai.training import load_warm_start, should_stop


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "source.pt"
        self.source = torch.nn.Linear(3, 2)
        self.config = {"architecture": "fixture", "labels": ["BCC", "MEL"], "warm_start": str(self.checkpoint)}
        torch.save({"model": self.source.state_dict(), "config": self.config, "dataset_version": "frozen", "epoch": 4}, self.checkpoint)

    def test_warm_start_copies_weights_and_records_source(self):
        target = torch.nn.Linear(3, 2)
        provenance = load_warm_start(target, self.config, "frozen")
        torch.testing.assert_close(target.weight, self.source.weight)
        self.assertEqual(provenance["epoch"], 4)
        self.assertEqual(len(provenance["sha256"]), 64)

    def test_reordered_classes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            load_warm_start(torch.nn.Linear(3, 2), {**self.config, "labels": ["MEL", "BCC"]}, "frozen")

    def test_changed_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset"):
            load_warm_start(torch.nn.Linear(3, 2), self.config, "changed")

    def test_incompatible_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            load_warm_start(torch.nn.Linear(3, 2), {**self.config, "architecture": "other"}, "frozen")

    def test_incomplete_weights_cannot_silently_load(self):
        with self.assertRaises(RuntimeError):
            load_warm_start(torch.nn.Linear(4, 2), self.config, "frozen")

    def test_early_stopping_waits_for_learning_rate_decay(self):
        config = {"early_stopping_patience": 4, "minimum_epochs": 8}
        self.assertFalse(should_stop(config, epoch=5, best_epoch=0))
        self.assertTrue(should_stop(config, epoch=8, best_epoch=0))
        self.assertFalse(should_stop(config, epoch=8, best_epoch=7))
        self.assertFalse(should_stop({"early_stopping_patience": 0}, epoch=50, best_epoch=0))


if __name__ == "__main__":
    unittest.main()
