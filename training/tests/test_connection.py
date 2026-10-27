import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from dermoai.client import Studio


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.studio = Studio.__new__(Studio)
        self.studio.config = {"project_id": 2}
        self.studio.client = Mock()
        self.backend = SimpleNamespace(id=1, url="http://127.0.0.1:9090", state="CO")

    def test_existing_disconnected_backend_is_not_reported_connected(self):
        self.backend.state = "DI"
        self.studio.client.ml.list.return_value = [self.backend]
        self.assertFalse(self.studio.connect_backend(approved=True)["connected"])
        self.studio.client.ml.create.assert_not_called()

    def test_saved_backend_is_rechecked_after_webhook_failure(self):
        self.studio.client.ml.list.side_effect = [[], [self.backend]]
        self.studio.client.ml.create.side_effect = RuntimeError("Webhook registration failed")
        result = self.studio.connect_backend(approved=True)
        self.assertTrue(result["connected"])
        self.assertIn("registration_warning", result)
        self.studio.client.ml.create.assert_called_once()

    def test_unsaved_backend_failure_is_propagated(self):
        self.studio.client.ml.list.return_value = []
        self.studio.client.ml.create.side_effect = RuntimeError("Unavailable")
        with self.assertRaisesRegex(RuntimeError, "Unavailable"):
            self.studio.connect_backend(approved=True)

    def test_connection_requires_approval(self):
        with self.assertRaises(PermissionError):
            self.studio.connect_backend()
        self.studio.client.ml.list.assert_not_called()


if __name__ == "__main__":
    unittest.main()
