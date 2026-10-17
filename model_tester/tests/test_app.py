import io
import sys
import unittest
from pathlib import Path

from PIL import Image


TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from app import CANONICAL_LABELS, create_app  # noqa: E402


class FakeEngine:
    def status(self):
        return {
            "available": True,
            "loaded": True,
            "device": "CPU",
            "model_version": "v2-test",
            "architecture": "ConvNeXt V2 Base",
            "class_count": 12,
        }

    def predict_bytes(self, data):
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        predictions = [
            {
                "class_index": index,
                "code": code,
                "name": code,
                "group": "other modeled class",
                "probability": (12 - index) / 78,
                "percentage": round((12 - index) / 78 * 100, 2),
                "rank": index + 1,
            }
            for index, code in enumerate(CANONICAL_LABELS)
        ]
        return {
            "prediction": predictions[0],
            "predictions": predictions,
            "image": {"width": width, "height": height, "format": "PNG"},
            "model": {"version": "v2-test", "architecture": "ConvNeXt V2 Base", "device": "CPU", "input_size": 224, "inference_ms": 1.0},
            "research_only": True,
            "notice": "Test output",
        }


def sample_image():
    stream = io.BytesIO()
    Image.new("RGB", (96, 64), (158, 112, 93)).save(stream, format="PNG")
    stream.seek(0)
    return stream


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(FakeEngine())
        self.app.testing = True
        self.client = self.app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"See what the model sees", response.data)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])

    def test_prediction_requires_image(self):
        response = self.client.post("/api/predict", data={})
        self.assertEqual(response.status_code, 400)

    def test_prediction_returns_all_canonical_classes(self):
        response = self.client.post(
            "/api/predict",
            data={"image": (sample_image(), "sample.png")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload["predictions"]), 12)
        self.assertEqual([entry["code"] for entry in payload["predictions"]], CANONICAL_LABELS)
        self.assertNotIn("cancer_class_signal", payload)
        self.assertTrue(payload["research_only"])


if __name__ == "__main__":
    unittest.main()
