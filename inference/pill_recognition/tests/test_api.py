import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pill_recognition.api import create_app
from pill_recognition.schemas import (
    PillDetection,
    ProductCandidate,
    RecognitionResult,
    VisionObservation,
)


class FakePipeline:
    def recognize(self, image_rgb):
        assert image_rgb.shape == (12, 16, 3)
        return RecognitionResult(
            image_width=16,
            image_height=12,
            pill_count=1,
            model_version="fake",
            detections=[
                PillDetection(
                    pill_id=1,
                    bbox=(1, 2, 10, 11),
                    crop_bbox=(0, 1, 11, 12),
                    detector_confidence=0.92,
                    vision=VisionObservation(),
                    candidates=[
                        ProductCandidate(
                            rank=1,
                            pill_id="K-000001",
                            score=88.0,
                            product_name="테스트정",
                            ingredient="성분A",
                        )
                    ],
                    status="needs_confirmation",
                    status_reason="review required",
                )
            ],
        )


def test_health_returns_runtime_policy(monkeypatch):
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["recognizer"] == "retrieval"
    assert response.json()["top_k"] == 3


def test_recognize_accepts_uploaded_image():
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("pill.jpg", image_bytes(16, 12), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pill_count"] == 1
    assert payload["detections"][0]["status"] == "needs_confirmation"
    assert payload["detections"][0]["status_reason"] == "review required"
    assert payload["detections"][0]["candidates"][0]["pill_id"] == "K-000001"


def test_recognize_rejects_non_image_file():
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def image_bytes(width: int, height: int) -> bytes:
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
