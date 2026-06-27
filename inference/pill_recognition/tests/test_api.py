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
from pill_recognition_legacy.aihub_classifier import AIHubProductInfo


class FakePipeline:
    def recognize(self, image_rgb):
        assert image_rgb.shape == (12, 16, 3)
        return fake_result("fake", "needs_confirmation")

    def recognize_crop(self, image_rgb):
        assert image_rgb.shape == (12, 16, 3)
        return fake_result("fake-crop", "needs_confirmation")


def fake_result(model_version: str, status: str) -> RecognitionResult:
    return RecognitionResult(
        image_width=16,
        image_height=12,
        pill_count=1,
        model_version=model_version,
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
                status=status,
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


def test_recognize_crop_accepts_uploaded_single_pill_crop():
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/crops/recognize",
        files={"file": ("pill-crop.jpg", image_bytes(16, 12), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "fake-crop"
    assert payload["pill_count"] == 1
    assert payload["detections"][0]["bbox"] == [1, 2, 10, 11]
    assert payload["detections"][0]["candidates"][0]["product_name"] == "테스트정"


def test_recognize_rejects_non_image_file():
    app = create_app(lambda: FakePipeline())
    client = TestClient(app)

    response = client.post(
        "/recognize",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_product_search_returns_aihub_metadata_matches():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-000001": AIHubProductInfo(
                pill_id="K-000001",
                product_name="대화와르파린나트륨정",
                ingredient="와르파린나트륨",
                print_front="W분할선2",
                drug_shape="원형",
                color_class1="하양",
            )
        },
    )
    client = TestClient(app)

    response = client.get(
        "/products/search",
        params={
            "imprint": "W2",
            "shape": "원형",
            "color": "하양",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["query"]["limit"] == 5
    assert payload["results"][0]["pill_id"] == "K-000001"
    assert payload["results"][0]["ingredient"] == "와르파린나트륨"
    assert payload["results"][0]["matched"] == "각인 exact, 모양, 색"


def test_product_search_requires_at_least_one_query_field():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.get("/products/search")

    assert response.status_code == 400


def test_product_search_reports_missing_product_metadata():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.get("/products/search", params={"text": "아시클로버"})

    assert response.status_code == 503


def test_product_refine_combines_image_candidates_with_metadata_search():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-WARFARIN": AIHubProductInfo(
                pill_id="K-WARFARIN",
                product_name="대화와르파린나트륨정",
                ingredient="와르파린나트륨",
                print_front="W분할선2",
                drug_shape="원형",
                color_class1="하양",
            ),
            "K-OTHER": AIHubProductInfo(
                pill_id="K-OTHER",
                product_name="다른정",
                ingredient="다른성분",
                print_front="AB",
                drug_shape="장방형",
                color_class1="노랑",
            ),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={
            "candidates": [
                {"pill_id": "K-OTHER", "score": 95.0, "source": "retrieval"},
                {"pill_id": "K-WARFARIN", "score": 55.0, "source": "retrieval"},
            ],
            "imprint": "W2",
            "shape": "원형",
            "color": "하양",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["results"][0]["pill_id"] == "K-WARFARIN"
    assert payload["results"][0]["image_score"] == 55.0
    assert payload["results"][0]["metadata_score"] == 170.0
    assert payload["results"][0]["score"] == 225.0
    assert payload["results"][0]["matched"] == "image candidate + 각인 exact, 모양, 색"


def test_product_refine_can_rank_existing_candidates_without_query():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-LOW": AIHubProductInfo(pill_id="K-LOW", product_name="낮은후보"),
            "K-HIGH": AIHubProductInfo(pill_id="K-HIGH", product_name="높은후보"),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={
            "candidates": [
                {"pill_id": "K-LOW", "score": 30},
                {"pill_id": "K-HIGH", "score": 88},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["pill_id"] == "K-HIGH"


def test_product_refine_boosts_candidate_seen_in_multiple_views():
    app = create_app(
        lambda: FakePipeline(),
        product_index_factory=lambda: {
            "K-SINGLE": AIHubProductInfo(pill_id="K-SINGLE", product_name="한면후보"),
            "K-MULTI": AIHubProductInfo(pill_id="K-MULTI", product_name="양면후보"),
        },
    )
    client = TestClient(app)

    response = client.post(
        "/products/refine",
        json={
            "candidates": [
                {
                    "pill_id": "K-SINGLE",
                    "score": 91,
                    "source": "aihub_resnet_retrieval",
                    "view": "front",
                },
                {
                    "pill_id": "K-MULTI",
                    "score": 88,
                    "source": "aihub_resnet_retrieval",
                    "view": "front",
                },
                {
                    "pill_id": "K-MULTI",
                    "score": 74,
                    "source": "aihub_resnet_retrieval",
                    "view": "back",
                },
            ],
            "limit": 2,
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["pill_id"] == "K-MULTI"
    assert result["image_score"] == 93.0
    assert result["image_score_max"] == 88.0
    assert result["image_evidence_count"] == 2
    assert result["views"] == ["back", "front"]
    assert result["candidate_sources"] == ["aihub_resnet_retrieval"]
    assert result["matched"] == "image candidate x2"


def test_product_refine_requires_candidates_or_query():
    app = create_app(lambda: FakePipeline(), product_index_factory=lambda: {})
    client = TestClient(app)

    response = client.post("/products/refine", json={})

    assert response.status_code == 400


def image_bytes(width: int, height: int) -> bytes:
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
