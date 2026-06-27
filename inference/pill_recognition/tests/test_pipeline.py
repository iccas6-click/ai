import numpy as np

from pill_recognition.pipeline import PillRecognitionPipeline
from pill_recognition.schemas import VisionObservation
from pill_recognition.settings import Settings
from pill_recognition_legacy.aihub_classifier import AIHubProductInfo
from pill_recognition_legacy.schemas import Candidate


class FakeDetector:
    def predict(self, image_rgb):
        return [
            (
                (10, 10, 30, 30),
                [Candidate(rank=1, class_id=0, class_name="pill", confidence=0.9)],
            )
        ]


class FakeVisionProvider:
    name = "fake-vision"

    def inspect_crop(self, crop_rgb):
        return VisionObservation(
            imprint_front="W2",
            shape="원형",
            color="하양",
            confidence=0.8,
        )


class FailingVisionProvider:
    name = "failing-vision"

    def inspect_crop(self, crop_rgb):
        raise RuntimeError("boom")


def test_pipeline_uses_vision_clues_to_search_product_db():
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=FakeDetector(),
        vision_provider=FakeVisionProvider(),
        product_index={
            "K-000001": AIHubProductInfo(
                pill_id="K-000001",
                product_name="대화와르파린나트륨정",
                print_front="W분할선2",
                drug_shape="원형",
                color_class1="하양",
            )
        },
    )

    result = pipeline.recognize(np.zeros((64, 64, 3), dtype=np.uint8) + 255)

    assert result.model_version == "rtmdet-single-class+fake-vision+aihub-db"
    assert result.detections[0].candidates[0].pill_id == "K-000001"
    assert result.detections[0].status == "needs_confirmation"


def test_pipeline_keeps_running_when_vision_provider_fails():
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=FakeDetector(),
        vision_provider=FailingVisionProvider(),
        product_index={},
    )

    result = pipeline.recognize(np.zeros((64, 64, 3), dtype=np.uint8) + 255)

    assert result.detections[0].status == "needs_manual_search"
    assert "provider failed" in result.detections[0].vision.notes
