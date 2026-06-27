import numpy as np

from pill_recognition.pipeline import PillRecognitionPipeline
from pill_recognition.schemas import ProductCandidate, VisionObservation
from pill_recognition.settings import Settings
from pill_recognition_legacy.aihub_classifier import AIHubProductInfo
from pill_recognition_legacy.schemas import Candidate


class FakeDetector:
    def predict(self, image_rgb):
        return [
            (
                (10, 10, 30, 30),
                [Candidate(rank=1, class_id=0, class_name="pill", confidence=0.9)],
            ),
            (
                (35, 10, 55, 30),
                [Candidate(rank=1, class_id=0, class_name="pill", confidence=0.8)],
            ),
        ]


class FakeRetriever:
    model_version = "fake-retriever"
    calls = 0

    def predict_batch(self, crops_rgb, top_k):
        self.calls += 1
        return [
            [
                ProductCandidate(
                    rank=1,
                    pill_id=f"K-{index:06d}",
                    score=87.5,
                    source="fake_retrieval",
                    product_name=f"검색 후보 {index}",
                    ingredient="와르파린나트륨",
                )
            ]
            for index, _ in enumerate(crops_rgb, start=1)
        ]


class SingleFakeDetector:
    def predict(self, image_rgb):
        return [
            (
                (10, 10, 30, 30),
                [Candidate(rank=1, class_id=0, class_name="pill", confidence=0.9)],
            )
        ]


class EmptyRetriever:
    model_version = "empty-retriever"

    def predict_batch(self, crops_rgb, top_k):
        return [[] for _ in crops_rgb]


class FailingVisionProvider:
    name = "failing-vision"

    def inspect_crop(self, crop_rgb):
        raise RuntimeError("boom")


def test_pipeline_uses_retriever_batch_for_detected_crops():
    retriever = FakeRetriever()
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=FakeDetector(),
        retriever=retriever,
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

    assert result.model_version == "rtmdet-single-class+fake-retriever"
    assert result.detections[0].candidates[0].pill_id == "K-000001"
    assert result.detections[0].candidates[0].source == "fake_retrieval"
    assert result.detections[0].candidates[0].ingredient == "와르파린나트륨"
    assert result.pill_count == 2
    assert retriever.calls == 1
    assert result.detections[0].status == "needs_confirmation"


def test_pipeline_marks_no_candidate_when_retriever_returns_empty():
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=SingleFakeDetector(),
        retriever=EmptyRetriever(),
        product_index={},
    )

    result = pipeline.recognize(np.zeros((64, 64, 3), dtype=np.uint8) + 255)

    assert result.detections[0].status == "no_candidate"
