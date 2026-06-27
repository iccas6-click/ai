import numpy as np

from pill_recognition.pipeline import PillRecognitionPipeline, determine_status
from pill_recognition.schemas import (
    ProductCandidate,
    VisionObservation,
    VisionProductCandidate,
)
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


class ExplodingDetector:
    def predict(self, image_rgb):
        raise AssertionError("crop recognition must not call detector")


class AmbiguousRetriever:
    model_version = "ambiguous-retriever"

    def predict_batch(self, crops_rgb, top_k):
        return [
            [
                ProductCandidate(
                    rank=1,
                    pill_id="K-000001",
                    score=88.0,
                    product_name="후보1",
                ),
                ProductCandidate(
                    rank=2,
                    pill_id="K-000002",
                    score=86.0,
                    product_name="후보2",
                ),
            ]
            for _ in crops_rgb
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


class FakeVisionProvider:
    name = "fake-vision"

    def inspect_crops(self, crops_rgb):
        return [
            VisionObservation(
                shape="장방형",
                color="노랑",
                confidence=0.7,
                product_candidates=[
                    VisionProductCandidate(
                        product_name="비전후보정",
                        ingredient="성분B",
                        confidence=0.7,
                    )
                ],
            )
            for _ in crops_rgb
        ]


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
    assert result.detections[0].vision.color == "하양"
    assert result.detections[0].vision.shape == "원형"
    assert result.detections[0].status == "needs_confirmation"
    assert result.detections[0].status_reason


def test_pipeline_recognize_crop_skips_detector_and_returns_single_crop_result():
    retriever = FakeRetriever()
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=ExplodingDetector(),
        retriever=retriever,
        product_index={},
    )

    result = pipeline.recognize_crop(np.zeros((32, 48, 3), dtype=np.uint8) + 255)

    assert result.model_version == "single-crop+fake-retriever"
    assert result.image_width == 48
    assert result.image_height == 32
    assert result.pill_count == 1
    assert result.detections[0].bbox == (0, 0, 48, 32)
    assert result.detections[0].detector_confidence == 1.0
    assert result.detections[0].candidates[0].pill_id == "K-000001"
    assert result.detections[0].vision.color == "하양"
    assert retriever.calls == 1


def test_pipeline_preserves_vision_observation_for_provider_recognizer():
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3, recognizer="gemini"),
        detector=SingleFakeDetector(),
        vision_provider=FakeVisionProvider(),
        product_index={},
    )

    result = pipeline.recognize(np.zeros((64, 64, 3), dtype=np.uint8) + 255)

    assert result.detections[0].vision.color == "노랑"
    assert result.detections[0].vision.shape == "장방형"
    assert result.detections[0].candidates[0].source == "gemini"
    assert result.detections[0].candidates[0].product_name == "비전후보정"


def test_pipeline_marks_no_candidate_when_retriever_returns_empty():
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=SingleFakeDetector(),
        retriever=EmptyRetriever(),
        product_index={},
    )

    result = pipeline.recognize(np.zeros((64, 64, 3), dtype=np.uint8) + 255)

    assert result.detections[0].status == "no_candidate"
    assert result.detections[0].status_reason


def test_pipeline_marks_ambiguous_when_top_candidates_are_close():
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3, candidate_ambiguity_margin=3),
        detector=SingleFakeDetector(),
        retriever=AmbiguousRetriever(),
        product_index={},
    )

    result = pipeline.recognize(np.zeros((64, 64, 3), dtype=np.uint8) + 255)

    assert result.detections[0].status == "ambiguous"
    assert "Top-2" in result.detections[0].status_reason


def test_determine_status_marks_low_confidence():
    status, reason = determine_status(
        [ProductCandidate(rank=1, pill_id="K-000001", score=42.0)],
        min_score=70,
    )

    assert status == "low_confidence"
    assert "below" in reason
