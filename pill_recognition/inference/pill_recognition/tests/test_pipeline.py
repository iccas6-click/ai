import numpy as np

from pill_recognition.pipeline import (
    PillRecognitionPipeline,
    determine_status,
    recognize_crops_with_aihub_classifier,
)
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
    allowed_pill_ids = None
    index_positions_by_pill_id = {
        "K-000001": [0],
        "K-000002": [1],
        "K-000777": [2, 3],
    }

    def predict_batch(self, crops_rgb, top_k, allowed_pill_ids=None):
        self.calls += 1
        self.allowed_pill_ids = allowed_pill_ids
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

    def predict_batch(self, crops_rgb, top_k, allowed_pill_ids=None):
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

    def predict_batch(self, crops_rgb, top_k, allowed_pill_ids=None):
        return [[] for _ in crops_rgb]


class FakeAIHubClassifier:
    model_version = "fake-aihub-classifier"
    class_ids_by_name = {
        "K-WRONG": 0,
        "K-RIGHT": 1,
        "K-SCOPED": 2,
    }
    allowed_pill_ids = None
    calls = 0

    def predict_batch(self, crops_rgb, top_k, allowed_pill_ids=None):
        self.calls += 1
        self.allowed_pill_ids = allowed_pill_ids
        rows = []
        for index, _ in enumerate(crops_rgb):
            if index % 2 == 0:
                rows.append(
                    [
                        Candidate(
                            rank=1,
                            class_id=0,
                            class_name="K-WRONG",
                            confidence=0.60,
                            product_name="오답정",
                        )
                    ]
                )
            else:
                rows.append(
                    [
                        Candidate(
                            rank=1,
                            class_id=1,
                            class_name="K-RIGHT",
                            confidence=0.92,
                            product_name="정답정",
                            ingredient="정답성분",
                            print_front="R",
                            drug_shape="원형",
                            color_class1="하양",
                        )
                    ]
                )
        return rows


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
    assert set(result.timings_ms) == {
        "quality",
        "detector",
        "recognition",
        "postprocess",
        "total",
    }
    assert result.timings_ms["total"] >= 0
    assert result.candidate_scope == {}


def test_pipeline_passes_allowed_pill_scope_to_retriever():
    retriever = FakeRetriever()
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=SingleFakeDetector(),
        retriever=retriever,
        product_index={},
    )

    result = pipeline.recognize(
        np.zeros((64, 64, 3), dtype=np.uint8) + 255,
        allowed_pill_ids={"K-000001", "K-000777", "K-MISSING"},
    )

    assert retriever.allowed_pill_ids == {"K-000001", "K-000777", "K-MISSING"}
    assert result.candidate_scope == {
        "enabled": True,
        "allowed_count": 3,
        "metadata_match_count": 0,
        "unknown_metadata_pill_ids": ["K-000001", "K-000777", "K-MISSING"],
        "retrieval_id_match_count": 2,
        "retrieval_index_position_count": 3,
        "unknown_retrieval_pill_ids": ["K-MISSING"],
    }


def test_pipeline_warns_when_allowed_scope_has_no_retrieval_match():
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=SingleFakeDetector(),
        retriever=EmptyRetriever(),
        product_index={},
    )

    result = pipeline.recognize(
        np.zeros((64, 64, 3), dtype=np.uint8) + 255,
        allowed_pill_ids={"K-MISSING"},
    )

    assert result.candidate_scope["unknown_retrieval_pill_ids"] == ["K-MISSING"]
    assert any("No allowed pill IDs" in warning for warning in result.warnings)


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
    assert set(result.timings_ms) == {
        "preprocess",
        "recognition",
        "postprocess",
        "total",
    }


def test_pipeline_recognize_crops_batch_uses_single_retriever_call():
    retriever = FakeRetriever()
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=ExplodingDetector(),
        retriever=retriever,
        product_index={},
    )

    result = pipeline.recognize_crops_batch(
        [
            np.zeros((32, 48, 3), dtype=np.uint8) + 255,
            np.zeros((40, 28, 3), dtype=np.uint8) + 255,
        ]
    )

    assert result.model_version == "crop-batch+fake-retriever"
    assert result.image_width == 48
    assert result.image_height == 40
    assert result.pill_count == 2
    assert [detection.pill_id for detection in result.detections] == [1, 2]
    assert [detection.candidates[0].pill_id for detection in result.detections] == [
        "K-000001",
        "K-000002",
    ]
    assert retriever.calls == 1
    assert result.timings_ms["recognition"] >= 0


def test_aihub_classifier_recognizer_merges_best_crop_variant():
    classifier = FakeAIHubClassifier()

    results = recognize_crops_with_aihub_classifier(
        classifier,
        [np.zeros((32, 48, 3), dtype=np.uint8) + 255],
        top_k=1,
        query_preprocess="none+foreground",
        allowed_pill_ids={"K-RIGHT"},
    )

    assert classifier.calls == 1
    assert classifier.allowed_pill_ids == {"K-RIGHT"}
    assert results[0][0].pill_id == "K-RIGHT"
    assert results[0][0].score == 92.0
    assert results[0][0].source == "aihub_resnet152_classifier"
    assert results[0][0].matched == "AIHub ResNet152 classifier (foreground)"


def test_pipeline_can_use_aihub_official_classifier_for_crop_recognition():
    classifier = FakeAIHubClassifier()
    pipeline = PillRecognitionPipeline(
        settings=Settings(
            top_k=1,
            recognizer="aihub_classifier",
            aihub_classifier_query_preprocess="none+foreground",
        ),
        detector=ExplodingDetector(),
        aihub_classifier=classifier,
        product_index={},
    )

    result = pipeline.recognize_crop(
        np.zeros((32, 48, 3), dtype=np.uint8) + 255,
        allowed_pill_ids={"K-RIGHT", "K-MISSING"},
    )

    assert result.model_version == (
        "single-crop+fake-aihub-classifier:none+foreground"
    )
    assert result.detections[0].candidates[0].pill_id == "K-RIGHT"
    assert result.detections[0].candidates[0].product_name == "정답정"
    assert result.candidate_scope == {
        "enabled": True,
        "allowed_count": 2,
        "metadata_match_count": 0,
        "unknown_metadata_pill_ids": ["K-MISSING", "K-RIGHT"],
        "classifier_id_match_count": 1,
        "unknown_classifier_pill_ids": ["K-MISSING"],
    }


def test_pipeline_returns_capture_quality_warnings_for_bad_input():
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=SingleFakeDetector(),
        retriever=EmptyRetriever(),
        product_index={},
    )

    result = pipeline.recognize(np.zeros((128, 128, 3), dtype=np.uint8))

    assert any("resolution is low" in warning for warning in result.warnings)
    assert any("too dark" in warning for warning in result.warnings)


def test_pipeline_returns_crop_quality_warnings_for_batch_crops():
    pipeline = PillRecognitionPipeline(
        settings=Settings(top_k=3),
        detector=ExplodingDetector(),
        retriever=EmptyRetriever(),
        product_index={},
    )

    result = pipeline.recognize_crops_batch(
        [np.zeros((40, 40, 3), dtype=np.uint8) + 250]
    )

    assert any("crop 1" in warning for warning in result.warnings)
    assert any("resolution is low" in warning for warning in result.warnings)


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
