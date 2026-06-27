from pill_recognition.app import (
    format_bbox,
    format_candidates,
    format_cautions,
    format_ingredient,
    recognize,
    warmup,
)
from pill_recognition.schemas import (
    PillDetection,
    ProductCandidate,
    RecognitionResult,
    VisionObservation,
)


class FakePipeline:
    allowed_pill_ids = None

    def recognize(self, image, allowed_pill_ids=None):
        self.allowed_pill_ids = allowed_pill_ids
        return RecognitionResult(
            image_width=8,
            image_height=6,
            pill_count=1,
            model_version="fake",
            detections=[
                PillDetection(
                    pill_id=1,
                    bbox=(1, 2, 3, 4),
                    crop_bbox=(1, 2, 3, 4),
                    detector_confidence=0.9,
                    vision=VisionObservation(),
                    candidates=[
                        ProductCandidate(
                            rank=1,
                            pill_id="K-000001",
                            score=88,
                            product_name="테스트정",
                        )
                    ],
                    status="needs_confirmation",
                    status_reason="review",
                )
            ],
            candidate_scope={"enabled": True, "allowed_count": 2},
        )


def test_format_bbox_uses_compact_coordinates():
    assert format_bbox((1, 2, 30, 40)) == "1,2,30,40"


def test_format_ingredient_splits_aihub_pipe_separator():
    assert format_ingredient("텔미사르탄|암로디핀베실산염") == "텔미사르탄, 암로디핀베실산염"


def test_format_candidates_focuses_on_product_and_ingredient():
    candidates = [
        ProductCandidate(
            rank=1,
            pill_id="GEMINI",
            score=87,
            source="gemini",
            product_name="대화와르파린나트륨정",
            ingredient="와르파린나트륨",
            caution_points=["출혈 위험이 있을 수 있음", "복용 전 전문가 확인"],
        )
    ]

    assert format_candidates(candidates) == (
        "1. 제품명: 대화와르파린나트륨정 | 성분: 와르파린나트륨 | "
        "주의: 출혈 위험이 있을 수 있음; 복용 전 전문가 확인 | 점수 87"
    )


def test_format_cautions_uses_dash_for_empty_values():
    assert format_cautions([]) == "-"


def test_recognize_passes_allowed_pill_scope_to_pipeline(monkeypatch):
    import numpy as np

    pipeline = FakePipeline()
    monkeypatch.setattr("pill_recognition.app.get_pipeline", lambda: pipeline)
    monkeypatch.setattr(
        "pill_recognition.app.draw_detections",
        lambda image, result: image,
    )

    _, rows, raw = recognize(
        np.zeros((6, 8, 3), dtype=np.uint8),
        '["K-000001","K-000002"]',
    )

    assert pipeline.allowed_pill_ids == {"K-000001", "K-000002"}
    assert rows[0][0] == 1
    assert raw["candidate_scope"] == {"enabled": True, "allowed_count": 2}


def test_warmup_respects_disabled_setting(monkeypatch):
    class Settings:
        warmup_on_startup = False

    def fail_get_pipeline():
        raise AssertionError("pipeline should not be loaded")

    monkeypatch.setattr("pill_recognition.app.get_settings", lambda: Settings())
    monkeypatch.setattr("pill_recognition.app.get_pipeline", fail_get_pipeline)

    warmup()
