from pill_recognition.app import format_bbox, format_candidates, format_ingredient
from pill_recognition.schemas import ProductCandidate


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
        )
    ]

    assert format_candidates(candidates) == (
        "1. 제품명: 대화와르파린나트륨정 | 성분: 와르파린나트륨 | gemini | 점수 87"
    )
