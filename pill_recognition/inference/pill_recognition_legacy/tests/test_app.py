from pill_recognition_legacy.app import format_bbox, format_candidates, format_ingredient
from pill_recognition_legacy.schemas import Candidate


def test_format_bbox_uses_compact_coordinates():
    assert format_bbox((1, 2, 30, 40)) == "1,2,30,40"


def test_format_candidates_lists_ranked_aihub_candidates():
    candidates = [
        Candidate(
            rank=1,
            class_id=10,
            class_name="K-000010",
            confidence=0.81234,
            product_name="테스트정",
            ingredient="성분A|성분B",
            company="테스트제약",
            item_seq="199900001",
            etc_otc_code="일반의약품",
        ),
        Candidate(rank=2, class_id=11, class_name="K-000011", confidence=0.10234),
    ]

    assert format_candidates(candidates) == (
        "1. K-000010 | 테스트정 | 성분: 성분A, 성분B | 테스트제약 | 199900001 | 일반의약품 (0.812)\n"
        "2. K-000011 (0.102)"
    )


def test_format_ingredient_splits_aihub_pipe_separator():
    assert format_ingredient("텔미사르탄|암로디핀베실산염") == "텔미사르탄, 암로디핀베실산염"
