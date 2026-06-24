from pill_recognition.pipeline import determine_status, expand_bbox
from pill_recognition.schemas import Candidate


def candidate(name: str, confidence: float = 0.9) -> Candidate:
    return Candidate(
        rank=1,
        class_id=0,
        class_name=name,
        confidence=confidence,
    )


def test_expand_bbox_adds_padding_and_clamps_to_image():
    assert expand_bbox((5, 10, 25, 30), 100, 100, 0.25) == (0, 5, 30, 35)


def test_aihub_candidate_requires_user_confirmation():
    assert determine_status(
        [candidate("detected-pill")],
        [candidate("K-000001")],
        [],
    ) == "needs_confirmation"


def test_matching_github_candidates_keep_existing_identified_status():
    assert determine_status(
        [candidate("pill-a", 0.8)],
        [],
        [candidate("pill-a", 0.9)],
    ) == "identified"
