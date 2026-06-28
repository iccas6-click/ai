from pathlib import Path

from PIL import Image

from pill_recognition.draft_real_annotations import (
    draft_annotation_payload,
    find_image_paths,
)
from pill_recognition.schemas import (
    PillDetection,
    ProductCandidate,
    VisionObservation,
)


def test_find_image_paths_filters_supported_suffixes(tmp_path):
    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "b.png")
    (tmp_path / "c.txt").write_text("not an image", encoding="utf-8")

    paths = find_image_paths(tmp_path, "*")

    assert [path.name for path in paths] == ["a.jpg", "b.png"]


def test_draft_annotation_payload_prefills_top_candidate():
    detection = PillDetection(
        pill_id=1,
        bbox=(10, 20, 110, 140),
        crop_bbox=(8, 18, 112, 142),
        detector_confidence=0.91,
        vision=VisionObservation(),
        candidates=[
            ProductCandidate(
                rank=1,
                pill_id="K-000001",
                score=88,
                product_name="테스트정",
                ingredient="성분A",
                company="테스트제약",
            ),
            ProductCandidate(
                rank=2,
                pill_id="K-000002",
                score=76,
                product_name="후보정",
            ),
        ],
        status="needs_confirmation",
    )

    payload = draft_annotation_payload(Path("IMG_0001.jpg"), [detection])

    assert payload["image"] == "IMG_0001.jpg"
    assert payload["pills"][0]["class_name"] == "K-000001"
    assert payload["pills"][0]["bbox_xyxy"] == [10, 20, 110, 140]
    assert payload["pills"][0]["needs_review"] is True
    assert payload["pills"][0]["candidate_hints"][1]["class_name"] == "K-000002"


def test_draft_annotation_payload_handles_empty_candidates():
    detection = PillDetection(
        pill_id=1,
        bbox=(10, 20, 110, 140),
        crop_bbox=(8, 18, 112, 142),
        detector_confidence=0.91,
        vision=VisionObservation(),
        candidates=[],
        status="no_candidate",
    )

    payload = draft_annotation_payload(Path("IMG_0001.jpg"), [detection])

    assert payload["pills"][0]["class_name"] == ""
    assert payload["pills"][0]["product_name"] is None
    assert payload["pills"][0]["candidate_hints"] == []
