import json

from PIL import Image

from pill_recognition.validate_real_dataset import validate_real_dataset


def test_validate_real_dataset_accepts_reviewed_annotation(tmp_path):
    write_dataset(
        tmp_path,
        annotation={
            "image": "photo.jpg",
            "allowed_pill_ids": ["K-000001", "K-000002"],
            "pills": [
                {
                    "pill_id": 1,
                    "class_name": "K-000001",
                    "product_name": "테스트정",
                    "bbox_xyxy": [1, 2, 20, 22],
                    "needs_review": False,
                }
            ],
        },
    )

    report = validate_real_dataset(
        tmp_path,
        product_ids={"K-000001", "K-000002"},
        retrieval_ids={"K-000001", "K-000002"},
    )

    assert report["summary"] == {
        "annotations": 1,
        "pills": 1,
        "errors": 0,
        "warnings": 0,
    }
    assert report["issues"] == []


def test_validate_real_dataset_reports_review_and_mapping_errors(tmp_path):
    write_dataset(
        tmp_path,
        annotation={
            "image": "photo.jpg",
            "allowed_pill_ids": ["K-ALLOWED", "K-MISSING"],
            "pills": [
                {
                    "pill_id": 1,
                    "class_name": "K-OTHER",
                    "bbox_xyxy": [1, 2, 120, 22],
                    "needs_review": True,
                }
            ],
        },
    )

    report = validate_real_dataset(
        tmp_path,
        product_ids={"K-ALLOWED"},
        retrieval_ids={"K-ALLOWED"},
    )

    codes = {issue["code"] for issue in report["issues"]}
    assert report["summary"]["errors"] == 7
    assert "needs_review" in codes
    assert "bbox_out_of_bounds" in codes
    assert "class_not_in_allowed_pill_ids" in codes
    assert "class_id_missing_product_metadata" in codes
    assert "class_id_missing_retrieval_index" in codes
    assert "allowed_id_missing_product_metadata" in codes
    assert "allowed_id_missing_retrieval_index" in codes


def test_validate_real_dataset_can_allow_needs_review_for_draft_reports(tmp_path):
    write_dataset(
        tmp_path,
        annotation={
            "image": "photo.jpg",
            "pills": [
                {
                    "pill_id": 1,
                    "class_name": "K-000001",
                    "bbox_xyxy": [1, 2, 20, 22],
                    "needs_review": True,
                }
            ],
        },
    )

    report = validate_real_dataset(
        tmp_path,
        product_ids={"K-000001"},
        retrieval_ids={"K-000001"},
        allow_needs_review=True,
    )

    assert report["summary"]["errors"] == 0


def write_dataset(tmp_path, annotation):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (32, 32), "white").save(images / "photo.jpg")
    (annotations / "photo.json").write_text(
        json.dumps(annotation, ensure_ascii=False),
        encoding="utf-8",
    )
