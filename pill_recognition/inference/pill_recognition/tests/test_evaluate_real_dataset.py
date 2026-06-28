import json
from pathlib import Path

import pytest
from PIL import Image

from pill_recognition.evaluate_real_dataset import (
    RealEvaluationExample,
    evaluation_scope_for_example,
    load_allowed_pill_ids,
    load_real_examples,
    load_real_ground_truth,
    resolve_image_path,
)
from pill_recognition.evaluate_pipeline_dataset import GroundTruthPill


def test_load_real_examples_uses_annotation_image_field(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (32, 32), "white").save(images / "photo.jpg")
    (annotations / "sample.json").write_text(
        json.dumps(
            {
                "image": "photo.jpg",
                "allowed_pill_ids": ["K-000001", "K-000999"],
                "pills": [
                    {
                        "class_name": "K-000001",
                        "product_name": "테스트정",
                        "bbox_xyxy": [1, 2, 20, 22],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    examples = load_real_examples(tmp_path)

    assert len(examples) == 1
    assert examples[0].image_path == images / "photo.jpg"
    assert examples[0].ground_truth[0].class_name == "K-000001"
    assert examples[0].ground_truth[0].bbox == (1.0, 2.0, 20.0, 22.0)
    assert examples[0].allowed_pill_ids == {"K-000001", "K-000999"}


def test_resolve_image_path_falls_back_to_annotation_stem(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    annotation_path = tmp_path / "annotations" / "photo_a.json"
    Image.new("RGB", (16, 16), "white").save(images / "photo_a.png")

    image_path = resolve_image_path(images, annotation_path, {})

    assert image_path == images / "photo_a.png"


def test_load_real_ground_truth_requires_bbox(tmp_path):
    annotation_path = tmp_path / "bad.json"

    with pytest.raises(ValueError, match="bbox_xyxy"):
        load_real_ground_truth(
            annotation_path,
            {"pills": [{"class_name": "K-000001"}]},
        )


def test_load_real_ground_truth_rejects_invalid_bbox(tmp_path):
    annotation_path = tmp_path / "bad.json"

    with pytest.raises(ValueError, match="x2>x1"):
        load_real_ground_truth(
            annotation_path,
            {
                "pills": [
                    {
                        "class_name": "K-000001",
                        "bbox_xyxy": [10, 10, 5, 20],
                    }
                ]
            },
        )


def test_load_allowed_pill_ids_requires_list():
    with pytest.raises(ValueError, match="allowed_pill_ids"):
        load_allowed_pill_ids({"allowed_pill_ids": "K-000001"})


def test_evaluation_scope_for_example_supports_annotation_and_ground_truth():
    example = RealEvaluationExample(
        image_path=Path("photo.jpg"),
        annotation_path=Path("photo.json"),
        allowed_pill_ids={"K-ANNOTATION"},
        ground_truth=[
            GroundTruthPill(
                index=0,
                class_name="K-GT",
                product_name="정답정",
                bbox=(1, 2, 3, 4),
            )
        ],
    )

    assert evaluation_scope_for_example(example, "none") == set()
    assert evaluation_scope_for_example(example, "annotation") == {"K-ANNOTATION"}
    assert evaluation_scope_for_example(example, "ground-truth") == {"K-GT"}
