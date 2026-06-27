import json

import pytest
from PIL import Image

from pill_recognition.evaluate_real_dataset import (
    load_real_examples,
    load_real_ground_truth,
    resolve_image_path,
)


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
