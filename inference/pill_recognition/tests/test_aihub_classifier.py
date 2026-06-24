import json

import pytest

from pill_recognition.aihub_classifier import load_aihub_class_names


def test_load_aihub_class_names(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "pill_label_path_sharp_score": [
                    [0, "K-000001", 1.0, 0.5, 2.0],
                    [1, "K-000002", 1.0, 0.5, 2.0],
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_aihub_class_names(mapping_path) == {
        0: "K-000001",
        1: "K-000002",
    }


def test_load_aihub_class_names_rejects_non_contiguous_ids(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "pill_label_path_sharp_score": [
                    [0, "K-000001"],
                    [2, "K-000003"],
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        load_aihub_class_names(mapping_path)
