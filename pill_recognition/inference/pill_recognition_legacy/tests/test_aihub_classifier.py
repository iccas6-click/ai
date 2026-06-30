import json

import pytest
import torch

from pill_recognition_legacy.aihub_classifier import (
    AIHubPillClassifier,
    load_aihub_class_names,
    load_aihub_product_master,
    rotate_crop,
)


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


def test_load_aihub_product_master_reads_product_metadata(tmp_path):
    class_dir = tmp_path / "K-000001"
    class_dir.mkdir()
    (class_dir / "sample.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "drug_N": "K-000001",
                        "item_seq": "196400046",
                        "dl_name": "게루삼정 200mg/PTP",
                        "dl_company": "삼남제약(주)",
                        "di_etc_otc_code": "일반의약품",
                        "dl_material": "건조수산화알루미늄 겔|침강탄산칼슘",
                        "chart": "흰색의 원형정제",
                        "img_key": "http://example.com/drug.jpg",
                        "print_front": "S분할선N",
                        "print_back": "200",
                        "drug_shape": "원형",
                        "color_class1": "하양",
                        "color_class2": "노랑",
                        "form_code_name": "나정",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    products = load_aihub_product_master(tmp_path, {"K-000001"})

    assert products["K-000001"].product_name == "게루삼정 200mg/PTP"
    assert products["K-000001"].company == "삼남제약(주)"
    assert products["K-000001"].item_seq == "196400046"
    assert products["K-000001"].etc_otc_code == "일반의약품"
    assert products["K-000001"].ingredient == "건조수산화알루미늄 겔|침강탄산칼슘"
    assert products["K-000001"].print_front == "S분할선N"
    assert products["K-000001"].print_back == "200"
    assert products["K-000001"].drug_shape == "원형"
    assert products["K-000001"].color_class1 == "하양"
    assert products["K-000001"].color_class2 == "노랑"
    assert products["K-000001"].form_code_name == "나정"


def test_rotate_crop_returns_contiguous_rotated_array():
    import numpy as np

    crop = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    rotated = rotate_crop(crop, 1)

    assert rotated.flags["C_CONTIGUOUS"]
    assert rotated.shape == (2, 2, 3)


def test_aihub_classifier_scopes_probabilities_to_allowed_pill_ids():
    classifier = AIHubPillClassifier.__new__(AIHubPillClassifier)
    classifier.class_ids_by_name = {
        "K-ONE": 0,
        "K-TWO": 1,
        "K-THREE": 2,
    }

    probabilities = torch.tensor([[0.7, 0.2, 0.1]])
    scoped = classifier._scope_probabilities(
        probabilities,
        allowed_pill_ids={"K-TWO", "K-THREE", "K-MISSING"},
    )

    assert scoped.tolist() == [[0.0, pytest.approx(2 / 3), pytest.approx(1 / 3)]]


def test_aihub_classifier_scopes_to_zero_when_allowed_ids_are_unknown():
    classifier = AIHubPillClassifier.__new__(AIHubPillClassifier)
    classifier.class_ids_by_name = {"K-ONE": 0}

    scoped = classifier._scope_probabilities(
        torch.tensor([[1.0]]),
        allowed_pill_ids={"K-MISSING"},
    )

    assert scoped.tolist() == [[0.0]]


def test_aihub_classifier_predict_batch_filters_candidates_outside_allowed_scope():
    classifier = AIHubPillClassifier.__new__(AIHubPillClassifier)
    classifier.class_names = {
        0: "K-ONE",
        1: "K-TWO",
        2: "K-THREE",
    }
    classifier.class_ids_by_name = {
        "K-ONE": 0,
        "K-TWO": 1,
        "K-THREE": 2,
    }
    classifier.product_master = {}
    classifier._predict_probabilities = lambda crops: torch.tensor([[0.7, 0.2, 0.1]])

    candidates = classifier.predict_batch(
        [object()],
        top_k=3,
        allowed_pill_ids={"K-TWO"},
    )[0]

    assert [candidate.class_name for candidate in candidates] == ["K-TWO"]
