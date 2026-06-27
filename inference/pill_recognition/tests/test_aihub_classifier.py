import json

import pytest

from pill_recognition.aihub_classifier import (
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
