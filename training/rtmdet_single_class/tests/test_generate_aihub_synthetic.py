import json

import cv2
import numpy as np

from training.rtmdet_single_class.scripts.generate_aihub_synthetic import (
    box_iou,
    build_asset_index,
    extract_foreground_mask,
    format_yolo_label,
    mask_bbox,
)


def test_build_asset_index_reads_class_metadata(tmp_path):
    class_dir = tmp_path / "K-000001"
    class_dir.mkdir()
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(class_dir / "sample.png"), image)
    (class_dir / "sample.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "dl_name": "테스트정",
                        "dl_company": "테스트제약",
                        "item_seq": "199900001",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    index = build_asset_index(tmp_path)

    assert sorted(index) == ["K-000001"]
    assert index["K-000001"].product_name == "테스트정"
    assert len(index["K-000001"].image_paths) == 1


def test_format_yolo_label_uses_normalized_center_size():
    assert format_yolo_label((100, 200, 300, 500), 1000, 1000) == (
        "0 0.20000000 0.35000000 0.20000000 0.30000000"
    )


def test_box_iou():
    assert round(box_iou((0, 0, 100, 100), (50, 50, 150, 150)), 4) == 0.1429


def test_extract_foreground_mask_finds_center_object():
    image = np.full((96, 96, 3), 240, dtype=np.uint8)
    cv2.circle(image, (48, 48), 20, (30, 80, 200), thickness=-1)

    mask = extract_foreground_mask(image)
    bbox = mask_bbox(mask)

    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x1 < 35
    assert y1 < 35
    assert x2 > 60
    assert y2 > 60
