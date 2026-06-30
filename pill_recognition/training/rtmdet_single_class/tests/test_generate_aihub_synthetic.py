import json
import random

import cv2
import numpy as np

from training.rtmdet_single_class.scripts.generate_aihub_synthetic import (
    box_iou,
    build_asset_index,
    compute_mask_quality,
    extract_foreground_mask,
    feather_alpha,
    format_yolo_label,
    is_usable_mask_quality,
    match_patch_to_background,
    mask_bbox,
    synthesize_scene,
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


def test_feather_alpha_adds_soft_edge_and_quality_metrics():
    alpha = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(alpha, (40, 40), 24, 255, thickness=-1)

    feathered = feather_alpha(alpha, radius=4.0)
    quality = compute_mask_quality(feathered)

    assert feathered.dtype == np.uint8
    assert np.logical_and(feathered > 0, feathered < 255).sum() > 0
    assert quality["soft_edge_ratio"] > 0
    assert is_usable_mask_quality(quality)


def test_match_patch_to_background_makes_small_lighting_adjustment():
    rng = random.Random(7)
    patch = np.full((40, 40, 3), 210, dtype=np.uint8)
    region = np.full((40, 40, 3), 120, dtype=np.uint8)
    alpha = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(alpha, (20, 20), 14, 255, thickness=-1)

    adjusted = match_patch_to_background(patch, alpha, region, rng)

    assert adjusted.shape == patch.shape
    assert adjusted.dtype == np.uint8
    assert adjusted.mean() < patch.mean()
    assert adjusted.mean() > 180


def test_synthesize_scene_retries_until_requested_pill_count(tmp_path):
    for index, color in enumerate([(30, 80, 220), (220, 80, 30), (80, 180, 80)]):
        class_dir = tmp_path / f"K-00000{index + 1}"
        class_dir.mkdir()
        image = np.full((96, 96, 3), 240, dtype=np.uint8)
        cv2.ellipse(image, (48, 48), (24, 16), 0, 0, 360, color, thickness=-1)
        cv2.imwrite(str(class_dir / "sample.png"), image)

    class_index = build_asset_index(tmp_path)
    scene = synthesize_scene(
        class_index=class_index,
        class_names=list(class_index),
        image_size=384,
        pill_count=3,
        background_mode="simple",
        rng=random.Random(11),
    )

    assert scene["metadata"]["requested_pill_count"] == 3
    assert scene["metadata"]["pill_count"] == 3
    assert scene["metadata"]["placement_attempts"] >= 3
    assert len(scene["yolo_labels"]) == 3
    assert scene["metadata"]["pills"][0]["mask_quality"]["soft_edge_ratio"] > 0
    assert scene["metadata"]["pills"][0]["placed"]["alpha_feathered"] is True
