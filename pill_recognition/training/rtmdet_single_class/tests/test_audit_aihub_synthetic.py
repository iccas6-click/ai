import json

import cv2
import numpy as np
from PIL import Image

from training.rtmdet_single_class.scripts.audit_aihub_synthetic import (
    audit_metadata_rows,
    load_metadata_rows,
    render_preview_grid,
)


def test_audit_metadata_rows_summarizes_counts_and_mask_quality():
    rows = [
        {
            "_stem": "sample_a",
            "requested_pill_count": 2,
            "pill_count": 2,
            "placement_attempts": 2,
            "pills": [
                {
                    "pill_id": 1,
                    "class_name": "K-000001",
                    "mask_quality": {
                        "area_ratio": 0.6,
                        "bbox_fill_ratio": 0.7,
                        "soft_edge_ratio": 0.08,
                    },
                },
                {
                    "pill_id": 2,
                    "class_name": "K-000002",
                    "mask_quality": {
                        "area_ratio": 0.5,
                        "bbox_fill_ratio": 0.6,
                        "soft_edge_ratio": 0.04,
                    },
                },
            ],
        },
        {
            "_stem": "sample_b",
            "requested_pill_count": 3,
            "pill_count": 2,
            "placement_attempts": 20,
            "pills": [
                {
                    "pill_id": 1,
                    "class_name": "K-000003",
                    "mask_quality": {
                        "area_ratio": 0.3,
                        "bbox_fill_ratio": 0.2,
                        "soft_edge_ratio": 0.001,
                    },
                }
            ],
        },
    ]

    report = audit_metadata_rows(rows)

    assert report["summary"]["images"] == 2
    assert report["summary"]["requested_pills"] == 5
    assert report["summary"]["placed_pills"] == 4
    assert report["summary"]["count_match_rate"] == 0.5
    assert report["summary"]["warnings"]["count_mismatch_images"] == 1
    assert report["summary"]["warnings"]["high_attempt_images"] == 1
    assert report["summary"]["warnings"]["low_quality_masks"] == 1
    assert report["summary"]["mask_quality"]["pills"] == 3


def test_load_metadata_rows_and_render_preview_grid(tmp_path):
    dataset = tmp_path / "dataset"
    image_dir = dataset / "images" / "train"
    metadata_dir = dataset / "metadata" / "train"
    image_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    image = np.full((96, 96, 3), 240, dtype=np.uint8)
    cv2.circle(image, (48, 48), 20, (80, 120, 220), thickness=-1)
    cv2.imwrite(str(image_dir / "scene_001.jpg"), image)
    (metadata_dir / "scene_001.json").write_text(
        json.dumps(
            {
                "requested_pill_count": 1,
                "pill_count": 1,
                "placement_attempts": 1,
                "pills": [
                    {
                        "pill_id": 1,
                        "class_name": "K-000001",
                        "bbox_xyxy": [20, 20, 76, 76],
                        "mask_quality": {
                            "area_ratio": 0.5,
                            "bbox_fill_ratio": 0.6,
                            "soft_edge_ratio": 0.05,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = load_metadata_rows(dataset, "train")
    output = tmp_path / "preview.jpg"
    render_preview_grid(dataset, "train", rows, output, max_images=1, thumb_size=128)

    assert rows[0]["_stem"] == "scene_001"
    assert output.exists()
    preview = Image.open(output)
    assert preview.size == (128, 128)
