import numpy as np

from pill_recognition.postprocess import bbox_iou, group_predictions


def test_bbox_iou_for_overlapping_boxes():
    assert bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175


def test_group_predictions_returns_top_candidates_per_location():
    bboxes = np.array(
        [
            [0, 0, 100, 100],
            [2, 2, 98, 98],
            [200, 200, 300, 300],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    labels = np.array([0, 1, 2], dtype=np.int64)

    groups = group_predictions(
        bboxes,
        scores,
        labels,
        ["pill-a", "pill-b", "pill-c"],
        confidence_threshold=0.25,
        iou_threshold=0.5,
        top_k=3,
    )

    assert len(groups) == 2
    assert [candidate.class_name for candidate in groups[0][1]] == [
        "pill-a",
        "pill-b",
    ]
