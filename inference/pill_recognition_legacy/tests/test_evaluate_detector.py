from pill_recognition_legacy.evaluate_detector import (
    build_row,
    match_detections,
    summarize,
    yolo_to_xyxy,
)


def test_yolo_to_xyxy_converts_normalized_coordinates():
    assert yolo_to_xyxy(0.5, 0.5, 0.25, 0.5, 400, 200) == (
        150.0,
        50.0,
        250.0,
        150.0,
    )


def test_match_detections_greedily_matches_each_box_once():
    predictions = [
        (0, 0, 100, 100),
        (5, 5, 95, 95),
        (200, 200, 300, 300),
    ]
    ground_truth = [
        (0, 0, 100, 100),
        (210, 210, 290, 290),
    ]

    matches = match_detections(predictions, ground_truth, iou_threshold=0.5)

    assert [(match.prediction_index, match.ground_truth_index) for match in matches] == [
        (0, 0),
        (2, 1),
    ]


def test_build_row_counts_detection_errors():
    row = build_row(
        "sample.jpg",
        ground_truth=[(0, 0, 100, 100), (200, 200, 300, 300)],
        predictions=[
            {"bbox": (0, 0, 100, 100), "confidence": 0.9},
            {"bbox": (500, 500, 600, 600), "confidence": 0.8},
        ],
        matches=match_detections(
            [(0, 0, 100, 100), (500, 500, 600, 600)],
            [(0, 0, 100, 100), (200, 200, 300, 300)],
            iou_threshold=0.5,
        ),
    )

    assert row["count_exact"] is True
    assert row["true_positive"] == 1
    assert row["false_positive"] == 1
    assert row["false_negative"] == 1


def test_summarize_detector_metrics():
    rows = [
        {
            "gt_count": 2,
            "pred_count": 2,
            "count_exact": True,
            "count_abs_error": 0,
            "true_positive": 2,
            "false_positive": 0,
            "false_negative": 0,
            "matches": [{"iou": 0.9}, {"iou": 0.8}],
        },
        {
            "gt_count": 2,
            "pred_count": 1,
            "count_exact": False,
            "count_abs_error": 1,
            "true_positive": 1,
            "false_positive": 0,
            "false_negative": 1,
            "matches": [{"iou": 0.7}],
        },
    ]

    assert summarize(rows, iou_threshold=0.5, confidence_threshold=0.25) == {
        "image_count": 2,
        "ground_truth_boxes": 4,
        "predicted_boxes": 3,
        "iou_threshold": 0.5,
        "confidence_threshold": 0.25,
        "count_exact_accuracy": 0.5,
        "count_mean_abs_error": 0.5,
        "precision": 1.0,
        "recall": 0.75,
        "f1": 0.8571,
        "mean_matched_iou": 0.8,
        "true_positive": 3,
        "false_positive": 0,
        "false_negative": 1,
    }
