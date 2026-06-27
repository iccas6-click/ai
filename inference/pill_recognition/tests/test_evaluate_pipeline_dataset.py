from pill_recognition.evaluate_pipeline_dataset import (
    GroundTruthPill,
    build_error_report,
    evaluate_result,
    match_detections,
    summarize,
)
from pill_recognition.schemas import (
    PillDetection,
    ProductCandidate,
    RecognitionResult,
    VisionObservation,
)


def test_match_detections_assigns_best_unique_pairs():
    matches = match_detections(
        predictions=[(0, 0, 10, 10), (80, 80, 100, 100)],
        ground_truth=[(1, 1, 11, 11), (0, 0, 10, 10)],
        iou_threshold=0.5,
    )

    assert [(m.prediction_index, m.ground_truth_index) for m in matches] == [(0, 1)]


def test_evaluate_result_counts_detection_and_recognition_topk():
    result = RecognitionResult(
        image_width=100,
        image_height=100,
        pill_count=2,
        model_version="fake",
        detections=[
            detection(
                bbox=(0, 0, 10, 10),
                candidates=[
                    candidate("K-WRONG", 1),
                    candidate("K-001", 2),
                    candidate("K-OTHER", 3),
                ],
            ),
            detection(bbox=(80, 80, 95, 95), candidates=[candidate("K-FP", 1)]),
        ],
    )
    truth = [
        GroundTruthPill(
            index=0,
            class_name="K-001",
            product_name="정답약",
            bbox=(0, 0, 10, 10),
        )
    ]

    row = evaluate_result("sample.jpg", truth, result, iou_threshold=0.5)

    assert row["true_positive"] == 1
    assert row["false_positive"] == 1
    assert row["false_negative"] == 0
    assert row["recognition_top1"] == 0
    assert row["recognition_top3"] == 1
    assert row["detections"][0]["status"] == "needs_confirmation"
    assert row["recognition"][0]["candidates"][1]["pill_id"] == "K-001"


def test_summarize_reports_end_to_end_rates():
    rows = [
        {
            "gt_count": 2,
            "pred_count": 2,
            "count_exact": True,
            "count_abs_error": 0,
            "true_positive": 2,
            "false_positive": 0,
            "false_negative": 0,
            "mean_matched_iou": 0.8,
            "recognition_top1": 1,
            "recognition_top3": 2,
            "recognition_top5": 2,
        },
        {
            "gt_count": 2,
            "pred_count": 1,
            "count_exact": False,
            "count_abs_error": 1,
            "true_positive": 1,
            "false_positive": 0,
            "false_negative": 1,
            "mean_matched_iou": 0.7,
            "recognition_top1": 0,
            "recognition_top3": 1,
            "recognition_top5": 1,
        },
    ]

    summary = summarize(rows, iou_threshold=0.5, top_k=5)

    assert summary["detector_recall"] == 0.75
    assert summary["recognition_top3_on_matched"] == 1.0
    assert summary["end_to_end_top3_on_gt"] == 0.75
    assert summary["status_counts"] == {}


def test_build_error_report_groups_actionable_failures():
    result = RecognitionResult(
        image_width=100,
        image_height=100,
        pill_count=2,
        model_version="fake",
        detections=[
            detection(
                bbox=(0, 0, 10, 10),
                candidates=[candidate("K-WRONG", 1), candidate("K-OTHER", 2)],
                status="ambiguous",
            ),
            detection(
                bbox=(80, 80, 95, 95),
                candidates=[candidate("K-FP", 1)],
                status="low_confidence",
            ),
        ],
        warnings=["image: blurry"],
    )
    truth = [
        GroundTruthPill(
            index=0,
            class_name="K-001",
            product_name="정답약",
            bbox=(0, 0, 10, 10),
        ),
        GroundTruthPill(
            index=1,
            class_name="K-MISSED",
            product_name="미검출약",
            bbox=(40, 40, 55, 55),
        ),
    ]
    row = evaluate_result("sample.jpg", truth, result, iou_threshold=0.5)

    report = build_error_report([row], top_k=3)

    assert report["counts"]["count_mismatch_images"] == 0
    assert report["counts"]["detector_misses"] == 1
    assert report["counts"]["false_positives"] == 1
    assert report["counts"]["recognition_top3_misses"] == 1
    assert report["counts"]["status_review_detections"] == 2
    assert report["counts"]["warning_images"] == 1
    assert report["detector_misses"][0]["expected"] == "K-MISSED"
    assert report["recognition_top3_misses"][0]["expected"] == "K-001"


def detection(bbox, candidates, status="needs_confirmation"):
    return PillDetection(
        pill_id=1,
        bbox=bbox,
        crop_bbox=bbox,
        detector_confidence=0.9,
        vision=VisionObservation(),
        candidates=candidates,
        status=status,
    )


def candidate(pill_id: str, rank: int):
    return ProductCandidate(
        rank=rank,
        pill_id=pill_id,
        score=90 - rank,
        product_name=f"제품 {pill_id}",
    )
