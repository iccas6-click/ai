import json

from pill_recognition_legacy.compare_detector_evaluations import (
    compare_evaluations,
    metric_delta,
    row_score,
)


def test_metric_delta_subtracts_baseline_from_candidate():
    delta = metric_delta(
        {"precision": 0.8, "recall": 0.9, "false_positive": 3},
        {"precision": 0.85, "recall": 0.88, "false_positive": 1},
    )

    assert delta["precision"] == 0.05
    assert delta["recall"] == -0.02
    assert delta["false_positive"] == -2


def test_row_score_penalizes_false_negative_more_than_false_positive():
    assert row_score(
        {
            "true_positive": 4,
            "false_positive": 0,
            "false_negative": 1,
            "count_abs_error": 1,
            "mean_matched_iou": 0.9,
        }
    ) < row_score(
        {
            "true_positive": 4,
            "false_positive": 1,
            "false_negative": 0,
            "count_abs_error": 1,
            "mean_matched_iou": 0.9,
        }
    )


def test_compare_evaluations_reports_wins_and_regressions(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    write_eval(
        baseline,
        summary={"precision": 0.8, "recall": 0.75, "f1": 0.77},
        rows=[
            row("a.jpg", gt=2, pred=2, tp=2, fp=0, fn=0, iou=0.90),
            row("b.jpg", gt=2, pred=2, tp=2, fp=0, fn=0, iou=0.95),
            row("c.jpg", gt=3, pred=3, tp=3, fp=0, fn=0, iou=0.95),
        ],
    )
    write_eval(
        candidate,
        summary={"precision": 0.9, "recall": 0.8, "f1": 0.85},
        rows=[
            row("a.jpg", gt=2, pred=2, tp=2, fp=0, fn=0, iou=0.90),
            row("b.jpg", gt=2, pred=1, tp=1, fp=0, fn=1, iou=0.80),
            row("c.jpg", gt=3, pred=3, tp=3, fp=0, fn=0, iou=0.99),
        ],
    )

    comparison = compare_evaluations(baseline, candidate)

    assert comparison["summary"]["delta"]["precision"] == 0.1
    assert comparison["summary"]["paired"]["candidate_wins"] == 1
    assert comparison["summary"]["paired"]["baseline_wins"] == 1
    assert comparison["summary"]["paired"]["ties"] == 1
    assert comparison["improvements"][0]["image"] == "c.jpg"
    assert comparison["regressions"][0]["image"] == "b.jpg"


def write_eval(path, summary, rows):
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "results.json").write_text(json.dumps(rows), encoding="utf-8")


def row(image, gt, pred, tp, fp, fn, iou):
    return {
        "image": image,
        "gt_count": gt,
        "pred_count": pred,
        "count_abs_error": abs(gt - pred),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "mean_matched_iou": iou,
    }
