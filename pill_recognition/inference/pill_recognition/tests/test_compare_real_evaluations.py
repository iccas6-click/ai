import json

from pill_recognition.compare_real_evaluations import (
    compare_real_evaluations,
    metric_delta,
    row_score,
)


def test_metric_delta_compares_key_real_eval_metrics():
    delta = metric_delta(
        {
            "recognition_top3_on_matched": 0.4,
            "end_to_end_top3_on_gt": 0.3,
            "mean_total_ms": 120,
        },
        {
            "recognition_top3_on_matched": 0.7,
            "end_to_end_top3_on_gt": 0.5,
            "mean_total_ms": 110,
        },
    )

    assert delta["recognition_top3_on_matched"] == 0.3
    assert delta["end_to_end_top3_on_gt"] == 0.2
    assert delta["mean_total_ms"] == -10


def test_row_score_prioritizes_recognition_top3_improvement():
    baseline = row("a.jpg", top1=0, top3=0, top5=1)
    candidate = row("a.jpg", top1=0, top3=1, top5=1)

    assert row_score(candidate) > row_score(baseline)


def test_compare_real_evaluations_reports_scope_wins_and_regressions(tmp_path):
    baseline_path = tmp_path / "unscoped.json"
    candidate_path = tmp_path / "scoped.json"
    write_eval(
        baseline_path,
        summary={
            "recognition_top3_on_matched": 0.33,
            "end_to_end_top3_on_gt": 0.33,
            "mean_total_ms": 120,
        },
        rows=[
            row("a.jpg", top1=0, top3=0, top5=1, scope_mode="none"),
            row("b.jpg", top1=1, top3=1, top5=1, scope_mode="none"),
            row("c.jpg", top1=0, top3=1, top5=1, scope_mode="none"),
        ],
    )
    write_eval(
        candidate_path,
        summary={
            "recognition_top3_on_matched": 0.67,
            "end_to_end_top3_on_gt": 0.67,
            "mean_total_ms": 118,
        },
        rows=[
            row("a.jpg", top1=1, top3=1, top5=1, scope_mode="annotation"),
            row("b.jpg", top1=0, top3=0, top5=0, scope_mode="annotation"),
            row("c.jpg", top1=0, top3=1, top5=1, scope_mode="annotation"),
        ],
    )

    comparison = compare_real_evaluations(
        baseline_path,
        candidate_path,
        baseline_name="unscoped",
        candidate_name="annotation-scope",
    )

    assert comparison["summary"]["delta"]["recognition_top3_on_matched"] == 0.34
    assert comparison["summary"]["delta"]["mean_total_ms"] == -2
    assert comparison["summary"]["paired"]["candidate_wins"] == 1
    assert comparison["summary"]["paired"]["baseline_wins"] == 1
    assert comparison["summary"]["paired"]["ties"] == 1
    assert comparison["improvements"][0]["image"] == "a.jpg"
    assert comparison["regressions"][0]["image"] == "b.jpg"
    assert comparison["runs"]["candidate"]["name"] == "annotation-scope"


def write_eval(path, summary, rows):
    path.write_text(
        json.dumps({"summary": summary, "rows": rows}),
        encoding="utf-8",
    )


def row(
    image,
    top1,
    top3,
    top5,
    scope_mode="none",
    tp=1,
    fp=0,
    fn=0,
):
    return {
        "image": image,
        "gt_count": 1,
        "pred_count": 1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "recognition_top1": top1,
        "recognition_top3": top3,
        "recognition_top5": top5,
        "warnings": [],
        "scope_mode": scope_mode,
        "candidate_scope": {"enabled": scope_mode != "none"},
    }
