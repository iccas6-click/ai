from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = [
    "detector_precision",
    "detector_recall",
    "detector_f1",
    "recognition_top1_on_matched",
    "recognition_top3_on_matched",
    "recognition_top5_on_matched",
    "end_to_end_top1_on_gt",
    "end_to_end_top3_on_gt",
    "end_to_end_top5_on_gt",
    "mean_total_ms",
    "p95_total_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two real-smartphone pill recognition evaluation JSON files."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--name-baseline", default="baseline")
    parser.add_argument("--name-candidate", default="candidate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_real_evaluations(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        baseline_name=args.name_baseline,
        candidate_name=args.name_candidate,
    )
    print(json.dumps(comparison["summary"], ensure_ascii=False, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def compare_real_evaluations(
    baseline_path: Path,
    candidate_path: Path,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> dict[str, Any]:
    baseline = load_evaluation(baseline_path)
    candidate = load_evaluation(candidate_path)
    paired_rows = compare_rows(baseline["rows"], candidate["rows"])
    return {
        "runs": {
            "baseline": {"name": baseline_name, "path": str(baseline_path)},
            "candidate": {"name": candidate_name, "path": str(candidate_path)},
        },
        "summary": {
            "baseline": baseline["summary"],
            "candidate": candidate["summary"],
            "delta": metric_delta(baseline["summary"], candidate["summary"]),
            "paired": summarize_paired_rows(paired_rows),
        },
        "improvements": [
            row for row in paired_rows if row["winner"] == "candidate"
        ][:50],
        "regressions": [
            row for row in paired_rows if row["winner"] == "baseline"
        ][:50],
        "ties": [row for row in paired_rows if row["winner"] == "tie"][:50],
    }


def load_evaluation(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("summary"), dict):
        raise ValueError(f"Evaluation JSON must contain a summary object: {path}")
    if not isinstance(payload.get("rows"), list):
        raise ValueError(f"Evaluation JSON must contain a rows list: {path}")
    return payload


def metric_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | None]:
    deltas = {}
    for key in METRIC_KEYS:
        base_value = baseline.get(key)
        candidate_value = candidate.get(key)
        if isinstance(base_value, int | float) and isinstance(candidate_value, int | float):
            deltas[key] = round(float(candidate_value) - float(base_value), 6)
        else:
            deltas[key] = None
    return deltas


def compare_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_image = {row["image"]: row for row in baseline_rows}
    candidate_by_image = {row["image"]: row for row in candidate_rows}
    images = sorted(set(baseline_by_image) | set(candidate_by_image))
    rows = []
    for image in images:
        baseline = baseline_by_image.get(image)
        candidate = candidate_by_image.get(image)
        baseline_score = row_score(baseline)
        candidate_score = row_score(candidate)
        winner = "tie"
        if candidate_score > baseline_score:
            winner = "candidate"
        elif baseline_score > candidate_score:
            winner = "baseline"
        rows.append(
            {
                "image": image,
                "winner": winner,
                "baseline": compact_row(baseline),
                "candidate": compact_row(candidate),
                "score_delta": round(candidate_score - baseline_score, 6),
            }
        )
    return rows


def row_score(row: dict[str, Any] | None) -> float:
    if row is None:
        return -10_000.0
    true_positive = float(row.get("true_positive") or 0)
    false_positive = float(row.get("false_positive") or 0)
    false_negative = float(row.get("false_negative") or 0)
    recognition_top1 = float(row.get("recognition_top1") or 0)
    recognition_top3 = float(row.get("recognition_top3") or 0)
    recognition_top5 = float(row.get("recognition_top5") or 0)
    warning_count = len(row.get("warnings") or [])
    return (
        true_positive * 100
        + recognition_top1 * 40
        + recognition_top3 * 30
        + recognition_top5 * 10
        - false_positive * 25
        - false_negative * 50
        - warning_count * 2
    )


def compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "gt_count": row.get("gt_count"),
        "pred_count": row.get("pred_count"),
        "true_positive": row.get("true_positive"),
        "false_positive": row.get("false_positive"),
        "false_negative": row.get("false_negative"),
        "recognition_top1": row.get("recognition_top1"),
        "recognition_top3": row.get("recognition_top3"),
        "recognition_top5": row.get("recognition_top5"),
        "scope_mode": row.get("scope_mode"),
        "candidate_scope": row.get("candidate_scope"),
    }


def summarize_paired_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_deltas = [float(row["score_delta"]) for row in rows]
    return {
        "images": len(rows),
        "candidate_wins": sum(1 for row in rows if row["winner"] == "candidate"),
        "baseline_wins": sum(1 for row in rows if row["winner"] == "baseline"),
        "ties": sum(1 for row in rows if row["winner"] == "tie"),
        "mean_score_delta": round(mean(score_deltas), 6) if score_deltas else None,
    }


if __name__ == "__main__":
    main()
