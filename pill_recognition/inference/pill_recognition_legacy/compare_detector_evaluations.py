from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = [
    "count_exact_accuracy",
    "count_mean_abs_error",
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "false_positive",
    "false_negative",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two detector evaluation output directories."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--name-baseline",
        default="baseline",
        help="Human-readable baseline run name.",
    )
    parser.add_argument(
        "--name-candidate",
        default="candidate",
        help="Human-readable candidate run name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_evaluations(
        baseline_dir=args.baseline,
        candidate_dir=args.candidate,
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


def compare_evaluations(
    baseline_dir: Path,
    candidate_dir: Path,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> dict[str, Any]:
    baseline_summary = load_json(baseline_dir / "summary.json")
    candidate_summary = load_json(candidate_dir / "summary.json")
    baseline_rows = load_rows(baseline_dir / "results.json")
    candidate_rows = load_rows(candidate_dir / "results.json")
    paired_rows = compare_rows(baseline_rows, candidate_rows)
    return {
        "runs": {
            "baseline": {"name": baseline_name, "dir": str(baseline_dir)},
            "candidate": {"name": candidate_name, "dir": str(candidate_dir)},
        },
        "summary": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "delta": metric_delta(baseline_summary, candidate_summary),
            "paired": summarize_paired_rows(paired_rows),
        },
        "improvements": [
            row for row in paired_rows if row["winner"] == "candidate"
        ][:50],
        "regressions": [
            row for row in paired_rows if row["winner"] == "baseline"
        ][:50],
        "ties": [
            row for row in paired_rows if row["winner"] == "tie"
        ][:50],
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
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
    count_abs_error = float(row.get("count_abs_error") or 0)
    mean_iou = float(row.get("mean_matched_iou") or 0)
    return true_positive * 100 + mean_iou - false_positive * 25 - false_negative * 50 - count_abs_error


def compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "gt_count": row.get("gt_count"),
        "pred_count": row.get("pred_count"),
        "count_abs_error": row.get("count_abs_error"),
        "true_positive": row.get("true_positive"),
        "false_positive": row.get("false_positive"),
        "false_negative": row.get("false_negative"),
        "mean_matched_iou": row.get("mean_matched_iou"),
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
