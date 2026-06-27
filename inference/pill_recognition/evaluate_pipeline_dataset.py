from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

from pill_recognition_legacy.postprocess import bbox_iou

from .pipeline import PillRecognitionPipeline
from .schemas import PillDetection, RecognitionResult
from .settings import Settings


@dataclass(frozen=True)
class GroundTruthPill:
    index: int
    class_name: str
    product_name: str | None
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class PipelineMatch:
    prediction_index: int
    ground_truth_index: int
    iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full RTMDet + retrieval pipeline on synthetic scenes."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/processed/rtmdet-aihub-synthetic-realistic-max10"),
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    if args.top_k is not None:
        settings = settings.__class__(**{**settings.__dict__, "top_k": args.top_k})
    pipeline = PillRecognitionPipeline(settings=settings)

    image_dir = args.dataset_root / "images" / args.split
    metadata_dir = args.dataset_root / "metadata" / args.split
    image_paths = sorted(path for path in image_dir.glob(args.pattern) if path.is_file())
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        metadata_path = metadata_dir / f"{image_path.stem}.json"
        ground_truth = load_ground_truth(metadata_path)
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        result = pipeline.recognize(image_rgb)
        row = evaluate_result(image_path.name, ground_truth, result, args.iou_threshold)
        rows.append(row)
        print(json.dumps(compact_row(row), ensure_ascii=False), flush=True)
        if index % 50 == 0:
            print(f"evaluated {index}/{len(image_paths)} images", flush=True)

    summary = summarize(rows, args.iou_threshold, settings.top_k)
    analysis = build_error_report(rows, top_k=settings.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"summary": summary, "analysis": analysis, "rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2), flush=True)


def load_ground_truth(path: Path) -> list[GroundTruthPill]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pills = []
    for index, pill in enumerate(payload.get("pills", [])):
        x1, y1, x2, y2 = pill["bbox_xyxy"]
        pills.append(
            GroundTruthPill(
                index=index,
                class_name=str(pill["class_name"]),
                product_name=pill.get("product_name"),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    return pills


def evaluate_result(
    image_name: str,
    ground_truth: list[GroundTruthPill],
    result: RecognitionResult,
    iou_threshold: float,
) -> dict:
    matches = match_detections(
        [detection.bbox for detection in result.detections],
        [pill.bbox for pill in ground_truth],
        iou_threshold,
    )
    recognition_rows = []
    for match in matches:
        detection = result.detections[match.prediction_index]
        truth = ground_truth[match.ground_truth_index]
        predicted_ids = [candidate.pill_id for candidate in detection.candidates]
        recognition_rows.append(
            {
                "prediction_index": match.prediction_index,
                "ground_truth_index": match.ground_truth_index,
                "iou": match.iou,
                "expected": truth.class_name,
                "expected_product_name": truth.product_name,
                "predicted": predicted_ids,
                "top1": predicted_ids[:1] == [truth.class_name],
                "top3": truth.class_name in predicted_ids[:3],
                "top5": truth.class_name in predicted_ids[:5],
                "status": detection.status,
                "status_reason": detection.status_reason,
                "top_candidate": candidate_to_row(
                    detection.candidates[0] if detection.candidates else None
                ),
                "candidates": [
                    candidate_to_row(candidate) for candidate in detection.candidates
                ],
            }
        )

    true_positive = len(matches)
    false_positive = len(result.detections) - true_positive
    false_negative = len(ground_truth) - true_positive
    return {
        "image": image_name,
        "gt_count": len(ground_truth),
        "pred_count": len(result.detections),
        "count_exact": len(ground_truth) == len(result.detections),
        "count_abs_error": abs(len(ground_truth) - len(result.detections)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "detector_precision": safe_divide(true_positive, len(result.detections)),
        "detector_recall": safe_divide(true_positive, len(ground_truth)),
        "mean_matched_iou": round(mean(match.iou for match in matches), 4)
        if matches
        else None,
        "recognition_top1": count_true(recognition_rows, "top1"),
        "recognition_top3": count_true(recognition_rows, "top3"),
        "recognition_top5": count_true(recognition_rows, "top5"),
        "ground_truth": [truth.__dict__ for truth in ground_truth],
        "detections": [detection_to_row(detection) for detection in result.detections],
        "matches": [match.__dict__ for match in matches],
        "recognition": recognition_rows,
        "warnings": result.warnings,
        "timings_ms": result.timings_ms,
    }


def match_detections(
    predictions: list[tuple[float, float, float, float]],
    ground_truth: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> list[PipelineMatch]:
    pairs = []
    for pred_index, pred_box in enumerate(predictions):
        for gt_index, gt_box in enumerate(ground_truth):
            iou = bbox_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                pairs.append((iou, pred_index, gt_index))

    matches = []
    used_predictions = set()
    used_ground_truth = set()
    for iou, pred_index, gt_index in sorted(pairs, reverse=True):
        if pred_index in used_predictions or gt_index in used_ground_truth:
            continue
        matches.append(
            PipelineMatch(
                prediction_index=pred_index,
                ground_truth_index=gt_index,
                iou=round(float(iou), 4),
            )
        )
        used_predictions.add(pred_index)
        used_ground_truth.add(gt_index)
    return sorted(matches, key=lambda match: match.prediction_index)


def summarize(rows: list[dict], iou_threshold: float, top_k: int) -> dict:
    image_count = len(rows)
    gt_total = sum(row["gt_count"] for row in rows)
    pred_total = sum(row["pred_count"] for row in rows)
    true_positive = sum(row["true_positive"] for row in rows)
    false_positive = sum(row["false_positive"] for row in rows)
    false_negative = sum(row["false_negative"] for row in rows)
    matched_total = true_positive
    top1 = sum(row["recognition_top1"] for row in rows)
    top3 = sum(row["recognition_top3"] for row in rows)
    top5 = sum(row["recognition_top5"] for row in rows)
    mean_ious = [
        row["mean_matched_iou"] for row in rows if row["mean_matched_iou"] is not None
    ]
    total_timings = [
        float(row["timings_ms"]["total"])
        for row in rows
        if row.get("timings_ms") and "total" in row["timings_ms"]
    ]
    return {
        "images": image_count,
        "gt_total": gt_total,
        "pred_total": pred_total,
        "iou_threshold": iou_threshold,
        "top_k": top_k,
        "count_exact": safe_divide(count_true(rows, "count_exact"), image_count),
        "mean_count_abs_error": round(
            mean(row["count_abs_error"] for row in rows), 4
        )
        if rows
        else 0,
        "detector_precision": safe_divide(true_positive, true_positive + false_positive),
        "detector_recall": safe_divide(true_positive, true_positive + false_negative),
        "detector_f1": f1_score(true_positive, false_positive, false_negative),
        "mean_matched_iou": round(mean(mean_ious), 4) if mean_ious else None,
        "matched_pills": matched_total,
        "recognition_top1_on_matched": safe_divide(top1, matched_total),
        "recognition_top3_on_matched": safe_divide(top3, matched_total),
        "recognition_top5_on_matched": safe_divide(top5, matched_total),
        "end_to_end_top1_on_gt": safe_divide(top1, gt_total),
        "end_to_end_top3_on_gt": safe_divide(top3, gt_total),
        "end_to_end_top5_on_gt": safe_divide(top5, gt_total),
        "warning_images": sum(1 for row in rows if row.get("warnings")),
        "status_counts": count_detection_statuses(rows),
        "mean_total_ms": round(mean(total_timings), 3) if total_timings else None,
        "p95_total_ms": percentile(total_timings, 95) if total_timings else None,
    }


def build_error_report(rows: list[dict], top_k: int, limit: int = 30) -> dict:
    count_mismatch = []
    detector_misses = []
    false_positives = []
    recognition_misses = []
    status_review = []
    warning_images = []

    for row in rows:
        image = row["image"]
        if row["gt_count"] != row["pred_count"]:
            count_mismatch.append(
                {
                    "image": image,
                    "gt_count": row["gt_count"],
                    "pred_count": row["pred_count"],
                    "count_abs_error": row["count_abs_error"],
                }
            )

        matched_gt = {match["ground_truth_index"] for match in row["matches"]}
        matched_pred = {match["prediction_index"] for match in row["matches"]}

        for truth in row["ground_truth"]:
            if truth["index"] not in matched_gt:
                detector_misses.append(
                    {
                        "image": image,
                        "expected": truth["class_name"],
                        "expected_product_name": truth.get("product_name"),
                        "bbox": truth["bbox"],
                    }
                )

        for index, detection in enumerate(row["detections"]):
            if index not in matched_pred:
                false_positives.append(
                    {
                        "image": image,
                        "prediction_index": index,
                        "bbox": detection["bbox"],
                        "detector_confidence": detection["detector_confidence"],
                        "status": detection.get("status"),
                        "top_candidate": detection["candidates"][0]
                        if detection["candidates"]
                        else None,
                    }
                )
            if detection.get("status") in {"no_candidate", "low_confidence", "ambiguous"}:
                status_review.append(
                    {
                        "image": image,
                        "prediction_index": index,
                        "status": detection.get("status"),
                        "status_reason": detection.get("status_reason"),
                        "top_candidate": detection["candidates"][0]
                        if detection["candidates"]
                        else None,
                    }
                )

        for item in row["recognition"]:
            if not item.get("top3"):
                recognition_misses.append(
                    {
                        "image": image,
                        "prediction_index": item["prediction_index"],
                        "ground_truth_index": item["ground_truth_index"],
                        "iou": item["iou"],
                        "expected": item["expected"],
                        "expected_product_name": item.get("expected_product_name"),
                        "predicted": item["predicted"][:top_k],
                        "status": item.get("status"),
                        "status_reason": item.get("status_reason"),
                        "top_candidate": item.get("top_candidate"),
                    }
                )

        if row.get("warnings"):
            warning_images.append({"image": image, "warnings": row["warnings"]})

    return {
        "counts": {
            "count_mismatch_images": len(count_mismatch),
            "detector_misses": len(detector_misses),
            "false_positives": len(false_positives),
            "recognition_top3_misses": len(recognition_misses),
            "status_review_detections": len(status_review),
            "warning_images": len(warning_images),
        },
        "count_mismatch": count_mismatch[:limit],
        "detector_misses": detector_misses[:limit],
        "false_positives": false_positives[:limit],
        "recognition_top3_misses": recognition_misses[:limit],
        "status_review": status_review[:limit],
        "warning_images": warning_images[:limit],
    }


def compact_row(row: dict) -> dict:
    return {
        "image": row["image"],
        "gt_count": row["gt_count"],
        "pred_count": row["pred_count"],
        "true_positive": row["true_positive"],
        "false_positive": row["false_positive"],
        "false_negative": row["false_negative"],
        "recognition_top3": row["recognition_top3"],
    }


def detection_to_row(detection: PillDetection) -> dict:
    return {
        "bbox": detection.bbox,
        "crop_bbox": detection.crop_bbox,
        "detector_confidence": detection.detector_confidence,
        "status": detection.status,
        "status_reason": detection.status_reason,
        "candidates": [candidate_to_row(candidate) for candidate in detection.candidates],
    }


def candidate_to_row(candidate) -> dict | None:
    if candidate is None:
        return None
    return {
        "pill_id": candidate.pill_id,
        "score": candidate.score,
        "product_name": candidate.product_name,
        "ingredient": candidate.ingredient,
        "company": candidate.company,
    }


def count_true(rows: list[dict], key: str) -> int:
    return sum(1 for row in rows if row.get(key))


def count_detection_statuses(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for detection in row.get("detections", []):
            status = str(detection.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / denominator, 6) if denominator else 0.0


def f1_score(true_positive: int, false_positive: int, false_negative: int) -> float:
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    return safe_divide(2 * precision * recall, precision + recall)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(value, 3)


if __name__ == "__main__":
    main()
