from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import cv2

from .assets import download_model_assets
from .detector import RTMDetPillDetector
from .postprocess import bbox_iou
from .settings import Settings


@dataclass(frozen=True)
class DetectionMatch:
    prediction_index: int
    ground_truth_index: int
    iou: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RTMDet pill detector against YOLO-format labels."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("../datasets/processed/rtmdet-single-class/images/val"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("../datasets/processed/rtmdet-single-class/labels/val"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--annotated-limit", type=int)
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.confidence_threshold is not None:
        settings = settings.__class__(
            **{
                **settings.__dict__,
                "confidence_threshold": args.confidence_threshold,
            }
        )

    assets = download_model_assets(settings)
    detector = RTMDetPillDetector(assets.checkpoint, assets.class_names, settings)
    image_paths = sorted(path for path in args.images.glob(args.pattern) if path.is_file())
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "annotated"
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, image_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        label_path = args.labels / f"{image_path.stem}.txt"
        ground_truth = load_yolo_boxes(label_path, width, height)
        predictions = [
            {
                "bbox": bbox,
                "confidence": candidates[0].confidence if candidates else 0.0,
            }
            for bbox, candidates in detector.predict(image_rgb)
        ]
        matches = match_detections(
            [prediction["bbox"] for prediction in predictions],
            ground_truth,
            args.iou_threshold,
        )
        row = build_row(image_path.name, ground_truth, predictions, matches)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if args.save_annotated and (
            args.annotated_limit is None or index < args.annotated_limit
        ):
            annotated = draw_detector_evaluation(
                image_bgr,
                ground_truth,
                predictions,
                matches,
            )
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)

    summary = summarize(rows, args.iou_threshold, settings.confidence_threshold)
    (args.output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "results.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


def load_yolo_boxes(path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        return []

    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label row in {path}: {line}")
        _, cx, cy, box_w, box_h = (float(value) for value in parts)
        boxes.append(yolo_to_xyxy(cx, cy, box_w, box_h, width, height))
    return boxes


def yolo_to_xyxy(
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    width = box_width * image_width
    height = box_height * image_height
    x1 = center_x * image_width - width / 2
    y1 = center_y * image_height - height / 2
    x2 = x1 + width
    y2 = y1 + height
    return (
        max(0.0, x1),
        max(0.0, y1),
        min(float(image_width), x2),
        min(float(image_height), y2),
    )


def match_detections(
    predictions: list[tuple[float, float, float, float]],
    ground_truth: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> list[DetectionMatch]:
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
            DetectionMatch(
                prediction_index=pred_index,
                ground_truth_index=gt_index,
                iou=round(float(iou), 4),
            )
        )
        used_predictions.add(pred_index)
        used_ground_truth.add(gt_index)
    return sorted(matches, key=lambda match: match.prediction_index)


def build_row(
    image_name: str,
    ground_truth: list[tuple[float, float, float, float]],
    predictions: list[dict],
    matches: list[DetectionMatch],
) -> dict:
    true_positive = len(matches)
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    return {
        "image": image_name,
        "gt_count": len(ground_truth),
        "pred_count": len(predictions),
        "count_exact": len(ground_truth) == len(predictions),
        "count_abs_error": abs(len(ground_truth) - len(predictions)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "mean_matched_iou": (
            round(mean(match.iou for match in matches), 4) if matches else None
        ),
        "ground_truth": [tuple(round(value, 2) for value in box) for box in ground_truth],
        "predictions": predictions,
        "matches": [match.__dict__ for match in matches],
    }


def draw_detector_evaluation(
    image_bgr,
    ground_truth: list[tuple[float, float, float, float]],
    predictions: list[dict],
    matches: list[DetectionMatch],
):
    canvas = image_bgr.copy()
    matched_predictions = {match.prediction_index for match in matches}
    matched_ground_truth = {match.ground_truth_index for match in matches}

    for index, box in enumerate(ground_truth):
        color = (34, 197, 94) if index in matched_ground_truth else (0, 0, 255)
        draw_box(canvas, box, color, f"GT {index + 1}")

    for index, prediction in enumerate(predictions):
        color = (0, 165, 255) if index in matched_predictions else (255, 0, 255)
        confidence = prediction.get("confidence", 0.0)
        draw_box(canvas, prediction["bbox"], color, f"P {index + 1} {confidence:.2f}")

    cv2.putText(
        canvas,
        "GT=green/red  Pred=orange/magenta",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GT=green/red  Pred=orange/magenta",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_box(image, box, color, label: str) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def summarize(rows: list[dict], iou_threshold: float, confidence_threshold: float) -> dict:
    true_positive = sum(row["true_positive"] for row in rows)
    false_positive = sum(row["false_positive"] for row in rows)
    false_negative = sum(row["false_negative"] for row in rows)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    return {
        "image_count": len(rows),
        "ground_truth_boxes": sum(row["gt_count"] for row in rows),
        "predicted_boxes": sum(row["pred_count"] for row in rows),
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "count_exact_accuracy": ratio(row["count_exact"] for row in rows),
        "count_mean_abs_error": round(mean(row["count_abs_error"] for row in rows), 4)
        if rows
        else None,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None,
        "mean_matched_iou": round(
            mean(
                match["iou"]
                for row in rows
                for match in row["matches"]
            ),
            4,
        )
        if true_positive
        else None,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def ratio(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image",
        "gt_count",
        "pred_count",
        "count_exact",
        "count_abs_error",
        "true_positive",
        "false_positive",
        "false_negative",
        "mean_matched_iou",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


if __name__ == "__main__":
    main()
