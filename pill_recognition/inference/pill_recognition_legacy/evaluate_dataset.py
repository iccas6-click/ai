from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

import cv2

from .pipeline import PillRecognitionPipeline
from .settings import Settings
from .visualization import draw_detections


@dataclass(frozen=True)
class EvaluationTarget:
    image: str
    expected_count: int | None = None
    expected_class_names: list[str] = field(default_factory=list)
    expected_item_seqs: list[str] = field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the pill recognition pipeline on an image directory."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-annotated", action="store_true")
    args = parser.parse_args()

    targets = load_targets(args.images, args.manifest, args.pattern, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output_dir / "annotated"
    if not args.no_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PillRecognitionPipeline(Settings.from_env())
    rows = []
    for target in targets:
        image_path = args.images / target.image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = pipeline.recognize(image_rgb)
        row = build_result_row(target, result)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if not args.no_annotated:
            annotated = draw_detections(image_rgb, result)
            cv2.imwrite(
                str(annotated_dir / image_path.name),
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
            )

    summary = summarize(rows)
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


def load_targets(
    image_root: Path,
    manifest: Path | None,
    pattern: str,
    limit: int | None,
) -> list[EvaluationTarget]:
    if manifest is None:
        image_paths = sorted(path for path in image_root.glob(pattern) if path.is_file())
        if limit is not None:
            image_paths = image_paths[:limit]
        return [EvaluationTarget(image=path.name) for path in image_paths]

    if manifest.suffix.lower() == ".json":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        rows = payload["images"] if isinstance(payload, dict) else payload
    else:
        with manifest.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))

    targets = [target_from_row(row) for row in rows]
    return targets[:limit] if limit is not None else targets


def target_from_row(row) -> EvaluationTarget:
    return EvaluationTarget(
        image=str(row["image"]),
        expected_count=parse_optional_int(row.get("expected_count")),
        expected_class_names=parse_list(row.get("expected_class_names")),
        expected_item_seqs=parse_list(row.get("expected_item_seqs")),
    )


def parse_optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def parse_list(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).replace(",", ";").split(";") if item.strip()]


def build_result_row(target: EvaluationTarget, result) -> dict:
    detections = []
    predicted_class_names = []
    predicted_item_seqs = []
    top3_class_names = []
    top3_item_seqs = []

    for detection in result.detections:
        rtmdet = detection.rtmdet_candidates[0]
        candidates = detection.aihub_candidates
        top1 = candidates[0] if candidates else None
        predicted_class_names.append(top1.class_name if top1 else "")
        predicted_item_seqs.append(top1.item_seq or "" if top1 else "")
        top3_class_names.extend(candidate.class_name for candidate in candidates[:3])
        top3_item_seqs.extend(candidate.item_seq or "" for candidate in candidates[:3])
        detections.append(
            {
                "pill_id": detection.pill_id,
                "bbox": detection.bbox,
                "detector_confidence": rtmdet.confidence,
                "aihub_top3": [asdict(candidate) for candidate in candidates[:3]],
                "status": detection.status,
            }
        )

    return {
        "image": target.image,
        "expected_count": target.expected_count,
        "detected_count": result.pill_count,
        "count_exact": (
            target.expected_count == result.pill_count
            if target.expected_count is not None
            else None
        ),
        "count_abs_error": (
            abs(target.expected_count - result.pill_count)
            if target.expected_count is not None
            else None
        ),
        "expected_class_names": target.expected_class_names,
        "predicted_class_names": predicted_class_names,
        "class_top1_all_hit": all_present(
            target.expected_class_names,
            predicted_class_names,
        ),
        "class_top3_all_hit": all_present(target.expected_class_names, top3_class_names),
        "expected_item_seqs": target.expected_item_seqs,
        "predicted_item_seqs": predicted_item_seqs,
        "item_seq_top1_all_hit": all_present(target.expected_item_seqs, predicted_item_seqs),
        "item_seq_top3_all_hit": all_present(target.expected_item_seqs, top3_item_seqs),
        "warnings": result.warnings,
        "detections": detections,
    }


def all_present(expected: list[str], predicted: list[str]) -> bool | None:
    if not expected:
        return None
    predicted_set = {value for value in predicted if value}
    return all(value in predicted_set for value in expected)


def summarize(rows: list[dict]) -> dict:
    count_rows = [row for row in rows if row["expected_count"] is not None]
    class_rows = [row for row in rows if row["class_top1_all_hit"] is not None]
    item_rows = [row for row in rows if row["item_seq_top1_all_hit"] is not None]
    return {
        "image_count": len(rows),
        "count_labeled_images": len(count_rows),
        "count_exact_accuracy": ratio(row["count_exact"] for row in count_rows),
        "count_mean_abs_error": (
            round(mean(row["count_abs_error"] for row in count_rows), 4)
            if count_rows
            else None
        ),
        "class_labeled_images": len(class_rows),
        "class_top1_all_hit_accuracy": ratio(row["class_top1_all_hit"] for row in class_rows),
        "class_top3_all_hit_accuracy": ratio(row["class_top3_all_hit"] for row in class_rows),
        "item_seq_labeled_images": len(item_rows),
        "item_seq_top1_all_hit_accuracy": ratio(
            row["item_seq_top1_all_hit"] for row in item_rows
        ),
        "item_seq_top3_all_hit_accuracy": ratio(
            row["item_seq_top3_all_hit"] for row in item_rows
        ),
    }


def ratio(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image",
        "expected_count",
        "detected_count",
        "count_exact",
        "count_abs_error",
        "expected_class_names",
        "predicted_class_names",
        "class_top1_all_hit",
        "class_top3_all_hit",
        "expected_item_seqs",
        "predicted_item_seqs",
        "item_seq_top1_all_hit",
        "item_seq_top3_all_hit",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ";".join(row[key]) if isinstance(row[key], list) else row[key]
                    for key in fieldnames
                }
            )


if __name__ == "__main__":
    main()
