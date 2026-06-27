from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .evaluate_pipeline_dataset import (
    GroundTruthPill,
    build_error_report,
    compact_row,
    evaluate_result,
    summarize,
)
from .pipeline import PillRecognitionPipeline
from .settings import Settings


@dataclass(frozen=True)
class RealEvaluationExample:
    image_path: Path
    annotation_path: Path
    ground_truth: list[GroundTruthPill]
    allowed_pill_ids: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full RTMDet + retrieval pipeline on real smartphone photos."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/evaluation/real-smartphone"),
    )
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--scope-mode",
        choices=["none", "annotation", "ground-truth"],
        default="none",
        help=(
            "Limit retrieval to no scope, annotation allowed_pill_ids, or the "
            "ground-truth K-IDs as an oracle user medication list."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    if args.top_k is not None:
        settings = settings.__class__(**{**settings.__dict__, "top_k": args.top_k})
    pipeline = PillRecognitionPipeline(settings=settings)

    examples = load_real_examples(
        dataset_root=args.dataset_root,
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        pattern=args.pattern,
        limit=args.limit,
    )
    rows = []
    for index, example in enumerate(examples, start=1):
        image_rgb = np.asarray(Image.open(example.image_path).convert("RGB"))
        allowed_pill_ids = evaluation_scope_for_example(example, args.scope_mode)
        result = pipeline.recognize(image_rgb, allowed_pill_ids=allowed_pill_ids)
        row = evaluate_result(
            example.image_path.name,
            example.ground_truth,
            result,
            args.iou_threshold,
        )
        row["annotation"] = str(example.annotation_path)
        row["scope_mode"] = args.scope_mode
        row["allowed_pill_ids"] = sorted(allowed_pill_ids)
        row["candidate_scope"] = result.candidate_scope
        rows.append(row)
        print(json.dumps(compact_row(row), ensure_ascii=False), flush=True)
        if index % 20 == 0:
            print(f"evaluated {index}/{len(examples)} images", flush=True)

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


def load_real_examples(
    dataset_root: Path,
    images_dir: Path | None = None,
    annotations_dir: Path | None = None,
    pattern: str = "*.json",
    limit: int | None = None,
) -> list[RealEvaluationExample]:
    images_root = images_dir or dataset_root / "images"
    annotations_root = annotations_dir or dataset_root / "annotations"
    if not images_root.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_root}")
    if not annotations_root.exists():
        raise FileNotFoundError(f"Annotations directory does not exist: {annotations_root}")

    annotation_paths = sorted(path for path in annotations_root.glob(pattern) if path.is_file())
    if limit is not None:
        annotation_paths = annotation_paths[:limit]

    examples = []
    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = resolve_image_path(images_root, annotation_path, payload)
        ground_truth = load_real_ground_truth(annotation_path, payload)
        allowed_pill_ids = load_allowed_pill_ids(payload)
        examples.append(
            RealEvaluationExample(
                image_path=image_path,
                annotation_path=annotation_path,
                ground_truth=ground_truth,
                allowed_pill_ids=allowed_pill_ids,
            )
        )
    return examples


def evaluation_scope_for_example(
    example: RealEvaluationExample,
    scope_mode: str,
) -> set[str]:
    if scope_mode == "none":
        return set()
    if scope_mode == "annotation":
        return set(example.allowed_pill_ids)
    if scope_mode == "ground-truth":
        return {
            pill.class_name
            for pill in example.ground_truth
            if str(pill.class_name or "").strip()
        }
    raise ValueError(f"Unsupported scope mode: {scope_mode}")


def load_allowed_pill_ids(payload: dict) -> set[str]:
    values = payload.get("allowed_pill_ids") or []
    if not isinstance(values, list):
        raise ValueError("allowed_pill_ids must be a list when provided.")
    return {str(value).strip() for value in values if str(value).strip()}


def resolve_image_path(images_root: Path, annotation_path: Path, payload: dict) -> Path:
    image_name = payload.get("image")
    if image_name:
        image_path = images_root / str(image_name)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Annotation image does not exist: {annotation_path} -> {image_path}"
            )
        return image_path

    matches = []
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_root / f"{annotation_path.stem}{suffix}"
        if candidate.exists():
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Cannot find image for annotation {annotation_path}. "
            "Set the annotation 'image' field explicitly."
        )
    raise ValueError(f"Multiple image files match annotation {annotation_path}: {matches}")


def load_real_ground_truth(
    annotation_path: Path,
    payload: dict,
) -> list[GroundTruthPill]:
    pills = payload.get("pills")
    if not isinstance(pills, list):
        raise ValueError(f"Annotation must contain a pills list: {annotation_path}")

    ground_truth = []
    for index, pill in enumerate(pills):
        if not isinstance(pill, dict):
            raise ValueError(f"Invalid pill row in {annotation_path}: {pill}")
        class_name = str(pill.get("class_name") or "").strip()
        if not class_name:
            raise ValueError(f"Missing class_name in {annotation_path} pill #{index + 1}")
        bbox = parse_bbox(annotation_path, pill, index)
        ground_truth.append(
            GroundTruthPill(
                index=index,
                class_name=class_name,
                product_name=pill.get("product_name"),
                bbox=bbox,
            )
        )
    return ground_truth


def parse_bbox(
    annotation_path: Path,
    pill: dict,
    index: int,
) -> tuple[float, float, float, float]:
    bbox = pill.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(
            f"bbox_xyxy [x1, y1, x2, y2] is required in "
            f"{annotation_path} pill #{index + 1}"
        )
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"bbox_xyxy must satisfy x2>x1 and y2>y1 in "
            f"{annotation_path} pill #{index + 1}: {bbox}"
        )
    return x1, y1, x2, y2


if __name__ == "__main__":
    main()
