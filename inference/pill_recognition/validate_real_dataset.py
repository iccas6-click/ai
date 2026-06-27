from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .evaluate_real_dataset import (
    load_allowed_pill_ids,
    load_real_ground_truth,
    resolve_image_path,
)
from .product_db import load_product_index
from .settings import Settings


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    annotation: str
    message: str
    pill_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "annotation": self.annotation,
            "pill_index": self.pill_index,
            "message": self.message,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real-smartphone pill recognition evaluation annotations."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/evaluation/real-smartphone"),
    )
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-needs-review",
        action="store_true",
        help="Do not fail when a pill row still has needs_review=true.",
    )
    parser.add_argument(
        "--skip-asset-checks",
        action="store_true",
        help="Skip AIHub product metadata and retrieval index K-ID checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    product_ids = None
    retrieval_ids = None
    if not args.skip_asset_checks:
        settings = Settings.from_env()
        product_ids = load_product_ids(settings.aihub_mapping)
        retrieval_ids = load_retrieval_ids(settings.retrieval_index)

    report = validate_real_dataset(
        dataset_root=args.dataset_root,
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        pattern=args.pattern,
        product_ids=product_ids,
        retrieval_ids=retrieval_ids,
        allow_needs_review=args.allow_needs_review,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if report["summary"]["errors"] > 0:
        raise SystemExit(1)


def validate_real_dataset(
    dataset_root: Path,
    images_dir: Path | None = None,
    annotations_dir: Path | None = None,
    pattern: str = "*.json",
    product_ids: set[str] | None = None,
    retrieval_ids: set[str] | None = None,
    allow_needs_review: bool = False,
) -> dict[str, Any]:
    images_root = images_dir or dataset_root / "images"
    annotations_root = annotations_dir or dataset_root / "annotations"
    issues: list[ValidationIssue] = []

    if not images_root.exists():
        issues.append(
            issue("error", "missing_images_dir", str(images_root), "Images directory is missing.")
        )
    if not annotations_root.exists():
        issues.append(
            issue(
                "error",
                "missing_annotations_dir",
                str(annotations_root),
                "Annotations directory is missing.",
            )
        )
    if issues:
        return validation_report([], issues)

    annotation_paths = sorted(path for path in annotations_root.glob(pattern) if path.is_file())
    validated = []
    for annotation_path in annotation_paths:
        row = validate_annotation(
            images_root=images_root,
            annotation_path=annotation_path,
            product_ids=product_ids,
            retrieval_ids=retrieval_ids,
            allow_needs_review=allow_needs_review,
        )
        validated.append(row)
        issues.extend(issue_from_dict(item) for item in row["issues"])
    return validation_report(validated, issues)


def validate_annotation(
    images_root: Path,
    annotation_path: Path,
    product_ids: set[str] | None = None,
    retrieval_ids: set[str] | None = None,
    allow_needs_review: bool = False,
) -> dict[str, Any]:
    issues: list[ValidationIssue] = []
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        issues.append(
            issue("error", "invalid_json", str(annotation_path), f"Invalid JSON: {error}")
        )
        return annotation_row(annotation_path, None, 0, set(), issues)

    try:
        image_path = resolve_image_path(images_root, annotation_path, payload)
        width, height = Image.open(image_path).size
    except Exception as error:
        issues.append(
            issue(
                "error",
                "image_resolution_failed",
                str(annotation_path),
                f"Cannot resolve/open image: {type(error).__name__}: {error}",
            )
        )
        image_path = None
        width = height = 0

    try:
        ground_truth = load_real_ground_truth(annotation_path, payload)
    except Exception as error:
        issues.append(
            issue(
                "error",
                "ground_truth_invalid",
                str(annotation_path),
                f"Invalid ground truth: {type(error).__name__}: {error}",
            )
        )
        ground_truth = []

    try:
        allowed_pill_ids = load_allowed_pill_ids(payload)
    except Exception as error:
        issues.append(
            issue(
                "error",
                "allowed_pill_ids_invalid",
                str(annotation_path),
                f"Invalid allowed_pill_ids: {type(error).__name__}: {error}",
            )
        )
        allowed_pill_ids = set()

    pills = payload.get("pills") if isinstance(payload, dict) else []
    if isinstance(pills, list):
        for index, pill in enumerate(pills, start=1):
            if isinstance(pill, dict):
                validate_pill_row(
                    annotation_path=annotation_path,
                    pill=pill,
                    index=index,
                    image_size=(width, height),
                    product_ids=product_ids,
                    retrieval_ids=retrieval_ids,
                    allowed_pill_ids=allowed_pill_ids,
                    allow_needs_review=allow_needs_review,
                    issues=issues,
                )

    for allowed_id in sorted(allowed_pill_ids):
        validate_id_presence(
            annotation_path=annotation_path,
            pill_index=None,
            pill_id=allowed_id,
            product_ids=product_ids,
            retrieval_ids=retrieval_ids,
            issues=issues,
            prefix="allowed",
        )

    return annotation_row(
        annotation_path,
        image_path,
        len(ground_truth),
        allowed_pill_ids,
        issues,
    )


def validate_pill_row(
    annotation_path: Path,
    pill: dict[str, Any],
    index: int,
    image_size: tuple[int, int],
    product_ids: set[str] | None,
    retrieval_ids: set[str] | None,
    allowed_pill_ids: set[str],
    allow_needs_review: bool,
    issues: list[ValidationIssue],
) -> None:
    class_name = str(pill.get("class_name") or "").strip()
    if class_name:
        validate_id_presence(
            annotation_path,
            index,
            class_name,
            product_ids,
            retrieval_ids,
            issues,
            prefix="class",
        )
        if allowed_pill_ids and class_name not in allowed_pill_ids:
            issues.append(
                issue(
                    "error",
                    "class_not_in_allowed_pill_ids",
                    str(annotation_path),
                    f"{class_name} is not present in allowed_pill_ids.",
                    index,
                )
            )

    if pill.get("needs_review") is True and not allow_needs_review:
        issues.append(
            issue(
                "error",
                "needs_review",
                str(annotation_path),
                "Pill row still has needs_review=true.",
                index,
            )
        )

    bbox = pill.get("bbox_xyxy")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and image_size != (0, 0):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width, height = image_size
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            issues.append(
                issue(
                    "error",
                    "bbox_out_of_bounds",
                    str(annotation_path),
                    f"bbox {bbox} is outside image bounds {width}x{height}.",
                    index,
                )
            )


def validate_id_presence(
    annotation_path: Path,
    pill_index: int | None,
    pill_id: str,
    product_ids: set[str] | None,
    retrieval_ids: set[str] | None,
    issues: list[ValidationIssue],
    prefix: str,
) -> None:
    if product_ids is not None and pill_id not in product_ids:
        issues.append(
            issue(
                "error",
                f"{prefix}_id_missing_product_metadata",
                str(annotation_path),
                f"{pill_id} is not present in AIHub product metadata.",
                pill_index,
            )
        )
    if retrieval_ids is not None and pill_id not in retrieval_ids:
        issues.append(
            issue(
                "error",
                f"{prefix}_id_missing_retrieval_index",
                str(annotation_path),
                f"{pill_id} is not present in the retrieval index.",
                pill_index,
            )
        )


def annotation_row(
    annotation_path: Path,
    image_path: Path | None,
    pill_count: int,
    allowed_pill_ids: set[str],
    issues: list[ValidationIssue],
) -> dict[str, Any]:
    return {
        "annotation": str(annotation_path),
        "image": str(image_path) if image_path else None,
        "pill_count": pill_count,
        "allowed_pill_ids": sorted(allowed_pill_ids),
        "issues": [item.to_dict() for item in issues],
    }


def validation_report(rows: list[dict[str, Any]], issues: list[ValidationIssue]) -> dict[str, Any]:
    return {
        "summary": {
            "annotations": len(rows),
            "pills": sum(row["pill_count"] for row in rows),
            "errors": sum(1 for item in issues if item.severity == "error"),
            "warnings": sum(1 for item in issues if item.severity == "warning"),
        },
        "issues": [item.to_dict() for item in issues],
        "rows": rows,
    }


def issue(
    severity: str,
    code: str,
    annotation: str,
    message: str,
    pill_index: int | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        annotation=annotation,
        pill_index=pill_index,
        message=message,
    )


def issue_from_dict(payload: dict[str, Any]) -> ValidationIssue:
    return ValidationIssue(
        severity=payload["severity"],
        code=payload["code"],
        annotation=payload["annotation"],
        pill_index=payload.get("pill_index"),
        message=payload["message"],
    )


def load_product_ids(mapping_path: Path | None) -> set[str] | None:
    if mapping_path is None:
        return None
    return set(load_product_index(mapping_path))


def load_retrieval_ids(index_path: Path) -> set[str] | None:
    if not index_path.exists():
        return None
    import torch

    payload = torch.load(index_path, map_location="cpu", weights_only=True)
    pill_ids = payload.get("pill_ids")
    if not isinstance(pill_ids, list):
        raise ValueError(f"Retrieval index has no pill_ids list: {index_path}")
    return {str(pill_id) for pill_id in pill_ids}


if __name__ == "__main__":
    main()
