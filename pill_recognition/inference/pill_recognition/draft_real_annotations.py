from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import PillRecognitionPipeline
from .schemas import PillDetection
from .settings import Settings


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draft real-smartphone evaluation annotations from detector output."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/evaluation/real-smartphone"),
    )
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing draft annotations.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Write annotation files even when no pill is detected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    settings = settings.__class__(**{**settings.__dict__, "top_k": args.top_k})
    pipeline = PillRecognitionPipeline(settings=settings)

    images_dir = args.images_dir or args.dataset_root / "images"
    annotations_dir = args.annotations_dir or args.dataset_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_image_paths(images_dir, args.pattern)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    written = 0
    skipped = 0
    for index, image_path in enumerate(image_paths, start=1):
        output_path = annotations_dir / f"{image_path.stem}.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        result = pipeline.recognize(image_rgb)
        if not result.detections and not args.include_empty:
            skipped += 1
            continue

        payload = draft_annotation_payload(image_path, result.detections)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written += 1
        print(
            json.dumps(
                {
                    "image": image_path.name,
                    "annotation": str(output_path),
                    "detections": len(result.detections),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if index % 20 == 0:
            print(f"processed {index}/{len(image_paths)} images", flush=True)

    print(
        json.dumps(
            {"images": len(image_paths), "written": written, "skipped": skipped},
            ensure_ascii=False,
        ),
        flush=True,
    )


def find_image_paths(images_dir: Path, pattern: str) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    return sorted(
        path
        for path in images_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def draft_annotation_payload(
    image_path: Path,
    detections: list[PillDetection],
) -> dict:
    return {
        "image": image_path.name,
        "capture": {
            "device": None,
            "background": None,
            "lighting": None,
            "notes": "Draft generated from RTMDet/retrieval. Review bbox and class_name.",
        },
        "pills": [
            draft_pill_row(index, detection)
            for index, detection in enumerate(detections, start=1)
        ],
    }


def draft_pill_row(index: int, detection: PillDetection) -> dict:
    top_candidate = detection.candidates[0] if detection.candidates else None
    return {
        "pill_id": index,
        "class_name": top_candidate.pill_id if top_candidate else "",
        "product_name": top_candidate.product_name if top_candidate else None,
        "bbox_xyxy": list(detection.bbox),
        "detector_confidence": detection.detector_confidence,
        "needs_review": True,
        "candidate_hints": [
            {
                "rank": candidate.rank,
                "class_name": candidate.pill_id,
                "product_name": candidate.product_name,
                "ingredient": candidate.ingredient,
                "company": candidate.company,
                "score": candidate.score,
            }
            for candidate in detection.candidates
        ],
        "notes": "Set class_name to the verified AIHub K-ID before evaluation.",
    }


if __name__ == "__main__":
    main()
