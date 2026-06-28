from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "processed" / "rtmdet-single-class"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitStats:
    images: int = 0
    annotations: int = 0
    missing_labels: int = 0
    invalid_labels: int = 0


def find_yolo_roots(source: Path) -> list[Path]:
    roots = []
    for images_dir in source.rglob("images"):
        parent = images_dir.parent
        if (parent / "labels").is_dir() and any(
            (images_dir / split).is_dir() for split in ("train", "val", "validation")
        ):
            roots.append(parent)
    return sorted(set(roots))


def prepare_sources(sources: list[Path], output: Path) -> dict[str, SplitStats]:
    layouts = []
    for source in sources:
        found = find_yolo_roots(source)
        if not found:
            raise FileNotFoundError(f"No YOLO images/labels layout found under {source}")
        layouts.extend((source.name, layout) for layout in found)

    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split in ("train", "val"):
        summaries[split] = prepare_split(layouts, split, output)

    (output / "pill.yaml").write_text(
        "names:\n  0: pill\n",
        encoding="utf-8",
    )
    manifest = {
        split: vars(stats)
        for split, stats in summaries.items()
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summaries


def prepare_split(
    layouts: list[tuple[str, Path]],
    split: str,
    output: Path,
) -> SplitStats:
    image_output = output / "images" / split
    label_output = output / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    stats = SplitStats()
    annotation_id = 1

    for source_name, layout in layouts:
        source_split = "validation" if split == "val" and (layout / "images" / "validation").is_dir() else split
        images_dir = layout / "images" / source_split
        labels_dir = layout / "labels" / source_split
        if not images_dir.is_dir():
            continue

        prefix = slug(f"{source_name}-{layout.name}")
        for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                stats.missing_labels += 1
                continue

            output_name = f"{prefix}-{image_path.name}"
            output_image = image_output / output_name
            output_label = label_output / f"{Path(output_name).stem}.txt"
            link_or_copy(image_path, output_image)

            with Image.open(image_path) as image:
                width, height = image.size
            image_id = len(coco_images) + 1
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": output_name,
                    "width": width,
                    "height": height,
                }
            )

            converted_lines = []
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    _, x_center, y_center, box_width, box_height = parse_yolo_line(line)
                except ValueError as error:
                    stats.invalid_labels += 1
                    raise ValueError(f"{label_path}:{line_number}: {error}") from error

                converted_lines.append(
                    f"0 {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
                )
                x = (x_center - box_width / 2) * width
                y = (y_center - box_height / 2) * height
                w = box_width * width
                h = box_height * height
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

            output_label.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
            stats.images += 1
            stats.annotations += len(converted_lines)

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "pill", "supercategory": "pill"}],
    }
    (output / f"{split}_coco.json").write_text(
        json.dumps(coco, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 YOLO fields, got {len(parts)}")
    class_id = int(float(parts[0]))
    values = tuple(float(value) for value in parts[1:])
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError("normalized bbox values must be between 0 and 1")
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("bbox width and height must be positive")
    return (class_id, *values)


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collapse YOLO pill classes into one class")
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summaries = prepare_sources(
        [source.expanduser().resolve() for source in args.sources],
        args.output.expanduser().resolve(),
    )
    for split, stats in summaries.items():
        print(split, vars(stats))


if __name__ == "__main__":
    main()
