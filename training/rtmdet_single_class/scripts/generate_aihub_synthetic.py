from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AIHUB_ROOT = (
    PROJECT_ROOT
    / "inference"
    / "aihub_official_code"
    / "package"
    / "평가용 데이터셋"
    / "pill_data"
    / "pill_data_croped"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "processed" / "rtmdet-aihub-synthetic-max10"


@dataclass(frozen=True)
class ClassAssets:
    class_name: str
    image_paths: list[Path]
    product_name: str | None = None
    company: str | None = None
    item_seq: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-pill RTMDet training scenes from AI Hub cropped pills."
    )
    parser.add_argument("--aihub-root", type=Path, default=DEFAULT_AIHUB_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-count", type=int, default=8000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--min-pills", type=int, default=1)
    parser.add_argument("--max-pills", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists. Use --overwrite: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    class_index = build_asset_index(args.aihub_root)
    if not class_index:
        raise FileNotFoundError(f"No AI Hub K-* assets found under {args.aihub_root}")

    manifest = {
        "source": str(args.aihub_root),
        "image_size": args.image_size,
        "min_pills": args.min_pills,
        "max_pills": args.max_pills,
        "seed": args.seed,
        "class_count": len(class_index),
        "splits": {},
    }
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        split_manifest = generate_split(
            output=args.output,
            split=split,
            count=count,
            class_index=class_index,
            image_size=args.image_size,
            min_pills=args.min_pills,
            max_pills=args.max_pills,
            rng=rng,
            jpeg_quality=args.jpeg_quality,
        )
        manifest["splits"][split] = split_manifest

    (args.output / "pill.yaml").write_text("names:\n  0: pill\n", encoding="utf-8")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2), flush=True)


def build_asset_index(aihub_root: Path) -> dict[str, ClassAssets]:
    class_index: dict[str, ClassAssets] = {}
    for class_dir in sorted(aihub_root.glob("K-*")):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        metadata = load_product_metadata(class_dir)
        images = sorted(class_dir.glob("*.png"))
        if not images:
            continue
        class_index[class_name] = ClassAssets(
            class_name=class_name,
            image_paths=images,
            product_name=metadata.get("dl_name"),
            company=metadata.get("dl_company"),
            item_seq=metadata.get("item_seq"),
        )
    return class_index


def load_product_metadata(class_dir: Path) -> dict:
    sample_json = next(class_dir.glob("*.json"), None)
    if sample_json is None:
        return {}
    try:
        payload = json.loads(sample_json.read_text(encoding="utf-8"))
        rows = payload.get("images", [])
        return rows[0] if rows else {}
    except (OSError, json.JSONDecodeError):
        return {}


def generate_split(
    output: Path,
    split: str,
    count: int,
    class_index: dict[str, ClassAssets],
    image_size: int,
    min_pills: int,
    max_pills: int,
    rng: random.Random,
    jpeg_quality: int,
) -> dict:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    metadata_dir = output / "metadata" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    class_names = list(class_index)
    count_distribution: dict[int, int] = {}

    for image_index in range(count):
        pill_count = rng.randint(min_pills, max_pills)
        scene = synthesize_scene(
            class_index=class_index,
            class_names=class_names,
            image_size=image_size,
            pill_count=pill_count,
            rng=rng,
        )
        image_name = f"aihub_synth_{split}_{image_index:06d}.jpg"
        label_name = f"aihub_synth_{split}_{image_index:06d}.txt"
        metadata_name = f"aihub_synth_{split}_{image_index:06d}.json"
        cv2.imwrite(
            str(image_dir / image_name),
            scene["image"],
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        (label_dir / label_name).write_text(
            "\n".join(scene["yolo_labels"]) + "\n",
            encoding="utf-8",
        )
        (metadata_dir / metadata_name).write_text(
            json.dumps(scene["metadata"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        image_id = image_index + 1
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": image_size,
                "height": image_size,
            }
        )
        for pill in scene["metadata"]["pills"]:
            x1, y1, x2, y2 = pill["bbox_xyxy"]
            width = x2 - x1
            height = y2 - y1
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        count_distribution[pill_count] = count_distribution.get(pill_count, 0) + 1

        if (image_index + 1) % 500 == 0:
            print(f"{split}: generated {image_index + 1}/{count}", flush=True)

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "pill", "supercategory": "pill"}],
    }
    (output / f"{split}_coco.json").write_text(
        json.dumps(coco, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "images": count,
        "annotations": len(coco_annotations),
        "count_distribution": dict(sorted(count_distribution.items())),
    }


def synthesize_scene(
    class_index: dict[str, ClassAssets],
    class_names: list[str],
    image_size: int,
    pill_count: int,
    rng: random.Random,
) -> dict:
    canvas = random_background(image_size, rng)
    placed_boxes: list[tuple[int, int, int, int]] = []
    yolo_labels = []
    pills = []

    selected_classes = rng.sample(class_names, k=min(pill_count, len(class_names)))
    while len(selected_classes) < pill_count:
        selected_classes.append(rng.choice(class_names))

    for pill_index, class_name in enumerate(selected_classes, start=1):
        asset = class_index[class_name]
        source_image = rng.choice(asset.image_paths)
        patch, alpha = load_pill_patch(source_image, rng)
        if patch is None or alpha is None:
            continue
        box, pasted = paste_patch(canvas, patch, alpha, placed_boxes, rng)
        if box is None:
            continue
        placed_boxes.append(box)
        yolo_labels.append(format_yolo_label(box, image_size, image_size))
        pills.append(
            {
                "pill_id": pill_index,
                "class_name": asset.class_name,
                "product_name": asset.product_name,
                "company": asset.company,
                "item_seq": asset.item_seq,
                "source_image": str(source_image),
                "bbox_xyxy": list(box),
                "bbox_yolo": yolo_labels[-1],
                "placed": pasted,
            }
        )

    return {
        "image": canvas,
        "yolo_labels": yolo_labels,
        "metadata": {
            "image_width": image_size,
            "image_height": image_size,
            "pill_count": len(pills),
            "pills": pills,
        },
    }


def random_background(image_size: int, rng: random.Random) -> np.ndarray:
    base_color = np.array(
        [rng.randint(80, 230), rng.randint(80, 230), rng.randint(80, 230)],
        dtype=np.float32,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas[:] = base_color

    y_gradient = np.linspace(-rng.randint(10, 35), rng.randint(10, 35), image_size)
    x_gradient = np.linspace(-rng.randint(10, 35), rng.randint(10, 35), image_size)
    canvas += y_gradient[:, None, None]
    canvas += x_gradient[None, :, None]

    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(
        0,
        rng.uniform(3.0, 12.0),
        canvas.shape,
    )
    canvas += noise

    if rng.random() < 0.35:
        spacing = rng.randint(24, 80)
        line_color = tuple(
            int(value)
            for value in np.clip(
                base_color + rng.choice([-1, 1]) * rng.randint(10, 25),
                0,
                255,
            )
        )
        for offset in range(0, image_size, spacing):
            cv2.line(canvas, (offset, 0), (offset, image_size), line_color, 1)
            cv2.line(canvas, (0, offset), (image_size, offset), line_color, 1)

    canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.3, 1.2))
    return np.clip(canvas, 0, 255).astype(np.uint8)


def load_pill_patch(
    image_path: Path,
    rng: random.Random,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    mask = extract_foreground_mask(image)
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None
    x1, y1, x2, y2 = bbox
    patch = image[y1:y2, x1:x2]
    alpha = mask[y1:y2, x1:x2]
    patch, alpha = augment_patch(patch, alpha, rng)
    return patch, alpha


def extract_foreground_mask(image: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [
            image[:8, :, :].reshape(-1, 3),
            image[-8:, :, :].reshape(-1, 3),
            image[:, :8, :].reshape(-1, 3),
            image[:, -8:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    mask = (distance > 18).astype(np.uint8) * 255
    if mask.mean() < 8 or mask.mean() > 230:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mask.mean() > 127:
            mask = 255 - mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [largest], -1, 255, thickness=-1)
    return cv2.GaussianBlur(cleaned, (5, 5), 0)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 16)
    if not len(xs) or not len(ys):
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return x1, y1, x2, y2


def augment_patch(
    patch: np.ndarray,
    alpha: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    scale = rng.uniform(0.55, 1.25)
    patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_AREA)

    angle = rng.uniform(-180, 180)
    patch, alpha = rotate_patch(patch, alpha, angle)

    brightness = rng.uniform(0.82, 1.18)
    contrast = rng.uniform(0.88, 1.12)
    patch = np.clip((patch.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255)
    patch = np.clip(patch * brightness, 0, 255).astype(np.uint8)
    return patch, alpha


def rotate_patch(
    patch: np.ndarray,
    alpha: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = patch.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]
    rotated_patch = cv2.warpAffine(
        patch,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rotated_alpha = cv2.warpAffine(
        alpha,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    bbox = mask_bbox(rotated_alpha)
    if bbox is None:
        return rotated_patch, rotated_alpha
    x1, y1, x2, y2 = bbox
    return rotated_patch[y1:y2, x1:x2], rotated_alpha[y1:y2, x1:x2]


def paste_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    alpha: np.ndarray,
    placed_boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int, int, int] | None, dict | None]:
    canvas_height, canvas_width = canvas.shape[:2]
    patch_height, patch_width = patch.shape[:2]
    if patch_width >= canvas_width or patch_height >= canvas_height:
        return None, None

    for _ in range(100):
        x1 = rng.randint(0, canvas_width - patch_width)
        y1 = rng.randint(0, canvas_height - patch_height)
        x2 = x1 + patch_width
        y2 = y1 + patch_height
        box = (x1, y1, x2, y2)
        if all(box_iou(box, previous) < 0.08 for previous in placed_boxes):
            break
    else:
        return None, None

    shadow_offset = rng.randint(3, 9)
    shadow_alpha = cv2.GaussianBlur(alpha, (0, 0), rng.uniform(2.0, 5.0))
    apply_shadow(canvas, shadow_alpha, x1 + shadow_offset, y1 + shadow_offset)

    region = canvas[y1:y2, x1:x2]
    blend_alpha = (alpha.astype(np.float32) / 255.0)[:, :, None]
    canvas[y1:y2, x1:x2] = (
        patch.astype(np.float32) * blend_alpha
        + region.astype(np.float32) * (1.0 - blend_alpha)
    ).astype(np.uint8)
    return box, {"x": x1, "y": y1, "width": patch_width, "height": patch_height}


def apply_shadow(canvas: np.ndarray, alpha: np.ndarray, x1: int, y1: int) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    patch_height, patch_width = alpha.shape[:2]
    x2 = min(canvas_width, x1 + patch_width)
    y2 = min(canvas_height, y1 + patch_height)
    if x1 >= canvas_width or y1 >= canvas_height or x2 <= 0 or y2 <= 0:
        return
    source_x1 = max(0, -x1)
    source_y1 = max(0, -y1)
    target_x1 = max(0, x1)
    target_y1 = max(0, y1)
    target_x2 = x2
    target_y2 = y2
    shadow = alpha[source_y1:source_y1 + target_y2 - target_y1, source_x1:source_x1 + target_x2 - target_x1]
    factor = 1.0 - (shadow.astype(np.float32) / 255.0 * 0.22)[:, :, None]
    canvas[target_y1:target_y2, target_x1:target_x2] = (
        canvas[target_y1:target_y2, target_x1:target_x2].astype(np.float32) * factor
    ).astype(np.uint8)


def format_yolo_label(box: tuple[int, int, int, int], image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2) / image_width
    center_y = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"


def box_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


if __name__ == "__main__":
    main()
