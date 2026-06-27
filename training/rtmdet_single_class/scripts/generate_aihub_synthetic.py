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
    parser.add_argument(
        "--background-mode",
        choices=("realistic", "simple"),
        default="realistic",
    )
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
        "background_mode": args.background_mode,
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
            background_mode=args.background_mode,
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
            product_name=clean_text(metadata.get("dl_name")),
            company=clean_text(metadata.get("dl_company")),
            item_seq=clean_text(metadata.get("item_seq")),
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


def clean_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def generate_split(
    output: Path,
    split: str,
    count: int,
    class_index: dict[str, ClassAssets],
    image_size: int,
    min_pills: int,
    max_pills: int,
    background_mode: str,
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
    placed_count_distribution: dict[int, int] = {}

    for image_index in range(count):
        pill_count = rng.randint(min_pills, max_pills)
        scene = synthesize_scene(
            class_index=class_index,
            class_names=class_names,
            image_size=image_size,
            pill_count=pill_count,
            background_mode=background_mode,
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
        actual_count = scene["metadata"]["pill_count"]
        placed_count_distribution[actual_count] = (
            placed_count_distribution.get(actual_count, 0) + 1
        )

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
        "requested_count_distribution": dict(sorted(count_distribution.items())),
        "placed_count_distribution": dict(sorted(placed_count_distribution.items())),
    }


def synthesize_scene(
    class_index: dict[str, ClassAssets],
    class_names: list[str],
    image_size: int,
    pill_count: int,
    background_mode: str,
    rng: random.Random,
) -> dict:
    canvas, background_name = random_background(image_size, rng, background_mode)
    placed_boxes: list[tuple[int, int, int, int]] = []
    yolo_labels = []
    pills = []

    selected_classes = rng.sample(class_names, k=min(pill_count, len(class_names)))
    while len(selected_classes) < pill_count:
        selected_classes.append(rng.choice(class_names))

    max_attempts = max(40, pill_count * 12)
    attempts = 0
    while len(pills) < pill_count and attempts < max_attempts:
        attempts += 1
        class_name = (
            selected_classes[attempts - 1]
            if attempts <= len(selected_classes)
            else rng.choice(class_names)
        )
        asset = class_index[class_name]
        source_image = rng.choice(asset.image_paths)
        patch, alpha, mask_quality = load_pill_patch(source_image, rng)
        if patch is None or alpha is None or mask_quality is None:
            continue
        box, pasted = paste_patch(canvas, patch, alpha, placed_boxes, rng)
        if box is None:
            continue
        placed_boxes.append(box)
        yolo_labels.append(format_yolo_label(box, image_size, image_size))
        pill_index = len(pills) + 1
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
                "mask_quality": mask_quality,
                "placed": pasted,
            }
        )

    return {
        "image": canvas,
        "yolo_labels": yolo_labels,
        "metadata": {
            "image_width": image_size,
            "image_height": image_size,
            "background": background_name,
            "requested_pill_count": pill_count,
            "pill_count": len(pills),
            "placement_attempts": attempts,
            "pills": pills,
        },
    }


def random_background(
    image_size: int,
    rng: random.Random,
    mode: str = "realistic",
) -> tuple[np.ndarray, str]:
    if mode == "simple":
        return simple_background(image_size, rng), "simple"
    style = rng.choice(("wood", "paper", "marble", "fabric", "skin", "countertop"))
    generators = {
        "wood": wood_background,
        "paper": paper_background,
        "marble": marble_background,
        "fabric": fabric_background,
        "skin": skin_background,
        "countertop": countertop_background,
    }
    canvas = generators[style](image_size, rng)
    return apply_photo_lighting(canvas, rng), style


def simple_background(image_size: int, rng: random.Random) -> np.ndarray:
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


def wood_background(image_size: int, rng: random.Random) -> np.ndarray:
    base = np.array(
        [rng.randint(92, 138), rng.randint(125, 175), rng.randint(165, 215)],
        dtype=np.float32,
    )
    y = np.linspace(0, 1, image_size)[:, None]
    x = np.linspace(0, 1, image_size)[None, :]
    grain = (
        np.sin((x * rng.uniform(18, 34) + rng.random()) * math.pi * 2) * 12
        + np.sin((x * rng.uniform(42, 80) + y * rng.uniform(1, 4)) * math.pi * 2) * 5
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas[:] = base
    canvas += grain[:, :, None]
    for _ in range(rng.randint(3, 8)):
        center = (
            rng.randint(-image_size // 3, image_size + image_size // 3),
            rng.randint(0, image_size),
        )
        radius = rng.randint(80, 220)
        color = tuple(int(v) for v in np.clip(base - rng.randint(8, 22), 0, 255))
        cv2.ellipse(canvas, center, (radius, max(10, radius // 8)), 0, 0, 360, color, 2)
    return finalize_surface(canvas, rng, noise=7.0, blur=0.6)


def paper_background(image_size: int, rng: random.Random) -> np.ndarray:
    tone = rng.randint(218, 246)
    base = np.array(
        [tone - rng.randint(0, 8), tone, tone + rng.randint(0, 5)],
        dtype=np.float32,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas[:] = base
    for _ in range(260):
        x1 = rng.randint(0, image_size)
        y1 = rng.randint(0, image_size)
        length = rng.randint(12, 80)
        angle = rng.uniform(-0.4, 0.4)
        x2 = int(x1 + math.cos(angle) * length)
        y2 = int(y1 + math.sin(angle) * length)
        color = tuple(int(v) for v in np.clip(base - rng.randint(8, 22), 0, 255))
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1)
    return finalize_surface(canvas, rng, noise=5.0, blur=0.4)


def marble_background(image_size: int, rng: random.Random) -> np.ndarray:
    base = np.array(
        [rng.randint(188, 226), rng.randint(190, 228), rng.randint(190, 232)],
        dtype=np.float32,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas[:] = base
    for _ in range(rng.randint(14, 30)):
        points = []
        x = rng.randint(-100, image_size + 100)
        for y in range(-80, image_size + 120, rng.randint(60, 120)):
            x += rng.randint(-80, 80)
            points.append((x, y))
        color = tuple(int(v) for v in np.clip(base - rng.randint(20, 55), 0, 255))
        for p1, p2 in zip(points, points[1:]):
            cv2.line(canvas, p1, p2, color, rng.choice((1, 1, 2, 3)))
    return finalize_surface(canvas, rng, noise=6.0, blur=1.2)


def fabric_background(image_size: int, rng: random.Random) -> np.ndarray:
    base = np.array(
        [rng.randint(95, 195), rng.randint(105, 205), rng.randint(105, 215)],
        dtype=np.float32,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas[:] = base
    spacing = rng.randint(8, 18)
    warp_color = tuple(int(v) for v in np.clip(base + rng.randint(8, 20), 0, 255))
    weft_color = tuple(int(v) for v in np.clip(base - rng.randint(8, 20), 0, 255))
    for offset in range(0, image_size, spacing):
        cv2.line(canvas, (offset, 0), (offset, image_size), warp_color, 1)
        cv2.line(canvas, (0, offset), (image_size, offset), weft_color, 1)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.8)
    return finalize_surface(canvas, rng, noise=9.0, blur=0.5)


def skin_background(image_size: int, rng: random.Random) -> np.ndarray:
    base = np.array(
        [rng.randint(120, 178), rng.randint(155, 205), rng.randint(185, 235)],
        dtype=np.float32,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas[:] = base
    center = (
        rng.randint(image_size // 4, image_size * 3 // 4),
        image_size + rng.randint(40, 180),
    )
    for radius in range(rng.randint(180, 240), image_size + 360, rng.randint(44, 72)):
        color = tuple(int(v) for v in np.clip(base - rng.randint(8, 28), 0, 255))
        cv2.ellipse(canvas, center, (radius, max(24, radius // 5)), 0, 190, 350, color, 1)
    for _ in range(rng.randint(12, 24)):
        x1 = rng.randint(0, image_size)
        y1 = rng.randint(0, image_size)
        x2 = min(image_size, max(0, x1 + rng.randint(-180, 180)))
        y2 = min(image_size, max(0, y1 + rng.randint(-80, 80)))
        color = tuple(int(v) for v in np.clip(base - rng.randint(5, 18), 0, 255))
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1)
    return finalize_surface(canvas, rng, noise=7.0, blur=1.0)


def countertop_background(image_size: int, rng: random.Random) -> np.ndarray:
    base = np.array(
        [rng.randint(120, 205), rng.randint(125, 210), rng.randint(125, 210)],
        dtype=np.float32,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    canvas[:] = base
    speckle = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(
        0,
        28,
        (image_size, image_size),
    )
    speckle = cv2.GaussianBlur(speckle.astype(np.float32), (0, 0), rng.uniform(1.5, 3.0))
    canvas += speckle[:, :, None]
    for _ in range(rng.randint(40, 90)):
        color = tuple(int(v) for v in np.clip(base + rng.randint(-45, 45), 0, 255))
        cv2.circle(
            canvas,
            (rng.randint(0, image_size), rng.randint(0, image_size)),
            rng.randint(1, 4),
            color,
            -1,
        )
    return finalize_surface(canvas, rng, noise=4.0, blur=0.6)


def finalize_surface(
    canvas: np.ndarray,
    rng: random.Random,
    noise: float,
    blur: float,
) -> np.ndarray:
    random_noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(
        0,
        noise,
        canvas.shape,
    )
    canvas += random_noise
    if blur > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def apply_photo_lighting(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = canvas.shape[:2]
    y, x = np.ogrid[:height, :width]
    center_x = rng.uniform(width * 0.25, width * 0.75)
    center_y = rng.uniform(height * 0.15, height * 0.55)
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    distance = distance / distance.max()
    vignette = 1.05 - distance * rng.uniform(0.18, 0.38)
    canvas = np.clip(canvas.astype(np.float32) * vignette[:, :, None], 0, 255)
    if rng.random() < 0.45:
        stripe_x = rng.randint(-width // 4, width)
        sigma = width * rng.uniform(0.12, 0.22)
        stripe = np.exp(-((x - stripe_x) ** 2) / (2 * sigma**2))
        canvas += stripe[:, :, None] * rng.uniform(8, 28)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def load_pill_patch(
    image_path: Path,
    rng: random.Random,
) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, None, None
    mask = extract_foreground_mask(image)
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None, None
    x1, y1, x2, y2 = bbox
    patch = image[y1:y2, x1:x2]
    alpha = mask[y1:y2, x1:x2]
    patch, alpha = augment_patch(patch, alpha, rng)
    alpha = feather_alpha(alpha, radius=rng.uniform(2.5, 5.5))
    quality = compute_mask_quality(alpha)
    if not is_usable_mask_quality(quality):
        return None, None, None
    return patch, alpha, quality


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
    mask = (distance > 34).astype(np.uint8) * 255
    if mask.mean() < 5:
        mask = (distance > 22).astype(np.uint8) * 255
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
    center = np.array([image.shape[1] / 2, image.shape[0] / 2])
    largest = max(
        contours,
        key=lambda contour: cv2.contourArea(contour)
        - np.linalg.norm(contour.reshape(-1, 2).mean(axis=0) - center) * 2.0,
    )
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [largest], -1, 255, thickness=-1)
    cleaned = cv2.erode(cleaned, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
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


def feather_alpha(alpha: np.ndarray, radius: float = 4.0) -> np.ndarray:
    binary = (alpha > 24).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 3)
    signed_distance = inside - outside
    soft = np.clip((signed_distance + radius) / (2.0 * radius), 0.0, 1.0)
    return (soft * 255).astype(np.uint8)


def compute_mask_quality(alpha: np.ndarray) -> dict:
    foreground = alpha > 32
    bbox = mask_bbox(alpha)
    foreground_area = int(foreground.sum())
    total_area = int(alpha.shape[0] * alpha.shape[1])
    if bbox is None:
        return {
            "area_ratio": 0.0,
            "bbox_fill_ratio": 0.0,
            "soft_edge_ratio": 0.0,
        }
    x1, y1, x2, y2 = bbox
    bbox_area = max(1, int((x2 - x1) * (y2 - y1)))
    soft_edges = np.logical_and(alpha > 0, alpha < 255)
    return {
        "area_ratio": round(foreground_area / max(1, total_area), 4),
        "bbox_fill_ratio": round(foreground_area / bbox_area, 4),
        "soft_edge_ratio": round(int(soft_edges.sum()) / max(1, total_area), 4),
    }


def is_usable_mask_quality(quality: dict) -> bool:
    return (
        0.08 <= float(quality["area_ratio"]) <= 0.96
        and float(quality["bbox_fill_ratio"]) >= 0.28
    )


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
    apply_shadow(
        canvas,
        shadow_alpha,
        x1 + shadow_offset,
        y1 + shadow_offset,
        strength=rng.uniform(0.14, 0.28),
    )

    region = canvas[y1:y2, x1:x2]
    patch = match_patch_to_background(patch, alpha, region, rng)
    blend_alpha = (alpha.astype(np.float32) / 255.0)[:, :, None]
    canvas[y1:y2, x1:x2] = (
        patch.astype(np.float32) * blend_alpha
        + region.astype(np.float32) * (1.0 - blend_alpha)
    ).astype(np.uint8)
    return box, {
        "x": x1,
        "y": y1,
        "width": patch_width,
        "height": patch_height,
        "shadow_offset": shadow_offset,
        "alpha_feathered": True,
    }


def apply_shadow(
    canvas: np.ndarray,
    alpha: np.ndarray,
    x1: int,
    y1: int,
    strength: float = 0.22,
) -> None:
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
    factor = 1.0 - (shadow.astype(np.float32) / 255.0 * strength)[:, :, None]
    canvas[target_y1:target_y2, target_x1:target_x2] = (
        canvas[target_y1:target_y2, target_x1:target_x2].astype(np.float32) * factor
    ).astype(np.uint8)


def match_patch_to_background(
    patch: np.ndarray,
    alpha: np.ndarray,
    region: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    foreground = alpha > 64
    background = alpha < 8
    if foreground.sum() < 16 or background.sum() < 16:
        return patch
    patch_float = patch.astype(np.float32)
    region_float = region.astype(np.float32)
    foreground_mean = patch_float[foreground].mean(axis=0)
    background_mean = region_float[background].mean(axis=0)
    target_luma = float(background_mean.mean())
    foreground_luma = float(foreground_mean.mean())
    luma_delta = np.clip((target_luma - foreground_luma) * rng.uniform(0.04, 0.12), -12, 12)
    color_delta = np.clip((background_mean - foreground_mean) * rng.uniform(0.015, 0.045), -8, 8)
    adjusted = patch_float + luma_delta + color_delta[None, None, :]
    if rng.random() < 0.35:
        adjusted = cv2.GaussianBlur(adjusted, (0, 0), rng.uniform(0.15, 0.45))
    return np.clip(adjusted, 0, 255).astype(np.uint8)


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
