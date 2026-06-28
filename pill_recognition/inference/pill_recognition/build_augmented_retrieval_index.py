from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .retrieval import load_aihub_resnet_encoder, retrieval_transform
from .settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AIHub ResNet retrieval index from synthetic-style reference crops."
    )
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument(
        "--index-mode",
        choices=["prototype", "reference"],
        default="prototype",
        help="prototype averages embeddings per class; reference stores every augmented reference image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    if settings.aihub_weights is None or settings.aihub_mapping is None:
        raise SystemExit("AI Hub weights/mapping are not configured")

    crop_root = settings.aihub_mapping.parent
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or settings.device)
    rng = random.Random(args.seed)

    encoder = load_aihub_resnet_encoder(settings.aihub_weights).to(device).eval()
    transform = retrieval_transform()

    pill_dirs = sorted(
        path for path in crop_root.iterdir() if path.is_dir() and path.name.startswith("K-")
    )
    if args.limit_classes:
        pill_dirs = pill_dirs[: args.limit_classes]

    pill_ids = []
    index_embeddings = []
    manifest = []
    for class_index, pill_dir in enumerate(pill_dirs, start=1):
        image_paths = sorted(pill_dir.glob("*.png"))
        if not image_paths:
            continue
        generated = []
        for sample_index in range(args.samples_per_class):
            image_path = rng.choice(image_paths)
            crop = build_augmented_reference_crop(image_path, rng)
            if crop is None:
                continue
            generated.append(crop)
            manifest.append(
                {
                    "pill_id": pill_dir.name,
                    "source": str(image_path),
                    "sample_index": sample_index,
                }
            )
        if not generated:
            continue

        embeddings = []
        for batch_crops in chunks(generated, args.batch_size):
            batch = torch.stack(
                [transform(Image.fromarray(crop).convert("RGB")) for crop in batch_crops]
            ).to(device)
            with torch.inference_mode():
                features = encoder(batch).flatten(1)
                features = torch.nn.functional.normalize(features, dim=1)
            cpu_features = features.cpu()
            embeddings.append(cpu_features)
            if args.index_mode == "reference":
                index_embeddings.extend(feature for feature in cpu_features)
                pill_ids.extend(pill_dir.name for _ in batch_crops)
        if args.index_mode == "prototype":
            prototype = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
            prototype = torch.nn.functional.normalize(prototype, dim=1)
            pill_ids.append(pill_dir.name)
            index_embeddings.append(prototype.squeeze(0))
        if class_index % 50 == 0:
            print(f"indexed {class_index}/{len(pill_dirs)} classes")

    payload = {
        "version": 1,
        "encoder": "aihub-resnet152-fc-identity-augmented-reference",
        "index_mode": args.index_mode,
        "samples_per_class": args.samples_per_class,
        "pill_ids": pill_ids,
        "embeddings": torch.stack(index_embeddings).half(),
        "reference_count": len(manifest),
        "class_count": len(set(pill_ids)),
    }
    torch.save(payload, output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(
        f"saved {output_path} with {payload['class_count']} classes, "
        f"{len(pill_ids)} index embeddings from {payload['reference_count']} "
        f"augmented reference crops"
    )


def build_augmented_reference_crop(image_path: Path, rng: random.Random) -> np.ndarray | None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask = extract_foreground_mask(image)
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    patch = image[y1:y2, x1:x2]
    alpha = mask[y1:y2, x1:x2]
    patch, alpha = augment_patch(patch, alpha, rng)
    return composite_on_background(patch, alpha, rng)


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
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

    patch, alpha = rotate_patch(patch, alpha, rng.uniform(-180, 180))
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


def composite_on_background(
    patch: np.ndarray,
    alpha: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    background = random_crop_background(patch.shape[1], patch.shape[0], rng)
    shadow_offset = rng.randint(3, 9)
    shadow_alpha = cv2.GaussianBlur(alpha, (0, 0), rng.uniform(2.0, 5.0))
    apply_shadow(background, shadow_alpha, shadow_offset, shadow_offset)
    blend_alpha = (alpha.astype(np.float32) / 255.0)[:, :, None]
    return (
        patch.astype(np.float32) * blend_alpha
        + background.astype(np.float32) * (1.0 - blend_alpha)
    ).astype(np.uint8)


def random_crop_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    mode = rng.choice(["paper", "marble", "fabric", "wood", "gray"])
    base = np.zeros((height, width, 3), dtype=np.uint8)
    if mode == "paper":
        base[:] = np.array([rng.randint(226, 250), rng.randint(226, 250), rng.randint(220, 246)])
    elif mode == "gray":
        value = rng.randint(175, 235)
        base[:] = np.array([value, value, value])
    elif mode == "wood":
        base[:] = np.array([rng.randint(150, 205), rng.randint(112, 165), rng.randint(74, 120)])
        for y in range(0, height, rng.randint(8, 20)):
            base[y : y + 2] = np.clip(base[y : y + 2].astype(int) - rng.randint(8, 22), 0, 255)
    else:
        base[:] = np.array([rng.randint(185, 235), rng.randint(185, 235), rng.randint(185, 235)])
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, 10, base.shape)
        base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if mode == "fabric":
            step = rng.randint(7, 13)
            base[:, ::step] = np.clip(base[:, ::step].astype(int) - 18, 0, 255)
    return cv2.GaussianBlur(base, (3, 3), 0)


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
    shadow = alpha[
        source_y1 : source_y1 + y2 - target_y1,
        source_x1 : source_x1 + x2 - target_x1,
    ]
    factor = 1.0 - (shadow.astype(np.float32) / 255.0 * 0.22)[:, :, None]
    canvas[target_y1:y2, target_x1:x2] = (
        canvas[target_y1:y2, target_x1:x2].astype(np.float32) * factor
    ).astype(np.uint8)


def chunks(items: list[np.ndarray], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


if __name__ == "__main__":
    main()
