from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit generated AIHub synthetic RTMDet scenes before training."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Synthetic dataset root with images/, labels/, metadata/.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--preview-output", type=Path, default=None)
    parser.add_argument("--max-preview-images", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_metadata_rows(args.dataset_root, args.split, args.limit)
    report = audit_metadata_rows(rows)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.preview_output:
        render_preview_grid(
            dataset_root=args.dataset_root,
            split=args.split,
            rows=rows,
            output=args.preview_output,
            max_images=args.max_preview_images,
        )


def load_metadata_rows(
    dataset_root: Path,
    split: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    metadata_dir = dataset_root / "metadata" / split
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory does not exist: {metadata_dir}")
    paths = sorted(path for path in metadata_dir.glob("*.json") if path.is_file())
    if limit is not None:
        paths = paths[:limit]
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_metadata_path"] = str(path)
        payload["_stem"] = path.stem
        rows.append(payload)
    return rows


def audit_metadata_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    requested_counts = Counter()
    placed_counts = Counter()
    count_mismatches = []
    empty_images = []
    high_attempt_images = []
    mask_quality_rows = []
    low_quality_masks = []

    for row in rows:
        requested = int(row.get("requested_pill_count", row.get("pill_count", 0)))
        placed = int(row.get("pill_count", len(row.get("pills", []))))
        attempts = int(row.get("placement_attempts", placed))
        requested_counts[requested] += 1
        placed_counts[placed] += 1
        if requested != placed:
            count_mismatches.append(
                {
                    "image": row.get("_stem"),
                    "requested": requested,
                    "placed": placed,
                    "placement_attempts": attempts,
                }
            )
        if placed == 0:
            empty_images.append({"image": row.get("_stem"), "requested": requested})
        if requested and attempts > requested * 4:
            high_attempt_images.append(
                {
                    "image": row.get("_stem"),
                    "requested": requested,
                    "placed": placed,
                    "placement_attempts": attempts,
                }
            )

        for pill in row.get("pills", []):
            quality = pill.get("mask_quality") or {}
            if not quality:
                continue
            quality_row = {
                "image": row.get("_stem"),
                "pill_id": pill.get("pill_id"),
                "class_name": pill.get("class_name"),
                "area_ratio": float(quality.get("area_ratio", 0.0)),
                "bbox_fill_ratio": float(quality.get("bbox_fill_ratio", 0.0)),
                "soft_edge_ratio": float(quality.get("soft_edge_ratio", 0.0)),
            }
            mask_quality_rows.append(quality_row)
            if (
                quality_row["bbox_fill_ratio"] < 0.28
                or quality_row["soft_edge_ratio"] < 0.005
            ):
                low_quality_masks.append(quality_row)

    image_count = len(rows)
    placed_total = sum(placed * count for placed, count in placed_counts.items())
    requested_total = sum(
        requested * count for requested, count in requested_counts.items()
    )
    summary = {
        "images": image_count,
        "requested_pills": requested_total,
        "placed_pills": placed_total,
        "count_match_rate": round(
            (image_count - len(count_mismatches)) / image_count,
            6,
        )
        if image_count
        else 0.0,
        "requested_count_distribution": dict(sorted(requested_counts.items())),
        "placed_count_distribution": dict(sorted(placed_counts.items())),
        "mask_quality": summarize_mask_quality(mask_quality_rows),
        "warnings": {
            "count_mismatch_images": len(count_mismatches),
            "empty_images": len(empty_images),
            "high_attempt_images": len(high_attempt_images),
            "low_quality_masks": len(low_quality_masks),
        },
    }
    return {
        "summary": summary,
        "count_mismatches": count_mismatches[:50],
        "empty_images": empty_images[:50],
        "high_attempt_images": high_attempt_images[:50],
        "low_quality_masks": low_quality_masks[:50],
    }


def summarize_mask_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "pills": 0,
            "area_ratio": {},
            "bbox_fill_ratio": {},
            "soft_edge_ratio": {},
        }
    return {
        "pills": len(rows),
        "area_ratio": summarize_values([row["area_ratio"] for row in rows]),
        "bbox_fill_ratio": summarize_values(
            [row["bbox_fill_ratio"] for row in rows]
        ),
        "soft_edge_ratio": summarize_values(
            [row["soft_edge_ratio"] for row in rows]
        ),
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 4),
        "mean": round(mean(values), 4),
        "p50": round(percentile(values, 50), 4),
        "p95": round(percentile(values, 95), 4),
        "max": round(max(values), 4),
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def render_preview_grid(
    dataset_root: Path,
    split: str,
    rows: list[dict[str, Any]],
    output: Path,
    max_images: int = 16,
    thumb_size: int = 320,
) -> None:
    tiles = []
    for row in rows[:max_images]:
        image_path = resolve_image_path(dataset_root, split, str(row["_stem"]))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = draw_scene_boxes(image, row)
        tile = resize_to_tile(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), thumb_size)
        tiles.append(tile)
    if not tiles:
        raise RuntimeError("No preview tiles were rendered.")
    grid = make_grid(tiles, columns=max(1, int(np.ceil(np.sqrt(len(tiles))))))
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output)


def resolve_image_path(dataset_root: Path, split: str, stem: str) -> Path:
    image_dir = dataset_root / "images" / split
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for metadata stem: {stem}")


def draw_scene_boxes(image_bgr: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    canvas = image_bgr.copy()
    for pill in row.get("pills", []):
        bbox = pill.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 210, 90), 3)
        label = f"{pill.get('pill_id', '?')} {pill.get('class_name', '')}"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 210, 90),
            2,
            cv2.LINE_AA,
        )
    return canvas


def resize_to_tile(image_rgb: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    scale = min(tile_size / width, tile_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    tile = np.full((tile_size, tile_size, 3), 245, dtype=np.uint8)
    offset_x = (tile_size - new_width) // 2
    offset_y = (tile_size - new_height) // 2
    tile[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
    return tile


def make_grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    tile_size = tiles[0].shape[0]
    rows = int(np.ceil(len(tiles) / columns))
    grid = np.full((rows * tile_size, columns * tile_size, 3), 235, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        y1 = row * tile_size
        x1 = column * tile_size
        grid[y1:y1 + tile_size, x1:x1 + tile_size] = tile
    return grid


if __name__ == "__main__":
    main()
