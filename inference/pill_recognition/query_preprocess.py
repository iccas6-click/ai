from __future__ import annotations

import cv2
import numpy as np


AIHUB_DARK_BACKGROUND = np.array([28, 8, 19], dtype=np.uint8)


def preprocess_query_crop(crop_rgb: np.ndarray, mode: str = "none") -> np.ndarray:
    if mode in {"", "none", "off", "false", "0"}:
        return crop_rgb
    if mode not in {"foreground_dark", "foreground", "grabcut_dark", "grabcut"}:
        raise ValueError(f"Unsupported retrieval query preprocess mode: {mode}")

    image = _as_rgb_uint8(crop_rgb)
    mask = (
        extract_pill_grabcut_mask(image)
        if mode in {"grabcut", "grabcut_dark"}
        else extract_pill_foreground_mask(image)
    )
    bbox = _mask_bbox(mask)
    if bbox is None:
        return image

    x1, y1, x2, y2 = _expand_bbox(bbox, image.shape[1], image.shape[0], 0.10)
    patch = image[y1:y2, x1:x2]
    alpha = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return image

    if mode in {"foreground", "grabcut"}:
        return _square_pad(patch, _border_median(image))
    return _composite_square(patch, alpha, AIHUB_DARK_BACKGROUND)


def extract_pill_grabcut_mask(image_rgb: np.ndarray) -> np.ndarray:
    image = _as_rgb_uint8(image_rgb)
    height, width = image.shape[:2]
    if height < 16 or width < 16:
        return extract_pill_foreground_mask(image)

    inset_x = max(2, round(width * 0.04))
    inset_y = max(2, round(height * 0.04))
    rect = (
        inset_x,
        inset_y,
        max(1, width - (2 * inset_x)),
        max(1, height - (2 * inset_y)),
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            mask,
            rect,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return extract_pill_foreground_mask(image)

    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    if not 0.03 <= _mask_area_ratio(binary) <= 0.85:
        return extract_pill_foreground_mask(image)
    return _clean_centered_mask(binary)


def extract_pill_foreground_mask(image_rgb: np.ndarray) -> np.ndarray:
    image = _as_rgb_uint8(image_rgb)
    height, width = image.shape[:2]
    border_width = max(4, min(height, width) // 18)
    border = np.concatenate(
        [
            image[:border_width, :, :].reshape(-1, 3),
            image[-border_width:, :, :].reshape(-1, 3),
            image[:, :border_width, :].reshape(-1, 3),
            image[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border.astype(np.float32), axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    threshold = max(22.0, min(48.0, float(np.percentile(distance, 82))))
    mask = (distance > threshold).astype(np.uint8) * 255

    if _mask_area_ratio(mask) < 0.03 or _mask_area_ratio(mask) > 0.85:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu.mean() > 127:
            otsu = 255 - otsu
        if 0.03 <= _mask_area_ratio(otsu) <= 0.85:
            mask = otsu

    kernel_size = max(3, int(round(min(height, width) * 0.025)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return _clean_centered_mask(mask)


def _clean_centered_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    center = np.array([width / 2.0, height / 2.0])
    contour = max(
        contours,
        key=lambda item: cv2.contourArea(item)
        - np.linalg.norm(item.reshape(-1, 2).mean(axis=0) - center) * 1.5,
    )
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [contour], -1, 255, thickness=-1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(cleaned, (5, 5), 0)


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected RGB image array")
    return np.clip(array[:, :, :3], 0, 255).astype(np.uint8)


def _mask_area_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask > 16) / mask.size) if mask.size else 0.0


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 16)
    if not len(xs) or not len(ys):
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    area_ratio = (x2 - x1) * (y2 - y1) / mask.size
    if area_ratio < 0.02:
        return None
    return x1, y1, x2, y2


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = round((x2 - x1) * ratio)
    pad_y = round((y2 - y1) * ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _square_pad(patch: np.ndarray, background: np.ndarray) -> np.ndarray:
    height, width = patch.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[:] = background.astype(np.uint8)
    x = (side - width) // 2
    y = (side - height) // 2
    canvas[y : y + height, x : x + width] = patch
    return canvas


def _composite_square(
    patch: np.ndarray,
    alpha: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    height, width = patch.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[:] = background
    x = (side - width) // 2
    y = (side - height) // 2
    blend_alpha = (alpha.astype(np.float32) / 255.0)[:, :, None]
    region = canvas[y : y + height, x : x + width]
    region[:] = (
        patch.astype(np.float32) * blend_alpha
        + region.astype(np.float32) * (1.0 - blend_alpha)
    ).astype(np.uint8)
    return canvas


def _border_median(image: np.ndarray) -> np.ndarray:
    border_width = max(4, min(image.shape[:2]) // 18)
    border = np.concatenate(
        [
            image[:border_width, :, :].reshape(-1, 3),
            image[-border_width:, :, :].reshape(-1, 3),
            image[:, :border_width, :].reshape(-1, 3),
            image[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border.astype(np.float32), axis=0).astype(np.uint8)
