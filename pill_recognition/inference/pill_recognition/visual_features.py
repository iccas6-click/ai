from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropVisualFeatures:
    color: str | None = None
    shape: str | None = None


def estimate_crop_visual_features(crop_rgb: np.ndarray) -> CropVisualFeatures:
    if not isinstance(crop_rgb, np.ndarray) or crop_rgb.ndim != 3 or crop_rgb.shape[2] < 3:
        return CropVisualFeatures()

    crop = np.asarray(crop_rgb[:, :, :3], dtype=np.uint8)
    height, width = crop.shape[:2]
    if height < 4 or width < 4:
        return CropVisualFeatures()

    color = estimate_dominant_color(crop)
    shape = estimate_shape(width, height)
    return CropVisualFeatures(color=color, shape=shape)


def estimate_dominant_color(crop_rgb: np.ndarray) -> str | None:
    pixels = crop_rgb[_central_ellipse_mask(crop_rgb.shape[0], crop_rgb.shape[1])]
    if pixels.size == 0:
        pixels = crop_rgb.reshape(-1, 3)

    # Median is robust against small imprints, reflections, and background corners.
    red, green, blue = np.median(pixels.astype(np.float32), axis=0) / 255.0
    hue, saturation, value = colorsys.rgb_to_hsv(float(red), float(green), float(blue))
    hue_deg = hue * 360.0

    if value < 0.22:
        return "검정"
    if saturation < 0.13:
        if value > 0.72:
            return "하양"
        return "회색"
    if 15 <= hue_deg < 50 and value < 0.58:
        return "갈색"
    if hue_deg < 12 or hue_deg >= 345:
        return "빨강" if saturation > 0.45 else "분홍"
    if 12 <= hue_deg < 28:
        return "주황"
    if 28 <= hue_deg < 68:
        return "노랑"
    if 68 <= hue_deg < 165:
        return "초록"
    if 165 <= hue_deg < 255:
        return "파랑"
    if 255 <= hue_deg < 320:
        return "보라"
    return "분홍"


def estimate_shape(width: int, height: int) -> str | None:
    if width <= 0 or height <= 0:
        return None
    ratio = max(width / height, height / width)
    if ratio < 1.18:
        return "원형"
    if ratio < 1.75:
        return "타원형"
    return "장방형"


def _central_ellipse_mask(height: int, width: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    ry = max(height * 0.42, 1.0)
    rx = max(width * 0.42, 1.0)
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
