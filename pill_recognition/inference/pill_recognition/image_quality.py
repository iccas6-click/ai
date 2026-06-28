from __future__ import annotations

import cv2
import numpy as np


def assess_image_quality(image_rgb: np.ndarray, context: str = "image") -> list[str]:
    """Return capture-quality warnings that should trigger app-side retake UX."""
    if not isinstance(image_rgb, np.ndarray):
        return [f"{context}: invalid image array."]

    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[2] < 3 or array.size == 0:
        return [f"{context}: invalid RGB image."]

    image = np.clip(array[:, :, :3], 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    warnings: list[str] = []

    if _is_low_resolution(width, height, context):
        warnings.append(
            f"{context}: resolution is low ({width}x{height}); retake closer or upload a larger crop."
        )

    luma = (
        0.299 * image[:, :, 0].astype(np.float32)
        + 0.587 * image[:, :, 1].astype(np.float32)
        + 0.114 * image[:, :, 2].astype(np.float32)
    )
    mean_luma = float(luma.mean())
    std_luma = float(luma.std())

    if mean_luma < 40.0:
        warnings.append(f"{context}: image is too dark; retake with more light.")
    elif mean_luma > 242.0:
        warnings.append(f"{context}: image is overexposed; reduce glare and retake.")

    if std_luma < 10.0:
        warnings.append(
            f"{context}: contrast is low; use a plain background and clearer lighting."
        )

    if _looks_blurry(luma, std_luma, context):
        warnings.append(f"{context}: image appears blurry; hold the camera steady and retake.")

    return warnings


def _is_low_resolution(width: int, height: int, context: str) -> bool:
    min_side = min(width, height)
    max_side = max(width, height)
    if context.startswith("crop"):
        return min_side < 64 or max_side < 96
    return min_side < 256 or max_side < 512


def _looks_blurry(luma: np.ndarray, std_luma: float, context: str) -> bool:
    if min(luma.shape[:2]) < 32 or std_luma < 12.0:
        return False
    gray = np.clip(luma, 0, 255).astype(np.uint8)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    threshold = 18.0 if context.startswith("crop") else 12.0
    return variance < threshold
