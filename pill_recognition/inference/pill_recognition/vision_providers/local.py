from __future__ import annotations

import numpy as np

from ..schemas import VisionObservation


class LocalVisionProvider:
    name = "local-heuristic"

    def inspect_crop(self, crop_rgb: np.ndarray) -> VisionObservation:
        color = estimate_color(crop_rgb)
        return VisionObservation(
            color=color,
            confidence=0.2 if color else 0.0,
            notes="local provider does not identify product/ingredient; configure Gemini.",
            raw={"provider": self.name},
        )

    def inspect_crops(self, crops_rgb: list[np.ndarray]) -> list[VisionObservation]:
        return [self.inspect_crop(crop_rgb) for crop_rgb in crops_rgb]


def estimate_color(crop_rgb: np.ndarray) -> str | None:
    if crop_rgb.size == 0:
        return None
    pixels = crop_rgb.reshape(-1, crop_rgb.shape[-1]).astype(np.float32)
    median = np.median(pixels, axis=0)
    red, green, blue = median.tolist()
    brightness = float(np.mean(median))
    saturation = float(max(median) - min(median))

    if brightness > 210 and saturation < 35:
        return "하양"
    if brightness < 55 and saturation < 35:
        return "검정"
    if saturation < 25:
        return "회색"
    if red > green + 35 and red > blue + 35:
        return "빨강" if green < 110 else "주황"
    if green > red + 25 and green > blue + 25:
        return "초록"
    if blue > red + 25 and blue > green + 20:
        return "파랑"
    if red > 150 and blue > 130 and green < 150:
        return "보라"
    if red > 180 and green > 150 and blue < 130:
        return "노랑"
    return None
