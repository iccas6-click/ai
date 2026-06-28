from __future__ import annotations

import cv2
import numpy as np

from .schemas import RecognitionResult


def draw_detections(image_rgb: np.ndarray, result: RecognitionResult) -> np.ndarray:
    canvas = np.ascontiguousarray(image_rgb.copy())
    for detection in result.detections:
        x1, y1, x2, y2 = detection.bbox
        label = f"{detection.pill_id}: {top_label(detection)}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (28, 180, 80), 3)
        cv2.putText(
            canvas,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (28, 180, 80),
            2,
            cv2.LINE_AA,
        )
    return canvas


def top_label(detection) -> str:
    if detection.candidates:
        return detection.candidates[0].pill_id
    if detection.vision.imprint_front:
        return detection.vision.imprint_front
    return detection.status
