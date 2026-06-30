from __future__ import annotations

import cv2
import numpy as np

from .schemas import RecognitionResult


def draw_detections(
    image_rgb: np.ndarray,
    result: RecognitionResult,
) -> np.ndarray:
    canvas = image_rgb.copy()
    for detection in result.detections:
        x1, y1, x2, y2 = detection.bbox
        color = (34, 197, 94) if detection.status == "identified" else (245, 158, 11)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

        top_candidate = (
            detection.aihub_candidates[0]
            if detection.aihub_candidates
            else detection.cnn_candidates[0]
            if detection.cnn_candidates
            else detection.rtmdet_candidates[0]
        )
        label = f"#{detection.pill_id} {top_candidate.class_name} {top_candidate.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas
