from __future__ import annotations

import cv2
import numpy as np

from .aihub_classifier import AIHubPillClassifier
from .assets import download_model_assets
from .classifier import EfficientNetPillClassifier
from .detector import RTMDetPillDetector
from .schemas import PillDetection, RecognitionResult
from .settings import Settings


class PillRecognitionPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        assets = download_model_assets(self.settings)
        self.detector = RTMDetPillDetector(
            assets.checkpoint,
            assets.class_names,
            self.settings,
        )
        self.aihub_classifier = self._load_optional_aihub_classifier()
        self.classifier = self._load_optional_classifier()

    def _load_optional_aihub_classifier(self) -> AIHubPillClassifier | None:
        if not self.settings.aihub_weights or not self.settings.aihub_mapping:
            return None
        return AIHubPillClassifier(
            self.settings.aihub_weights,
            self.settings.aihub_mapping,
            self.settings.device,
            self.settings.aihub_rotation_tta,
        )

    def _load_optional_classifier(self) -> EfficientNetPillClassifier | None:
        if not self.settings.cnn_weights or not self.settings.cnn_mapping:
            return None
        return EfficientNetPillClassifier(
            self.settings.cnn_weights,
            self.settings.cnn_mapping,
            self.settings.device,
        )

    def recognize(self, image_rgb: np.ndarray) -> RecognitionResult:
        image_rgb = ensure_rgb_uint8(image_rgb)
        height, width = image_rgb.shape[:2]
        detected_crops = []

        for pill_id, (bbox, rtmdet_candidates) in enumerate(
            self.detector.predict(image_rgb),
            start=1,
        ):
            x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
            crop_box = expand_bbox(
                (x1, y1, x2, y2),
                width,
                height,
                self.settings.crop_padding_ratio,
            )
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
            crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            detected_crops.append(
                (pill_id, (x1, y1, x2, y2), rtmdet_candidates, crop)
            )

        valid_crops = [crop for _, _, _, crop in detected_crops if crop.size]
        aihub_predictions = (
            self.aihub_classifier.predict_batch(valid_crops, self.settings.top_k)
            if self.aihub_classifier is not None
            else [[] for _ in valid_crops]
        )

        detections = []
        prediction_index = 0
        for pill_id, bbox, rtmdet_candidates, crop in detected_crops:
            aihub_candidates = []
            if crop.size:
                aihub_candidates = aihub_predictions[prediction_index]
                prediction_index += 1
            cnn_candidates = (
                self.classifier.predict(crop, self.settings.top_k)
                if self.classifier is not None and crop.size
                else []
            )
            status = determine_status(
                rtmdet_candidates,
                aihub_candidates,
                cnn_candidates,
            )
            detections.append(
                PillDetection(
                    pill_id=pill_id,
                    bbox=bbox,
                    status=status,
                    rtmdet_candidates=rtmdet_candidates,
                    aihub_candidates=aihub_candidates,
                    cnn_candidates=cnn_candidates,
                )
            )

        warnings = []
        if self.aihub_classifier is None:
            warnings.append(
                "AI Hub weights are not configured; 1,000-class candidates are unavailable."
            )
        if self.classifier is None:
            warnings.append("GitHub EfficientNet verifier is not configured.")
        if not detections:
            warnings.append("No pill was detected. Retake the photo with separated pills.")

        return RecognitionResult(
            image_width=width,
            image_height=height,
            pill_count=len(detections),
            model_version=self._model_version(),
            detections=detections,
            warnings=warnings,
        )

    def _model_version(self) -> str:
        if self.settings.detector_checkpoint is not None:
            versions = ["rtmdet-single-class"]
        else:
            versions = [f"{self.settings.model_repo_id}@{self.settings.model_revision[:8]}"]
        if self.aihub_classifier is not None:
            versions.append(self.aihub_classifier.model_version)
        return "+".join(versions)


def determine_status(rtmdet_candidates, aihub_candidates, cnn_candidates) -> str:
    if not rtmdet_candidates:
        return "retake_required"
    if aihub_candidates:
        return "needs_confirmation"
    if not cnn_candidates:
        return "needs_confirmation"
    same_top1 = rtmdet_candidates[0].class_name == cnn_candidates[0].class_name
    if (
        same_top1
        and rtmdet_candidates[0].confidence >= 0.5
        and cnn_candidates[0].confidence >= 0.85
    ):
        return "identified"
    return "needs_confirmation"


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    padding_x = round((x2 - x1) * padding_ratio)
    padding_y = round((y2 - y1) * padding_ratio)
    return clamp_bbox(
        (x1 - padding_x, y1 - padding_y, x2 + padding_x, y2 + padding_y),
        width,
        height,
    )


def clamp_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(1, min(x2, width)),
        max(1, min(y2, height)),
    )


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return np.clip(array, 0, 255).astype(np.uint8)
