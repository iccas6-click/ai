from __future__ import annotations

import cv2
import numpy as np

from .product_db import ProductSearchQuery, load_product_index, search_products
from .schemas import PillDetection, ProductCandidate, RecognitionResult, VisionObservation
from .settings import Settings
from .vision_providers import create_vision_provider


class PillRecognitionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        detector=None,
        vision_provider=None,
        product_index: dict | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.detector = detector or self._load_detector()
        self.vision_provider = vision_provider or create_vision_provider(self.settings)
        self.product_index = (
            product_index
            if product_index is not None
            else load_product_index(self.settings.aihub_mapping)
        )

    def _load_detector(self):
        from pill_recognition_legacy.assets import download_model_assets
        from pill_recognition_legacy.detector import RTMDetPillDetector
        from pill_recognition_legacy.settings import Settings as LegacySettings

        legacy_settings = LegacySettings.from_env()
        assets = download_model_assets(legacy_settings)
        return RTMDetPillDetector(
            assets.checkpoint,
            assets.class_names,
            legacy_settings,
        )

    def recognize(self, image_rgb: np.ndarray) -> RecognitionResult:
        image_rgb = ensure_rgb_uint8(image_rgb)
        height, width = image_rgb.shape[:2]
        detections = []

        for pill_id, (bbox, detector_candidates) in enumerate(
            self.detector.predict(image_rgb),
            start=1,
        ):
            bbox = clamp_bbox(bbox, width, height)
            crop_bbox = expand_bbox(
                bbox,
                width,
                height,
                self.settings.crop_padding_ratio,
            )
            x1, y1, x2, y2 = crop_bbox
            crop = image_rgb[y1:y2, x1:x2]
            observation = (
                self.vision_provider.inspect_crop(crop)
                if crop.size
                else VisionObservation(notes="empty crop")
            )
            candidates = rank_product_candidates(
                search_products(
                    self.product_index,
                    product_query_from_observation(observation, self.settings.top_k),
                ),
                self.settings.top_k,
            )
            detector_confidence = (
                float(detector_candidates[0].confidence) if detector_candidates else 0.0
            )
            detections.append(
                PillDetection(
                    pill_id=pill_id,
                    bbox=bbox,
                    crop_bbox=crop_bbox,
                    detector_confidence=round(detector_confidence, 4),
                    vision=observation,
                    candidates=candidates,
                    status=determine_status(observation, candidates),
                )
            )

        warnings = []
        if not self.product_index:
            warnings.append("AI Hub product metadata is unavailable.")
        if not detections:
            warnings.append("No pill was detected. Retake the photo with separated pills.")

        return RecognitionResult(
            image_width=width,
            image_height=height,
            pill_count=len(detections),
            model_version=f"rtmdet-single-class+{self.vision_provider.name}+aihub-db",
            detections=detections,
            warnings=warnings,
        )


def product_query_from_observation(
    observation: VisionObservation,
    limit: int,
) -> ProductSearchQuery:
    imprints = [
        value
        for value in (observation.imprint_front, observation.imprint_back)
        if value
    ]
    text_parts = []
    if observation.text:
        text_parts.append(observation.text)
    text_parts.extend(observation.possible_product_names)
    return ProductSearchQuery(
        imprint=" ".join(imprints),
        shape=observation.shape or "",
        color=observation.color or "",
        text=" ".join(text_parts),
        limit=limit,
    )


def rank_product_candidates(rows: list[dict], limit: int) -> list[ProductCandidate]:
    candidates = []
    for rank, row in enumerate(rows[:limit], start=1):
        candidates.append(
            ProductCandidate(
                rank=rank,
                pill_id=row["pill_id"],
                score=int(row.get("score", 0)),
                product_name=row.get("product_name"),
                ingredient=row.get("ingredient"),
                company=row.get("company"),
                item_seq=row.get("item_seq"),
                etc_otc_code=row.get("etc_otc_code"),
                print_front=row.get("print_front"),
                print_back=row.get("print_back"),
                drug_shape=row.get("drug_shape"),
                color_class1=row.get("color_class1"),
                color_class2=row.get("color_class2"),
                matched=row.get("matched"),
            )
        )
    return candidates


def determine_status(
    observation: VisionObservation,
    candidates: list[ProductCandidate],
) -> str:
    if not candidates:
        return "needs_manual_search"
    if candidates[0].score >= 100 and (observation.confidence or 0) >= 0.6:
        return "needs_confirmation"
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
        max(0, min(int(x1), width - 1)),
        max(0, min(int(y1), height - 1)),
        max(1, min(int(x2), width)),
        max(1, min(int(y2), height)),
    )


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    return np.clip(array, 0, 255).astype(np.uint8)
