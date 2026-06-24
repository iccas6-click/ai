from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from mmdet.apis import inference_detector, init_detector

from .model_config import build_rtmdet_config
from .postprocess import group_predictions
from .schemas import Candidate
from .settings import Settings


class RTMDetPillDetector:
    def __init__(
        self,
        checkpoint_path: Path,
        class_names_path: Path,
        settings: Settings,
    ) -> None:
        self.settings = settings
        self.class_names = load_class_names(class_names_path)
        config = build_rtmdet_config(self.class_names, settings.image_size)
        self.model = init_detector(
            config,
            str(checkpoint_path),
            device=settings.device,
        )

    def predict(
        self,
        image_rgb: np.ndarray,
    ) -> list[tuple[tuple[int, int, int, int], list[Candidate]]]:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        prediction = inference_detector(self.model, image_bgr).pred_instances

        grouped = group_predictions(
            prediction.bboxes.detach().cpu().numpy(),
            prediction.scores.detach().cpu().numpy(),
            prediction.labels.detach().cpu().numpy(),
            self.class_names,
            self.settings.confidence_threshold,
            self.settings.grouping_iou_threshold,
            self.settings.top_k,
        )
        return grouped[: self.settings.max_detections]


def load_class_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file)

    names = data.get("names", data)
    if isinstance(names, dict):
        return [str(names[index]).strip() for index in sorted(names)]
    if isinstance(names, list):
        return [str(name).strip() for name in names]
    raise ValueError("pill.yaml does not contain a valid names mapping")
