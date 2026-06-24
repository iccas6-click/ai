from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path = PROJECT_ROOT / "artifacts"
    model_repo_id: str = "wony98/healtheat-pill-rtmdet-v4"
    model_revision: str = "91fa48ea31327c7c724e7c104a61b55119a6ae31"
    checkpoint_filename: str = "v4.pth"
    class_names_filename: str = "pill.yaml"
    detector_checkpoint: Path | None = None
    detector_class_names: Path | None = None
    image_size: int = 1536
    confidence_threshold: float = 0.25
    grouping_iou_threshold: float = 0.5
    top_k: int = 3
    max_detections: int = 10
    crop_padding_ratio: float = 0.12
    device: str = "cuda:0"
    aihub_weights: Path | None = None
    aihub_mapping: Path | None = None
    cnn_weights: Path | None = None
    cnn_mapping: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        device = os.getenv("PILL_DEVICE", "cuda:0")
        detector_checkpoint = os.getenv("PILL_DETECTOR_CHECKPOINT")
        detector_class_names = os.getenv("PILL_DETECTOR_CLASSES")
        aihub_weights = os.getenv("PILL_AIHUB_WEIGHTS")
        aihub_mapping = os.getenv("PILL_AIHUB_MAPPING")
        cnn_weights = os.getenv("PILL_CNN_WEIGHTS")
        cnn_mapping = os.getenv("PILL_CNN_MAPPING")
        default_aihub_dir = (
            PROJECT_ROOT / "aihub_official_code" / "docker img" / "proj_pill"
        )
        default_aihub_weights = default_aihub_dir / "pill_resnet152_dataclass01_aug0.pt"
        default_aihub_mapping = default_aihub_dir / "pill_label_path_sharp_score.json"
        default_cnn_dir = PROJECT_ROOT / "artifacts" / "cnn"
        default_cnn_weights = default_cnn_dir / "cls119_classifier_v4.pt"
        default_cnn_mapping = default_cnn_dir / "class_mapping.csv"
        default_detector_dir = PROJECT_ROOT / "artifacts" / "rtmdet-single-class"
        default_detector_checkpoint = default_detector_dir / "model.pth"
        default_detector_classes = default_detector_dir / "pill.yaml"

        return cls(
            device=device,
            detector_checkpoint=(
                Path(detector_checkpoint).expanduser()
                if detector_checkpoint
                else default_detector_checkpoint
                if default_detector_checkpoint.exists()
                else None
            ),
            detector_class_names=(
                Path(detector_class_names).expanduser()
                if detector_class_names
                else default_detector_classes if default_detector_classes.exists() else None
            ),
            aihub_weights=(
                Path(aihub_weights).expanduser()
                if aihub_weights
                else default_aihub_weights if default_aihub_weights.exists() else None
            ),
            aihub_mapping=(
                Path(aihub_mapping).expanduser()
                if aihub_mapping
                else default_aihub_mapping if default_aihub_mapping.exists() else None
            ),
            cnn_weights=(
                Path(cnn_weights).expanduser()
                if cnn_weights
                else default_cnn_weights if default_cnn_weights.exists() else None
            ),
            cnn_mapping=(
                Path(cnn_mapping).expanduser()
                if cnn_mapping
                else default_cnn_mapping if default_cnn_mapping.exists() else None
            ),
        )
