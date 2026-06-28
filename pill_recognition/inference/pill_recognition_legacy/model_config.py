from __future__ import annotations

from pathlib import Path

import mmdet
from mmengine import Config


def build_rtmdet_config(class_names: list[str], image_size: int) -> Config:
    config_path = (
        Path(mmdet.__file__).resolve().parent
        / ".mim"
        / "configs"
        / "rtmdet"
        / "rtmdet_tiny_8xb32-300e_coco.py"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"MMDetection RTMDet config not found: {config_path}")

    config = Config.fromfile(config_path)
    config.model.bbox_head.num_classes = len(class_names)
    config.model.backbone.init_cfg = None
    config.load_from = None

    metainfo = {"classes": tuple(class_names)}
    config.test_dataloader.dataset.metainfo = metainfo
    config.val_dataloader.dataset.metainfo = metainfo

    for pipeline in (
        config.test_dataloader.dataset.pipeline,
        config.val_dataloader.dataset.pipeline,
    ):
        _set_inference_size(pipeline, image_size)

    return config


def _set_inference_size(pipeline: list[dict], image_size: int) -> None:
    size = (image_size, image_size)
    for step in pipeline:
        if step.get("type") == "Resize":
            step["scale"] = size
        elif step.get("type") == "Pad":
            step["size"] = size
