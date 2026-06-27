from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from .settings import Settings


@dataclass(frozen=True)
class ModelAssets:
    checkpoint: Path
    class_names: Path


def download_model_assets(settings: Settings) -> ModelAssets:
    if settings.detector_checkpoint and settings.detector_class_names:
        if not settings.detector_checkpoint.exists():
            raise FileNotFoundError(settings.detector_checkpoint)
        if not settings.detector_class_names.exists():
            raise FileNotFoundError(settings.detector_class_names)
        return ModelAssets(
            checkpoint=settings.detector_checkpoint,
            class_names=settings.detector_class_names,
        )

    local_dir = settings.artifact_dir / "rtmdet-v4"
    local_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "repo_id": settings.model_repo_id,
        "revision": settings.model_revision,
        "local_dir": local_dir,
    }
    checkpoint = hf_hub_download(
        filename=settings.checkpoint_filename,
        **common,
    )
    class_names = hf_hub_download(
        filename=settings.class_names_filename,
        **common,
    )

    return ModelAssets(Path(checkpoint), Path(class_names))
