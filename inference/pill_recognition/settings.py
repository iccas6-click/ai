from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    device: str = "cuda:0"
    crop_padding_ratio: float = 0.12
    top_k: int = 3
    max_batch_crops: int = 12
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 12_000_000
    candidate_min_score: float = 70.0
    candidate_ambiguity_margin: float = 3.0
    recognizer: str = "retrieval"
    gemini_model: str = "gemini-3.5-flash"
    gemini_api_key: str | None = None
    aihub_weights: Path | None = None
    aihub_mapping: Path | None = None
    retrieval_index: Path = PROJECT_ROOT / "artifacts" / "retrieval" / "aihub_resnet_index.pt"
    retrieval_metadata_rerank: bool = False
    retrieval_query_preprocess: str = "none"
    warmup_on_startup: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        packaged_aihub_mapping = (
            PROJECT_ROOT
            / "aihub_official_code"
            / "package"
            / "평가용 데이터셋"
            / "pill_data"
            / "pill_data_croped"
            / "pill_label_path_sharp_score.json"
        )
        legacy_aihub_mapping = (
            PROJECT_ROOT
            / "aihub_official_code"
            / "docker img"
            / "proj_pill"
            / "pill_label_path_sharp_score.json"
        )
        packaged_aihub_weights = (
            PROJECT_ROOT
            / "aihub_official_code"
            / "package"
            / "평가용 데이터셋"
            / "pill_data"
            / "proj_pill"
            / "pill_resnet152_dataclass01_aug0.pt"
        )
        legacy_aihub_weights = (
            PROJECT_ROOT
            / "aihub_official_code"
            / "docker img"
            / "proj_pill"
            / "pill_resnet152_dataclass01_aug0.pt"
        )
        env_mapping = os.getenv("PILL_AIHUB_MAPPING")
        env_weights = os.getenv("PILL_AIHUB_WEIGHTS")
        env_retrieval_index = os.getenv("PILL_RETRIEVAL_INDEX")
        default_mapping = (
            packaged_aihub_mapping
            if packaged_aihub_mapping.exists()
            else legacy_aihub_mapping
        )
        default_weights = (
            packaged_aihub_weights
            if packaged_aihub_weights.exists()
            else legacy_aihub_weights
        )
        mapping = Path(env_mapping).expanduser() if env_mapping else default_mapping
        weights = Path(env_weights).expanduser() if env_weights else default_weights

        return cls(
            device=os.getenv("PILL_DEVICE", "cuda:0"),
            crop_padding_ratio=float(os.getenv("PILL_CROP_PADDING_RATIO", "0.12")),
            top_k=int(os.getenv("PILL_TOP_K", "3")),
            max_batch_crops=int(os.getenv("PILL_MAX_BATCH_CROPS", "12")),
            max_upload_bytes=int(
                os.getenv("PILL_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
            ),
            max_image_pixels=int(os.getenv("PILL_MAX_IMAGE_PIXELS", "12000000")),
            candidate_min_score=float(os.getenv("PILL_CANDIDATE_MIN_SCORE", "70")),
            candidate_ambiguity_margin=float(
                os.getenv("PILL_CANDIDATE_AMBIGUITY_MARGIN", "3")
            ),
            recognizer=os.getenv("PILL_RECOGNIZER", "retrieval").strip().lower(),
            gemini_model=os.getenv("PILL_GEMINI_MODEL", "gemini-3.5-flash"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            aihub_weights=weights if weights.exists() else None,
            aihub_mapping=mapping if mapping.exists() else None,
            retrieval_index=(
                Path(env_retrieval_index).expanduser()
                if env_retrieval_index
                else cls.retrieval_index
            ),
            retrieval_metadata_rerank=parse_bool(
                os.getenv("PILL_RETRIEVAL_METADATA_RERANK"),
                default=False,
            ),
            retrieval_query_preprocess=os.getenv(
                "PILL_RETRIEVAL_QUERY_PREPROCESS",
                "none",
            ).strip().lower(),
            warmup_on_startup=parse_bool(
                os.getenv("PILL_WARMUP_ON_STARTUP"),
                default=True,
            ),
        )


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}
