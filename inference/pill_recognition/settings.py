from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    device: str = "cuda:0"
    crop_padding_ratio: float = 0.12
    top_k: int = 5
    vision_provider: str = "local"
    gemini_model: str = "gemini-3.5-flash"
    gemini_api_key: str | None = None
    aihub_mapping: Path | None = None

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
        env_mapping = os.getenv("PILL_AIHUB_MAPPING")
        default_mapping = (
            packaged_aihub_mapping
            if packaged_aihub_mapping.exists()
            else legacy_aihub_mapping
        )
        mapping = Path(env_mapping).expanduser() if env_mapping else default_mapping

        return cls(
            device=os.getenv("PILL_DEVICE", "cuda:0"),
            crop_padding_ratio=float(os.getenv("PILL_CROP_PADDING_RATIO", "0.12")),
            top_k=int(os.getenv("PILL_TOP_K", "5")),
            vision_provider=os.getenv("PILL_VISION_PROVIDER", "local").strip().lower(),
            gemini_model=os.getenv("PILL_GEMINI_MODEL", "gemini-3.5-flash"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            aihub_mapping=mapping if mapping.exists() else None,
        )
