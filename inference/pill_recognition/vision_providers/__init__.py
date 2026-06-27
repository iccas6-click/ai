from __future__ import annotations

from ..settings import Settings
from .base import VisionProvider
from .local import LocalVisionProvider


def create_vision_provider(settings: Settings) -> VisionProvider:
    if settings.vision_provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("PILL_VISION_PROVIDER=gemini requires GEMINI_API_KEY.")
        from .gemini import GeminiVisionProvider

        return GeminiVisionProvider(settings.gemini_api_key, settings.gemini_model)
    if settings.vision_provider in {"local", "local_heuristic", ""}:
        return LocalVisionProvider()
    raise ValueError(f"Unsupported vision provider: {settings.vision_provider}")
