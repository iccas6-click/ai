from __future__ import annotations

from ..settings import Settings
from .base import VisionProvider
from .local import LocalVisionProvider


def create_vision_provider(settings: Settings) -> VisionProvider:
    if settings.recognizer == "gemini":
        if not settings.allow_gemini_recognizer:
            raise RuntimeError(
                "PILL_RECOGNIZER=gemini is experimental. "
                "Set PILL_ENABLE_EXPERIMENTAL_GEMINI=1 to use it outside "
                "the production retrieval path."
            )
        if not settings.gemini_api_key:
            raise RuntimeError("PILL_RECOGNIZER=gemini requires GEMINI_API_KEY.")
        from .gemini import GeminiVisionProvider

        return GeminiVisionProvider(settings.gemini_api_key, settings.gemini_model)
    if settings.recognizer in {"local", "local_heuristic", ""}:
        return LocalVisionProvider()
    raise ValueError(f"Unsupported vision recognizer: {settings.recognizer}")
