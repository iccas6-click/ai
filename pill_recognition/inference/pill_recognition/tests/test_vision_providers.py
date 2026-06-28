import pytest

from pill_recognition.settings import Settings
from pill_recognition.vision_providers import create_vision_provider
from pill_recognition.vision_providers.local import LocalVisionProvider


def test_create_vision_provider_defaults_to_local_for_local_recognizer():
    provider = create_vision_provider(Settings(recognizer="local"))

    assert isinstance(provider, LocalVisionProvider)


def test_gemini_recognizer_requires_explicit_experimental_flag():
    with pytest.raises(RuntimeError, match="experimental"):
        create_vision_provider(
            Settings(
                recognizer="gemini",
                gemini_api_key="test-key",
                allow_gemini_recognizer=False,
            )
        )


def test_gemini_recognizer_requires_api_key_when_enabled():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        create_vision_provider(
            Settings(
                recognizer="gemini",
                allow_gemini_recognizer=True,
                gemini_api_key=None,
            )
        )
