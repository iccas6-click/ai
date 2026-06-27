import numpy as np
import pytest

from pill_recognition.query_preprocess import (
    AIHUB_DARK_BACKGROUND,
    preprocess_query_crop,
)


def test_preprocess_query_crop_returns_original_when_disabled():
    image = np.full((80, 120, 3), 240, dtype=np.uint8)

    result = preprocess_query_crop(image, "none")

    assert result is image


def test_preprocess_query_crop_foreground_dark_centers_object_on_dark_square():
    image = np.full((120, 180, 3), 240, dtype=np.uint8)
    image[40:80, 70:110] = [210, 120, 80]

    result = preprocess_query_crop(image, "foreground_dark")

    assert result.shape[0] == result.shape[1]
    assert np.allclose(result[0, 0], AIHUB_DARK_BACKGROUND, atol=2)
    center = result[
        result.shape[0] // 2 - 4 : result.shape[0] // 2 + 4,
        result.shape[1] // 2 - 4 : result.shape[1] // 2 + 4,
    ]
    assert center[:, :, 0].mean() > 150


def test_preprocess_query_crop_rejects_unknown_mode():
    image = np.full((80, 120, 3), 240, dtype=np.uint8)

    with pytest.raises(ValueError):
        preprocess_query_crop(image, "wat")
