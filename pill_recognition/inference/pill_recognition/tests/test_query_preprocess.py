import numpy as np
import pytest

from pill_recognition.retrieval import infer_variant_count, query_preprocess_modes
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


def test_preprocess_query_crop_grabcut_dark_centers_object_on_dark_square():
    image = np.full((120, 180, 3), [230, 220, 120], dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    cv2.ellipse(image, (90, 60), (42, 24), 0, 0, 360, (210, 130, 80), -1)

    result = preprocess_query_crop(image, "grabcut_dark")

    assert result.shape[0] == result.shape[1]
    assert np.allclose(result[0, 0], AIHUB_DARK_BACKGROUND, atol=3)
    center = result[
        result.shape[0] // 2 - 4 : result.shape[0] // 2 + 4,
        result.shape[1] // 2 - 4 : result.shape[1] // 2 + 4,
    ]
    assert center[:, :, 0].mean() > 150


def test_preprocess_query_crop_rejects_unknown_mode():
    image = np.full((80, 120, 3), 240, dtype=np.uint8)

    with pytest.raises(ValueError):
        preprocess_query_crop(image, "wat")


def test_query_preprocess_modes_expands_multi_aliases():
    assert query_preprocess_modes("none") == ["none"]
    assert query_preprocess_modes("multi_foreground") == ["none", "foreground"]
    assert query_preprocess_modes("multi_grabcut") == ["none", "grabcut_dark"]
    assert query_preprocess_modes("none+foreground+none") == ["none", "foreground"]


def test_infer_variant_count_requires_fixed_variant_count():
    assert infer_variant_count(2, crop_count=2) == 1
    assert infer_variant_count(4, crop_count=2) == 2
    with pytest.raises(ValueError):
        infer_variant_count(5, crop_count=2)
