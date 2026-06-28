import numpy as np

from pill_recognition.image_quality import assess_image_quality


def test_assess_image_quality_flags_low_resolution_crop():
    image = np.full((40, 50, 3), 180, dtype=np.uint8)

    warnings = assess_image_quality(image, context="crop 1")

    assert any("resolution is low" in warning for warning in warnings)


def test_assess_image_quality_flags_dark_image():
    image = np.full((300, 600, 3), 18, dtype=np.uint8)

    warnings = assess_image_quality(image, context="image")

    assert any("too dark" in warning for warning in warnings)


def test_assess_image_quality_flags_overexposed_low_contrast_image():
    image = np.full((300, 600, 3), 250, dtype=np.uint8)

    warnings = assess_image_quality(image, context="image")

    assert any("overexposed" in warning for warning in warnings)
    assert any("contrast is low" in warning for warning in warnings)
