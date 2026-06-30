import random

import numpy as np

from pill_recognition.build_augmented_retrieval_index import (
    composite_on_background,
    extract_foreground_mask,
    mask_bbox,
)


def test_extract_foreground_mask_finds_center_object():
    image = np.full((80, 80, 3), 245, dtype=np.uint8)
    image[25:55, 20:60] = [80, 120, 200]

    bbox = mask_bbox(extract_foreground_mask(image))

    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x1 <= 24
    assert y1 <= 29
    assert x2 >= 56
    assert y2 >= 51


def test_composite_on_background_preserves_patch_shape():
    patch = np.full((40, 60, 3), [200, 100, 80], dtype=np.uint8)
    alpha = np.full((40, 60), 255, dtype=np.uint8)

    composite = composite_on_background(patch, alpha, random.Random(1))

    assert composite.shape == patch.shape
    assert composite.dtype == np.uint8
