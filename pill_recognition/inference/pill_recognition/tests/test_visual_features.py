import numpy as np

from pill_recognition.visual_features import estimate_crop_visual_features


def test_estimate_crop_visual_features_detects_white_round_crop():
    crop = np.full((80, 80, 3), 245, dtype=np.uint8)

    features = estimate_crop_visual_features(crop)

    assert features.color == "하양"
    assert features.shape == "원형"


def test_estimate_crop_visual_features_detects_yellow_oblong_crop():
    crop = np.zeros((60, 150, 3), dtype=np.uint8)
    crop[:, :] = [230, 190, 35]

    features = estimate_crop_visual_features(crop)

    assert features.color == "노랑"
    assert features.shape == "장방형"


def test_estimate_crop_visual_features_ignores_invalid_input():
    features = estimate_crop_visual_features(object())

    assert features.color is None
    assert features.shape is None
