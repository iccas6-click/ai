from pill_recognition.app import format_bbox, format_ingredient


def test_format_bbox_uses_compact_coordinates():
    assert format_bbox((1, 2, 30, 40)) == "1,2,30,40"


def test_format_ingredient_splits_aihub_pipe_separator():
    assert format_ingredient("텔미사르탄|암로디핀베실산염") == "텔미사르탄, 암로디핀베실산염"
