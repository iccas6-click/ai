from training.rtmdet_single_class.scripts.prepare_single_class import parse_yolo_line


def test_parse_yolo_line_accepts_normalized_bbox():
    assert parse_yolo_line("117 0.5 0.4 0.2 0.1") == (117, 0.5, 0.4, 0.2, 0.1)
