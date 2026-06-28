from pill_recognition_legacy.evaluate_dataset import (
    EvaluationTarget,
    all_present,
    parse_list,
    summarize,
    target_from_row,
)


def test_parse_list_accepts_semicolon_and_comma_values():
    assert parse_list("K-000001; K-000002,K-000003") == [
        "K-000001",
        "K-000002",
        "K-000003",
    ]


def test_target_from_row_parses_optional_fields():
    target = target_from_row(
        {
            "image": "real_001.jpg",
            "expected_count": "3",
            "expected_class_names": "K-000001;K-000002",
            "expected_item_seqs": "199900001",
        }
    )

    assert target == EvaluationTarget(
        image="real_001.jpg",
        expected_count=3,
        expected_class_names=["K-000001", "K-000002"],
        expected_item_seqs=["199900001"],
    )


def test_all_present_requires_every_expected_value():
    assert all_present(["K-1", "K-2"], ["K-2", "K-1", "K-3"]) is True
    assert all_present(["K-1", "K-4"], ["K-1", "K-2"]) is False
    assert all_present([], ["K-1"]) is None


def test_summarize_counts_available_labels_only():
    rows = [
        {
            "expected_count": 2,
            "count_exact": True,
            "count_abs_error": 0,
            "class_top1_all_hit": True,
            "class_top3_all_hit": True,
            "item_seq_top1_all_hit": None,
            "item_seq_top3_all_hit": None,
        },
        {
            "expected_count": 3,
            "count_exact": False,
            "count_abs_error": 1,
            "class_top1_all_hit": False,
            "class_top3_all_hit": True,
            "item_seq_top1_all_hit": True,
            "item_seq_top3_all_hit": True,
        },
    ]

    assert summarize(rows) == {
        "image_count": 2,
        "count_labeled_images": 2,
        "count_exact_accuracy": 0.5,
        "count_mean_abs_error": 0.5,
        "class_labeled_images": 2,
        "class_top1_all_hit_accuracy": 0.5,
        "class_top3_all_hit_accuracy": 1.0,
        "item_seq_labeled_images": 1,
        "item_seq_top1_all_hit_accuracy": 1.0,
        "item_seq_top3_all_hit_accuracy": 1.0,
    }
