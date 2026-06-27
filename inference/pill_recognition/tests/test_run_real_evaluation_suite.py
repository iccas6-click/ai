import argparse
from pathlib import Path

from pill_recognition.run_real_evaluation_suite import build_suite_commands


def test_build_suite_commands_creates_three_evaluations_and_three_comparisons(tmp_path):
    args = argparse.Namespace(
        dataset_root=Path("datasets/evaluation/real-smartphone"),
        images_dir=None,
        annotations_dir=None,
        output_dir=tmp_path / "outputs",
        prefix="real-smartphone",
        pattern="*.json",
        limit=None,
        iou_threshold=0.5,
        top_k=5,
    )

    commands = build_suite_commands(args)

    assert [command.name for command in commands] == [
        "evaluate:none",
        "evaluate:annotation",
        "evaluate:ground-truth",
        "compare:annotation-vs-unscoped",
        "compare:oracle-vs-unscoped",
        "compare:oracle-vs-annotation",
    ]
    assert commands[0].output == tmp_path / "outputs" / "real-smartphone-unscoped.json"
    assert "--scope-mode" in commands[0].args
    assert "none" in commands[0].args
    assert commands[3].output == (
        tmp_path / "outputs" / "real-smartphone-annotation-vs-unscoped.json"
    )
    assert "pill_recognition.compare_real_evaluations" in commands[3].args


def test_build_suite_commands_passes_optional_input_dirs_and_limit(tmp_path):
    args = argparse.Namespace(
        dataset_root=Path("dataset"),
        images_dir=Path("custom-images"),
        annotations_dir=Path("custom-annotations"),
        output_dir=tmp_path,
        prefix="smoke",
        pattern="IMG_*.json",
        limit=10,
        iou_threshold=0.6,
        top_k=3,
    )

    commands = build_suite_commands(args)
    evaluate_args = commands[0].args

    assert ["--images-dir", "custom-images"] == evaluate_args[
        evaluate_args.index("--images-dir") : evaluate_args.index("--images-dir") + 2
    ]
    assert ["--annotations-dir", "custom-annotations"] == evaluate_args[
        evaluate_args.index("--annotations-dir") : evaluate_args.index("--annotations-dir") + 2
    ]
    assert ["--limit", "10"] == evaluate_args[
        evaluate_args.index("--limit") : evaluate_args.index("--limit") + 2
    ]
    assert ["--top-k", "3"] == evaluate_args[
        evaluate_args.index("--top-k") : evaluate_args.index("--top-k") + 2
    ]
