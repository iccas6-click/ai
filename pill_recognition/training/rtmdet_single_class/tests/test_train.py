from pathlib import Path

from mmengine.config import ConfigDict

from training.rtmdet_single_class.scripts.train import (
    apply_smoke_overrides,
    apply_runtime_overrides,
    normalized_data_root,
)


def test_normalized_data_root_adds_trailing_slash():
    assert normalized_data_root(Path("datasets/example")) == "datasets/example/"


def test_apply_runtime_overrides_updates_dataset_and_evaluator_paths():
    cfg = ConfigDict(
        data_root="old/",
        work_dir="old-work",
        train_dataloader=ConfigDict(dataset=ConfigDict(data_root="old/")),
        val_dataloader=ConfigDict(dataset=ConfigDict(data_root="old/")),
        test_dataloader=ConfigDict(dataset=ConfigDict(data_root="old/")),
        val_evaluator=ConfigDict(ann_file="old/val_coco.json"),
        test_evaluator=ConfigDict(ann_file="old/val_coco.json"),
    )

    apply_runtime_overrides(
        cfg,
        data_root=Path("datasets/new-synthetic"),
        work_dir=Path("training/runs/new-synthetic"),
    )

    assert cfg.data_root == "datasets/new-synthetic/"
    assert cfg.train_dataloader.dataset.data_root == "datasets/new-synthetic/"
    assert cfg.val_dataloader.dataset.data_root == "datasets/new-synthetic/"
    assert cfg.test_dataloader.dataset.data_root == "datasets/new-synthetic/"
    assert cfg.val_evaluator.ann_file == "datasets/new-synthetic/val_coco.json"
    assert cfg.test_evaluator.ann_file == "datasets/new-synthetic/val_coco.json"
    assert cfg.work_dir == "training/runs/new-synthetic"


def test_apply_smoke_overrides_keeps_explicit_work_dir():
    cfg = ConfigDict(
        work_dir="explicit-work",
        train_cfg=ConfigDict(max_epochs=12),
        train_pipeline_stage2=["stage2"],
        train_dataloader=ConfigDict(
            batch_size=8,
            num_workers=4,
            persistent_workers=True,
            dataset=ConfigDict(indices=None, pipeline=["stage1"]),
        ),
        val_dataloader=ConfigDict(
            batch_size=4,
            num_workers=4,
            persistent_workers=True,
            dataset=ConfigDict(indices=None),
        ),
        test_dataloader=ConfigDict(dataset=ConfigDict(indices=None)),
        default_hooks=ConfigDict(
            logger=ConfigDict(interval=50),
            checkpoint=ConfigDict(interval=5),
        ),
    )

    apply_smoke_overrides(
        cfg,
        default_work_dir=Path("training/runs/smoke"),
        keep_work_dir=True,
        keep_num_workers=True,
    )

    assert cfg.work_dir == "explicit-work"
    assert cfg.train_cfg.max_epochs == 1
    assert cfg.train_dataloader.batch_size == 1
    assert cfg.train_dataloader.num_workers == 4
    assert cfg.train_dataloader.dataset.indices == 8
    assert cfg.train_dataloader.dataset.pipeline == ["stage2"]
    assert cfg.val_dataloader.batch_size == 1
    assert cfg.val_dataloader.num_workers == 4
    assert cfg.val_dataloader.dataset.indices == 4
    assert cfg.test_dataloader == cfg.val_dataloader
    assert cfg.default_hooks.logger.interval == 1
    assert cfg.default_hooks.checkpoint.interval == 1
