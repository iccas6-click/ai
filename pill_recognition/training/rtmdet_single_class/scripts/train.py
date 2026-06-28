from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from mmengine.config import Config
from mmengine.runner import Runner


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "training"
    / "rtmdet_single_class"
    / "configs"
    / "rtmdet_tiny_single_class.py"
)
SOURCE_CHECKPOINT = PROJECT_ROOT / "inference" / "artifacts" / "rtmdet-v4" / "v4.pth"
ADAPTED_CHECKPOINT = PROJECT_ROOT / "training" / "runs" / "pretrained" / "rtmdet-v4-single-class-init.pth"
SOURCE_REPO_ID = "wony98/healtheat-pill-rtmdet-v4"
SOURCE_REVISION = "91fa48ea31327c7c724e7c104a61b55119a6ae31"


def ensure_source_checkpoint() -> Path:
    if SOURCE_CHECKPOINT.exists():
        return SOURCE_CHECKPOINT
    SOURCE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=SOURCE_REPO_ID,
            revision=SOURCE_REVISION,
            filename=SOURCE_CHECKPOINT.name,
            local_dir=SOURCE_CHECKPOINT.parent,
        )
    )


def adapt_checkpoint(source: Path, destination: Path) -> Path:
    if destination.exists():
        return destination
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    for state_key in ("state_dict", "ema_state_dict"):
        state = checkpoint.get(state_key)
        if isinstance(state, dict):
            checkpoint[state_key] = {
                key: value
                for key, value in state.items()
                if "bbox_head.rtm_cls" not in key
            }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-class RTMDet")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override dataset root containing train_coco.json, val_coco.json, images/.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Override MMDetection work_dir for checkpoints and logs.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override train dataloader workers. Use 0 for WSL and 4 or more on Linux.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from random initialization instead of the RTMDet v4 checkpoint.",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    cfg.resume = args.resume
    apply_runtime_overrides(cfg, data_root=args.data_root, work_dir=args.work_dir)
    if args.num_workers is not None:
        if args.num_workers < 0:
            parser.error("--num-workers must be zero or greater")
        cfg.train_dataloader.num_workers = args.num_workers
        cfg.train_dataloader.persistent_workers = False
    if not args.resume and not args.no_pretrained:
        source_checkpoint = ensure_source_checkpoint()
        cfg.load_from = str(adapt_checkpoint(source_checkpoint, ADAPTED_CHECKPOINT))

    if args.smoke:
        apply_smoke_overrides(
            cfg,
            default_work_dir=PROJECT_ROOT / "training" / "runs" / "smoke",
            keep_work_dir=args.work_dir is not None,
            keep_num_workers=args.num_workers is not None,
        )

    Runner.from_cfg(cfg).train()


def apply_runtime_overrides(
    cfg,
    data_root: Path | None = None,
    work_dir: Path | None = None,
) -> None:
    if data_root is not None:
        root = normalized_data_root(data_root)
        cfg.data_root = root
        cfg.train_dataloader.dataset.data_root = root
        cfg.val_dataloader.dataset.data_root = root
        cfg.test_dataloader.dataset.data_root = root
        cfg.val_evaluator.ann_file = root + "val_coco.json"
        cfg.test_evaluator.ann_file = root + "val_coco.json"
    if work_dir is not None:
        cfg.work_dir = str(work_dir)


def normalized_data_root(path: Path) -> str:
    text = str(path)
    return text if text.endswith("/") else text + "/"


def apply_smoke_overrides(
    cfg,
    default_work_dir: Path,
    keep_work_dir: bool = False,
    keep_num_workers: bool = False,
) -> None:
    if not keep_work_dir:
        cfg.work_dir = str(default_work_dir)
    cfg.train_cfg.max_epochs = 1
    cfg.train_dataloader.batch_size = 1
    if not keep_num_workers:
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
    cfg.train_dataloader.dataset.indices = 8
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline_stage2
    cfg.val_dataloader.batch_size = 1
    if not keep_num_workers:
        cfg.val_dataloader.num_workers = 0
        cfg.val_dataloader.persistent_workers = False
    cfg.val_dataloader.dataset.indices = 4
    cfg.test_dataloader = cfg.val_dataloader
    cfg.default_hooks.logger.interval = 1
    cfg.default_hooks.checkpoint.interval = 1


if __name__ == "__main__":
    main()
