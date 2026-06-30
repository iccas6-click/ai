from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .query_preprocess import preprocess_query_crop
from .settings import Settings
from pill_recognition_legacy.aihub_classifier import load_aihub_class_names


class SyntheticPillCropDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        class_ids_by_name: dict[str, int],
        transform,
        query_preprocess: str,
        limit_images: int | None = None,
        include_source_crops: bool = False,
    ) -> None:
        self.transform = transform
        self.query_preprocess = query_preprocess
        self.samples = []
        image_dir = dataset_root / "images" / split
        metadata_dir = dataset_root / "metadata" / split
        image_paths = sorted(
            path
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp")
            for path in image_dir.glob(pattern)
            if path.is_file()
        )
        if limit_images is not None:
            image_paths = image_paths[:limit_images]
        for image_path in image_paths:
            metadata_path = metadata_dir / f"{image_path.stem}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for pill in metadata.get("pills", []):
                class_id = class_ids_by_name.get(str(pill.get("class_name")))
                if class_id is None:
                    continue
                self.samples.append(
                    {
                        "kind": "scene",
                        "image_path": str(image_path),
                        "bbox": tuple(map(int, pill["bbox_xyxy"])),
                        "class_id": class_id,
                    }
                )
                if include_source_crops and pill.get("source_image"):
                    source_path = Path(pill["source_image"])
                    if source_path.exists():
                        self.samples.append(
                            {
                                "kind": "source",
                                "image_path": str(source_path),
                                "bbox": None,
                                "class_id": class_id,
                            }
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        if sample["bbox"] is not None:
            x1, y1, x2, y2 = sample["bbox"]
            image = image[y1:y2, x1:x2]
            image = preprocess_query_crop(image, self.query_preprocess)
        tensor = self.transform(Image.fromarray(image).convert("RGB"))
        return tensor, int(sample["class_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune AIHub ResNet152 on synthetic scene crops."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/processed/rtmdet-aihub-synthetic-realistic-max10-v2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--query-preprocess", default="none")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-val-images", type=int)
    parser.add_argument("--include-source-crops", action="store_true")
    parser.add_argument(
        "--trainable-blocks",
        choices=["fc", "layer4", "layer3-layer4", "all"],
        default="layer4",
        help="Backbone blocks to unfreeze. All modes also train fc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    if settings.aihub_weights is None or settings.aihub_mapping is None:
        raise FileNotFoundError("AIHub weights and mapping must be configured.")

    class_names = load_aihub_class_names(settings.aihub_mapping)
    class_ids_by_name = {name: class_id for class_id, name in class_names.items()}
    checkpoint = torch.load(settings.aihub_weights, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("AIHub checkpoint does not contain a model state dict.")

    device = torch.device(settings.device)
    output_classes = int(state_dict["fc.weight"].shape[0])
    model = models.resnet152(weights=None, num_classes=output_classes)
    model.load_state_dict(state_dict)
    freeze_model(model, args.trainable_blocks)
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    train_dataset = SyntheticPillCropDataset(
        args.dataset_root,
        "train",
        class_ids_by_name,
        transform,
        args.query_preprocess,
        limit_images=args.max_train_images,
        include_source_crops=args.include_source_crops,
    )
    val_dataset = SyntheticPillCropDataset(
        args.dataset_root,
        "val",
        class_ids_by_name,
        transform,
        args.query_preprocess,
        limit_images=args.max_val_images,
        include_source_crops=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    start = perf_counter()
    report_rows = []
    best_top3 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if batch_index % 250 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": len(train_loader),
                            "loss": round(sum(losses[-50:]) / min(50, len(losses)), 6),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / max(1, len(losses)), 6),
            **metrics,
        }
        report_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row["val_top3"] > best_top3:
            best_top3 = row["val_top3"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "model": best_state,
            "epoch": f"{checkpoint.get('epoch', 'unknown')}-synthetic-layer4",
            "synthetic_finetune": {
                "dataset_root": str(args.dataset_root),
                "query_preprocess": args.query_preprocess,
                "include_source_crops": args.include_source_crops,
                "trainable_blocks": args.trainable_blocks,
                "train_examples": len(train_dataset),
                "val_examples": len(val_dataset),
                "best_val_top3": best_top3,
            },
        },
        args.output,
    )
    report = {
        "output": str(args.output),
        "dataset_root": str(args.dataset_root),
        "query_preprocess": args.query_preprocess,
        "include_source_crops": args.include_source_crops,
        "trainable_blocks": args.trainable_blocks,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "elapsed_sec": round(perf_counter() - start, 2),
        "best_val_top3": best_top3,
        "epochs": report_rows,
    }
    report_output = args.report_output or args.output.with_suffix(".json")
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def freeze_model(model: nn.Module, trainable_blocks: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    modules = [model.fc]
    if trainable_blocks in {"layer4", "layer3-layer4", "all"}:
        modules.append(model.layer4)
    if trainable_blocks in {"layer3-layer4", "all"}:
        modules.append(model.layer3)
    if trainable_blocks == "all":
        modules.extend([model.conv1, model.bn1, model.layer1, model.layer2])

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    top1 = top3 = top5 = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            _, indices = torch.topk(logits, k=5, dim=1)
            total += int(labels.numel())
            top1 += int((indices[:, :1] == labels[:, None]).any(dim=1).sum().item())
            top3 += int((indices[:, :3] == labels[:, None]).any(dim=1).sum().item())
            top5 += int((indices[:, :5] == labels[:, None]).any(dim=1).sum().item())
    return {
        "val_top1": round(top1 / total, 6) if total else 0.0,
        "val_top3": round(top3 / total, 6) if total else 0.0,
        "val_top5": round(top5 / total, 6) if total else 0.0,
    }


if __name__ == "__main__":
    main()
