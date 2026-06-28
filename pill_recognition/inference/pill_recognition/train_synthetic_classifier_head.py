from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

from .query_preprocess import preprocess_query_crop
from .settings import Settings
from pill_recognition_legacy.aihub_classifier import load_aihub_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune only the AIHub ResNet152 classifier head on synthetic scene crops."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/processed/rtmdet-aihub-synthetic-realistic-max10-v2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--query-preprocess", default="grabcut_dark")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-val-images", type=int)
    parser.add_argument(
        "--include-source-crops",
        action="store_true",
        help="Also train on the original AIHub crop referenced by each synthetic pill.",
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
    initial_fc = nn.Linear(model.fc.in_features, output_classes)
    initial_fc.load_state_dict(model.fc.state_dict())
    model.fc = nn.Identity()
    model.to(device).eval()

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

    start = perf_counter()
    train_features, train_labels = extract_split_features(
        args.dataset_root,
        "train",
        class_ids_by_name,
        model,
        transform,
        device,
        args.query_preprocess,
        args.feature_batch_size,
        limit_images=args.max_train_images,
        include_source_crops=args.include_source_crops,
    )
    val_features, val_labels = extract_split_features(
        args.dataset_root,
        "val",
        class_ids_by_name,
        model,
        transform,
        device,
        args.query_preprocess,
        args.feature_batch_size,
        limit_images=args.max_val_images,
        include_source_crops=False,
    )

    head = initial_fc.to(device)
    train_report = train_head(
        head,
        train_features,
        train_labels,
        val_features,
        val_labels,
        device,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    tuned_state = dict(state_dict)
    tuned_state["fc.weight"] = head.weight.detach().cpu()
    tuned_state["fc.bias"] = head.bias.detach().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "model": tuned_state,
            "epoch": f"{checkpoint.get('epoch', 'unknown')}-synthetic-head",
            "synthetic_head": {
                "dataset_root": str(args.dataset_root),
                "query_preprocess": args.query_preprocess,
                "include_source_crops": args.include_source_crops,
                "train_examples": int(train_labels.numel()),
                "val_examples": int(val_labels.numel()),
            },
        },
        args.output,
    )

    report = {
        "output": str(args.output),
        "dataset_root": str(args.dataset_root),
        "query_preprocess": args.query_preprocess,
        "include_source_crops": args.include_source_crops,
        "train_examples": int(train_labels.numel()),
        "val_examples": int(val_labels.numel()),
        "elapsed_sec": round(perf_counter() - start, 2),
        "epochs": train_report,
    }
    report_output = args.report_output or args.output.with_suffix(".json")
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def extract_split_features(
    dataset_root: Path,
    split: str,
    class_ids_by_name: dict[str, int],
    model: nn.Module,
    transform,
    device: torch.device,
    query_preprocess: str,
    batch_size: int,
    limit_images: int | None = None,
    include_source_crops: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    pending_images = []
    pending_labels = []
    features = []
    labels = []
    for index, image_path in enumerate(image_paths, start=1):
        metadata_path = metadata_dir / f"{image_path.stem}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        image = np.asarray(Image.open(image_path).convert("RGB"))
        for pill in metadata.get("pills", []):
            class_id = class_ids_by_name.get(str(pill.get("class_name")))
            if class_id is None:
                continue
            x1, y1, x2, y2 = map(int, pill["bbox_xyxy"])
            crop = image[y1:y2, x1:x2]
            pending_images.append(preprocess_query_crop(crop, query_preprocess))
            pending_labels.append(class_id)
            if include_source_crops and pill.get("source_image"):
                source_path = Path(pill["source_image"])
                if source_path.exists():
                    pending_images.append(np.asarray(Image.open(source_path).convert("RGB")))
                    pending_labels.append(class_id)
            if len(pending_images) >= batch_size:
                flush_feature_batch(
                    pending_images,
                    pending_labels,
                    model,
                    transform,
                    device,
                    features,
                    labels,
                )
                pending_images = []
                pending_labels = []
        if index % 250 == 0:
            print(f"{split}: prepared {index}/{len(image_paths)} images", flush=True)

    flush_feature_batch(
        pending_images,
        pending_labels,
        model,
        transform,
        device,
        features,
        labels,
    )
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def flush_feature_batch(
    images: list[np.ndarray],
    labels_in: list[int],
    model: nn.Module,
    transform,
    device: torch.device,
    features_out: list[torch.Tensor],
    labels_out: list[torch.Tensor],
) -> None:
    if not images:
        return
    batch = torch.stack(
        [transform(Image.fromarray(image).convert("RGB")) for image in images]
    ).to(device)
    with torch.inference_mode():
        features = model(batch).detach().cpu().to(torch.float32)
    features_out.append(features)
    labels_out.append(torch.tensor(labels_in, dtype=torch.long))


def train_head(
    head: nn.Linear,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> list[dict]:
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    report = []
    for epoch in range(1, epochs + 1):
        head.train()
        losses = []
        for feature_batch, label_batch in train_loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(feature_batch)
            loss = loss_fn(logits, label_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate_head(head, val_features, val_labels, device)
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / max(1, len(losses)), 6),
            **metrics,
        }
        report.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return report


def evaluate_head(
    head: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    head.eval()
    top1 = top3 = top5 = total = 0
    with torch.inference_mode():
        for start in range(0, labels.numel(), batch_size):
            feature_batch = features[start : start + batch_size].to(device)
            label_batch = labels[start : start + batch_size].to(device)
            logits = head(feature_batch)
            _, indices = torch.topk(logits, k=5, dim=1)
            total += int(label_batch.numel())
            top1 += int((indices[:, :1] == label_batch[:, None]).any(dim=1).sum().item())
            top3 += int((indices[:, :3] == label_batch[:, None]).any(dim=1).sum().item())
            top5 += int((indices[:, :5] == label_batch[:, None]).any(dim=1).sum().item())
    return {
        "val_top1": round(top1 / total, 6) if total else 0.0,
        "val_top3": round(top3 / total, 6) if total else 0.0,
        "val_top5": round(top5 / total, 6) if total else 0.0,
    }


if __name__ == "__main__":
    main()
