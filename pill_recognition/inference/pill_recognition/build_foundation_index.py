from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .foundation_retrieval import (
    DEFAULT_TORCHHUB_MODEL,
    DEFAULT_TORCHHUB_REPO,
    FoundationEncoderConfig,
    encode_images,
    foundation_transform,
    load_torchhub_encoder,
)
from .settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build foundation image retrieval index")
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-classes", type=int, default=None)
    parser.add_argument("--torchhub-repo", default=DEFAULT_TORCHHUB_REPO)
    parser.add_argument("--torchhub-model", default=DEFAULT_TORCHHUB_MODEL)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--index-mode",
        choices=["prototype", "reference"],
        default="prototype",
        help="prototype averages embeddings per class; reference stores every sampled reference image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    if settings.aihub_mapping is None:
        raise SystemExit("AI Hub mapping is not configured")
    crop_root = settings.aihub_mapping.parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or settings.device)
    encoder_config = FoundationEncoderConfig(
        torchhub_repo=args.torchhub_repo,
        torchhub_model=args.torchhub_model,
        image_size=args.image_size,
    )

    encoder = load_torchhub_encoder(args.torchhub_repo, args.torchhub_model).to(device).eval()
    transform = foundation_transform(args.image_size)

    pill_dirs = sorted(
        path for path in crop_root.iterdir() if path.is_dir() and path.name.startswith("K-")
    )
    if args.limit_classes:
        pill_dirs = pill_dirs[: args.limit_classes]

    pill_ids = []
    index_embeddings = []
    manifest = []
    for class_index, pill_dir in enumerate(pill_dirs, start=1):
        image_paths = sorted(pill_dir.glob("*.png"))[: args.samples_per_class]
        if not image_paths:
            continue
        embeddings = []
        for batch_paths in chunks(image_paths, args.batch_size):
            batch = []
            valid_paths = []
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                except OSError:
                    continue
                batch.append(transform(image))
                valid_paths.append(path)
            if not batch:
                continue
            tensor = torch.stack(batch).to(device)
            with torch.inference_mode():
                features = encode_images(encoder, tensor)
                features = torch.nn.functional.normalize(features, dim=1)
            cpu_features = features.cpu()
            embeddings.append(cpu_features)
            manifest.extend(str(path) for path in valid_paths)
            if args.index_mode == "reference":
                index_embeddings.extend(feature for feature in cpu_features)
                pill_ids.extend(pill_dir.name for _ in valid_paths)
        if not embeddings:
            continue
        if args.index_mode == "prototype":
            prototype = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
            prototype = torch.nn.functional.normalize(prototype, dim=1)
            pill_ids.append(pill_dir.name)
            index_embeddings.append(prototype.squeeze(0))
        if class_index % 50 == 0:
            print(f"indexed {class_index}/{len(pill_dirs)} classes")

    payload = {
        "version": 1,
        "encoder": encoder_config.name,
        "index_mode": args.index_mode,
        "samples_per_class": args.samples_per_class,
        "pill_ids": pill_ids,
        "embeddings": torch.stack(index_embeddings).half(),
        "reference_count": len(manifest),
        "class_count": len(set(pill_ids)),
        "image_size": args.image_size,
    }
    torch.save(payload, args.output)
    print(
        f"saved {args.output} with {payload['class_count']} classes, "
        f"{len(pill_ids)} index embeddings "
        f"from {payload['reference_count']} reference images"
    )


def chunks(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


if __name__ == "__main__":
    main()
