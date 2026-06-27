from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .settings import PROJECT_ROOT


DATASET_REPO_ID = "wony98/healtheat-pill-yolo"
DATASET_REVISION = "970f9dc9b38886512868b184d965d81573bef4fb"
VAL_PATH = "images/val"
ANGLE_PATTERN = re.compile(r"_(70|75|90)_000_200\.png$")


def list_validation_images() -> list[str]:
    api = HfApi()
    return sorted(
        entry.path
        for entry in api.list_repo_tree(
            DATASET_REPO_ID,
            path_in_repo=VAL_PATH,
            repo_type="dataset",
            revision=DATASET_REVISION,
            recursive=False,
        )
        if entry.path.endswith(".png")
    )


def select_diverse_images(paths: list[str], count: int, seed: int) -> list[str]:
    randomizer = random.Random(seed)
    buckets: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        match = ANGLE_PATTERN.search(path)
        buckets[match.group(1) if match else "other"].append(path)

    selected = []
    used_combinations = set()
    bucket_names = sorted(buckets)
    for bucket in buckets.values():
        randomizer.shuffle(bucket)

    while len(selected) < min(count, len(paths)):
        added = False
        for bucket_name in bucket_names:
            bucket = buckets[bucket_name]
            while bucket:
                path = bucket.pop()
                combination = ANGLE_PATTERN.sub("_ANGLE_000_200.png", path)
                if combination in used_combinations:
                    continue
                used_combinations.add(combination)
                selected.append(path)
                added = True
                break
            if len(selected) >= count:
                break
        if not added:
            break

    return selected


def download(paths: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        print(f"[{index:03d}/{total:03d}] {path}")
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=path,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=destination,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pill validation images")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "samples" / "diverse",
    )
    args = parser.parse_args()

    paths = list_validation_images()
    selected = paths if args.all else select_diverse_images(paths, args.count, args.seed)
    print(f"Validation images: {len(paths)}, selected: {len(selected)}")
    download(selected, args.destination)


if __name__ == "__main__":
    main()
