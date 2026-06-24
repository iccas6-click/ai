from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import hf_hub_download, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "datasets" / "raw"


def download_synthetic_v3() -> Path:
    destination = RAW_ROOT / "healtheat-pill-synthetic-v3"
    archive = Path(
        hf_hub_download(
            repo_id="wony98/healtheat-pill-synthetic-v3",
            repo_type="dataset",
            filename="synthetic_v3.zip",
            local_dir=destination,
        )
    )
    extracted = destination / "extracted"
    marker = extracted / ".complete"
    if not marker.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        extract_zip_safely(archive, extracted)
        marker.touch()
    return extracted


def download_yolo_dataset() -> Path:
    destination = RAW_ROOT / "healtheat-pill-yolo"
    snapshot_download(
        repo_id="wony98/healtheat-pill-yolo",
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=["images/**", "labels/**", "pill.yaml", "README.md"],
        max_workers=8,
    )
    return destination


def extract_zip_safely(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Archive contains an unsafe path: {member.filename}")
        zip_file.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RTMDet training datasets")
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=["synthetic-v3", "yolo"],
    )
    args = parser.parse_args()

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        path = (
            download_synthetic_v3()
            if dataset == "synthetic-v3"
            else download_yolo_dataset()
        )
        print(f"{dataset}: {path}")


if __name__ == "__main__":
    main()
