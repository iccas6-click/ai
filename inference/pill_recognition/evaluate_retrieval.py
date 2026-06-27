from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .retrieval import AIHubResNetRetriever
from .settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AIHub retrieval on held-out crops")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--offset", type=int, default=64)
    parser.add_argument("--limit-classes", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    retriever = AIHubResNetRetriever.from_settings(settings)
    crop_root = settings.aihub_mapping.parent
    pill_dirs = sorted(
        path for path in crop_root.iterdir() if path.is_dir() and path.name.startswith("K-")
    )
    if args.limit_classes:
        pill_dirs = pill_dirs[: args.limit_classes]

    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    rows = []
    for class_index, pill_dir in enumerate(pill_dirs, start=1):
        paths = sorted(pill_dir.glob("*.png"))[
            args.offset : args.offset + args.samples_per_class
        ]
        if not paths:
            continue
        crops = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
        predictions = retriever.predict_batch(crops, max(args.top_k, 5))
        for path, candidates in zip(paths, predictions):
            total += 1
            predicted_ids = [candidate.pill_id for candidate in candidates]
            expected = pill_dir.name
            top1 += int(predicted_ids[:1] == [expected])
            top3 += int(expected in predicted_ids[:3])
            top5 += int(expected in predicted_ids[:5])
            rows.append(
                {
                    "image": str(path),
                    "expected": expected,
                    "predicted": predicted_ids[: args.top_k],
                }
            )
        if class_index % 50 == 0:
            print(f"evaluated {class_index}/{len(pill_dirs)} classes")

    summary = {
        "total": total,
        "top1": top1 / total if total else 0,
        "top3": top3 / total if total else 0,
        "top5": top5 / total if total else 0,
        "samples_per_class": args.samples_per_class,
        "offset": args.offset,
        "retrieval_query_preprocess": settings.retrieval_query_preprocess,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
