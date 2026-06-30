from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .pipeline import PillRecognitionPipeline
from .visualization import draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-pill recognition")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    pipeline = PillRecognitionPipeline()
    result = pipeline.recognize(image_rgb)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if args.output:
        annotated = draw_detections(image_rgb, result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(args.output),
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
        )


if __name__ == "__main__":
    main()
