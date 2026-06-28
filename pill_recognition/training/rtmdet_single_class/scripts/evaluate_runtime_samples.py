from __future__ import annotations

import json
from pathlib import Path

import cv2

from pill_recognition.pipeline import PillRecognitionPipeline
from pill_recognition.settings import Settings
from pill_recognition.visualization import draw_detections


def main() -> None:
    inference_root = Path(__file__).resolve().parents[3] / "inference"
    samples = sorted((inference_root / "artifacts" / "samples").glob("multi-pill*.png"))
    output_dir = inference_root / "outputs" / "rtmdet-single-class"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    print(f"detector: {settings.detector_checkpoint}", flush=True)
    print(f"classes: {settings.detector_class_names}", flush=True)
    print(f"aihub: {settings.aihub_weights}", flush=True)
    pipeline = PillRecognitionPipeline(settings)

    summary = []
    for path in samples:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = pipeline.recognize(image_rgb)
        annotated = draw_detections(image_rgb, result)
        cv2.imwrite(
            str(output_dir / path.name),
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
        )

        detections = []
        for detection in result.detections:
            top = detection.aihub_candidates[0] if detection.aihub_candidates else None
            detections.append(
                {
                    "pill_id": detection.pill_id,
                    "bbox": detection.bbox,
                    "detector_confidence": round(
                        detection.rtmdet_candidates[0].confidence,
                        4,
                    ),
                    "aihub_top1": top.class_name if top else None,
                    "aihub_confidence": round(top.confidence, 4) if top else None,
                }
            )

        row = {
            "image": path.name,
            "expected_count": 4,
            "detected_count": result.pill_count,
            "model_version": result.model_version,
            "detections": detections,
        }
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
