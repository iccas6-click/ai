from __future__ import annotations

from functools import lru_cache

import gradio as gr
import numpy as np

from .pipeline import PillRecognitionPipeline
from .settings import PROJECT_ROOT
from .visualization import draw_detections


@lru_cache(maxsize=1)
def get_pipeline() -> PillRecognitionPipeline:
    return PillRecognitionPipeline()


def recognize(image: np.ndarray | None):
    if image is None:
        return None, [], {"error": "이미지를 업로드해 주세요."}

    pipeline = get_pipeline()
    result = pipeline.recognize(image)
    annotated = draw_detections(image, result)
    rows = []
    for detection in result.detections:
        rtmdet = detection.rtmdet_candidates[0]
        aihub = detection.aihub_candidates[0] if detection.aihub_candidates else None
        cnn = detection.cnn_candidates[0] if detection.cnn_candidates else None
        rows.append(
            [
                detection.pill_id,
                rtmdet.class_name,
                rtmdet.confidence,
                aihub.class_name if aihub else "-",
                aihub.confidence if aihub else "-",
                cnn.class_name if cnn else "-",
                cnn.confidence if cnn else "-",
                detection.status,
            ]
        )
    return annotated, rows, result.to_dict()


def build_app() -> gr.Blocks:
    sample_root = PROJECT_ROOT / "artifacts" / "samples"
    sample_paths = sorted(sample_root.rglob("*.png")) if sample_root.exists() else []

    with gr.Blocks(title="CLICK 알약 인식 Baseline") as app:
        gr.Markdown("# CLICK 알약 인식 Baseline")
        with gr.Row():
            source = gr.Image(type="numpy", label="여러 알약 사진")
            annotated = gr.Image(type="numpy", label="탐지 결과")
        if sample_paths:
            gr.Examples(
                examples=[[str(path)] for path in sample_paths],
                inputs=source,
                label=f"테스트 이미지 {len(sample_paths)}장",
            )
        run_button = gr.Button("알약 인식", variant="primary")
        table = gr.Dataframe(
            headers=[
                "번호",
                "RTMDet Top-1",
                "RTMDet confidence",
                "AI Hub Top-1",
                "AI Hub confidence",
                "GitHub CNN Top-1",
                "GitHub CNN confidence",
                "상태",
            ],
            interactive=False,
        )
        raw_result = gr.JSON(label="전체 Top-3 결과")
        run_button.click(
            fn=recognize,
            inputs=source,
            outputs=[annotated, table, raw_result],
        )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
