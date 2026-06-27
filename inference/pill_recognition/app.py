from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

from .pipeline import PillRecognitionPipeline
from .product_search import ProductSearchQuery, load_product_index, search_products
from .settings import PROJECT_ROOT, Settings
from .visualization import draw_detections


DETECTOR_EVALUATION_RUNS = {
    "synthetic-v3 val 800": PROJECT_ROOT / "outputs" / "evaluation" / "rtmdet-val-800",
    "AIHub synthetic max10 val 1000": PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "rtmdet-aihub-synthetic-max10-val",
    "AIHub realistic max10 val 1000": PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "rtmdet-aihub-synthetic-realistic-max10-val",
}
DEFAULT_DETECTOR_EVALUATION = "AIHub realistic max10 val 1000"


@lru_cache(maxsize=1)
def get_pipeline() -> PillRecognitionPipeline:
    return PillRecognitionPipeline()


@lru_cache(maxsize=1)
def get_product_index() -> dict:
    return load_product_index(Settings.from_env().aihub_mapping)


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
                format_bbox(detection.bbox),
                f"{rtmdet.confidence:.3f}",
                format_candidates(detection.aihub_candidates),
                f"{cnn.class_name} ({cnn.confidence:.3f})" if cnn else "-",
                detection.status,
            ]
        )
    return annotated, rows, result.to_dict()


def search_product_db(imprint: str, shape: str, color: str, text: str):
    query = ProductSearchQuery(
        imprint=imprint or "",
        shape=shape or "",
        color=color or "",
        text=text or "",
        limit=30,
    )
    results = search_products(get_product_index(), query)
    rows = [
        [
            row.get("score"),
            row.get("pill_id"),
            row.get("product_name") or "-",
            format_ingredient(row.get("ingredient") or ""),
            row.get("print_front") or "-",
            row.get("print_back") or "-",
            row.get("drug_shape") or "-",
            joined_colors(row),
            row.get("company") or "-",
            row.get("item_seq") or "-",
            row.get("etc_otc_code") or "-",
            row.get("matched") or "-",
        ]
        for row in results
    ]
    return rows, results


def joined_colors(row: dict) -> str:
    colors = [row.get("color_class1"), row.get("color_class2")]
    colors = [color for color in colors if color]
    return "/".join(colors) if colors else "-"


def format_bbox(bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    return f"{x1},{y1},{x2},{y2}"


def format_candidates(candidates) -> str:
    if not candidates:
        return "-"
    return "\n".join(
        f"{candidate.rank}. {format_candidate_identity(candidate)} ({candidate.confidence:.3f})"
        for candidate in candidates
    )


def format_candidate_identity(candidate) -> str:
    parts = [candidate.class_name]
    if candidate.product_name:
        parts.append(candidate.product_name)
    if candidate.ingredient:
        parts.append(f"성분: {format_ingredient(candidate.ingredient)}")
    for value in (candidate.company, candidate.item_seq, candidate.etc_otc_code):
        if value:
            parts.append(str(value))
    return " | ".join(parts)


def format_ingredient(value: str) -> str:
    return ", ".join(part.strip() for part in str(value).split("|") if part.strip())


def load_detector_eval_summary(run_name: str = DEFAULT_DETECTOR_EVALUATION) -> dict:
    summary_path = detector_eval_root(run_name) / "summary.json"
    if not summary_path.exists():
        return {"message": "RTMDet 평가 결과가 아직 없습니다."}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_detector_eval_images(
    run_name: str = DEFAULT_DETECTOR_EVALUATION,
    limit: int = 120,
) -> list[str]:
    annotated_dir = detector_eval_root(run_name) / "annotated"
    if not annotated_dir.exists():
        return []
    rows = load_detector_eval_rows(run_name)
    error_names = [
        row["image"]
        for row in rows
        if row.get("false_positive", 0) or row.get("false_negative", 0)
    ]
    ordered_names = error_names + [
        row["image"] for row in rows if row["image"] not in set(error_names)
    ]
    paths = [annotated_dir / name for name in ordered_names]
    return [str(path) for path in paths if path.exists()][:limit]


def load_detector_eval_rows(run_name: str = DEFAULT_DETECTOR_EVALUATION) -> list[dict]:
    result_path = detector_eval_root(run_name) / "results.json"
    if not result_path.exists():
        return []
    return json.loads(result_path.read_text(encoding="utf-8"))


def detector_eval_root(run_name: str) -> Path:
    return DETECTOR_EVALUATION_RUNS.get(
        run_name,
        DETECTOR_EVALUATION_RUNS[DEFAULT_DETECTOR_EVALUATION],
    )


def build_app() -> gr.Blocks:
    sample_root = PROJECT_ROOT / "artifacts" / "samples"
    sample_paths = (
        sorted(
            path
            for path in sample_root.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if sample_root.exists()
        else []
    )

    with gr.Blocks(title="CLICK 알약 인식 Baseline") as app:
        with gr.Tab("인식 데모"):
            gr.Markdown(
                "# CLICK 알약 인식 Baseline\n"
                "RTMDet 단일 클래스 탐지로 알약 위치를 잡고, 각 crop을 AI Hub ResNet152 1,000종 분류기에 넣어 Top-3 후보를 반환합니다."
            )
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
                    "BBox x1,y1,x2,y2",
                    "탐지 confidence",
                    "AI Hub Top-3 제품/성분 후보",
                    "GitHub CNN 옵션",
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

        with gr.Tab("제품 DB 검색"):
            gr.Markdown(
                "# AI Hub 1000종 제품 DB 검색\n"
                "사진 분류가 흔들릴 때 각인, 색, 모양, 제품명/성분 일부를 입력해 후보를 좁힙니다."
            )
            with gr.Row():
                imprint_input = gr.Textbox(label="각인 문자", placeholder="예: CKD, 500, AC")
                shape_input = gr.Dropdown(
                    choices=["", "원형", "타원형", "장방형", "반원형", "삼각형", "사각형", "기타"],
                    value="",
                    label="모양",
                )
                color_input = gr.Dropdown(
                    choices=[
                        "",
                        "하양",
                        "노랑",
                        "주황",
                        "분홍",
                        "빨강",
                        "갈색",
                        "초록",
                        "파랑",
                        "보라",
                        "회색",
                        "검정",
                        "투명",
                    ],
                    value="",
                    label="색",
                )
            text_input = gr.Textbox(
                label="제품명/성분/업체/품목기준코드",
                placeholder="예: 메트포르민, 트윈스타, 201005083",
            )
            search_button = gr.Button("DB 후보 검색", variant="primary")
            search_table = gr.Dataframe(
                headers=[
                    "점수",
                    "K-ID",
                    "제품명",
                    "성분",
                    "앞면 각인",
                    "뒷면 각인",
                    "모양",
                    "색",
                    "업체",
                    "품목기준코드",
                    "일반/전문",
                    "매칭 근거",
                ],
                interactive=False,
            )
            search_json = gr.JSON(label="검색 raw 결과")
            search_button.click(
                fn=search_product_db,
                inputs=[imprint_input, shape_input, color_input, text_input],
                outputs=[search_table, search_json],
            )

        with gr.Tab("RTMDet 평가"):
            gr.Markdown(
                "# RTMDet Validation\n"
                "초록/빨강은 정답 박스, 주황/자홍은 예측 박스입니다. 빨강과 자홍은 매칭 실패 케이스입니다."
            )
            detector_run = gr.Dropdown(
                choices=list(DETECTOR_EVALUATION_RUNS),
                value=DEFAULT_DETECTOR_EVALUATION,
                label="평가셋",
            )
            refresh_button = gr.Button("평가 결과 새로고침")
            detector_summary = gr.JSON(
                value=lambda: load_detector_eval_summary(DEFAULT_DETECTOR_EVALUATION),
                label="summary.json",
            )
            detector_gallery = gr.Gallery(
                value=lambda: load_detector_eval_images(DEFAULT_DETECTOR_EVALUATION),
                label="Annotated validation images",
                columns=3,
                height=720,
                show_label=True,
            )
            refresh_button.click(
                fn=lambda run_name: (
                    load_detector_eval_summary(run_name),
                    load_detector_eval_images(run_name),
                ),
                inputs=detector_run,
                outputs=[detector_summary, detector_gallery],
            )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
