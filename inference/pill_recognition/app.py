from __future__ import annotations

from functools import lru_cache

import gradio as gr
import numpy as np

from .pipeline import PillRecognitionPipeline
from .product_db import ProductSearchQuery, load_product_index, search_products
from .scope import parse_allowed_pill_ids
from .settings import Settings
from .visualization import draw_detections


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_pipeline() -> PillRecognitionPipeline:
    return PillRecognitionPipeline(get_settings())


@lru_cache(maxsize=1)
def get_product_index() -> dict:
    return load_product_index(get_settings().aihub_mapping)


def recognize(image: np.ndarray | None, allowed_pill_ids_text: str = ""):
    if image is None:
        return None, [], {"error": "이미지를 업로드해 주세요."}

    allowed_pill_ids = parse_allowed_pill_ids(allowed_pill_ids_text)
    result = get_pipeline().recognize(image, allowed_pill_ids=allowed_pill_ids)
    annotated = draw_detections(image, result)
    rows = []
    for detection in result.detections:
        rows.append(
            [
                detection.pill_id,
                format_bbox(detection.bbox),
                f"{detection.detector_confidence:.3f}",
                format_candidates(detection.candidates),
                detection.status,
                detection.status_reason or "-",
            ]
        )
    return annotated, rows, result.to_dict()


def search_product_db(imprint: str, shape: str, color: str, text: str):
    results = search_products(
        get_product_index(),
        ProductSearchQuery(
            imprint=imprint or "",
            shape=shape or "",
            color=color or "",
            text=text or "",
            limit=30,
        ),
    )
    return product_rows(results), results


def product_rows(results: list[dict]) -> list[list]:
    return [
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


def format_bbox(bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    return f"{x1},{y1},{x2},{y2}"


def format_candidates(candidates) -> str:
    if not candidates:
        return "-"
    return "\n".join(
        f"{candidate.rank}. 제품명: {candidate.product_name or '-'} | "
        f"성분: {format_ingredient(candidate.ingredient or '') or '-'} | "
        f"주의: {format_cautions(candidate.caution_points)} | "
        f"점수 {candidate.score}"
        for candidate in candidates
    )


def format_ingredient(value: str) -> str:
    return ", ".join(part.strip() for part in str(value).split("|") if part.strip())


def format_cautions(values: list[str]) -> str:
    return "; ".join(value.strip() for value in values if value.strip()) or "-"


def joined_colors(row: dict) -> str:
    colors = [row.get("color_class1"), row.get("color_class2")]
    colors = [color for color in colors if color]
    return "/".join(colors) if colors else "-"


def build_app() -> gr.Blocks:
    settings = get_settings()
    with gr.Blocks(title="CLICK 알약 인식 v2") as app:
        with gr.Tab("인식 데모"):
            gr.Markdown(
                "# CLICK 알약 인식 v2\n"
                f"RTMDet로 알약 위치를 찾고, `{settings.recognizer}` recognizer가 제품 후보 Top-{settings.top_k}를 반환합니다."
            )
            with gr.Row():
                source = gr.Image(type="numpy", label="여러 알약 사진")
                annotated = gr.Image(type="numpy", label="탐지 결과")
            allowed_pill_ids_input = gr.Textbox(
                label="복약목록 K-ID scope",
                placeholder='예: ["K-000059","K-000069"] 또는 K-000059,K-000069',
            )
            run_button = gr.Button("알약 후보 찾기", variant="primary")
            table = gr.Dataframe(
                headers=[
                    "번호",
                    "BBox x1,y1,x2,y2",
                    "탐지 confidence",
                    "제품명/성분 후보",
                    "상태",
                    "상태 이유",
                ],
                interactive=False,
            )
            raw_result = gr.JSON(label="전체 결과")
            run_button.click(
                fn=recognize,
                inputs=[source, allowed_pill_ids_input],
                outputs=[annotated, table, raw_result],
            )

        with gr.Tab("제품 DB 검색"):
            gr.Markdown("# AI Hub 1000종 제품 DB 검색")
            with gr.Row():
                imprint_input = gr.Textbox(label="각인 문자", placeholder="예: CKD, W2, D5")
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
                placeholder="예: 아시클로버, 와르파린, 201005083",
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
    return app


def warmup() -> None:
    if get_settings().warmup_on_startup:
        get_pipeline().warmup(load_detector=True)


if __name__ == "__main__":
    warmup()
    build_app().launch(server_name="127.0.0.1", server_port=7860)
