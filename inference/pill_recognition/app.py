from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from .pipeline import PillRecognitionPipeline
from .product_db import ProductSearchQuery, load_product_index, search_products
from .scope import parse_allowed_pill_ids
from .settings import Settings
from .visualization import draw_detections
from pill_recognition_legacy.aihub_classifier import AIHubPillClassifier
from pill_recognition_legacy.schemas import Candidate


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_pipeline() -> PillRecognitionPipeline:
    return PillRecognitionPipeline(get_settings())


@lru_cache(maxsize=1)
def get_product_index() -> dict:
    return load_product_index(get_settings().aihub_mapping)


@lru_cache(maxsize=1)
def get_aihub_classifier() -> AIHubPillClassifier:
    settings = get_settings()
    if settings.aihub_weights is None or settings.aihub_mapping is None:
        raise FileNotFoundError("AIHub weights or mapping path is not configured.")
    return AIHubPillClassifier(
        settings.aihub_weights,
        settings.aihub_mapping,
        settings.device,
    )


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


def aihub_predict_upload(image: np.ndarray | None, top_k: int = 5):
    top_k = clamp_int(top_k, 1, 20)
    if image is None:
        return [], {"error": "AIHub crop 이미지를 업로드해 주세요."}
    classifier = get_aihub_classifier()
    candidates = classifier.predict_batch([ensure_rgb_uint8(image)], top_k=top_k)[0]
    return aihub_candidate_rows(candidates), {
        "model_version": classifier.model_version,
        "top_k": top_k,
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def load_aihub_dataset_sample(pill_id: str, sample_index: int = 0):
    pill_id = (pill_id or "").strip()
    if not pill_id:
        return None, {"error": "K-ID를 입력해 주세요. 예: K-004378"}
    pill_dir = get_aihub_crop_root() / pill_id
    if not pill_dir.is_dir():
        return None, {"error": f"AIHub crop directory not found: {pill_id}"}
    paths = aihub_crop_paths(pill_dir)
    if not paths:
        return None, {"error": f"No crop images found for {pill_id}"}
    index = clamp_int(sample_index, 0, len(paths) - 1)
    product = get_product_index().get(pill_id)
    return np.asarray(Image.open(paths[index]).convert("RGB")), {
        "pill_id": pill_id,
        "sample_index": index,
        "sample_count": len(paths),
        "image_path": str(paths[index]),
        "product": asdict(product) if product else None,
    }


def evaluate_aihub_classifier_dataset(
    samples_per_class: int = 1,
    offset: int = 64,
    limit_classes: int = 0,
    top_k: int = 5,
    batch_size: int = 32,
):
    samples_per_class = clamp_int(samples_per_class, 1, 100)
    offset = clamp_int(offset, 0, 1_000_000)
    limit_classes = max(0, int(limit_classes or 0))
    top_k = clamp_int(top_k, 1, 20)
    batch_size = clamp_int(batch_size, 1, 128)
    classifier = get_aihub_classifier()
    pill_dirs = sorted(
        path
        for path in get_aihub_crop_root().iterdir()
        if path.is_dir() and path.name.startswith("K-")
    )
    if limit_classes:
        pill_dirs = pill_dirs[:limit_classes]

    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    misses = []
    pending_images = []
    pending_expected = []
    pending_paths = []

    def flush_batch():
        nonlocal total, top1, top3, top5, misses, pending_images, pending_expected, pending_paths
        if not pending_images:
            return
        predictions = classifier.predict_batch(pending_images, top_k=max(top_k, 5))
        for image_path, expected, candidates in zip(
            pending_paths,
            pending_expected,
            predictions,
        ):
            predicted_ids = [candidate.class_name for candidate in candidates]
            total += 1
            top1 += int(predicted_ids[:1] == [expected])
            top3_hit = expected in predicted_ids[:3]
            top3 += int(top3_hit)
            top5 += int(expected in predicted_ids[:5])
            if not top3_hit and len(misses) < 200:
                misses.append(
                    [
                        str(image_path),
                        expected,
                        format_aihub_candidates(candidates[:top_k]),
                    ]
                )
        pending_images = []
        pending_expected = []
        pending_paths = []

    for pill_dir in pill_dirs:
        for path in aihub_crop_paths(pill_dir)[offset : offset + samples_per_class]:
            pending_images.append(np.asarray(Image.open(path).convert("RGB")))
            pending_expected.append(pill_dir.name)
            pending_paths.append(path)
            if len(pending_images) >= batch_size:
                flush_batch()
    flush_batch()

    return {
        "model_version": classifier.model_version,
        "crop_root": str(get_aihub_crop_root()),
        "classes": len(pill_dirs),
        "total": total,
        "samples_per_class": samples_per_class,
        "offset": offset,
        "top_k": top_k,
        "top1": safe_rate(top1, total),
        "top3": safe_rate(top3, total),
        "top5": safe_rate(top5, total),
        "top1_count": top1,
        "top3_count": top3,
        "top5_count": top5,
    }, misses


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


def aihub_candidate_rows(candidates: list[Candidate]) -> list[list]:
    return [
        [
            candidate.rank,
            candidate.class_name,
            candidate.class_id,
            round(candidate.confidence * 100, 4),
            candidate.product_name or "-",
            format_ingredient(candidate.ingredient or ""),
            candidate.print_front or "-",
            candidate.print_back or "-",
            candidate.drug_shape or "-",
            "/".join(
                color
                for color in [candidate.color_class1, candidate.color_class2]
                if color
            )
            or "-",
            candidate.company or "-",
            candidate.item_seq or "-",
            candidate.etc_otc_code or "-",
        ]
        for candidate in candidates
    ]


def format_aihub_candidates(candidates: list[Candidate]) -> str:
    return "\n".join(
        f"{candidate.rank}. {candidate.class_name} | {candidate.product_name or '-'} | "
        f"{format_ingredient(candidate.ingredient or '') or '-'} | "
        f"{candidate.confidence * 100:.2f}%"
        for candidate in candidates
    )


def get_aihub_crop_root() -> Path:
    settings = get_settings()
    if settings.aihub_mapping is None:
        raise FileNotFoundError("AIHub mapping path is not configured.")
    return settings.aihub_mapping.parent


def aihub_crop_paths(pill_dir: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        path
        for path in pill_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected RGB image")
    return np.clip(array[:, :, :3], 0, 255).astype(np.uint8)


def clamp_int(value, minimum: int, maximum: int) -> int:
    return max(minimum, min(int(value or minimum), maximum))


def safe_rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


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

        with gr.Tab("AIHub 공식 모델 검증"):
            gr.Markdown(
                "# AIHub ResNet152 1000종 단독 검증\n"
                "RTMDet와 retrieval index를 거치지 않고 AIHub 제공 `.pt` classifier를 crop 이미지에 바로 적용합니다."
            )
            with gr.Row():
                aihub_source = gr.Image(type="numpy", label="단일 알약 crop")
                with gr.Column():
                    aihub_top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=5,
                        label="Top-K",
                    )
                    aihub_predict_button = gr.Button("공식 pt로 예측", variant="primary")
            aihub_prediction_table = gr.Dataframe(
                headers=[
                    "순위",
                    "K-ID",
                    "class_id",
                    "확률(%)",
                    "제품명",
                    "성분",
                    "앞면 각인",
                    "뒷면 각인",
                    "모양",
                    "색",
                    "업체",
                    "품목기준코드",
                    "일반/전문",
                ],
                interactive=False,
            )
            aihub_prediction_json = gr.JSON(label="공식 pt 예측 raw 결과")
            aihub_predict_button.click(
                fn=aihub_predict_upload,
                inputs=[aihub_source, aihub_top_k],
                outputs=[aihub_prediction_table, aihub_prediction_json],
            )

            gr.Markdown("## AIHub 1000종 crop 샘플 불러오기")
            with gr.Row():
                sample_pill_id = gr.Textbox(
                    label="정답 K-ID",
                    value="K-004378",
                    placeholder="예: K-004378",
                )
                sample_index = gr.Number(label="샘플 index", value=64, precision=0)
                sample_button = gr.Button("샘플 로드")
            sample_info = gr.JSON(label="샘플 정보")
            sample_button.click(
                fn=load_aihub_dataset_sample,
                inputs=[sample_pill_id, sample_index],
                outputs=[aihub_source, sample_info],
            )

            gr.Markdown("## AIHub 1000종 crop subset 평가")
            with gr.Row():
                eval_samples_per_class = gr.Number(
                    label="클래스당 샘플 수",
                    value=1,
                    precision=0,
                )
                eval_offset = gr.Number(
                    label="각 클래스 시작 offset",
                    value=64,
                    precision=0,
                )
                eval_limit_classes = gr.Number(
                    label="평가 클래스 수(0=1000 전체)",
                    value=0,
                    precision=0,
                )
            with gr.Row():
                eval_top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=5,
                    label="평가 Top-K",
                )
                eval_batch_size = gr.Number(label="batch size", value=32, precision=0)
                eval_button = gr.Button("공식 pt 평가 실행", variant="primary")
            eval_summary = gr.JSON(label="평가 summary")
            eval_misses = gr.Dataframe(
                headers=["이미지 경로", "정답 K-ID", "Top-K 예측"],
                interactive=False,
            )
            eval_button.click(
                fn=evaluate_aihub_classifier_dataset,
                inputs=[
                    eval_samples_per_class,
                    eval_offset,
                    eval_limit_classes,
                    eval_top_k,
                    eval_batch_size,
                ],
                outputs=[eval_summary, eval_misses],
            )
    return app


def warmup() -> None:
    if get_settings().warmup_on_startup:
        get_pipeline().warmup(load_detector=True)


if __name__ == "__main__":
    warmup()
    build_app().launch(server_name="127.0.0.1", server_port=7860)
