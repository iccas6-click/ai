from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json
import os
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MULTI_PILL_DATASET = "rtmdet-aihub-synthetic-realistic-max10-v2"
PREFERRED_MULTI_PILL_DATASETS = [
    "rtmdet-aihub-synthetic-realistic-clean-v3-pilot",
    "rtmdet-aihub-synthetic-realistic-max10-v2",
]


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


@lru_cache(maxsize=1)
def multi_pill_dataset_choices() -> list[str]:
    processed_root = REPO_ROOT / "datasets" / "processed"
    if not processed_root.is_dir():
        return []
    datasets = [
        path.name
        for path in processed_root.iterdir()
        if (path / "images" / "val").is_dir()
        and (path / "metadata" / "val").is_dir()
    ]
    preferred = [name for name in PREFERRED_MULTI_PILL_DATASETS if name in datasets]
    remaining = sorted(name for name in datasets if name not in preferred)
    return preferred + remaining


def default_multi_pill_dataset() -> str:
    choices = multi_pill_dataset_choices()
    return choices[0] if choices else DEFAULT_MULTI_PILL_DATASET


@lru_cache(maxsize=16)
def multi_pill_sample_choices(dataset_name: str) -> list[str]:
    image_dir = multi_pill_dataset_root(dataset_name) / "images" / "val"
    if not image_dir.is_dir():
        return []
    choices = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        metadata = load_multi_pill_metadata(dataset_name, image_path.stem)
        pill_count = metadata.get("pill_count")
        pills = metadata.get("pills") or []
        names = ", ".join(
            str(pill.get("product_name") or pill.get("class_name") or "-")
            for pill in pills[:3]
        )
        suffix = f"{pill_count}알" if pill_count else "?알"
        if len(pills) > 3:
            names = f"{names} 외 {len(pills) - 3}개"
        choices.append(f"{image_path.stem} | {suffix} | {names}")
    return choices


def multi_pill_dataset_root(dataset_name: str | None = None) -> Path:
    dataset = (dataset_name or default_multi_pill_dataset()).strip()
    return REPO_ROOT / "datasets" / "processed" / dataset


def load_multi_pill_metadata(dataset_name: str, sample_stem: str) -> dict:
    metadata_path = (
        multi_pill_dataset_root(dataset_name)
        / "metadata"
        / "val"
        / f"{sample_stem}.json"
    )
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_multi_pill_sample_path(dataset_name: str, selection: str) -> Path:
    sample_stem = (selection or "").split("|", 1)[0].strip()
    image_dir = multi_pill_dataset_root(dataset_name) / "images" / "val"
    for suffix in [".jpg", ".jpeg", ".png", ".webp"]:
        image_path = image_dir / f"{sample_stem}{suffix}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Multi-pill sample not found: {sample_stem}")


def update_multi_pill_samples(dataset_name: str):
    choices = multi_pill_sample_choices(dataset_name)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def load_multi_pill_sample(dataset_name: str, selection: str):
    if not selection:
        return None, [], {"error": "샘플을 선택해 주세요."}
    image_path = resolve_multi_pill_sample_path(dataset_name, selection)
    metadata = load_multi_pill_metadata(dataset_name, image_path.stem)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    return image, multi_pill_ground_truth_rows(metadata), {
        "dataset": dataset_name,
        "image_path": str(image_path),
        "metadata_path": str(
            multi_pill_dataset_root(dataset_name)
            / "metadata"
            / "val"
            / f"{image_path.stem}.json"
        ),
        "pill_count": metadata.get("pill_count"),
        "background": metadata.get("background"),
        "ground_truth": metadata.get("pills", []),
    }


def recognize_multi_pill_sample(
    dataset_name: str,
    selection: str,
    allowed_pill_ids_text: str = "",
):
    image, ground_truth, sample_info = load_multi_pill_sample(dataset_name, selection)
    if image is None:
        return None, ground_truth, [], sample_info, sample_info
    annotated, prediction_rows, raw_result = recognize(image, allowed_pill_ids_text)
    return annotated, ground_truth, prediction_rows, raw_result, sample_info


def multi_pill_ground_truth_rows(metadata: dict) -> list[list]:
    rows = []
    for pill in metadata.get("pills", []) or []:
        rows.append(
            [
                pill.get("pill_id"),
                pill.get("class_name"),
                pill.get("product_name") or "-",
                pill.get("company") or "-",
                format_bbox(tuple(pill.get("bbox_xyxy", [0, 0, 0, 0]))),
            ]
        )
    return rows


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

        with gr.Tab("멀티알약 테스트셋"):
            gr.Markdown(
                "# 멀티알약 테스트셋\n"
                "서버의 `datasets/processed/*/images/val` 이미지를 직접 불러와 인식합니다."
            )
            dataset_choices = multi_pill_dataset_choices()
            selected_dataset = default_multi_pill_dataset()
            sample_choices = multi_pill_sample_choices(selected_dataset)
            dataset_selector = gr.Dropdown(
                choices=dataset_choices,
                value=selected_dataset if dataset_choices else None,
                label="데이터셋 선택",
                interactive=True,
            )
            sample_selector = gr.Dropdown(
                choices=sample_choices,
                value=sample_choices[0] if sample_choices else None,
                label="테스트 이미지 선택",
                interactive=True,
            )
            with gr.Row():
                dataset_source = gr.Image(type="numpy", label="선택한 테스트 이미지")
                dataset_annotated = gr.Image(type="numpy", label="탐지/인식 결과")
            dataset_allowed_scope = gr.Textbox(
                label="복약목록 K-ID scope",
                placeholder='비워두면 1000종 전체에서 후보를 냅니다. 예: ["K-004378","K-001732"]',
            )
            with gr.Row():
                load_sample_button = gr.Button("샘플 보기")
                run_sample_button = gr.Button("이 샘플 인식", variant="primary")
            ground_truth_table = gr.Dataframe(
                headers=["번호", "정답 K-ID", "제품명", "업체", "정답 BBox"],
                label="합성셋 정답",
                interactive=False,
            )
            prediction_table = gr.Dataframe(
                headers=[
                    "번호",
                    "BBox x1,y1,x2,y2",
                    "탐지 confidence",
                    "제품명/성분 후보",
                    "상태",
                    "상태 이유",
                ],
                label="모델 예측",
                interactive=False,
            )
            sample_info = gr.JSON(label="샘플 정보")
            sample_result = gr.JSON(label="전체 인식 결과")
            dataset_selector.change(
                fn=update_multi_pill_samples,
                inputs=[dataset_selector],
                outputs=[sample_selector],
            )
            load_sample_button.click(
                fn=load_multi_pill_sample,
                inputs=[dataset_selector, sample_selector],
                outputs=[dataset_source, ground_truth_table, sample_info],
            )
            run_sample_button.click(
                fn=recognize_multi_pill_sample,
                inputs=[dataset_selector, sample_selector, dataset_allowed_scope],
                outputs=[
                    dataset_annotated,
                    ground_truth_table,
                    prediction_table,
                    sample_result,
                    sample_info,
                ],
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
    build_app().launch(
        server_name=os.getenv("PILL_GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("PILL_GRADIO_PORT", "7860")),
    )
