from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]
PILL_INFERENCE_ROOT = REPO_ROOT / "pill_recognition" / "inference"
if str(PILL_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PILL_INFERENCE_ROOT))


@router.post("/pill/recognize")
async def recognize_pill(
    file: UploadFile = File(...),
    allowed_pill_ids: list[str] | None = Form(None),
    allowed_item_seqs: list[str] | None = Form(None),
    allowed_product_names: list[str] | None = Form(None),
):
    """알약 인식 엔드포인트.

    Gradio/전용 API에서 쓰는 RTMDet + AIHub retrieval 파이프라인을 통합 AI
    서버에서도 동일하게 호출한다.
    """
    from pill_recognition.api import (
        attach_api_timings,
        attach_input_scope_resolution,
        elapsed_ms,
        get_pipeline,
        get_product_index,
        read_upload_image,
        resolve_allowed_scope_from_form,
    )

    request_start = perf_counter()
    decode_start = perf_counter()
    image_rgb = await read_upload_image(file)
    decode_ms = elapsed_ms(decode_start)
    pill_scope, input_scope_resolution = resolve_allowed_scope_from_form(
        get_product_index,
        allowed_pill_ids=allowed_pill_ids,
        allowed_item_seqs=allowed_item_seqs,
        allowed_product_names=allowed_product_names,
    )
    pipeline_get_start = perf_counter()
    pipeline = get_pipeline()
    pipeline_get_ms = elapsed_ms(pipeline_get_start)
    pipeline_call_start = perf_counter()
    result = pipeline.recognize(image_rgb, allowed_pill_ids=pill_scope)
    attach_api_timings(
        result,
        request_start=request_start,
        upload_decode_ms=decode_ms,
        pipeline_get_ms=pipeline_get_ms,
        pipeline_call_ms=elapsed_ms(pipeline_call_start),
    )
    attach_input_scope_resolution(result, input_scope_resolution)
    return result.to_dict()
