from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.extraction.llm_extractor import extract_from_ocr
from src.matching.matcher import match_and_enrich
from src.ocr.reader import run_ocr
from src.schema.result import (
    ErrorCode,
    RecognitionStatus,
    SupplementRecognitionResult,
)

load_dotenv()

_CONFIDENCE_THRESHOLD = 0.7


def recognize(image: Path | str | np.ndarray, request_id: str | None = None) -> SupplementRecognitionResult:
    rid = request_id or f"rec_supplement_{uuid.uuid4().hex[:8]}"

    try:
        ocr_result = run_ocr(image)
    except Exception as e:
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.MODEL_INFERENCE_FAILED,
            error_detail=str(e),
        )

    if not ocr_result.full_text.strip():
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.OCR_TEXT_NOT_FOUND,
        )

    try:
        product = extract_from_ocr(ocr_result)
    except Exception as e:
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.MODEL_INFERENCE_FAILED,
            error_detail=str(e),
        )

    product = match_and_enrich(product)

    if product.product_code is None:
        warnings = ["공공 데이터에서 일치하는 제품을 찾지 못했습니다. OCR 기반 결과를 반환합니다."]
        status = RecognitionStatus.NEEDS_CONFIRMATION
    elif product.confidence < _CONFIDENCE_THRESHOLD:
        warnings = []
        status = RecognitionStatus.NEEDS_CONFIRMATION
    else:
        warnings = []
        status = RecognitionStatus.COMPLETED

    return SupplementRecognitionResult(
        request_id=rid,
        status=status,
        product=product,
        needs_confirmation=(status == RecognitionStatus.NEEDS_CONFIRMATION),
        warnings=warnings,
    )
