from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import load_dotenv

from supplement_recognition.src.extraction.llm_extractor import extract_product_name
from supplement_recognition.src.matching.matcher import match_and_enrich
from supplement_recognition.src.schema.result import (
    ErrorCode,
    RecognitionStatus,
    SupplementRecognitionResult,
)

load_dotenv()

_CONFIDENCE_THRESHOLD = 0.7


def recognize(image_path: Path | str, request_id: str | None = None) -> SupplementRecognitionResult:
    rid = request_id or f"rec_supplement_{uuid.uuid4().hex[:8]}"

    # 1. Gemini Vision으로 이미지에서 제품명 직접 추출
    try:
        product_name = extract_product_name(image_path)
    except Exception as e:
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.MODEL_INFERENCE_FAILED,
            error_detail=str(e),
        )

    if not product_name.strip():
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.OCR_TEXT_NOT_FOUND,
        )

    # 2. FULLTEXT + RapidFuzz로 DB 매칭
    product = match_and_enrich(product_name)

    if product.product_code is None:
        warnings = [f"'{product_name}' 제품을 DB에서 찾지 못했습니다."]
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
