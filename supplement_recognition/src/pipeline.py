from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.ocr.reader import extract_text
from src.extraction.llm_extractor import extract_product_name
from src.matching.matcher import match_and_enrich
from src.schema.result import (
    ErrorCode,
    RecognitionStatus,
    SupplementRecognitionResult,
)

load_dotenv()

_CONFIDENCE_THRESHOLD = 0.7


def recognize(image_path: Path | str, request_id: str | None = None) -> SupplementRecognitionResult:
    rid = request_id or f"rec_supplement_{uuid.uuid4().hex[:8]}"

    # 1. EasyOCR로 텍스트 추출
    try:
        ocr_text = extract_text(image_path)
    except Exception as e:
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.MODEL_INFERENCE_FAILED,
            error_detail=str(e),
        )

    if not ocr_text.strip():
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.OCR_TEXT_NOT_FOUND,
        )

    # 2. Gemini로 OCR 텍스트에서 제품명 추출
    try:
        product_name = extract_product_name(ocr_text)
    except Exception as e:
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.MODEL_INFERENCE_FAILED,
            error_detail=str(e),
        )

    # 3. FULLTEXT + RapidFuzz로 DB 매칭
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
