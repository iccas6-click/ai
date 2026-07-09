from __future__ import annotations

import re
import uuid
from pathlib import Path

from dotenv import load_dotenv

from supplement_recognition.src.enrichment.ingredient_lookup import lookup_ingredients_from_gemini
from supplement_recognition.src.extraction.image_preprocessor import preprocess
from supplement_recognition.src.extraction.llm_extractor import extract_product_candidates
from supplement_recognition.src.matching.matcher import match_and_enrich
from supplement_recognition.src.matching.mfds_client import search_top_products
from supplement_recognition.src.schema.result import (
    ErrorCode,
    RecognitionStatus,
    SupplementProduct,
    SupplementRecognitionResult,
)

load_dotenv()

_CONFIDENCE_THRESHOLD = 0.7


def _extract_brand(candidates: list[str]) -> str | None:
    """Gemini 후보1(브랜드 포함)과 후보2(브랜드 제외) 차이로 브랜드명 추출."""
    if len(candidates) < 2:
        return None
    words0 = candidates[0].split()
    words1_set = set(candidates[1].split())
    brand_words = [w for w in words0 if w not in words1_set]
    brand = " ".join(brand_words).strip()
    return brand if len(brand) >= 2 else None


def _normalize_manufacturer(name: str | None) -> str:
    """제조사명 정규화 — 공백·괄호·(주)/(유) 표기 차이 제거."""
    if not name:
        return ""
    n = re.sub(r"[\s ]+", "", name)   # 모든 공백 제거
    n = re.sub(r"[()（）]", "", n)           # 괄호 제거
    n = re.sub(r"(주|유|합자|합명)$", "", n)  # 법인 형태 제거
    return n.lower()


def recognize(image_path: Path | str, request_id: str | None = None) -> SupplementRecognitionResult:
    rid = request_id or f"rec_supplement_{uuid.uuid4().hex[:8]}"

    # 1. 이미지 전처리
    try:
        processed_path = preprocess(image_path)
    except Exception:
        processed_path = Path(image_path)

    # 2. Gemini Vision — 제품명 후보 추출 (최대 3개)
    try:
        name_candidates = extract_product_candidates(processed_path)
    except Exception as e:
        if processed_path != Path(image_path) and processed_path.exists():
            processed_path.unlink()
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.MODEL_INFERENCE_FAILED,
            error_detail=str(e),
        )
    else:
        if processed_path != Path(image_path) and processed_path.exists():
            processed_path.unlink()

    name_candidates = [c for c in name_candidates if c.strip()]
    if not name_candidates:
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.FAILED,
            error_code=ErrorCode.OCR_TEXT_NOT_FOUND,
        )

    # 3. 후보 전부 DB 매칭 — confidence 최고 결과 선택
    brand_hint = _extract_brand(name_candidates)
    best_product = None
    best_name = name_candidates[0]
    for candidate in name_candidates:
        product = match_and_enrich(candidate, brand_hint=brand_hint)
        if product.product_code is not None:
            if best_product is None or product.confidence > best_product.confidence:
                best_product = product
                best_name = candidate

    product = best_product if best_product is not None else match_and_enrich(name_candidates[0])
    product_name = best_name

    # 4. DB 미등재 — Gemini로 식약처 기준 성분 추출 후 반환
    if product.product_code is None:
        gemini_ingredients = lookup_ingredients_from_gemini(product_name)
        warnings = [f"'{product_name}' 제품이 식약처 DB에 등록되어 있지 않습니다."]
        if gemini_ingredients:
            warnings.append("성분 정보는 Gemini가 식약처 자료를 기준으로 추출했습니다. 반드시 확인하세요.")
        return SupplementRecognitionResult(
            request_id=rid,
            status=RecognitionStatus.NEEDS_CONFIRMATION,
            product=SupplementProduct(
                product_name=product_name,
                ingredients=gemini_ingredients,
                ingredients_source="gemini",
                confidence=0.0,
            ),
            needs_confirmation=True,
            warnings=warnings,
        )

    # 5. 동명이제 감지
    status = (
        RecognitionStatus.COMPLETED
        if product.confidence >= _CONFIDENCE_THRESHOLD
        else RecognitionStatus.NEEDS_CONFIRMATION
    )

    candidate_products: list[SupplementProduct] = []
    if status == RecognitionStatus.COMPLETED:
        top = search_top_products(product_name, top_k=6)
        best_mfr = _normalize_manufacturer(product.manufacturer)

        for m in top:
            if m.product_code == product.product_code:
                continue
            other_mfr = _normalize_manufacturer(m.manufacturer)
            # 동일 제조사(표기 차이) — 최신 코드(숫자 큰 쪽)가 best보다 크면 교체
            if other_mfr == best_mfr:
                if m.product_code and product.product_code:
                    if m.product_code > product.product_code:
                        product = SupplementProduct(
                            product_code=m.product_code,
                            product_name=m.product_name,
                            manufacturer=m.manufacturer,
                            confidence=round(m.similarity / 100, 2),
                        )
            else:
                # 진짜 다른 제조사 → candidates로 반환
                candidate_products.append(
                    SupplementProduct(
                        product_code=m.product_code,
                        product_name=m.product_name,
                        manufacturer=m.manufacturer,
                        confidence=round(m.similarity / 100, 2),
                    )
                )

    return SupplementRecognitionResult(
        request_id=rid,
        status=status,
        product=product,
        candidates=candidate_products,
        needs_confirmation=(status == RecognitionStatus.NEEDS_CONFIRMATION),
        warnings=[],
    )
