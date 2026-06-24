from __future__ import annotations

from src.matching.mfds_client import search_product
from src.schema.result import SupplementProduct


def match_and_enrich(ocr_text: str) -> SupplementProduct:
    """OCR 텍스트로 FULLTEXT + RapidFuzz 매칭 후 SupplementProduct 반환."""
    mfds = search_product(ocr_text)

    if mfds is None:
        return SupplementProduct(
            product_name=ocr_text[:100],
            confidence=0.5,
        )

    return SupplementProduct(
        product_code=mfds.product_code,
        product_name=mfds.product_name,
        manufacturer=mfds.manufacturer,
        main_function=mfds.main_function,
        base_standard=mfds.base_standard,
        confidence=0.85,
    )
