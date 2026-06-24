from __future__ import annotations

from src.matching.mfds_client import search_product
from src.schema.result import SupplementProduct


def match_and_enrich(product_name: str) -> SupplementProduct:
    """제품명으로 DB 검색 후 SupplementProduct 반환."""
    mfds = search_product(product_name)

    if mfds is None:
        return SupplementProduct(
            product_name=product_name,
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
