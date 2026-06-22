from __future__ import annotations

from src.matching.mfds_client import search_product
from src.schema.result import SupplementProduct


def match_and_enrich(product: SupplementProduct) -> SupplementProduct:
    """MFDS DB에서 제품을 검색해 product_code와 confidence를 보정한다."""
    mfds = search_product(product.product_name)

    if mfds is None:
        return product.model_copy(update={"confidence": 0.5})

    return product.model_copy(
        update={
            "product_code": mfds.product_code,
            "product_name": mfds.product_name,
            "confidence": 0.85,
        }
    )
