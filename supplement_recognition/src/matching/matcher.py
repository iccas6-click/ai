from __future__ import annotations

from supplement_recognition.src.extraction.ingredient_parser import extract_ingredients
from supplement_recognition.src.matching.mfds_client import search_product
from supplement_recognition.src.schema.result import SupplementProduct

_SIMILARITY_THRESHOLD = 70


def match_and_enrich(product_name: str) -> SupplementProduct:
    """제품명으로 FULLTEXT + RapidFuzz 매칭 후 SupplementProduct 반환."""
    mfds = search_product(product_name)

    if mfds is None or mfds.similarity < _SIMILARITY_THRESHOLD:
        return SupplementProduct(
            product_name=product_name[:100],
            confidence=0.0,
        )

    ingredients = extract_ingredients(mfds.main_function, mfds.base_standard)

    return SupplementProduct(
        product_code=mfds.product_code,
        product_name=mfds.product_name,
        manufacturer=mfds.manufacturer,
        product_image_url=mfds.product_image_url,
        main_function=mfds.main_function,
        base_standard=mfds.base_standard,
        ingredients=ingredients,
        confidence=round(mfds.similarity / 100, 2),
    )
