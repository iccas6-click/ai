from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pill_recognition_legacy.aihub_classifier import (
    AIHubProductInfo,
    load_aihub_product_master,
)


@dataclass(frozen=True)
class ProductSearchQuery:
    imprint: str = ""
    shape: str = ""
    color: str = ""
    text: str = ""
    limit: int = 20


def load_product_index(mapping_path: Path | None) -> dict[str, AIHubProductInfo]:
    if mapping_path is None:
        return {}
    return load_aihub_product_master(mapping_path.parent)


def search_products(
    products: dict[str, AIHubProductInfo],
    query: ProductSearchQuery,
) -> list[dict]:
    scored = []
    for product in products.values():
        score, reasons = score_product(product, query)
        if score <= 0:
            continue
        row = asdict(product)
        row["score"] = score
        row["matched"] = ", ".join(reasons)
        scored.append(row)

    scored.sort(
        key=lambda row: (
            -row["score"],
            row.get("product_name") or "",
            row["pill_id"],
        )
    )
    return scored[: max(1, query.limit)]


def score_product(
    product: AIHubProductInfo,
    query: ProductSearchQuery,
) -> tuple[int, list[str]]:
    score = 0
    reasons = []

    imprints = query_imprint_variants(query.imprint)
    if imprints:
        product_imprints = {
            variant
            for value in (product.print_front, product.print_back)
            for variant in imprint_variants(value)
        }
        if any(imprint == value for imprint in imprints for value in product_imprints):
            score += 100
            reasons.append("각인 exact")
        elif any(
            imprint in value or value in imprint
            for imprint in imprints
            for value in product_imprints
        ):
            score += 70
            reasons.append("각인 partial")
        else:
            score -= 20

    shape = query.shape.strip()
    if shape and product.drug_shape:
        if shape == product.drug_shape:
            score += 35
            reasons.append("모양")
        elif shape in product.drug_shape or product.drug_shape in shape:
            score += 18
            reasons.append("모양 partial")

    color = query.color.strip()
    if color:
        colors = {product.color_class1 or "", product.color_class2 or ""}
        if color in colors:
            score += 35
            reasons.append("색")

    text = normalize_text(query.text)
    if text:
        haystacks = {
            "제품명": product.product_name,
            "성분": product.ingredient,
            "업체": product.company,
            "품목기준코드": product.item_seq,
        }
        for label, value in haystacks.items():
            if value and text in normalize_text(value):
                score += 30
                reasons.append(label)
                break

    return score, reasons


def normalize_token(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", str(value)).upper()


def imprint_variants(value: str | None) -> set[str]:
    token = normalize_token(value or "")
    if not token:
        return set()
    variants = {token}
    without_split = token.replace("분할선", "").replace("분할", "")
    if without_split:
        variants.add(without_split)
    return variants


def query_imprint_variants(value: str | None) -> set[str]:
    raw = str(value or "").strip()
    if not raw:
        return set()
    variants = set()
    for part in re.split(r"[\s,/|]+", raw):
        variants.update(imprint_variants(part))
    variants.update(imprint_variants(raw))
    return variants


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).upper()
