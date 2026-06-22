from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import httpx

_BASE_URL = "http://apis.data.go.kr/1471000/HtfsInfoService03"


@dataclass
class MfdsProduct:
    product_code: str
    product_name: str
    functional_ingredients: list[str]


def search_product(product_name: str) -> Optional[MfdsProduct]:
    """식약처 건강기능식품 API로 제품명 검색. 일치 없으면 None 반환."""
    api_key = os.environ.get("MFDS_API_KEY", "")
    if not api_key:
        return None

    params = {
        "serviceKey": api_key,
        "type": "json",
        "numOfRows": 5,
        "pageNo": 1,
        "PRDT_NM": product_name,
    }

    try:
        response = httpx.get(
            f"{_BASE_URL}/getHtfsItem03",
            params=params,
            timeout=10.0,
        )
        response.raise_for_status()
        body = response.json()
        items = body.get("body", {}).get("items", [])
        if not items:
            return None
        item = items[0]
        return MfdsProduct(
            product_code=item.get("PRDT_NO", ""),
            product_name=item.get("PRDT_NM", product_name),
            functional_ingredients=_parse_ingredients(item.get("MTRAL_NM", "")),
        )
    except Exception:
        return None


def _parse_ingredients(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]
