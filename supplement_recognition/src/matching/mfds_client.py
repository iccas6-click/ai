from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import httpx

_BASE_URL = "https://apis.data.go.kr/1471000/HtfsInfoService03"


@dataclass
class MfdsProduct:
    product_code: str   # STTEMNT_NO
    product_name: str   # PRDUCT
    manufacturer: str   # ENTRPS
    main_function: str  # MAIN_FNCTN


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
        "PRDUCT": product_name,
    }

    try:
        response = httpx.get(
            f"{_BASE_URL}/getHtfsItem01",
            params=params,
            timeout=10.0,
        )
        response.raise_for_status()
        body = response.json()
        items = body.get("body", {}).get("items", [])
        if not items:
            return None
        item = items[0].get("item", {})
        return MfdsProduct(
            product_code=item.get("STTEMNT_NO", ""),
            product_name=item.get("PRDUCT", product_name).strip(),
            manufacturer=item.get("ENTRPS", ""),
            main_function=item.get("MAIN_FNCTN", ""),
        )
    except Exception:
        return None
