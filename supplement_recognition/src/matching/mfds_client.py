from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import mysql.connector
from rapidfuzz import fuzz


@dataclass
class MfdsProduct:
    product_code: str
    product_name: str
    manufacturer: str
    main_function: str
    base_standard: str
    similarity: float = 0.0


def _get_conn():
    return mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ["MYSQL_PORT"]),
        database=os.environ["MYSQL_DATABASE"],
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        charset="utf8mb4",
    )


def search_product(product_name: str, top_k: int = 30) -> Optional[MfdsProduct]:
    """
    1단계: FULLTEXT로 제품명 기반 후보 top_k개 추출
    2단계: RapidFuzz로 후보 중 가장 유사한 제품 선택
    """
    try:
        conn = _get_conn()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT sttemnt_no, prduct, entrps, main_fnctn, base_standard, "
            "MATCH(prduct) AGAINST(%s IN BOOLEAN MODE) AS score "
            "FROM supplement_info "
            "WHERE MATCH(prduct) AGAINST(%s IN BOOLEAN MODE) "
            "ORDER BY score DESC "
            "LIMIT %s",
            (product_name, product_name, top_k),
        )
        candidates = cursor.fetchall()
        cursor.close()
        conn.close()

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda r: fuzz.partial_ratio(product_name, r["prduct"]),
        )
        best["_similarity"] = fuzz.partial_ratio(product_name, best["prduct"])

        return MfdsProduct(
            product_code=best["sttemnt_no"],
            product_name=best["prduct"].strip(),
            manufacturer=best["entrps"],
            main_function=best["main_fnctn"],
            base_standard=best["base_standard"],
            similarity=best["_similarity"],
        )
    except Exception:
        return None
