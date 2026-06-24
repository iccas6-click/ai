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


def _get_conn():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", 3306)),
        database=os.environ.get("MYSQL_DATABASE", "click_db"),
        user=os.environ.get("MYSQL_USER", "click_user"),
        password=os.environ.get("MYSQL_PASSWORD", "click0623"),
        charset="utf8mb4",
    )


def search_product(ocr_text: str, top_k: int = 30) -> Optional[MfdsProduct]:
    """
    1단계: FULLTEXT로 OCR 텍스트 기반 후보 top_k개 추출
    2단계: RapidFuzz로 후보 중 가장 유사한 제품 선택
    """
    try:
        conn = _get_conn()
        cursor = conn.cursor(dictionary=True)

        # FULLTEXT 후보 추출
        cursor.execute(
            "SELECT sttemnt_no, prduct, entrps, main_fnctn, base_standard, "
            "MATCH(prduct) AGAINST(%s IN BOOLEAN MODE) AS score "
            "FROM supplement_info "
            "WHERE MATCH(prduct) AGAINST(%s IN BOOLEAN MODE) "
            "ORDER BY score DESC "
            "LIMIT %s",
            (ocr_text, ocr_text, top_k),
        )
        candidates = cursor.fetchall()
        cursor.close()
        conn.close()

        if not candidates:
            return None

        # RapidFuzz로 재정렬
        best = max(
            candidates,
            key=lambda r: fuzz.partial_ratio(ocr_text, r["prduct"]),
        )

        return MfdsProduct(
            product_code=best["sttemnt_no"],
            product_name=best["prduct"].strip(),
            manufacturer=best["entrps"],
            main_function=best["main_fnctn"],
            base_standard=best["base_standard"],
        )
    except Exception:
        return None
