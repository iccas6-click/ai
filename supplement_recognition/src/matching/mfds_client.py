from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import mysql.connector


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


def search_product(product_name: str) -> Optional[MfdsProduct]:
    """로컬 DB에서 제품명 부분 일치 검색. 없으면 None 반환."""
    try:
        conn = _get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT sttemnt_no, prduct, entrps, main_fnctn, base_standard "
            "FROM supplement_info "
            "WHERE prduct LIKE %s "
            "ORDER BY CHAR_LENGTH(prduct) ASC "
            "LIMIT 1",
            (f"%{product_name}%",),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return None

        return MfdsProduct(
            product_code=row["sttemnt_no"],
            product_name=row["prduct"].strip(),
            manufacturer=row["entrps"],
            main_function=row["main_fnctn"],
            base_standard=row["base_standard"],
        )
    except Exception:
        return None
