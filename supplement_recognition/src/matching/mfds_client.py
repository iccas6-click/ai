from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import mysql.connector
from mysql.connector import Error as MySQLError
from rapidfuzz import fuzz


@dataclass
class MfdsProduct:
    product_code: str
    product_name: str
    manufacturer: str
    main_function: str
    base_standard: str
    product_image_url: str | None = None
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


def _like_terms(product_name: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", product_name).strip()
    terms = [cleaned]
    terms.extend(part for part in re.split(r"[\s()/,，]+", cleaned) if len(part) >= 2)

    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique[:8]


def _fetch_fulltext_candidates(cursor, product_name: str, top_k: int) -> list[dict]:
    cursor.execute(
        "SELECT sttemnt_no, prduct, entrps, main_fnctn, base_standard, "
        "MATCH(prduct) AGAINST(%s IN BOOLEAN MODE) AS score "
        "FROM supplement_info "
        "WHERE MATCH(prduct) AGAINST(%s IN BOOLEAN MODE) "
        "ORDER BY score DESC "
        "LIMIT %s",
        (product_name, product_name, top_k),
    )
    return cursor.fetchall()


def _fetch_like_candidates(cursor, product_name: str, top_k: int) -> list[dict]:
    terms = _like_terms(product_name)
    if not terms:
        return []

    clauses = ["prduct LIKE %s" for _ in terms]
    params = [f"%{term}%" for term in terms]
    cursor.execute(
        "SELECT sttemnt_no, prduct, entrps, main_fnctn, base_standard, 0 AS score "
        "FROM supplement_info "
        f"WHERE {' OR '.join(clauses)} "
        "LIMIT %s",
        (*params, max(top_k * 8, 80)),
    )
    return cursor.fetchall()


def search_product(product_name: str, top_k: int = 30) -> Optional[MfdsProduct]:
    """
    1단계: FULLTEXT로 제품명 기반 후보 추출
    2단계: FULLTEXT 인덱스가 없거나 후보가 없으면 LIKE 후보 추출
    3단계: RapidFuzz로 후보 중 가장 유사한 제품 선택
    """
    conn = None
    cursor = None
    try:
        conn = _get_conn()
        cursor = conn.cursor(dictionary=True)

        try:
            candidates = _fetch_fulltext_candidates(cursor, product_name, top_k)
        except MySQLError:
            candidates = []

        if not candidates:
            candidates = _fetch_like_candidates(cursor, product_name, top_k)

        if not candidates:
            return None

        def _score(candidate_name: str) -> float:
            # partial_ratio만 쓰면 짧은 쿼리가 긴 제품명 안에 부분 포함될 때 100점을 줘서
            # 엉뚱한 제품이 매칭되는 문제가 있음.
            # length_ratio로 길이 차이에 패널티를 주어 보정.
            partial = fuzz.partial_ratio(product_name, candidate_name)
            length_ratio = min(len(product_name), len(candidate_name)) / max(
                len(product_name), len(candidate_name), 1
            )
            return partial * (0.5 + 0.5 * length_ratio)

        best = max(candidates, key=lambda r: _score(r["prduct"]))
        best["_similarity"] = _score(best["prduct"])

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
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
