from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import mysql.connector
from mysql.connector import Error as MySQLError
from rapidfuzz import fuzz, utils as fuzz_utils

from supplement_recognition.src.enrichment.official_image_lookup import (
    lookup_official_product_image,
)


@dataclass
class MfdsProduct:
    product_code: str
    product_name: str
    manufacturer: str
    main_function: str
    base_standard: str
    product_image_url: str | None = None
    product_image_source_url: str | None = None
    similarity: float = 0.0
    prebuilt_ingredients: list[str] = None  # supplement_product_markers에서 조회

    def __post_init__(self):
        if self.prebuilt_ingredients is None:
            self.prebuilt_ingredients = []


def _get_conn():
    return mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ["MYSQL_PORT"]),
        database=os.environ["MYSQL_DATABASE"],
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        charset="utf8mb4",
    )


def _normalize_query(name: str) -> str:
    """단일 알파벳·숫자가 앞 토큰과 공백으로 분리된 경우 붙임: '메가도스 B' → '메가도스B'."""
    return re.sub(r"(\w)\s+([A-Za-z0-9])(?=\s|$)", r"\1\2", name)


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


def _ensure_image_columns(cursor) -> bool:
    try:
        cursor.execute("SHOW COLUMNS FROM supplement_info")
        columns = {row["Field"] for row in cursor.fetchall()}
        alters = []
        if "product_image_url" not in columns:
            alters.append("ADD COLUMN product_image_url TEXT COMMENT '공식 제품 이미지 URL'")
        if "product_image_source_url" not in columns:
            alters.append("ADD COLUMN product_image_source_url TEXT COMMENT '공식 제품 이미지 출처 URL'")
        if "product_image_checked_at" not in columns:
            alters.append("ADD COLUMN product_image_checked_at TIMESTAMP NULL COMMENT '공식 이미지 확인 시각'")
        if alters:
            cursor.execute(f"ALTER TABLE supplement_info {', '.join(alters)}")
        return True
    except MySQLError:
        return False


def _select_fields(include_image_columns: bool) -> str:
    fields = "id, sttemnt_no, prduct, entrps, main_fnctn, base_standard"
    if include_image_columns:
        fields += ", product_image_url, product_image_source_url, product_image_checked_at"
    return fields


def _fetch_markers(cursor, supplement_info_id: int) -> list[str]:
    """supplement_product_markers에서 파싱된 성분명 조회. 테이블 없으면 빈 리스트."""
    try:
        cursor.execute(
            "SELECT marker_text FROM supplement_product_markers "
            "WHERE supplement_info_id = %s ORDER BY marker_id",
            (supplement_info_id,),
        )
        return [row["marker_text"] for row in cursor.fetchall()]
    except MySQLError:
        return []


def _fetch_fulltext_candidates(
    cursor,
    product_name: str,
    top_k: int,
    include_image_columns: bool,
) -> list[dict]:
    cursor.execute(
        f"SELECT {_select_fields(include_image_columns)}, "
        "MATCH(prduct) AGAINST(%s IN BOOLEAN MODE) AS score "
        "FROM supplement_info "
        "WHERE MATCH(prduct) AGAINST(%s IN BOOLEAN MODE) "
        "ORDER BY score DESC "
        "LIMIT %s",
        (product_name, product_name, top_k),
    )
    return cursor.fetchall()


def _fetch_like_candidates(
    cursor,
    product_name: str,
    top_k: int,
    include_image_columns: bool,
) -> list[dict]:
    terms = _like_terms(product_name)
    if not terms:
        return []

    clauses = ["prduct LIKE %s" for _ in terms]
    params = [f"%{term}%" for term in terms]
    cursor.execute(
        f"SELECT {_select_fields(include_image_columns)}, 0 AS score "
        "FROM supplement_info "
        f"WHERE {' OR '.join(clauses)} "
        "LIMIT %s",
        (*params, max(top_k * 8, 80)),
    )
    return cursor.fetchall()


def _cache_product_image(cursor, product_code: str, image_url: str | None, source_url: str | None) -> None:
    try:
        cursor.execute(
            "UPDATE supplement_info "
            "SET product_image_url = %s, product_image_source_url = %s, product_image_checked_at = NOW() "
            "WHERE sttemnt_no = %s",
            (image_url, source_url, product_code),
        )
    except MySQLError:
        pass


def search_product(product_name: str, top_k: int = 30, brand_hint: str | None = None) -> Optional[MfdsProduct]:
    """
    1단계: FULLTEXT로 제품명 기반 후보 추출
    2단계: FULLTEXT 인덱스가 없거나 후보가 없으면 LIKE 후보 추출
    3단계: RapidFuzz로 후보 중 가장 유사한 제품 선택
    brand_hint: Gemini가 추출한 브랜드명 — 일치 시 보너스, 불일치 시 패널티 적용
    """
    # 단일 알파벳/숫자가 앞 토큰과 공백으로 분리된 경우 붙임 ('메가도스 B' → '메가도스B')
    normalized_name = _normalize_query(product_name)

    conn = None
    cursor = None
    try:
        conn = _get_conn()
        cursor = conn.cursor(dictionary=True)
        include_image_columns = _ensure_image_columns(cursor)
        if include_image_columns:
            conn.commit()

        try:
            candidates = _fetch_fulltext_candidates(cursor, normalized_name, top_k, include_image_columns)
        except MySQLError:
            candidates = []

        if not candidates:
            candidates = _fetch_like_candidates(cursor, normalized_name, top_k, include_image_columns)

        if not candidates:
            return None

        # 브랜드 힌트 정규화 (비교 시 공백·대소문자 무시)
        brand_normalized = brand_hint.replace(" ", "").lower() if brand_hint else None

        def _score(candidate_name: str) -> float:
            token_set = fuzz.token_set_ratio(normalized_name, candidate_name)
            partial = fuzz.partial_ratio(normalized_name, candidate_name)
            length_ratio = min(len(normalized_name), len(candidate_name)) / max(
                len(normalized_name), len(candidate_name), 1
            )
            base = token_set * 0.6 + partial * 0.4
            score = base * (0.7 + 0.3 * length_ratio)

            # 브랜드 보정: 같은 브랜드면 미세 보너스 (DB에 브랜드명 없는 경우 패널티 금지)
            if brand_normalized and len(brand_normalized) >= 2:
                cand_normalized = candidate_name.replace(" ", "").lower()
                if brand_normalized in cand_normalized:
                    score *= 1.05

            return min(score, 100.0)

        best = max(candidates, key=lambda r: _score(r["prduct"]))
        best["_similarity"] = _score(best["prduct"])
        image_url = best.get("product_image_url")
        image_source_url = best.get("product_image_source_url")
        has_image_lookup_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        if include_image_columns and has_image_lookup_key and not image_url and not best.get("product_image_checked_at"):
            image = lookup_official_product_image(best["prduct"].strip(), best["entrps"])
            if image:
                image_url = image.image_url
                image_source_url = image.source_url
            _cache_product_image(cursor, best["sttemnt_no"], image_url, image_source_url)
            conn.commit()

        # supplement_product_markers 테이블에서 파싱된 성분 조회 (없으면 빈 리스트)
        markers = _fetch_markers(cursor, best["id"]) if "id" in best else []

        return MfdsProduct(
            product_code=best["sttemnt_no"],
            product_name=best["prduct"].strip(),
            manufacturer=best["entrps"],
            main_function=best["main_fnctn"],
            base_standard=best["base_standard"],
            product_image_url=image_url,
            product_image_source_url=image_source_url,
            similarity=best["_similarity"],
            prebuilt_ingredients=markers,
        )
    except Exception:
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def search_top_products(product_name: str, top_k: int = 5) -> list[MfdsProduct]:
    """동명이제 감지용 — 유사도 상위 N개 반환 (score 차이 10 이내, 다른 제조사)."""
    normalized_name = _normalize_query(product_name)
    conn = None
    cursor = None
    try:
        conn = _get_conn()
        cursor = conn.cursor(dictionary=True)
        include_image_columns = _ensure_image_columns(cursor)
        if include_image_columns:
            conn.commit()

        try:
            candidates = _fetch_fulltext_candidates(cursor, normalized_name, 50, include_image_columns)
        except Exception:
            candidates = []
        if not candidates:
            candidates = _fetch_like_candidates(cursor, normalized_name, 50, include_image_columns)
        if not candidates:
            return []

        def _score(name: str) -> float:
            token_set = fuzz.token_set_ratio(normalized_name, name)
            partial = fuzz.partial_ratio(normalized_name, name)
            length_ratio = min(len(normalized_name), len(name)) / max(len(normalized_name), len(name), 1)
            base = token_set * 0.6 + partial * 0.4
            return base * (0.7 + 0.3 * length_ratio)

        scored = [(r, _score(r["prduct"])) for r in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_score = scored[0][1] if scored else 0
        results = []
        seen_names: set[str] = set()
        for row, score in scored:
            if score < _SIMILARITY_THRESHOLD:
                break
            if score < best_score - 10:
                break
            name_key = row["prduct"].strip().lower()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            markers = _fetch_markers(cursor, row["id"]) if "id" in row else []
            results.append(MfdsProduct(
                product_code=row["sttemnt_no"],
                product_name=row["prduct"].strip(),
                manufacturer=row["entrps"],
                main_function=row["main_fnctn"],
                base_standard=row["base_standard"],
                product_image_url=row.get("product_image_url"),
                product_image_source_url=row.get("product_image_source_url"),
                similarity=score,
                prebuilt_ingredients=markers,
            ))
            if len(results) >= top_k:
                break
        return results
    except Exception:
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


_SIMILARITY_THRESHOLD = 70
