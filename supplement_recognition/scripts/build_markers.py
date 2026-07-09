"""
supplement_info 데이터를 파싱해 supplement_product_markers 테이블에 저장하는 스크립트.

테이블 구조:
  marker_id            INT AUTO_INCREMENT PK
  supplement_info_id   INT  FK → supplement_info.id
  marker_text          VARCHAR(255)  파싱된 성분명 원문
  marker_text_normalized VARCHAR(255) 정규화된 성분명 (소문자·공백 제거)
  marker_source_column VARCHAR(50)   'main_fnctn' | 'base_standard'
  marker_type          VARCHAR(50)   'ingredient'
  supplement_id        VARCHAR(50)   백엔드 supplement_map.supplement_id (매핑 시)
  mapping_status       VARCHAR(50)   'mapped' | 'unmapped'

사용법: python supplement_recognition/scripts/build_markers.py
"""
from __future__ import annotations

import os
import re
import sys

import mysql.connector
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from supplement_recognition.src.extraction.ingredient_parser import extract_ingredients, parse_from_main_fnctn, parse_from_base_standard


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS supplement_product_markers (
    marker_id              INT AUTO_INCREMENT PRIMARY KEY,
    supplement_info_id     INT          NOT NULL,
    marker_text            VARCHAR(255) NOT NULL,
    marker_text_normalized VARCHAR(255) NOT NULL,
    marker_source_column   VARCHAR(50)  NOT NULL,
    marker_type            VARCHAR(50)  NOT NULL DEFAULT 'ingredient',
    supplement_id          VARCHAR(50)  NULL,
    mapping_status         VARCHAR(50)  NOT NULL DEFAULT 'unmapped',
    UNIQUE KEY uq_marker (supplement_info_id, marker_text_normalized),
    INDEX idx_marker_text (marker_text_normalized),
    INDEX idx_supplement_id (supplement_id),
    FOREIGN KEY (supplement_info_id) REFERENCES supplement_info(id)
        ON DELETE CASCADE
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
"""


def _get_conn():
    return mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ["MYSQL_PORT"]),
        database=os.environ["MYSQL_DATABASE"],
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        charset="utf8mb4",
    )


def _normalize(text: str) -> str:
    """소문자 변환 + 공백·특수문자 제거."""
    return re.sub(r"[\s\-·•,，()（）\[\]]+", "", text).lower()


def build_markers() -> None:
    conn = _get_conn()
    cursor = conn.cursor(dictionary=True)

    print("supplement_product_markers 테이블 생성 중...")
    cursor.execute(CREATE_TABLE_SQL)
    conn.commit()

    print("supplement_info 전체 로드 중...")
    cursor.execute("SELECT id, main_fnctn, base_standard FROM supplement_info")
    rows = cursor.fetchall()
    print(f"  총 {len(rows):,}건")

    inserted = 0
    skipped = 0

    for row in rows:
        info_id = row["id"]
        from_main = [(name, "main_fnctn") for name in parse_from_main_fnctn(row["main_fnctn"] or "")]
        from_base = [(name, "base_standard") for name in parse_from_base_standard(row["base_standard"] or "")]

        # main_fnctn 우선, 중복 normalized 제거
        seen: set[str] = set()
        markers: list[tuple[str, str]] = []
        for text, source in from_main + from_base:
            norm = _normalize(text)
            if norm and norm not in seen:
                seen.add(norm)
                markers.append((text, source))

        for marker_text, source_col in markers:
            norm = _normalize(marker_text)
            try:
                cursor.execute(
                    """
                    INSERT IGNORE INTO supplement_product_markers
                        (supplement_info_id, marker_text, marker_text_normalized,
                         marker_source_column, marker_type, mapping_status)
                    VALUES (%s, %s, %s, %s, 'ingredient', 'unmapped')
                    """,
                    (info_id, marker_text[:255], norm[:255], source_col),
                )
                if cursor.rowcount:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  [WARN] id={info_id} marker={marker_text}: {e}")

        if inserted % 5000 == 0 and inserted > 0:
            conn.commit()
            print(f"  중간 커밋: {inserted:,}건 저장됨")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"\n완료: 저장 {inserted:,}건 / 중복 스킵 {skipped:,}건")


if __name__ == "__main__":
    build_markers()
