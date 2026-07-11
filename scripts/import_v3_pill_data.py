"""
drug-supplement schema v3 CSV → AI DB import

대상 테이블:
  - pill_products
  - pill_product_ingredients
  - supplement_product_markers (기존 데이터 교체)

실행:
  python scripts/import_v3_pill_data.py --csv-dir "../drug-supplement schema v3"

환경변수 (.env):
  MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv


def connect(env_path: Path) -> mysql.connector.MySQLConnection:
    load_dotenv(env_path)
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
    )


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def import_pill_products(cursor, rows: list[dict]) -> int:
    sql = """
        INSERT INTO pill_products (pill_product_id, product_name, product_name_normalized)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            product_name = VALUES(product_name),
            product_name_normalized = VALUES(product_name_normalized)
    """
    data = [(r["pill_product_id"], r["product_name"], r["product_name_normalized"]) for r in rows]
    cursor.executemany(sql, data)
    return len(data)


def import_pill_product_ingredients(cursor, rows: list[dict]) -> int:
    sql = """
        INSERT INTO pill_product_ingredients
            (pill_product_id, ingredient_name, ingredient_name_normalized, canonical_drug_id)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            ingredient_name = VALUES(ingredient_name),
            canonical_drug_id = VALUES(canonical_drug_id)
    """
    data = [
        (r["pill_product_id"], r["ingredient_name"], r["ingredient_name_normalized"], r["canonical_drug_id"])
        for r in rows
    ]
    cursor.executemany(sql, data)
    return len(data)


# v3 marker_source_column 값 "product" → AI DB 실제 컬럼명 "prduct"
_SOURCE_COLUMN_MAP = {
    "product": "prduct",
}


def import_supplement_product_markers(cursor, rows: list[dict]) -> int:
    cursor.execute("DELETE FROM supplement_product_markers")

    sql = """
        INSERT INTO supplement_product_markers
            (supplement_info_id, marker_text, marker_text_normalized,
             marker_source_column, marker_type, supplement_id, mapping_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    data = [
        (
            int(r["supplement_info_id"]),
            r["marker_text"],
            r["marker_text_normalized"],
            _SOURCE_COLUMN_MAP.get(r.get("marker_source_column", ""), r.get("marker_source_column", "")),
            r.get("marker_type", ""),
            r["supplement_id"],
            r.get("mapping_status", "confirmed"),
        )
        for r in rows
    ]

    batch = 5000
    for i in range(0, len(data), batch):
        cursor.executemany(sql, data[i : i + batch])

    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="../drug-supplement schema v3")
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    env_path = Path(args.env)

    conn = connect(env_path)
    conn.autocommit = False
    cursor = conn.cursor()

    try:
        print("pill_products import 중...")
        rows = load_csv(csv_dir / "pill_products.csv")
        n = import_pill_products(cursor, rows)
        print(f"  {n}행 처리")

        print("pill_product_ingredients import 중...")
        rows = load_csv(csv_dir / "pill_product_ingredients.csv")
        n = import_pill_product_ingredients(cursor, rows)
        print(f"  {n}행 처리")

        print("supplement_product_markers 교체 중...")
        rows = load_csv(csv_dir / "supplement_product_markers.csv")
        n = import_supplement_product_markers(cursor, rows)
        print(f"  {n}행 처리")

        conn.commit()
        print("완료")
    except Exception as e:
        conn.rollback()
        print(f"오류 발생, 롤백: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
