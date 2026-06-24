"""
로컬 DB에서 건강기능식품 검색.
사용법: python scripts/search_db.py 칼슘
"""
import os
import sys

import mysql.connector
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

def search(keyword: str):
    conn = mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", 3306)),
        database=os.environ.get("MYSQL_DATABASE", "click_db"),
        user=os.environ.get("MYSQL_USER", "click_user"),
        password=os.environ.get("MYSQL_PASSWORD", "click0623"),
        charset="utf8mb4",
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT sttemnt_no, prduct, entrps, main_fnctn "
        "FROM supplement_info WHERE prduct LIKE %s LIMIT 10",
        (f"%{keyword}%",),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print(f"'{keyword}' 검색 결과 없음")
        return

    print(f"'{keyword}' 검색 결과 {len(rows)}건:\n")
    for r in rows:
        print(f"  [{r['sttemnt_no']}] {r['prduct']}")
        print(f"  업체: {r['entrps']}")
        print(f"  기능: {r['main_fnctn'][:80]}...")
        print()

if __name__ == "__main__":
    keyword = sys.argv[1] if len(sys.argv) > 1 else input("검색어: ")
    search(keyword)
