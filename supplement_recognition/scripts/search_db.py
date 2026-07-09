"""
로컬 DB에서 건강기능식품 대화형 검색.

실행 (ai/ 루트에서):
    python supplement_recognition/scripts/search_db.py

명령어:
    검색어 입력  →  제품 검색 (부분 일치)
    숫자 입력    →  해당 번호 제품 상세 정보 (성분 등)
    q / quit    →  종료
"""
import os
import sys

import mysql.connector
from dotenv import load_dotenv

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()


def get_conn():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", 3306)),
        database=os.environ.get("MYSQL_DATABASE", "click_db"),
        user=os.environ.get("MYSQL_USER", "click_user"),
        password=os.environ.get("MYSQL_PASSWORD", "click0623"),
        charset="utf8mb4",
    )


def search(conn, keyword: str, limit: int = 100) -> list[dict]:
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT sttemnt_no, prduct, entrps "
        "FROM supplement_info WHERE prduct LIKE %s LIMIT %s",
        (f"%{keyword}%", limit),
    )
    rows = cur.fetchall()
    cur.close()
    return rows


def detail(conn, sttemnt_no: str) -> dict | None:
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT sttemnt_no, prduct, entrps, main_fnctn, base_standard "
        "FROM supplement_info WHERE sttemnt_no = %s",
        (sttemnt_no,),
    )
    row = cur.fetchone()
    cur.close()
    return row


def print_results(rows: list[dict]) -> None:
    print()
    for i, r in enumerate(rows, 1):
        print(f"  {i:2d}. {r['prduct']}")
        print(f"      업체: {r['entrps']}  코드: {r['sttemnt_no']}")
    print()


def print_detail(row: dict) -> None:
    print(f"\n{'─'*55}")
    print(f"제품명: {row['prduct']}")
    print(f"업체:   {row['entrps']}")
    print(f"코드:   {row['sttemnt_no']}")
    if row.get("main_fnctn"):
        print(f"\n[기능성]\n{row['main_fnctn']}")
    if row.get("base_standard"):
        print(f"\n[기준규격]\n{row['base_standard']}")
    print(f"{'─'*55}\n")


def main() -> None:
    print("건강기능식품 DB 검색  (종료: q)\n")
    try:
        conn = get_conn()
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        print("Docker가 실행 중인지, .env 설정이 맞는지 확인하세요.")
        sys.exit(1)

    last_rows: list[dict] = []

    while True:
        try:
            raw = input("검색어 또는 번호> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            break

        # 숫자 입력 → 이전 검색 결과에서 상세 보기
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(last_rows):
                row = detail(conn, last_rows[idx]["sttemnt_no"])
                if row:
                    print_detail(row)
                    print(f'✔ 이미지 파일명으로 쓸 제품명: "{row["prduct"]}"')
            else:
                print(f"  1~{len(last_rows)} 사이 번호를 입력하세요.\n")
            continue

        # 텍스트 입력 → 검색
        rows = search(conn, raw)
        if not rows:
            print(f"  '{raw}' 검색 결과 없음\n")
            last_rows = []
            continue

        print(f"  '{raw}' 검색 결과 {len(rows)}건 (번호 입력 시 상세 보기):")
        print_results(rows)
        last_rows = rows

    conn.close()
    print("종료")


if __name__ == "__main__":
    main()
