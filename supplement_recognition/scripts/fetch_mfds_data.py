"""
MFDS 건강기능식품 전체 데이터를 API에서 받아 MySQL에 저장하는 스크립트.
"""
import os
import time
import httpx
import mysql.connector
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

API_KEY   = os.environ["MFDS_API_KEY"]
BASE_URL  = "https://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01"
ROWS      = 100   # 한 번에 가져올 행 수

DB = dict(
    host     = os.environ.get("MYSQL_HOST", "localhost"),
    port     = int(os.environ.get("MYSQL_PORT", 3306)),
    database = os.environ.get("MYSQL_DATABASE", "click_db"),
    user     = os.environ.get("MYSQL_USER", "click_user"),
    password = os.environ.get("MYSQL_PASSWORD", "click0623"),
    charset  = "utf8mb4",
)

INSERT_SQL = """
INSERT INTO supplement_info
    (sttemnt_no, prduct, entrps, regist_dt, distb_pd,
     sungsang, srv_use, prsrv_pd, intake_hint1, main_fnctn, base_standard)
VALUES
    (%(sttemnt_no)s, %(prduct)s, %(entrps)s, %(regist_dt)s, %(distb_pd)s,
     %(sungsang)s, %(srv_use)s, %(prsrv_pd)s, %(intake_hint1)s, %(main_fnctn)s, %(base_standard)s)
ON DUPLICATE KEY UPDATE
    prduct       = VALUES(prduct),
    entrps       = VALUES(entrps),
    main_fnctn   = VALUES(main_fnctn),
    base_standard= VALUES(base_standard)
"""


def fetch_page(page: int) -> tuple[list[dict], int]:
    params = {
        "serviceKey": API_KEY,
        "type": "json",
        "numOfRows": ROWS,
        "pageNo": page,
    }
    resp = httpx.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    body = resp.json().get("body", {})
    total = body.get("totalCount", 0)
    items = [entry.get("item", {}) for entry in body.get("items", [])]
    return items, total


def row(item: dict) -> dict:
    return {
        "sttemnt_no":  item.get("STTEMNT_NO", "")[:30],
        "prduct":      (item.get("PRDUCT") or "").strip()[:200],
        "entrps":      (item.get("ENTRPS") or "")[:200],
        "regist_dt":   (item.get("REGIST_DT") or "")[:10],
        "distb_pd":    (item.get("DISTB_PD") or "")[:100],
        "sungsang":    item.get("SUNGSANG") or "",
        "srv_use":     item.get("SRV_USE") or "",
        "prsrv_pd":    (item.get("PRSRV_PD") or "")[:200],
        "intake_hint1":item.get("INTAKE_HINT1") or "",
        "main_fnctn":  item.get("MAIN_FNCTN") or "",
        "base_standard":item.get("BASE_STANDARD") or "",
    }


def main():
    conn = mysql.connector.connect(**DB)
    cursor = conn.cursor()

    # 1페이지로 전체 건수 파악
    _, total = fetch_page(1)
    total_pages = (total + ROWS - 1) // ROWS
    print(f"Total {total:,} records / {total_pages} pages - starting...")

    saved = 0
    start = time.time()

    for page in range(1, total_pages + 1):
        try:
            items, _ = fetch_page(page)
        except Exception as e:
            print(f"  [ERROR] page {page}: {e} - retrying")
            time.sleep(2)
            try:
                items, _ = fetch_page(page)
            except Exception as e2:
                print(f"  [FAIL] page {page} skipped: {e2}")
                continue

        rows = [row(i) for i in items if i.get("STTEMNT_NO")]
        if rows:
            cursor.executemany(INSERT_SQL, rows)
            conn.commit()
            saved += len(rows)

        elapsed = time.time() - start
        remaining = (elapsed / page) * (total_pages - page)
        print(f"  [{page}/{total_pages}] saved {saved:,} | "
              f"elapsed {elapsed/60:.1f}min | remaining ~{remaining/60:.1f}min")

        time.sleep(0.3)  # API 호출 간격

    cursor.close()
    conn.close()
    print(f"\nDone! Total {saved:,} records saved | {(time.time()-start)/60:.1f}min")


if __name__ == "__main__":
    main()
