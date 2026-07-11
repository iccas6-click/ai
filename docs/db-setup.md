# AI DB 구축 가이드

AI DB(`click_db`, 포트 3306) 단독 세팅 가이드입니다.

> **두 DB를 함께 세팅하는 경우** → [`backend/docs/db-setup.md`](../../backend/docs/db-setup.md)를 먼저 보세요. Backend DB(3307)와 AI DB(3306) 전체 순서가 정리되어 있습니다.

---

## 전제 조건

- `drug-supplement schema v3/` 폴더 보유 (팀 드라이브에서 받아야 함)
- Backend DB(`click_backend_db`, 포트 3307)가 먼저 구축되어 있어야 합니다. `pill_product_ingredients`의 `canonical_drug_id`가 Backend DB의 `canonical_drug_entities`를 참조합니다 (cross-DB 참조, 앱 레벨에서 일관성 보장).

---

## `.env` 설정

`.env.example`을 복사해 `.env` 생성:

```
GEMINI_API_KEY=your_gemini_api_key
CBNUAI_API_KEY=your_cbnuai_api_key

MYSQL_DATABASE=click_db
MYSQL_USER=click_user
MYSQL_PASSWORD=your_password
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306

PILL_MYSQL_HOST=127.0.0.1
PILL_MYSQL_PORT=3307
PILL_MYSQL_DATABASE=click_backend_db
PILL_MYSQL_USER=click_user
PILL_MYSQL_PASSWORD=your_backend_password
```

---

## 컨테이너 실행

```powershell
docker compose up -d db
```

컨테이너명: `click_supplement_db` / 포트: `3306`

테이블 정의는 `supplement_recognition/db/init.sql`이 자동 실행됩니다.

---

## 데이터 적재 순서

### 1단계 — supplement_info (44,885행)

`supplement_product_markers`가 이 테이블의 FK를 참조하므로 반드시 먼저 적재해야 합니다.

```powershell
python -c "
import csv, os, mysql.connector
from dotenv import load_dotenv
from pathlib import Path

load_dotenv('.env')
conn = mysql.connector.connect(
    host=os.environ.get('MYSQL_HOST', '127.0.0.1'),
    port=int(os.environ.get('MYSQL_PORT', 3306)),
    user=os.environ['MYSQL_USER'],
    password=os.environ['MYSQL_PASSWORD'],
    database=os.environ['MYSQL_DATABASE'],
)
cursor = conn.cursor()

csv_path = Path(r'C:\경로\drug-supplement schema v3\supplement_info.csv')
with open(csv_path, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

sql = '''INSERT INTO supplement_info
    (sttemnt_no, prduct, entrps, regist_dt, distb_pd, sungsang,
     srv_use, prsrv_pd, intake_hint1, main_fnctn, base_standard)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE prduct=VALUES(prduct), entrps=VALUES(entrps)'''

data = [(r.get('sttemnt_no',''), r.get('prduct',''), r.get('entrps',''),
         r.get('regist_dt',''), r.get('distb_pd',''), r.get('sungsang',''),
         r.get('srv_use',''), r.get('prsrv_pd',''), r.get('intake_hint1',''),
         r.get('main_fnctn',''), r.get('base_standard','')) for r in rows]

for i in range(0, len(data), 2000):
    cursor.executemany(sql, data[i:i+2000])

conn.commit()
print(f'supplement_info {len(data)}행 적재 완료')
cursor.close(); conn.close()
"
```

### 2단계 — pill_products / pill_product_ingredients / supplement_product_markers

```powershell
python scripts/import_v3_pill_data.py --csv-dir "C:\경로\drug-supplement schema v3"
```

적재 순서 (내부 자동):
1. `pill_products` — 4,525행
2. `pill_product_ingredients` — 892행
3. `supplement_product_markers` — 69,845행

---

## 적재 확인

```powershell
docker exec -it click_supplement_db mysql -u click_user -p click_db
```

```sql
SELECT 'supplement_info'           AS tbl, COUNT(*) AS cnt FROM supplement_info
UNION ALL SELECT 'supplement_product_markers', COUNT(*) FROM supplement_product_markers
UNION ALL SELECT 'pill_products',              COUNT(*) FROM pill_products
UNION ALL SELECT 'pill_product_ingredients',   COUNT(*) FROM pill_product_ingredients;
```

정상 결과:

| tbl | cnt |
|---|---:|
| supplement_info | 44,885 |
| supplement_product_markers | 69,845 |
| pill_products | 4,525 |
| pill_product_ingredients | 892 |

```sql
-- 건기식 브랜드 → supplement_id 연결 확인
SELECT si.prduct, spm.supplement_id
FROM supplement_info si
JOIN supplement_product_markers spm ON spm.supplement_info_id = si.id
WHERE si.prduct LIKE '%오메가3%'
LIMIT 5;

EXIT;
```
