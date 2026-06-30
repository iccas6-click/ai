# CLICK AI Recognition

CLICK 서비스에서 사용하는 **의약품 인식**과 **건강기능식품 인식** 파이프라인을 개발하는 저장소입니다.

이미지에서 제품 후보와 성분 정보를 추출해 구조화된 인식 결과를 만드는 것까지가 이 저장소의 책임입니다.
성분 간 상호작용 판정, 위험도 결정, 사용자용 설명 생성은 `click/backend`에서 담당합니다.

---

## 디렉터리 구조

```
ai/
├── app/                        # FastAPI 서버 (port 8001)
│   ├── main.py
│   └── api/v1/
│       ├── supplement.py       # POST /api/v1/supplement/recognize
│       └── pill.py             # POST /api/v1/pill/recognize
├── supplement_recognition/     # 건강기능식품 인식 파이프라인
├── pill_recognition/           # 의약품 인식 파이프라인
├── requirements.txt
├── docker-compose.yml
└── .env
```

---

## 서버 실행

### 환경 변수 설정

`.env` 파일을 `ai/` 루트에 생성:

```env
CBNUAI_API_KEY=           # 충북대 AI Gateway API 키 (1순위)
GEMINI_API_KEY=           # Google Gemini API 키 (fallback용)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=click_db
MYSQL_USER=click_user
MYSQL_PASSWORD=
```

### DB 실행

```bash
docker compose up -d
```

### AI 서버 실행

```bash
# ai/ 루트에서 실행
uvicorn app.main:app --reload --port 8001
```

### API 확인

- Swagger: `http://localhost:8001/docs`
- 헬스체크: `http://localhost:8001/health`

---

## 건강기능식품 파이프라인

자세한 내용 → [`supplement_recognition/PIPELINE.md`](./supplement_recognition/PIPELINE.md)

### 처리 흐름

```
이미지 입력
  ↓
[이미지 전처리]
  auto-crop / 리사이즈 / 노이즈 제거 / 대비·선명도 강화
  ↓
[Gemini Vision] 제품명 추출
  충북대 AI Gateway (gemini-3.5-flash) → 실패 시 Google Gemini API fallback
  최대 2회 재시도
  ↓
[MFDS DB 매칭]
  FULLTEXT 검색 → RapidFuzz partial_ratio 재랭킹 (유사도 70% 이상)
  ↓
[성분 파싱]
  main_fnctn [브래킷] 패턴 + base_standard 비성분 필터링
  ↓
SupplementRecognitionResult 반환
{
  status, product: { product_name, ingredients: [...], confidence, ... }
}
```

### API 엔드포인트

```
POST /api/v1/supplement/recognize
Content-Type: multipart/form-data
Body: image (JPG/PNG)

Response:
{
  "request_id": "rec_supplement_xxxx",
  "status": "completed",
  "product": {
    "product_code": "20230001234567",
    "product_name": "TWK10 100억 분말",
    "manufacturer": "...",
    "ingredients": ["Lactobacillus plantarum TWK10 프로바이오틱스", "프로바이오틱스", "아연", ...],
    "main_function": "...",
    "base_standard": "...",
    "confidence": 0.95
  },
  "needs_confirmation": false,
  "warnings": []
}
```

---

## 의약품 파이프라인

위치: [`pill_recognition/`](./pill_recognition/) — 팀원 담당

---

## 공통 상태 및 오류 코드

| 상태 | 의미 |
|---|---|
| `completed` | 인식 성공 |
| `needs_confirmation` | DB 미매칭 또는 낮은 신뢰도 — 사용자 확인 필요 |
| `failed` | 인식 실패 |

| 오류 코드 | 의미 |
|---|---|
| `INVALID_FILE` | JPG/PNG 외 파일 형식 |
| `OCR_TEXT_NOT_FOUND` | Gemini가 제품명을 추출하지 못함 |
| `PRODUCT_NOT_MATCHED` | MFDS DB에서 일치 제품 없음 |
| `MODEL_INFERENCE_FAILED` | Gemini API 호출 실패 |

---

## 남은 작업

- [ ] Backend 다중 성분 엔드포인트 연동 (`ingredients` 리스트 → `/interactions`)
- [ ] 실제 이미지 정확도 테스트 (목표: 15~20장)
- [ ] `src/ocr/reader.py` 제거 (EasyOCR 잔재, 미사용)
