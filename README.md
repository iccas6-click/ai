# CLICK AI Recognition

CLICK 서비스에서 사용하는 **의약품 인식**과 **건강기능식품 인식** 파이프라인을 개발하는 저장소입니다.

이미지에서 제품 후보와 성분 정보를 추출해 구조화된 인식 결과를 만드는 것까지가 이 저장소의 책임입니다.
성분 간 상호작용 판정, 위험도 결정, 사용자용 설명 생성은 `click/backend`에서 담당합니다.

---

## 이번 브랜치의 main 대비 변경점

- `/api/v1/pill/recognize`를 낱알 탐지 중심에서 처방전·약봉투·복약 안내문 문서 인식 중심으로 전환했습니다.
- 기존 프론트가 보내는 `recognizer`, `allowed_pill_ids`, `allowed_item_seqs`, `allowed_product_names` form field는 호환성 때문에 유지하지만, 서버는 문서 인식 파이프라인만 사용합니다.
- `app/services/prescription_recognition.py`를 추가해 Gemini Vision으로 약품명, 용량, 복용법, 성분 후보를 구조화합니다.
- 인식된 약품명은 MySQL 약 제품/성분 캐시와 legacy 제품-성분 테이블로 성분명을 보강합니다.
- 성분 미확정 항목은 LLM 제품명→성분명 보조 추론과 `needs_confirmation` 상태로 사용자 확인 흐름을 유지합니다.
- 건강기능식품 라벨 인식은 Gemini 제품명 추출, MFDS DB 매칭, RapidFuzz 재랭킹, 원료 파싱 구조를 유지합니다.

---

## 디렉터리 구조

```
ai/
├── app/                            # FastAPI 서버 (port 8001)
│   ├── main.py
│   ├── core/config.py
│   └── api/v1/
│       ├── supplement.py           # POST /api/v1/supplement/recognize
│       └── pill.py                 # POST /api/v1/pill/recognize
├── app/services/
│   └── prescription_recognition.py # 처방전·약봉투·복약 안내문 문서 인식
├── pill_recognition/               # 과거 낱알 인식 학습·추론 자산
│   ├── datasets/                   # 로컬 학습·평가 데이터, 내용 Git 제외
│   ├── training/                   # 학습 설정과 데이터 변환·평가 스크립트
│   ├── requirements/               # 런타임·학습 의존성
│   └── inference/                  # 학습과 분리된 추론 서비스
│       ├── artifacts/              # 추론 모델 가중치, Git 제외
│       ├── aihub_official_code/    # AI Hub 공식 배포 파일, Git 제외
│       ├── outputs/                # 추론 결과, Git 제외
│       ├── pill_recognition/       # v2: RTMDet + AI Hub ResNet retrieval/classifier
│       └── pill_recognition_legacy/ # v1 baseline 보존
├── supplement_recognition/         # 건강기능식품 라벨 인식 파이프라인
│   ├── src/
│   │   ├── pipeline.py             # 파이프라인 오케스트레이션
│   │   ├── extraction/
│   │   │   ├── llm_extractor.py    # Gemini Vision 제품명 추출 (retry + fallback)
│   │   │   ├── image_preprocessor.py # 이미지 전처리 (crop/denoise/enhance)
│   │   │   └── ingredient_parser.py  # 성분명 파싱 (main_fnctn / base_standard)
│   │   ├── matching/
│   │   │   ├── mfds_client.py      # MySQL FULLTEXT + RapidFuzz 매칭
│   │   │   └── matcher.py          # 매칭 결과 + 성분 파싱 결합
│   │   └── schema/result.py        # 응답 스키마 (Pydantic)
│   ├── scripts/
│   │   ├── evaluate_accuracy.py    # 정확도 측정 (이미지 파일명 = 정답)
│   │   ├── search_db.py            # DB 대화형 검색
│   │   ├── test_pipeline.py        # 단일 이미지 파이프라인 테스트
│   │   └── fetch_mfds_data.py      # MFDS 데이터 수집
│   ├── db/init.sql                 # DB 초기화 스크립트
│   ├── data/                       # Git 제외 (테스트 이미지 등)
│   │   ├── samples/                # 정확도 측정용 테스트 이미지
│   │   └── mfds_cache/             # MFDS 캐시
│   └── PIPELINE.md                 # 파이프라인 상세 문서
├── docker-compose.yml
├── requirements.txt
└── README.md
```

학습 데이터의 구체적인 배치 규칙은 [`pill_recognition/datasets/README.md`](./pill_recognition/datasets/README.md), RTMDet 단일 클래스 학습 흐름은 [`pill_recognition/training/rtmdet_single_class/README.md`](./pill_recognition/training/rtmdet_single_class/README.md)를 따릅니다.

---

## 공통 처리 흐름

```mermaid
flowchart LR
    A["이미지 입력"] --> B["입력 검증"]
    B --> C["이미지 전처리"]
    C --> D{"인식 대상"}
    D -->|"의약품"| E["처방전·약봉투 문서 인식"]
    D -->|"건강기능식품"| F["Gemini Vision 제품명 추출"]
    E --> G["약품명·용량·성분 후보 구조화"]
    F --> G
    G --> H["DB/API 후보 보강 및 표준화"]
    H --> I["신뢰도 및 확인 필요 여부 반환"]
```

공통 원칙:
- 인식 결과를 확정 사실이 아닌 **후보와 신뢰도**로 반환합니다.
- 신뢰도가 낮거나 여러 제품이 유사하면 `low_confidence`, `ambiguous`, `needs_confirmation` 상태를 구분합니다.
- 인식 실패 시에도 사용자가 제품과 성분을 직접 입력할 수 있도록 실패 원인을 구분합니다.

---

## 의약품 파이프라인

현재 앱에서 사용하는 의약품 인식은 처방전, 약봉투, 복약 안내문처럼 약품명이 텍스트로 적힌 문서를 대상으로 합니다.

### 현재 앱 연동 흐름

```
이미지 입력
  ↓
[FastAPI] app/api/v1/pill.py
  multipart 이미지를 수신하고 문서 인식 서비스로 전달
  ↓
[Gemini Vision] app/services/prescription_recognition.py
  처방 의약품명, 용량, 복용 정보, 성분 후보를 JSON으로 추출
  ↓
[약품/성분 보강]
  official_drug_products / official_drug_product_ingredients 우선 조회
  pill_product_ingredients legacy fallback 조회
  필요 시 LLM 제품명→성분명 보조 추론
  ↓
[응답 구조화]
  medications[], detections[], confidence, needs_confirmation 반환
```

낱알 이미지에서 모양을 보고 개별 알약을 맞히던 RTMDet 기반 코드는 학습·비교용 자산으로 남아 있지만, 현재 앱의 처방약 인식 요청은 위 문서 인식 경로를 사용합니다.

### 현재 정확도 (테스트 이미지 20장 기준)

| 지표 | 결과 |
|---|---|
| Gemini 약품명 인식 F1 | **95.7%** (Precision 93.7%, Recall 97.8%) |
| pill_products 제품명 매칭률 | **94.7%** (89/94건, LEFT JOIN LIKE 기준) |
| 상호작용 DB 역조회 정확도 | **100%** (475/475건, 옵션2) |
| 처방약×건기식 상호작용 감지율 | **8.7%** (40/462 조합, 처방전 14종 × 건기식 33종, 옵션3) |
| 상호작용 있는 처방약 비율 | **100%** (14/14종 — 처방전 약물 전부 최소 1개 건기식과 상호작용) |

> 상세 측정 결과 → [`docs/accuracy-report.md`](./docs/accuracy-report.md)

### 과거 낱알 인식 자산

위치: [`pill_recognition/inference/pill_recognition/`](./pill_recognition/inference/pill_recognition/)

실행 방법과 환경 구성은 [`pill_recognition/inference/pill_recognition/README.md`](./pill_recognition/inference/pill_recognition/README.md)를 참고합니다.

### 현재 구현 상태 (v2)

- 한 이미지에서 여러 알약을 `pill` 단일 클래스로 탐지
- RTMDet Bounding Box 기준으로 알약별 crop 생성
- 요청 또는 환경변수로 인식 엔진 선택
  - `retrieval`: RTMDet + AIHub ResNet152 feature retrieval
  - `aihub_classifier`: RTMDet + AIHub 공식 1,000-class classifier
  - `codeit`: `ZerofZero/codeit10_pj1` 기반 RTMDet + EfficientNet classifier
- retrieval/classifier 후보는 색상·모양·제형 metadata로 재정렬 가능
- 결과는 제품명, 성분, 업체, 품목기준코드, 일반/전문 여부와 함께 반환
- FastAPI `/recognize`, `/crops/recognize`, `/crops/recognize-batch` endpoint 제공
- 독립 pill API는 multipart form field `recognizer=codeit|retrieval|aihub_classifier`로 요청별 엔진 선택 가능

```bash
cd pill_recognition/inference
source ../../.venv/bin/activate
python -m pill_recognition.api --host 0.0.0.0 --port 8001
```

---

## 건강기능식품 파이프라인

자세한 내용 → [`supplement_recognition/PIPELINE.md`](./supplement_recognition/PIPELINE.md)

### 처리 흐름

```
이미지 입력
  ↓
[이미지 전처리]  image_preprocessor.py
  auto-crop / 512~1024px 정규화 / GaussianBlur 노이즈 제거 / 대비·선명도 강화
  ↓
[Gemini Vision]  llm_extractor.py
  1순위: 충북대 AI Gateway (gemini-3.5-flash, CBNUAI_API_KEY)
  2순위: Google Gemini API (gemini-2.0-flash, GEMINI_API_KEY)  ← 자동 fallback
  최대 2회 재시도 (지수 백오프)
  ↓
[MFDS DB 매칭]  mfds_client.py
  MySQL FULLTEXT (ngram) 상위 후보 → RapidFuzz + 길이 패널티 재랭킹
  FULLTEXT 실패 시 LIKE/RapidFuzz fallback
  유사도 70% 미만이면 needs_confirmation 반환
  ↓
[성분 파싱]  ingredient_parser.py
  main_fnctn: [성분명] 브래킷 패턴 추출
  base_standard: 비성분 키워드 필터링 후 콜론 앞 성분명 추출
  복합 성분(의 합), 영문 약어(DHA/EPA) 처리
  ↓
[공식 이미지 보강]  enrichment/official_image_lookup.py
  MFDS/공식 원문에 있는 제품 이미지 URL을 함께 반환
  ↓
SupplementRecognitionResult 반환
  { status, product: { product_name, ingredients: [...], confidence }, needs_confirmation }
```

### 현재 정확도 (테스트 이미지 50장 기준)

| 지표 | 결과 |
|---|---|
| Gemini 제품명 추출 성공률 | 50/50 = **100%** |
| Gemini 추출명 vs 정답 유사도 | **93.4%** |
| DB Top-1 정확 매칭률 | 42/50 = **84.0%** |
| 성분 해석율 F1 | **79.6%** (Precision 100%, Recall 66.1%) |

### 실패 케이스 분류 (8건)

| 원인 | 건수 | 케이스 |
|---|---|---|
| Gemini 인식 성공 + DB 오매칭 | **6건** | 고려은단 오메가3, 락토핏 골드, 밀크씨슬 이뮨 바이탈 샷, 센트룸 멀티 구미, 얼라이브 원스데일리 밀크씨슬, 칼슘앤마그네슘비타민D아연 |
| Gemini 인식 성공 + DB 미등재 | **1건** | 세노비스 칼슘+비타민D |
| Gemini 인식 실패 → DB 매칭 실패 | **1건** | 비타민 활기력샷 ("활기력" → "활력"으로 오인식) |

- **DB 오매칭**: Gemini가 올바른 제품명을 추출했으나 토큰이 겹치는 유사 제품이 먼저 매칭됨
- **DB 미등재**: Gemini 추출은 정확하나 해당 브랜드 제품 자체가 DB에 없음 (코드로 해결 불가)
- **Gemini 오인식**: Gemini가 제품명 일부를 잘못 읽어 DB 매칭도 실패

### API

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
    "product_name": "센트룸 실버 맨",
    "manufacturer": "...",
    "product_image_url": "https://...",
    "product_image_source_url": "https://...",
    "ingredients": ["비타민A", "비타민C", "아연", ...],
    "confidence": 0.97
  },
  "needs_confirmation": false
}
```

### 서버 실행

환경 변수 설정 (`.env`):
```env
CBNUAI_API_KEY=           # 충북대 AI Gateway API 키 (1순위)
GEMINI_API_KEY=           # Google Gemini API 키 (fallback용)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=click_db
MYSQL_USER=click_user
MYSQL_PASSWORD=
```

```bash
docker compose up -d
uvicorn app.main:app --reload --port 8002
```

- Swagger: `http://localhost:8002/docs`
- 헬스체크: `http://localhost:8002/health`

---

## 공통 상태 코드

| 상태 | 의미 |
|---|---|
| `queued` | 인식 요청이 접수됨 |
| `processing` | 모델 또는 OCR이 실행 중임 |
| `completed` | 확인 가능한 구조화 결과가 생성됨 |
| `needs_confirmation` | 후보가 생성됐으며 사용자 확인이 필요함 |
| `ambiguous` | 상위 후보 점수 차이가 작아 비교 확인이 필요함 |
| `low_confidence` | 후보 점수가 낮아 직접 검색 또는 재촬영이 필요함 |
| `no_candidate` | 제품 후보를 생성하지 못함 |
| `partial` | 일부 이미지만 인식됨 |
| `failed` | 결과를 생성하지 못함 |

### 건강기능식품 오류 코드

| 오류 코드 | 의미 |
|---|---|
| `INVALID_FILE` | JPG/PNG 외 파일 형식 |
| `OCR_TEXT_NOT_FOUND` | Gemini가 제품명을 추출하지 못함 |
| `PRODUCT_NOT_MATCHED` | MFDS DB에서 일치 제품 없음 |
| `MODEL_INFERENCE_FAILED` | Gemini API 호출 실패 |

---

## 남은 작업

- [x] Backend 다중 성분 엔드포인트 연동 (`ingredients` 리스트 → `/api/v1/interactions/analyze`)
- [x] 제품 공식 이미지 URL 응답
- [x] 알약 인식 엔진 요청별 선택
- [ ] 실제 이미지 정확도 테스트 확대
