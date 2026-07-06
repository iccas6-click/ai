# 건강기능식품 인식 파이프라인

## 개요

사용자가 촬영한 건강기능식품 포장 이미지에서 제품명을 추출하고, MFDS(식약처) DB에서 일치 제품을 찾아 성분 목록을 반환하는 파이프라인입니다.

```
이미지 → 전처리 → Gemini Vision → MFDS DB 매칭 → 성분 파싱 → 공식 이미지 보강 → 결과 반환
```

---

## 디렉터리 구조

```
supplement_recognition/
├── src/
│   ├── extraction/
│   │   ├── llm_extractor.py      # Gemini Vision 제품명 추출
│   │   ├── image_preprocessor.py # 이미지 전처리
│   │   └── ingredient_parser.py  # 성분명 파싱
│   ├── matching/
│   │   ├── db.py                 # MySQL 연결 / FULLTEXT 검색
│   │   ├── mfds_client.py        # MFDS product search + fallback
│   │   └── matcher.py            # 유사도 재랭킹 + 성분 파싱 결합
│   ├── enrichment/
│   │   └── official_image_lookup.py # 공식 제품 이미지 URL 보강
│   ├── schema/
│   │   └── result.py             # 응답 스키마 (Pydantic)
│   ├── pipeline.py               # 전체 파이프라인 오케스트레이션
│   └── ocr/
│       └── reader.py             # (미사용) EasyOCR 잔재 — 제거 예정
└── data/                         # gitignore됨 (DB 덤프 등)
```

---

## 파이프라인 단계

### 1단계 — 이미지 전처리 (`image_preprocessor.py`)

사용자가 촬영한 사진은 배경 노이즈, 기울어짐, 낮은 해상도 등이 있을 수 있습니다. Gemini에 보내기 전에 전처리를 수행합니다.

| 단계 | 처리 내용 |
|---|---|
| auto-crop | Canny 엣지 검출로 제품 영역 추출. 면적이 원본의 20% 미만이면 크롭 건너뜀 |
| 리사이즈 | 짧은 변 512px 이상, 긴 변 1024px 이하로 정규화 |
| 노이즈 제거 | GaussianBlur (radius 0.5) |
| 대비·선명도 강화 | 대비 1.3×, 선명도 2.0×, 밝기 1.05× |

전처리 결과는 `_preprocessed_{원본파일명}` 로 임시 저장되고 파이프라인 완료 후 삭제됩니다.

**오류 처리**: 전처리 실패 시 원본 이미지로 그대로 진행합니다.

---

### 2단계 — Gemini Vision 제품명 추출 (`llm_extractor.py`)

전처리된 이미지를 Gemini Vision에 보내 제품명을 추출합니다.

#### API 우선순위

| 순위 | API | 모델 | 환경 변수 |
|---|---|---|---|
| 1순위 | 충북대 AI Gateway | `gemini-3.5-flash` | `CBNUAI_API_KEY` |
| 2순위 (fallback) | Google Gemini API | `gemini-2.0-flash` | `GEMINI_API_KEY` |

1순위 키가 없거나 호출 실패 시 자동으로 2순위로 전환됩니다.

#### 재시도 정책

- 최대 2회 재시도
- 재시도 간격: 1초 → 2초 (지수 백오프)
- 두 API 모두 실패하면 `MODEL_INFERENCE_FAILED` 오류 반환

#### 프롬프트

```
이 이미지는 한국 건강기능식품의 포장입니다.
제품명(상품명)만 정확하게 추출해주세요.
제조사, 성분명, 광고 문구는 포함하지 마세요.
제품명이 명확하지 않으면 가장 눈에 띄는 이름을 반환해주세요.
```

---

### 3단계 — MFDS DB 매칭 (`mfds_client.py`, `matcher.py`)

추출된 제품명으로 식약처 MFDS 데이터베이스(44,766건)에서 일치 제품을 찾습니다.

#### 검색 방식

1. **FULLTEXT 검색** — MySQL ngram parser 기반 후보 검색
2. **Fallback 검색** — FULLTEXT index가 없거나 결과가 약하면 LIKE 기반 후보를 추가 검색
3. **RapidFuzz 재랭킹** — 제품명 유사도와 길이 패널티로 후보 재평가
4. **임계값 필터** — 유사도 70% 미만이면 미매칭으로 처리

#### 미매칭 처리

유사도가 낮거나 DB에 제품이 없으면 `needs_confirmation: true`로 반환합니다. 성분 목록은 비어 있으며, 프론트엔드에서 사용자에게 직접 입력을 요청해야 합니다.

---

### 4단계 — 성분 파싱 (`ingredient_parser.py`)

MFDS 데이터에는 `main_fnctn`(기능성 내용)과 `base_standard`(기준·규격) 두 필드에 성분 정보가 있습니다.

#### `main_fnctn` 파싱 (`parse_from_main_fnctn`)

`[성분명]` 형태의 브래킷 패턴에서 성분을 추출합니다.

```
예: "[프로바이오틱스] 장 건강에 도움을 줄 수 있음 → 프로바이오틱스"
예: "[오메가-3 지방산(EPA 및 DHA)] 혈중 중성지방 개선 → 오메가-3 지방산"
```

- 개별인정 번호(`제2015-49호` 등) 제거
- 괄호 내 설명 제거 (단, 약어는 유지: DHA, EPA)

#### `base_standard` 파싱 (`parse_from_base_standard`)

비성분 라인을 필터링한 후 `성분명: 기준값` 구조에서 콜론 앞 부분을 성분명으로 추출합니다.

**비성분 키워드 필터** (아래 포함 시 해당 라인 제외):
```
성상, 세균수, 대장균, 납, 카드뮴, 비소, 수은, 붕해, 용출,
총균수, 진균수, 메틸수은, 총비소, 잔류, 이물, 산가, 과산화물가,
수분, 회분, 조단백, 조지방, 총 플라보노이드, CFU, Plate Count,
Yeast, Mould, E. coli, S., Hexane, 헥산, 국문, 영문, 미생물 규격, ...
```

**특수 케이스 처리**:

| 케이스 | 예시 | 처리 방식 |
|---|---|---|
| 복합 성분 (`의 합`) | `EPA 및 DHA의 합` | 전체 표현을 하나의 성분으로 유지 |
| 영문 약어 | `Docosahexaenoic acid (DHA)...` | 콜론 없이 긴 경우 괄호 내 약어 추출 |
| 번호 목록 | `①`, `1.`, `(1)`, `•` 등 | 번호·기호 제거 후 성분명만 추출 |

#### 결합 및 중복 제거 (`extract_ingredients`)

`main_fnctn`과 `base_standard` 결과를 합치고 중복을 제거합니다. 짧은 성분명(2자 이하)은 제외합니다.

### 5단계 — 공식 이미지 보강 (`official_image_lookup.py`)

MFDS 매칭 결과에 제품 이미지 정보가 있으면 `product_image_url`과 `product_image_source_url`을 함께 반환합니다. 프론트는 사용자가 촬영한 사진 대신 공식 제품 이미지를 인식 결과 확인 화면에 표시할 수 있습니다.

이미지를 찾지 못해도 인식 자체는 실패하지 않습니다. 이 경우 이미지 필드만 `null`로 반환됩니다.

---

## 응답 스키마

```python
class SupplementProduct(BaseModel):
    product_code: Optional[str]     # MFDS 제품 코드
    product_name: str               # DB 매칭 제품명 (또는 Gemini 추출명)
    manufacturer: Optional[str]     # 제조사
    product_image_url: Optional[str] # 공식 제품 이미지 URL
    product_image_source_url: Optional[str] # 이미지 출처 URL
    main_function: Optional[str]    # 기능성 내용 원문
    base_standard: Optional[str]    # 기준·규격 원문
    ingredients: list[str]          # 파싱된 성분명 리스트
    confidence: float               # 유사도 (0.0~1.0)

class SupplementRecognitionResult(BaseModel):
    request_id: str
    status: str                     # completed | needs_confirmation | failed
    product: Optional[SupplementProduct]
    needs_confirmation: bool
    warnings: list[str]
    error_code: Optional[str]
    error_message: Optional[str]
```

---

## 환경 변수

| 변수 | 설명 | 필수 |
|---|---|---|
| `CBNUAI_API_KEY` | 충북대 AI Gateway API 키 | 1순위 (선택) |
| `GEMINI_API_KEY` | Google Gemini API 키 | fallback (선택, 하나는 있어야 함) |
| `MYSQL_HOST` | MySQL 호스트 | 필수 |
| `MYSQL_PORT` | MySQL 포트 (기본 3306) | 필수 |
| `MYSQL_DATABASE` | 데이터베이스명 (`click_db`) | 필수 |
| `MYSQL_USER` | DB 사용자 | 필수 |
| `MYSQL_PASSWORD` | DB 비밀번호 | 필수 |

---

## 한계점

### 1. Gemini 제품명 추출 정확도

- 포장에 여러 이름이 있을 때 Gemini가 제품명 대신 광고 문구나 성분명을 반환할 수 있음
- 이미지 각도·조명·반사광에 따라 성능 차이 있음
- 현재 이미지 정확도 테스트 미완료 (목표: 15~20장 실측)

### 2. 규칙 기반 성분 파싱

- MFDS `base_standard` 필드 형식이 제품마다 다르며, 예외 케이스가 발생할 수 있음
- 새로운 비성분 패턴이 추가되면 `_NON_INGREDIENT_KEYWORDS` 수동 업데이트 필요
- 성분명이 50자 이상인 경우 약어 추출 fallback을 사용하지만 완벽하지 않음

### 3. DB 미등재 제품

- 식약처 DB에 없는 신제품이나 수입품은 `needs_confirmation` 반환
- 이 경우 성분 목록이 비어 있으며, 사용자 직접 입력 외 방법 없음

### 4. Backend 연동

- AI 서버는 `ingredients` 리스트와 공식 이미지 URL을 반환합니다.
- 프론트는 알약 인식 결과와 건강기능식품 `ingredients`를 합쳐 Backend `/api/v1/interactions/analyze`로 보냅니다.
- Backend는 `supplement_map`/`supplement_aliases` 기준으로 성분을 canonical entity에 연결합니다.

---

## 남은 작업

- [ ] `src/ocr/reader.py` 제거 (EasyOCR 잔재, 미사용)
- [ ] 실제 이미지 15~20장으로 정확도 측정
- [x] Backend 다중 성분 엔드포인트 연동
- [x] 공식 제품 이미지 URL 응답
